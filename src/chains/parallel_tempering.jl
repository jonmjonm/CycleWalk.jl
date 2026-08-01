# =============================================================================
# Parallel tempering — swap mechanics and driver
#
# See docs/parallel_tempering_implementation.md (tasks 3-6) for the design
# rationale. Types (BetaLattice, Replica, PTEnsemble, HeatBath, PTDiagnostics) live
# in parallel_tempering_types.jl; the AnnealPath seam (energy_at, weights_at,
# configure_measure!) lives in anneal_path.jl and is shared with annealed_smc.jl.
# =============================================================================

# ------------------------------------------------------------------ swap schedule
"""
    even_odd_pairs(M, round) -> Vector{Tuple{Int,Int}}

Adjacent rung pairs to attempt this round, alternating between the even and odd
pairings:

    round odd  ->  (1,2) (3,4) (5,6) ...
    round even ->  (2,3) (4,5) (6,7) ...

Pairs within a round are disjoint, so the whole round is one serial pass with no
conflicts. This is the deterministic even-odd (DEO) / non-reversible scheme of Syed,
Bouchard-Cote, Deligiannidis & Doucet (2022), whose round-trip rate is O(1) in the
number of rungs rather than the diffusive O(1/M^2) of random-pair schemes. Do NOT
replace it with "pick one random adjacent pair", which is what
`ParallelTempering.java:184` does and which performs M times less exchange per round.
"""
function even_odd_pairs(M::Int, round::Int)
    start = isodd(round) ? 1 : 2
    return [(k, k+1) for k in start:2:(M-1)]
end

"""
    idle_rungs(M, round) -> Vector{Int}

Rungs that appear in no pair this round. The count depends on both parities and is 0, 1,
or 2 — do NOT hard-code a case analysis, derive it from [`even_odd_pairs`](@ref):

    M=4, odd round : (1,2)(3,4)      -> idle = []
    M=4, even round: (2,3)           -> idle = [1, 4]
    M=5, odd round : (1,2)(3,4)      -> idle = [5]
    M=5, even round: (2,3)(4,5)      -> idle = [1]

The heat bath fires only when its rung is idle, so it costs nothing (design §1.9).
"""
idle_rungs(M::Int, round::Int) =
    setdiff(1:M, Iterators.flatten(even_odd_pairs(M, round)))

# ------------------------------------------------------------------ swap logratio
"""
    swap_logratio(path, beta_i, beta_j, phi_i, phi_j) -> Float64

Log acceptance ratio for exchanging the states whose cached potentials are `phi_i`,
`phi_j` between path points `beta_i`, `beta_j`:

    log α = ⟨ w(beta_i) − w(beta_j) , phi_i − phi_j ⟩

Arithmetic only — no energy evaluation. Derived in the module docs; note the operand
order, which follows from ν ∝ exp(−energy) (NOT exp(+energy)). For a pure β-ladder this
equals `(beta_i − beta_j) * (E_i − E_j)`, which is the independent cross-check.
"""
@inline function swap_logratio(path::AnnealPath, beta_i::Float64, beta_j::Float64,
                               phi_i::NTuple{K,Float64}, phi_j::NTuple{K,Float64}) where {K}
    return _dot(_sub(weights_at(path, beta_i), weights_at(path, beta_j)),
                _sub(phi_i, phi_j))
end

"""
    swap_logratio(w_i, w_j, phi_i, phi_j) -> Float64

Same acceptance ratio, taking two explicit weight tuples rather than a path and two
path points. Used by the heat-bath exchange (§2.4, `try_heat_bath!` in
`pt_heat_bath.jl`), whose partner's weights come from a `HeatBath`'s own measure
rather than sitting on the path.
"""
@inline function swap_logratio(w_i::NTuple{K,Float64}, w_j::NTuple{K,Float64},
                               phi_i::NTuple{K,Float64}, phi_j::NTuple{K,Float64}) where {K}
    return _dot(_sub(w_i, w_j), _sub(phi_i, phi_j))
end

# ------------------------------------------------------------------ swap round
"""
    swap_round!(ensemble, round, rng, diagnostics) -> Int

Attempt every pair from [`even_odd_pairs`](@ref) and return the number accepted. Runs
SERIALLY on the calling thread: once `phi` is cached each swap is a handful of flops, so
spawning a task per pair (as the MSMS reference does) costs orders of magnitude more
than the work (design §1.5).

A swap exchanges the two rungs' `beta_index` and the `walker_at_rung` entries. States
never move.
"""
function swap_round!(ensemble::PTEnsemble{K}, round::Int, rng::AbstractRNG,
                     diag::PTDiagnostics) where {K}
    M = nrungs(ensemble)
    accepted = 0
    for (i, j) in even_odd_pairs(M, round)
        wi, wj = ensemble.walker_at_rung[i], ensemble.walker_at_rung[j]
        logα = swap_logratio(ensemble.path, ensemble.lattice[i], ensemble.lattice[j],
                             ensemble.replicas[wi].phi, ensemble.replicas[wj].phi)
        diag.attempts[i] += 1
        diag.accept_prob_sum[i] += min(1.0, exp(logα))
        if log(rand(rng)) < logα
            ensemble.walker_at_rung[i], ensemble.walker_at_rung[j] = wj, wi
            ensemble.replicas[wi].beta_index = j
            ensemble.replicas[wj].beta_index = i
            diag.accepts[i] += 1
            accepted += 1
        end
    end
    return accepted
end

# ------------------------------------------------------------------ round bookkeeping
"""
    record_round!(ensemble, diagnostics)

Update occupancy and round-trip counts after a swap round. A round trip is a walker
travelling rung M -> rung 1 -> rung M; `last_end_visited` tracks which end it saw last,
and reaching rung M having last seen rung 1 completes one.
"""
function record_round!(ensemble::PTEnsemble, diag::PTDiagnostics)
    M = nrungs(ensemble)
    for r in ensemble.replicas
        diag.occupancy[r.replica_id, r.beta_index] += 1
        if r.beta_index == 1
            r.last_end_visited = 1
        elseif r.beta_index == M
            r.last_end_visited == 1 && (diag.round_trips[r.replica_id] += 1)
            r.last_end_visited = M
        end
    end
end

# ------------------------------------------------------------------ potentials
"""
    refresh_potentials!(ensemble)

Recompute every replica's `phi = (e₁(state), …, e_K(state))` from `ensemble.scores`.
Cheap: each energy reads its identifier-guarded per-district cache (in sync after
`advance!`), so this is arithmetic, not fresh log-dets — the PT analogue of annealed
SMC's `refresh_potentials!` for particles.
"""
function refresh_potentials!(ensemble::PTEnsemble{K}) where {K}
    for r in ensemble.replicas
        r.phi = map(e -> e(r.state)::Float64, ensemble.scores)
    end
    return nothing
end

# ------------------------------------------------------------------ output
"""
    emit_maps!(writers, ensemble, write_rungs, round)

Write one Atlas map per configured rung for this round. `write_rungs` is `:target`
(`writers` a single `Writer`, only rung `M` — the sample stream), `:all` (`writers` a
`Vector{Writer}` of length `M`, index `k` -> rung `k`, file suffix left to the
caller), or `:none` (no-op, e.g. for tests and tuning runs). No-op if `writers ===
nothing`.

Every emitted map's `MapParam` carries `pt/replica_id`, `pt/beta_index`, `pt/beta`,
`pt/bath_swaps`, and `pt/round`; maps are named `"round<r>_beta<k>"`. PT samples are
unweighted MCMC states (weight 1), so writers should be built with the default
`weight_type=Int64` (unlike AIS/ASMC, which need `Float64`).
"""
function emit_maps!(writers::Union{Vector{Writer},Writer,Nothing},
                    ensemble::PTEnsemble, write_rungs::Symbol, round::Int)
    writers === nothing && return nothing
    write_rungs === :none && return nothing
    M = nrungs(ensemble)
    rungs = write_rungs === :target ? (M:M) : (1:M)
    for k in rungs
        writer = writers isa Vector ? writers[k] : writers
        replica = ensemble.replicas[ensemble.walker_at_rung[k]]
        extra = Dict{String,Any}(
            "pt/replica_id" => replica.replica_id,
            "pt/beta_index" => k,
            "pt/beta"       => ensemble.lattice[k],
            "pt/bath_swaps" => replica.bath_swaps,
            "pt/round"      => round,
        )
        name = "round" * string(round) * "_beta" * string(k)
        addMap(writer.atlas.io,
              build_output_map(writer, replica.state, name, 1, replica.diagnostics;
                               extra_data=extra))
    end
    return nothing
end

# ------------------------------------------------------------------ driver
"""
    run_parallel_tempering!(partition, proposal, measure, lattice, swap_interval,
                            n_rounds, rng; path=nothing, backend=SerialBackend(proposal),
                            heat_bath=nothing, init_steps=0, write_rungs=:target,
                            output_every=1, writers=nothing,
                            run_diagnostics=RunDiagnostics(), diagnostics=nothing,
                            seed=nothing)
        -> (ensemble, diagnostics)

Run parallel tempering: `lattice.betas[k]` gives rung `k`'s path point (rung 1
hottest/most tempered, rung `M` coldest/target). `M` replicas ("walkers") are cloned
from `partition`, one per rung, each with its own rng (seeded sequentially from `rng`,
so results are independent of worker count — see the module invariants), diagnostics,
and `work_measure`. Every `swap_interval` MH steps (via `backend`, default serial), a
deterministic even/odd swap round is attempted between adjacent rungs
([`swap_round!`](@ref)), an optional [`HeatBath`](@ref) exchange fires on the rung the
swap pattern leaves idle, and round diagnostics (occupancy, round trips, straggler
gap) are recorded. `diagnostics` is reset on entry ([`reset_pt_diagnostics!`](@ref)),
so one object may drive several runs (mirrors `reset_schedule!` in `annealed_smc.jl`).

`path` (default: `linear_path` from `measure`'s target weights) is the swappable path
geometry shared with annealed SMC — see `AnnealPath`. `init_steps` burns every replica
in at its OWN assigned rung before swapping starts; with `init_steps <= 0` and more
than one rung the ladder starts fully correlated (a warning, not an error — PT still
works, just decorrelates more slowly).

`write_rungs`/`output_every`/`writers` control map output — see [`emit_maps!`](@ref).
"""
function run_parallel_tempering!(
    partition::LinkCutPartition,
    proposal::Union{Function,Vector{Tuple{T,Function}}},
    measure::Measure,
    lattice::BetaLattice,
    swap_interval::Int,
    n_rounds::Int,
    rng::AbstractRNG;
    path::Union{AnnealPath,Function,Nothing}=nothing,
    backend::PTBackend=SerialBackend(proposal),
    heat_bath::Union{HeatBath,Nothing}=nothing,
    init_steps::Int=0,
    write_rungs::Symbol=:target,          # :target | :all | :none
    output_every::Int=1,                  # emit every N swap rounds
    writers::Union{Vector{Writer},Writer,Nothing}=nothing,
    run_diagnostics::RunDiagnostics=RunDiagnostics(),
    diagnostics::Union{PTDiagnostics,Nothing}=nothing,
    seed=nothing,
) where {T<:Real}
    swap_interval >= 1 ||
        throw(ArgumentError("swap_interval must be >= 1, got $swap_interval"))
    n_rounds >= 1 || throw(ArgumentError("n_rounds must be >= 1, got $n_rounds"))
    write_rungs in (:target, :all, :none) ||
        throw(ArgumentError("write_rungs must be :target, :all, or :none, got :$write_rungs"))

    M = length(lattice)
    if init_steps <= 0 && M > 1
        @warn "run_parallel_tempering! with init_steps=$init_steps starts the ladder " *
              "fully correlated (every replica is an identical clone of `partition`); " *
              "PT still works, but decorrelates more slowly. Consider init_steps > 0."
    end

    diag = diagnostics === nothing ? PTDiagnostics(M) : diagnostics
    reset_pt_diagnostics!(diag)

    scores, target_w = measure_scores_and_targets(measure)
    K = length(scores)
    p = path === nothing ? linear_path(target_w) :
        path isa Function ? LinearPath{K}(path) : path

    if writers !== nothing
        metadata = parallel_tempering_run_metadata(measure, lattice, swap_interval,
            n_rounds; seed=seed, path=p, proposal=proposal, backend=backend,
            init_steps=init_steps, write_rungs=write_rungs,
            output_every=output_every, heat_bath=heat_bath)
        for w in (writers isa Vector ? writers : (writers,))
            stamp_run_metadata!(w, metadata)
        end
    end

    prev_blas_threads = BLAS.get_num_threads()
    Threads.nthreads() > 1 && BLAS.set_num_threads(1)
    try

    # ---- build the ensemble: one replica per rung, seeded sequentially -------
    replicas = Vector{Replica{K}}(undef, M)
    for w in 1:M
        st = clone_for_annealing(partition)
        replicas[w] = Replica{K}(st, w, map(e -> e(st)::Float64, scores),
                                 PCG.PCGStateOneseq(UInt64, rand(rng, UInt64)),
                                 deepcopy(run_diagnostics), deepcopy(measure),
                                 w, 0, 0)
    end
    ensemble = PTEnsemble{K}(replicas, collect(1:M), lattice, p, scores)

    if init_steps > 0
        advance!(backend, ensemble, init_steps)
        refresh_potentials!(ensemble)
    end

    # ---- main loop ------------------------------------------------------------
    for round in 1:n_rounds
        advance!(backend, ensemble, swap_interval)
        push!(diag.straggler_gap, backend_straggler_gap(backend))
        refresh_potentials!(ensemble)
        swap_round!(ensemble, round, rng, diag)
        heat_bath !== nothing && try_heat_bath!(ensemble, heat_bath, round, rng, diag)
        record_round!(ensemble, diag)
        round % output_every == 0 && emit_maps!(writers, ensemble, write_rungs, round)
    end

    return ensemble, diag

    finally
        Threads.nthreads() > 1 && BLAS.set_num_threads(prev_blas_threads)
    end
end
