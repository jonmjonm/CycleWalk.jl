# =============================================================================
# Annealed SMC sampler (Del Moral–Doucet–Jasra 2006)
#
# Evolves a POPULATION of N particles together along a PATH through measure space
# parametrised by t ∈ [0,1]. The path returns, at each t, the weight of every
# energy term in the target `Measure` (in `scores` order):
#
#     path(t) :: NTuple{K,Float64}     # e.g. linear diagonal: t .* (γ, iso)
#
# so path geometry (diagonal / staged-L / nonzero-base / fitted) is a fully
# swappable knob, independent of the sampler. `modify_measure!` is NOT used here;
# the path replaces it (build one from a target measure with `linear_path`).
#
# The ONLY thing distinguishing the "fixed γ+iso path" and "adaptive tempering"
# variants is `next_t`. Everything else — potentials, weights, ESS, resampling,
# rejuvenation, logZ — is shared.
#
# SIGN CONVENTION: `get_log_energy` (and therefore `energy_at` below) returns an
# ENERGY, not a log-density. The target is ν(τ) ∝ exp(−E(τ,t)) — see `get_delta_energy`,
# and §6 of the Cycle Walk paper, which defines ν_γ(τ) ∝ π(ξ_τ)/Tree(ξ_τ)^γ ∝
# exp(−γ·J_Tree − J) with that exponent being exactly what `get_log_energy` sums. So
# every incremental log-weight is E(t_prev) − E(t), NOT E(t) − E(t_prev). Inverting it
# makes the sampler estimate Z_base/Z_target, and — because the weights also drive
# resampling here — degrades the particle population itself, so it cannot be repaired
# by negating logZ afterwards. `test_annealed_smc.jl` pins the sign against the exactly
# known log Z(1)/Z(0) = log(117/654) on the 4×4 test graph.
#
# Potentials: each particle caches φ = (energy_1(state), …, energy_K(state)), the
# PATH-INDEPENDENT raw per-term energies, as an NTuple{K} (isbits, no heap alloc).
# Reading them is cheap: get_log_spanning_forests / get_isoperimetric_score read
# identifier-guarded per-district caches that update_partition! keeps in sync, so
# refresh is O(num_dists) arithmetic, not fresh log-dets (those live in the MH
# proposals). Given φ, the log-energy at any t and any path is
#     E(state, t) = ⟨ path(t), φ ⟩
# so the adaptive ESS root-find is a pure arithmetic sweep — no energy calls.
#
# Schedules are STATEFUL and are reset by `run_annealed_smc!` on entry (see
# [`reset_schedule!`](@ref)), so the same object may be reused across runs. Before that
# reset existed, a second run with the same schedule silently performed zero blocks and
# returned logZ = 0.0.
# =============================================================================

# ------------------------------------------------------------------ tuple utils
@inline _dot(a::NTuple{K,Float64}, b::NTuple{K,Float64}) where {K} = sum(map(*, a, b))
@inline _sub(a::NTuple{K,Float64}, b::NTuple{K,Float64}) where {K} = map(-, a, b)

# ------------------------------------------------------------------ particle
"""
    Particle{K}

One SMC particle: a `LinkCutPartition` state, its accumulated log importance
weight since the last resample (`logW`), the cached path-independent per-term
potentials `phi = (e₁(state), …, e_K(state))`, and an independent rejuvenation
RNG. `phi` is refreshed once per block after rejuvenation.
"""
mutable struct Particle{K}
    state::LinkCutPartition
    logW::Float64
    phi::NTuple{K,Float64}
    rng::PCG.PCGStateOneseq
end

# ------------------------------------------------------------------ schedules
abstract type AnnealSchedule end

"""
    FixedSchedule(grid; ess_frac=0.5)

Precomputed increasing `t`-grid ending at 1.0 (e.g. `range(0,1,length=K+1)`).
Resamples adaptively when ESS drops below `ess_frac * N`. The simple variant to
build and validate first.
"""
mutable struct FixedSchedule <: AnnealSchedule
    grid::Vector{Float64}
    k::Int                          # index of last t returned; t_prev = grid[k+1]
    ess_frac::Float64
    FixedSchedule(grid; ess_frac=0.5) = new(collect(grid), 0, ess_frac)
end

"""
    AdaptiveTempering(; ess_target=0.5)

Chooses `t_next` online by bisection so the population ESS drops to `ess_target * N`,
in the coupled form (resample every step). `t_prev` tracks the last point returned and
is maintained by [`next_t`](@ref); [`reset_schedule!`](@ref) returns it to 0.

Requires `init_steps > 0`: with an unburned population every particle is an identical
clone, so every incremental weight is equal, ESS ≡ N, and the bisection jumps straight
to t = 1 on that false signal. `run_annealed_smc!` rejects the combination.
"""
mutable struct AdaptiveTempering <: AnnealSchedule
    ess_target::Float64
    t_prev::Float64
    AdaptiveTempering(; ess_target=0.5) = new(ess_target, 0.0)
end

"""
    reset_schedule!(schedule) -> schedule

Return a stateful [`AnnealSchedule`](@ref) to its pre-run position so the same object
can drive another run. `run_annealed_smc!` calls this on entry.

Schedules carry a cursor — `FixedSchedule` its grid index `k`, `AdaptiveTempering` its
`t_prev` — which is left at the end of the grid when a run finishes. Reusing one
without resetting made [`done`](@ref) true immediately: the driver performed zero
blocks and returned `logZ = 0.0` with an empty trace, no error, and no warning.
"""
reset_schedule!(s::FixedSchedule)     = (s.k = 0;       s)
reset_schedule!(s::AdaptiveTempering) = (s.t_prev = 0.0; s)

# --- next_t: THE ONLY BRANCH POINT --------------------------------------------

"""
    next_t(sched, particles, path, t_prev) -> t_next

Return the next schedule point in (t_prev, 1]. `particles` carry current `logW`
and potentials `phi`; `path(t)` gives per-term weights, so an adaptive strategy
scores candidate `t` with no energy calls.

A schedule owns its own cursor: each method advances whatever state [`done`](@ref)
reads, so the driver never reaches into the schedule and a new subtype needs no change
there.
"""
function next_t(sched::FixedSchedule, particles, path, t_prev)
    sched.k += 1
    return sched.grid[sched.k + 1]       # grid[1]==0 is base; first call → grid[2]
end

function next_t(sched::AdaptiveTempering, particles, path, t_prev)
    # Adaptive tempering: pick the largest t ≤ 1 whose incremental reweighting from
    # t_prev leaves ESS ≥ ess_target*N. Routed through the energy_at seam, so it
    # works for any AnnealPath (arithmetic-only for LinearPath; energy evals for a
    # general path). In the coupled form every step resamples, so on entry all
    # logW ≈ 0 ⇒ trial_ess(t) is monotone decreasing from N; bisection is exact.
    N = length(particles)
    target = sched.ess_target * N
    buf = Vector{Float64}(undef, N)          # TODO(perf): hoist out of the hot loop
    trial_ess = function (t)
        for i in 1:N
            p = particles[i]
            buf[i] = p.logW + energy_at(path, p, t_prev) - energy_at(path, p, t)
        end
        return ess_from_logw(buf)
    end
    if trial_ess(1.0) >= target              # can reach the target in one jump
        sched.t_prev = 1.0
        return 1.0
    end
    lo, hi = t_prev, 1.0
    for _ in 1:60                            # ~60 halvings ⇒ machine precision on t
        mid = 0.5 * (lo + hi)
        trial_ess(mid) >= target ? (lo = mid) : (hi = mid)
    end
    sched.t_prev = lo                        # the schedule owns its own cursor
    return lo
end

# --- resample decision --------------------------------------------------------
should_resample(sched::FixedSchedule, ess::Float64, N::Int) = ess < sched.ess_frac * N
should_resample(sched::AdaptiveTempering, ess::Float64, N::Int) = true   # coupled

done(sched::FixedSchedule)     = sched.k + 1 >= length(sched.grid)
done(sched::AdaptiveTempering) = sched.t_prev >= 1.0

# ------------------------------------------------------------------ path + seam
"""
    AnnealPath

A schedule path through measure space. Subtypes supply the two seams the driver
routes through — [`energy_at`](@ref) (incremental weights / adaptive ESS) and
[`configure_measure!`](@ref) (rejuvenation) — so the driver itself assumes nothing
about how the measure depends on `t`.
"""
abstract type AnnealPath end

"""
    LinearPath{K}(weights_at)

Log-LINEAR path: `weights_at(t)::NTuple{K,Float64}` gives each energy term's weight
at `t` (in `scores` order); the energies themselves don't depend on `t`. So
`E(state,t) = ⟨weights_at(t), phi⟩` with the cached path-independent `phi`, and both
seams are arithmetic-only. Covers all reweighting, incl. turning terms on/off (weight
0 → positive). A bare `t ↦ NTuple` passed as `path` is wrapped as one of these.
"""
struct LinearPath{K,F} <: AnnealPath
    weights_at::F
end
LinearPath{K}(f::F) where {K,F} = LinearPath{K,F}(f)
@inline weights_at(p::LinearPath, t) = p.weights_at(t)

"""
    linear_path(target_w) -> LinearPath

The diagonal path `t ↦ t .* target_w`, reproducing the linear γ+iso schedule
`modify_measure!` used. `target_w` is the target measure's per-term weights, in
`scores` order (see `annealed_smc_scores_and_targets`). Swap for a staged-L / nonzero-base /
fitted path: any `t ↦ NTuple{K}` starting at base weights (t=0), ending at
`target_w` (t=1) — still a `LinearPath`.
"""
linear_path(target_w::NTuple{K,Float64}) where {K} =
    LinearPath{K}(t -> map(w -> t * w, target_w))

# --- the seam ----------------------------------------------------------------
# THE two functions that touch how the measure depends on t. Everything else
# (schedules, resample, rejuvenate, driver) is written against these, so a
# non-log-linear path is added by defining a new AnnealPath subtype with its own
# `energy_at` / `configure_measure!` — no change to the driver.
#
# Future GeneralPath sketch (annealing a param INSIDE an energy — an exponent,
# tolerance threshold, soft-constraint temperature — so raw energies depend on t):
#     struct GeneralPath <: AnnealPath; configure!::Function; scratch::Measure; end
#     energy_at(gp::GeneralPath, p, t)  = (gp.configure!(gp.scratch, t);
#                                          get_log_energy(p.state, gp.scratch))
#     configure_measure!(gp::GeneralPath, m, scores, t) = gp.configure!(m, t)
# Correct but pays real energy evals per t → the adaptive root-find is no longer
# free (N evals per bisection step). `phi`/refresh_potentials! are the LinearPath
# optimization and are simply unused for such a path.

"""
    energy_at(path, particle, t) -> Float64

Log-energy of `particle.state` under the measure at schedule point `t`. All
incremental-weight and adaptive-ESS math goes through this. `LinearPath` reads the
cached potentials (`⟨weights(t), phi⟩`) — arithmetic only.

This is an ENERGY: the density at `t` is `ν_t ∝ exp(−energy_at(path, p, t))`. Callers
forming a log-density ratio therefore want `energy_at(t_prev) − energy_at(t)` (see this
file's header).
"""
@inline energy_at(path::LinearPath, p::Particle, t::Float64) =
    _dot(weights_at(path, t), p.phi)

"""
    configure_measure!(path, m, scores, t)

Set measure `m` to schedule point `t` for rejuvenation MH. `LinearPath` writes the
per-term weights; a general path would also set any inside-energy parameters.
"""
function configure_measure!(path::LinearPath, m::Measure,
                            scores::NTuple{K,Function}, t::Float64) where {K}
    set_measure_weights!(m, scores, weights_at(path, t))
end

"""
    annealed_smc_scores_and_targets(measure) -> (scores::NTuple{K,Function}, target_w::NTuple{K,Float64})

Freeze the target `measure` into an ordered tuple of energy functions and their
target weights. `scores` order is the single source of truth aligning each
particle's `phi`, `path(t)`, and the rejuvenation measure.
"""
function annealed_smc_scores_and_targets(measure::Measure)
    scores = Tuple(measure.scores)
    target_w = map(e -> Float64(measure.weights[e]), scores)
    return scores, target_w
end

# ------------------------------------------------------------------ primitives
"""
    ess_from_logw(logw) -> Float64

Effective sample size (Σw)²/Σw² from unnormalized log-weights, stably.
"""
function ess_from_logw(logw::AbstractVector{Float64})
    m = maximum(logw)
    w = exp.(logw .- m)
    s = sum(w)
    return s * s / sum(abs2, w)
end

"""
    refresh_potentials!(particles, scores)

Recompute each particle's `phi = (e₁(state), …, e_K(state))`. Cheap: each energy
reads its identifier-guarded per-district cache (in sync after rejuvenation), so
this is arithmetic, not fresh log-dets. `map` over the `scores` tuple keeps it
type-stable and non-allocating (NTuple result).
"""
function refresh_potentials!(particles::Vector{Particle{K}},
                             scores::NTuple{K,Function}) where {K}
    for p in particles
        p.phi = map(e -> e(p.state)::Float64, scores)
    end
end

"""
    incremental_logweight!(particles, path, t_prev, t)

Add the SMC incremental log-weight `log[ν_t(x)/ν_{t_prev}(x)]` to each particle (at its
PRE-move state, before rejuvenation). Since `energy_at` returns an ENERGY and
`ν ∝ exp(−E)` (see this file's header), that increment is
`energy_at(t_prev) − energy_at(t)` — note the order. Generic form routes through the
[`energy_at`](@ref) seam, so it is correct for any `AnnealPath`.
"""
function incremental_logweight!(particles::Vector{Particle{K}}, path::AnnealPath,
                                t_prev::Float64, t::Float64) where {K}
    for p in particles
        p.logW += energy_at(path, p, t_prev) - energy_at(path, p, t)
    end
end

# Fast path: for a LinearPath the weight delta is shared across particles, so
# compute Δw once and dot it against each cached phi (avoids per-particle weights_at).
function incremental_logweight!(particles::Vector{Particle{K}}, path::LinearPath,
                                t_prev::Float64, t::Float64) where {K}
    dw = _sub(weights_at(path, t_prev), weights_at(path, t))
    for p in particles
        p.logW += _dot(dw, p.phi)
    end
end

# ------------------------------------------------------------------ resampling
"""
    systematic_resample(logw, rng) -> Vector{Int}

Systematic (low-variance) resampling: N parent indices drawn ∝ exp(logw).
"""
function systematic_resample(logw::AbstractVector{Float64}, rng::AbstractRNG)
    N = length(logw)
    m = maximum(logw)
    w = exp.(logw .- m); w ./= sum(w)
    positions = (rand(rng) .+ (0:N-1)) ./ N
    idx = Vector{Int}(undef, N)
    c = w[1]; i = 1
    for j in 1:N
        while positions[j] > c && i < N
            i += 1; c += w[i]
        end
        idx[j] = i
    end
    return idx
end

"""
    resample!(particles, rng) -> logZ_inc

Resample states in place ∝ exp(logW), reset all logW to 0, and return the
log-normalizer increment `logsumexp(logW) − log N`. Survivors are deep-copied
into killed slots via `clone_for_annealing` (independent `energy_data`). `phi` is
left stale here — the driver refreshes it after rejuvenation, before next use.
"""
function resample!(particles::Vector{Particle{K}}, rng::AbstractRNG) where {K}
    N = length(particles)
    logw = [p.logW for p in particles]
    m = maximum(logw)
    logZ_inc = m + log(sum(x -> exp(x - m), logw)) - log(N)

    parents = systematic_resample(logw, rng)
    # Clone survivors into fresh states first so a survivor chosen multiple times
    # never aliases another slot's energy_data.
    # TODO(perf): skip the clone when parents[j]==j and j is chosen exactly once.
    new_states = [clone_for_annealing(particles[parents[j]].state) for j in 1:N]
    for j in 1:N
        particles[j].state = new_states[j]
        particles[j].logW  = 0.0
    end
    return logZ_inc
end

# ------------------------------------------------------------------ rejuvenation
"""
    set_measure_weights!(m, scores, w)

Set `m`'s per-term weights to `w` (in `scores` order) for rejuvenation at the
current `t`. Replaces the `modify_measure!` round-trip.
"""
function set_measure_weights!(m::Measure, scores::NTuple{K,Function},
                              w::NTuple{K,Float64}) where {K}
    for e in 1:K
        m.weights[scores[e]] = w[e]
    end
end

"""
    rejuvenate!(particles, measure, proposal, steps, diags)

Apply `steps` MH moves to every particle at the FIXED current measure (no prestep
hook), in parallel across particles — the payoff over AIS's serial base chain. Each
particle has an independent state, `energy_data`, and rng, and its OWN diagnostics
`diags[i]` (run_metropolis_hastings! mutates diagnostics, so a shared object would
race), so there is no shared mutable state. Uses `Threads.@threads`; give the
process threads with `julia -t N`. The caller pins BLAS to 1 thread for the run
(the spanning-forest energy calls LAPACK — see the driver's BLAS guard).
"""
function rejuvenate!(particles::Vector{Particle{K}}, measure::Measure,
                     proposal, steps::Int,
                     diags::Vector{RunDiagnostics}) where {K}
    Threads.@threads for i in eachindex(particles)
        run_metropolis_hastings!(particles[i].state, proposal, measure, steps,
                                 particles[i].rng;
                                 output_initial=false,
                                 run_diagnostics=diags[i])
    end
end

# ------------------------------------------------------------------ driver
"""
    run_annealed_smc!(partition, proposal, measure, schedule, n_particles,
                      rejuv_steps, rng; path=nothing, init_steps=0,
                      collect_steps=0, collect_every=100,
                      writer=nothing, run_diagnostics=RunDiagnostics())
        -> (particles, logZ, trace)

Run an annealed SMC sampler. `schedule` (`FixedSchedule` | `AdaptiveTempering`) is
the sole thing distinguishing the two variants; `path` (default: `linear_path`
from `measure`'s target weights) is the swappable path geometry. Returns the final
(near-equally-weighted) particles, the log-normalizer estimate `logZ`, and a
per-block trace `(t, ess, resampled)`.

`schedule` is reset on entry ([`reset_schedule!`](@ref)), so one schedule object may
drive several runs.

`init_steps` burns each particle in independently under the base measure before
annealing starts, which is what makes the population *diverse* at t=0 — without it all
N particles are the same plan. It is **required** with `AdaptiveTempering` (an
`ArgumentError` otherwise, since ESS ≡ N would send it straight to t=1) and strongly
recommended with `FixedSchedule`, which otherwise marches its grid over a degenerate
population.

Post-anneal sample amplification ("option 1"): with `collect_steps > 0`, after the
population reaches t=1 the sampler keeps applying MH moves at the TARGET measure
(each particle is then a valid target-invariant chain) and, to `writer`, emits every
particle's state once per `collect_every` steps — `n_particles × (collect_steps ÷
collect_every)` maps, each carrying its particle's log importance weight. This yields
far more than `n_particles` samples without enlarging the population (the N chains are
independent/well-dispersed; states within one chain are autocorrelated — tune
`collect_every`). With `collect_steps = 0` (default) the writer gets one map per
particle (the annealing-final state), the original behavior.

`resample_before_collect` (default `true`, only when `collect_steps > 0`): systematic-
resample to EQUAL weights at t=1 before amplifying, so no heavy particle dominates its
whole collected chain and every emitted sample is unweighted. `logZ` is finalized
first, so this resample does not affect it. Set `false` to keep the importance weights.
"""
function run_annealed_smc!(
    partition::LinkCutPartition,
    proposal::Union{Function,Vector{Tuple{T,Function}}},
    measure::Measure,
    schedule::AnnealSchedule,
    n_particles::Int,
    rejuv_steps::Int,
    rng::AbstractRNG;
    path::Union{AnnealPath,Function,Nothing}=nothing,
    init_steps::Int=0,
    collect_steps::Int=0,
    collect_every::Int=100,
    resample_before_collect::Bool=true,
    writer::Union{Writer,Nothing}=nothing,
    run_diagnostics::RunDiagnostics=RunDiagnostics(),
    seed=nothing,
) where {T<:Real}

    # An unburned population is N identical clones, so every incremental weight is the
    # same, ESS ≡ N, and AdaptiveTempering's bisection leaps to t=1 on that false
    # signal — one block, no annealing, a meaningless logZ. FixedSchedule marches its
    # grid regardless and only loses diversity, so it is merely inadvisable there.
    if init_steps <= 0 && schedule isa AdaptiveTempering
        throw(ArgumentError(
            "run_annealed_smc! with AdaptiveTempering requires init_steps > 0: with " *
            "init_steps=$init_steps every particle is an identical clone of " *
            "`partition`, so the population ESS is always N and the schedule jumps " *
            "straight to t=1. Burn the population in first (a few hundred to a few " *
            "thousand steps), or use FixedSchedule."))
    end
    # Schedules carry a cursor and finish a run exhausted; reset so the same object can
    # be reused (otherwise `done` is true immediately and the run is a silent no-op).
    reset_schedule!(schedule)

    scores, target_w = annealed_smc_scores_and_targets(measure)
    K = length(scores)
    path === nothing && (path = linear_path(target_w))
    path isa Function && (path = LinearPath{K}(path))   # bare t↦NTuple ⇒ linear
    work_measure = deepcopy(measure)

    # Stamp this run's metadata onto the atlas header before any particle is written.
    if writer !== nothing
        stamp_run_metadata!(writer,
            annealed_smc_run_metadata(measure, schedule, n_particles, rejuv_steps;
                                      seed=seed, path=path, proposal=proposal,
                                      init_steps=init_steps, collect_steps=collect_steps,
                                      collect_every=collect_every,
                                      resample_before_collect=resample_before_collect))
    end

    # per-particle diagnostics: run_metropolis_hastings! mutates its diagnostics,
    # so threaded rejuvenation needs one object per particle (no shared state)
    rejuv_diags = RunDiagnostics[deepcopy(run_diagnostics) for _ in 1:n_particles]

    # BLAS pin: the spanning-forest energy calls LAPACK; with several Julia threads
    # each spawning its own BLAS pool the machine oversubscribes (the same issue
    # run_annealed_importance_sampling! guards against). Pin BLAS to 1 thread for the
    # threaded run and restore on exit.
    prev_blas_threads = BLAS.get_num_threads()
    Threads.nthreads() > 1 && BLAS.set_num_threads(1)
    try

    # ---- initialise population at t=0 ---------------------------------------
    # Clone `partition` N times, then burn each in independently under the BASE
    # measure (path(0.0)) for `init_steps` steps so the population is DIVERSE at
    # t=0 — the annealing analog of AIS's base chain. This matters: with
    # init_steps=0 all particles are identical, so every base→target weight is
    # equal and adaptive tempering leaps straight to t=1 on a false ESS≈N signal
    # (the fixed schedule masks it by marching the grid regardless). phi is
    # state-only, so refresh after the burn-in.
    particles = Vector{Particle{K}}(undef, n_particles)
    for i in 1:n_particles
        st = clone_for_annealing(partition)
        particles[i] = Particle{K}(st, 0.0, map(e -> e(st)::Float64, scores),
                                   PCG.PCGStateOneseq(UInt64, rand(rng, UInt64)))
    end
    if init_steps > 0
        configure_measure!(path, work_measure, scores, 0.0)
        rejuvenate!(particles, work_measure, proposal, init_steps, rejuv_diags)
        refresh_potentials!(particles, scores)
    end

    logZ = 0.0
    t_prev = 0.0
    trace = NamedTuple[]

    # ---- main SMC loop -------------------------------------------------------
    while !done(schedule)
        t = next_t(schedule, particles, path, t_prev)      # ← ONLY branch point

        incremental_logweight!(particles, path, t_prev, t) # pre-move weights
        ess = ess_from_logw([p.logW for p in particles])

        resampled = should_resample(schedule, ess, n_particles)
        resampled && (logZ += resample!(particles, rng))

        configure_measure!(path, work_measure, scores, t)
        rejuvenate!(particles, work_measure, proposal, rejuv_steps, rejuv_diags)
        refresh_potentials!(particles, scores)             # phi changed after moves

        push!(trace, (t=t, ess=ess, resampled=resampled))
        t_prev = t
    end

    # ---- final normalizer ----------------------------------------------------
    let logw = [p.logW for p in particles]
        m = maximum(logw)
        logZ += m + log(sum(x -> exp(x - m), logw)) - log(n_particles)
    end

    # ---- writer: final particles, optionally amplified as target-chains -----
    # Each particle at t=1 is a target sample; write its state with the LOG
    # importance weight `logW` as the map weight (AIS convention, so analyze.jl's
    # is_ais reader exponentiates it; after a final resample logW=0 ⇒ equal weight).
    # Build the writer with weight_type=Float64 + push_writer! the observables
    # analyze.jl reads. With collect_steps>0, keep sampling at the target and emit
    # every particle each collect_every steps (option-1 amplification). Serial write.
    if writer !== nothing
        write_particles = (suffix) -> for i in 1:n_particles
            addMap(writer.atlas.io,
                   build_output_map(writer, particles[i].state,
                                    "particle" * string(i) * suffix,
                                    particles[i].logW, run_diagnostics))
        end
        if collect_steps <= 0
            write_particles("")                   # "particle<i>" — one map/particle
        else
            # Optionally resample to EQUAL weights before amplifying, so no heavy
            # particle dominates its whole collected chain (logZ is already
            # finalized above, so discard this resample's normalizer increment).
            resample_before_collect && resample!(particles, rng)
            configure_measure!(path, work_measure, scores, 1.0)   # target measure
            every = max(collect_every, 1)
            for r in 1:div(collect_steps, every)
                # advance every chain `every` steps at the target, then record all
                rejuvenate!(particles, work_measure, proposal, every, rejuv_diags)
                write_particles("_s" * string(r * every))         # "particle<i>_s<step>"
            end
        end
    end

    return particles, logZ, trace

    finally
        Threads.nthreads() > 1 && BLAS.set_num_threads(prev_blas_threads)
    end
end
