# =============================================================================
# Parallel tempering — core types
#
# See docs/parallel_tempering_implementation.md (task 2) for the design rationale.
# =============================================================================

# ------------------------------------------------------------------ beta lattice
"""
    BetaLattice(betas)

The inverse-temperature lattice: rung `k` sits at path point `betas[k]`. Strictly
increasing, `betas[1] >= 0`, `betas[end] == 1.0` (rung `M` is the target — note that is
"the configured target", not γ = 1; see the module docs).

Rung 1 is the HOTTEST (most tempered, easiest to sample); rung `M` is the COLDEST
(the target). Samples for downstream analysis come from rung `M`.

Each stored value is the SAME `t` an [`AnnealPath`](@ref) takes — "β" here is just the
replica-exchange name for it, used because a `BetaLattice` holds several fixed values
of that one parameter at once (see `AnnealPath`'s docs for the full note). A replica's
current β is `lattice[replica.beta_index]`, passed straight to `weights_at`/
`energy_at`/`configure_measure!` as `t`.
"""
struct BetaLattice
    betas::Vector{Float64}
    function BetaLattice(betas::AbstractVector{<:Real})
        b = collect(Float64, betas)
        length(b) >= 1 || throw(ArgumentError("a lattice needs at least one rung"))
        issorted(b) && allunique(b) ||
            throw(ArgumentError("betas must be strictly increasing, got $b"))
        b[1] >= 0 || throw(ArgumentError("betas must be >= 0, got $(b[1])"))
        b[end] == 1.0 ||
            throw(ArgumentError("the last rung must be the target (beta = 1.0), got $(b[end])"))
        return new(b)
    end
end

Base.length(l::BetaLattice) = length(l.betas)
Base.getindex(l::BetaLattice, k::Int) = l.betas[k]
Base.lastindex(l::BetaLattice) = length(l.betas)

"""
    linear_betas(M) -> BetaLattice

`M` rungs evenly spaced on [0, 1].
"""
linear_betas(M::Int) = BetaLattice(range(0, 1; length=M))

"""
    geometric_betas(M; beta_min=0.05) -> BetaLattice

`M` rungs geometrically spaced in TEMPERATURE (= 1/β) between `beta_min` and 1 — the
ladder shape of `ParallelTempering.java:37`, which builds the same set of values in the
opposite order (it runs cold-to-hot; we run hot-to-cold, so rung 1 is always the most
tempered). A test comparing the two must `sort` or `reverse` one of them.

Cannot include β = 0 (infinite temperature); use `linear_betas`, or prepend 0
explicitly, if you want the exact ν_0 end.
"""
function geometric_betas(M::Int; beta_min::Float64=0.05)
    0 < beta_min < 1 || throw(ArgumentError("beta_min must be in (0,1), got $beta_min"))
    M >= 2 || return BetaLattice([1.0])
    logT = range(log(1/beta_min), log(1.0); length=M)   # temperature high -> low
    return BetaLattice([1/exp(x) for x in logT])
end

# ------------------------------------------------------------------ replica
"""
    Replica{K}

One PT walker. `state`/`rng`/`diagnostics`/`work_measure` are private to this replica
and must never be shared (see the module invariants). `beta_index` is the rung it
currently occupies; `phi` is the cached raw per-term energies of `state`, refreshed
once per block.

`work_measure` is a private scratch `Measure` that `configure_measure!` mutates in
place to advance this replica at its own rung (see `advance_replica!` in
`pt_backends.jl`) — the PT analogue of annealed SMC's per-particle `RunDiagnostics`
rule: a shared `work_measure` would race under any concurrent backend.

`replica_id` is fixed for the life of the run and follows the walker up and down the
ladder — it is what makes round-trip diagnostics possible. `bath_swaps` counts heat-bath
replacements, which break the walker's lineage.
"""
mutable struct Replica{K}
    state::LinkCutPartition
    beta_index::Int
    phi::NTuple{K,Float64}
    rng::PCG.PCGStateOneseq
    diagnostics::RunDiagnostics
    work_measure::Measure
    replica_id::Int
    bath_swaps::Int
    last_end_visited::Int      # 0 = neither end yet; else 1 or M — for round trips
end

# ------------------------------------------------------------------ ensemble
"""
    PTEnsemble{K}

`replicas` is indexed by WALKER id and is never permuted; `walker_at_rung[k]` is the
walker currently at rung `k`. The invariant

    replicas[walker_at_rung[k]].beta_index == k    for every k

is checked by [`check_ensemble`](@ref) and must hold on entry to and exit from every
swap round. A swap exchanges rung labels, never array slots — see the module docs for
why (it is what lets one swap implementation serve both the threaded and distributed
backends).
"""
mutable struct PTEnsemble{K}
    replicas::Vector{Replica{K}}
    walker_at_rung::Vector{Int}
    lattice::BetaLattice
    path::AnnealPath
    scores::NTuple{K,Function}
end

nrungs(e::PTEnsemble) = length(e.lattice)

"""
    check_ensemble(e) -> Bool

Assert the rung/walker bijection. Call it in tests and under `@debug`; it is O(M).
"""
function check_ensemble(e::PTEnsemble)
    M = nrungs(e)
    length(e.walker_at_rung) == M || return false
    sort(e.walker_at_rung) == collect(1:M) || return false
    return all(k -> e.replicas[e.walker_at_rung[k]].beta_index == k, 1:M)
end

# ------------------------------------------------------------------ heat bath
"""
    HeatBath(source_path, measure, samples, rung, graph)

Independent draws from a reference `measure`, read from a stored Atlas, exchanged
against `rung` (default: the hottest, rung 1) on the rounds the even/odd pattern leaves
it idle. `graph` is the `MultiLevelGraph` a sample's districting is rebuilt against
(`MultiLevelPartition(graph, assignment)`) — a `LinkCutPartition` only carries the
raw `BaseGraph`, not the hierarchical graph reconstruction needs, so the bath must
carry its own (the SAME graph the running ensemble uses).

Each stored sample is consumed AT MOST ONCE — sampling with replacement from a small
pool would correlate the exchanges and break the independence the move relies on. The
run errors rather than wrapping around when the pool is exhausted, so size
`length(samples) >= n_rounds ÷ 2`.

Validity note: the move is an exact MH step only if each `y` is an independent draw from
`measure`. Drawn from a stored MCMC atlas they are approximately independent at best —
use a burn-in and a wide stride, and treat the resulting bias as a modelling assumption.

Construction (`parse_bath_measure`, `parse_bath_samples`) and the exchange move
(`try_heat_bath!`) live in `pt_heat_bath.jl`.
"""
struct HeatBath
    source_path::String
    measure::Measure
    samples::Vector{Dict{Tuple{Vararg{String}},Int}}   # districting assignments
    rung::Int
    graph::MultiLevelGraph
end

# ------------------------------------------------------------------ diagnostics
"""
    PTDiagnostics(M)

Per-run PT diagnostics. STATEFUL — `run_parallel_tempering!` resets it on entry (see
[`reset_pt_diagnostics!`](@ref)), so one object may drive several runs. This mirrors the
`reset_schedule!` fix in `annealed_smc.jl`, which exists because a stateful schedule
silently made a second run a no-op.

- `attempts` / `accepts`: per adjacent pair `(k, k+1)`, length `M-1`
- `accept_prob_sum`: running sum of min(1, α) per pair, so the mean can be reported
- `occupancy[w, k]`: blocks walker `w` spent at rung `k`. Flat rows mean a well-placed
  lattice; a walker stuck in one band means the lattice is too coarse there.
- `round_trips[w]`: completed rung-M -> rung-1 -> rung-M journeys. **The single most
  informative PT diagnostic** and the one neither reference implementation computes.
- `bath_attempts` / `bath_accepts`
- `straggler_gap`: per block, `max - mean` of the per-replica advance time. Use it to
  size `swap_interval`.
"""
mutable struct PTDiagnostics
    attempts::Vector{Int}
    accepts::Vector{Int}
    accept_prob_sum::Vector{Float64}
    occupancy::Matrix{Int}
    round_trips::Vector{Int}
    bath_attempts::Int
    bath_accepts::Int
    straggler_gap::Vector{Float64}
end

PTDiagnostics(M::Int) = PTDiagnostics(zeros(Int, max(M-1,0)), zeros(Int, max(M-1,0)),
                                      zeros(Float64, max(M-1,0)), zeros(Int, M, M),
                                      zeros(Int, M), 0, 0, Float64[])

"""
    reset_pt_diagnostics!(d) -> d

Return a stateful [`PTDiagnostics`](@ref) to its pre-run position so the same object
can drive another run. `run_parallel_tempering!` calls this on entry.
"""
function reset_pt_diagnostics!(d::PTDiagnostics)
    fill!(d.attempts, 0); fill!(d.accepts, 0); fill!(d.accept_prob_sum, 0.0)
    fill!(d.occupancy, 0); fill!(d.round_trips, 0)
    d.bath_attempts = 0; d.bath_accepts = 0
    empty!(d.straggler_gap)
    return d
end

"""
    swap_rate(d) -> Vector{Float64}

Empirical acceptance rate per adjacent pair. A well-placed lattice has these roughly
EQUAL across pairs — that is the tuning target (design §1.10), not "as high as possible".
"""
swap_rate(d::PTDiagnostics) =
    [d.attempts[k] == 0 ? NaN : d.accepts[k]/d.attempts[k] for k in eachindex(d.attempts)]
