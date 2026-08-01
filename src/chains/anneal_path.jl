# =============================================================================
# AnnealPath — the shared "path through measure space" seam
#
# Extracted from annealed_smc.jl (see docs/parallel_tempering_implementation.md,
# task 1) so parallel tempering can reuse it without `include`ing the SMC file.
# Annealed SMC and parallel tempering both temper a `Measure` along t/β via the
# same two seams: `energy_at` (arithmetic-only incremental weights, for anything
# with a `phi` field) and `configure_measure!` (rejuvenation/advance).
#
# SIGN CONVENTION: `get_log_energy` (and therefore `energy_at` below) returns an
# ENERGY, not a log-density. The target is ν(τ) ∝ exp(−E(τ,t)). See
# annealed_smc.jl's header and §6 of the Cycle Walk paper.
# =============================================================================

# ------------------------------------------------------------------ tuple utils
@inline _dot(a::NTuple{K,Float64}, b::NTuple{K,Float64}) where {K} = sum(map(*, a, b))
@inline _sub(a::NTuple{K,Float64}, b::NTuple{K,Float64}) where {K} = map(-, a, b)

# ------------------------------------------------------------------ path + seam
"""
    AnnealPath

A schedule path through measure space. Subtypes supply the two seams samplers
route through — [`energy_at`](@ref) (incremental weights / adaptive ESS) and
[`configure_measure!`](@ref) (rejuvenation) — so a driver itself assumes nothing
about how the measure depends on `t`.

`t` is the ONE path parameter this whole file's seam is built around — there is no
second concept hiding under a different name. Annealed SMC calls it `t` because a
single population evolves along one schedule value over the course of a run.
Parallel tempering calls the same argument **β** (see `BetaLattice` in
`parallel_tempering_types.jl`) because a PT run holds several fixed values of it at
once, one per rung, in the replica-exchange/statistical-physics convention the
design docs cite. `ensemble.lattice[replica.beta_index]` IS the `Float64` passed as
`t` to `weights_at`/`energy_at`/`configure_measure!` — same slot, two names for the
two contexts it's used in.
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
`scores` order (see `measure_scores_and_targets`). Swap for a staged-L / nonzero-base /
fitted path: any `t ↦ NTuple{K}` starting at base weights (t=0), ending at
`target_w` (t=1) — still a `LinearPath`.
"""
linear_path(target_w::NTuple{K,Float64}) where {K} =
    LinearPath{K}(t -> map(w -> t * w, target_w))

# --- the seam ----------------------------------------------------------------
# THE two functions that touch how the measure depends on t. Everything else
# (schedules, resample, rejuvenate, swap mechanics, drivers) is written against
# these, so a non-log-linear path is added by defining a new AnnealPath subtype
# with its own `energy_at` / `configure_measure!` — no change to the callers.

"""
    energy_at(path, phi, t) -> Float64

Log-energy of cached per-term potentials `phi` under the measure at schedule point
`t`. All incremental-weight, adaptive-ESS, and swap-acceptance math goes through
this. `LinearPath` reads the cached potentials (`⟨weights(t), phi⟩`) — arithmetic
only.

This is an ENERGY: the density at `t` is `ν_t ∝ exp(−energy_at(path, phi, t))`.
Callers forming a log-density ratio therefore want
`energy_at(t_prev) − energy_at(t)`.

Takes the raw `phi::NTuple` rather than a particle/replica so both `annealed_smc.jl`
and `parallel_tempering.jl` share one implementation; each defines a thin forwarding
method for its own state type (e.g. `energy_at(path, p::Particle, t) =
energy_at(path, p.phi, t)`).
"""
@inline energy_at(path::LinearPath, phi::NTuple{K,Float64}, t::Float64) where {K} =
    _dot(weights_at(path, t), phi)

"""
    configure_measure!(path, m, scores, t)

Set measure `m` to schedule point `t` for rejuvenation/advance MH. `LinearPath`
writes the per-term weights; a general path would also set any inside-energy
parameters.
"""
function configure_measure!(path::LinearPath, m::Measure,
                            scores::NTuple{K,Function}, t::Float64) where {K}
    set_measure_weights!(m, scores, weights_at(path, t))
end

"""
    set_measure_weights!(m, scores, w)

Set `m`'s per-term weights to `w` (in `scores` order) for rejuvenation/advance at the
current `t`. Replaces the `modify_measure!` round-trip.
"""
function set_measure_weights!(m::Measure, scores::NTuple{K,Function},
                              w::NTuple{K,Float64}) where {K}
    for e in 1:K
        m.weights[scores[e]] = w[e]
    end
end

"""
    measure_scores_and_targets(measure) -> (scores::NTuple{K,Function}, target_w::NTuple{K,Float64})

Freeze the target `measure` into an ordered tuple of energy functions and their
target weights. `scores` order is the single source of truth aligning cached
potentials (`phi`), `path(t)`, and the rejuvenation/advance measure.
"""
function measure_scores_and_targets(measure::Measure)
    scores = Tuple(measure.scores)
    target_w = map(e -> Float64(measure.weights[e]), scores)
    return scores, target_w
end

# Old name, kept as an exported alias: used by test/test_run_metadata.jl,
# examples/run_asmc_toml.jl, and examples/validation/*.jl.
const annealed_smc_scores_and_targets = measure_scores_and_targets

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
