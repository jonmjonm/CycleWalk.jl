# =============================================================================
# Run metadata builders
#
# Each sampler entry point (run_metropolis_hastings!, run_annealed_importance_sampling!,
# run_annealed_smc!) has a matching builder here that returns a `Dict{String, Any}`
# describing the run: the human-readable `"chain.run"` tag plus a nested
# `"chain.parameters"` dict of the settings that define the run (schedule, walker
# count, chain lengths, resampling, annealing endpoints, seed, proposal, …).
#
# The Atlas file writes its parameter dict (the file's third line) to disk the moment
# the `Writer` is constructed, so these builders are meant to be called BEFORE the
# writer and their result passed through `Writer(...; additional_parameters=...)`.
# The target measure itself (energies, weights, population bounds, constraints) is
# already recorded by the writer, so the builders add only run/annealing-specific info.
# =============================================================================

"""
    describe_proposal(proposal) -> Any

Human-readable description of a `run_*!` proposal argument: the proposal's registered
name (see [`name_proposal!`](@ref)) for a single closure, or a vector of
`Dict("weight"=>w, "proposal"=>name)` for a weighted mixture (in mixture order).
"""
describe_proposal(proposal::Function) = proposal_name(proposal)
function describe_proposal(proposal::Vector{Tuple{T, Function}}) where {T<:Real}
    return [Dict{String, Any}("weight" => w, "proposal" => proposal_name(f))
            for (w, f) in proposal]
end

# Label for one energy/score function: its measure description if set, else its name.
function _score_label(measure::Measure, f::Function)
    d = get(measure.descriptions, f, "")
    return isempty(d) ? string(f) : d
end

# Map an ordered `scores` tuple and matching weight tuple to a label => weight dict.
function _label_weights(scores, wtuple, measure::Measure)
    return Dict{String, Any}(_score_label(measure, scores[k]) => wtuple[k]
                             for k in eachindex(scores))
end

# All of a measure's term weights, labelled by score description/name.
function _weights_dict(measure::Measure)
    return Dict{String, Any}(_score_label(measure, f) => w
                             for (f, w) in measure.weights)
end

_total_steps(steps::Integer) = steps
_total_steps(steps::Tuple{Integer, Integer}) = steps[2] - steps[1]

# Record `seed` under "seed" only when one was supplied (the automatic path may not
# have it, since a run function sees only the rng object, not its original seed).
_maybe_seed!(params, seed) = (seed !== nothing && (params["seed"] = seed); params)

"""
    metropolis_hastings_run_metadata(proposal, steps;
                                     seed=nothing, output_freq=250, extra=Dict())

Build the run-metadata dict for a standard Metropolized CycleWalk run
([`run_metropolis_hastings!`](@ref)). `run_metropolis_hastings!` stamps this
automatically; call it directly only to precompute the dict. `extra` is merged into the
parameters dict; `seed` is recorded when given.
"""
function metropolis_hastings_run_metadata(
    proposal::Union{Function, Vector{Tuple{T, Function}}},
    steps::Union{Integer, Tuple{Integer, Integer}};
    seed=nothing,
    output_freq::Int=250,
    extra::AbstractDict=Dict{String, Any}(),
) where {T<:Real}
    params = Dict{String, Any}(
        "function"    => "run_metropolis_hastings!",
        "steps"       => _total_steps(steps),
        "output_freq" => output_freq,
        "proposal"    => describe_proposal(proposal),
        "threads"     => Threads.nthreads(),
    )
    _maybe_seed!(params, seed)
    merge!(params, Dict{String, Any}(extra))
    return Dict{String, Any}(
        "chain.run"        => "standard metropolized CycleWalk",
        "chain.parameters" => params,
    )
end

"""
    annealed_importance_sampling_run_metadata(measure, modify_measure!, total_steps,
        base_steps_per_sample, steps_per_annealing;
        seed=nothing, proposal=nothing, ntasks=1, schedule="linear", extra=Dict())

Build the run-metadata dict for an annealed-importance-sampling CycleWalk run
([`run_annealed_importance_sampling!`](@ref)). `run_annealed_importance_sampling!`
stamps this automatically. The base and target annealing weights are recorded by
evaluating `modify_measure!(copy, 0, 1)` (base) against the target `measure` (both
labelled by score); `schedule` names the interpolation shape (the closure body itself
cannot be introspected).
"""
function annealed_importance_sampling_run_metadata(
    measure::Measure,
    modify_measure!::Function,
    total_steps::Integer,
    base_steps_per_sample::Integer,
    steps_per_annealing::Integer;
    seed=nothing,
    proposal::Union{Function, Vector, Nothing}=nothing,
    ntasks::Int=1,
    schedule::String="linear",
    extra::AbstractDict=Dict{String, Any}(),
)
    base = deepcopy(measure)
    modify_measure!(base, 0, 1)
    params = Dict{String, Any}(
        "function"              => "run_annealed_importance_sampling!",
        "schedule"              => schedule,
        "total_steps"           => total_steps,
        "base_steps_per_sample" => base_steps_per_sample,
        "steps_per_annealing"   => steps_per_annealing,
        "n_samples"             => div(total_steps, base_steps_per_sample),
        "ntasks"                => ntasks,
        "threads"               => Threads.nthreads(),
        "weights.base"          => _weights_dict(base),
        "weights.target"        => _weights_dict(measure),
    )
    _maybe_seed!(params, seed)
    proposal !== nothing && (params["proposal"] = describe_proposal(proposal))
    merge!(params, Dict{String, Any}(extra))
    return Dict{String, Any}(
        "chain.run"        => "annealed importance sampling CycleWalk",
        "chain.parameters" => params,
    )
end

# Schedule-specific fields for an annealed-SMC run.
function _schedule_metadata(s::FixedSchedule)
    return Dict{String, Any}(
        "kind"       => "fixed",
        "grid"       => copy(s.grid),
        "blocks"     => length(s.grid) - 1,
        "ess_frac"   => s.ess_frac,
        "resampling" => "adaptive: resample when ESS < ess_frac * N",
    )
end
function _schedule_metadata(s::AdaptiveTempering)
    return Dict{String, Any}(
        "kind"        => "adaptive",
        "ess_target"  => s.ess_target,
        "resampling"  => "every step (coupled)",
    )
end

"""
    annealed_smc_run_metadata(measure, schedule, n_particles, rejuv_steps;
        seed=nothing, path=nothing, proposal=nothing, init_steps=0, collect_steps=0,
        collect_every=100, resample_before_collect=true, extra=Dict())

Build the run-metadata dict for an annealed-SMC CycleWalk run
([`run_annealed_smc!`](@ref)). `run_annealed_smc!` stamps this automatically. The
schedule (fixed grid / adaptive tempering, with its resampling rule), the annealing-path
endpoints (`weights_at(path, 0)` and `weights_at(path, 1)`, labelled by score), the
walker count, rejuvenation/init lengths, and any post-anneal amplification settings are
all recorded. `path` mirrors the driver's default (`linear_path` from the target weights
when `nothing`).
"""
function annealed_smc_run_metadata(
    measure::Measure,
    schedule::AnnealSchedule,
    n_particles::Integer,
    rejuv_steps::Integer;
    seed=nothing,
    path::Union{AnnealPath, Function, Nothing}=nothing,
    proposal::Union{Function, Vector, Nothing}=nothing,
    init_steps::Int=0,
    collect_steps::Int=0,
    collect_every::Int=100,
    resample_before_collect::Bool=true,
    extra::AbstractDict=Dict{String, Any}(),
)
    scores, target_w = annealed_smc_scores_and_targets(measure)
    K = length(scores)
    p = path === nothing ? linear_path(target_w) :
        path isa Function ? LinearPath{K}(path) : path

    params = Dict{String, Any}(
        "function"                => "run_annealed_smc!",
        "schedule"                => _schedule_metadata(schedule),
        "n_particles"             => n_particles,
        "rejuv_steps"             => rejuv_steps,
        "init_steps"              => init_steps,
        "collect_steps"           => collect_steps,
        "collect_every"           => collect_every,
        "resample_before_collect" => resample_before_collect,
        "threads"                 => Threads.nthreads(),
        "weights.base"            => _label_weights(scores, weights_at(p, 0.0), measure),
        "weights.target"          => _label_weights(scores, weights_at(p, 1.0), measure),
    )
    _maybe_seed!(params, seed)
    proposal !== nothing && (params["proposal"] = describe_proposal(proposal))
    merge!(params, Dict{String, Any}(extra))
    return Dict{String, Any}(
        "chain.run"        => "annealed sequential monte carlo CycleWalk",
        "chain.parameters" => params,
    )
end
