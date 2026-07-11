"""
    track_weight_and_modify_measure!(cur_step, partition, total_steps, weight,
                                     measure, modify_measure!)

Pre-step hook for annealed importance sampling. Advance `measure` one annealing
step via `modify_measure!(measure, cur_step, total_steps)` and accumulate the log
importance weight in place: `weight.value` gains the change in the log-energy of
the current `partition` under the measure before vs. after the modification.
"""
function track_weight_and_modify_measure!(
    cur_step::Int,
    partition::LinkCutPartition,
    total_steps::Int,
    weight::MutableFloat,
    measure::Measure,
    modify_measure!::Function
)
    e1 = get_log_energy(partition, measure)
    modify_measure!(measure, cur_step, total_steps)
    e2 = get_log_energy(partition, measure)
    weight.value += e2 - e1
end

"""
    run_annealed_importance_sampling!(partition, proposal, measure, modify_measure!,
                                      total_steps, base_steps_per_sample,
                                      steps_per_annealing, rng;
                                      writer=nothing,
                                      run_diagnostics=RunDiagnostics())

Run annealed importance sampling (AIS) and return the vector of log importance
weights, one per annealing run.

`modify_measure!(measure, cur_step, total_steps)` defines the annealing schedule:
called with `(0, 1)` it must set `measure` to the base measure, and stepping
`cur_step` from 1 to `total_steps` it must interpolate toward the target measure.
The base chain runs on `partition` under the base measure for
`base_steps_per_sample` Metropolis–Hastings steps between samples; each sample is
then deep-copied and annealed toward the target for `steps_per_annealing` steps
while [`track_weight_and_modify_measure!`](@ref) accumulates its log weight. The
base chain takes `total_steps` steps in all, so `total_steps ÷
base_steps_per_sample` annealing runs are performed. If a `writer` is supplied,
each annealed plan is written at the end of its run (the associated importance
weight is currently only available in the returned vector).
"""
function run_annealed_importance_sampling!(
    partition::LinkCutPartition,
    proposal::Union{Function,Vector{Tuple{T, Function}}},
    measure::Measure,
    modify_measure!::Function,
    total_steps::Int,
    base_steps_per_sample::Int,
    steps_per_annealing::Int,
    rng::AbstractRNG;
    writer::Union{Writer, Nothing}=nothing,
    run_diagnostics::RunDiagnostics=RunDiagnostics()
)::Vector{Float64} where T <: Real
    base_measure = deepcopy(measure)
    modify_measure!(base_measure, 0, 1)

    outer_steps = div(total_steps, base_steps_per_sample)
    log_weights = Vector{Float64}(undef, outer_steps)
    for ii = 1:outer_steps
        run_metropolis_hastings!(partition, proposal, base_measure,
                                 base_steps_per_sample, rng)
        partition_to_anneal = deepcopy(partition)
        log_weight = MutableFloat(0.0)
        annealing_measure = deepcopy(base_measure)
        annealing_rng = deepcopy(rng)
        run_metropolis_hastings!(partition_to_anneal, proposal,
                                 annealing_measure, steps_per_annealing,
                                 annealing_rng; writer=writer,
                                 output_freq=steps_per_annealing,
                                 output_initial=false,
                                 run_diagnostics=run_diagnostics,
                                 prestepf=track_weight_and_modify_measure!,
                                 prestepargs=(partition_to_anneal,
                                              steps_per_annealing, log_weight,
                                              annealing_measure,
                                              modify_measure!))
        log_weights[ii] = log_weight.value
    end
    return log_weights
end
