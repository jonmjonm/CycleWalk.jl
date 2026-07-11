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
    anneal_sample!(partition, proposal, base_measure, modify_measure!,
                   steps_per_annealing, seed, run_diagnostics)

Anneal one base-chain sample toward the target measure and return its log
importance weight. `partition` is mutated in place (callers pass a deep copy of
the base-chain state), the annealing chain runs on its own
`PCG.PCGStateOneseq(UInt64, seed)`, and `base_measure` is deep-copied before being
annealed, so concurrent calls share no mutable state.
"""
function anneal_sample!(
    partition::LinkCutPartition,
    proposal::Union{Function,Vector{Tuple{T, Function}}},
    base_measure::Measure,
    modify_measure!::Function,
    steps_per_annealing::Int,
    seed::UInt64,
    run_diagnostics::RunDiagnostics
)::Float64 where T <: Real
    log_weight = MutableFloat(0.0)
    annealing_measure = deepcopy(base_measure)
    annealing_rng = PCG.PCGStateOneseq(UInt64, seed)
    run_metropolis_hastings!(partition, proposal, annealing_measure,
                             steps_per_annealing, annealing_rng;
                             output_initial=false,
                             run_diagnostics=run_diagnostics,
                             prestepf=track_weight_and_modify_measure!,
                             prestepargs=(partition, steps_per_annealing,
                                          log_weight, annealing_measure,
                                          modify_measure!))
    return log_weight.value
end

"""
    run_annealed_importance_sampling!(partition, proposal, measure, modify_measure!,
                                      total_steps, base_steps_per_sample,
                                      steps_per_annealing, rng;
                                      writer=nothing,
                                      run_diagnostics=RunDiagnostics(),
                                      ntasks=1)

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
base_steps_per_sample` annealing runs are performed.

Annealing runs execute on `ntasks` concurrent tasks (use `julia -t N` to give
them threads). Each run gets an independent RNG seeded from a draw made
sequentially on the base chain's `rng`, so the log weights are reproducible and
identical for every `ntasks`. If a `writer` is supplied — construct it with
`weight_type=Float64` — sample `i` is written as map `"sample<i>"` with its log
importance weight as the map's weight, and maps land on disk in sample order
regardless of which task finishes first.
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
    run_diagnostics::RunDiagnostics=RunDiagnostics(),
    ntasks::Int=1
)::Vector{Float64} where T <: Real
    @assert ntasks >= 1

    # The spanning-forest energy takes a log-determinant per district, which calls
    # into (OpenBLAS) LAPACK. BLAS defaults to many threads, so with several
    # annealing tasks each launching its own BLAS pool the machine is massively
    # oversubscribed (observed: a multi-task run running *slower* than a single
    # task). Annealing parallelism is already across runs, and the per-district
    # matrices are small, so pin BLAS to one thread for the duration of a
    # multi-task run and restore the previous setting on exit.
    prev_blas_threads = BLAS.get_num_threads()
    ntasks > 1 && BLAS.set_num_threads(1)
    try

    base_measure = deepcopy(measure)
    modify_measure!(base_measure, 0, 1)

    outer_steps = div(total_steps, base_steps_per_sample)
    log_weights = Vector{Float64}(undef, outer_steps)

    work_channel = Channel{Tuple{Int, LinkCutPartition, UInt64,
                                 RunDiagnostics}}(ntasks)
    map_channel = Channel{Any}(2*ntasks)

    # lone consumer of map_channel: holds back out-of-order arrivals so maps
    # reach the file in sample order
    writer_task = Threads.@spawn begin
        pending = Dict{Int, Any}()
        next_map = 1
        for (idx, out_map) in map_channel
            pending[idx] = out_map
            while haskey(pending, next_map)
                addMap(writer.atlas.io, pop!(pending, next_map))
                next_map += 1
            end
        end
    end
    bind(map_channel, writer_task)

    workers = [Threads.@spawn(
        for (idx, sample, seed, diagnostics) in work_channel
            log_weight = anneal_sample!(sample, proposal, base_measure,
                                        modify_measure!, steps_per_annealing,
                                        seed, diagnostics)
            log_weights[idx] = log_weight
            if writer !== nothing
                out_map = build_output_map(writer, sample,
                                           "sample"*string(idx),
                                           log_weight, diagnostics)
                put!(map_channel, (idx, out_map))
            end
        end) for _ in 1:ntasks]
    foreach(t -> bind(work_channel, t), workers)

    try
        for ii = 1:outer_steps
            run_metropolis_hastings!(partition, proposal, base_measure,
                                     base_steps_per_sample, rng)
            seed = rand(rng, UInt64)
            put!(work_channel, (ii, deepcopy(partition), seed,
                                deepcopy(run_diagnostics)))
        end
    catch e
        # a dead worker closes work_channel (via bind) and put! throws; fall
        # through so the waits below surface the worker's own error
        e isa InvalidStateException || rethrow()
    finally
        close(work_channel)
    end
    foreach(wait, workers)
    close(map_channel)
    wait(writer_task)

    return log_weights

    finally
        ntasks > 1 && BLAS.set_num_threads(prev_blas_threads)
    end
end
