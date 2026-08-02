"""
    run_metropolis_hastings!(partition, proposal, measure, steps, rng;
                             writer=nothing, output_freq=250,
                             run_diagnostics=RunDiagnostics(),
                             prestepf=(x...)->nothing, prestepargs=(),
                             output_initial=true, sample_hook=nothing, weight=1)

Run the Metropolis–Hastings sampler in place on `partition`. Each step draws a
proposal (a single closure, or one sampled from a weighted mixture whose weights must
sum to 1), multiplies its proposal probability ratio by the energy ratio from
`measure`, and accepts with that probability — applying accepted moves with
[`update_partition!`](@ref). `steps` is a step count or an `(initial, final)` range.
Every `output_freq` steps the current plan and any registered observables/diagnostics
are written to `writer`; set `output_initial=false` to suppress the map otherwise
written before the first step. `prestepf(step, prestepargs...)` is called before
each proposal — annealed importance sampling uses it to anneal `measure` and
accumulate the log importance weight. `weight` is recorded as each output map's
sampling weight; passing a [`MutableFloat`](@ref) lets `prestepf` update it as the
run progresses.

When supplied, `sample_hook(partition, step)` is called immediately after every
regular `output_freq` sample is written. It is not called for the optional initial
output. This supports streaming, in-process diagnostic accumulation without reading
the Atlas back from disk.
"""
function run_metropolis_hastings!(
    partition::LinkCutPartition,
    proposal::Union{Function,Vector{Tuple{T, Function}}},
    measure::Measure,
    steps::Union{Int,Tuple{Int,Int}},
    rng::AbstractRNG;
    writer::Union{Writer, Nothing}=nothing,
    output_freq::Int=250,
    run_diagnostics::RunDiagnostics=RunDiagnostics(),
    prestepf::Function=(x...)->nothing,
    prestepargs::Tuple=(),
    output_initial::Bool=true,
    sample_hook::Union{Nothing, Function}=nothing,
    weight::Union{Real, MutableFloat}=1,
    seed=nothing
) where T <: Real
    # Want this, but need to redefine it: precompute_node_tree_counts!(partition)
    check_proposals_weights(proposal)

    # Stamp this run's metadata onto the atlas header before any map is written. Inner
    # annealing chains (AIS/ASMC rejuvenation) pass writer=nothing, so only a genuine
    # standalone MH run records the "standard metropolized CycleWalk" tag.
    if writer !== nothing
        stamp_run_metadata!(writer,
            metropolis_hastings_run_metadata(proposal, steps;
                                             seed=seed, output_freq=output_freq))
    end

    initial_step, final_step = set_step_bounds(steps)
    if (initial_step == 0 || initial_step == 1) && output_initial
        output(partition, measure, initial_step, 0, writer; weight=weight)
    end

    for step = initial_step:final_step
        prestepf(step, prestepargs...)
        proposal!, proposal_index = get_random_proposal(proposal, rng)
        proposal_diagnostics = get_proposal_diagnostics(run_diagnostics, 
                                                        proposal!)
        p, update = proposal!(partition, rng, diagnostics=proposal_diagnostics)
        if p == 0
            if mod(step, output_freq) == 0 && step != initial_step
                output(partition, measure, step, 0, writer, run_diagnostics;
                       weight=weight)
                sample_hook !== nothing && sample_hook(partition, step)
            end
            continue
        end
        p *= get_delta_energy(partition, measure, update)
        update_acceptance_ratio_diagnostic!(proposal_diagnostics, p)
        # post_step!(proposal_diagnostics, p, partition, measure, writer, update)

        if rand(rng) < p
            update_partition!(partition, update)
        end
        if mod(step, output_freq) == 0
            output(partition, measure, step, 0, writer, run_diagnostics;
                   weight=weight)
            sample_hook !== nothing && sample_hook(partition, step)
        end
    end
end

"""
    update_partition!(partition, update)

Apply an accepted [`Update`](@ref) to `partition` in place: perform the link/cut
edits, recompute the two changed districts' roots (swapping their labels if
`update.swap_link11` requires it), commit the new cross-district edges and the
node→district assignment, bump `partition.identifier`, and roll each cached energy's
data forward via [`update_energy_data!`](@ref).
"""
function update_partition!(
    partition::LinkCutPartition,
    update::Update{T}
) where T <: Int
    for cut in update.cuts
        evert!(partition.lct.nodes[cut[1]])
        cut!(partition.lct.nodes[cut[2]])
    end
    for link in update.links
        evert!(partition.lct.nodes[link[1]])
        link!(partition.lct.nodes[link[1]], partition.lct.nodes[link[2]])
    end

    distPair = update.changed_districts
    new_roots = (find_root!(partition.lct.nodes[update.cuts[1][1]]),
                 find_root!(partition.lct.nodes[update.cuts[1][2]]))
    # swap if needed
    l11node = partition.lct.nodes[update.links[1][1]]
    l11dist_cur = partition.node_to_dist[update.links[1][1]]
    r11_new = find_root!(l11node)
    r11_root_ind_new = (r11_new != new_roots[1]) + 1
    r11_root_ind_cur = (l11dist_cur != distPair[1]) + 1
    if (r11_root_ind_new == r11_root_ind_cur) ⊻ !update.swap_link11
        new_roots = (new_roots[2], new_roots[1])
    end


    # modify roots
    old_root1 = partition.district_roots[distPair[1]]
    old_root2 = partition.district_roots[distPair[2]]
    partition.district_roots[distPair[1]] = new_roots[1].vertex
    partition.district_roots[distPair[2]] = new_roots[2].vertex
    delete!(partition.roots_to_district, old_root1)
    delete!(partition.roots_to_district, old_root2)
    partition.roots_to_district[new_roots[1].vertex] = distPair[1]
    partition.roots_to_district[new_roots[2].vertex] = distPair[2]

    # new_cross_d_edg 
    del_keys = [k for k in keys(partition.cross_district_edges) 
                if (distPair[1] in k || distPair[2] in k) && 
                    !haskey(update.new_cross_d_edg, k)]
    for dk in del_keys
        delete!(partition.cross_district_edges, dk)
    end
    for (dists, edgeset) in update.new_cross_d_edg
        partition.cross_district_edges[dists] = edgeset
    end

    partition.node_to_dist .= partition.node_to_dist_update

    partition.identifier += 1
    for (key, eData) in partition.energy_data
        update_energy_data!(eData, partition, update)
    end 
end


"""
    set_step_bounds(steps)

Normalize the `steps` argument to an `(initial_step, final_step)` tuple: a bare count
`n` becomes `(1, n)`, while a tuple is returned unchanged.
"""
@inline function set_step_bounds(steps::Union{Tuple{T,T}, T}) where T<:Int
    if typeof(steps)<:Tuple
        return steps
    else
        return 1, steps
    end
end


"""
    get_random_proposal(proposal, rng)

Select a proposal to attempt this step, returning `(proposal_fn, index)`. A single
function is returned as-is (index 1); a weighted mixture is inverse-CDF sampled by
walking the running cumulative weight, without allocating a cumsum.
"""
@inline function get_random_proposal(
    proposal::Union{Function, Vector{Tuple{T, Function}}},
    rng::AbstractRNG
) where T <: Real
    if !(typeof(proposal) <: Vector)
        return proposal, 1
    end
    # Inverse-CDF sample without allocating a weights vector / cumsum / BitVector
    # (this runs every MCMC step). Walk the running cumulative weight directly.
    u = rand(rng)
    cum = zero(T)
    @inbounds for i = 1:length(proposal)
        cum += proposal[i][1]
        if cum > u
            return proposal[i][2], i
        end
    end
    # Fall back to the last proposal (guards float round-off when the weights
    # sum to 1 and u is just below the final boundary).
    return proposal[end][2], length(proposal)
end


"""
    check_proposals_weights(proposal)

Validate a proposal mixture before a run: throw an `ArgumentError` if the mixture
weights do not sum to 1. A single-function proposal passes trivially.
"""
@inline function check_proposals_weights(
    proposal::Union{Function, Vector{Tuple{T, Function}}}
) where T<:Real
    if !(typeof(proposal) <: Vector)
        return
    end
    weight_sum = sum(proposal[i][1] for i = 1:length(proposal))
    if weight_sum != 1
        throw(
            ArgumentError(
                "Chance of choosing a proposal must sum to 1",
            ),
        )
    end
end

"""
    post_step!(proposal_diagnostics, acceptance_ratio, partition, measure, writer, update)

Post-step diagnostics hook (currently records the acceptance ratio into
`proposal_diagnostics`). Not called on the main MCMC path; retained for diagnostic
experimentation.
"""
function post_step!(
    proposal_diagnostics::Union{ProposalDiagnostics, Nothing},
    acceptance_ratio::Float64,
    partition::LinkCutPartition,
    measure::Measure,
    writer::Writer,
    update::Update{T}
) where T<:Int
    update_acceptance_ratio_diagnostic!(proposal_diagnostics, acceptance_ratio)
    # for report_func in values(writer.map_output_data)
    # for (desc, report_func) in writer.map_output_data
    #     @show desc, report_func∈measure.scores, report_func
    # end
end
