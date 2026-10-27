"""
    BalanceData <: AbstractEnergyData

Cache of each district's total population, used to check the population
(balance) constraint incrementally. `dist_pops` is the current per-district
population and `dist_pops_update` the proposed values; `identifier` /
`update_identifier` track which partition state and which proposed update the two
buffers correspond to.
"""
mutable struct BalanceData <: AbstractEnergyData
    identifier::Int64
    update_identifier::Vector{Int64}
    dist_pops::Vector{Float64}
    dist_pops_update::Vector{Float64}
end

"""
    BalanceData(partition)

Build a [`BalanceData`](@ref) for `partition`, computing each district's total
population from scratch by summing the cached per-node populations.
"""
function BalanceData(partition::LinkCutPartition)
    identifier = partition.identifier
    update_identifier = Vector{Int64}(undef, 0)
    dist_pops = zeros(Float64, partition.num_dists)
    dist_pops_update = zeros(Float64, partition.num_dists)

    node_pops = partition.node_pops          # cached concrete Float64 populations
    for ni in 1:partition.graph.num_nodes
        di = partition.node_to_dist[ni]
        dist_pops[di] += node_pops[ni]
    end

    return BalanceData(identifier, update_identifier, dist_pops,
                       dist_pops_update)
end

"""
    reset_update_buffer!(data::BalanceData)

Forget which update `dist_pops_update` was computed for, so the next query with an
`update` recomputes it.
"""
function reset_update_buffer!(data::BalanceData)
    empty!(data.update_identifier)
    return
end

"""
    update_balance_data!(data, partition, update)

Recompute `data.dist_pops_update` for the proposed plan: copy the current district
populations, then re-sum the populations of the changed districts from the proposed
assignment (`node_to_dist_update`). Records `update.identifier` so the proposed
buffer is not recomputed for the same update.
"""
function update_balance_data!(
    data::BalanceData,
    partition::LinkCutPartition,
    update::Update
)
    data.update_identifier = copy(update.identifier)
    node_pops = partition.node_pops          # cached concrete Float64 populations

    changed_districts = collect(update.changed_districts)
    data.dist_pops_update .= data.dist_pops
    for di in changed_districts
        data.dist_pops_update[di] = 0.0
    end

    for ni in 1:partition.graph.num_nodes
        di = partition.node_to_dist_update[ni]
        if di in changed_districts
            data.dist_pops_update[di] += node_pops[ni]
        end
    end
end

"""
    satisfies_constraint(partition, population_constraint, districts=...; update=nothing)::Bool

Return whether every district in `districts` has population within
`[min_pop, max_pop]`, using the cached [`BalanceData`](@ref) (rebuilt if stale, and
extended with the proposed populations when `update` is given). With an `update`,
only the changed districts are checked.
"""
function satisfies_constraint(
    partition::LinkCutPartition,
    population_constraint::PopulationConstraint,
    districts::Union{Tuple{Vararg{T}}, Vector{T}}
        = collect(1:partition.num_dists);
    update::Union{Update, Nothing}=nothing
)::Bool where T<:Int
    if !haskey(partition.energy_data, BalanceData)
        partition.energy_data[BalanceData] = BalanceData(partition)
    end
    data = partition.energy_data[BalanceData]

    if partition.identifier != data.identifier
        partition.energy_data[BalanceData] = BalanceData(partition)
        data = partition.energy_data[BalanceData]
    end
    if update !== nothing && update.identifier != data.update_identifier
        update_balance_data!(data, partition, update)
    end

    active_districts = districts
    active_dist_pops = data.dist_pops
    if update !== nothing
        active_districts = collect(update.changed_districts)
        active_dist_pops = data.dist_pops_update
    end

    return satisfies_constraint(population_constraint, active_dist_pops,
                                active_districts)
end

"""
    satisfies_constraint(pop_constraint, dist_pops, districts)::Bool

Low-level check: return `true` if `dist_pops[di]` lies within the constraint's
`[min_pop, max_pop]` for every `di` in `districts`.
"""
function satisfies_constraint(
    pop_constraint::PopulationConstraint,
    dist_pops::Vector{Float64},
    districts::Union{Tuple{Vararg{T}}, Vector{T}}
)::Bool where T<:Int
    for di in districts
        pop = dist_pops[di]
        if !(pop_constraint.min_pop <= pop <= pop_constraint.max_pop)
            return false
        end
    end
    return true
end

"""
    update_energy_data!(data::BalanceData, partition, update)

Commit the accepted `update` into the population cache: copy the proposed populations
of the changed districts into `dist_pops` and sync the identifier.
"""
function update_energy_data!(
    data::BalanceData,
    partition::LinkCutPartition,
    update::Update{T}
) where T<:Int
    if update.identifier != data.update_identifier
        update_balance_data!(data, partition, update)
    end

    data.identifier = partition.identifier
    for di in update.changed_districts
        data.dist_pops[di] = data.dist_pops_update[di]
    end
end
