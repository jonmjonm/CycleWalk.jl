"""
    RegionalSplitData <: AbstractEnergyData

Shared cache of how a `region` attribute splits across districts, used by all of the
region constraints (pack / cap / budgeted). Maintains the two inverse maps
`regions_to_dists` (region name → set of districts it touches) and `dists_to_regions`
(district → set of regions it intersects), plus `_update` copies for a proposed move.
`identifier` / `update_identifier` track the partition state and proposed update the
buffers correspond to; `region` is the node-attribute column being split on.
"""
mutable struct RegionalSplitData <: AbstractEnergyData
    identifier::Int64
    update_identifier::Vector{Int64}
    regions_to_dists::Dict{String, Set{Int64}}
    dists_to_regions::Dict{Int64, Set{String}}
    regions_to_dists_update::Dict{String, Set{Int64}}
    dists_to_regions_update::Dict{Int64, Set{String}}
    region::String
end

"""
    RegionalSplitData(partition, region)

Build a [`RegionalSplitData`](@ref) for the given `region` attribute by scanning every
node once and recording, for the current assignment, which districts each region
touches and which regions each district intersects.
"""
function RegionalSplitData(partition::LinkCutPartition, region::String)
    identifier = partition.identifier
    update_identifier = Vector{Int64}(undef, 0)
    
    regions_to_dists = Dict{String, Set{Int64}}()
    dists_to_regions = Dict{Int64, Set{String}}()
    regions_to_dists_update = Dict{String, Set{Int64}}()
    dists_to_regions_update = Dict{Int64, Set{String}}()
    for ni in 1:length(partition.node_to_dist)
        name = partition.graph.node_attributes[ni][region]
        dist = partition.node_to_dist[ni]
        push!(get!(regions_to_dists, name, Set{Int64}()), dist)
        push!(get!(dists_to_regions, dist, Set{String}()), name)
    end
    return RegionalSplitData(identifier, update_identifier, 
                             regions_to_dists, dists_to_regions, 
                             regions_to_dists_update, 
                             dists_to_regions_update, region)
end

"""
    reset_update_buffer!(data::RegionalSplitData)

Forget which update the `_update` region maps were built for, so the next query with
an `update` rebuilds them.
"""
function reset_update_buffer!(data::RegionalSplitData)
    empty!(data.update_identifier)
    return
end

"""
    update_regional_split_data!(data, partition, update)

Recompute the `_update` region↔district maps for the proposed plan. Shallow-copies
the committed maps (sharing unaffected value sets), drops the entries for the changed
districts and the regions they touch, and rebuilds just those from the proposed
assignment `node_to_dist_update`. All nodes are scanned so unchanged districts that
also contain parts of an affected region are handled correctly.
"""
function update_regional_split_data!(
    data::RegionalSplitData,
    partition::LinkCutPartition,
    update::Update
)
    data.update_identifier = copy(update.identifier)
    region_col = data.region

    affected_regions = Set{String}()
    for di in update.changed_districts
        for region in data.dists_to_regions[di]
            push!(affected_regions, region)
        end
    end

    # Shallow-copy both dicts: unaffected entries share their value Vectors
    # with the committed dicts (safe because we replace, never mutate them).
    data.dists_to_regions_update = copy(data.dists_to_regions)
    data.regions_to_dists_update = copy(data.regions_to_dists)

    # Drop the entries to rebuild
    for di in update.changed_districts
        delete!(data.dists_to_regions_update, di)
    end
    for region in affected_regions
        delete!(data.regions_to_dists_update, region)
    end

    # Rebuild from node_to_dist_update (already the full proposed state).
    # Must iterate all nodes to correctly handle unchanged districts that
    # also contain parts of affected regions.
    for ni in 1:partition.graph.num_nodes
        region = partition.graph.node_attributes[ni][region_col]
        if region ∉ affected_regions
            continue
        end
        di = partition.node_to_dist_update[ni]
        push!(get!(data.dists_to_regions_update, di, Set{String}()), region)
        push!(get!(data.regions_to_dists_update, region, Set{Int64}()), di)
    end
end

"""
    update_energy_data!(data::RegionalSplitData, partition, update)

Commit the accepted `update` into the regional-split cache: ensure the proposed maps
are current, then make them the committed `regions_to_dists` / `dists_to_regions` and
sync the identifier.
"""
function update_energy_data!(
    data::RegionalSplitData,
    partition::LinkCutPartition,
    update::Update{T}
) where T<:Int
    if update.identifier != data.update_identifier
        update_regional_split_data!(data, partition, update)
    end

    data.identifier = partition.identifier
    data.regions_to_dists = copy(data.regions_to_dists_update)
    data.dists_to_regions = copy(data.dists_to_regions_update)
end