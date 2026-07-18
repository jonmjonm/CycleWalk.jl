"""
    IsoperimetricData <: AbstractEnergyData

Cache backing the isoperimetric (Polsby–Popper) energy. Stores each district's
`areas` and `perimeters`, their proposed-update scratch buffers (with `-1.0` marking
"not yet computed"), and the partition `identifier` last synced to.
"""
mutable struct IsoperimetricData <: AbstractEnergyData
    identifier::Int64
    areas::Vector{Float64}
    perimeters::Vector{Float64}
    areas_update::Vector{Float64}
    perimeters_update::Vector{Float64}
end

"""
    IsoperimetricData(partition)

Allocate an empty [`IsoperimetricData`](@ref) sized to `partition`'s districts, with
the identifier set one behind the partition's so areas/perimeters are computed on
first use.
"""
function IsoperimetricData(partition::LinkCutPartition)
    identifier = partition.identifier - 1
    areas = Vector{Float64}(undef, partition.num_dists)
    perimeters= Vector{Float64}(undef, partition.num_dists)
    areas_update  = Vector{Float64}(undef, partition.num_dists)
    perimeters_update = Vector{Float64}(undef, partition.num_dists)
    areas_update .= -1.0
    return IsoperimetricData(identifier, areas, perimeters, 
                             areas_update, perimeters_update)
end

"""
    set_areas_and_perimeters!(partition)

Recompute, from scratch, every district's area and perimeter into the
[`IsoperimetricData`](@ref) cache for the *current* partition. Area is the sum of
node areas; perimeter is the sum of node border lengths plus the perimeter
contribution of each cross-district edge (counted for both incident districts).
"""
function set_areas_and_perimeters!(partition::LinkCutPartition)

    if !haskey(partition.energy_data, IsoperimetricData)
        partition.energy_data[IsoperimetricData] = 
            IsoperimetricData(partition)
    end
    iso_data = partition.energy_data[IsoperimetricData]

    graph = partition.graph
    num_dists = partition.num_dists
    
    node_to_dist = partition.node_to_dist
    areas = iso_data.areas
    perimeters = iso_data.perimeters
    edge_dict = partition.cross_district_edges
    iso_data.identifier = partition.identifier

    border_len_col = graph.node_border_col
    area_col = graph.area_col
    edge_perimeter_col = graph.edge_perimeter_col

    areas .= 0
    perimeters .= 0

    for ni = 1:length(node_to_dist)
        areas[node_to_dist[ni]] += graph.node_attributes[ni][area_col]
        perimeters[node_to_dist[ni]] += 
            graph.node_attributes[ni][border_len_col]
    end
    for ((d1,d2), edges) in edge_dict
        for edge in edges
            e = Set([src(edge), dst(edge)])
            perimeters[d1] += graph.edge_attributes[e][edge_perimeter_col]
            perimeters[d2] += graph.edge_attributes[e][edge_perimeter_col]
        end
    end
end

"""
    set_areas_and_perimeters!(partition, di, update)

Compute district `di`'s area and perimeter for the *proposed* partition into the
update scratch buffers. If `di` is unchanged by `update`, its values are copied from
the synced cache; otherwise they are recomputed from the proposed assignment
(`node_to_dist_update`) and the update's recomputed cross-district edges.
"""
function set_areas_and_perimeters!(
    partition::LinkCutPartition,
    di::T,
    update::Update{T}
) where T <: Int
    if !haskey(partition.energy_data, IsoperimetricData)
        partition.energy_data[IsoperimetricData] = 
            IsoperimetricData(partition)
    end
    iso_data = partition.energy_data[IsoperimetricData]
    
    graph = partition.graph
    num_dists = partition.num_dists
    
    node_to_dist = partition.node_to_dist_update
    areas = iso_data.areas_update
    perimeters = iso_data.perimeters_update
    edge_dict = update.new_cross_d_edg
    delta_dists = update.changed_districts

    border_len_col = graph.node_border_col
    area_col = graph.area_col
    edge_perimeter_col = graph.edge_perimeter_col

    if !(di in update.changed_districts)
        @assert partition.identifier == iso_data.identifier
        areas[di] = iso_data.areas[di]
        perimeters[di] = iso_data.perimeters[di]
    else
        nodes = [ii for ii = 1:partition.graph.num_nodes 
                 if node_to_dist[ii]==di]
        for ni in nodes
            areas[di] += graph.node_attributes[ni][area_col]
            perimeters[di] += graph.node_attributes[ni][border_len_col]
        end
        for ((d1,d2), edges) in edge_dict
            if di != d1 && di != d2
                continue
            end
            for edge in edges
                e = Set([src(edge), dst(edge)])
                perimeters[di] += graph.edge_attributes[e][edge_perimeter_col]
            end
        end
    end
end

"""
    get_isoperimetric_scores(partition, districts=...; update=nothing)::Vector{Float64}

Return each district's isoperimetric ratio `perimeter² / area` (an inverse
Polsby–Popper compactness score; larger means less compact), using the cached
[`IsoperimetricData`](@ref). With `update === nothing` the current partition is
scored (refreshing the cache if stale); with an `update` the proposed values are
computed into the scratch buffers for the requested districts.
"""
function get_isoperimetric_scores(
    partition::LinkCutPartition,
    districts::Union{Tuple{Vararg{T}}, Vector{T}}
        =collect(1:partition.num_dists);
    update::Union{Update{T}, Nothing}=nothing
)::Vector{Float64} where T <: Int
    isos = Vector{Float64}(undef, length(districts))

    if !haskey(partition.energy_data, IsoperimetricData)
        partition.energy_data[IsoperimetricData] = 
            IsoperimetricData(partition)
    end
    iso_data = partition.energy_data[IsoperimetricData]

    if update === nothing
        if partition.identifier != iso_data.identifier
            set_areas_and_perimeters!(partition)
        end
        areas = iso_data.areas
        perimeters = iso_data.perimeters
    else
        areas = iso_data.areas_update
        perimeters = iso_data.perimeters_update
        # if partition.update_identifier != update.identifier
            areas .= 0
            perimeters .= 0
            for di in districts
                set_areas_and_perimeters!(partition, di, update)
            end
        # end
        # update.identifier = partition.update_identifier
    end

    for (ii, di) in enumerate(districts)
        isos[ii] = (perimeters[di]^2)/areas[di]
        # @show di, areas[di], iso_data.areas_update[di], iso_data.areas[di]
        # @show di, perimeters[di], iso_data.perimeters_update[di], iso_data.perimeters[di]
    end

    return isos
end

"""
    get_isoperimetric_scores(node_to_dist, graph::BaseGraph, num_dists)::Vector{Float64}

Partition-free isoperimetric (Polsby–Popper) scores `perimeter² / area` per district,
computed from just a node-to-district assignment and the base graph — no
`LinkCutPartition`, and therefore no random spanning tree. This is the same
quantity as [`set_areas_and_perimeters!`](@ref) plus the score formula, but the
cross-district edges are derived directly (an edge contributes to a boundary iff its
endpoints lie in different districts) instead of being read from a partition's cached
`cross_district_edges`.

Intended for scoring a stored districting after the fact (e.g. recomputing an atlas's
map data), where the spanning tree a `LinkCutPartition` draws is pure overhead.
`node_to_dist[ni]` is the district (`1:num_dists`) of node `ni`; the returned vector
is indexed by that same district numbering. Requires `graph`'s `area_col`,
`node_border_col` and `edge_perimeter_col` to be set (as isoperimetric scoring always
does). Matches the `LinkCutPartition` method's values exactly.
"""
function get_isoperimetric_scores(
    node_to_dist::Vector{Int},
    graph::BaseGraph,
    num_dists::Int,
)::Vector{Float64}
    areas = zeros(Float64, num_dists)
    perimeters = zeros(Float64, num_dists)
    area_col = graph.area_col
    border_len_col = graph.node_border_col
    edge_perimeter_col = graph.edge_perimeter_col

    for ni = 1:length(node_to_dist)
        di = node_to_dist[ni]
        areas[di] += graph.node_attributes[ni][area_col]
        perimeters[di] += graph.node_attributes[ni][border_len_col]
    end
    for edge in edges(graph.simple_graph)
        u, v = src(edge), dst(edge)
        du, dv = node_to_dist[u], node_to_dist[v]
        du == dv && continue
        perim = graph.edge_attributes[Set([u, v])][edge_perimeter_col]
        perimeters[du] += perim
        perimeters[dv] += perim
    end

    return Float64[perimeters[di]^2 / areas[di] for di in 1:num_dists]
end

"""
    get_isoperimetric_score(partition, districts=...; update=nothing, exponent=1.0,
                            omit_least_compact=0, pow_on_sum=nothing)::Float64

Summed isoperimetric energy: the sum over districts of `(perimeter²/area)^exponent`
(see [`get_isoperimetric_scores`](@ref)). Push this onto a `Measure` to penalize
non-compact plans.

!!! note
    The `omit_least_compact` and `pow_on_sum` keywords are currently accepted but not
    applied (the corresponding logic is disabled); only `exponent` affects the result.
"""
function get_isoperimetric_score(
    partition::LinkCutPartition,
    districts::Union{Tuple{Vararg{T}}, Vector{T}}
        =collect(1:partition.num_dists);
    update::Union{Update{T}, Nothing}=nothing,
    omit_least_compact::Int=0,
    pow_on_sum::Union{Nothing,Float64}=nothing,
    exponent::F=1.0
)::Float64 where {T <: Int, F <: Real}
    # if omit_least_compact > 0 || pow_on_sum != nothing
        isos = get_isoperimetric_scores(partition, districts, update=update)
    # else
    #    isos = get_isoperimetric_scores(partition, update=update)
    # end

    if exponent != 1
        isos .^= exponent
    end

    return sum(isos)
end

"""
    update_energy_data!(eData::IsoperimetricData, partition, update)

Commit the accepted `update` into the isoperimetric cache: copy the proposed areas
and perimeters for the changed districts into the synced fields and reset their
scratch slots. If any proposed value was never computed, the scratch buffer is fully
invalidated so values are recomputed on next query.
"""
function update_energy_data!(
    eData::IsoperimetricData,
    partition::LinkCutPartition,
    update::Update{T}
) where {T<:Int}
    for di in update.changed_districts
        if eData.areas_update[di] == -1
            eData.areas_update .= -1.0
            return
        end
    end
    eData.identifier = partition.identifier
    for di in update.changed_districts
        eData.areas[di] = eData.areas_update[di]
        eData.perimeters[di] = eData.perimeters_update[di]
        eData.areas_update[di] = -1.0
    end
end
