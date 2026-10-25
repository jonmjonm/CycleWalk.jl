"""
    LogDistrictTreesData <: AbstractEnergyData

Cache backing the district-quotient-graph energy. `weights` is the dense,
symmetric `num_dists × num_dists` matrix whose `(a, b)` entry is the total weight of
the edges crossing between districts `a` and `b`, and `log_trees` is the log
weighted spanning-tree count of the graph it defines. `weights_update` and
`log_trees_update` hold the proposed values staged by the last `update` query, and
`identifier` is the partition identifier the current values were last synced to.
"""
mutable struct LogDistrictTreesData <: AbstractEnergyData
    identifier::Int64
    weights::Matrix{Float64}
    log_trees::Float64
    weights_update::Matrix{Float64}
    log_trees_update::Float64
end

"""
    LogDistrictTreesData(partition)

Allocate an empty [`LogDistrictTreesData`](@ref) for `partition`. The stored
identifier is one behind the partition's so the values are built on first use.
"""
function LogDistrictTreesData(partition::LinkCutPartition)
    identifier = partition.identifier - 1
    n = partition.num_dists
    return LogDistrictTreesData(identifier, zeros(n, n), 0.0, zeros(n, n), 0.0)
end

"""
    log_nspanning_weights(W)::Float64

Log weighted spanning-tree count of the graph with symmetric weight matrix `W`
(Kirchhoff, first row/column of the Laplacian deleted); `-Inf` if the graph is
disconnected.
"""
function log_nspanning_weights(W::Matrix{Float64})::Float64
    n = size(W, 1)
    n <= 1 && return 0.0
    L = Matrix{Float64}(undef, n - 1, n - 1)
    @inbounds for j = 2:n, i = 2:n
        L[i-1, j-1] = -W[i, j]
    end
    @inbounds for i = 2:n
        L[i-1, i-1] = sum(@view W[i, :]) - W[i, i]
    end
    fact = cholesky!(Symmetric(L); check=false)
    return issuccess(fact) ? logdet(fact) : -Inf
end

"""
    get_log_district_trees_data(partition)::LogDistrictTreesData

Fetch the partition's [`LogDistrictTreesData`](@ref) (creating it on first use) and
rebuild its current values from `partition.cross_district_edges` if the partition
has moved since it was last synced.
"""
function get_log_district_trees_data(
    partition::LinkCutPartition
)::LogDistrictTreesData
    data = get!(partition.energy_data, LogDistrictTreesData) do
        LogDistrictTreesData(partition)
    end
    if partition.identifier != data.identifier
        data.identifier = partition.identifier
        fill!(data.weights, 0.0)
        for ((a, b), edgeset) in partition.cross_district_edges
            pair_sum = 0.0
            for e in edgeset
                pair_sum += weight(e)
            end
            data.weights[a, b] = pair_sum
            data.weights[b, a] = pair_sum
        end
        data.log_trees = log_nspanning_weights(data.weights)
    end
    return data
end

"""
    stage_log_district_trees!(data, update)

Stage the proposed weights and log tree count for `update`: copy the current
weights, clear the rows and columns of the changed districts, and refill them from
`update.new_cross_d_edg`.
"""
function stage_log_district_trees!(
    data::LogDistrictTreesData,
    update::Update{T}
) where {T<:Int}
    W = data.weights_update
    copyto!(W, data.weights)
    for d in update.changed_districts
        W[d, :] .= 0.0
        W[:, d] .= 0.0
    end
    for ((a, b), edgeset) in update.new_cross_d_edg
        pair_sum = 0.0
        for e in edgeset
            pair_sum += weight(e)
        end
        W[a, b] = pair_sum
        W[b, a] = pair_sum
    end
    data.log_trees_update = log_nspanning_weights(W)
    return data.log_trees_update
end

"""
    get_district_quotient_graph(partition)::SimpleWeightedGraph

The weighted district quotient graph, built from the cached weight matrix.
"""
function get_district_quotient_graph(partition::LinkCutPartition)
    return SimpleWeightedGraph(get_log_district_trees_data(partition).weights)
end

"""
    get_log_district_trees(partition, districts=...; update=nothing)::Float64

Log number of (weighted) spanning trees of the district adjacency graph — one vertex
per district, each adjacent pair joined by an edge weighted by the total weight of
edges crossing between them — using the cached [`LogDistrictTreesData`](@ref). With
an `update`, the proposed value is computed from the current weights by replacing the
rows and columns of the changed districts with the update's recomputed boundary
weights, and staged for [`update_energy_data!`](@ref). Measures how richly connected
the district-level graph is.
"""
function get_log_district_trees(
    partition::LinkCutPartition,
    districts::Union{Tuple{Vararg{T}}, Vector{T}}
        =collect(1:partition.num_dists);
    update::Union{Update{T}, Nothing}=nothing
)::Float64 where T <: Int
    data = get_log_district_trees_data(partition)
    update === nothing && return data.log_trees
    return stage_log_district_trees!(data, update)
end

"""
    update_energy_data!(eData::LogDistrictTreesData, partition, update)

Commit the accepted `update`: restage it (so the committed values always belong to
`update`, whatever proposal was scored last) and swap the staged weights in.
"""
function update_energy_data!(
    eData::LogDistrictTreesData,
    partition::LinkCutPartition,
    update::Update{T}
) where {T<:Int}
    # unsynced before this accept (never built, or invalidated): rebuild on next use
    eData.identifier == partition.identifier - 1 || return
    stage_log_district_trees!(eData, update)
    eData.weights, eData.weights_update = eData.weights_update, eData.weights
    eData.log_trees = eData.log_trees_update
    eData.identifier = partition.identifier
end
