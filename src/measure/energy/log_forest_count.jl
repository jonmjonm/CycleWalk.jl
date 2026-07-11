"""
    LogForestEnergyData <: AbstractEnergyData

Cache backing the spanning-forest energy. Holds the current per-district log
spanning-tree counts (`log_trees`), a scratch buffer for the proposed values
(`log_trees_update`, with `-1.0` marking "not yet computed"), and the partition
`identifier` the cache was last synced to.
"""
mutable struct LogForestEnergyData <: AbstractEnergyData
    identifier::Int64
    log_trees::Vector{Float64}
    log_trees_update::Vector{Float64}
end

"""
    LogForestEnergyData(partition)

Allocate an empty [`LogForestEnergyData`](@ref) for `partition`, sized to its number
of districts. The stored identifier is set one behind the partition's so the counts
are recomputed on first use.
"""
function LogForestEnergyData(partition::LinkCutPartition)
    identifier = partition.identifier - 1
    log_trees = Vector{Float64}(undef, partition.num_dists)
    log_trees_update = Vector{Float64}(undef, partition.num_dists)
    log_trees_update .= -1.0
    return LogForestEnergyData(identifier, log_trees, log_trees_update)
end

"""
    get_log_spanning_trees(node_to_dist, simple_graph, di)::Float64

Low-level kernel: the log number of (weighted) spanning trees of district `di`, i.e.
of the subgraph of `simple_graph` induced by the nodes assigned to `di` in
`node_to_dist`, via Kirchhoff's matrix-tree theorem.

The grounded Laplacian of the induced subgraph (its Laplacian with the last district
node's row/column deleted) is assembled directly into a dense matrix and its
log-determinant taken by Cholesky — the grounded Laplacian of a connected graph is
symmetric positive definite. This avoids materializing an `induced_subgraph`, a
sparse Laplacian, and a dense copy of it on every call, which is the dominant
allocator during sampling. A disconnected district has no spanning tree, so its
(singular) grounded Laplacian yields `-Inf`.
"""
function get_log_spanning_trees(
    node_to_dist::Vector{Int64},
    simple_graph::SimpleWeightedGraph,
    di::T
)::Float64 where T <: Int64
    n = length(node_to_dist)
    # `nodes` lists the district's vertices; `pos` maps a global vertex index to its
    # 1-based position within the district (0 if not in it). The last district vertex
    # is grounded (dropped), giving an (m = k-1)-dimensional cofactor.
    nodes = Vector{Int}(undef, n)
    pos = zeros(Int, n)
    k = 0
    @inbounds for ii = 1:n
        if node_to_dist[ii] == di
            k += 1
            nodes[k] = ii
            pos[ii] = k
        end
    end
    m = k - 1
    m <= 0 && return 0.0  # single-node (or empty) district has exactly one spanning tree

    L = zeros(Float64, m, m)
    weights = simple_graph.weights  # SparseMatrixCSC of edge weights
    rows = rowvals(weights)
    vals = nonzeros(weights)
    @inbounds for a = 1:k
        u = nodes[a]
        pu = pos[u]
        pu > m && continue  # u is the grounded vertex; its row/column is dropped
        for idx in nzrange(weights, u)
            v = rows[idx]
            pv = pos[v]
            pv == 0 && continue  # neighbor not in this district
            w = vals[idx]
            L[pu, pu] += w                 # weighted degree (diagonal)
            pv <= m && (L[pu, pv] -= w)    # off-diagonal (skip grounded vertex)
        end
    end

    fact = cholesky!(Symmetric(L); check=false)
    return issuccess(fact) ? logdet(fact) : -Inf
end

"""
    get_log_spanning_trees(partition, districts=...; update=nothing)::Vector{Float64}

Return the log spanning-tree count of each district in `districts`, using the cached
[`LogForestEnergyData`](@ref). With `update === nothing` the current partition is
scored (recomputing all districts only when the cache is stale); with an `update`
the proposed assignment (`node_to_dist_update`) is scored for the changed districts.
"""
function get_log_spanning_trees(
    partition::LinkCutPartition,
    districts::Union{Tuple{Vararg{T}}, Vector{T}}
        =collect(1:partition.num_dists);
    update::Union{Update{T}, Nothing}=nothing
)::Vector{Float64} where T <: Int
    log_spanning_trees = Vector{Float64}(undef, length(districts))

    if haskey(partition.energy_data, LogForestEnergyData)
        log_tree_data = partition.energy_data[LogForestEnergyData]
    else
        log_tree_data = LogForestEnergyData(partition)
        partition.energy_data[LogForestEnergyData] = log_tree_data
    end

    simple_graph = partition.graph.simple_graph

    if update === nothing
        node_to_dist = partition.node_to_dist
        log_trees = log_tree_data.log_trees
        if partition.identifier != log_tree_data.identifier
            log_tree_data.identifier = partition.identifier
            for ii = 1:partition.num_dists
                log_trees[ii] = get_log_spanning_trees(node_to_dist, 
                                                       simple_graph, ii)
            end
        end
    else
        node_to_dist = partition.node_to_dist_update
        log_trees = log_tree_data.log_trees_update
        for di in districts
            log_trees[di] = get_log_spanning_trees(node_to_dist, simple_graph,
                                                   di)
        end
    end

    for (ii, di) in enumerate(districts)
        log_spanning_trees[ii] = log_trees[di]
    end
    return log_spanning_trees
end

"""
    get_log_spanning_forests(partition, districts=...; update=nothing)::Float64

Sum of [`get_log_spanning_trees`](@ref) over `districts` — the log number of spanning
forests of the plan. This is the spanning-forest energy: push it onto a `Measure`
with weight γ (γ=0 gives the spanning-forest measure, γ=1 the partition measure).
"""
function get_log_spanning_forests(
    partition::LinkCutPartition,
    districts::Union{Tuple{Vararg{T}}, Vector{T}}
        = collect(1:partition.num_dists);
    update::Union{Update{T}, Nothing}=nothing
)::Float64 where T <: Int
    return sum(get_log_spanning_trees(partition, districts, update=update))
end

"""
    update_energy_data!(eData::LogForestEnergyData, partition, update)

Commit the accepted `update` into the spanning-forest cache: copy the proposed log
counts for the changed districts into `log_trees` and reset their scratch slots. If
any proposed value was never computed (still `-1.0`), the scratch buffer is fully
invalidated so the counts are recomputed on next query.
"""
function update_energy_data!(
    eData::LogForestEnergyData,
    partition::LinkCutPartition,
    update::Update{T}
) where {T<:Int}
    # @show "in tree based updater"
    for di in update.changed_districts
        if eData.log_trees_update[di] == -1
            eData.log_trees_update .= -1.0
            return
        end
    end
    eData.identifier = partition.identifier
    for di in update.changed_districts
        eData.log_trees[di] = eData.log_trees_update[di]
        eData.log_trees_update[di] = -1.0
    end
end