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
    DENSE_CHOLESKY_MAX_M

Districts with cofactor dimension `m` below this use a dense Cholesky
(`get_log_spanning_trees`'s dense path); at or above it, a sparse one (CHOLMOD).
Chosen from direct wall-clock benchmarks (assembly + factorization together, BLAS
pinned to 1 thread as `run_parallel_tempering!` actually runs it) on real induced
subgraphs from ct (districts ~120-146 nodes: dense wins, ratio 1.0-1.3x),  nc
(districts 125-286 nodes: crosses over around 150-170), oh and hex50 (districts
~350-600 nodes: sparse wins 3-4x) — see the benchmark record in
`docs/pt_profiling_notes.md`. 160 sits just above ct's largest district and just
below nc's crossover, erring toward dense (the safer/simpler default) at the
boundary since real per-district sparsity is noisier than a clean function of `m`
alone (nc's own districts don't sort monotonically by `m` in the 125-170 range).
"""
const DENSE_CHOLESKY_MAX_M = 160

"""
    get_log_spanning_trees(node_to_dist, simple_graph, di)::Float64

Low-level kernel: the log number of (weighted) spanning trees of district `di`, i.e.
of the subgraph of `simple_graph` induced by the nodes assigned to `di` in
`node_to_dist`, via Kirchhoff's matrix-tree theorem.

The grounded Laplacian of the induced subgraph (its Laplacian with the last district
node's row/column deleted) is assembled directly from `simple_graph`'s sparse
adjacency (no `induced_subgraph`, no intermediate sparse-then-dense copy) and its
log-determinant taken by Cholesky — the grounded Laplacian of a connected graph is
symmetric positive definite. Below [`DENSE_CHOLESKY_MAX_M`](@ref) nodes this builds a
dense matrix (LAPACK `potrf!`); at or above it, a sparse one (CHOLMOD) — dense wins
at small district sizes (less assembly/factorization overhead), sparse wins by
several times at large ones (the induced subgraph of a real precinct or hex-lattice
map is near-planar, so its fill-in stays low). A disconnected district has no
spanning tree, so its (singular) grounded Laplacian yields `-Inf` on either path.
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

    weights = simple_graph.weights  # SparseMatrixCSC of edge weights
    rows = rowvals(weights)
    vals = nonzeros(weights)

    if m < DENSE_CHOLESKY_MAX_M
        L = zeros(Float64, m, m)
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
    else
        Is = Int[]
        Js = Int[]
        Vs = Float64[]
        @inbounds for a = 1:k
            u = nodes[a]
            pu = pos[u]
            pu > m && continue
            diag = 0.0
            for idx in nzrange(weights, u)
                v = rows[idx]
                pv = pos[v]
                pv == 0 && continue
                w = vals[idx]
                diag += w
                if pv <= m
                    push!(Is, pu); push!(Js, pv); push!(Vs, -w)
                end
            end
            push!(Is, pu); push!(Js, pu); push!(Vs, diag)
        end
        L = sparse(Is, Js, Vs, m, m)
        fact = cholesky(Symmetric(L); check=false)
        return issuccess(fact) ? logdet(fact) : -Inf
    end
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
    get_log_spanning_trees(node_to_dist, graph::BaseGraph, num_dists)::Vector{Float64}

Partition-free per-district log spanning-tree counts, computed straight from a
node-to-district assignment and the base graph — no `LinkCutPartition`, and therefore
no random spanning tree. Applies the low-level kernel
[`get_log_spanning_trees(node_to_dist, simple_graph, di)`](@ref) to each district over
`graph.simple_graph`. The result is indexed by `node_to_dist`'s own district numbering.

This is the uniform `(node_to_dist, graph::BaseGraph, num_dists)` signature shared with
the other partition-free writers, so a caller can discover the fast path by method
existence and fall back to the `LinkCutPartition` method otherwise. Intended for
scoring a stored districting after the fact (e.g. recomputing an atlas's map data).
"""
function get_log_spanning_trees(
    node_to_dist::Vector{Int},
    graph::BaseGraph,
    num_dists::Int,
)::Vector{Float64}
    sg = graph.simple_graph
    return Float64[get_log_spanning_trees(node_to_dist, sg, di) for di in 1:num_dists]
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
    get_log_spanning_forests(node_to_dist, graph::BaseGraph, num_dists)::Float64

Partition-free log spanning-FOREST count: the sum over districts of the partition-free
[`get_log_spanning_trees(node_to_dist, graph, num_dists)`](@ref). Shares the uniform
`(node_to_dist, graph::BaseGraph, num_dists)` signature so it is discoverable as a fast
path (with the `LinkCutPartition` method as the fallback). No partition or spanning
tree is built.
"""
function get_log_spanning_forests(
    node_to_dist::Vector{Int},
    graph::BaseGraph,
    num_dists::Int,
)::Float64
    return sum(get_log_spanning_trees(node_to_dist, graph, num_dists))
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