"""
    ordered_edge_key(u, v)

Canonicalized undirected edge key `(min(u,v), max(u,v))` used for stable 
    set/dict membership checks.
"""
@inline function ordered_edge_key(u::Int, v::Int)::Tuple{Int, Int}
    return (min(u, v), max(u, v))
end

"""
    ordered_edge_key(e)

Canonicalized key for a weighted graph edge.
"""
@inline function ordered_edge_key(e::SimpleWeightedEdge)::Tuple{Int, Int}
    return ordered_edge_key(src(e), dst(e))
end

"""
    has_edge(v, e)
"""
@inline function has_edge(v::Vector{E}, e::E)::Bool where E <: Edge
    return e ∈ v || reverse(e) ∈ v
end

"""
    has_edge(v, e)
"""
@inline function has_edge(
    v::Vector{E}, 
    e::Tuple{T, T}
)::Bool where {E <: Edge, T <: Int}
    return has_edge(v, Edge(e[1], e[2]))
end

"""
    sample_non_lct_edge(partition, rng; forbidden_edges=Set())

Draw a uniformly random graph edge whose endpoints are not adjacent in the 
current link-cut forest and whose canonical key is not in `forbidden_edges`.
"""
function sample_non_lct_edge(
    partition::LinkCutPartition,
    rng::AbstractRNG;
    forbidden_edges::Set{Tuple{Int, Int}}=Set{Tuple{Int, Int}}(),
)::Tuple{Int, Int}
    while true
        e = rand_dict_key(rng, partition.graph.edge_attributes)
        u, v = collect(e)
        uv_key = ordered_edge_key(u, v)
        uv_key in forbidden_edges && continue
        neighbors_huh(partition.lct[u], partition.lct[v]) && continue
        return uv_key
    end
end

"""
    build_district_quotient_graph(num_dists, cross_district_edges)

Construct the weighted district quotient graph and return
`(quotient_graph, dist_pair_to_qedge)` where each quotient edge weight is the 
sum of weights across `cross_district_edges[(d1,d2)]`.
"""
function build_district_quotient_graph(
    num_dists::Int,
    cross_district_edges::Dict{Tuple{Int64, Int64}, Set{SimpleWeightedEdge}},
)
    qg = SimpleWeightedGraph(num_dists)
    pair_to_qedge = Dict{Tuple{Int, Int}, Edge{Int}}()

    for (dist_pair, edgeset) in cross_district_edges
        total_weight = 0.0
        for e in edgeset
            total_weight += weight(e)
        end
        add_edge!(qg, dist_pair[1], dist_pair[2], total_weight)
        pair_to_qedge[dist_pair] = Edge(dist_pair[1], dist_pair[2])
    end
    return qg, pair_to_qedge
end

"""
    build_district_quotient_graph(partition)

Partition-specialized convenience wrapper around
[`build_district_quotient_graph`](@ref).
"""
function build_district_quotient_graph(
    partition::LinkCutPartition,
)
    return build_district_quotient_graph(
        partition.num_dists,
        partition.cross_district_edges,
    )
end

"""
    sample_weighted_edge(edgeset, rng)

Sample one concrete edge from `edgeset` proportional to its edge weight.
"""
function sample_weighted_edge(
    edgeset::Set{<:SimpleWeightedEdge},
    rng::AbstractRNG,
)
    total_weight = 0.0
    for e in edgeset
        total_weight += weight(e)
    end

    rand_samp = rand(rng) * total_weight
    running = 0.0
    for e in edgeset
        running += weight(e)
        if rand_samp <= running
            return e
        end
    end

    # Floating-point guard; loop above should always return.
    return first(edgeset)
end

"""
    link_sampled_cross_edges!(partition, sampled_cross_edges)

Temporarily link the sampled cross-district edges into the link-cut forest to
realize the district-tree augmentation used for the BUD cycle construction.
Returns the linked endpoint pairs so callers can undo the edits with
[`unlink_sampled_cross_edges!`](@ref).
"""
function link_quotient_path!(
    partition::LinkCutPartition,
    sampled_cross_edges::Dict
)::Vector{Tuple{Int, Int}}
    linked = Vector{Tuple{Int, Int}}(undef, 0)
    for e in sampled_cross_edges
        u = src(e)
        v = dst(e)
        u_node = partition.lct.nodes[u]
        v_node = partition.lct.nodes[v]
        evert!(u_node)
        link!(u_node, v_node)
        push!(linked, (u, v))
    end
    return linked
end

"""
    specify_quotient_path!(partition, specified_cross_edges, quotient_path, rng)

For each consecutive pair of districts `(d1, d2)` along `quotient_path` (a
vector of quotient-tree `Node`s whose `vertex` is the district), ensure a
cross-district edge is specified in `specified_cross_edges` (keyed by
`Edge(d1,d2)` in either orientation).

Pairs already specified are left untouched. Otherwise an edge is drawn with
[`sample_weighted_edge`](@ref) from `partition.cross_district_edges` and stored
under `Edge(d1, d2)` as a sorted `(u, v)` tuple. Nothing is linked; see
[`add_specified_edges!`](@ref).
"""
function specify_quotient_path!(
    partition::LinkCutPartition,
    specified_cross_edges::Dict{Edge{Int}, Tuple{Int, Int}},
    quotient_path::AbstractVector{<:Node},
    rng::AbstractRNG,
)
    for i in 1:length(quotient_path)-1
        d1 = quotient_path[i].vertex
        d2 = quotient_path[i+1].vertex
        key = Edge(d1, d2)
        if haskey(specified_cross_edges, key) ||
           haskey(specified_cross_edges, reverse(key))
            continue
        end
        dist_pair = haskey(partition.cross_district_edges, (d1, d2)) ?
                    (d1, d2) : (d2, d1)
        edgeset = partition.cross_district_edges[dist_pair]
        cross_edge = sample_weighted_edge(edgeset, rng)
        specified_cross_edges[key] = ordered_edge_key(cross_edge)
    end
    return specified_cross_edges
end

"""
    add_specified_edges!(partition, specified_cross_edges, quotient_path)

Link into `partition.lct` the specified cross-district edge for each
consecutive district pair along `quotient_path` (looked up in
`specified_cross_edges` in either orientation). Only edges on the path are
linked.

Returns the linked endpoint pairs, in order, so callers can undo the links with
[`unlink_sampled_cross_edges!`](@ref). If a link fails partway, the links made
so far are undone before the error is rethrown.
"""
function add_specified_edges!(
    partition::LinkCutPartition,
    specified_cross_edges::Dict{Edge{Int}, Tuple{Int, Int}},
    quotient_path::AbstractVector{<:Node},
)::Vector{Tuple{Int, Int}}
    linked = Tuple{Int, Int}[]
    for i in 1:length(quotient_path)-1
        d1 = quotient_path[i].vertex
        d2 = quotient_path[i+1].vertex
        key = haskey(specified_cross_edges, Edge(d1, d2)) ? Edge(d1, d2) :
                                                            Edge(d2, d1)
        haskey(specified_cross_edges, key) || throw(ArgumentError(
            "no specified cross edge for districts ($d1, $d2)"))
        u, v = specified_cross_edges[key]
        u_node = partition.lct.nodes[u]
        v_node = partition.lct.nodes[v]
        evert!(u_node)
        link!(u_node, v_node)
        push!(linked, (u, v))
    end
    return linked
end

"""
    unlink_sampled_cross_edges!(partition, linked_edges)

Undo temporary links created by [`link_sampled_cross_edges!`](@ref), restoring the
original district forest structure.
"""
function unlink_sampled_cross_edges!(
    partition::LinkCutPartition,
    linked_edges::Vector{Tuple{Int, Int}},
)
    for (u, v) in reverse(linked_edges)
        u_node = partition.lct.nodes[u]
        v_node = partition.lct.nodes[v]
        evert!(u_node)
        cut!(v_node)
    end
    return nothing
end

"""
    cut_cross_district_edges!(partition, quotient_path, specified_cross_edges)

Undo [`add_specified_edges!`](@ref): cut from `partition.lct` the specified
cross-district edge for each consecutive district pair along `quotient_path`,
splitting the linked trees back into one tree per district. The edges are removed
from the lct only; `cross_district_edges` and `node_to_dist` are untouched.

Cutting can leave a district rooted at a different vertex than before, so
`district_roots` and `roots_to_district` are updated for exactly the districts on
`quotient_path` (one root lookup per cut endpoint), keeping them valid without
restoring the old roots.
"""
function cut_cross_district_edges!(
    partition::LinkCutPartition,
    quotient_path::AbstractVector{<:Node},
    specified_cross_edges::Dict{Edge{Int}, Tuple{Int, Int}},
)
    cut_edges = Tuple{Int, Int}[]
    for i in 1:length(quotient_path)-1
        d1 = quotient_path[i].vertex
        d2 = quotient_path[i+1].vertex
        key = haskey(specified_cross_edges, Edge(d1, d2)) ? Edge(d1, d2) :
                                                            Edge(d2, d1)
        haskey(specified_cross_edges, key) || throw(ArgumentError(
            "no specified cross edge for districts ($d1, $d2)"))
        u, v = specified_cross_edges[key]
        evert!(partition.lct.nodes[u])
        cut!(partition.lct.nodes[v])
        push!(cut_edges, (u, v))
    end

    # Every cut endpoint lies in a district on the path, and every such district
    # has an endpoint of a cut edge, so this covers all of the affected roots.
    # Drop all stale root entries first so a deletion cannot clobber a new entry.
    ## TO DO: Clean this up -- I don't need the reallocation in the middle for loop
    new_roots = Dict{Int, Int}() # district => new root vertex
    for (u, v) in cut_edges, w in (u, v)
        new_roots[partition.node_to_dist[w]] =
            find_root!(partition.lct.nodes[w]).vertex
    end
    for d in keys(new_roots)
        delete!(partition.roots_to_district, partition.district_roots[d])
    end
    for (d, r) in new_roots
        partition.district_roots[d] = r
        partition.roots_to_district[r] = d
    end
    return nothing
end

"""
    get_cycle_path_for_edge!(partition, edge)

With the quotient-tree links already present, return the tree path between the
endpoints of `edge`; adding `edge` would close this path into the proposal cycle.
"""
function get_cycle_path_for_edge!(
    lct::LinkCutTree,
    edge::Tuple{Int, Int},
)::Vector{<:Node}
    u_node = lct.nodes[edge[1]]
    v_node = lct.nodes[edge[2]]
    evert!(u_node)
    expose!(v_node)
    return findPath(v_node)
end

"""
    get_cycle_path_for_edge!(partition, edge)

With the quotient-tree links already present, return the tree path between the
endpoints of `edge`; adding `edge` would close this path into the proposal cycle.
"""
function get_cycle_path_for_edge!(
    partition::LinkCutPartition,
    edge::Tuple{Int, Int},
)::Vector{<:Node}
    return get_cycle_path_for_edge!(partition.lct, edge)
end

"""
    get_collapsed_cycle_weights_path(path, cut_remainder)

Collapsed cycle weights for a single tree `path` (ordered root-first, i.e. from
the vertex the tree was rooted at by [`topological_sort`](@ref) to the other
endpoint of the closing edge). `cut_remainder[v]` is the weight of the subtree
below `v`, so entry `i` is `cut_remainder[path[i]] - cut_remainder[path[i+1]]`
(the last entry is just `cut_remainder[path[end]]`): the weight of `path[i]` plus
everything hanging off it away from the path. The entries sum to the total
weight of the tree.
"""
function get_collapsed_cycle_weights_path(
    path::AbstractVector{<:Node},
    cut_remainder::Dict{Int, Float64},
)::Vector{Float64}
    m = length(path)
    weights = Vector{Float64}(undef, m)
    for i in 1:m
        weights[i] = cut_remainder[path[i].vertex]
        if i < m
            weights[i] -= cut_remainder[path[i+1].vertex]
        end
    end
    return weights
end

"""
    get_collapsed_cycle_weights_path(path, partition)

Population-only fast path of [`get_collapsed_cycle_weights_path`](@ref). Roots
the tree at `path[1]` and reads each path vertex's subtree population from the
maintained sums of the augmented link-cut tree (`O(log n)` each) instead of
sorting the whole merged tree, so the cost depends on the length of the path
rather than on the number of nodes in the cycle's districts.
"""
function get_collapsed_cycle_weights_path(
    path::AbstractVector{<:Node},
    partition::LinkCutPartition,
)::Vector{Float64}
    lct = partition.lct
    m = length(path)
    evert!(lct.nodes[path[1].vertex])
    weights = Vector{Float64}(undef, m)
    next_pop = 0.0
    for i in m:-1:1
        cur_pop = subtree_pop(lct.nodes[path[i].vertex])
        weights[i] = cur_pop - next_pop
        next_pop = cur_pop
    end
    return weights
end

"""
    edge_supports_all_k_splits(cycle_weights, edge_ind, k, min_pop, max_pop)

Check the cheaper BUD criterion requested by the user: starting from cut edge
`edge_ind`, while traversing the cycle counter-clockwise, does there exist at least
one opposite-side cut producing each split size `t` vs `k-t` for all `t=1..k-1`
under district population bounds `[min_pop, max_pop]`.
"""
function edge_supports_all_k_splits(
    cycle_weights::Vector{Float64},
    edge_ind::Int,
    k::Int, # number of districts the cycle touches
    min_pop::Float64,
    max_pop::Float64,
    total_pop::Float64
)::Bool
    m = length(cycle_weights)
    @assert (m >= 2 && k >= 2)

    found = falses(k - 1)
    remaining = k - 1
    idx = (edge_ind == m) ? 1 : (edge_ind + 1)
    cur_pop = 0.0

    for _ in 1:(m - 1)
        cur_pop += cycle_weights[idx]

        k < ceil(cur_pop/max_pop)

        other_pop = total_pop - cur_pop
        st_idx = Int(ceil((cur_pop-1)/max_pop))
        end_idx = k - Int(ceil((total_pop - (cur_pop+1))/max_pop))
        for t in max(st_idx, 1):min(end_idx, k - 1)
            found[t] && continue
            lower_t = t * min_pop
            upper_t = t * max_pop
            lower_other = (k - t) * min_pop
            upper_other = (k - t) * max_pop
            if lower_t <= cur_pop <= upper_t &&
               lower_other <= other_pop <= upper_other
                found[t] = true
                remaining -= 1
                remaining == 0 && return true
            end
        end
        idx = (idx == m) ? 1 : (idx + 1)
    end
    return false
end

"""
    find_candidate_cycle_cut_edges(cycle_weights, k, constraints)

Return cycle edge indices that satisfy the cheaper `1..k-1` split-existence check
implemented by [`edge_supports_all_k_splits`](@ref).
"""
function find_candidate_cycle_cut_edges(
    cycle_weights::Vector{Float64},
    districts::Int,
    constraints::Constraints,
)::Vector{Int}
    @assert length(cycle_weights) >= 2 && districts >= 2

    min_pop = Float64(constraints.population_constraint.min_pop)
    max_pop = Float64(constraints.population_constraint.max_pop)
    tot_pop = sum(cycle_weights)
    candidates = Int[]
    for ei in 1:length(cycle_weights)
        if edge_supports_all_k_splits(cycle_weights, ei, districts, min_pop, 
                                      max_pop, tot_pop)
            push!(candidates, ei)
        end
    end
    return candidates
end

"""
    CutTable

DP table for splitting a path of `m` nodes into `k` districts with population in
`[min_pop, max_pop]`, built by [`build_cut_table`](@ref). `S[p+1]` is the
population of nodes `1..p`. Entry `W_t(p)` is the total weight of the ways to
split nodes `1..p` into `t` valid districts, where a way is weighted by the
product of the inverse edge weights of its interior cuts (including the cut at
`p` when `p < m`). Row `t` is only stored on the band `lo[t+1]:hi[t+1]` of
boundaries that can lie on a full segmentation; row `t` is column `t+1` of `W`.
"""
struct CutTable
    S::Vector{Float64}
    W::Matrix{Float64}
    lo::Vector{Int}
    hi::Vector{Int}
    k::Int
    m::Int
    min_pop::Float64
    max_pop::Float64
end

@inline function table_weight(tab::CutTable, t::Int, p::Int)::Float64
    (tab.lo[t+1] <= p <= tab.hi[t+1]) || return 0.0
    return tab.W[p - tab.lo[t+1] + 1, t+1]
end

"""
    build_cut_table(node_pops, inv_edge_weights, k, min_pop, max_pop)

Build the [`CutTable`](@ref) for a path of nodes with populations `node_pops`.
`inv_edge_weights[c]` is `1/weight` of the edge between nodes `c` and `c+1`.
Returns `nothing` if some row has no boundary that can lie on a full
segmentation. The total weight `Z` of all valid segmentations is
`table_weight(tab, k, m)` (zero when there is none).

Row `t` is only nonzero where the population before the boundary can come from
`t` districts and the population after it from `k-t` districts, which bounds the
band with four binary searches on the prefix sums.
"""
function build_cut_table(
    node_pops::AbstractVector{<:Real},
    inv_edge_weights::AbstractVector{<:Real},
    k::Int,
    pop_bounds::Tuple{Float64,Float64}
)::Union{Nothing, CutTable}
    m = length(node_pops)
    S = zeros(Float64, m + 1)
    for i in 1:m
        S[i+1] = S[i] + node_pops[i]
    end
    total = S[m+1]

    min_pop, max_pop = pop_bounds

    # This may be overly clever: It may be better just to store a m x k+1 array
    lo = Vector{Int}(undef, k + 1)
    hi = Vector{Int}(undef, k + 1)
    lo[1] = hi[1] = 0
    lo[k+1] = hi[k+1] = m
    for t in 1:(k-1)
        lo_t = searchsortedfirst(S, max(t * min_pop, total - (k - t) * max_pop)) - 1
        hi_t = searchsortedlast(S, min(t * max_pop, total - (k - t) * min_pop)) - 1
        lo[t+1] = max(lo_t, 1)
        hi[t+1] = min(hi_t, m - 1)
        lo[t+1] > hi[t+1] && return nothing
    end

    W = zeros(Float64, maximum(hi .- lo) + 1, k + 1)
    W[1, 1] = 1.0
    for t in 1:k
        for q in lo[t+1]:hi[t+1]
            s = S[q+1]
            # previous boundaries p with segment p+1..q in [min_pop, max_pop]
            plo = max(searchsortedfirst(S, s - max_pop) - 1, lo[t])
            phi = min(searchsortedlast(S, s - min_pop) - 1, hi[t], q - 1)
            acc = 0.0
            for p in plo:phi
                acc += W[p - lo[t] + 1, t]
            end
            W[q - lo[t+1] + 1, t+1] = q < m ? acc * inv_edge_weights[q] : acc
        end
    end
    return CutTable(S, W, lo, hi, k, m, min_pop, max_pop)
end

"""
    select_next_cut(tab, q, t, rng; forced=nothing)

One backward step of the DP walk: given that nodes up to boundary `q` are split
into `t` districts, choose the previous boundary `p` (the last of the `t`
districts is nodes `p+1..q`) with probability proportional to `W_{t-1}(p)` among
the boundaries giving that district a valid population. With `forced` set to a
boundary, nothing is sampled and that boundary is scored instead.

Returns `(p, log_prob)`, or `nothing` if there is no valid choice (or `forced`
is not one).
"""
function select_next_cut(
    tab::CutTable,
    q::Int,
    t::Int,
    rng::AbstractRNG;
    forced::Union{Nothing, Int}=nothing,
)::Union{Nothing, Tuple{Int, Float64}}
    s = tab.S[q+1]
    plo = max(searchsortedfirst(tab.S, s - tab.max_pop) - 1, tab.lo[t])
    phi = min(searchsortedlast(tab.S, s - tab.min_pop) - 1, tab.hi[t], q - 1)
    total = 0.0
    for p in plo:phi
        total += table_weight(tab, t - 1, p)
    end
    total > 0 || return nothing

    if forced !== nothing
        (plo <= forced <= phi) || @assert false "Can only select a valid cut"
        w = table_weight(tab, t - 1, forced)
        w > 0 || return nothing
        return forced, log(w / total)
    end

    r = rand(rng) * total
    running = 0.0
    chosen = 0
    for p in plo:phi
        w = table_weight(tab, t - 1, p)
        w > 0 || continue
        chosen = p  # also the floating-point fallback: the last positive p
        running += w
        r <= running && break
    end
    return chosen, log(table_weight(tab, t - 1, chosen) / total)
end

"""
    rotated_cut_table(node_pops, inv_w, e, k, min_pop, max_pop)

[`build_cut_table`](@ref) for the cycle cut open at edge `e`: the path starts at
cycle node `mod1(e+1, m)`, and path edge `c` is cycle edge `mod1(e+c, m)`.
"""
function rotated_cut_table(
    node_pops::AbstractVector{<:Real},
    inv_w::AbstractVector{<:Real},
    e::Int,
    k::Int,
    pop_bounds::Tuple{Float64,Float64}
)
    m = length(node_pops)
    rot_pops = [node_pops[mod1(e + r, m)] for r in 1:m]
    rot_inv_w = [inv_w[mod1(e + c, m)] for c in 1:(m-1)]
    return build_cut_table(rot_pops, rot_inv_w, k, pop_bounds)
end

"""
    CycleCutData(node_pops, edge_weights, candidate_edges, districts, pop_bounds)

Everything about one cycle that the cut selection needs. Cycle node `i` has
population `node_pops[i]`; edge `i` joins nodes `i` and `i+1` (edge `m` closes
the cycle) with weight `edge_weights[i]`.

For each plausible first cut `e` in `candidate_edges` it builds the DP table of
the path obtained by cutting at `e`, and keeps `Z[e]`, the total weight (product
of `1/w` over the cuts) of the ways to place the other `districts-1` cuts. Edges
with `Z[e] == 0` cannot start any valid collection and are dropped from
`feasible`. `D = Σ_feasible inv_w[e] * Z[e]` normalizes the first-edge draw.
Everything here depends only on the cycle, so the same `CycleCutData` gives the
forward and the reverse probability.
"""
struct CycleCutData
    node_pops::Vector{Float64}
    inv_w::Vector{Float64}
    cummulant::Vector{Float64}
    districts::Int
    path_len::Int
    min_pop::Float64
    max_pop::Float64
    Z::Vector{Float64}
    feasible::Vector{Int}
    D::Float64
end

function CycleCutData(
    node_pops::AbstractVector{<:Real},
    edge_weights::AbstractVector{<:Real},
    candidate_edges::AbstractVector{Int},
    districts::Int,
    pop_bounds::Tuple{Float64,Float64},
)
    m = length(node_pops)
    @assert m >= 2 && districts >= 2
    pops = Float64.(node_pops)
    inv_w = [1.0 / w for w in edge_weights]

    cummulant = zeros(Float64, m + 1)
    for i in 1:m
        cummulant[i+1] = cummulant[i] + pops[i]
    end

    Z = zeros(Float64, m)
    feasible = Int[]
    for e in candidate_edges
        dptable = rotated_cut_table(pops, inv_w, e, districts, pop_bounds)
        dptable === nothing && continue
        Z[e] = table_weight(dptable, districts, m)
        Z[e] > 0 && push!(feasible, e)
    end
    D = sum(inv_w[e] * Z[e] for e in feasible; init=0.0)

    return CycleCutData(pops, inv_w, cummulant, districts, m, pop_bounds[1],
                        pop_bounds[2], Z, feasible, D)
end

"""
    CycleCutData(node_pops, districts, partition, constraints, path, edge)

Build the [`CycleCutData`](@ref) of the cycle formed by the tree `path` (cycle
node `i` is `path[i]`) and the closing non-tree `edge`. The candidate first
edges come from [`find_candidate_cycle_cut_edges`](@ref).
"""
function CycleCutData(
    node_pops::AbstractVector{<:Real},
    districts::Int,
    partition::LinkCutPartition,
    constraints::Constraints,
    path::AbstractVector{<:Node},
    edge::Tuple{Int, Int},
)
    m = length(node_pops)
    wmx = weights(partition.graph.simple_graph)
    edge_weights = [i < m ? wmx[path[i].vertex, path[i+1].vertex] :
                            wmx[edge[1], edge[2]] for i in 1:m]
    pop_bounds = (Float64(constraints.population_constraint.min_pop),
                  Float64(constraints.population_constraint.max_pop))
    candidate_edges = find_candidate_cycle_cut_edges(node_pops, districts,
                                                     constraints)
    return CycleCutData(node_pops, edge_weights, candidate_edges, districts,
                        pop_bounds)
end

"""
    valid_cycle_boundaries(cut_data, boundaries)

Whether `boundaries` are `districts` distinct cycle edges whose arcs all have
population in `[min_pop, max_pop]`.
"""
function valid_cycle_boundaries(
    cut_data::CycleCutData,
    boundaries::AbstractVector{Int},
)::Bool
    k = cut_data.districts
    length(boundaries) == k || return false
    Bs = sort(boundaries)
    (allunique(Bs) && 1 <= Bs[1] && Bs[end] <= cut_data.path_len) || return false
    S = cut_data.cummulant
    for j in 1:k
        a = Bs[j]
        pop = j < k ? S[Bs[j+1]+1] - S[a+1] : S[end] - S[a+1] + S[Bs[1]+1]
        cut_data.min_pop <= pop <= cut_data.max_pop || return false
    end
    return true
end

"""
    log_prob_cut_route(cut_data, first_edge, marked_edges)

Log-probability that [`select_cycle_cuts`](@ref) draws `first_edge` as the removed
(unmarked) edge and `marked_edges` as the other cuts (`-Inf` if that is not a
valid collection). The first edge is drawn with probability
`inv_w[e] * Z[e] / D` and the marked edges given `e` with probability
`Π inv_w[b] / Z[e]`, so the route has probability `Π_{b ∈ boundaries} inv_w[b] / D`:
`Z[e]` cancels, and `D` is the same for every route on the cycle.
"""
function log_prob_cut_route(
    cut_data::CycleCutData,
    first_edge::Int,
    marked_edges::AbstractVector{Int},
)::Float64
    boundaries = vcat(first_edge, marked_edges)
    valid_cycle_boundaries(cut_data, boundaries) || return -Inf
    cut_data.Z[first_edge] > 0 || return -Inf
    return sum(log(cut_data.inv_w[b]) for b in boundaries) - log(cut_data.D)
end

"""
    select_cycle_cuts(cut_data, rng)

Select the cuts that split the cycle into districts. A first edge `e` is removed,
drawn from `cut_data.feasible` with probability `inv_w[e] * Z[e] / D`; the cycle
is then a path, and the other `districts-1` (marked) cuts are drawn by walking
back through its DP table with [`select_next_cut`](@ref), so a full route has
probability `Π inv_w[b] / D` over its cuts `b`.

Returns `((e, marked_edges), log_prob)`: the removed edge, the sorted marked
cycle edges, and the log-probability of that route (the same value
[`log_prob_cut_route`](@ref) returns for it). Returns `nothing` if no valid
collection exists.
"""
function select_cycle_cuts(
    cut_data::CycleCutData,
    rng::AbstractRNG,
)::Union{Nothing, Tuple{Tuple{Int, Vector{Int}}, Float64}}
    isempty(cut_data.feasible) && return nothing

    # select the edge to remove
    r = rand(rng) * cut_data.D
    cum_sum = 0.0
    e = cut_data.feasible[end]
    for cand in cut_data.feasible
        cum_sum += cut_data.inv_w[cand] * cut_data.Z[cand]
        if r <= cum_sum
            e = cand
            break
        end
    end

    # walk back through dp table of path opened at e to pick the marked cuts
    dptab = rotated_cut_table(cut_data.node_pops, cut_data.inv_w, e,
                              cut_data.districts,
                              (cut_data.min_pop, cut_data.max_pop))
    @assert dptab !== nothing
    marked_edges = Int[]
    cur_ind = cut_data.path_len
    log_prob = log(cut_data.inv_w[e] * cut_data.Z[e]) - log(cut_data.D)
    for t in cut_data.districts:-1:1
        step = select_next_cut(dptab, cur_ind, t, rng)
        @assert step !== nothing
        step_ind, step_log_prob = step
        log_prob += step_log_prob
        step_ind > 0 && push!(marked_edges,
                              mod1(e + step_ind, cut_data.path_len))
        cur_ind = step_ind
    end
    sort!(marked_edges)
    return (e, marked_edges), log_prob
end

"""
    build_quotient_lct(edges; safeflag=true)

Build a `LinkCutTree` on vertices `1:length(edges)+1` whose edges are `edges`
(anything with `src`/`dst`, e.g. a Wilson quotient tree).

With `safeflag=true` (default) every add is validated: endpoints must lie in
`1:length(edges)+1`, no edge may be a self-loop, and no edge may join two
vertices that are already connected. Since a tree on `n` vertices has `n-1`
edges, passing all checks means `edges` is a spanning tree. Violations throw an
`ArgumentError`. Set `safeflag=false` to skip the checks when the input is
known to be a tree.
"""
function build_quotient_lct(edges::AbstractVector; safeflag::Bool=true)
    n = length(edges) + 1
    lct = LinkCutTree{Int}(n)
    for e in edges
        u, v = src(e), dst(e)
        if safeflag
            (1 <= u <= n && 1 <= v <= n) || throw(ArgumentError(
                "edge ($u, $v) has an endpoint outside 1:$n"))
            u == v && throw(ArgumentError("self-loop at vertex $u"))
            find_root!(lct.nodes[u]) === find_root!(lct.nodes[v]) &&
                throw(ArgumentError(
                    "edge ($u, $v) closes a cycle; edges are not a tree"))
        end
        evert!(lct.nodes[u])
        link!(lct.nodes[u], lct.nodes[v])
    end
    return lct
end

"""
    get_valid_cross_edge!(specified_cross_edges, partition, rng, edge)

Specify the cross-district edge for the district pair of the sampled non-tree
`edge`, ensuring the specified edge differs from `edge` itself. If it coincides,
`edge` is resampled (never choosing an already specified edge) and the process
repeats for the new edge's district pair.

`specified_cross_edges` (keyed by `Edge(d1,d2)`, looked up in either
orientation) is updated in place; existing specifications are reused. Returns
`(edge, (d1, d2))` for the final `edge`. If the resampled edge is internal
(`d1 == d2`) it is returned immediately, unspecified, so the caller can run an
internal forest walk.
"""
function get_valid_cross_edge!(
    specified_cross_edges::Dict{Edge{T}, Tuple{T, T}},
    partition::LinkCutPartition,
    quotient_tree::Vector{Edge},
    edge::Tuple{T, T},
    dist_pair::Tuple{T, T},
    rng::AbstractRNG
) where T <: Int
    d1, d2 = dist_pair
    while true
        d1 == d2 && return edge, (d1, d2)
        if Edge(d1, d2) ∉ quotient_tree && Edge(d2, d1) ∉ quotient_tree
            return edge, (d1, d2)
        end
        key = Edge(d1, d2)
        if haskey(specified_cross_edges, reverse(key))
            key = reverse(key)
        end
        if !haskey(specified_cross_edges, key)
            dist_pair = haskey(partition.cross_district_edges, (d1, d2)) ?
                               (d1, d2) : (d2, d1)
            cross_edges = partition.cross_district_edges[dist_pair]
            cross_edge = sample_weighted_edge(cross_edges, rng)
            specified_cross_edges[key] = ordered_edge_key(cross_edge)
        end
        specified_cross_edges[key] != edge && return edge, (d1, d2)

        edge = sample_non_lct_edge(partition, rng;
                       forbidden_edges=Set(values(specified_cross_edges)))
        d1 = partition.node_to_dist[edge[1]]
        d2 = partition.node_to_dist[edge[2]]
    end
end
    

"""
    cycle_edge_vertices(path, i)

Vertex pair of cycle edge `i`: edge `i < length(path)` joins `path[i]` and
`path[i+1]`, and edge `length(path)` is the closing non-tree edge joining the two
ends of the path.
"""
@inline function cycle_edge_vertices(path::AbstractVector{<:Node}, i::Int)
    m = length(path)
    return i < m ? [path[i].vertex, path[i+1].vertex] :
                   [path[m].vertex, path[1].vertex]
end

"""
    get_links_and_cuts(path, old_boundaries, new_boundaries)

Edge edits that take the current forest to the proposed one. Each boundary
argument is `(removed_edge, marked_edges)` in cycle edge indices. Old boundaries
that are not new boundaries become links, and new boundaries that are not old
boundaries become cuts; boundaries in both are omitted from both lists.
"""
function get_links_and_cuts(
    path::AbstractVector{<:Node},
    old_boundaries::Tuple{Int, Vector{Int}},
    new_boundaries::Tuple{Int, Vector{Int}},
)
    old_set = Set(vcat(old_boundaries[1], old_boundaries[2]))
    new_set = Set(vcat(new_boundaries[1], new_boundaries[2]))
    links = [cycle_edge_vertices(path, i) for i in sort!(collect(setdiff(old_set, new_set)))]
    cuts = [cycle_edge_vertices(path, i) for i in sort!(collect(setdiff(new_set, old_set)))]
    return links, cuts
end

"""
    assign_cycle_district_map!(partition, path, path_labels)

Write the proposed labels into `node_to_dist_update` for the nodes whose label
changes, and return those nodes. `path_labels[i]` is the new label of `path[i]`,
and `node_to_dist_update` must already equal `node_to_dist`. Must be called on the
merged cycle tree, before any tentative cut.

With the tree rooted at `path[1]` and exposed at `path[m]`, the preferred path is
exactly the cycle path, so every other node lies in a subtree hanging off one path
vertex (a virtual child of it) and shares that vertex's piece. Only the subtrees of
path vertices whose label changes are traversed, so the cost is the number of
relabeled nodes plus `m` rather than the size of the changed districts.
"""
function assign_cycle_district_map!(
    partition::LinkCutPartition,
    path::AbstractVector{<:Node},
    path_labels::Vector{Int},
)
    nodes = partition.lct.nodes
    m = length(path)
    evert!(nodes[path[1].vertex])
    expose!(nodes[path[m].vertex])
    # function barrier: the node type is only concrete at runtime
    return _assign_cycle_district_map!(nodes, path, path_labels,
                                       partition.node_to_dist,
                                       partition.node_to_dist_update)
end

function _assign_cycle_district_map!(
    nodes::Vector{Union{Nothing, N}},
    path::AbstractVector{<:Node},
    path_labels::Vector{Int},
    node_to_dist::Vector{Int},
    node_to_dist_update::Vector{Int},
) where {N<:Node}
    m = length(path)
    moved = Int[]
    stack = N[]
    for i in 1:m
        v = path[i].vertex
        new_label = path_labels[i]
        node_to_dist[v] == new_label && continue
        node_to_dist_update[v] = new_label
        push!(moved, v)
        c = first_path_child(nodes[v])
        while c !== nothing
            push!(stack, c)
            c = next_path_sibling(c)
        end
        while !isempty(stack)
            node = pop!(stack)
            node_to_dist_update[node.vertex] = new_label
            push!(moved, node.vertex)
            for ii = 1:2
                node.children[ii] !== nothing && push!(stack, node.children[ii])
            end
            c = first_path_child(node)
            while c !== nothing
                push!(stack, c)
                c = next_path_sibling(c)
            end
        end
    end
    return moved
end

"""
    update_log_proposal_ratio!(log_p, log_qtree_count_old, partition, path,
                               old_marked_edges, new_boundaries, links, cuts,
                               constraints)

Score a BUD proposal. Must be called while the quotient-path cross edges are
linked, so the lct holds the merged cycle tree. The proposed forest is reached by
cutting the new boundaries that lie on the path and linking the closing edge if it
is not a new boundary. Component `j` along the path takes the old label of piece
`j` (the component joined across the closing edge takes the label of piece 0).

Returns `(log_p, update)` with `log_p` increased by
`log_qtree_count_old - log_qtree_count_new`, or `(-Inf, nothing)` if the
constraints fail. The lct, `district_roots` and `roots_to_district` are restored
to the merged-tree state, and `node_to_dist_update` holds the proposed assignment
on success.
"""
function update_log_proposal_ratio!(
    log_p::Float64,
    log_qtree_count_old::Float64,
    partition::LinkCutPartition,
    path::AbstractVector{<:Node},
    old_marked_edges::Vector{Int},
    new_boundaries::Vector{Int},
    links::Vector{Vector{Int}},
    cuts::Vector{Vector{Int}},
    constraints::Constraints,
)
    m = length(path)
    nodes = partition.lct.nodes
    node_to_dist = partition.node_to_dist

    # old labels of the pieces along the path
    old_sorted = sort(old_marked_edges)
    labels = [node_to_dist[path[1].vertex];
              [node_to_dist[path[i+1].vertex] for i in old_sorted]]
    k = length(labels)
    changed = Tuple(sort(labels))

    new_path_cuts = sort([b for b in new_boundaries if b < m])
    closing_is_boundary = m in new_boundaries
    piece_starts = [1; [p + 1 for p in new_path_cuts]]
    piece_labels = [labels[min(j, k)] for j in 1:length(piece_starts)]
    # without a closing boundary the last piece is joined to piece 0 across `m`
    !closing_is_boundary && (piece_labels[end] = labels[1])

    old_roots = [partition.district_roots[d] for d in changed]

    # new label of every path vertex: piece j spans piece_starts[j] up to the next
    # start (the merged tree is still intact here)
    path_labels = Vector{Int}(undef, m)
    for (j, start) in enumerate(piece_starts)
        stop = j < length(piece_starts) ? piece_starts[j+1] - 1 : m
        path_labels[start:stop] .= piece_labels[j]
    end
    partition.node_to_dist_update .= partition.node_to_dist
    moved = assign_cycle_district_map!(partition, path, path_labels)

    # tentative forest
    for p in new_path_cuts
        evert!(nodes[path[p].vertex])
        cut!(nodes[path[p+1].vertex])
    end
    if !closing_is_boundary
        evert!(nodes[path[m].vertex])
        link!(nodes[path[m].vertex], nodes[path[1].vertex])
    end

    for r in old_roots
        delete!(partition.roots_to_district, r)
    end
    for (s, d) in zip(piece_starts, piece_labels)
        r = find_root!(nodes[path[s].vertex]).vertex
        partition.district_roots[d] = r
        partition.roots_to_district[r] = d
    end

    new_cross_d_edg = find_cross_district_edges_moved(partition, changed, moved)
    update = Update(changed, links, cuts, new_cross_d_edg, false)

    # `update.identifier` does not fix the labeling, so staged buffers from an earlier
    # proposal must not be reused
    for eData in values(partition.energy_data)
        reset_update_buffer!(eData)
    end
    constraints_ok = satisfies_constraints(partition, constraints,
                                           collect(changed); update=update)

    log_qtree_count_new = -Inf
    if constraints_ok
        log_qtree_count_new = get_log_district_trees(partition; update=update)
    else
        partition.node_to_dist_update .= partition.node_to_dist
    end

    # revert to the merged tree; cut the closing edge before relinking so no
    # link joins two nodes of the same tree
    if !closing_is_boundary
        evert!(nodes[path[m].vertex])
        cut!(nodes[path[1].vertex])
    end
    for p in new_path_cuts
        u = nodes[path[p].vertex]
        evert!(u)
        link!(u, nodes[path[p+1].vertex])
    end
    for d in changed
        delete!(partition.roots_to_district, partition.district_roots[d])
    end
    for (d, r) in zip(changed, old_roots)
        partition.district_roots[d] = r
        partition.roots_to_district[r] = d
    end

    constraints_ok || return -Inf, nothing
    return log_p + log_qtree_count_old - log_qtree_count_new, update
end

"""
    bud_walk!(partition, constraints, rng; diagnostics=nothing)

Scaffold for a BUD-style proposal:
1. Sample a random edge not in the link-cut forest.
2. If internal, run an internal forest walk and return.
3. If cross-district, build a weighted district quotient graph, sample a 
   quotient spanning tree (Wilson), and sample one concrete cross edge for each 
   quotient edge.
4. If the initially sampled edge lies in that newly sampled full-tree 
   augmentation, resample until an edge outside both trees is found.
5. If that resampled edge is internal, run an internal forest walk and return.
6. Otherwise, the chosen edge would create a cycle in the sampled full tree.

This scaffold currently stops at step 6 and returns `(0, nothing)`.
"""
function bud_walk!(
    partition::LinkCutPartition,
    constraints::Constraints,
    rng::AbstractRNG;
    diagnostics::Union{Nothing, ProposalDiagnostics}=nothing,
)
    edge = sample_non_lct_edge(partition, rng)
    d1 = partition.node_to_dist[edge[1]]
    d2 = partition.node_to_dist[edge[2]]
    d1 == d2 && return internal_forest_walk!(partition, rng; 
                                             diagnostics=diagnostics, edge=edge)

    qg = get_district_quotient_graph(partition)
    quotient_tree = wilson_rst(qg, rng, weights(qg))
    
    specified_cross_edges = Dict{Edge{Int64}, Tuple{Int64, Int64}}()
    edge, (d1, d2) = get_valid_cross_edge!(specified_cross_edges, partition, 
                                           quotient_tree, edge, (d1, d2), rng)
    d1 == d2 && return internal_forest_walk!(partition, rng;
                                             diagnostics=diagnostics, edge=edge)

    quotient_lct = build_quotient_lct(quotient_tree; safeflag=false)
    quotient_path = get_cycle_path_for_edge!(quotient_lct, (d1, d2))
    dists_in_cycle = length(quotient_path)
    specify_quotient_path!(partition, specified_cross_edges, quotient_path, rng)
    linked_cross_edges = add_specified_edges!(partition, specified_cross_edges,
                                              quotient_path)
    
    path = get_cycle_path_for_edge!(partition, edge)
    cycle_weights = get_collapsed_cycle_weights_path(path, partition)

    cut_data = CycleCutData(cycle_weights, dists_in_cycle, partition, 
                            constraints, path, edge) # cached computations
    new_edge_cuts = select_cycle_cuts(cut_data, rng)
    if new_edge_cuts === nothing
        cut_cross_district_edges!(partition, quotient_path, 
                                  specified_cross_edges)
        return 0, nothing
    end
    (cut_e, new_marked_edges), log_prob_forward_mark = new_edge_cuts

    # Reverse route: the removed edge is the sampled non-tree `edge` (cycle edge
    # m) and the marked edges are the path edges that currently separate
    # districts.
    m = length(path)
    old_marked_edges = [i for i in 1:(m-1) if partition.node_to_dist[path[i].vertex] !=
                                              partition.node_to_dist[path[i+1].vertex]]
    log_prob_reverse_mark = log_prob_cut_route(cut_data, m, old_marked_edges)
    @assert log_prob_reverse_mark > -Inf
    

    #### proposal ratio is what: I still need ratio of quotient district graphs
    log_p = log_prob_reverse_mark - log_prob_forward_mark
    # The lifted weight w(F)w(M) is the weight of the spanning tree F ∪ M, and
    # the move swaps cut_e out of that tree and `edge` in.
    log_p += log(cut_data.inv_w[cut_e]) - log(cut_data.inv_w[m])
    
    links, cuts = get_links_and_cuts(path, (m, old_marked_edges), 
                                     (cut_e, new_marked_edges),)
    @assert length(links) == length(cuts) 
    if length(links) == 0
        cut_cross_district_edges!(partition, quotient_path, 
                                  specified_cross_edges)
        return 0, nothing
    end
    
    log_qtree_count_old = get_log_district_trees(partition)
    log_p, update = update_log_proposal_ratio!(
        log_p, log_qtree_count_old, partition, path, old_marked_edges,
        vcat(cut_e, new_marked_edges), links, cuts, constraints)

    cut_cross_district_edges!(partition, quotient_path, specified_cross_edges)
    update === nothing && return 0, nothing
    return exp(log_p), update
end

"""
    build_bud_walk(constraints)

Return a BUD-walk proposal closure with captured `constraints`.
"""
function build_bud_walk(
    constraints::Constraints,
)
    f(p::LinkCutPartition,
      r::AbstractRNG;
      diagnostics::Union{Nothing, ProposalDiagnostics}=nothing) =
        bud_walk!(p, constraints, r; diagnostics=diagnostics)
    return f
end
