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
        specified_cross_edges[key] = ordered_edge_key(
                                         sample_weighted_edge(edgeset, rng))
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
    edge_supports_all_k_splits(cycle_weights, edge_ind, k, min_pop, max_pop)

Check the cheaper BUD criterion requested by the user: starting from cut edge
`edge_ind`, while traversing the cycle counter-clockwise, does there exist at least
one opposite-side cut producing each split size `t` vs `k-t` for all `t=1..k-1`
under district population bounds `[min_pop, max_pop]`.
"""
function edge_supports_all_k_splits(
    cycle_weights::Vector{Float64},
    edge_ind::Int,
    k::Int,
    min_pop::Float64,
    max_pop::Float64,
    total_pop::Float64
)::Bool
    m = length(cycle_weights)
    (m < 2 || k < 2) && return false

    found = falses(k - 1)
    remaining = k - 1
    idx = (edge_ind == m) ? 1 : (edge_ind + 1)
    cur_pop = 0.0

    for _ in 1:(m - 1)
        cur_pop += cycle_weights[idx]

        k < ceil(cur_pop/max_pop)

        other_pop = total_pop - cur_pop
        st_idx = int(ceil((cur_pop-1)/max_pop))
        end_idx = k - int(ceil((total_pop - (cur_pop+1))/max_pop))
        for t in st_idx:end_idx
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
    k::Int,
    constraints::Constraints,
)::Vector{Int}
    m = length(cycle_weights)
    (m < 2 || k < 2) && return Int[]

    min_pop = Float64(constraints.population_constraint.min_pop)
    max_pop = Float64(constraints.population_constraint.max_pop)
    tot_pop = sum(cycle_weights)
    candidates = Int[]
    for ei in 1:m
        if edge_supports_all_k_splits(cycle_weights, ei, k, min_pop, max_pop, 
                                      tot_pop)
            push!(candidates, ei)
        end
    end
    return candidates
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
    if d1 == d2
        return internal_forest_walk!(partition, rng; diagnostics=diagnostics,
                                     edge=edge)
    end

    qg, _ = build_district_quotient_graph(partition)
    quotient_tree = wilson_rst(qg, rng, weights(qg))
    
    specified_cross_edges = Dict{Edge{Int}, Tuple{Int, Int}}()

    while true
        specified_edge = sample_weighted_edge(
                                  partition.cross_district_edges[(d1, d2)], rng)
        specified_edge = ordered_edge_key(specified_edge)
        specified_cross_edges[Edge(d1, d2)] = ordered_edge_key(specified_edge)
        if specified_edge != ordered_edge_key(edge)
            break
        end
        edge = sample_non_lct_edge(partition, rng;
                                   forbidden_edges=keys(specified_cross_edges))
        d1 = partition.node_to_dist[edge[1]]
        d2 = partition.node_to_dist[edge[2]]
        d1 == d2 && return internal_forest_walk!(partition, rng;
                                                 diagnostics=diagnostics,
                                                 edge=edge)
    end
    log_weighted_qtree_count = log_nspanning(qg)

#####
    ## TODO: store the quotient_tree as an lct, then find the impacted districts
    ##       and only specify/link those edges (don't do the whole thing)
    quotient_lct = build_quotient_lct(quotient_tree; safeflag=false)
    quotient_path = get_cycle_path_for_edge!(quotient_lct, (d1, d2))
    k = length(quotient_path)
    specify_quotient_path!(partition, specified_cross_edges, quotient_path, rng)
    linked_cross_edges = add_specified_edges!(partition, specified_cross_edges,
                                              quotient_path)
    
    u1, u2 = partition.lct[src(edge)], partition.lct[dst(edge)]
    evert!(u1)
    expose!(u2)
    path = findPath(u2)
    cut_remainder = topological_sort(u1, partition)
    cycle_weights = get_collapsed_cycle_weights_path(path, cut_remainder)
    
    candidate_cut_edges = find_candidate_cycle_cut_edges(cycle_weights, k,
                                                         constraints)
    # finally
    #     unlink_sampled_cross_edges!(partition, linked_cross_edges)
    # end

    # This is the per-edge probability under the sampling model.
    num_non_tree_edges = ne(partition.graph.simple_graph) -
                         (nv(partition.graph.simple_graph) - 1)
    if num_non_tree_edges <= 0
        return 0, nothing
    end
    p_edge = 1.0 / num_non_tree_edges

    # Placeholder values to keep scaffold outputs available during development.
    _ = candidate_cut_edges
    _ = cycle_weights
    _ = p_edge
    _ = log_weighted_qtree_count

    # At this point, adding `edge` to the sampled full tree forms a cycle.
    return 0, nothing
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
