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
    lct_neighbors_huh(partition, u, v)

Compatibility wrapper for LinkCutTreesAugmented's `neighbors_huh`. If the 
    updated API is unavailable, falls back to checking whether the unique tree 
    path has length 2.
"""
function lct_neighbors_huh(
    partition::LinkCutPartition,
    u::Int,
    v::Int,
)::Bool
    if isdefined(LinkCutTreesAugmented, :neighbors_huh)
        neighbors_huh = getfield(LinkCutTreesAugmented, :neighbors_huh)
        return neighbors_huh(u, v, partition.lct)
    end

    u_node = partition.lct.nodes[u]
    v_node = partition.lct.nodes[v]
    evert!(u_node)
    expose!(v_node)
    return length(findPath(v_node)) == 2
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
        lct_neighbors_huh(partition, u, v) && continue
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
    sample_cross_edges_from_quotient_tree(partition, quotient_tree, rng)

For each district-edge of `quotient_tree`, select one concrete cross-district 
edge from the corresponding edge set, weighted by edge weight.
"""
function sample_cross_edges_from_quotient_tree(
    partition::LinkCutPartition,
    quotient_tree::Vector{Edge{Int}},
    rng::AbstractRNG,
)
    sampled = Vector{eltype(first(values(partition.cross_district_edges)))}(
        undef,
        length(quotient_tree),
    )
    for (ii, qe) in enumerate(quotient_tree)
        dist_pair = ordered_edge_key(src(qe), dst(qe))
        sampled[ii] = sample_weighted_edge(
            partition.cross_district_edges[dist_pair],
            rng,
        )
    end
    return sampled
end

"""
    link_sampled_cross_edges!(partition, sampled_cross_edges)

Temporarily link the sampled cross-district edges into the link-cut forest to
realize the district-tree augmentation used for the BUD cycle construction.
Returns the linked endpoint pairs so callers can undo the edits with
[`unlink_sampled_cross_edges!`](@ref).
"""
function link_sampled_cross_edges!(
    partition::LinkCutPartition,
    sampled_cross_edges::Vector{<:SimpleWeightedEdge},
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
    partition::LinkCutPartition,
    edge::Tuple{Int, Int},
)::Vector{<:Node}
    u_node = partition.lct.nodes[edge[1]]
    v_node = partition.lct.nodes[edge[2]]
    evert!(u_node)
    expose!(v_node)
    return findPath(v_node)
end

"""
    get_collapsed_cycle_weights_path(path, partition; field=partition.graph.pop_col)

Collapsed per-segment population weights for the single cycle induced by `path`
plus its closing proposal edge. The result length equals `length(path)` and sums to
the total population on the linked cycle component.
"""
function get_collapsed_cycle_weights_path(
    path::Vector{<:Node},
    partition::LinkCutPartition;
    field=partition.graph.pop_col,
)::Vector{Float64}
    n = length(path)
    n < 2 && return Float64[]

    collapsed = Vector{Float64}(undef, n)
    path_root = partition.lct.nodes[path[1].vertex]

    if field == partition.graph.pop_col
        evert!(path_root)
        cut_mass = Vector{Float64}(undef, n)
        for ii in 1:n
            cut_mass[ii] = subtree_pop(partition.lct.nodes[path[ii].vertex])
        end
        collapsed[1] = cut_mass[1] - cut_mass[2]
        for ii in 2:n-1
            collapsed[ii] = cut_mass[ii] - cut_mass[ii + 1]
        end
        collapsed[n] = cut_mass[n]
        return collapsed
    end

    cut_mass = topological_sort(path_root, partition; field=field)
    total = cut_mass[path[1].vertex]
    collapsed[1] = total - cut_mass[path[2].vertex]
    for ii in 2:n-1
        collapsed[ii] = cut_mass[path[ii].vertex] - cut_mass[path[ii + 1].vertex]
    end
    collapsed[n] = cut_mass[path[n].vertex]
    return collapsed
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
)::Bool
    m = length(cycle_weights)
    (m < 2 || k < 2) && return false

    found = falses(k - 1)
    remaining = k - 1
    total_pop = sum(cycle_weights)
    idx = (edge_ind == m) ? 1 : (edge_ind + 1)
    cur_pop = 0.0

    for _ in 1:(m - 1)
        cur_pop += cycle_weights[idx]
        other_pop = total_pop - cur_pop
        for t in 1:k-1
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
    candidates = Int[]
    for ei in 1:m
        if edge_supports_all_k_splits(cycle_weights, ei, k, min_pop, max_pop)
            push!(candidates, ei)
        end
    end
    return candidates
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
    k = length(quotient_tree) + 1
    sampled_cross_edges = sample_cross_edges_from_quotient_tree(
        partition,
        quotient_tree,
        rng,
    )

    # Track weighted district-tree count for future forward/backward proposal 
    # terms.
    log_weighted_qtree_count = log_nspanning(qg)

    sampled_cross_edge_keys = Set{Tuple{Int, Int}}(
        ordered_edge_key(e) for e in sampled_cross_edges
    )
    if edge in sampled_cross_edge_keys
        edge = sample_non_lct_edge(partition, rng;
                                   forbidden_edges=sampled_cross_edge_keys)
        d1 = partition.node_to_dist[edge[1]]
        d2 = partition.node_to_dist[edge[2]]
    end
    d1 == d2 && return internal_forest_walk!(partition, rng;
                                             diagnostics=diagnostics,
                                             edge=edge)
#####
    linked_cross_edges = link_sampled_cross_edges!(partition, sampled_cross_edges)
    cycle_weights = Float64[]
    candidate_cut_edges = Int[]
    try
        cycle_path = get_cycle_path_for_edge!(partition, edge)
        cycle_weights = get_collapsed_cycle_weights_path(cycle_path, partition)
        candidate_cut_edges = find_candidate_cycle_cut_edges(cycle_weights,
                                                             k,
                                                             constraints)
    finally
        unlink_sampled_cross_edges!(partition, linked_cross_edges)
    end

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
