"""
    get_rand_adjacent_dists(partition, rng)

Pick a uniformly random pair of adjacent districts (a key of
`partition.cross_district_edges`). Returns `(distPair, prob)` where `prob` is the
selection probability `1/number_of_adjacent_pairs`, needed for the Hastings ratio.
"""
function get_rand_adjacent_dists(
    partition::LinkCutPartition,
    rng::AbstractRNG
)
    distPair = rand_dict_key(rng, partition.cross_district_edges)
    return distPair, 1/length(partition.cross_district_edges)
end

"""
    get_rand_edges(distPair, partition, rng)

Pick two distinct uniformly random boundary edges between the adjacent districts
`distPair` (the two edges that will be linked to form the cycle). Returns
`(edge_pair, prob)`, or `(nothing, nothing)` if the pair shares at most one boundary
edge. `prob` is `1/(m*(m-1))` for `m` boundary edges, for the Hastings ratio.
"""
function get_rand_edges(
    distPair::Tuple{Int, Int},
    partition::LinkCutPartition,
    rng::AbstractRNG
)
    boundary_edges = length(partition.cross_district_edges[distPair])
    boundary_edges <= 1 && return nothing, nothing
    edge_pair = rand_set_element_pair(rng, 
                                      partition.cross_district_edges[distPair])
    return edge_pair, 1/(boundary_edges*(boundary_edges-1))
end

"""
    get_paths!(partition, edge_pair)

Given the two boundary edges in `edge_pair`, mutate the link-cut tree (via `evert!`
/ `expose!`) so that the cycle they create can be read off as two arms, and return
`(uPath, vPath)`: the tree paths from each linking edge toward the boundary between
the two districts. The endpoints are reoriented so that `uPath` and `vPath` lie in
the two different districts.
"""
function get_paths!(partition::LinkCutPartition, edge_pair::Tuple)
    u1 = partition.lct.nodes[src(edge_pair[1])]
    v1 = partition.lct.nodes[dst(edge_pair[1])]
    evert!(u1)
    evert!(v1)

    u2 = partition.lct.nodes[src(edge_pair[2])]
    v2 = partition.lct.nodes[dst(edge_pair[2])]
    r2 = find_root!(u2)
    if u1 != r2 
        # u1 and u2 are in different districts, redefine v2 as u2 so they are
        u2, v2 = v2, u2
    end
    
    expose!(u2)
    expose!(v2)
    uPath = findPath(u2)
    vPath = findPath(v2)
    return uPath, vPath
end

"""
    get_collapsed_cycle_weights(uPath, vPath, partition; field=partition.graph.pop_col)

Return the per-segment weights (population by default) along the cycle formed by the
two arms `uPath` and `vPath`. Each entry is the weight that would move across the
boundary if the cycle were cut between two adjacent path vertices. For the hot
population case this reads the link-cut tree's maintained subtree sums directly
(`O(log n)` per vertex); for other `field`s it falls back to the whole-tree
[`topological_sort`](@ref).
"""
function get_collapsed_cycle_weights(
    uPath::Vector{<:Node},
    vPath::Vector{<:Node},
    partition::LinkCutPartition;
    field=partition.graph.pop_col
)::Vector{Float64}
    # Population path (the hot case): query the maintained PopAug subtree sums
    # directly -- O(log n) per path vertex, no per-call Dict. Other fields (e.g.
    # `field=nothing` node counts, used by diagnostics) fall back to the
    # whole-tree topological_sort.
    if field == partition.graph.pop_col
        return get_collapsed_cycle_weights_subpop(uPath, vPath, partition)
    end
    return get_collapsed_cycle_weights_topo(uPath, vPath, partition; field=field)
end

"""
    get_collapsed_cycle_weights_subpop(uPath, vPath, partition)

Population-only fast path of [`get_collapsed_cycle_weights`](@ref). The cut-population
of a path vertex is its subtree population when the tree is rooted at the arm's base
(`u1` / `v1`), queried in `O(log n)` via the augmented link-cut tree; each segment's
collapsed weight is the difference of adjacent cut-populations.
"""
function get_collapsed_cycle_weights_subpop(
    uPath::Vector{<:Node},
    vPath::Vector{<:Node},
    partition::LinkCutPartition,
)::Vector{Float64}
    lct = partition.lct
    u1 = lct.nodes[uPath[1].vertex]
    v1 = lct.nodes[vPath[1].vertex]
    path_length = length(uPath) + length(vPath)
    collapsed_cycle_weight = Vector{Float64}(undef, path_length)

    evert!(u1)                       # root the cycle's u-arm at u1
    prev = 0.0
    for ii = 1:length(uPath)
        # uPath reversed: position ii corresponds to uPath[end-ii+1]
        cur = subtree_pop(lct.nodes[uPath[end-ii+1].vertex])
        collapsed_cycle_weight[ii] = ii == 1 ? cur : cur - prev
        prev = cur
    end

    evert!(v1)                       # root the v-arm at v1
    prev = 0.0
    for ii = 0:(length(vPath)-1)
        cur = subtree_pop(lct.nodes[vPath[end-ii].vertex])
        collapsed_cycle_weight[end-ii] = ii == 0 ? cur : cur - prev
        prev = cur
    end
    return collapsed_cycle_weight
end

"""
    get_collapsed_cycle_weights_topo(uPath, vPath, partition; field=partition.graph.pop_col)

General fallback of [`get_collapsed_cycle_weights`](@ref) for arbitrary `field`s
(e.g. node counts). Computes each arm's cut weights with a whole-tree
[`topological_sort`](@ref) and differences adjacent path vertices to get per-segment
collapsed weights.
"""
function get_collapsed_cycle_weights_topo(
    uPath::Vector{<:Node},
    vPath::Vector{<:Node},
    partition::LinkCutPartition;
    field=partition.graph.pop_col
)::Vector{Float64}
    uPath_rev = reverse(uPath)
    u1 = partition.lct.nodes[uPath[1].vertex]
    v1 = partition.lct.nodes[vPath[1].vertex]
    u_cut_pop_dict = topological_sort(u1, partition; field=field)
    v_cut_pop_dict = topological_sort(v1, partition; field=field)
    path_length = length(uPath) + length(vPath)
    collapsed_cycle_weight = Vector{Float64}(undef, path_length)
    for ii = 1:length(uPath_rev)
        vertex = uPath_rev[ii].vertex
        collapsed_cycle_weight[ii] = u_cut_pop_dict[vertex]
        if ii > 1
            next_vertex = uPath_rev[ii-1].vertex
            collapsed_cycle_weight[ii] -= u_cut_pop_dict[next_vertex]
        end
    end
    for ii = 0:(length(vPath)-1)
        vertex = vPath[end-ii].vertex
        collapsed_cycle_weight[end-ii] = v_cut_pop_dict[vertex]
        if ii > 0
            next_vertex = vPath[end-ii+1].vertex
            collapsed_cycle_weight[end-ii] -= v_cut_pop_dict[next_vertex]
        end
    end
    return collapsed_cycle_weight
end

# assumes other cut is at end and 1
# `pre` holds prefix sums: pre[i+1] == sum(cycle_weights[1:i]), pre[1] == 0, so
# sum(cycle_weights[a:b]) == pre[b+1] - pre[a].
"""
    find_first_valid_cut(pre, initial_cut_index, min_pop, max_pop, totpop_uv)

Starting from `initial_cut_index` and walking *down*, return the smallest cut index
for which the single cut `(1, index)` keeps both induced segments within
`[min_pop, max_pop]`. `pre` are the prefix sums of the cycle weights and `totpop_uv`
the cycle's total. Used to bound the inner loop of [`find_cuttable_edge_pairs`](@ref).
"""
function find_first_valid_cut(
    pre::Vector{Float64},
    initial_cut_index::Int,
    min_pop::Float64,
    max_pop::Float64,
    totpop_uv::Float64
)
    first_valid_cut = initial_cut_index
    while first_valid_cut > 1
        first_valid_cut -= 1
        @assert first_valid_cut > 0
        pop1 = pre[first_valid_cut + 1]          # sum(1:first_valid_cut)
        pop2 = totpop_uv - pop1
        if !(min_pop <= pop1 <= max_pop && min_pop <= pop2 <= max_pop)
            first_valid_cut += 1
            break
        end
    end
    return first_valid_cut
end

"""
    find_cuttable_edge_pairs(cycle_weights, initial_cut_index, partition, constraints)

Return the set of `(cut1, cut2)` index pairs whose two induced segments are each
population-balanced within the constraint's `[min_pop, max_pop]`, excluding the
current cut `(1, initial_cut_index)`.

PRECONDITION: `initial_cut_index` must itself be a balanced single cut, i.e. both
`sum(cycle_weights[1:initial_cut_index])` and its complement lie in
`[min_pop, max_pop]`. In production this always holds because `initial_cut_index`
is the current district boundary (`length(uPath)`), which satisfies the population
constraint. `find_first_valid_cut` walks *down* from `initial_cut_index` while the
single cut stays valid; if the initial index is itself unbalanced the walk stops
immediately and the search silently under-returns valid pairs. Callers feeding a
synthetic index (e.g. tests) must respect this.
"""
function find_cuttable_edge_pairs(
    cycle_weights::Vector{U},
    initial_cut_index::Int,
    partition::LinkCutPartition,
    constraints::Constraints
) where U <: Real
    path_length = length(cycle_weights)
    # Prefix sums so each segment population is an O(1) difference, instead of an
    # O(path) `sum(@view ...)` recomputed for every (cut1,cut2) pair (which was
    # O(path^3) and boxed a Float64 each time). For integer populations stored as
    # Float64 the prefix difference is bit-identical to the view sum.
    pre = Vector{Float64}(undef, path_length + 1)
    pre[1] = 0.0
    @inbounds for i in 1:path_length
        pre[i + 1] = pre[i] + cycle_weights[i]
    end
    totpop_uv = pre[path_length + 1]
    # concrete Float64 locals: the constraint's min_pop/max_pop fields are
    # abstract `Real`, which would make every `min_pop <= pop1` a dynamic call
    # that boxes pop1. For integer bounds the Float64 value is identical.
    min_pop::Float64 = constraints.population_constraint.min_pop
    max_pop::Float64 = constraints.population_constraint.max_pop
    possible_pairs = Set{Tuple{Int,Int}}()

    ### find valid cut with smallest index when paired with u1,v1
    first_valid_cut = find_first_valid_cut(pre, initial_cut_index,
                                           min_pop, max_pop, totpop_uv)

    for cut1 = 1:path_length
        found_cut = false
        for cut2 = max(cut1, first_valid_cut):path_length-1
            pop1 = pre[cut2 + 1] - pre[cut1]      # sum(cut1:cut2)
            pop2 = totpop_uv - pop1
            if min_pop <= pop1 <= max_pop && min_pop <= pop2 <= max_pop
                # @show "adding", cut1, cut2, pop1, pop2
                push!(possible_pairs, (cut1, cut2))
                if !found_cut
                    found_cut = true
                    first_valid_cut = cut2
                end
            elseif found_cut
                break
            end
        end
    end
    # don't consider the previous cut
    delete!(possible_pairs, (1, initial_cut_index)) 
    return possible_pairs
end

"""
    get_node_indices_from_paths(edge_ind, uPath, vPath)

Map a cycle edge index `edge_ind` back to the pair of link-cut tree `Node`s that
edge connects. The cycle is indexed as: the closing `u`–`v` boundary edge, then up
the `u` arm, across the far link, then down the `v` arm. Returns the `(node1, node2)`
endpoints of the indexed edge.
"""
function get_node_indices_from_paths(
    edge_ind::Int,
    uPath::Vector{<:Node},
    vPath::Vector{<:Node}
)
    if edge_ind==1 
        return (uPath[end], vPath[end])
    elseif edge_ind <= length(uPath)
        return (uPath[end-edge_ind+1], uPath[end-edge_ind+2])
    elseif edge_ind == length(uPath)+1
        return (uPath[1], vPath[1])
    elseif edge_ind > length(uPath)+1
        ind = edge_ind - length(uPath) - 1
        return (vPath[ind], vPath[ind+1])
    end
end

"""
    get_cuts_and_links(init_cut_edge_pair, final_cut_edge_pair)

Given the two edges originally added to form the cycle (`init_cut_edge_pair`) and the
two edges chosen to cut it (`final_cut_edge_pair`), return `(cuts, links)`: the edges
to actually remove and add to realize the move. If a chosen cut coincides with an
originally-added edge, that edge cancels out (a 1-tree move), leaving a single link.
"""
function get_cuts_and_links(
    init_cut_edge_pair::Tuple,
    final_cut_edge_pair::Tuple
)
    ie1 = [src(init_cut_edge_pair[1]), dst(init_cut_edge_pair[1])]
    ie2 = [src(init_cut_edge_pair[2]), dst(init_cut_edge_pair[2])]
    fe1 = [final_cut_edge_pair[1][1].vertex, final_cut_edge_pair[1][2].vertex]
    fe2 = [final_cut_edge_pair[2][1].vertex, final_cut_edge_pair[2][2].vertex]
    links = [ie1, ie2]
    cuts = [fe1, fe2]
    if ie1 in cuts
        filter!(e->e≠ie1, cuts)
        links = [ie2]
    elseif ie2 in cuts
        filter!(e->e≠ie2, cuts)
        links = [ie1]
    end
    return cuts, links
end

"""
    revert_tentative_proposal!(partition, distPair, links, cuts, old_root1, old_root2)

Undo the in-place link-cut edits made while scoring a proposal: re-cut the edges that
were linked, re-link the edges that were cut, and restore the district roots and
`roots_to_district` map to `old_root1`/`old_root2`. Leaves `partition` exactly as it
was before the tentative move.
"""
function revert_tentative_proposal!(
    partition::LinkCutPartition,
    distPair::Tuple{Int, Int},
    links::Vector{Vector{T}},
    cuts::Vector{Vector{T}},
    old_root1::Int,
    old_root2::Int
) where T <: Int
    new_root1 = partition.district_roots[distPair[1]]
    new_root2 = partition.district_roots[distPair[2]]

    for link in links
        evert!(partition.lct.nodes[link[2]])
        cut!(partition.lct.nodes[link[1]])
    end
    for cut in cuts
        u = partition.lct.nodes[cut[1]]
        v = partition.lct.nodes[cut[2]]
        evert!(u)
        link!(u,v)
    end

    partition.district_roots[distPair[1]] = old_root1
    partition.district_roots[distPair[2]] = old_root2
    delete!(partition.roots_to_district, new_root1)
    delete!(partition.roots_to_district, new_root2)
    partition.roots_to_district[old_root1] = distPair[1]
    partition.roots_to_district[old_root2] = distPair[2]
    evert!(partition.lct.nodes[old_root1])
    evert!(partition.lct.nodes[old_root2])
end

"""
    find_proposal_prob_ratio!(partition, distPair, links, cuts,
                              sum_edge_weight_products, w1w2_cuts_inv,
                              w1w2_links_inv, swap_link11, constraints)

Tentatively apply the `links`/`cuts` to the link-cut tree, recompute the changed
districts' roots, assignment, and cross-district edges, and check `constraints`. If
the proposal is valid, return `(prob, update)` where `prob` is the proposal
probability ratio (accounting for the change in number of adjacent district pairs,
the change in boundary-edge counts, and edge-weight factors) and `update` is the
[`Update`](@ref) describing the move; otherwise return `(0, nothing)`. Either way the
tentative edits are reverted before returning, so `partition` is left unchanged.
"""
function find_proposal_prob_ratio!(
    partition::LinkCutPartition,
    distPair::Tuple{Int, Int},
    links::Vector{Vector{T}},
    cuts::Vector{Vector{T}},
    sum_edge_weight_products::Float64,
    w1w2_cuts_inv::Float64,
    w1w2_links_inv::Float64,
    swap_link11::Bool,
    constraints::Constraints
) where T <: Int
    for cut in cuts
        cut!(partition.lct.nodes[cut[2]])
    end
    for link in links
        u = partition.lct.nodes[link[1]]
        v = partition.lct.nodes[link[2]]
        evert!(u)
        link!(u,v)
    end

    # @show partition.district_roots
    cut11_dist_init = partition.node_to_dist[cuts[1][1]]
    new_roots = (find_root!(partition.lct.nodes[cuts[1][1]]),
                 find_root!(partition.lct.nodes[cuts[1][2]]))
    # @show partition.district_roots
    # swap if needed
    l11node = partition.lct.nodes[links[1][1]]
    l11dist_cur = partition.node_to_dist[links[1][1]]
    r11_new = find_root!(l11node)
    r11_root_ind_new = (r11_new != new_roots[1]) + 1
    r11_root_ind_cur = (l11dist_cur != distPair[1]) + 1
    if (r11_root_ind_new == r11_root_ind_cur) ⊻ !swap_link11
        new_roots = (new_roots[2], new_roots[1])
    end

    # modify roots
    old_root1 = partition.district_roots[distPair[1]]
    old_root2 = partition.district_roots[distPair[2]]
    partition.district_roots[distPair[1]] = new_roots[1].vertex
    partition.district_roots[distPair[2]] = new_roots[2].vertex
    partition.roots_to_district[new_roots[1].vertex] = distPair[1]
    partition.roots_to_district[new_roots[2].vertex] = distPair[2]

    # modify district assignments
    partition.node_to_dist_update .= partition.node_to_dist
    assign_district_map!(partition, collect(distPair), update = true)

    new_cross_d_edg = Dict{Tuple{Int64,Int64}, Set{SimpleWeightedEdge}}()
    find_cross_district_edges!(partition, collect(distPair), new_cross_d_edg,
                               update=true)

    proposal_update = Update(distPair, links, cuts, new_cross_d_edg, swap_link11)
    energy_data_snapshot = deepcopy(partition.energy_data)
    constraints_ok = satisfies_constraints(partition, constraints,
                                           collect(distPair);
                                           update=proposal_update)
    partition.energy_data = energy_data_snapshot
    if !constraints_ok
        revert_tentative_proposal!(partition, distPair, links, cuts,
                                   old_root1, old_root2)
        partition.node_to_dist_update .= partition.node_to_dist
        return 0, nothing
    end

    old_keys = [k for k in keys(partition.cross_district_edges) 
                if distPair[1] in k || distPair[2] in k]

    delta_adj_dists = length(keys(new_cross_d_edg)) - length(old_keys)
    old_edges = length(partition.cross_district_edges[distPair])
    new_edges = length(new_cross_d_edg[distPair])
    # @show delta_adj_dists
    # @show old_edges, new_edges

    old_adj_dists = length(keys(partition.cross_district_edges))
    prob = old_adj_dists/(old_adj_dists+delta_adj_dists)
    prob*= old_edges*(old_edges-1)/(new_edges*(new_edges-1))

    # account for differences in cummulative sum on edges; only need if graph
    # is weighted
    prob*= sum_edge_weight_products
    prob/= (sum_edge_weight_products + w1w2_links_inv - w1w2_cuts_inv)

    # revert
    revert_tentative_proposal!(partition, distPair, links, cuts,
                               old_root1, old_root2)

    return prob, proposal_update
end

"""
    get_log_tree_count_ratio(partition, distPair)

Return the change in total log spanning-tree count for the two districts in
`distPair` under the proposed move: the post-move log counts minus the cached
pre-move log counts. (Helper for spanning-tree–weighted measures; relies on cached
per-district `log_tree_counts`.)
"""
function get_log_tree_count_ratio(
    partition::LinkCutPartition,
    distPair::Tuple{Int, Int}
)::Float64
    log_tree_count_ratio = 0
    log_count1 = partition.log_tree_counts[distPair[1]]
    log_count2 = partition.log_tree_counts[distPair[2]]
    log_tree_count_ratio -= log_count1+log_count2

    new_log_tree_counts = get_log_tree_counts(partition, distPair, update=true)
    log_tree_count_ratio += sum(new_log_tree_counts)
    return log_tree_count_ratio
end

"""
    get_link_path_ind(link_ind, uPath, vPath)

Return the cycle position (index into the collapsed cycle) of the vertex `link_ind`,
which must be one of the four arm endpoints (`uPath`/`vPath` first or last). Used to
locate the "link11" edge along the cycle for the assignment-swap check. Throws if
`link_ind` is not one of those endpoints.
"""
function get_link_path_ind(
    link_ind::T,
    uPath::Vector{<:Node},
    vPath::Vector{<:Node}
)::T where T <: Int
    if uPath[end].vertex == link_ind
        return 1
    elseif uPath[1].vertex == link_ind
        return length(uPath)
    elseif vPath[end].vertex == link_ind
        return length(uPath) + length(vPath)
    elseif vPath[1].vertex == link_ind
        return length(uPath)+1
    else
        throw("Couldn't find link11 index in appropriate spot")
    end
end

"""
    swap_assignment_check(path_ind, edge_inds, uPath, vPath, cycle_weights)::Bool

Determine whether the two new districts' root labels need to be swapped so that each
keeps a consistent district identity after the cut. It works out which side of the
two cuts (`edge_inds`) carries the larger population and which side the link-11 edge
(at `path_ind`) ends up on, returning the parity that
[`find_proposal_prob_ratio!`](@ref) uses as `swap_link11`.
"""
function swap_assignment_check(
    path_ind::T,
    edge_inds::Tuple{T,T},
    uPath::Vector{<:Node},
    vPath::Vector{<:Node},
    cycle_weights::Vector{Float64}
)::Bool where T <: Int
    # edge_inds interval assigned to u district?
    overlap1 = 0
    tot_pop = sum(cycle_weights)
    if edge_inds[1] <= length(uPath)
        overlap1 += sum(@view cycle_weights[edge_inds[1]:min(
                                                   length(uPath),edge_inds[2])])
    elseif edge_inds[1] > length(uPath)+1
        overlap1 += sum(@view cycle_weights[length(uPath)+1:edge_inds[1]-1])
    end
    if edge_inds[2] < length(cycle_weights)
        overlap1 += sum(@view cycle_weights[max(length(uPath)+1, edge_inds[2]+1):end])
    end
    # note: overlap2 = tot_pop - overlap1 and check is overlap1 > overlap2?
    uPathToInterval = (2*overlap1 > tot_pop) 
    # @show overlap1, tot_pop, tot_pop-overlap1

    l11_in_interval = (edge_inds[1] <= path_ind <= edge_inds[2])
    l11_in_uPath = (path_ind <= length(uPath))
    # @show l11_in_interval, l11_in_uPath, uPathToInterval

    return (l11_in_uPath ⊻ l11_in_interval) ⊻ !uPathToInterval
    # l11_in_uPath && l11_in_interval && uPathToInterval -> false
    # l11_in_uPath && !l11_in_interval && uPathToInterval -> true
    # !l11_in_uPath && l11_in_interval && uPathToInterval -> true
    # !l11_in_uPath && !l11_in_interval && uPathToInterval -> f
    # l11_in_uPath && l11_in_interval && !uPathToInterval -> t
    # l11_in_uPath && !l11_in_interval && !uPathToInterval -> f
    # !l11_in_uPath && l11_in_interval && !uPathToInterval -> f
    # !l11_in_uPath && !l11_in_interval && !uPathToInterval -> t
end

"""
    lifted_tree_cycle_walk!(partition, constraints, rng; diagnostics=nothing)

Perform one 2-tree (lifted) cycle-walk proposal. Pick a random adjacent district
pair and two boundary edges, form the cycle, enumerate the population-balanced cut
pairs respecting `constraints`, sample one proportional to inverse edge-weight
products, and score the resulting move. Returns `(prob, update)` — the acceptance
probability ratio and the [`Update`](@ref) — or `(0, nothing)` if no valid move
exists. Per-proposal `diagnostics` are gathered along the way. The partition is left
unchanged; an accepted move is applied later by `update_partition!`.
"""
function lifted_tree_cycle_walk!(
    partition::LinkCutPartition,
    constraints::Constraints,
    rng::AbstractRNG;
    diagnostics::Union{Nothing,ProposalDiagnostics}=nothing
)
    distPair, probDists = get_rand_adjacent_dists(partition, rng)
    edge_pair, edge_pairProb = get_rand_edges(distPair, partition, rng)
    if edge_pair === nothing
        gather_lifted_cycle_walk_diagnostics!(diagnostics)
        return 0, nothing
    end
    
    uPath, vPath = get_paths!(partition, edge_pair)
    cycle_weights = get_collapsed_cycle_weights(uPath, vPath, partition)

    edge_pair_inds = find_cuttable_edge_pairs(cycle_weights, length(uPath), 
                                              partition, constraints)
    if length(edge_pair_inds) == 0
        evert!(partition.lct.nodes[partition.district_roots[distPair[1]]])
        evert!(partition.lct.nodes[partition.district_roots[distPair[2]]])
        gather_lifted_cycle_walk_diagnostics!(diagnostics; 
                                              cycle_weights=cycle_weights)
        return 0, nothing
    end

    edge_pair_inds = collect(edge_pair_inds)
    cum_edge_weight_product = Vector{Float64}(undef, length(edge_pair_inds))
    graph = partition.graph.simple_graph
    for (ii, epi) in enumerate(edge_pair_inds)
        e1 = get_node_indices_from_paths(epi[1], uPath, vPath)
        e2 = get_node_indices_from_paths(epi[2]+1, uPath, vPath)
        w1 = graph.weights[e1[1].vertex, e1[2].vertex]
        w2 = graph.weights[e2[1].vertex, e2[2].vertex]
        cum_edge_weight_product[ii] = 1/(w1*w2) +
                                   (ii == 1 ? 0 : cum_edge_weight_product[ii-1])
    end
    randSamp = rand(rng)* cum_edge_weight_product[end]
    ei = 1
    while randSamp > cum_edge_weight_product[ei]
        ei += 1
    end
    edge_inds = edge_pair_inds[ei]

    e1 = get_node_indices_from_paths(edge_inds[1], uPath, vPath)
    e2 = get_node_indices_from_paths(edge_inds[2]+1, uPath, vPath)
    cuts, links = get_cuts_and_links(edge_pair, (e1, e2))

    w1w2_cuts_inv = 1.0/(graph.weights[e1[1].vertex, e1[2].vertex]*
                          graph.weights[e2[1].vertex, e2[2].vertex])
    w1w2_links_inv = 1.0/(graph.weights[src(edge_pair[1]), dst(edge_pair[1])]*
                          graph.weights[src(edge_pair[2]), dst(edge_pair[2])])

    path_ind_l11 = get_link_path_ind(links[1][1], uPath, vPath)
    swap_link11 = swap_assignment_check(path_ind_l11, edge_inds, uPath, vPath, 
                                        cycle_weights)

    p, update = find_proposal_prob_ratio!(partition, distPair, links, cuts, 
                                          cum_edge_weight_product[end],
                                          w1w2_cuts_inv, w1w2_links_inv, 
                                          swap_link11, constraints)

    gather_lifted_cycle_walk_diagnostics!(diagnostics; accept_ratio=p,
                                          cycle_weights=cycle_weights, 
                                          dist_pair=distPair,
                                          edge_pair=edge_pair, 
                                          edge_inds=edge_inds,
                                          edge_pair_inds=edge_pair_inds,
                                          swap_data=(path_ind_l11, swap_link11),
                                          len_uPath=length(uPath),
                                          partition=partition)
    return p, update
end


"""
    build_lifted_tree_cycle_walk(constraints)

Return a 2-tree cycle-walk proposal closure `f(partition, rng; diagnostics=nothing)`
that calls [`lifted_tree_cycle_walk!`](@ref) with the captured `constraints`. Exported
under the aliases `build_cycle_walk` and `build_two_tree_cycle_walk`. The returned
closure is what gets passed (often in a weighted mixture) to
[`run_metropolis_hastings!`](@ref).
"""
function build_lifted_tree_cycle_walk(
    constraints::Constraints
)
    f(p::LinkCutPartition, 
      r::AbstractRNG; 
      diagnostics::Union{Nothing, ProposalDiagnostics}=nothing) =
        lifted_tree_cycle_walk!(p, constraints, r; diagnostics=diagnostics)
    return name_proposal!(f, "two_tree_cycle_walk")
end

const build_cycle_walk = build_lifted_tree_cycle_walk
const build_two_tree_cycle_walk = build_lifted_tree_cycle_walk
