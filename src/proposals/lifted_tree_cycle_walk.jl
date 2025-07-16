function get_rand_adjacent_dists(
    partition::LinkCutPartition,
    rng::AbstractRNG
)
    distPair = rand_dict_key(rng, partition.cross_district_edges)
    return distPair, 1/length(partition.cross_district_edges)
end

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

function get_collapsed_cycle_weights(
    uPath::Vector{Node}, 
    vPath::Vector{Node}, 
    partition::LinkCutPartition;
    field=partition.graph.pop_col
)::Vector{Float64}
    uPath_rev = reverse(uPath)
    u1 = partition.lct.nodes[uPath[1].vertex]
    v1 = partition.lct.nodes[vPath[1].vertex]
    u_cut_pop_dict = topological_sort(u1, partition; field=field)
    v_cut_pop_dict = topological_sort(v1, partition; field=field)
    path_length = length(uPath) + length(vPath)
    # @show u_cut_pop_dict
    # @show v_cut_pop_dict
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
function find_first_valid_cut(
    cycle_weights::Vector, 
    initial_cut_index::Int, 
    min_pop::Real, 
    max_pop::Real,
    totpop_uv::Real
)
    first_valid_cut = initial_cut_index
    while first_valid_cut > 1
        first_valid_cut -= 1
        @assert first_valid_cut > 0
        pop1 = sum(cycle_weights[1:first_valid_cut])
        pop2 = totpop_uv - pop1
        if !(min_pop <= pop1 <= max_pop && min_pop <= pop2 <= max_pop)
            first_valid_cut += 1
            break
        end
    end
    return first_valid_cut
end

function find_cuttable_edge_pairs(
    cycle_weights::Vector{U}, 
    initial_cut_index::Int,
    partition::LinkCutPartition,
    constraints::Dict{Type{T} where T<:AbstractConstraint, AbstractConstraint}
) where U <: Real
    path_length = length(cycle_weights)
    totpop_uv = sum(cycle_weights)
    min_pop = constraints[PopulationConstraint].min_pop
    max_pop = constraints[PopulationConstraint].max_pop
    possible_pairs = Set()

    ### find valid cut with smallest index when paired with u1,v1
    first_valid_cut = find_first_valid_cut(cycle_weights, initial_cut_index, 
                                           min_pop, max_pop, totpop_uv)

    for cut1 = 1:path_length
        found_cut = false
        for cut2 = max(cut1, first_valid_cut):path_length-1
            pop1 = sum(cycle_weights[cut1:cut2])
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

function get_node_indices_from_paths(
    edge_ind::Int, 
    uPath::Vector{Node},
    vPath::Vector{Node}
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

function find_proposal_prob_ratio!(
    partition::LinkCutPartition, 
    distPair::Tuple{Int, Int}, 
    links::Vector{Vector{T}}, 
    cuts::Vector{Vector{T}},
    sum_edge_weight_products::Float64,
    w1w2_cuts_inv::Float64,
    w1w2_links_inv::Float64,
    swap_link11::Bool
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
    # revert roots
    partition.district_roots[distPair[1]] = old_root1
    partition.district_roots[distPair[2]] = old_root2
    delete!(partition.roots_to_district, new_roots[1].vertex)
    delete!(partition.roots_to_district, new_roots[2].vertex)
    partition.roots_to_district[old_root1] = distPair[1]
    partition.roots_to_district[old_root2] = distPair[2]
    evert!(partition.lct.nodes[old_root1])
    evert!(partition.lct.nodes[old_root2])

    return prob, new_cross_d_edg
end

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

function get_link_path_ind(
    link_ind::T, 
    uPath::Vector{Node}, 
    vPath::Vector{Node}
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

function swap_assignment_check(
    path_ind::T, 
    edge_inds::Tuple{T,T}, 
    uPath::Vector{Node}, 
    vPath::Vector{Node}, 
    cycle_weights::Vector{Float64}
)::Bool where T <: Int
    # edge_inds interval assigned to u district?
    overlap1 = 0
    tot_pop = sum(cycle_weights)
    if edge_inds[1] <= length(uPath)
        overlap1 += sum(cycle_weights[edge_inds[1]:min(
                                                   length(uPath),edge_inds[2])])
    elseif edge_inds[1] > length(uPath)+1
        overlap1 += sum(cycle_weights[length(uPath)+1:edge_inds[1]-1])
    end
    if edge_inds[2] < length(cycle_weights)
        overlap1 += sum(cycle_weights[max(length(uPath)+1, edge_inds[2]+1):end])
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

function lifted_tree_cycle_walk!(
    partition::LinkCutPartition,
    constraints::Dict{Type{T} where T<:AbstractConstraint, AbstractConstraint},
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

    p, new_cross_d_edg = find_proposal_prob_ratio!(partition, distPair, links, 
                                                   cuts, 
                                                   cum_edge_weight_product[end],
                                                   w1w2_cuts_inv,
                                                   w1w2_links_inv, 
                                                   swap_link11)

    gather_lifted_cycle_walk_diagnostics!(diagnostics; accept_ratio=p,
                                          cycle_weights=cycle_weights, 
                                          dist_pair=distPair,
                                          edge_pair=edge_pair, 
                                          edge_inds=edge_inds,
                                          edge_pair_inds=edge_pair_inds,
                                          swap_data=(path_ind_l11, swap_link11),
                                          len_uPath=length(uPath),
                                          partition=partition)
    # partition.update_index += 1
    return p, Update(distPair, links, cuts, new_cross_d_edg, swap_link11)
                     # partition.update_identifier+1)
end


""""""
function build_lifted_tree_cycle_walk(
    constraints::Dict{Type{T} where T<:AbstractConstraint, AbstractConstraint}
)
    f(p::LinkCutPartition, 
      r::AbstractRNG; 
      diagnostics::Union{Nothing, ProposalDiagnostics}=nothing) = 
        lifted_tree_cycle_walk!(p, constraints, r; diagnostics=diagnostics)
    return f
end

const build_cycle_walk = build_lifted_tree_cycle_walk
const build_two_tree_cycle_walk = build_lifted_tree_cycle_walk
