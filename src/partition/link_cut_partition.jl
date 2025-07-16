abstract type AbstractTreeState end

mutable struct LinkCutPartition <: AbstractTreeState
    num_dists::Int64
    cross_district_edges::Dict{Tuple{Int64,Int64}, Set{SimpleWeightedEdge}}
    district_roots::Vector{Int64}
    roots_to_district::Dict{Int64, Int64}
    energy_data::Dict{Union{DataType, Tuple{DataType, Tuple}}, 
                      AbstractEnergyData}
    node_to_dist::Vector{Int64}
    node_to_dist_update::Vector{Int64}
    lct::LinkCutTree
    node_col::String
    graph::BaseGraph
    identifier::Int64
    # update_identifier::Int64
end

""""""
function LinkCutPartition(
    partition::MultiLevelPartition,
    rng::AbstractRNG
)::LinkCutPartition
    edge_type = edgetype(partition.subgraphs[1].graphs_by_level[1][()])
    edges = Vector{edge_type}(undef, 0)

    base_graph = partition.graph.graphs_by_level[end]
    node_to_dist = Vector{Int64}(undef, base_graph.num_nodes)
    node_to_dist_update = Vector{Int64}(undef, base_graph.num_nodes)
    log_tree_counts = Vector{Int64}(undef, partition.num_dists)

    for di = 1:partition.num_dists
        dg = partition.subgraphs[di].graphs_by_level[1][()]
        vmap = partition.subgraphs[di].vmaps[1][()]
        tree_edges = wilson_rst(dg, rng)
        tree_edges = [edge_type(vmap[src(e)],vmap[dst(e)], 
                                get_weight(dg, src(e), dst(e))) 
                      for e in tree_edges]
        edges = [edges; tree_edges]
    end

    lct = link_cut_tree(base_graph.simple_graph, edges)
    district_roots, roots_to_district = get_district_roots(lct)
    @assert length(district_roots) == partition.num_dists

    energy_data = Dict{Union{DataType, Tuple{DataType, Tuple}}, 
                       AbstractEnergyData}()
    
    identifier = rand(rng, typemin(Int64):0)
    lcp = LinkCutPartition(partition.num_dists, 
                           Dict{Tuple{Int64,Int64}, Set{SimpleWeightedEdge}}(),
                           district_roots, roots_to_district, energy_data,
                           node_to_dist, node_to_dist_update, lct,
                           partition.graph.levels[1], base_graph,
                           identifier)#, identifier)
    assign_district_map!(lcp)
    find_cross_district_edges!(lcp)
    return lcp
end

""""""
function get_district_roots(lct::LinkCutTree)
    district_roots = Vector{Int}(undef, 0)
    roots_to_district = Dict{Int, Int}()
    for n in lct.nodes
        r = find_root!(n).vertex
        if r ∉ district_roots
            push!(district_roots, r)
            roots_to_district[r] = length(district_roots)
        end
    end 
    return district_roots, roots_to_district
end

""""""
function find_cross_district_edges!(
    lcp::LinkCutPartition,
    districts::Vector{Int} = collect(1:lcp.num_dists),
    cross_district_edges::Dict{Tuple{Int,Int}, Set{SimpleWeightedEdge}} 
    = lcp.cross_district_edges;
    update = false
)
    if update
        node_to_dist = lcp.node_to_dist_update
    else
        node_to_dist = lcp.node_to_dist
    end
    graph = lcp.graph
    simple_graph = graph.simple_graph
    visited_nodes = Vector{Bool}([false for _ = 1:nv(simple_graph)])
    for di in districts
        r = lcp.district_roots[di]
        queue = [r]
        while length(queue) > 0
            v = pop!(queue)
            visited_nodes[v] = true
            for n in neighbors(simple_graph, v)
                if visited_nodes[n] continue end
                if get_weight(simple_graph, v, n) == 0 continue end
                # rn = find_root!(lcp.lct.nodes[n]).vertex
                dj = node_to_dist[n]
                # if (rn == r) != (dj == di)
                #     @show "problem here"
                #     @show n, rn, r, di, dj
                #     @assert false
                # end
                # if rn == r
                if di == dj
                    push!(queue, n)
                else
                    # dj = lcp.roots_to_district[rn]
                    dij = (min(di,dj), max(di, dj))
                    if !haskey(cross_district_edges, dij)
                        cross_district_edges[dij] =Set{SimpleWeightedEdge}()
                    end 
                    w = get_weight(simple_graph, v, n)
                    e = SimpleWeightedEdge(v, n, w)
                    push!(cross_district_edges[dij], e)
                end
            end
        end
    end
    return cross_district_edges
end

function assign_district_map!(
    partition::LinkCutPartition, 
    districts::Vector{Int} = collect(1:partition.num_dists);
    update = false
) 
    if update
        node_to_dist = partition.node_to_dist_update
    else 
        node_to_dist = partition.node_to_dist
    end

    for di in districts
        ri = partition.district_roots[di]
        node_r = partition.lct.nodes[ri]
        expose!(node_r)
        queue = [node_r]
        while length(queue) > 0
            node = pop!(queue)
            node_to_dist[node.vertex] = di
            for ii = 1:2
                if node.children[ii] != nothing
                    push!(queue, node.children[ii])
                end
            end
            for n in node.pathChildren
                push!(queue, n)
            end
        end
    end
end

function sum_cc(
    node::Node, 
    graph::BaseGraph, 
    field::Union{Nothing,String}=nothing,
    start=true
) 
    if start
        expose!(node)
    end
    if field === nothing
        sum = 1
    else
        sum = graph.node_attributes[node.vertex][field]
    end
    for ii = 1:2
        if node.children[ii] != nothing
            sum += sum_cc(node.children[ii], graph, field, false)
        end
    end
    for n in node.pathChildren
        sum += sum_cc(n, graph, field, false)
    end
    return sum
end

""""""
function sum_cc(
    node::Node, 
    partition::LinkCutPartition, 
    field::Union{Nothing,String}=nothing,
    start=true
) 
    return sum_cc(node, partition.graph, field, start)
end


function topological_sort!(
    cut_remainder::Dict{Int, Real}, 
    node::Union{Node, Nothing}, 
    source::Node,
    graph::BaseGraph,
    field::Union{Nothing,String},
    reversed::Bool=false,
    mass::Real=0,
)
    if node === nothing
        return 0
    end

    remainder = 0
    reversed ⊻= node.reversed
    if !reversed; lc,rc = 1,2; else lc,rc=2,1 end
    remainder += topological_sort!(cut_remainder, node.children[rc], node, 
                                   graph, field, reversed, mass)
    for n in node.pathChildren
        remainder += topological_sort!(cut_remainder, n, node, graph, field)
    end
    
    index = node.vertex
    node_val = 1 # if nothing, just count nodes
    if field !== nothing
        node_val = graph.node_attributes[index][field]
    end 
    cut_remainder[index] = remainder + node_val + mass

    if source != node.children[lc]
        remainder += topological_sort!(cut_remainder, node.children[lc], node,
                                       graph, field, reversed,
                                       cut_remainder[index])
    end
    return remainder + node_val
end

""""""
function topological_sort(root::Node, graph::BaseGraph; 
                          field::Union{Nothing,String}=graph.pop_col)
    cut_remainder = Dict{Int,Real}()
    evert!(root)
    if !root.reversed; lc,rc = 1,2; else lc,rc=2,1 end

    # # the following should all be handled by the evert!; this is to confirm
    # @assert root.children[lc] === nothing
    # @assert root.parent === nothing
    # @assert root.pathParent === nothing

    total = topological_sort!(cut_remainder, root.children[rc], root, 
                              graph, field, root.reversed)

    for n in root.pathChildren
        total += topological_sort!(cut_remainder, n, root, graph, field)
    end
    root_pop = 1 # if field === nothing, just count number of nodes
    if field !== nothing
        root_pop = graph.node_attributes[root.vertex][field]
    end
    cut_remainder[root.vertex] = root_pop + total

    return cut_remainder
end

""""""
function topological_sort(root::Node, tree_rep::AbstractTreeState; 
                          field::Union{Nothing,String}=tree_rep.graph.pop_col)
    return topological_sort(root, tree_rep.graph; field=field)
end

"
Gets the diameter of the trees held in a LinkCutPartition
"
function get_diameters(partition::LinkCutPartition)
    return get_diameter.(partition.lct.nodes[partition.district_roots])
end

"
gets the neighbor list representations of the trees held in a LinkCutPartition
"

function get_neighbor_lists(partition::LinkCutPartition)
    roots = partition.lct.nodes[partition.district_roots]
    edgeLists = get_connected_edge_list.(roots)
    graphs = get_neighbor_lists(edgeLists)
    return graphs
end

"
gets the degree distribution of the link/cut trees held in a LinkCutPartition
"
function get_degree_distributions(partition::LinkCutPartition)
    edgeLists=LiftedTreeWalk.get_connected_edge_list.(
                                  partition.lct.nodes[partition.district_roots])
    distributionsList=get_degree_distribution.(edgeLists)
    return distributionsList
end 

"
calculates the average degree of the trees in a LinkCutPartition
"
function get_average_degrees(partition::LinkCutPartition)
    edgeLists=LiftedTreeWalk.get_connected_edge_list.(
                                  partition.lct.nodes[partition.district_roots])
    averageDegreeList=get_average_degree.(edgeLists)
    return averageDegreeList
end 


"
calculates the center of the trees in a LinkCutPartition and then the L^p norm 
of the distances of the vertices from that center vertex
"
function get_center_moments(partition::LinkCutPartition;p=1)
    edgeList=LiftedTreeWalk.get_connected_edge_list.(
                                  partition.lct.nodes[partition.district_roots])

    center_moments=get_center_moment.(edgeList,p=p)

    return center_moments

end

"
calculates the center of the trees in a LinkCutPartition and then the L^p norm 
of the distances of the leaves from that center vertex
"
function get_center_leaves_moments(partition::LinkCutPartition;p=1)
    edgeList=LiftedTreeWalk.get_connected_edge_list.(
                                  partition.lct.nodes[partition.district_roots])

    center_moments=get_center_leaves_moment.(edgeList,p=p)

    return center_moments

end














