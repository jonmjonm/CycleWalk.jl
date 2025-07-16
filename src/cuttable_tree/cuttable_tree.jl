mutable struct PotentiallyCuttalbleTree <: AbstractTreeState
    num_dists::Int64
    root::Int64
    cuts::Vector{Vector{Int64}}
    sort::Union{Nothing, Dict{Int64,Float64}}
    lct::LinkCutTree
    graph::BaseGraph
    node_col::String
    identifier::Int64
    # energy_data::Dict{Union{DataType, Tuple{DataType, Tuple}}, 
    #                   AbstractEnergyData}
    # node_to_dist::Vector{Int64}
    # node_to_dist_update::Vector{Int64}
end

""""""
function PotentiallyCuttalbleTree(
    node_col::String,
    constraints::Dict{Type{T} where T<:AbstractConstraint, AbstractConstraint},
    base_graph::BaseGraph,
    num_dists::Int64;
    rng::AbstractRNG=PCG.PCGStateOneseq(UInt64),
    verbose::Bool=false
)::LinkCutPartition

    graph = MultiLevelGraph(base_graph, [node_col])
    initial_partition = MultiLevelPartition(graph, constraints, num_dists; 
                                            rng=rng);
    tree_partition = LinkCutPartition(initial_partition, rng);

    district_graph = extractDistrictGraph(tree_partition)
    district_tree = wilson_rst(district_graph)
    lct = deepcopy(tree_partition.lct)
    linkForest!(lct, tree_partition, district_tree, rng)
    # cutset = determineCutSet(lct, num_cuts, constraints)
    # root = find_root(lct).vertex

    identifier = rand(rng, typemin(Int64):0)

    # fct = PotentiallyCuttalbleTree(num_dists-1, root, cutset, nothing, lct, 
    #                             base_graph, node_col, identifier) 

    # topological_sort!(fct)
    # evert!(partition.lct.nodes[link[1]])
    # link!(partition.lct.nodes[link[1]], partition.lct.nodes[link[2]])

    # if verbose
    #     @show collect(keys(initial_partition.district_to_nodes[1]))[1:min(10,end)]
    # end


    cuts = Vector{Vector{Int64}}(undef, 0)
    fct = PotentiallyCuttalbleTree(num_dists, find_root!(lct.nodes[1]), cuts,
                                   nothing, lct, base_graph, node_col, 
                                   identifier)
    return fct
end

""""""
function topological_sort!(fct::PotentiallyCuttalbleTree)
    fct.sort = topological_sort(fct.root, fct)
end



""""""
function extractDistrictGraph(tree_partition::LinkCutPartition)
    g = SimpleWeightedGraph(tree_partition.num_dists)
    for e in keys(tree_partition.cross_district_edges)
        add_edge!(g, e)
    end
    return g
end


""""""
function linkForest!(
    lct::LinkCutTree, 
    tree_partition::LinkCutPartition, 
    district_tree::Vector{Graphs.SimpleGraphs.SimpleEdge},
    rng::AbstractRNG
)
    for e in district_tree
        n1, n2 = src(e), dst(e)
        ek = (n1, n2)
        if n1 > n2 
            ek = (n2, n1)
        end
        newe = rand(rng, tree_partition.cross_district_edges[ek])
        u, v = lct.nodes[src(newe)], lct.nodes[dst(newe)]
        evert!(u)
        link!(u, v)
    end
end

# """"""
# function determineCutSet(
#     lct::LinkCutTree, 
#     num_dists::U, 
#     constraints::Dict{Type{T} where T<:AbstractConstraint, AbstractConstraint}
# ) where U <: Int

# end

""""""
function topological_sort_cut!(
    cut_remainder::Dict{I, T}, 
    cuttable_edge_info::Vector{Tuple{I, Vector{I}}},
    node::Union{Node, Nothing}, 
    source::Node,
    graph::BaseGraph,
    field::Union{Nothing,String},
    field_total::T,
    bounds::Tuple{T2, T2},
    num_dists::I2,
    reversed::Bool=false,
    mass::T=T(0)
) where {I<:Int, I2<:Int, T<:Real, T2<:Real}
    if node === nothing
        return 0
    end

    remainder = 0
    reversed ⊻= node.reversed
    if !reversed; lc,rc = 1,2; else lc,rc=2,1 end
    remainder += topological_sort_cut!(cut_remainder, cuttable_edges, 
                                       node.children[rc], node, graph, 
                                       field, field_total, bounds, num_dists, 
                                       reversed, mass)
    for n in node.pathChildren
        remainder += topological_sort_cut!(cut_remainder, cuttable_edges, n, 
                                           node, graph, field, field_total, 
                                           bounds, num_dists)
    end
    
    index = node.vertex
    node_val = 1 # if nothing, just count nodes
    if field !== nothing
        node_val = graph.node_attributes[index][field]
    end 
    cut_remainder[index] = remainder + node_val + mass

    # is cuttable?
    cuttable_info = cuttable_huh(cut_remainder[index], field_total, bounds, 
                                 num_dists)
    if cuttable_info[1]
        if cuttable_info[2] == cuttable_edges
        # nbrs = lower_neighbors(node)
        @show index, cuttable_info[2]
    end

    if source != node.children[lc]
        remainder += topological_sort_cut!(cut_remainder, cuttable_edges,
                                           node.children[lc], node, graph, 
                                           field, field_total, bounds, 
                                           num_dists, reversed, 
                                           cut_remainder[index])
    else
        @show "here???" # what case is this avoiding??
        @assert false
    end
    return remainder + node_val
end

""""""
function cuttable_huh(
    cut_val_lower::T,
    field_total::T, 
    bounds::Tuple{T, T},
    num_dists::Int64
) where T <: Real
    i_max = floor(cut_val_lower/bounds[1])
    i_min = ceil(cut_val_lower/bounds[2])
    !(i_min <= i_max) && return (false, nothing)

    cut_val_upper = field_total - cut_val_lower
    di_max = floor(cut_val_upper/bounds[1])
    di_min = ceil(cut_val_upper/bounds[2])
    !(di_min <= di_max) && return (false, nothing)

    !(i_min + di_min <= num_dists <= i_max + di_max) && return (false, nothing)

    for ii = i_min:i_max
        for jj = di_min:di_max
            if ii + jj == num_dists
                return (true, ii)
            end
        end
    end

    @assert false
end


""""""
function topological_sort_cut(
    root::Node, graph::BaseGraph, num_dists::Int64, 
    field::Union{Nothing, String},
    bounds::Tuple{I, I}
) where I<: Real
    
    field_total = graph.total_pop
    if field != graph.pop_col
        field_total = sum_cc(node, graph, field)
    end
    cut_remainder = Dict{typeof(root.vertex) ,typeof(field_total)}()
    cuttable_edges = Vector{Vector{typeof(root.vertex)}}(undef, 0)

    evert!(root)
    if !root.reversed; lc,rc = 1,2; else lc,rc=2,1 end

    # the following should all be handled by the evert!; this is to confirm
    @assert root.children[lc] === nothing
    @assert root.parent === nothing
    @assert root.pathParent === nothing

    total = topological_sort_cut!(cut_remainder, cuttable_edges, 
                                  root.children[rc], root, graph, field, 
                                  field_total, bounds, num_dists, 
                                  root.reversed)

    for n in root.pathChildren
        total += topological_sort_cut!(cut_remainder, cuttable_edges, n, root, 
                                       graph, field, field_total, bounds, 
                                       num_dists)
    end
    root_val = 1 # if field === nothing, just count number of nodes
    if field !== nothing
        root_val = graph.node_attributes[root.vertex][field]
    end
    cut_remainder[root.vertex] = root_val + total

    # check the whole graph total
    @assert field_total == cut_remainder[root.vertex]

    return cut_remainder, cuttable_edges
end






