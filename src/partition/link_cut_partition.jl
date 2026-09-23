"""
    LinkCutPartition <: AbstractPartition

A districting plan represented as a spanning forest — one spanning tree per
district — stored in an augmented link-cut tree (`lct`) so that cycle-walk
proposals can link/cut edges and query subtree populations in `O(log n)`.

Key fields:
- `num_dists`: number of districts.
- `lct`: the population-augmented `LinkCutTree` holding the spanning forest.
- `district_roots` / `roots_to_district`: the link-cut root vertex of each district
  and the inverse map.
- `node_to_dist` / `node_to_dist_update`: current and tentative (proposed) district
  assignment of each node.
- `cross_district_edges`: for each adjacent district pair, the set of graph edges
  crossing between them (the candidate cut/link edges).
- `energy_data`: cache of incrementally-maintained per-energy data
  (`AbstractEnergyData`), keyed by energy type.
- `node_col`, `graph`, `node_pops`: the node-id column, base graph, and a concrete
  cache of per-node populations.
- `identifier`: a counter bumped on every accepted update, used to invalidate caches.

Construct one with the [`LinkCutPartition`](@ref) constructors rather than directly.
"""
mutable struct LinkCutPartition <: AbstractPartition
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
    node_pops::Vector{Float64}   # cached graph.node_attributes[ni][pop_col], concrete
    identifier::Int64
    # update_identifier::Int64
end

"""
    clone_for_annealing(partition)

Return an independent copy of `partition` suitable for running an annealing chain on
its own task, cloning only the state that sampling mutates. The base graph
(`graph`), the node-id column (`node_col`), and the cached per-node populations
(`node_pops`) are read-only for the entire run, so they are *shared* rather than
deep-copied — on a large graph this is the bulk of a full `deepcopy(partition)`'s
bytes (the graph alone is ~75% of them). Everything a proposal touches (the link-cut
tree, district bookkeeping, energy caches, and node→district vectors) is copied.

This is what [`run_annealed_importance_sampling!`](@ref) uses to fan samples out to
worker tasks; it is *not* a general-purpose `deepcopy` replacement, since it assumes
the graph is never mutated while the clone is alive.
"""
function clone_for_annealing(p::LinkCutPartition)
    return LinkCutPartition(
        p.num_dists,
        deepcopy(p.cross_district_edges),   # Sets of edges are mutated
        copy(p.district_roots),
        copy(p.roots_to_district),
        deepcopy(p.energy_data),            # per-district caches are advanced in place
        copy(p.node_to_dist),
        copy(p.node_to_dist_update),
        copy(p.lct),                        # spanning forest — fast structural clone

        p.node_col,                         # immutable — shared
        p.graph,                            # read-only during sampling — shared
        p.node_pops,                        # read-only during sampling — shared
        p.identifier,
    )
end

"""
    LinkCutPartition(partition::MultiLevelPartition, rng=PCG.PCGStateOneseq(UInt64))

Build a `LinkCutPartition` from an existing `MultiLevelPartition` by drawing a
random spanning tree (Wilson's algorithm) of each district's subgraph and loading
all of the trees into a single population-augmented link-cut tree. District roots,
node-to-district assignments, and cross-district edges are computed from the result.
"""
function LinkCutPartition(
    partition::MultiLevelPartition,
    rng::AbstractRNG=PCG.PCGStateOneseq(UInt64)
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

    # PopAug tree: carries per-node population so subtree populations are
    # maintained incrementally (see get_collapsed_cycle_weights / subtree_pop).
    pop_col = base_graph.pop_col
    pops = Float64[Float64(base_graph.node_attributes[ni][pop_col])
                   for ni in 1:base_graph.num_nodes]
    lct = pop_link_cut_tree(base_graph.simple_graph, edges, pops)
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
                           pops, identifier)#, identifier)
    assign_district_map!(lcp)
    find_cross_district_edges!(lcp)
    return lcp
end

"""
    LinkCutPartition(graph::MultiLevelGraph, constraints, num_dists;
                     rng=PCG.PCGStateOneseq(UInt64), verbose=false)

Build an initial `LinkCutPartition` of `graph` into `num_dists` districts that
satisfies `constraints`. The constraints are interpreted to determine the graph
levels, an initial balanced `MultiLevelPartition` is drawn, flattened to a
node-to-district assignment, and wrapped in a link-cut tree. The resulting plan is
asserted to satisfy `constraints` (including population). This is the constructor
used in practice to seed a run.
"""
function LinkCutPartition(
    graph::MultiLevelGraph,
    constraints::Constraints,
    num_dists::U;
    rng::AbstractRNG=PCG.PCGStateOneseq(UInt64),
    verbose::Bool=false
) where U <: Int
    mfr_constraints, levels = interpret_constraints(constraints, graph)
    ml_graph = multi_level_graph(graph.graphs_by_level[end], levels)
    partition = MultiLevelPartition(ml_graph, mfr_constraints, num_dists; 
                                    rng=rng);
    node_to_dist = flatten_assignment(partition)
    partition = MultiLevelPartition(graph, node_to_dist)

    if verbose
        @show collect(keys(partition.district_to_nodes[1]))[1:min(10,end)]
    end
    lct_partition = LinkCutPartition(partition, rng)
    @assert satisfies_constraints(lct_partition, constraints; 
                                  check_population=true)
    return lct_partition;
end

"""
    relabel_districts!(partition, node_to_district)

Renumber `partition`'s districts so that every node carries the district index it has
in `node_to_district` (a `Dict` from each node's one-tuple id to its district — the
form an Atlas records, and the form [`flatten_assignment`](@ref) returns).

The `LinkCutPartition` constructors number districts by the order in which their trees
are first encountered in the link-cut tree, which has nothing to do with the labels of
the assignment a plan was built from. That is immaterial to sampling — the measure and
the constraints do not care what a district is called — but it matters when a recorded
plan is reloaded and the new samples are meant to line up with the old ones, as when
extending an existing Atlas (see `examples/run_cyclewalk_extend.jl`). Relabelling makes
district `d` after the reload the same district `d` that was written out.

`node_to_district` must describe the same plan `partition` already holds, differing
only in its labels; anything else throws, since the partition's forest cannot be
changed by renaming. Any cached `energy_data` is dropped, since it is keyed by
district; it is rebuilt from the partition the next time an energy is evaluated.
"""
function relabel_districts!(
    partition::LinkCutPartition,
    node_to_district::AbstractDict{<:Tuple{Vararg{String}}, <:Integer}
)
    graph = partition.graph
    num_dists = partition.num_dists

    # perm[current label] = target label, read off node by node
    perm = zeros(Int, num_dists)
    for ni = 1:graph.num_nodes
        key = (graph.node_attributes[ni][partition.node_col],)
        haskey(node_to_district, key) ||
            throw(ArgumentError("node $key is missing from the assignment"))
        target = node_to_district[key]
        (1 <= target <= num_dists) ||
            throw(ArgumentError("district $target of node $key is out of range " *
                                "1:$num_dists"))
        cur = partition.node_to_dist[ni]
        if perm[cur] == 0
            perm[cur] = target
        elseif perm[cur] != target
            throw(ArgumentError("the assignment is not a relabelling of the " *
                                "partition: district $cur holds nodes assigned to " *
                                "both $(perm[cur]) and $target"))
        end
    end
    length(Set(perm)) == num_dists ||
        throw(ArgumentError("the assignment is not a relabelling of the partition: " *
                            "its districts do not map one-to-one"))

    old_roots = copy(partition.district_roots)
    for cur = 1:num_dists
        partition.district_roots[perm[cur]] = old_roots[cur]
    end
    empty!(partition.roots_to_district)
    for di = 1:num_dists
        partition.roots_to_district[partition.district_roots[di]] = di
    end
    # Rebuild the derived state from the (now renumbered) roots rather than permuting
    # it in place: both are keyed by district, and recomputing once at load time is
    # cheap next to getting the permutation of a Dict's keys subtly wrong.
    assign_district_map!(partition)
    partition.node_to_dist_update .= partition.node_to_dist
    empty!(partition.cross_district_edges)
    find_cross_district_edges!(partition)
    empty!(partition.energy_data)   # per-district caches; rebuilt lazily on next use
    return partition
end

"""
    get_district_roots(lct)

Scan every node of the link-cut tree `lct` and collect the distinct tree roots,
returning `(district_roots, roots_to_district)`: a vector of root vertices (one per
connected tree / district) and a `Dict` mapping each root vertex back to its
district index.
"""
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

"""
    find_cross_district_edges!(lcp, districts=1:lcp.num_dists,
                               cross_district_edges=lcp.cross_district_edges;
                               update=false)

Recompute the cross-district (boundary) edges incident to `districts` and store them
in `cross_district_edges`, keyed by the ordered district pair. Uses the current
assignment `lcp.node_to_dist`, or the tentative `lcp.node_to_dist_update` when
`update=true`. Work is delegated to a typed inner worker (a function barrier over the
weight matrix) so the boundary-finding BFS specializes and stays allocation-light.
"""
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
    simple_graph = lcp.graph.simple_graph
    # The weighted adjacency is a SparseMatrixCSC, but `BaseGraph.simple_graph`
    # is stored with the abstract `SimpleWeightedGraph` type, so iterating its
    # CSC fields inline would be type-unstable. Pass the (concretely-typed at
    # runtime) weight matrix into a worker through a function barrier so the BFS
    # specializes and we can read each neighbor's weight straight from the column
    # instead of doing a per-edge binary-search `get_weight`.
    return _find_cross_district_edges!(lcp, districts, cross_district_edges,
                                       node_to_dist, weights(simple_graph),
                                       nv(simple_graph))
end

"""
    _find_cross_district_edges!(lcp, districts, cross_district_edges,
                                node_to_dist, wmat, num_nodes)

Typed worker behind [`find_cross_district_edges!`](@ref). Performs a BFS from each
district root over the concrete CSC weight matrix `wmat`, reading each neighbor's
weight straight from the column; intra-district edges extend the BFS, inter-district
edges are recorded under their ordered district-pair key. Zero-weight entries are
skipped. Returns `cross_district_edges`.
"""
function _find_cross_district_edges!(
    lcp::LinkCutPartition,
    districts::Vector{Int},
    cross_district_edges::Dict{Tuple{Int,Int}, Set{SimpleWeightedEdge}},
    node_to_dist::Vector{Int},
    wmat,                       # SparseMatrixCSC; untyped so no SparseArrays import
    num_nodes::Int,
)
    # Column `v` of `wmat` stores each neighbor `n = rowval[k]` together with its
    # weight `nzval[k] == get_weight(graph, v, n)` (symmetric, undirected graph).
    colptr = wmat.colptr
    rowval = wmat.rowval
    nzval = wmat.nzval
    visited_nodes = falses(num_nodes)
    for di in districts
        r = lcp.district_roots[di]
        queue = [r]
        while length(queue) > 0
            v = pop!(queue)
            visited_nodes[v] = true
            for k in colptr[v]:(colptr[v + 1] - 1)
                n = rowval[k]
                if visited_nodes[n] continue end
                w = nzval[k]
                if w == 0 continue end
                dj = node_to_dist[n]
                if di == dj
                    push!(queue, n)
                else
                    dij = (min(di,dj), max(di, dj))
                    edgeset = get!(() -> Set{SimpleWeightedEdge}(),
                                   cross_district_edges, dij)
                    push!(edgeset, SimpleWeightedEdge(v, n, w))
                end
            end
        end
    end
    return cross_district_edges
end

"""
    assign_district_map!(partition, districts=1:partition.num_dists; update=false)

Walk each district's link-cut tree from its root and write that district's index
into every member node of the assignment vector — `partition.node_to_dist`, or the
tentative `partition.node_to_dist_update` when `update=true`. Used to (re)materialize
the flat node→district map after the forest structure changes.
"""
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
            c = first_path_child(node)
            while c !== nothing
                push!(queue, c)
                c = next_path_sibling(c)
            end
        end
    end
end

"""
    sum_cc(node, partition, field, start=true)

Sum the node-attribute `field` over the entire connected component (district tree)
containing `node`, recursing through both the splay-tree children and the
path-children of the link-cut representation. `start=true` exposes `node` first; the
recursive calls pass `start=false`.
"""
function sum_cc(
    node::Node,
    partition::LinkCutPartition,
    field::String,
    start=true
)
    if start
        expose!(node)
    end
    sum = partition.graph.node_attributes[node.vertex][field]
    for ii = 1:2
        if node.children[ii] != nothing
            sum += sum_cc(node.children[ii], partition, field, false)
        end
    end
    c = first_path_child(node)
    while c !== nothing
        sum += sum_cc(c, partition, field, false)
        c = next_path_sibling(c)
    end
    return sum
end


"""
    topological_sort!(cut_remainder, node, source, partition, field, reversed=false, mass=0.0)

Recursive worker for [`topological_sort`](@ref). Traverses the link-cut subtree at
`node` (honoring lazy `reversed` flags), accumulating into `cut_remainder[v]` the
subtree weight that would be separated from the root if the edge above `v` were cut.
`field` selects the node attribute summed (or `nothing` to count nodes); `mass`
carries the running weight contributed by ancestors. Returns the total weight of the
subtree rooted at `node`.
"""
function topological_sort!(
    cut_remainder::Dict{Int, Float64},
    node::Union{Node, Nothing},
    source::Node,
    partition::LinkCutPartition,
    field::Union{Nothing,String},
    reversed::Bool=false,
    mass::Float64=0.0,
)
    if node === nothing
        return 0.0
    end

    remainder = 0.0
    reversed ⊻= node.reversed
    if !reversed; lc,rc = 1,2; else lc,rc=2,1 end
    remainder += topological_sort!(cut_remainder, node.children[rc], node, 
                                   partition, field, reversed, mass)
    c = first_path_child(node)
    while c !== nothing
        remainder += topological_sort!(cut_remainder, c, node, partition, field)
        c = next_path_sibling(c)
    end

    index = node.vertex
    # typed local: the `node_attributes` read is `Any`, so assert Float64 to keep
    # inference concrete and avoid boxing the result and everything downstream.
    node_val::Float64 = 1.0 # if nothing, just count nodes
    if field !== nothing
        node_val = partition.graph.node_attributes[index][field]
    end
    cut_remainder[index] = remainder + node_val + mass

    if source != node.children[lc]
        remainder += topological_sort!(cut_remainder, node.children[lc], node,
                                       partition, field, reversed,
                                       cut_remainder[index])
    end
    return remainder + node_val
end

"""
    topological_sort(root, partition; field=partition.graph.pop_col)

Root the district tree at `root` and return a `Dict` mapping each vertex `v` to its
"cut weight": the total `field` value (population by default; node count if
`field=nothing`) of the subtree hanging below `v`. Cutting the edge just above `v`
would separate exactly that much weight from `root`. This is the whole-tree fallback
used by [`get_collapsed_cycle_weights`](@ref) for non-population fields.
"""
function topological_sort(root::Node, partition::LinkCutPartition;
                          field::Union{Nothing,String}=partition.graph.pop_col)
    cut_remainder = Dict{Int,Float64}()
    evert!(root)
    if !root.reversed; lc,rc = 1,2; else lc,rc=2,1 end

    # # the following should all be handled by the evert!; this is to confirm
    # @assert root.children[lc] === nothing
    # @assert root.parent === nothing
    # @assert root.pathParent === nothing

    total = topological_sort!(cut_remainder, root.children[rc], root, 
                              partition, field, root.reversed)

    c = first_path_child(root)
    while c !== nothing
        total += topological_sort!(cut_remainder, c, root, partition, field)
        c = next_path_sibling(c)
    end
    root_pop::Float64 = 1.0 # if field === nothing, just count number of nodes
    if field !== nothing
        root_pop = partition.graph.node_attributes[root.vertex][field]
    end
    cut_remainder[root.vertex] = root_pop + total

    return cut_remainder
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














