"""
    get_rand_internal_edge(partition, rng)

Return a uniformly random graph edge `(u, v)` whose endpoints lie in the *same*
district (i.e. an edge internal to one district tree, including the tree's own edges).
Rejection-samples graph edges until both endpoints share a link-cut root.
"""
function get_rand_internal_edge(
    partition::LinkCutPartition,
    rng::AbstractRNG
)
    while true
        edges = partition.graph.edge_attributes
        e = rand_dict_key(rng, edges)
        u, v = collect(e)
        ru = find_root!(partition.lct.nodes[u])
        rv = find_root!(partition.lct.nodes[v])
        ru == rv && return (u,v)
    end
end

"""
    repair_partition!(partition, linked_node, original_root)

After a 1-tree move re-roots a district tree, update the bookkeeping: find the new
root of the tree now containing `linked_node`, and migrate that district's entry in
`district_roots`/`roots_to_district` from `original_root` to the new root.
"""
function repair_partition!(
    partition::LinkCutPartition,
    linked_node::Node,
    original_root::Node
)
    new_root = find_root!(linked_node)
    district = partition.roots_to_district[original_root.vertex]
    delete!(partition.roots_to_district, original_root.vertex)
    partition.roots_to_district[new_root.vertex] = district
    partition.district_roots[district] = new_root.vertex 
end

"""
    getCummulativePathWeight(partition, path, link)

Return `(pathWeights, cumWeight)` for the cycle formed by adding `link` across the
tree `path`. `pathWeights` is the running cumulative sum of inverse edge weights
along `path` (starting at 0), and `cumWeight` additionally includes the inverse
weight of `link` — i.e. the total over all edges of the cycle. The cumulative
profile lets a cut edge be sampled proportional to inverse weight.
"""
function getCummulativePathWeight(
    partition::LinkCutPartition,
    path::Vector{<:Node},
    link::Tuple{Node, Node}
)
    pathWeights = Float64[0]
    g = partition.graph.simple_graph
    cumWeight = 0
    for n in 2:lastindex(path)
        w = g.weights[path[n-1].vertex,path[n].vertex]
        cumWeight += (1/w)
        append!(pathWeights,[cumWeight])
    end
    cumWeight += 1/g.weights[link[1].vertex, link[2].vertex]
    return pathWeights, cumWeight
end

"""
    internal_forest_walk!(partition, rng; diagnostics=nothing, edge=nothing)

Perform one 1-tree (internal forest) cycle-walk proposal *in place*. Add a random
internal edge `edge` (sampled if not given) to its district tree, forming a cycle,
then cut one cycle edge sampled proportional to inverse weight, yielding a new
spanning tree of the same district. Because the move never changes the node→district
assignment it is always accepted, so the partition is mutated directly and the
function returns `(0, nothing)` (no Metropolis correction needed). `edge` may be
supplied to force a specific move (used in testing).
"""
function internal_forest_walk!(
    partition::LinkCutPartition,
    rng::AbstractRNG;
    diagnostics::Union{Nothing,ProposalDiagnostics}=nothing,
    edge::Union{Tuple{Int64, Int64}, Nothing}=nothing
)
    if edge === nothing
        edge = get_rand_internal_edge(partition, rng)
    end

    u = partition.lct.nodes[edge[1]]
    v = partition.lct.nodes[edge[2]]
    r = find_root!(u)

    evert!(u)
    expose!(v)
    path = findPath(v)
    if length(path) == 2
        repair_partition!(partition, v, r)
        return 0, nothing
    end

    pathWeights, cumWeight = getCummulativePathWeight(partition, path, (u,v))

    # edge_ind = rand(rng, 1:length(D))
    randSamp = rand(rng)*cumWeight
    if randSamp > pathWeights[end]
        repair_partition!(partition, v, r)
        return 0, nothing
    end

    for edge_ind in 1:lastindex(pathWeights)
        if ((randSamp > pathWeights[edge_ind]) && 
            (randSamp <= pathWeights[edge_ind+1]))
            cut!(path[edge_ind+1])
            new_root = find_root!(v)
            link!(u,v)
            district = partition.roots_to_district[r.vertex]
            delete!(partition.roots_to_district, r.vertex)
            partition.roots_to_district[new_root.vertex] = district
            partition.district_roots[district] = new_root.vertex
            return 0, nothing
        end
    end
    println("Error: escaped edge detection")
    @assert false
end

"""
    build_internal_forest_walk(constraints)

Return a 1-tree cycle-walk proposal closure `f(partition, rng; diagnostics=nothing)`
that calls [`internal_forest_walk!`](@ref). Exported under the alias
`build_one_tree_cycle_walk`. `constraints` is accepted for a uniform proposal
interface but is unused, since an internal-forest move keeps every district's node
set fixed and therefore cannot violate constraints.
"""
function build_internal_forest_walk(
    constraints::Constraints
)
    f(p, r; diagnostics=nothing) = internal_forest_walk!(p, r;
                                                        diagnostics=diagnostics)
    return name_proposal!(f, "one_tree_cycle_walk")
end

const build_one_tree_cycle_walk = build_internal_forest_walk
const one_tree_cycle_walk! = internal_forest_walk!
