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
        st_idx = Int(ceil((cur_pop-1)/max_pop))
        end_idx = k - Int(ceil((total_pop - (cur_pop+1))/max_pop))
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
    sample_inverse_weight_cycle_edge(partition, path, candidate_cut_edges, edge, rng)

Sample one index from `candidate_cut_edges` with probability proportional to
1/weight of the corresponding cycle edge. Cycle edge `i` joins `path[i]` and
`path[i+1]`; the last cycle edge is the closing `edge`. Returns
`(cut_ind, p)` where `p` is the probability of the sampled index, or
`(0, 0.0)` if there are no candidates.
"""
function sample_inverse_weight_cycle_edge(
    partition::LinkCutPartition,
    path::AbstractVector{<:Node},
    candidate_cut_edges::Vector{Int},
    edge::Tuple{Int, Int},
    rng::AbstractRNG,
)::Tuple{Int, Float64}
    isempty(candidate_cut_edges) && return 0, 0.0
    m = length(path)
    wmx = weights(partition.graph.simple_graph)
    inv_w = Vector{Float64}(undef, length(candidate_cut_edges))
    for (j, i) in enumerate(candidate_cut_edges)
        u, v = i < m ? (path[i].vertex, path[i+1].vertex) : edge
        inv_w[j] = 1.0 / wmx[u, v]
    end
    total = sum(inv_w)
    rand_samp = rand(rng) * total
    running = 0.0
    for j in eachindex(inv_w)
        running += inv_w[j]
        if rand_samp <= running
            return candidate_cut_edges[j], inv_w[j] / total
        end
    end
    # Floating-point guard; loop above should always return.
    return candidate_cut_edges[end], inv_w[end] / total
end

"""
    path_boundary_reach(S, from, to, steps, min_pop, max_pop, forward)

Boundaries `p` (a cut after node `p`, `0 <= p <= m`) reachable from boundary
`from` (forward) or `to` (backward) using exactly `steps` contiguous segments,
each with population in `[min_pop, max_pop]`. `S[p+1]` is the population of
nodes `1..p`. Returns a `BitVector` indexed by `p+1`.

Forward: `reach[p+1]` means nodes `from+1..p` split into `steps` districts.
Backward: `reach[p+1]` means nodes `p+1..to` split into `steps` districts.
"""
function path_boundary_reach(
    S::Vector{Float64},
    from::Int,
    to::Int,
    steps::Int,
    min_pop::Float64,
    max_pop::Float64,
    forward::Bool,
)::BitVector
    reach = falses(length(S))
    reach[(forward ? from : to) + 1] = true
    for _ in 1:steps
        nxt = falses(length(S))
        for p in from:to
            reach[p+1] || continue
            if forward
                for q in (p+1):to
                    pop = S[q+1] - S[p+1]
                    pop > max_pop && break
                    pop >= min_pop && (nxt[q+1] = true)
                end
            else
                for q in (p-1):-1:from
                    pop = S[p+1] - S[q+1]
                    pop > max_pop && break
                    pop >= min_pop && (nxt[q+1] = true)
                end
            end
        end
        reach = nxt
    end
    return reach
end

"""
    peel_cycle_districts(node_pops, edge_weights, labels, k, min_pop, max_pop,
                         rng; forced_cuts=nothing)

Split the path of `m = length(node_pops)` collapsed nodes into `k` districts by
sequentially peeling one district off the end whose node has the smaller entry
of `labels`. `edge_weights[c]` is the weight of the edge between nodes `c` and
`c+1`. At each peel the candidate cuts are exactly those giving a peeled
district with population in `[min_pop, max_pop]` whose remainder can still be
split into the remaining districts (an exact 1D feasibility check). One is
chosen with probability proportional to 1/weight.

Returns `(cuts, log_prob)` with `cuts` the sorted cut positions (`c` means the
edge between nodes `c` and `c+1`) and `log_prob` the log-probability of those
choices given `labels`, or `nothing` if no valid peeling exists. With
`forced_cuts` (sorted) no randomness is used: the given cuts are replayed and
their log-probability returned (`nothing` if they are not valid choices), which
gives the reverse-move probability.
"""
function peel_cycle_districts(
    node_pops::AbstractVector{<:Real},
    edge_weights::AbstractVector{<:Real},
    labels::AbstractVector{<:Real},
    k::Int,
    min_pop::Float64,
    max_pop::Float64,
    rng::AbstractRNG;
    forced_cuts::Union{Nothing, AbstractVector{Int}}=nothing,
)::Union{Nothing, Tuple{Vector{Int}, Float64}}
    m = length(node_pops)
    S = zeros(Float64, m + 1)
    for i in 1:m
        S[i+1] = S[i] + node_pops[i]
    end

    a, b, j = 1, m, k
    f_lo, f_hi = 1, forced_cuts === nothing ? 0 : length(forced_cuts)
    cuts = Int[]
    log_prob = 0.0
    @show "here"
    while j > 1
        left = labels[a] < labels[b]
        cands = Int[]
        if left # peel nodes a..c; nodes c+1..b must split into j-1
            reach = path_boundary_reach(S, a-1, b, j-1, min_pop, max_pop, false)
            @show reach
            for c in a:(b-1)
                pop = S[c+1] - S[a]
                reach[c+1] && min_pop <= pop <= max_pop && push!(cands, c)
            end
        else # peel nodes c+1..b; nodes a..c must split into j-1
            reach = path_boundary_reach(S, a-1, b, j-1, min_pop, max_pop, true)
            for c in a:(b-1)
                pop = S[b+1] - S[c+1]
                reach[c+1] && min_pop <= pop <= max_pop && push!(cands, c)
            end
        end
        isempty(cands) && return nothing

        inv_w = [1.0 / edge_weights[c] for c in cands]
        total = sum(inv_w)
        if forced_cuts === nothing
            r = rand(rng) * total
            idx = findfirst(>=(r), cumsum(inv_w))
            idx === nothing && (idx = length(cands))
        else
            c_forced = left ? forced_cuts[f_lo] : forced_cuts[f_hi]
            idx = findfirst(==(c_forced), cands)
            idx === nothing && return nothing
        end
        c = cands[idx]
        log_prob += log(inv_w[idx] / total)
        push!(cuts, c)
        if left
            a = c + 1
            f_lo += 1
        else
            b = c
            f_hi -= 1
        end
        j -= 1
    end
    return sort!(cuts), log_prob
end

"""
    log_prob_cut_selection(node_pops, edge_weights, labels, boundary_edges,
                           candidate_edges, min_pop, max_pop, rng)

Log-probability that the cut-selection procedure (first cut sampled from
`candidate_edges` proportional to 1/weight, then [`peel_cycle_districts`](@ref))
produces the district boundaries `boundary_edges` on a cycle of `m` nodes, given
the node `labels`.

All inputs are in cycle indexing: `node_pops[i]` and `labels[i]` belong to cycle
node `i`, and `edge_weights[i]` is the weight of the edge between nodes `i` and
`i+1` (edge `m` closes the cycle). `boundary_edges` are the `k` cycle edges that
separate the districts.

Any boundary can be the first cut, so the result is the log of the sum over the
`k` boundaries `e` of `P(first = e) * P(peel gives the other k-1 | first = e)`,
with the cycle rotated to start after `e` and `labels` carried along with the
nodes. Boundaries that are not candidates (or whose forced peeling is invalid)
contribute zero. Returns `-Inf` if every term is zero.

Use it on the new boundaries for the forward probability and on the old ones for
the reverse probability; the candidates depend only on the cycle, so both
directions share them.
"""
function log_prob_cut_selection(
    node_pops::AbstractVector{<:Real},
    edge_weights::AbstractVector{<:Real},
    labels::AbstractVector{<:Real},
    boundary_edges::AbstractVector{Int},
    candidate_edges::AbstractVector{Int},
    min_pop::Float64,
    max_pop::Float64,
    rng::AbstractRNG,
)::Float64
    m = length(node_pops)
    k = length(boundary_edges)
    total_inv_w = sum(1.0 / edge_weights[i] for i in candidate_edges; init=0.0)
    terms = Float64[]
    for e in boundary_edges
        e in candidate_edges || continue
        rot_pops = [node_pops[mod1(e + r, m)] for r in 1:m]
        rot_edge_w = [edge_weights[mod1(e + c, m)] for c in 1:(m-1)]
        rot_labels = [labels[mod1(e + r, m)] for r in 1:m]
        forced = sort!([mod1(b - e, m) for b in boundary_edges if b != e])
        peeled = peel_cycle_districts(rot_pops, rot_edge_w, rot_labels, k,
                                      min_pop, max_pop, rng;
                                      forced_cuts=forced)
        peeled === nothing && continue
        push!(terms, log((1.0 / edge_weights[e]) / total_inv_w) + peeled[2])
    end
    isempty(terms) && return -Inf
    top = maximum(terms)
    return top + log(sum(exp(t - top) for t in terms))
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

    qg, _ = build_district_quotient_graph(partition)
    quotient_tree = wilson_rst(qg, rng, weights(qg))
    
    specified_cross_edges = Dict{Edge{Int64}, Tuple{Int64, Int64}}()
    edge, (d1, d2) = get_valid_cross_edge!(specified_cross_edges, partition, 
                                           quotient_tree, edge, (d1, d2), rng)
    d1 == d2 && return internal_forest_walk!(partition, rng;
                                             diagnostics=diagnostics, edge=edge)

    # log_weighted_qtree_count = log_nspanning(qg)

#####
    ## TODO: store the quotient_tree as an lct, then find the impacted districts
    ##       and only specify/link those edges (don't do the whole thing)
    quotient_lct = build_quotient_lct(quotient_tree; safeflag=false)
    quotient_path = get_cycle_path_for_edge!(quotient_lct, (d1, d2))
    k = length(quotient_path)
    specify_quotient_path!(partition, specified_cross_edges, quotient_path, rng)
    linked_cross_edges = add_specified_edges!(partition, specified_cross_edges,
                                              quotient_path)
    
    path = get_cycle_path_for_edge!(partition, edge)
    cut_remainder = topological_sort(path[1], partition)
    cycle_weights = get_collapsed_cycle_weights_path(path, cut_remainder)
    
    candidate_cut_edges = find_candidate_cycle_cut_edges(cycle_weights, k,
                                                         constraints)

    cut_ind, p_cut = sample_inverse_weight_cycle_edge(partition, path,
                                                      candidate_cut_edges, 
                                                      edge, rng)
    if cut_ind == length(cycle_weights)
        cut_cross_district_edges!(partition, quotient_path, 
                                  specified_cross_edges)
        return 0, nothing
    end

    # Rotate the cycle so it starts just after the cut edge: rotated node r is
    # cycle node mod1(cut_ind + r, m); rotated edge c is cycle edge
    # mod1(cut_ind + c, m).
    m = length(path)
    wmx = weights(partition.graph.simple_graph)
    cycle_edge_weight(i) = i < m ? wmx[path[i].vertex, path[i+1].vertex] :
                                   wmx[edge[1], edge[2]]
    rot_pops = [cycle_weights[mod1(cut_ind + r, m)] for r in 1:m]
    rot_edge_w = [cycle_edge_weight(mod1(cut_ind + c, m)) for c in 1:(m-1)]
    labels = sortperm(rand(rng, m)) # random permutation of node labels
    peeled = peel_cycle_districts(rot_pops, rot_edge_w, labels, k,
                                  Float64(constraints.population_constraint.min_pop),
                                  Float64(constraints.population_constraint.max_pop),
                                  rng)
    if peeled === nothing
        cut_cross_district_edges!(partition, quotient_path, specified_cross_edges)
        return 0, nothing
    end
    cuts, log_p_peel = peeled

    # Forward and reverse cut-selection probabilities, conditioned on the same
    # labels (carried with the cycle nodes). New boundaries: the first cut plus
    # the peeled cuts; old boundaries: the closing edge and the path edges that
    # currently separate districts.
    min_pop = Float64(constraints.population_constraint.min_pop)
    max_pop = Float64(constraints.population_constraint.max_pop)
    cycle_edge_ws = [cycle_edge_weight(i) for i in 1:m]
    cycle_labels = similar(labels)
    for r in 1:m
        cycle_labels[mod1(cut_ind + r, m)] = labels[r]
    end
    new_boundaries = vcat(cut_ind, [mod1(cut_ind + c, m) for c in cuts])
    old_boundaries = [i for i in 1:(m-1) if partition.node_to_dist[path[i].vertex] !=
                                            partition.node_to_dist[path[i+1].vertex]]
    push!(old_boundaries, m)
    log_q_forward = log_prob_cut_selection(cycle_weights, cycle_edge_ws,
                                           cycle_labels, new_boundaries,
                                           candidate_cut_edges, min_pop, max_pop,
                                           rng)
    log_q_reverse = log_prob_cut_selection(cycle_weights, cycle_edge_ws,
                                           cycle_labels, old_boundaries,
                                           candidate_cut_edges, min_pop, max_pop,
                                           rng)
    # TODO:
        # - pick a plausible edge with probability proportional to 1/edge_weight
        # - check if the tree is cuttable
        # - use the random partition to compute forward and backward probabilities

    # unlink_sampled_cross_edges!(partition, linked_cross_edges)

    # This is the per-edge probability under the sampling model.
    num_non_tree_edges = ne(partition.graph.simple_graph) -
                         (nv(partition.graph.simple_graph) - 1)
    if num_non_tree_edges <= 0
        return 0, nothing
    end
    p_edge = 1.0 / num_non_tree_edges

    # Placeholder values to keep scaffold outputs available during development.
    _ = candidate_cut_edges
    _ = cuts
    _ = log_p_peel
    _ = p_cut
    _ = log_q_forward
    _ = log_q_reverse
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
