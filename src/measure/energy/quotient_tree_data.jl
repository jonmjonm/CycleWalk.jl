"""
    QuotientTreeData <: AbstractEnergyData

Cache of the marked tree `M` of the BUD walk: a spanning tree of the district
quotient graph in which every quotient edge is realized by one concrete cross-district
edge of the base graph. Together with the forest `F` it makes the spanning tree
`F ∪ M` of the lifted measure `w(F) w(M) / τ(Q)`, so `M` is part of the lifted state
and persists between steps.

Edges live in slots `1:num_dists-1`. Slot `s` holds the concrete edge `edges[s]`
(ordered vertex pair), its weight, and the district pair `ends[s]` it currently joins;
`ends` is derived from `node_to_dist` and refreshed whenever labels change. `slot_of`
maps an edge to its slot and `adj[d]` lists the slots at district `d`. `identifier` is
the partition identifier the tree was last synced to; a mismatch means the tree may no
longer match the partition and is redrawn (see [`refresh_quotient_tree!`](@ref)).
"""
mutable struct QuotientTreeData <: AbstractEnergyData
    identifier::Int64
    edges::Vector{Tuple{Int,Int}}
    edge_weights::Vector{Float64}
    ends::Vector{Tuple{Int,Int}}
    slot_of::Dict{Tuple{Int,Int},Int}
    adj::Vector{Vector{Int}}
end

"""
    QuotientTreeData(partition)

Allocate an empty [`QuotientTreeData`](@ref) for `partition`. The stored identifier is
one behind the partition's so the tree is drawn on first use.
"""
function QuotientTreeData(partition::LinkCutPartition)
    n = partition.num_dists
    return QuotientTreeData(partition.identifier - 1, Tuple{Int,Int}[],
                            Float64[], Tuple{Int,Int}[],
                            Dict{Tuple{Int,Int},Int}(), [Int[] for _ in 1:n])
end

"""
    QuotientPathVertex(vertex)

Minimal stand-in for a quotient-tree `Node`: the functions that link, cut and read
the quotient path only use the `vertex` field.
"""
struct QuotientPathVertex
    vertex::Int
end

"""
    detach_slot!(data, s)

Remove slot `s`'s edge from the lookup and adjacency structures (the slot's arrays are
left to be overwritten by [`attach_slot!`](@ref)).
"""
function detach_slot!(data::QuotientTreeData, s::Int)
    a, b = data.ends[s]
    delete!(data.slot_of, data.edges[s])
    deleteat!(data.adj[a], findfirst(==(s), data.adj[a]))
    deleteat!(data.adj[b], findfirst(==(s), data.adj[b]))
    return nothing
end

"""
    attach_slot!(data, s, key, w, node_to_dist)

Put the cross edge `key` (weight `w`) in slot `s`, appending the slot if `s` is one
past the last, and record the districts it joins under `node_to_dist`.
"""
function attach_slot!(
    data::QuotientTreeData,
    s::Int,
    key::Tuple{Int,Int},
    w::Float64,
    node_to_dist::Vector{Int},
)
    a = node_to_dist[key[1]]
    b = node_to_dist[key[2]]
    if s > length(data.edges)
        push!(data.edges, key)
        push!(data.edge_weights, w)
        push!(data.ends, (a, b))
    else
        data.edges[s] = key
        data.edge_weights[s] = w
        data.ends[s] = (a, b)
    end
    data.slot_of[key] = s
    push!(data.adj[a], s)
    push!(data.adj[b], s)
    return nothing
end

"""
    refresh_quotient_ends!(data, node_to_dist)

Recompute the district pair of every slot from `node_to_dist` and repair the
adjacency lists where it changed. `O(num_dists)`.
"""
function refresh_quotient_ends!(data::QuotientTreeData, node_to_dist::Vector{Int})
    for s in eachindex(data.edges)
        u, v = data.edges[s]
        new_ends = (node_to_dist[u], node_to_dist[v])
        new_ends == data.ends[s] && continue
        oa, ob = data.ends[s]
        deleteat!(data.adj[oa], findfirst(==(s), data.adj[oa]))
        deleteat!(data.adj[ob], findfirst(==(s), data.adj[ob]))
        data.ends[s] = new_ends
        push!(data.adj[new_ends[1]], s)
        push!(data.adj[new_ends[2]], s)
    end
    return nothing
end

"""
    rebuild_quotient_tree!(data, partition, rng)

Draw `M` from its conditional law given the forest: a Wilson spanning tree of the
weighted quotient graph, with one concrete edge per quotient edge drawn in proportion
to its weight. This is the exact distribution `w(M) / τ(Q)`.
"""
function rebuild_quotient_tree!(
    data::QuotientTreeData,
    partition::LinkCutPartition,
    rng::AbstractRNG,
)
    empty!(data.edges)
    empty!(data.edge_weights)
    empty!(data.ends)
    empty!(data.slot_of)
    for a in data.adj
        empty!(a)
    end

    qg = get_district_quotient_graph(partition)
    tree = wilson_rst(qg, rng, weights(qg))
    for (s, e) in enumerate(tree)
        a, b = src(e), dst(e)
        pair = haskey(partition.cross_district_edges, (a, b)) ? (a, b) : (b, a)
        cross_edge = sample_weighted_edge(partition.cross_district_edges[pair], rng)
        attach_slot!(data, s, ordered_edge_key(cross_edge),
                     Float64(weight(cross_edge)), partition.node_to_dist)
    end
    data.identifier = partition.identifier
    return nothing
end

"""
    quotient_tree_path(data, a, b)

Districts and slots along the path from district `a` to district `b` in the tree:
`(districts, slots)` with `length(districts) == length(slots) + 1`.
"""
function quotient_tree_path(data::QuotientTreeData, a::Int, b::Int)
    n = length(data.adj)
    parent_vert = zeros(Int, n)
    parent_slot = zeros(Int, n)
    parent_vert[a] = a
    stack = Int[a]
    while !isempty(stack)
        x = pop!(stack)
        x == b && break
        for s in data.adj[x]
            ea, eb = data.ends[s]
            y = ea == x ? eb : ea
            parent_vert[y] != 0 && continue
            parent_vert[y] = x
            parent_slot[y] = s
            push!(stack, y)
        end
    end
    parent_vert[b] == 0 && error(
        "districts $a and $b are not connected in the quotient tree")
    districts = Int[]
    slots = Int[]
    y = b
    while y != a
        push!(districts, y)
        push!(slots, parent_slot[y])
        y = parent_vert[y]
    end
    push!(districts, a)
    return reverse!(districts), reverse!(slots)
end

"""
    quotient_updown_step!(data, partition, rng)

One heat-bath update of `M`, reversible with respect to `w(M) / τ(Q)` given the
forest. Draw a cross edge `e_in` uniformly from all cross-district edges (a pair in
proportion to its edge count, then an edge uniformly within it). If it is already in
`M` nothing happens. Otherwise it closes a cycle with the `M` path between its
districts, and one edge of that cycle is dropped with probability proportional to
`1/weight`, `e_in` itself included (which leaves `M` unchanged). Returns whether `M`
changed.
"""
function quotient_updown_step!(
    data::QuotientTreeData,
    partition::LinkCutPartition,
    rng::AbstractRNG,
)::Bool
    cross = partition.cross_district_edges
    total = 0
    for es in values(cross)
        total += length(es)
    end
    total == 0 && return false

    r = rand(rng, 1:total)
    e_in = nothing
    for es in values(cross)
        if r <= length(es)
            for (i, e) in enumerate(es)
                if i == r
                    e_in = e
                    break
                end
            end
            break
        end
        r -= length(es)
    end

    key = ordered_edge_key(e_in)
    haskey(data.slot_of, key) && return false
    w_in = Float64(weight(e_in))
    node_to_dist = partition.node_to_dist
    _, slots = quotient_tree_path(data, node_to_dist[key[1]], node_to_dist[key[2]])

    z = 1 / w_in
    for s in slots
        z += 1 / data.edge_weights[s]
    end
    x = rand(rng) * z - 1 / w_in
    x <= 0 && return false
    chosen = slots[end]  # floating-point guard
    for s in slots
        x -= 1 / data.edge_weights[s]
        if x <= 0
            chosen = s
            break
        end
    end
    detach_slot!(data, chosen)
    attach_slot!(data, chosen, key, w_in, node_to_dist)
    return true
end

"""
    refresh_quotient_tree!(partition, rng; wilson_prob=0.0, updown_steps=0)

Return the partition's [`QuotientTreeData`](@ref), refreshed for the next BUD step.
The tree is drawn by Wilson if it is missing or out of sync with the partition, or
with probability `wilson_prob`. Otherwise it takes `updown_steps` heat-bath steps
([`quotient_updown_step!`](@ref)). Either refresh leaves the lifted measure invariant.
"""
function refresh_quotient_tree!(
    partition::LinkCutPartition,
    rng::AbstractRNG;
    wilson_prob::Float64=0.0,
    updown_steps::Int=0,
)::QuotientTreeData
    data = get!(partition.energy_data, QuotientTreeData) do
        QuotientTreeData(partition)
    end
    if data.identifier != partition.identifier ||
       (wilson_prob > 0 && rand(rng) < wilson_prob)
        rebuild_quotient_tree!(data, partition, rng)
    else
        for _ in 1:updown_steps
            quotient_updown_step!(data, partition, rng)
        end
    end
    return data
end

"""
    update_energy_data!(eData::QuotientTreeData, partition, update)

Carry the marked tree across an accepted `update`. A BUD move (`update.marked_change`)
swaps its removed edges for its added ones and refreshes the district pairs the labels
moved. An update that changes a single district (an internal forest move) leaves `M`
valid. Anything else, or a cache that was not in sync before this accept, leaves the
identifier behind so the tree is redrawn on next use.
"""
function update_energy_data!(
    eData::QuotientTreeData,
    partition::LinkCutPartition,
    update::Update{T}
) where {T<:Int}
    eData.identifier == partition.identifier - 1 || return
    if update.marked_change !== nothing
        removed, added = update.marked_change
        slots = Int[get(eData.slot_of, key, 0) for key in removed]
        any(==(0), slots) && return
        for s in slots
            detach_slot!(eData, s)
        end
        for (s, key) in zip(slots, added)
            attach_slot!(eData, s, key, Float64(get_weight(
                partition.graph.simple_graph, key[1], key[2])),
                partition.node_to_dist)
        end
        refresh_quotient_ends!(eData, partition.node_to_dist)
        eData.identifier = partition.identifier
    elseif length(update.changed_districts) == 1
        eData.identifier = partition.identifier
    end
    return
end
