"""
    VotingData <: AbstractEnergyData

Cache backing the partisan observables for one pair of vote columns. Stores each
district's two-party vote totals (`dem_votes`, `rep_votes`) and the first party's
vote-share percentage (`dem_margins`), plus matching `_update` scratch buffers (with
`-1` in `dem_margins_update` marking "not yet computed"). The "dem"/"rep" naming is
just first-/second-column convention.
"""
mutable struct VotingData <: AbstractEnergyData
    identifier::Int64
    dem_votes::Vector{Float64}
    rep_votes::Vector{Float64}
    dem_margins::Vector{Float64}
    dem_votes_update::Vector{Float64}
    rep_votes_update::Vector{Float64}
    dem_margins_update::Vector{Float64}
end

"""
    VotingData(partition)

Allocate an empty [`VotingData`](@ref) sized to `partition`'s districts, with the
identifier one behind the partition's so vote totals are computed on first use.
"""
function VotingData(partition::LinkCutPartition)
    identifier = partition.identifier - 1
    dem_votes = Vector{Float64}(undef, partition.num_dists)
    rep_votes = Vector{Float64}(undef, partition.num_dists)
    dem_margins = Vector{Float64}(undef, partition.num_dists)
    dem_votes_update = Vector{Float64}(undef, partition.num_dists)
    rep_votes_update = Vector{Float64}(undef, partition.num_dists)
    dem_margins_update = Vector{Float64}(undef, partition.num_dists)
    dem_margins_update .= -1.0
    return VotingData(identifier, dem_votes, rep_votes, dem_margins,
                      dem_votes_update, rep_votes_update, dem_margins_update)
end

"""
    set_vote_data!(partition, votes1, votes2, vote_data, update=nothing)

Recompute each district's `votes1`/`votes2` totals and the first party's vote-share
percentage into `vote_data`. With `update === nothing` all districts are tallied for
the current assignment (and the cache identifier synced); with an `update` only the
changed districts are tallied into the `_update` buffers from the proposed assignment.
"""
function set_vote_data!(
    partition::LinkCutPartition,
    votes1::String,
    votes2::String,
    vote_data::VotingData,
    update::Union{Update{T}, Nothing}=nothing
) where T<:Int
    node_attributes = partition.graph.node_attributes
    num_dists = partition.num_dists
    num_nodes = partition.graph.num_nodes

    node_to_dist = partition.node_to_dist
    dem_votes = vote_data.dem_votes
    rep_votes = vote_data.rep_votes
    dem_margins = vote_data.dem_margins
    nodes = 1:num_nodes
    districts = 1:num_dists

    if update !== nothing
        node_to_dist = partition.node_to_dist_update
        dem_votes = vote_data.dem_votes_update
        rep_votes = vote_data.rep_votes_update
        dem_margins = vote_data.dem_margins_update
        nodes = [ii for ii = 1:num_nodes 
                 if node_to_dist[ii]∈update.changed_districts]
        districts = update.changed_districts
    end

    for di in districts
        dem_votes[di] = 0
        rep_votes[di] = 0
    end

    for ii in nodes
        dem_votes[node_to_dist[ii]] += node_attributes[ii][votes1]
        rep_votes[node_to_dist[ii]] += node_attributes[ii][votes2]
    end
    for di in districts
        dem_margins[di] = 100.0*dem_votes[di]/(dem_votes[di]+rep_votes[di])
    end

    if update === nothing
        vote_data.identifier = partition.identifier
    end
end

"""
    get_partisan_margins(partition, votes1, votes2, districts=...; update=nothing)::Vector{Float64}

Return each district's first-party vote share (as a percentage,
`100·votes1/(votes1+votes2)`) for the `votes1`/`votes2` columns, using the cached
[`VotingData`](@ref) (refreshed if stale, or extended for the changed districts when
an `update` is given).
"""
function get_partisan_margins(
    partition::LinkCutPartition,
    votes1::String,
    votes2::String,
    districts::Vector{Int} = collect(1:partition.num_dists);
    update::Union{Update{T}, Nothing}=nothing
)::Vector{Float64} where T <: Int
    margins = Vector{Float64}(undef, length(districts))

    if !haskey(partition.energy_data, (VotingData, (votes1, votes2)))
        partition.energy_data[(VotingData, (votes1, votes2))] = 
            VotingData(partition)
    end
    vote_data = partition.energy_data[(VotingData, (votes1, votes2))]

    if update === nothing
        if partition.identifier != vote_data.identifier
            set_vote_data!(partition, votes1, votes2, vote_data)
        end
        margins .= vote_data.dem_margins
    else
        margins .= vote_data.dem_margins
        set_vote_data!(partition, votes1, votes2, vote_data, update)
        for di in update.changed_districts
            margins[di] = vote_data.dem_margins_update[di]
        end
    end

    return margins
end

"""
    get_partisan_seats(partition, votes1, votes2, districts=...; update=nothing)::Float64

Return the number of districts the first party wins for the `votes1`/`votes2` columns
— the count of districts whose first-party vote share (see
[`get_partisan_margins`](@ref)) exceeds 50%.
"""
function get_partisan_seats(
    partition::LinkCutPartition,
    votes1::String,
    votes2::String,
    districts::Vector{Int} = collect(1:partition.num_dists);
    update::Union{Update{T}, Nothing}=nothing
)::Float64 where T <: Int
    leans = get_partisan_margins(partition, votes1, votes2, districts; 
                                 update=update)
    return length([1 for l in leans if l > 50.0])
end

"""
    get_partisan_margins(node_to_dist, graph::BaseGraph, num_dists, votes1, votes2)::Vector{Float64}

Partition-free per-district first-party vote share `100·votes1/(votes1+votes2)`,
computed straight from a node-to-district assignment and the base graph's vote
columns — no `LinkCutPartition`, and therefore no random spanning tree (the tally is
independent of it). Same quantity as the `LinkCutPartition` method; the result is
indexed by `node_to_dist`'s own district numbering. Intended for scoring a stored
districting after the fact (e.g. an atlas's maps); requires `graph` to carry the
`votes1`/`votes2` node attributes.
"""
function get_partisan_margins(
    node_to_dist::Vector{Int},
    graph::BaseGraph,
    num_dists::Int,
    votes1::String,
    votes2::String,
)::Vector{Float64}
    v1 = zeros(Float64, num_dists)
    v2 = zeros(Float64, num_dists)
    node_attributes = graph.node_attributes
    for ni = 1:graph.num_nodes
        di = node_to_dist[ni]
        v1[di] += node_attributes[ni][votes1]
        v2[di] += node_attributes[ni][votes2]
    end
    return Float64[100.0 * v1[di] / (v1[di] + v2[di]) for di in 1:num_dists]
end

"""
    get_partisan_seats(node_to_dist, graph::BaseGraph, num_dists, votes1, votes2)::Float64

Partition-free count of districts the first party wins — the number of districts whose
[`get_partisan_margins`](@ref) vote share exceeds 50%. No partition or spanning tree.
"""
function get_partisan_seats(
    node_to_dist::Vector{Int},
    graph::BaseGraph,
    num_dists::Int,
    votes1::String,
    votes2::String,
)::Float64
    margins = get_partisan_margins(node_to_dist, graph, num_dists, votes1, votes2)
    return Float64(count(>(50.0), margins))
end

# The `build_get_partisan_*` factories return callable structs (not closures) that
# subtype `Function` -- so they still satisfy `push_writer!`/`Writer`'s
# `get_data::Function` and behave as `f(partition)` as before -- while ALSO exposing
# the partition-free `f(node_to_dist, ::BaseGraph, num_dists)` method above. A caller
# scoring stored districtings can then discover the fast path by method existence
# (`hasmethod(f, (Vector{Int}, BaseGraph, Int))`) and skip the partition rebuild.

"""
    PartisanMargins(votes1, votes2) <: Function

Callable observable for [`get_partisan_margins`](@ref) on the captured vote columns.
Call it as `f(partition, districts; update)` (the writer path) or
`f(node_to_dist, graph::BaseGraph, num_dists)` (the partition-free path).
"""
struct PartisanMargins <: Function
    votes1::String
    votes2::String
end
(pm::PartisanMargins)(p::LinkCutPartition, d::Vector{Int}=collect(1:p.num_dists);
                      update=nothing) =
    get_partisan_margins(p, pm.votes1, pm.votes2, d; update=update)
(pm::PartisanMargins)(node_to_dist::Vector{Int}, graph::BaseGraph, num_dists::Int) =
    get_partisan_margins(node_to_dist, graph, num_dists, pm.votes1, pm.votes2)

"""
    PartisanSeats(votes1, votes2) <: Function

Callable observable for [`get_partisan_seats`](@ref); see [`PartisanMargins`](@ref).
"""
struct PartisanSeats <: Function
    votes1::String
    votes2::String
end
(ps::PartisanSeats)(p::LinkCutPartition, d::Vector{Int}=collect(1:p.num_dists);
                    update=nothing) =
    get_partisan_seats(p, ps.votes1, ps.votes2, d; update=update)
(ps::PartisanSeats)(node_to_dist::Vector{Int}, graph::BaseGraph, num_dists::Int) =
    get_partisan_seats(node_to_dist, graph, num_dists, ps.votes1, ps.votes2)

"""
    build_get_partisan_margins(votes1, votes2) -> PartisanMargins

Return an observable computing [`get_partisan_margins`](@ref) for the captured vote
columns. Register it with a `Writer` to record per-district vote shares each output
step; it also serves the partition-free path (see [`PartisanMargins`](@ref)).
"""
build_get_partisan_margins(votes1::String, votes2::String) =
    PartisanMargins(votes1, votes2)

"""
    build_get_partisan_seats(votes1, votes2) -> PartisanSeats

Return an observable computing [`get_partisan_seats`](@ref) for the captured vote
columns. Register it with a `Writer` to record the first party's seat count each
output step; it also serves the partition-free path (see [`PartisanSeats`](@ref)).
"""
build_get_partisan_seats(votes1::String, votes2::String) =
    PartisanSeats(votes1, votes2)


"""
    update_energy_data!(eData::VotingData, partition, update)

Commit the accepted `update` into the voting cache: copy the proposed vote totals and
margins for the changed districts into the synced fields and reset their scratch
slots. If any proposed margin was never computed, the scratch buffer is fully
invalidated so values are recomputed on next query.
"""
function update_energy_data!(
    eData::VotingData,
    partition::LinkCutPartition,
    update::Update{T}
) where {T<:Int}
    for di in update.changed_districts
        if eData.dem_margins_update[di] == -1
            eData.dem_margins_update .= -1.0
            return
        end
    end
    eData.identifier = partition.identifier
    for di in update.changed_districts
        eData.dem_votes[di] = eData.dem_votes_update[di]
        eData.rep_votes[di] = eData.rep_votes_update[di]
        eData.dem_margins[di] = eData.dem_margins_update[di]
        eData.dem_margins_update[di] = -1.0
    end
end

