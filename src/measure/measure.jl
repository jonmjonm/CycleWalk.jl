"""
    Measure

The (unnormalized) target distribution as a weighted sum of score/energy functions.
A plan's log-density is `-∑ weights[score] · score(partition)` over `scores`;
`descriptions` holds a human-readable label per score for output metadata. Build one
with [`push_energy!`](@ref).
"""
mutable struct Measure
    weights::Dict{Function, Float64}
    scores::Set{Function}
    descriptions::Dict{Function, String}
    # TO DO: constraints::Dict{Type{T} where T<:AbstractConstraint, AbstractConstraint}
end

"""
    Measure()

Create an empty [`Measure`](@ref) with no scores. Add weighted energies with
[`push_energy!`](@ref).
"""
function Measure()
    scores = Set{Function}()
    weights = Dict{Function, Float64}()
    descriptions = Dict{Function, String}()
    return Measure(weights, scores, descriptions)
end

"""
    push_energy!(measure, score, weight; desc="", allow_zero=false)

Add the energy `score` to `measure` with the given `weight`; `desc` is an optional
label for output metadata (defaults to the function's name). `score` must have the
signature `score(partition, districts; update)`. A zero `weight` is ignored.

Pass `allow_zero=true` to keep a zero-weight score in the measure anyway. This is what
an annealing schedule needs: a run that ramps an energy's weight *down to* zero has a
zero target weight, and a score that never entered `measure.scores` can never be
annealed, since [`get_log_energy`](@ref) sums over `scores` (a weight written straight
into `measure.weights` for a score outside that set is silently ignored, and leaves
`weights` and `scores` out of sync). Keeping the score costs nothing per step —
`get_log_energy` skips zero weights when it evaluates — but it does mean the score
appears in the Atlas header's energy list with weight `0`.
"""
function push_energy!(
    measure::Measure,
    score::Function,
    weight::Real;
    desc::String="",
    allow_zero::Bool=false
)
    weight == 0 && !allow_zero && return
    @assert keys(measure.weights) == measure.scores
    @assert keys(measure.descriptions) == measure.scores
    push!(measure.scores, score)
    measure.weights[score] = weight
    if desc == ""
        desc = string(score)
    end
    measure.descriptions[score] = desc
end

"""
    get_delta_energy(partition, measure, update)

Return the Metropolis energy factor `exp(logE_current - logE_proposed)` for the move
`update`, evaluated only over the changed districts. Multiplying the proposal ratio
by this gives the acceptance probability. Values `> 1` favor acceptance.
"""
function get_delta_energy(
    partition::LinkCutPartition,
    measure::Measure,
    update::Update{T}
) where T <: Int
    score = 0
    changed_districts = update.changed_districts
    # @show get_log_energy(partition, measure, changed_districts)
    # @show get_log_energy(partition, measure, changed_districts, update)
    # @show changed_districts
    score += get_log_energy(partition, measure, changed_districts)
    score -= get_log_energy(partition, measure, changed_districts, update)
    # @show score, exp(score)
    return exp(score)
end

"""
    get_log_energy(partition, measure, districts=..., update=nothing)::Float64

Return the total weighted log-energy `∑ weights[e] · e(partition, districts; update)`
over every score `e` in `measure`. With an `update`, evaluates the proposed plan;
otherwise the current plan. Restricting `districts` (as `get_delta_energy` does)
evaluates only those districts' contributions.
"""
function get_log_energy(
    partition::LinkCutPartition,
    measure::Measure,
    districts::Union{Tuple{Vararg{T}}, Vector{T}}
        = collect(1:partition.num_dists),
    update::Union{Update{T}, Nothing}=nothing
)::Float64 where T <: Int
    score = 0.0
    for energy in measure.scores
        weight = measure.weights[energy]
        if weight == 0
            continue
        end
        # tmp = energy(partition, districts; update=update)
        # @show weight, tmp
        score += weight*energy(partition, districts; update=update)
    end
    return score
end

# ################################## TO DO: EVERYTHING BELOW SHOULD GO ELSEWHERE
"""
    get_cut_edge_count(partition::MultiLevelPartition)

Return the number of cut (cross-district) edges of a `MultiLevelPartition`, summing
the `"connections"` weight over boundary edges.
"""
function get_cut_edge_count(partition::MultiLevelPartition)
    return get_cut_edge_weights(partition, "connections")
end

"""
    get_cut_edge_perimeter(partition::MultiLevelPartition)

Return the total perimeter of the district boundaries of a `MultiLevelPartition`,
summing the edge-perimeter attribute over cut edges.
"""
function get_cut_edge_perimeter(partition::MultiLevelPartition)
    edge_perimeter_col = partition.graph.graphs_by_level[1].edge_perimeter_col
    return get_cut_edge_weights(partition, edge_perimeter_col)
end

"""
    get_cut_edge_sum(partition::LinkCutPartition; column="connections")

Return the total `column` weight of the cut (cross-district) edges of a
`LinkCutPartition` — by default a count of boundary edges. Suitable as a `Measure`
energy or a `Writer` observable.
"""
function get_cut_edge_sum(
    partition::LinkCutPartition;
    column::String="connections"
)
    graph = partition.graph
    total = 0
    for e in edges(graph.simple_graph)
        n1, n2 = src(e), dst(e)
        if partition.node_to_dist[n1] == partition.node_to_dist[n2]
            continue
        end
        total += graph.edge_attributes[Set([n1, n2])][column]
    end
    return total
end

