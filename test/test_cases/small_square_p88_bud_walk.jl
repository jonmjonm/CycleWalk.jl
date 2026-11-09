# Distribution tests for the BUD walk on the small square graph (4 districts,
# pop=8). Every plan (unlabeled partition into 4 connected districts of 4 nodes)
# is enumerated, weighted by spanning forests^(1-gamma) and the isoperimetric
# score, and the chain's plan frequencies are compared to that distribution.
#
# Uses `weight_small_square_base_graph` and `modify_edge_len_small_square_base_graph`
# from small_square_p88_weighted.jl and small_square_p88_polsby_popper.jl, so this
# file must be included after those.

using Graphs: neighbors, nv

"""
Canonical (first-occurrence) relabeling of a district assignment.
"""
function canonical_plan(assignment)
    relabel = Dict{Int, Int}()
    return Tuple(get!(relabel, d, length(relabel) + 1) for d in assignment)
end

"""
All unlabeled partitions of the small square graph into connected districts of
`dist_size` nodes, as canonical assignments.
"""
function enumerate_small_square_plans(dist_size::Int=4)
    g = small_square_base_graph.simple_graph
    n = nv(g)
    function connected(nodes)
        seen = Set([first(nodes)])
        stack = [first(nodes)]
        while !isempty(stack)
            u = pop!(stack)
            for v in neighbors(g, u)
                if v in nodes && !(v in seen)
                    push!(seen, v)
                    push!(stack, v)
                end
            end
        end
        return length(seen) == length(nodes)
    end
    plans = Set{Tuple{Vararg{Int}}}()
    assignment = zeros(Int, n)
    function recurse(label)
        first_free = findfirst(==(0), assignment)
        if first_free === nothing
            push!(plans, canonical_plan(assignment))
            return
        end
        free = [v for v in 1:n if assignment[v] == 0 && v != first_free]
        for rest in Iterators.product(ntuple(_ -> free, dist_size - 1)...)
            (issorted(rest) && allunique(rest)) || continue
            nodes = Set([first_free, rest...])
            connected(nodes) || continue
            for v in nodes
                assignment[v] = label
            end
            recurse(label + 1)
            for v in nodes
                assignment[v] = 0
            end
        end
    end
    recurse(1)
    return collect(plans)
end

"""
Exact plan probabilities `∝ forests^(1-gamma) * exp(-iso_weight * isoperimetric score)`.
"""
function exact_plan_distribution(
    graph::MultiLevelGraph,
    plans,
    gamma::Float64,
    iso_weight::Float64
)
    logw = Dict{Tuple{Vararg{Int}}, Float64}()
    for plan in plans
        node_to_district = Dict{Tuple{Vararg{String}}, Int}()
        for ii = 1:length(plan)
            node_to_district[graph.id_to_partitions[1][ii]] = plan[ii]
        end
        ml_partition = MultiLevelPartition(graph, node_to_district)
        rng = PCG.PCGStateOneseq(UInt64, 4052159124)
        partition = LinkCutPartition(ml_partition, rng)
        lw = (1.0 - gamma) * get_log_spanning_forests(partition)
        iso_weight != 0 && (lw -= iso_weight * get_isoperimetric_score(partition))
        logw[plan] = lw
    end
    mx = maximum(values(logw))
    total = sum(exp(v - mx) for v in values(logw))
    return Dict(p => exp(v - mx) / total for (p, v) in logw)
end

"""
Run `records` samples (every `thin` MH steps) of a `bud_frac` BUD / internal-walk
mixture and return the visited canonical plans' frequencies.
"""
function observed_plan_frequencies(
    graph::MultiLevelGraph,
    constraints::Constraints,
    measure::Measure;
    records::Int=300_000,
    thin::Int=4,
    seed::Int=1241909
)
    rng = PCG.PCGStateOneseq(UInt64, seed)
    partition = LinkCutPartition(graph, constraints, 4; rng=rng)
    proposal =  build_bud_walk(constraints)
    counts = Dict{Tuple{Vararg{Int}}, Int}()
    for _ in 1:records
        run_metropolis_hastings!(partition, proposal, measure, thin, rng)
        plan = canonical_plan(partition.node_to_dist)
        counts[plan] = get(counts, plan, 0) + 1
    end
    return Dict(p => c / records for (p, c) in counts)
end

function test_plan_distribution(graph, constraints, measure, gamma, iso_weight, plans)
    exact = exact_plan_distribution(graph, plans, gamma, iso_weight)
    observed = observed_plan_frequencies(graph, constraints, measure)

    # no plan outside the enumerated set
    @test all(haskey(exact, p) for p in keys(observed))

    # total variation distance over the whole space
    tv = 0.5 * sum(abs(get(observed, p, 0.0) - pr) for (p, pr) in exact)
    @test tv < 0.035

    # plans of equal probability (e.g. all 117 when gamma=1 on the unweighted
    # graph) are pooled so the heavy classes are tested to the usual tolerance
    class_exact = Dict{Float64, Float64}()
    class_observed = Dict{Float64, Float64}()
    for (p, pr) in exact
        c = round(pr, sigdigits=8)
        class_exact[c] = get(class_exact, c, 0.0) + pr
        class_observed[c] = get(class_observed, c, 0.0) + get(observed, p, 0.0)
    end
    @test all(is_close(class_observed[c], class_exact[c])
              for c in keys(class_exact) if class_exact[c] > 0.02)
end

name = "small square test graph, BUD walk plan distribution, 4 districts, pop=8"
@testset "$name" begin
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4,4))

    plans = enumerate_small_square_plans()
    @test length(plans) == 117

    weighted_graph = weight_small_square_base_graph()
    pp_graph = modify_edge_len_small_square_base_graph()
    iso_weight = 0.02

    @testset "unweighted, gamma=0" begin
        test_plan_distribution(small_square_graph, constraints, Measure(), 0.0,
                               0.0, plans)
    end
    @testset "unweighted, gamma=1" begin
        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, 1.0)
        test_plan_distribution(small_square_graph, constraints, measure, 1.0,
                               0.0, plans)
    end
    @testset "weighted, gamma=0" begin
        test_plan_distribution(weighted_graph, constraints, Measure(), 0.0, 0.0,
                               plans)
    end
    @testset "weighted, gamma=1" begin
        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, 1.0)
        test_plan_distribution(weighted_graph, constraints, measure, 1.0, 0.0,
                               plans)
    end
    @testset "polsby-popper, gamma=0" begin
        measure = Measure()
        push_energy!(measure, get_isoperimetric_score, iso_weight)
        test_plan_distribution(pp_graph, constraints, measure, 0.0, iso_weight,
                               plans)
    end
    @testset "polsby-popper, gamma=1" begin
        measure = Measure()
        push_energy!(measure, get_isoperimetric_score, iso_weight)
        push_energy!(measure, get_log_spanning_forests, 1.0)
        test_plan_distribution(pp_graph, constraints, measure, 1.0, iso_weight,
                               plans)
    end

    # the cut-edge histograms the lifted-walk tests use, driven by the BUD walk
    @testset "cut-edge histogram, unweighted" begin
        observed_cuts = get_observed_cut_edges(small_square_graph, constraints, 4,
                                               cycle_steps=200_000, walk=:bud)
        @test length(values(observed_cuts)) == 4
        ce_to_count = Dict(8=>256, 10=>(128+64+32), 11=>96, 12=>78)
        steps = sum(values(observed_cuts))
        @test all([is_close(v/steps, ce_to_count[k]/654)
                   for (k, v) in observed_cuts])
    end
end
