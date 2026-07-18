
# Tests for exported observables, energies, constraint checks, and run diagnostics
# that were not previously exercised by the test suite.
#
# Relies on globals defined in runtests.jl: `testdir`, `small_square_json`,
# `small_square_graph` (a 4x4-precinct / 2x2-county MultiLevelGraph, total pop 16).

# Build a fresh base graph with synthetic vote/election columns so the partisan and
# VRA observables can be exercised without mutating the shared test graphs.
function build_voting_graph()
    node_data = Set(["county", "pct", "pop", "area", "border_length"])
    bg = BaseGraph(small_square_json, "pop";
                   inc_node_data=node_data, area_col="area",
                   node_border_col="border_length", edge_perimeter_col="length")
    for ni in 1:bg.num_nodes
        bg.node_attributes[ni]["dem"]    = float(ni % 5)
        bg.node_attributes[ni]["rep"]    = float((ni * 2) % 7) + 1.0
        bg.node_attributes[ni]["totpop"] = 16.0
        bg.node_attributes[ni]["minpop"] = 4.0
    end
    return bg, MultiLevelGraph(bg, ["pct"])
end

function fresh_partition(seed)
    rng = PCG.PCGStateOneseq(UInt64, seed)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    return partition, constraints, rng
end


@testset "get_diameters" begin
    partition, _, _ = fresh_partition(10101)
    diameters = get_diameters(partition)
    @test diameters isa AbstractVector
    @test length(diameters) == partition.num_dists
    # Each district has 4 nodes, so its spanning-tree diameter is between 1 and 3 edges.
    @test all(1 .<= diameters .<= 3)
end

@testset "get_neighbor_lists" begin
    partition, _, _ = fresh_partition(20202)
    nbr_lists = get_neighbor_lists(partition)
    @test length(nbr_lists) == partition.num_dists
    # Every district tree has 4 vertices; the four trees together cover all 16 nodes.
    @test all(length(nl) == 4 for nl in nbr_lists)
    @test sum(length(nl) for nl in nbr_lists) == 16
    # A neighbor list is symmetric: if v lists u then u lists v.
    for nl in nbr_lists
        for (v, nbrs) in nl
            for u in nbrs
                @test v in nl[u]
            end
        end
    end
end

@testset "get_log_energy matches weighted sum of energies" begin
    partition, _, _ = fresh_partition(30303)
    measure = Measure()
    push_energy!(measure, get_log_spanning_forests, 2.0)
    push_energy!(measure, get_isoperimetric_score, 0.5)

    expected = 2.0 * get_log_spanning_forests(partition) +
               0.5 * get_isoperimetric_score(partition)
    result = get_log_energy(partition, measure)
    @test result isa Float64
    @test isfinite(result)
    @test result ≈ expected
end

@testset "satisfies_constraint / satisfies_constraints" begin
    partition, constraints, _ = fresh_partition(40404)

    # The seed partition is perfectly balanced (each district has population 4).
    @test satisfies_constraints(partition, constraints; check_population=true)
    @test satisfies_constraint(partition, PopulationConstraint(4, 4))

    # Bounds that exclude the actual district populations are not satisfied.
    @test !satisfies_constraint(partition, PopulationConstraint(5, 6))
    @test !satisfies_constraint(partition, PopulationConstraint(1, 3))
end

@testset "partisan observables (margins and seats)" begin
    _, voting_graph = build_voting_graph()
    rng = PCG.PCGStateOneseq(UInt64, 50505)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(voting_graph, constraints, 4; rng=rng)

    get_margins = build_get_partisan_margins("dem", "rep")
    get_seats   = build_get_partisan_seats("dem", "rep")

    margins = get_margins(partition)
    seats   = get_seats(partition)

    @test margins isa Vector{Float64}
    @test length(margins) == partition.num_dists
    @test all(0.0 .<= margins .<= 100.0)        # margins are dem vote-share percentages

    @test 0.0 <= seats <= partition.num_dists
    # Seats = number of districts where dem share strictly exceeds 50%.
    @test seats == count(>(50.0), margins)

    # --- partition-free path -------------------------------------------------
    # The builders return callables that subtype Function (so push_writer!/Writer
    # still accept them) and also expose f(node_to_dist, ::BaseGraph, num_dists).
    @test get_margins isa Function
    @test get_seats isa Function
    n2d, base, nd = partition.node_to_dist, partition.graph, partition.num_dists
    @test hasmethod(get_margins, Tuple{Vector{Int}, typeof(base), Int})
    @test hasmethod(get_seats, Tuple{Vector{Int}, typeof(base), Int})

    # Same tally as the LinkCutPartition path (same node numbering, no spanning tree).
    @test CycleWalk.get_partisan_margins(n2d, base, nd, "dem", "rep") == margins
    @test CycleWalk.get_partisan_seats(n2d, base, nd, "dem", "rep") == seats
    @test get_margins(n2d, base, nd) == margins       # via the functor
    @test get_seats(n2d, base, nd) == seats
end

@testset "performant VRA score and report" begin
    voting_base, voting_graph = build_voting_graph()
    rng = PCG.PCGStateOneseq(UInt64, 60606)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(voting_graph, constraints, 4; rng=rng)

    # One election with two (identical, for the test) stages; preferred candidate first.
    elections = [(("dem", "rep"), ("dem", "rep"))]

    vra_score_fn = build_performant_vra_score(voting_base, elections;
                                              target_districts=1)
    score = vra_score_fn(partition)
    @test score isa Float64
    @test isfinite(score)
    @test score >= 0.0

    vra_report_fn = build_performant_vra_report(voting_base, elections;
                                                target_districts=1)
    report = vra_report_fn(partition)
    @test report isa AbstractVector
    @test report[1] == 1        # first entry echoes the target-district count
end

@testset "run diagnostics accumulate one value per step" begin
    partition, constraints, rng = fresh_partition(70707)
    cycle_walk = build_lifted_tree_cycle_walk(constraints)

    run_diagnostics = RunDiagnostics()
    diags = (AcceptanceRatios(), CycleLengthDiagnostic(), DeltaNodesDiagnostic(),
             DeltaPopDiagnostic(), CuttableEdgePairsDiagnostic(),
             UniqueCuttableEdgesDiagnostic(), MaxSwappablePopulationDiagnostic(),
             AvgSwappablePopulationDiagnostic())
    for d in diags
        push_diagnostic!(run_diagnostics, cycle_walk, d)
    end

    nsteps = 250
    # output_freq larger than nsteps so diagnostics are never reset mid-run.
    run_metropolis_hastings!(partition, cycle_walk, Measure(), nsteps, rng;
                             run_diagnostics=run_diagnostics, output_freq=10_000)

    gathered = run_diagnostics[cycle_walk][2]
    @test length(gathered) == length(diags)
    for d in values(gathered)
        @test length(d.data_vec) == nsteps
        @test all(isfinite, d.data_vec)
    end

    # Cycle-length diagnostic records non-negative integer cycle lengths.
    cyc = gathered[CycleLengthDiagnostic]
    @test all(cyc.data_vec .>= 0)
    # Swappable-population fractions lie in [0, 1].
    avg_swap = gathered[AvgSwappablePopulationDiagnostic]
    @test all(0.0 .<= avg_swap.data_vec .<= 1.0)
end

# --- Known-broken exported API ----------------------------------------------
# These exported functions currently throw `UndefVarError` because they reference
# an undefined `LiftedTreeWalk` module (the geometry observables) or an undefined
# `operate_on_attribute` (the VRA target helper). The `@test_broken` markers
# document the breakage and will flag (as "Unexpected Pass") once it is fixed.
@testset "known-broken observables (documented)" begin
    partition, _, _ = fresh_partition(80808)
    @test_broken get_degree_distributions(partition) isa AbstractVector
    @test_broken get_average_degrees(partition) isa AbstractVector
    @test_broken get_center_moments(partition) isa AbstractVector
    @test_broken get_center_leaves_moments(partition) isa AbstractVector

    voting_base, _ = build_voting_graph()
    @test_broken get_target_vra_districts(voting_base, 4, "totpop", "minpop") isa Integer
end
