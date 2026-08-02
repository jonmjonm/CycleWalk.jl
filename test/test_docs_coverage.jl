
# Tests for APIs documented at quantifyinggerrymandering.pages.oit.duke.edu/codedoc/cycleWalk.html

@testset "PopulationConstraint convenience constructor" begin
    # 4×4 grid: total_pop=16, so ideal_pop for 4 districts = 4
    # Uses formula: min=ceil((1-tol)*ideal), max=floor((1+tol)*ideal)
    con_zero = PopulationConstraint(small_square_graph, 4, 0.0)
    @test con_zero.min_pop == 4
    @test con_zero.max_pop == 4

    # tolerance=0.05 on ideal=4: min=ceil(3.8)=4, max=floor(4.2)=4
    con_5pct = PopulationConstraint(small_square_graph, 4, 0.05)
    @test con_5pct.min_pop == 4
    @test con_5pct.max_pop == 4

    # 2 districts: ideal_pop=8, tolerance=0.15
    # min=ceil(6.8)=7, max=floor(9.2)=9
    con_2dist = PopulationConstraint(small_square_graph, 2, 0.15)
    @test con_2dist.min_pop == 7
    @test con_2dist.max_pop == 9

    # min ≤ max and they bracket the ideal population
    ideal = small_square_graph.graphs_by_level[1].total_pop / 4
    @test con_5pct.min_pop ≤ ideal ≤ con_5pct.max_pop
end

@testset "One-tree cycle walk proposal" begin
    rng = PCG.PCGStateOneseq(UInt64, 11111)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    proposal = build_one_tree_cycle_walk(constraints)
    run_metropolis_hastings!(partition, proposal, Measure(), 200, rng)
    @test partition.num_dists == 4
end

@testset "MCMC sample hook follows regular output cadence" begin
    rng = PCG.PCGStateOneseq(UInt64, 33334)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    proposal = build_one_tree_cycle_walk(constraints)
    observed_steps = Int[]
    run_metropolis_hastings!(partition, proposal, Measure(), 25, rng;
                             output_freq=5, output_initial=false,
                             sample_hook=(_, step) -> push!(observed_steps, step))
    @test observed_steps == [5, 10, 15, 20, 25]
end

@testset "Two-tree cycle walk proposal" begin
    rng = PCG.PCGStateOneseq(UInt64, 22222)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    proposal = build_two_tree_cycle_walk(constraints)
    run_metropolis_hastings!(partition, proposal, Measure(), 200, rng)
    @test partition.num_dists == 4
end

@testset "Writer produces non-empty Atlas output" begin
    rng = PCG.PCGStateOneseq(UInt64, 33333)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    measure = Measure()
    push_energy!(measure, get_log_spanning_forests, 1.0)
    proposal = build_lifted_tree_cycle_walk(constraints)

    mktempdir() do tmpdir
        output_path = joinpath(tmpdir, "test_output.jsonl.gz")
        writer = Writer(measure, constraints, partition, output_path)
        push_writer!(writer, get_log_spanning_forests)
        run_metropolis_hastings!(partition, proposal, measure, 50, rng;
                                 writer=writer, output_freq=10)
        close_writer(writer)
        @test isfile(output_path)
        @test filesize(output_path) > 0
    end
end

@testset "get_isoperimetric_score (scalar)" begin
    rng = PCG.PCGStateOneseq(UInt64, 55555)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    score  = get_isoperimetric_score(partition)
    scores = get_isoperimetric_scores(partition)
    @test score isa Float64
    @test isfinite(score)
    @test score > 0
    @test score ≈ sum(scores)
end

@testset "get_log_linking_edges" begin
    rng = PCG.PCGStateOneseq(UInt64, 66666)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    result = get_log_linking_edges(partition)
    @test result isa Float64
    @test isfinite(result)
    @test result >= 0
end

@testset "get_log_district_trees" begin
    rng = PCG.PCGStateOneseq(UInt64, 77777)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    result = get_log_district_trees(partition)
    @test result isa Float64
    @test isfinite(result)
end

@testset "get_cut_edge_sum" begin
    rng = PCG.PCGStateOneseq(UInt64, 88888)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    result = get_cut_edge_sum(partition)
    # 4 districts in a 4×4 grid require at least 3 boundary edges
    @test result >= 3
end

@testset "push_energy! with log linking edges and district trees" begin
    rng = PCG.PCGStateOneseq(UInt64, 12345)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    measure = Measure()
    push_energy!(measure, get_log_linking_edges, 1.0)
    push_energy!(measure, get_log_district_trees, 1.0)
    proposal = build_lifted_tree_cycle_walk(constraints)
    run_metropolis_hastings!(partition, proposal, measure, 200, rng)
    @test partition.num_dists == 4
end

@testset "push_writer! for log spanning trees, forests, and isoperimetric scores" begin
    rng = PCG.PCGStateOneseq(UInt64, 44444)
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    partition = LinkCutPartition(small_square_graph, constraints, 4; rng=rng)
    measure = Measure()
    push_energy!(measure, get_log_spanning_forests, 1.0)
    proposal = build_lifted_tree_cycle_walk(constraints)

    mktempdir() do tmpdir
        output_path = joinpath(tmpdir, "test_writer_observables.jsonl.gz")
        writer = Writer(measure, constraints, partition, output_path)

        push_writer!(writer, get_log_spanning_trees)
        push_writer!(writer, get_log_spanning_forests)
        push_writer!(writer, get_isoperimetric_scores)

        @test length(writer.map_output_data) == 3
        @test haskey(writer.map_output_data, string(get_log_spanning_trees))
        @test haskey(writer.map_output_data, string(get_log_spanning_forests))
        @test haskey(writer.map_output_data, string(get_isoperimetric_scores))

        run_metropolis_hastings!(partition, proposal, measure, 50, rng;
                                 writer=writer, output_freq=10)

        trees  = writer.map_param[string(get_log_spanning_trees)]
        forest = writer.map_param[string(get_log_spanning_forests)]
        scores = writer.map_param[string(get_isoperimetric_scores)]

        @test trees isa Vector{Float64}
        @test length(trees) == partition.num_dists
        @test all(isfinite, trees)

        @test forest isa Float64
        @test isfinite(forest)
        @test forest ≈ sum(trees)

        @test scores isa Vector{Float64}
        @test length(scores) == partition.num_dists
        @test all(isfinite, scores)

        close_writer(writer)
        @test isfile(output_path)
        @test filesize(output_path) > 0
    end
end
