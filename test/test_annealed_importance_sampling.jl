using CycleWalk
using RandomNumbers
using Test

@testset verbose = true "annealed_importance_sampling" begin
    ais_testdir = dirname(@__FILE__)
    ais_json = joinpath(ais_testdir, "test_graphs", "4x4pct_2x2cnty.json")
    ais_node_data = Set(["county", "pct", "pop", "area", "border_length"])
    ais_base_graph = BaseGraph(ais_json, "pop",
                               inc_node_data=ais_node_data,
                               area_col="area",
                               node_border_col="border_length",
                               edge_perimeter_col="length")
    ais_graph = MultiLevelGraph(ais_base_graph, ["pct"])

    num_dists = 4
    pop_dev = 0.1
    gamma = 0.7

    constraints = initialize_constraints()
    add_constraint!(constraints,
                    PopulationConstraint(ais_graph, num_dists, pop_dev))

    cycle_walk = build_lifted_tree_cycle_walk(constraints)
    internal_walk = build_internal_forest_walk(constraints)
    proposal = [(0.5, cycle_walk), (0.5, internal_walk)]

    # anneal the spanning-forest energy weight linearly from 0 to gamma
    function anneal_forest_weight!(m::Measure, step::Int, total::Int)
        m.weights[get_log_spanning_forests] = gamma*step/total
    end

    @testset "constant schedule gives zero log weights" begin
        rng = PCG.PCGStateOneseq(UInt64, 1241909)
        partition = LinkCutPartition(ais_graph, constraints, num_dists;
                                     rng=rng)
        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, gamma)
        log_weights = run_annealed_importance_sampling!(
            partition, proposal, measure, (m, s, t)->nothing,
            40, 10, 5, rng)
        @test length(log_weights) == 4
        @test all(log_weights .== 0.0)
    end

    @testset "single-step annealing matches direct energy difference" begin
        rng = PCG.PCGStateOneseq(UInt64, 90210)
        partition = LinkCutPartition(ais_graph, constraints, num_dists;
                                     rng=rng)
        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, gamma)
        # one outer sample, one annealing step: the base measure has weight 0,
        # so the log weight is exactly the target log-energy of the base-chain
        # state, which `partition` still holds after the run (annealing acts
        # on a deep copy)
        log_weights = run_annealed_importance_sampling!(
            partition, proposal, measure, anneal_forest_weight!,
            10, 10, 1, rng)
        @test length(log_weights) == 1
        expected = get_log_energy(partition, measure)
        @test log_weights[1] ≈ expected
    end

    @testset "multi-step annealing runs and returns finite weights" begin
        rng = PCG.PCGStateOneseq(UInt64, 4541901234)
        partition = LinkCutPartition(ais_graph, constraints, num_dists;
                                     rng=rng)
        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, gamma)
        user_weight_before = measure.weights[get_log_spanning_forests]
        log_weights = run_annealed_importance_sampling!(
            partition, proposal, measure, anneal_forest_weight!,
            60, 20, 25, rng)
        @test length(log_weights) == 3
        @test all(isfinite.(log_weights))
        # the caller's measure must come through the run unmodified
        @test measure.weights[get_log_spanning_forests] == user_weight_before
    end

    @testset "writer records log weights in the Atlas output" begin
        AIO = CycleWalk.AtlasIO
        rng = PCG.PCGStateOneseq(UInt64, 777333)
        partition = LinkCutPartition(ais_graph, constraints, num_dists;
                                     rng=rng)
        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, gamma)
        mktempdir() do tmpdir
            output_path = joinpath(tmpdir, "ais_output.jsonl.gz")
            writer = Writer(measure, constraints, partition, output_path;
                            weight_type=Float64)
            push_writer!(writer, get_log_spanning_forests)
            log_weights = run_annealed_importance_sampling!(
                partition, proposal, measure, anneal_forest_weight!,
                60, 20, 25, rng; writer=writer)
            close_writer(writer)

            io = AIO.smartOpen(output_path, "r")
            atlas = AIO.openAtlas(io)
            maps = AIO.nextMaps(atlas)
            close(io)
            @test atlas.weightType == Float64
            @test length(maps) == length(log_weights) == 3
            @test [m.weight for m in maps] ≈ log_weights
            # samples must land on disk in the order they left the base chain
            @test [m.name for m in maps] == ["sample1", "sample2", "sample3"]
        end
    end

    @testset "ntasks=4 reproduces ntasks=1 weights and file order" begin
        AIO = CycleWalk.AtlasIO
        function ais_run(ntasks::Int, output_path::String)
            rng = PCG.PCGStateOneseq(UInt64, 246810)
            partition = LinkCutPartition(ais_graph, constraints, num_dists;
                                         rng=rng)
            measure = Measure()
            push_energy!(measure, get_log_spanning_forests, gamma)
            writer = Writer(measure, constraints, partition, output_path;
                            weight_type=Float64)
            push_writer!(writer, get_log_spanning_forests)
            log_weights = run_annealed_importance_sampling!(
                partition, proposal, measure, anneal_forest_weight!,
                120, 15, 20, rng; writer=writer, ntasks=ntasks)
            close_writer(writer)
            io = AIO.smartOpen(output_path, "r")
            maps = AIO.nextMaps(AIO.openAtlas(io))
            close(io)
            return log_weights, maps
        end
        mktempdir() do tmpdir
            lw_serial, maps_serial = ais_run(1, joinpath(tmpdir, "s.jsonl.gz"))
            lw_conc, maps_conc = ais_run(4, joinpath(tmpdir, "c.jsonl.gz"))
            @test length(lw_serial) == 8
            # per-sample seeds are drawn on the base chain, so weights are
            # identical no matter how many tasks anneal them
            @test lw_conc == lw_serial
            @test [m.name for m in maps_conc] ==
                  ["sample"*string(i) for i = 1:8]
            @test [m.weight for m in maps_conc] ≈ lw_serial
            @test [m.districting for m in maps_conc] ==
                  [m.districting for m in maps_serial]
        end
    end

    @testset "path recorders capture the annealing path" begin
        AIO = CycleWalk.AtlasIO
        rng = PCG.PCGStateOneseq(UInt64, 135791)
        partition = LinkCutPartition(ais_graph, constraints, num_dists; rng=rng)
        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, gamma)
        mktempdir() do tmpdir
            output_path = joinpath(tmpdir, "ais_path.jsonl.gz")
            writer = Writer(measure, constraints, partition, output_path;
                            weight_type=Float64, path_target_points=10)
            push_writer!(writer, get_log_spanning_forests)
            push_path_writer!(writer, :log_weight)
            push_path_writer!(writer, :schedule_frac)
            push_path_writer!(writer, :delta_log_weight)
            push_path_writer!(writer, get_isoperimetric_score)
            log_weights = run_annealed_importance_sampling!(
                partition, proposal, measure, anneal_forest_weight!,
                60, 20, 30, rng; writer=writer, ntasks=2)
            close_writer(writer)

            io = AIO.smartOpen(output_path, "r")
            maps = AIO.nextMaps(AIO.openAtlas(io)); close(io)
            @test length(maps) == length(log_weights) == 3
            for k in ["path/log_weight", "path/schedule_frac",
                      "path/delta_log_weight", "path/get_isoperimetric_score"]
                @test haskey(maps[1].data, k)
            end
            # target_points=10 over 30 steps => stride 3 => 10 recorded points
            @test length(maps[1].data["path/log_weight"]) == 10
            @test Float64(maps[1].data["path/schedule_frac"][end]) ≈ 1.0
            # the final running log-weight must equal the map's importance weight
            for m in maps
                p = Float64.(m.data["path/log_weight"])
                @test p[end] ≈ m.weight
            end
        end
    end

    @testset "recording is opt-in: no recorders => no path data, same weights" begin
        AIO = CycleWalk.AtlasIO
        function ais_run(record::Bool, path::String)
            rng = PCG.PCGStateOneseq(UInt64, 222444)
            partition = LinkCutPartition(ais_graph, constraints, num_dists; rng=rng)
            measure = Measure()
            push_energy!(measure, get_log_spanning_forests, gamma)
            writer = Writer(measure, constraints, partition, path; weight_type=Float64)
            push_writer!(writer, get_log_spanning_forests)
            record && push_path_writer!(writer, :log_weight)
            lw = run_annealed_importance_sampling!(partition, proposal, measure,
                     anneal_forest_weight!, 60, 20, 25, rng; writer=writer)
            close_writer(writer)
            io = AIO.smartOpen(path, "r")
            maps = AIO.nextMaps(AIO.openAtlas(io)); close(io)
            return lw, maps
        end
        mktempdir() do tmpdir
            lw_off, maps_off = ais_run(false, joinpath(tmpdir, "off.jsonl.gz"))
            lw_on,  maps_on  = ais_run(true,  joinpath(tmpdir, "on.jsonl.gz"))
            # turning recording on must not change the sampled weights
            @test lw_off ≈ lw_on
            @test !any(k -> startswith(k, "path/"), keys(maps_off[1].data))
            @test any(k -> startswith(k, "path/"), keys(maps_on[1].data))
        end
    end
end
