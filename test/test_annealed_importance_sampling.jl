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
        # one outer sample, one annealing step: the base measure has weight 0, so the
        # log weight is exactly log[ν_target/ν_base] at the base-chain state, which
        # `partition` still holds after the run (annealing acts on a deep copy).
        # ν ∝ exp(−get_log_energy), so that is the NEGATIVE of the target log-energy.
        log_weights = run_annealed_importance_sampling!(
            partition, proposal, measure, anneal_forest_weight!,
            10, 10, 1, rng)
        @test length(log_weights) == 1
        expected = -get_log_energy(partition, measure)
        @test log_weights[1] ≈ expected
        # direction: raising γ from 0 penalises this state (positive energy), so the
        # importance weight must be < 1, i.e. log weight < 0
        @test log_weights[1] < 0
    end

    # ------------------------------------------------------------------------
    # Sign / correctness regression. AIS and ASMC accumulated their log weights
    # with the sign of the ENERGY rather than of the log-density, which made both
    # estimate Z_base/Z_target. Nothing caught it: every existing assertion either
    # restated the implementation or used a zero/constant schedule where the sign
    # is invisible. These tests pin the estimator to an exactly known answer.
    #
    # On the 4x4 test graph with PopulationConstraint(4,4) the partition function
    # of pi_gamma(xi) ∝ Tree(xi)^(1-gamma) is enumerated in
    # test/test_cases/small_square_p88_unweighted.jl:
    #     Z(0) = 256 + 224 + 96 + 78 = 654      (gamma = 0)
    #     Z(1) =   1 +  14 + 24 + 78 = 117      (gamma = 1)
    # so annealing gamma from 0 to 1 must give log(Z(1)/Z(0)) = log(117/654).
    # The measured spread over 5 seeds is <= 0.15; a flipped sign lands ~4.1 away,
    # so atol=0.5 separates the two by a factor of ~8 without being flaky.
    # ------------------------------------------------------------------------
    @testset "logZ matches the exactly known log Z(1)/Z(0)" begin
        AIO = CycleWalk.AtlasIO
        exact_logZ = log(117 / 654)

        # gamma=1 target, and the p88 hard population constraint (min=max=4) that the
        # enumeration assumes -- NOT the pop_dev=0.1 constraint used elsewhere here.
        p88_constraints = initialize_constraints()
        add_constraint!(p88_constraints, PopulationConstraint(4, 4))
        p88_proposal = [(0.1, build_lifted_tree_cycle_walk(p88_constraints)),
                        (0.9, build_internal_forest_walk(p88_constraints))]
        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, 1.0)
        ramp!(m, step, total) =
            (m.weights[get_log_spanning_forests] = step / total)

        rng = PCG.PCGStateOneseq(UInt64, 20260731)
        partition = LinkCutPartition(ais_graph, p88_constraints, 4; rng=rng)

        mktempdir() do tmpdir
            output_path = joinpath(tmpdir, "ais_logz.jsonl.gz")
            writer = Writer(measure, p88_constraints, partition, output_path;
                            weight_type=Float64)
            push_writer!(writer, get_cut_edge_sum)
            log_weights = run_annealed_importance_sampling!(
                partition, p88_proposal, measure, ramp!,
                120_000, 100, 400, rng; writer=writer)
            close_writer(writer)

            # logZ = log mean exp(log w)
            mx = maximum(log_weights)
            logZ = mx + log(sum(x -> exp(x - mx), log_weights)) - log(length(log_weights))
            @test logZ ≈ exact_logZ atol = 0.5
            # Z(1) < Z(0), so the estimate must be negative. This alone catches the
            # sign flip, independent of how tight the tolerance is.
            @test logZ < 0

            # Stronger than logZ: the weights must reweight the gamma=0 samples into
            # the gamma=1 distribution. Compare the weighted cut-edge histogram with
            # the enumerated one. (logZ is an average and could in principle be right
            # while individual weights were not.)
            io = AIO.smartOpen(output_path, "r")
            atlas = AIO.openAtlas(io)
            maps = AIO.nextMaps(atlas)
            close(io)
            @test length(maps) == length(log_weights)

            w = exp.(log_weights .- mx)
            acc = Dict{Int, Float64}()
            for (m, wi) in zip(maps, w)
                ce = Int(m.data["get_cut_edge_sum"])
                acc[ce] = get(acc, ce, 0.0) + wi
            end
            tot = sum(values(acc))
            truth = Dict(8 => 1/117, 10 => 14/117, 11 => 24/117, 12 => 78/117)
            @test Set(keys(acc)) == Set(keys(truth))
            # L1 distance between the reweighted and enumerated distributions.
            # Measured ~0.04 when correct; the sign flip gives ~0.5.
            l1 = sum(abs(acc[k]/tot - truth[k]) for k in keys(truth))
            @test l1 < 0.15
        end
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
