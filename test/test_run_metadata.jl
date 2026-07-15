# Tests for the run-metadata that each sampler stamps onto the Atlas header (the
# file's third line): the human-readable `chain.run` tag and the nested
# `chain.parameters` dict describing the run. Reads the header back via
# AtlasIO.openAtlas (whose `atlasParam` is the parsed third line).

@testset "run metadata (chain.run)" begin
    AIO = CycleWalk.AtlasIO

    run_metadata_json = joinpath(testdir, "test_graphs", "4x4pct_2x2cnty.json")
    node_data = Set(["county", "pct", "pop", "area", "border_length"])
    base_graph = BaseGraph(run_metadata_json, "pop"; inc_node_data=node_data,
                           area_col="area", node_border_col="border_length",
                           edge_perimeter_col="length")
    graph = MultiLevelGraph(base_graph, ["pct"])

    num_dists = 4
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(graph, num_dists, 0.1))

    cycle_walk = build_lifted_tree_cycle_walk(constraints)
    internal_walk = build_internal_forest_walk(constraints)
    proposal = [(0.5, cycle_walk), (0.5, internal_walk)]

    make_measure() = begin
        m = Measure()
        push_energy!(m, get_log_spanning_forests, 0.7)
        push_energy!(m, get_isoperimetric_score, 0.3)
        m
    end
    fresh_partition(seed) =
        LinkCutPartition(graph, constraints, num_dists;
                         rng=PCG.PCGStateOneseq(UInt64, seed))

    # read the parsed atlasParam (third line) back from an atlas file
    read_atlas_param(path) = begin
        io = AIO.smartOpen(path, "r")
        atlas = AIO.openAtlas(io)
        close(io)
        atlas.atlasParam
    end

    @testset "standard MH stamps its tag and parameters" begin
        mktempdir() do dir
            path = joinpath(dir, "mh.jsonl.gz")
            p = fresh_partition(1)
            m = make_measure()
            w = Writer(m, constraints, p, path)
            run_metropolis_hastings!(p, proposal, m, 200,
                                     PCG.PCGStateOneseq(UInt64, 1);
                                     writer=w, output_freq=100, seed=12345)
            close_writer(w)

            ap = read_atlas_param(path)
            @test ap["chain.run"] == "standard metropolized CycleWalk"
            params = ap["chain.parameters"]
            @test params["function"] == "run_metropolis_hastings!"
            @test params["steps"] == 200
            @test params["seed"] == 12345
            @test length(params["proposal"]) == 2
            @test Set(pp["proposal"] for pp in params["proposal"]) ==
                  Set(["two_tree_cycle_walk", "one_tree_cycle_walk"])
            # the writer's own header keys survive alongside the new metadata
            @test haskey(ap, "energies")
            @test ap["districts"] == num_dists
        end
    end

    @testset "AIS stamps its tag, schedule endpoints, and sample count" begin
        mktempdir() do dir
            path = joinpath(dir, "ais.jsonl.gz")
            p = fresh_partition(2)
            m = make_measure()
            base_w = Dict(get_log_spanning_forests => 0.0,
                          get_isoperimetric_score => 0.0)
            tgt_w = Dict(get_log_spanning_forests => 0.7,
                         get_isoperimetric_score => 0.3)
            modify_measure! = (mm, step, total) -> begin
                frac = step / total
                for (f, tw) in tgt_w
                    mm.weights[f] = base_w[f] + (tw - base_w[f]) * frac
                end
            end
            w = Writer(m, constraints, p, path; weight_type=Float64)
            run_annealed_importance_sampling!(p, proposal, m, modify_measure!,
                                              40, 20, 30,
                                              PCG.PCGStateOneseq(UInt64, 2);
                                              writer=w, seed=999, schedule="linear")
            close_writer(w)

            ap = read_atlas_param(path)
            @test ap["chain.run"] == "annealed importance sampling CycleWalk"
            params = ap["chain.parameters"]
            @test params["function"] == "run_annealed_importance_sampling!"
            @test params["schedule"] == "linear"
            @test params["n_samples"] == 2      # total_steps ÷ base_steps_per_sample
            @test params["steps_per_annealing"] == 30
            @test params["seed"] == 999
            @test params["weights.base"]["get_log_spanning_forests"] == 0.0
            @test params["weights.target"]["get_log_spanning_forests"] == 0.7
        end
    end

    @testset "ASMC stamps its tag, schedule, and path endpoints" begin
        mktempdir() do dir
            path = joinpath(dir, "asmc.jsonl.gz")
            p = fresh_partition(3)
            m = make_measure()
            sched = FixedSchedule(range(0.0, 1.0; length=6); ess_frac=0.5)
            w = Writer(m, constraints, p, path; weight_type=Float64)
            run_annealed_smc!(p, proposal, m, sched, 8, 5,
                              PCG.PCGStateOneseq(UInt64, 3);
                              init_steps=5, writer=w, seed=777)
            close_writer(w)

            ap = read_atlas_param(path)
            @test ap["chain.run"] == "annealed sequential monte carlo CycleWalk"
            params = ap["chain.parameters"]
            @test params["function"] == "run_annealed_smc!"
            @test params["n_particles"] == 8
            @test params["rejuv_steps"] == 5
            @test params["seed"] == 777
            @test params["schedule"]["kind"] == "fixed"
            @test params["schedule"]["blocks"] == 5
            @test params["weights.target"]["get_isoperimetric_score"] == 0.3
        end
    end

    @testset "user additional_parameters win over auto-stamped fields" begin
        mktempdir() do dir
            path = joinpath(dir, "override.jsonl.gz")
            p = fresh_partition(4)
            m = make_measure()
            w = Writer(m, constraints, p, path;
                       additional_parameters=Dict{String, Any}(
                           "chain.run" => "MY OVERRIDE", "my_custom" => 42))
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 4);
                                     writer=w, seed=5)
            close_writer(w)

            ap = read_atlas_param(path)
            @test ap["chain.run"] == "MY OVERRIDE"     # user key preserved
            @test ap["my_custom"] == 42                # custom key preserved
            @test haskey(ap, "chain.parameters")   # auto params still stamped
        end
    end

    @testset "seed is omitted when not supplied" begin
        mktempdir() do dir
            path = joinpath(dir, "noseed.jsonl.gz")
            p = fresh_partition(5)
            m = make_measure()
            w = Writer(m, constraints, p, path)
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 5); writer=w)
            close_writer(w)

            ap = read_atlas_param(path)
            @test !haskey(ap["chain.parameters"], "seed")
        end
    end

    @testset "close_writer alone writes a valid (map-less) header" begin
        mktempdir() do dir
            path = joinpath(dir, "empty.jsonl.gz")
            p = fresh_partition(6)
            m = make_measure()
            w = Writer(m, constraints, p, path)
            close_writer(w)                # no run, no maps

            ap = read_atlas_param(path)    # header still parses
            @test haskey(ap, "energies")
            @test !haskey(ap, "chain.run") # nothing stamped it
        end
    end

    @testset "proposal names: aliases, single vs mixture, untagged fallback" begin
        # every alias of the two builders resolves to its registered name
        @test proposal_name(build_one_tree_cycle_walk(constraints)) == "one_tree_cycle_walk"
        @test proposal_name(build_internal_forest_walk(constraints)) == "one_tree_cycle_walk"
        @test proposal_name(build_two_tree_cycle_walk(constraints)) == "two_tree_cycle_walk"
        @test proposal_name(build_lifted_tree_cycle_walk(constraints)) == "two_tree_cycle_walk"
        @test proposal_name(build_cycle_walk(constraints)) == "two_tree_cycle_walk"

        # an unregistered closure falls back to string(f)
        untagged = (p, r; diagnostics=nothing) -> nothing
        @test proposal_name(untagged) == string(untagged)

        # describe_proposal on a single closure is a bare name; on a mixture, a vector
        @test describe_proposal(build_one_tree_cycle_walk(constraints)) == "one_tree_cycle_walk"
        desc = describe_proposal(proposal)
        @test desc isa Vector
        @test Set(d["proposal"] for d in desc) ==
              Set(["two_tree_cycle_walk", "one_tree_cycle_walk"])
        @test [d["weight"] for d in desc] == [0.5, 0.5]

        # name_proposal! returns the closure unchanged and tags it
        f = (p, r; diagnostics=nothing) -> nothing
        @test name_proposal!(f, "my_move") === f
        @test proposal_name(f) == "my_move"
    end

    @testset "ASMC AdaptiveTempering schedule metadata" begin
        mktempdir() do dir
            path = joinpath(dir, "adaptive.jsonl.gz")
            p = fresh_partition(7)
            m = make_measure()
            sched = AdaptiveTempering(; ess_target=0.5)
            w = Writer(m, constraints, p, path; weight_type=Float64)
            run_annealed_smc!(p, proposal, m, sched, 8, 5,
                              PCG.PCGStateOneseq(UInt64, 7);
                              init_steps=5, writer=w, seed=321)
            close_writer(w)

            ap = read_atlas_param(path)
            sch = ap["chain.parameters"]["schedule"]
            @test sch["kind"] == "adaptive"
            @test sch["ess_target"] == 0.5
            @test occursin("every step", sch["resampling"])
        end
    end

    @testset "ASMC records non-zero annealing start (base) parameters" begin
        mktempdir() do dir
            path = joinpath(dir, "nonzero_base.jsonl.gz")
            p = fresh_partition(8)
            m = make_measure()
            # a LinearPath ramping from a NON-zero base (γ=0.2, iso=0.1) to the target
            scores, target_w = annealed_smc_scores_and_targets(m)
            K = length(scores)
            base_by_fn = Dict(get_log_spanning_forests => 0.2,
                              get_isoperimetric_score => 0.1)
            base_w = ntuple(k -> base_by_fn[scores[k]], K)
            path_fn = LinearPath{K}(t -> ntuple(k -> base_w[k] +
                                                     t * (target_w[k] - base_w[k]), K))
            sched = FixedSchedule(range(0.0, 1.0; length=6); ess_frac=0.5)
            w = Writer(m, constraints, p, path; weight_type=Float64)
            run_annealed_smc!(p, proposal, m, sched, 8, 5,
                              PCG.PCGStateOneseq(UInt64, 8);
                              path=path_fn, init_steps=5, writer=w, seed=654)
            close_writer(w)

            ap = read_atlas_param(path)
            params = ap["chain.parameters"]
            # weights.base reflects path(0) = the non-zero start, not zeros
            @test params["weights.base"]["get_log_spanning_forests"] == 0.2
            @test params["weights.base"]["get_isoperimetric_score"] == 0.1
            @test params["weights.target"]["get_log_spanning_forests"] == 0.7
            @test params["weights.target"]["get_isoperimetric_score"] == 0.3
        end
    end

    @testset "header is written exactly once (no duplicate preamble)" begin
        mktempdir() do dir
            path = joinpath(dir, "once.jsonl.gz")
            p = fresh_partition(9)
            m = make_measure()
            w = Writer(m, constraints, p, path)
            # a run that writes several maps, so the header + many map lines coexist
            run_metropolis_hastings!(p, proposal, m, 500,
                                     PCG.PCGStateOneseq(UInt64, 9);
                                     writer=w, output_freq=100, seed=42)
            close_writer(w)

            io = AIO.smartOpen(path, "r")
            lines = readlines(io)
            close(io)
            banner = count(l -> occursin("This is an Atlas for Redistricting Maps", l),
                           lines)
            @test banner == 1               # exactly one header preamble, never duplicated
            @test length(lines) > 3         # header (3 lines) plus at least one map
        end
    end
end
