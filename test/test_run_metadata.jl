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

    @testset "AIS stamps its tag and weight endpoints" begin
        mktempdir() do dir
            path = joinpath(dir, "ais.jsonl.gz")
            p = fresh_partition(2)
            m = make_measure()
            # a genuinely nonlinear schedule — get_log_spanning_forests reaches its
            # target quadratically, get_isoperimetric_score linearly. AnnealPath is
            # introspectable by construction (unlike the old modify_measure!), so
            # the header's weights.base/weights.target need no defensive sampling
            # to trust; they're read straight from the path.
            K = 2
            scores0, target_w = annealed_smc_scores_and_targets(m)
            gamma_idx = findfirst(==(get_log_spanning_forests), scores0)
            iso_idx   = findfirst(==(get_isoperimetric_score), scores0)
            path_fn = LinearPath{K}(t -> ntuple(k ->
                k == gamma_idx ? target_w[k]*t^2 : target_w[k]*t, K))

            w = Writer(m, constraints, p, path; weight_type=Float64)
            run_annealed_importance_sampling!(p, proposal, m, 40, 20, 30,
                                              PCG.PCGStateOneseq(UInt64, 2);
                                              path=path_fn, writer=w, seed=999)
            close_writer(w)

            ap = read_atlas_param(path)
            @test ap["chain.run"] == "annealed importance sampling CycleWalk"
            params = ap["chain.parameters"]
            @test params["function"] == "run_annealed_importance_sampling!"
            @test params["n_samples"] == 2      # total_steps ÷ base_steps_per_sample
            @test params["steps_per_annealing"] == 30
            @test params["seed"] == 999
            @test params["weights.base"]["get_log_spanning_forests"] == 0.0
            @test params["weights.target"]["get_log_spanning_forests"] == 0.7
            @test params["weights.target"]["get_isoperimetric_score"] == 0.3
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

    @testset "PT stamps its tag, lattice, backend, and weight endpoints" begin
        mktempdir() do dir
            path = joinpath(dir, "pt.jsonl.gz")
            p = fresh_partition(8)
            m = make_measure()
            lattice = linear_betas(4)
            w = Writer(m, constraints, p, path)
            run_parallel_tempering!(p, proposal, m, lattice, 5, 3,
                                    PCG.PCGStateOneseq(UInt64, 8);
                                    writers=w, seed=2024)
            close_writer(w)

            ap = read_atlas_param(path)
            @test ap["chain.run"] == "parallel tempering CycleWalk"
            params = ap["chain.parameters"]
            @test params["function"] == "run_parallel_tempering!"
            @test params["betas"] == [0.0, 1/3, 2/3, 1.0]
            @test params["n_rungs"] == 4
            @test params["swap_interval"] == 5
            @test params["n_rounds"] == 3
            @test params["total_steps"] == 15
            @test params["swap_scheme"] == "deterministic even/odd (non-reversible, DEO)"
            @test params["backend"] == "serial"
            @test params["workers"] == 1
            @test params["init_steps"] == 0
            @test params["write_rungs"] == "target"
            @test params["output_every"] == 1
            @test params["heat_bath"] === nothing
            @test params["seed"] == 2024
            @test params["weights.hot"]["get_log_spanning_forests"] == 0.0
            @test params["weights.target"]["get_isoperimetric_score"] == 0.3
            @test length(params["proposal"]) == 2
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

    @testset "execution metadata: user, script_name, and script (last key)" begin
        mktempdir() do dir
            # default include_script=true stamps user + script_name + full source
            path = joinpath(dir, "exec.jsonl.gz")
            p = fresh_partition(10)
            m = make_measure()
            w = Writer(m, constraints, p, path;
                       additional_parameters=Dict{String, Any}("popdev" => 0.1))
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 10);
                                     writer=w, seed=7)
            close_writer(w)

            ap = read_atlas_param(path)
            @test haskey(ap, "user")
            # under `] test`/runtests.jl a script IS running, so these are present
            if !isempty(Base.PROGRAM_FILE)
                @test haskey(ap, "script_name")
                @test ap["script_name"] == basename(Base.PROGRAM_FILE)
                @test haskey(ap, "script")
                @test ap["script"] == read(Base.PROGRAM_FILE, String)

                # "script" must be the header's final key (raw third line ordering)
                io = AIO.smartOpen(path, "r"); lines = readlines(io); close(io)
                line3 = lines[3]
                spos = first(findfirst("\"script\":", line3))
                for k in ("energies", "districts", "user", "script_name",
                          "chain.run", "chain.parameters", "seed", "popdev")
                    @test first(findfirst("\"$k\":", line3)) < spos
                end
            end
        end
    end

    @testset "real_name and user_full_name execution metadata" begin
        # real_name() is a best-effort, environment-dependent lookup: it must
        # always return a String and never raise, but its value (and whether
        # it is non-empty) depends on the OS user database.
        name = CycleWalk.real_name()
        @test name isa AbstractString

        mktempdir() do dir
            path = joinpath(dir, "fullname.jsonl.gz")
            p = fresh_partition(20)
            m = make_measure()
            w = Writer(m, constraints, p, path; include_script=false)
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 20);
                                     writer=w, seed=20)
            close_writer(w)

            ap = read_atlas_param(path)
            if isempty(name)
                # nothing to stamp -> key is omitted rather than blank
                @test !haskey(ap, "user_full_name")
            else
                @test ap["user_full_name"] == name
            end
        end
    end

    @testset "user_full_name additional_parameter wins over real_name" begin
        mktempdir() do dir
            path = joinpath(dir, "fullname_override.jsonl.gz")
            p = fresh_partition(21)
            m = make_measure()
            w = Writer(m, constraints, p, path; include_script=false,
                       additional_parameters=Dict{String, Any}(
                           "user_full_name" => "Ada Lovelace"))
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 21);
                                     writer=w, seed=21)
            close_writer(w)

            ap = read_atlas_param(path)
            @test ap["user_full_name"] == "Ada Lovelace"
        end
    end

    @testset "include_script=false omits the embedded source" begin
        mktempdir() do dir
            path = joinpath(dir, "noscript.jsonl.gz")
            p = fresh_partition(11)
            m = make_measure()
            w = Writer(m, constraints, p, path; include_script=false)
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 11);
                                     writer=w, seed=8)
            close_writer(w)

            ap = read_atlas_param(path)
            @test haskey(ap, "user")             # user is always stamped
            @test !haskey(ap, "script")          # but the source blob is omitted
        end
    end

    @testset "config_file embeds toml_config as the header's final key" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "config.toml")
            toml_text = "[plans]\nnum_dists = 4\npop_dev = 0.02\n"
            write(toml_path, toml_text)

            path = joinpath(dir, "config.jsonl.gz")
            p = fresh_partition(13)
            m = make_measure()
            w = Writer(m, constraints, p, path;
                       additional_parameters=Dict{String, Any}("popdev" => 0.02),
                       config_file=toml_path)
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 13);
                                     writer=w, seed=13)
            close_writer(w)

            ap = read_atlas_param(path)
            @test haskey(ap, "toml_config")
            @test ap["toml_config"] == toml_text

            # "toml_config" must be the header's absolute final key, after "script"
            # and its includes (when a script is running) and after everything else.
            io = AIO.smartOpen(path, "r"); lines = readlines(io); close(io)
            line3 = lines[3]
            cpos = first(findfirst("\"toml_config\":", line3))
            other_keys = ["energies", "districts", "user", "julia_version",
                          "chain.run", "chain.parameters", "seed", "popdev"]
            if !isempty(Base.PROGRAM_FILE)
                push!(other_keys, "script")
                # the include keys are only present when there is something to record
                for k in ("script_includes", "script_includes_unresolved",
                          "script_includes_skipped")
                    haskey(ap, k) && push!(other_keys, k)
                end
            end
            for k in other_keys
                @test first(findfirst("\"$k\":", line3)) < cpos
            end
        end
    end

    @testset "no config_file => no toml_config key" begin
        mktempdir() do dir
            path = joinpath(dir, "noconfig.jsonl.gz")
            p = fresh_partition(14)
            m = make_measure()
            w = Writer(m, constraints, p, path)   # config_file defaults to nothing
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 14);
                                     writer=w, seed=14)
            close_writer(w)

            ap = read_atlas_param(path)
            @test !haskey(ap, "toml_config")
        end
    end

    @testset "nonexistent config_file is silently skipped" begin
        mktempdir() do dir
            path = joinpath(dir, "missingconfig.jsonl.gz")
            p = fresh_partition(15)
            m = make_measure()
            w = Writer(m, constraints, p, path;
                       config_file=joinpath(dir, "does_not_exist.toml"))
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 15);
                                     writer=w, seed=15)
            close_writer(w)

            ap = read_atlas_param(path)
            @test !haskey(ap, "toml_config")
        end
    end

    @testset "user additional_parameters win over config_file" begin
        mktempdir() do dir
            toml_path = joinpath(dir, "config2.toml")
            write(toml_path, "[plans]\nnum_dists = 4\n")

            path = joinpath(dir, "config_override.jsonl.gz")
            p = fresh_partition(16)
            m = make_measure()
            w = Writer(m, constraints, p, path;
                       config_file=toml_path,
                       additional_parameters=Dict{String, Any}(
                           "toml_config" => "my own config"))
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 16);
                                     writer=w, seed=16)
            close_writer(w)

            ap = read_atlas_param(path)
            @test ap["toml_config"] == "my own config"
        end
    end

    @testset "user additional_parameters override execution metadata" begin
        mktempdir() do dir
            path = joinpath(dir, "exec_override.jsonl.gz")
            p = fresh_partition(12)
            m = make_measure()
            w = Writer(m, constraints, p, path;
                       additional_parameters=Dict{String, Any}(
                           "user" => "override-user", "script" => "my own script"))
            run_metropolis_hastings!(p, proposal, m, 100,
                                     PCG.PCGStateOneseq(UInt64, 12);
                                     writer=w, seed=9)
            close_writer(w)

            ap = read_atlas_param(path)
            @test ap["user"] == "override-user"  # user's key wins
            @test ap["script"] == "my own script"
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
