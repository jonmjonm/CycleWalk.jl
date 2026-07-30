# Tests for the pieces that let an existing atlas be extended with more samples,
# restarting the chain from the last plan the atlas recorded (see
# examples/run_cyclewalk_extend.jl):
#   * `Writer(...; write_header=false)`, which appends maps to a file that already
#     carries a header rather than embedding a second header block in its middle;
#   * rebuilding a `LinkCutPartition` from a recorded map's districting;
#   * continuing the step numbering across the seam via `steps::Tuple`.

@testset "atlas extension" begin
    AIO = CycleWalk.AtlasIO

    extend_json = joinpath(testdir, "test_graphs", "4x4pct_2x2cnty.json")
    node_data = Set(["county", "pct", "pop", "area", "border_length"])
    base_graph = BaseGraph(extend_json, "pop"; inc_node_data=node_data,
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
        push_energy!(m, get_log_spanning_forests, 1.0)
        m
    end
    fresh_partition(seed) =
        LinkCutPartition(graph, constraints, num_dists;
                         rng=PCG.PCGStateOneseq(UInt64, seed))

    # read an atlas back: its header params and every map it holds
    read_atlas(path) = begin
        io = AIO.smartOpen(path, "r")
        atlas = AIO.openAtlas(io)
        maps = AIO.Map[]
        while !eof(io)
            push!(maps, AIO.nextMap(atlas))
        end
        close(io)
        (atlas.atlasParam, maps)
    end

    map_step(m) = parse(Int, match(r"^step(\d+)$", m.name).captures[1])

    # Write a short run and return (path, its maps, the partition it ended on).
    function seed_atlas(dir, name, seed, steps, freq)
        path = joinpath(dir, name)
        p = fresh_partition(seed)
        m = make_measure()
        w = Writer(m, constraints, p, path)
        run_metropolis_hastings!(p, proposal, m, steps,
                                 PCG.PCGStateOneseq(UInt64, seed);
                                 writer=w, output_freq=freq, seed=seed)
        close_writer(w)
        return path, p
    end

    @testset "write_header=false appends without a second header" begin
        mktempdir() do dir
            path, _ = seed_atlas(dir, "append.jsonl.gz", 21, 400, 100)
            _, parent_maps = read_atlas(path)
            @test length(parent_maps) > 1
            last_step = map_step(parent_maps[end])

            # resume from the last recorded plan and append to the same file
            assignment = Dict{Tuple{Vararg{String}}, Int}(parent_maps[end].districting)
            rng = PCG.PCGStateOneseq(UInt64, 22)
            p2 = LinkCutPartition(MultiLevelPartition(graph, assignment), rng)
            m2 = make_measure()
            w2 = Writer(m2, constraints, p2, path; io_mode="a", write_header=false)
            run_metropolis_hastings!(p2, proposal, m2,
                                     (last_step+1, last_step+400), rng;
                                     writer=w2, output_freq=100, seed=22)
            close_writer(w2)

            # The whole file still parses as one atlas: one header, then every map
            # from both runs, with step numbers continuing monotonically.
            ap, maps = read_atlas(path)
            @test haskey(ap, "energies")                 # the parent's header survived
            @test length(maps) > length(parent_maps)
            steps = map_step.(maps)
            @test issorted(steps)
            @test length(unique(steps)) == length(steps)
            @test steps[1:length(parent_maps)] == map_step.(parent_maps)
            @test maximum(steps) > last_step
        end
    end

    @testset "write_header=true (default) still writes exactly one header" begin
        mktempdir() do dir
            path, _ = seed_atlas(dir, "plain.jsonl.gz", 23, 200, 100)
            io = AIO.smartOpen(path, "r"); lines = readlines(io); close(io)
            # line 1 is the format banner; it must appear once and only once
            @test count(l -> occursin("This is an Atlas for Redistricting Maps",
                                      l), lines) == 1
        end
    end

    @testset "a partition rebuilt from a recorded map has that map's plan" begin
        mktempdir() do dir
            path, _ = seed_atlas(dir, "resume.jsonl.gz", 24, 300, 100)
            _, maps = read_atlas(path)
            last_map = maps[end]

            assignment = Dict{Tuple{Vararg{String}}, Int}(last_map.districting)
            resumed = LinkCutPartition(MultiLevelPartition(graph, assignment),
                                       PCG.PCGStateOneseq(UInt64, 25))

            @test resumed.num_dists == num_dists
            @test satisfies_constraints(resumed, constraints; check_population=true)

            # The reload holds the recorded plan, but the constructor labels districts
            # by the link-cut tree's scan order, so the labels need not match yet:
            # nodes that shared a district still do, and nodes that did not still do
            # not.
            node_field = resumed.node_col
            keys_by_node = [(resumed.graph.node_attributes[ni][node_field],)
                            for ni = 1:resumed.graph.num_nodes]
            same_plan = all(
                (resumed.node_to_dist[ni] == resumed.node_to_dist[nj]) ==
                (assignment[keys_by_node[ni]] == assignment[keys_by_node[nj]])
                for ni = 1:resumed.graph.num_nodes,
                    nj = 1:resumed.graph.num_nodes)
            @test same_plan

            # …and relabelling restores the recorded labels exactly.
            relabel_districts!(resumed, assignment)
            @test all(resumed.node_to_dist[ni] == assignment[keys_by_node[ni]]
                      for ni = 1:resumed.graph.num_nodes)
            @test satisfies_constraints(resumed, constraints; check_population=true)
            # the relabelled partition is still a working chain state
            @test length(resumed.cross_district_edges) > 0
            m = make_measure()
            run_metropolis_hastings!(resumed, proposal, m, 200,
                                     PCG.PCGStateOneseq(UInt64, 26))
            @test satisfies_constraints(resumed, constraints; check_population=true)
        end
    end

    @testset "collect_included_sources follows a script's includes" begin
        mktempdir() do dir
            # a -> b -> c, plus b -> a (a cycle), a missing file, and a computed path
            write(joinpath(dir, "a.jl"), """
                include("b.jl")
                include("missing.jl")
                include(joinpath(somewhere, "computed.jl"))
                # include("commented_out.jl")
                """)
            write(joinpath(dir, "b.jl"), "include(\"sub/c.jl\")\ninclude(\"a.jl\")\n")
            mkpath(joinpath(dir, "sub"))
            write(joinpath(dir, "sub", "c.jl"), "x = 1\n")

            sources, unresolved, skipped =
                collect_included_sources(joinpath(dir, "a.jl"))

            # nested includes are followed, and resolved relative to their own file
            @test collect(keys(sources)) == ["b.jl", joinpath("sub", "c.jl")]
            @test sources[joinpath("sub", "c.jl")] == "x = 1\n"
            # the include back to the entry script is not re-read
            @test !haskey(sources, "a.jl")
            @test isempty(skipped)

            # a missing file and a computed path are reported, not followed;
            # a commented-out include is ignored entirely
            @test length(unresolved) == 2
            @test any(occursin("missing.jl", u) for u in unresolved)
            @test any(occursin("computed.jl", u) for u in unresolved)
            @test !any(occursin("commented_out", u) for u in unresolved)
        end
    end

    @testset "max_bytes caps the embedded source" begin
        mktempdir() do dir
            write(joinpath(dir, "a.jl"), "include(\"big.jl\")\ninclude(\"small.jl\")\n")
            write(joinpath(dir, "big.jl"), "#"^5000)
            write(joinpath(dir, "small.jl"), "y = 2\n")

            sources, _, skipped =
                collect_included_sources(joinpath(dir, "a.jl"); max_bytes=1000)
            @test skipped == ["big.jl"]
            # the cap skips the oversized file but keeps going
            @test collect(keys(sources)) == ["small.jl"]
        end
    end

    @testset "a script's includes reach the atlas header" begin
        # PROGRAM_FILE is what the writer embeds, and it is whatever is running the
        # tests, so drive a real script through a subprocess instead of faking it.
        mktempdir() do dir
            write(joinpath(dir, "helper.jl"), "const HELPER_MARKER = 12345\n")
            script = joinpath(dir, "runner.jl")
            write(script, """
                using CycleWalk, RandomNumbers
                include("helper.jl")
                graph_json = $(repr(extend_json))
                nd = Set(["county", "pct", "pop", "area", "border_length"])
                bg = BaseGraph(graph_json, "pop"; inc_node_data=nd, area_col="area",
                               node_border_col="border_length",
                               edge_perimeter_col="length")
                g = MultiLevelGraph(bg, ["pct"])
                cons = initialize_constraints()
                add_constraint!(cons, PopulationConstraint(g, 4, 0.1))
                p = LinkCutPartition(g, cons, 4; rng=PCG.PCGStateOneseq(UInt64, 5))
                m = Measure(); push_energy!(m, get_log_spanning_forests, 1.0)
                w = Writer(m, cons, p, $(repr(joinpath(dir, "out.jsonl.gz"))))
                close_writer(w)
                """)
            project = dirname(Base.active_project())
            run(pipeline(`$(Base.julia_cmd()) --project=$project $script`;
                         stdout=devnull, stderr=devnull))

            ap, _ = read_atlas(joinpath(dir, "out.jsonl.gz"))
            @test ap["script_name"] == "runner.jl"
            @test occursin("include(\"helper.jl\")", ap["script"])
            @test haskey(ap, "script_includes")
            @test ap["script_includes"]["helper.jl"] == "const HELPER_MARKER = 12345\n"
            @test haskey(ap, "julia_version")
        end
    end

    @testset "relabel_districts! rejects an assignment that is not a relabelling" begin
        p = fresh_partition(27)
        node_field = p.node_col
        assignment = Dict{Tuple{Vararg{String}}, Int}(
            (p.graph.node_attributes[ni][node_field],) => p.node_to_dist[ni]
            for ni = 1:p.graph.num_nodes)

        # merge two districts: no longer a permutation of the same plan
        merged = Dict(k => (v == num_dists ? 1 : v) for (k, v) in assignment)
        @test_throws ArgumentError relabel_districts!(p, merged)

        # move a single node across: districts no longer map one-to-one
        split = copy(assignment)
        some_key = first(k for (k, v) in assignment if v == 1)
        split[some_key] = 2
        @test_throws ArgumentError relabel_districts!(p, split)
    end

    @testset "a resumed chain keeps sampling the same distribution" begin
        # Restarting from a recorded plan redraws the spanning forest, because an
        # atlas stores the districting and not the forest. That redraw is a Gibbs
        # refresh of the forest coordinate and must not move the target: a chain that
        # is repeatedly stopped and resumed through the recorded assignment should
        # produce the same cut-edge distribution as one continuous chain.
        total_steps = 600_000
        sample_freq = 10
        n_segments = 60          # a stop-and-resume every 10k steps

        # `partition` -> the assignment an atlas records -> `partition`, which is the
        # round trip run_cyclewalk_extend.jl makes through a file.
        function through_assignment(partition, rng)
            assignment = CycleWalk.get_node_map(partition.node_col, partition)
            reloaded = LinkCutPartition(MultiLevelPartition(graph, assignment), rng)
            return relabel_districts!(reloaded, assignment)
        end

        function cut_edge_counts(seed; segments::Int)
            rng = PCG.PCGStateOneseq(UInt64, seed)
            partition = fresh_partition(seed)
            measure = make_measure()
            counts = Dict{Int64, Int64}()
            seg_steps = total_steps ÷ segments
            step = 0
            for si = 1:segments
                si == 1 || (partition = through_assignment(partition, rng))
                for _ = 1:(seg_steps ÷ sample_freq)
                    run_metropolis_hastings!(partition, proposal, measure,
                                             (step+1, step+sample_freq), rng)
                    step += sample_freq
                    ce = get_cut_edge_sum(partition, column="connections")
                    counts[ce] = get(counts, ce, 0) + 1
                end
            end
            return counts
        end

        continuous = cut_edge_counts(41; segments=1)
        resumed    = cut_edge_counts(42; segments=n_segments)

        n_c = sum(values(continuous))
        n_r = sum(values(resumed))
        @test n_c == n_r == total_steps ÷ sample_freq

        # Compare every cut-edge count that carries real mass in the continuous run
        # (is_close is looser below 1% mass, where the sampling error is larger).
        for (ce, count) in continuous
            p_c = count/n_c
            p_c < 0.001 && continue
            p_r = get(resumed, ce, 0)/n_r
            @test is_close(p_c, p_r)
        end
    end
end
