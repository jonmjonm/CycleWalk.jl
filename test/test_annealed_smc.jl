using CycleWalk
using RandomNumbers
using Test

# Tests for the annealed SMC sampler (src/chains/annealed_smc.jl). The file is
# factored into small pure helpers (ESS, resampling, paths, schedules) plus the
# run_annealed_smc! driver, so we unit-test the helpers deterministically and then
# drive the whole sampler on closed-form / invariant checks. Internal (unexported)
# helpers are reached through the CycleWalk namespace.

@testset verbose = true "annealed_smc" begin
    annealed_smc_testdir = dirname(@__FILE__)
    annealed_smc_json = joinpath(annealed_smc_testdir, "test_graphs", "4x4pct_2x2cnty.json")
    annealed_smc_node_data = Set(["county", "pct", "pop", "area", "border_length"])
    annealed_smc_base_graph = BaseGraph(annealed_smc_json, "pop",
                               inc_node_data=annealed_smc_node_data,
                               area_col="area",
                               node_border_col="border_length",
                               edge_perimeter_col="length")
    annealed_smc_graph = MultiLevelGraph(annealed_smc_base_graph, ["pct"])

    num_dists = 4
    pop_dev = 0.1
    gamma = 0.7
    iso = 0.3

    constraints = initialize_constraints()
    add_constraint!(constraints,
                    PopulationConstraint(annealed_smc_graph, num_dists, pop_dev))

    cycle_walk = build_lifted_tree_cycle_walk(constraints)
    internal_walk = build_internal_forest_walk(constraints)
    proposal = [(0.5, cycle_walk), (0.5, internal_walk)]

    # a fresh two-energy target measure (K = 2) for the driver tests
    make_measure() = begin
        m = Measure()
        push_energy!(m, get_log_spanning_forests, gamma)
        push_energy!(m, get_isoperimetric_score, iso)
        m
    end
    fresh_partition(seed) =
        LinkCutPartition(annealed_smc_graph, constraints, num_dists;
                         rng=PCG.PCGStateOneseq(UInt64, seed))

    # ---------------------------------------------------------------- primitives

    @testset "ess_from_logw: equal weights, degenerate, shift-invariant" begin
        # equal log-weights => every particle equally likely => ESS == N
        @test CycleWalk.ess_from_logw(zeros(8)) ≈ 8.0
        @test CycleWalk.ess_from_logw(fill(3.5, 5)) ≈ 5.0
        # one dominant weight => ESS -> 1
        @test CycleWalk.ess_from_logw([0.0, -50.0, -50.0, -50.0]) ≈ 1.0 atol=1e-6
        # invariant under a constant shift of the log-weights
        lw = [0.3, -1.2, 2.0, 0.5, -0.7]
        @test CycleWalk.ess_from_logw(lw) ≈ CycleWalk.ess_from_logw(lw .+ 17.0)
        # ESS is always within (0, N]
        @test 0 < CycleWalk.ess_from_logw(lw) <= length(lw)
    end

    @testset "systematic_resample: index set, balance, degenerate" begin
        rng = PCG.PCGStateOneseq(UInt64, 11)
        # equal weights => low-variance resampling keeps exactly one copy of each
        parents = CycleWalk.systematic_resample(zeros(8), rng)
        @test length(parents) == 8
        @test all(1 .<= parents .<= 8)
        @test sort(parents) == collect(1:8)
        # a single unit weight => every parent is that particle
        deg = CycleWalk.systematic_resample([0.0, -Inf, -Inf, -Inf], rng)
        @test all(deg .== 1)
        # a heavy particle is over-represented but others can survive
        parents2 = CycleWalk.systematic_resample([5.0, 0.0, 0.0, 0.0, 0.0], rng)
        @test length(parents2) == 5
        @test count(==(1), parents2) >= 2
    end

    @testset "resample!: logZ increment, weight reset, no state aliasing" begin
        st = fresh_partition(101)
        mkparts(logws) = [CycleWalk.Particle{2}(
                              CycleWalk.clone_for_annealing(st), lw, (0.0, 0.0),
                              PCG.PCGStateOneseq(UInt64, UInt64(i)))
                          for (i, lw) in enumerate(logws)]
        rng = PCG.PCGStateOneseq(UInt64, 202)

        # equal weights => logZ increment == the common log-weight
        parts = mkparts(fill(1.5, 6))
        logZ_inc = CycleWalk.resample!(parts, rng)
        @test logZ_inc ≈ 1.5
        # every weight is reset to zero after a resample
        @test all(p -> p.logW == 0.0, parts)
        # survivors are deep-copied: no two slots share the same state object
        states = [p.state for p in parts]
        @test length(unique(objectid.(states))) == length(states)

        # degenerate weights => logZ increment == logsumexp(logw) - log N
        parts2 = mkparts([0.0, -Inf, -Inf, -Inf])
        @test CycleWalk.resample!(parts2, rng) ≈ -log(4)
    end

    # ------------------------------------------------------------------- paths

    @testset "linear_path / weights_at / energy_at" begin
        target_w = (gamma, iso)
        path = CycleWalk.linear_path(target_w)
        @test CycleWalk.weights_at(path, 0.0) == (0.0, 0.0)
        @test all(CycleWalk.weights_at(path, 1.0) .≈ target_w)
        @test all(CycleWalk.weights_at(path, 0.5) .≈ (0.5 .* target_w))
        # energy_at for a LinearPath is <weights(t), phi>
        p = CycleWalk.Particle{2}(fresh_partition(303), 0.0, (2.0, -3.0),
                                  PCG.PCGStateOneseq(UInt64, 1))
        @test CycleWalk.energy_at(path, p, 0.5) ≈
              0.5 * (gamma * 2.0 + iso * -3.0)
        @test CycleWalk.energy_at(path, p, 0.0) ≈ 0.0
    end

    @testset "incremental_logweight! matches the energy_at seam" begin
        target_w = (gamma, iso)
        path = CycleWalk.linear_path(target_w)
        st = fresh_partition(404)
        phis = [(1.0, 0.5), (-2.0, 3.0), (0.25, -0.75)]
        parts = [CycleWalk.Particle{2}(CycleWalk.clone_for_annealing(st), 0.0,
                                       phi, PCG.PCGStateOneseq(UInt64, UInt64(i)))
                 for (i, phi) in enumerate(phis)]
        t_prev, t = 0.2, 0.7
        expected = [CycleWalk.energy_at(path, p, t) -
                    CycleWalk.energy_at(path, p, t_prev) for p in parts]
        CycleWalk.incremental_logweight!(parts, path, t_prev, t)
        @test [p.logW for p in parts] ≈ expected
    end

    @testset "annealed_smc_scores_and_targets round-trips a measure" begin
        m = make_measure()
        scores, target_w = annealed_smc_scores_and_targets(m)
        @test length(scores) == 2
        @test Set(scores) == m.scores
        # target weights are aligned to `scores` order
        for (e, w) in zip(scores, target_w)
            @test w == m.weights[e]
        end
        # configure_measure! at t=1 reproduces the original target weights
        work = deepcopy(m)
        path = CycleWalk.linear_path(target_w)
        CycleWalk.configure_measure!(path, work, scores, 1.0)
        @test all(work.weights[e] ≈ m.weights[e] for e in scores)
        CycleWalk.configure_measure!(path, work, scores, 0.0)
        @test all(work.weights[e] ≈ 0.0 for e in scores)
    end

    # --------------------------------------------------------------- schedules

    @testset "FixedSchedule walks the grid and terminates" begin
        sched = FixedSchedule([0.0, 0.5, 1.0]; ess_frac=0.5)
        @test !CycleWalk.done(sched)
        @test CycleWalk.next_t(sched, nothing, nothing, 0.0) ≈ 0.5
        @test !CycleWalk.done(sched)
        @test CycleWalk.next_t(sched, nothing, nothing, 0.5) ≈ 1.0
        @test CycleWalk.done(sched)
        # resample decision fires iff ess < ess_frac * N
        @test CycleWalk.should_resample(sched, 3.9, 8)      # 3.9 < 0.5*8
        @test !CycleWalk.should_resample(sched, 4.1, 8)
    end

    @testset "AdaptiveTempering.next_t hits the ESS target" begin
        N = 8
        st = fresh_partition(505)
        # K = 1 path with weights_at(t) = (t,), so energy_at(p,t) = t * phi
        path = CycleWalk.LinearPath{1}(t -> (t,))

        # identical potentials => ESS stays N for all t => reach t=1 in one jump
        eqparts = [CycleWalk.Particle{1}(st, 0.0, (2.0,),
                       PCG.PCGStateOneseq(UInt64, UInt64(i))) for i in 1:N]
        sched = AdaptiveTempering(; ess_target=0.5)
        @test CycleWalk.next_t(sched, eqparts, path, 0.0) ≈ 1.0

        # spread potentials so ESS at t=1 is far below the target => interior t
        xs = Float64.(1:N)
        parts = [CycleWalk.Particle{1}(st, 0.0, (xs[i],),
                     PCG.PCGStateOneseq(UInt64, UInt64(i))) for i in 1:N]
        sched2 = AdaptiveTempering(; ess_target=0.5)
        t_star = CycleWalk.next_t(sched2, parts, path, 0.0)
        @test 0.0 < t_star < 1.0
        # the chosen t reweights the population to (about) the target ESS
        ess_at_t = CycleWalk.ess_from_logw(t_star .* xs)
        @test ess_at_t ≈ 0.5 * N atol = 0.1
        # coupled variant always resamples
        @test CycleWalk.should_resample(sched2, 4.0, N)
    end

    # ------------------------------------------------------------------- driver

    @testset "deterministic logZ accounting (no rejuvenation, no resampling)" begin
        # With rejuv_steps=0 and init_steps=0 every particle is a frozen clone of
        # the initial partition, so phi never changes and the incremental weights
        # telescope: each particle's final logW == <target_w, phi> == the target
        # log-energy, and logZ == that same value. ess_frac=0 disables resampling.
        partition = fresh_partition(606)
        measure = make_measure()
        rng = PCG.PCGStateOneseq(UInt64, 606)
        sched = FixedSchedule(range(0.0, 1.0; length=4); ess_frac=0.0)
        particles, logZ, trace = run_annealed_smc!(
            partition, proposal, measure, sched, 8, 0, rng; init_steps=0)

        expected = get_log_energy(partition, measure)
        @test logZ ≈ expected
        @test all(p -> isapprox(p.logW, expected), particles)
        @test length(particles) == 8
        @test trace[end].t ≈ 1.0
        @test !any(r -> r.resampled, trace)
    end

    @testset "zero path => zero weights and logZ == 0" begin
        # a path that is identically zero injects no incremental weight anywhere
        partition = fresh_partition(707)
        measure = make_measure()
        K = length(measure.scores)
        rng = PCG.PCGStateOneseq(UInt64, 707)
        sched = FixedSchedule(range(0.0, 1.0; length=5); ess_frac=0.5)
        particles, logZ, trace = run_annealed_smc!(
            partition, proposal, measure, sched, 8, 5, rng;
            path = (t -> ntuple(_ -> 0.0, K)))
        @test logZ ≈ 0.0 atol = 1e-9
        @test all(p -> isapprox(p.logW, 0.0; atol=1e-9), particles)
        @test !any(r -> r.resampled, trace)   # ESS stays at N with equal weights
    end

    @testset "caller's measure is left unmodified" begin
        partition = fresh_partition(808)
        measure = make_measure()
        weights_before = deepcopy(measure.weights)
        rng = PCG.PCGStateOneseq(UInt64, 808)
        sched = FixedSchedule(range(0.0, 1.0; length=6); ess_frac=0.5)
        run_annealed_smc!(partition, proposal, measure, sched, 8, 5, rng;
                          init_steps=5)
        @test measure.weights == weights_before
    end

    @testset "reproducibility: same seed => identical logZ and weights" begin
        function annealed_smc_run()
            partition = fresh_partition(909)
            measure = make_measure()
            rng = PCG.PCGStateOneseq(UInt64, 909)
            sched = FixedSchedule(range(0.0, 1.0; length=6); ess_frac=0.5)
            particles, logZ, _ = run_annealed_smc!(
                partition, proposal, measure, sched, 8, 5, rng; init_steps=5)
            return logZ, [p.logW for p in particles]
        end
        logZ1, w1 = annealed_smc_run()
        logZ2, w2 = annealed_smc_run()
        @test logZ1 == logZ2
        @test w1 == w2
    end

    @testset "writer records one map per particle with logW weights" begin
        AIO = CycleWalk.AtlasIO
        partition = fresh_partition(1010)
        measure = make_measure()
        rng = PCG.PCGStateOneseq(UInt64, 1010)
        sched = FixedSchedule(range(0.0, 1.0; length=6); ess_frac=0.5)
        n_particles = 8
        mktempdir() do tmpdir
            output_path = joinpath(tmpdir, "annealed_smc_output.jsonl.gz")
            writer = Writer(measure, constraints, partition, output_path;
                            weight_type=Float64)
            push_writer!(writer, get_isoperimetric_score)
            particles, _, _ = run_annealed_smc!(
                partition, proposal, measure, sched, n_particles, 5, rng;
                init_steps=5, writer=writer)
            close_writer(writer)

            io = AIO.smartOpen(output_path, "r")
            maps = AIO.nextMaps(AIO.openAtlas(io))
            close(io)
            @test length(maps) == n_particles
            @test [m.name for m in maps] ==
                  ["particle" * string(i) for i in 1:n_particles]
            @test [m.weight for m in maps] ≈ [p.logW for p in particles]
        end
    end
end
