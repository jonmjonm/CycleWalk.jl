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

    @testset "incremental_logweight! is the log-density ratio (sign)" begin
        # The SMC increment is log[ν_t(x)/ν_{t_prev}(x)]. `energy_at` returns an
        # ENERGY and ν ∝ exp(−E), so the increment is E(t_prev) − E(t). Asserting it
        # against `energy_at(t) - energy_at(t_prev)` (as this test previously did)
        # merely restates the implementation and cannot catch a flipped sign.
        target_w = (gamma, iso)
        path = CycleWalk.linear_path(target_w)
        st = fresh_partition(404)
        phis = [(1.0, 0.5), (-2.0, 3.0), (0.25, -0.75)]
        mkparts() = [CycleWalk.Particle{2}(CycleWalk.clone_for_annealing(st), 0.0,
                                           phi, PCG.PCGStateOneseq(UInt64, UInt64(i)))
                     for (i, phi) in enumerate(phis)]
        t_prev, t = 0.2, 0.7

        parts = mkparts()
        expected = [CycleWalk.energy_at(path, p, t_prev) -
                    CycleWalk.energy_at(path, p, t) for p in parts]
        CycleWalk.incremental_logweight!(parts, path, t_prev, t)
        @test [p.logW for p in parts] ≈ expected

        # Independent derivation from the definition, not from `energy_at`: for a
        # linear path the increment is ⟨w(t_prev) − w(t), φ⟩ = (t_prev − t)·⟨w_target, φ⟩.
        @test [p.logW for p in parts] ≈
              [(t_prev - t) * (target_w[1]*phi[1] + target_w[2]*phi[2]) for phi in phis]

        # The generic AnnealPath method and the LinearPath fast path must agree; a
        # sign error in only one of the two would otherwise slip through.
        generic = mkparts()
        invoke(CycleWalk.incremental_logweight!,
               Tuple{Vector{CycleWalk.Particle{2}}, CycleWalk.AnnealPath,
                     Float64, Float64},
               generic, path, t_prev, t)
        @test [p.logW for p in generic] ≈ [p.logW for p in parts]

        # Direction: a particle with positive energy is penalised as t grows, so its
        # weight must fall. (phis[1] is positive in both coordinates.)
        @test parts[1].logW < 0
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
        # telescope: each particle's final logW == −<target_w, phi> == MINUS the
        # target log-energy (ν ∝ exp(−E)), and logZ == that same value. ess_frac=0
        # disables resampling.
        partition = fresh_partition(606)
        measure = make_measure()
        rng = PCG.PCGStateOneseq(UInt64, 606)
        sched = FixedSchedule(range(0.0, 1.0; length=4); ess_frac=0.0)
        particles, logZ, trace = run_annealed_smc!(
            partition, proposal, measure, sched, 8, 0, rng; init_steps=0)

        expected = -get_log_energy(partition, measure)
        @test logZ ≈ expected
        @test all(p -> isapprox(p.logW, expected), particles)
        @test length(particles) == 8
        @test trace[end].t ≈ 1.0
        @test !any(r -> r.resampled, trace)
    end

    # ------------------------------------------------------------------------
    # Sign / correctness regression -- see the matching block in
    # test_annealed_importance_sampling.jl for the derivation of the exact answer.
    # Annealing gamma from 0 to 1 on the 4x4 graph must give
    #     logZ = log(Z(1)/Z(0)) = log(117/654) = -1.72093.
    #
    # This matters more for ASMC than for AIS: the log weights also drive resampling
    # and the ESS test, so an inverted sign kills the wrong particles and corrupts
    # the population itself -- it cannot be repaired by negating logZ afterwards.
    # Measured spread over 5 seeds is <= 0.09; a flipped sign lands ~3.6 away.
    # ------------------------------------------------------------------------
    @testset "logZ matches the exactly known log Z(1)/Z(0)" begin
        exact_logZ = log(117 / 654)

        # the hard p88 population constraint (min=max=4) the enumeration assumes
        p88_constraints = initialize_constraints()
        add_constraint!(p88_constraints, PopulationConstraint(4, 4))
        p88_proposal = [(0.1, build_lifted_tree_cycle_walk(p88_constraints)),
                        (0.9, build_internal_forest_walk(p88_constraints))]

        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, 1.0)
        scores, target_w = annealed_smc_scores_and_targets(measure)
        K = length(scores)
        path = LinearPath{K}(t -> ntuple(k -> t * target_w[k], K))

        rng = PCG.PCGStateOneseq(UInt64, 20260731)
        partition = LinkCutPartition(annealed_smc_graph, p88_constraints, 4; rng=rng)
        sched = FixedSchedule(range(0, 1; length=41); ess_frac=0.5)
        _, logZ, trace = run_annealed_smc!(partition, p88_proposal, measure, sched,
                                           64, 200, rng; path=path, init_steps=1000)

        @test logZ ≈ exact_logZ atol = 0.5
        # Z(1) < Z(0): the estimate must be negative regardless of tolerance.
        @test logZ < 0
        # the population must not have collapsed -- if it did, the logZ agreement
        # above would be luck rather than a working sampler
        @test minimum(rc.ess for rc in trace) > 2.0
    end

    @testset "final particles are distributed as the target" begin
        # The logZ test above pins the NORMALIZER; this pins the SAMPLES. At t=1 the
        # weighted particle population must match the enumerated gamma=1 cut-edge
        # distribution {8=>1, 10=>14, 11=>24, 12=>78}/117. Amplify with collect_steps
        # so the histogram has enough draws to be worth comparing.
        p88_constraints = initialize_constraints()
        add_constraint!(p88_constraints, PopulationConstraint(4, 4))
        p88_proposal = [(0.1, build_lifted_tree_cycle_walk(p88_constraints)),
                        (0.9, build_internal_forest_walk(p88_constraints))]
        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, 1.0)
        scores, target_w = annealed_smc_scores_and_targets(measure)
        K = length(scores)
        path = LinearPath{K}(t -> ntuple(k -> t * target_w[k], K))

        rng = PCG.PCGStateOneseq(UInt64, 31337)
        partition = LinkCutPartition(annealed_smc_graph, p88_constraints, 4; rng=rng)
        sched = FixedSchedule(range(0, 1; length=41); ess_frac=0.5)
        particles, _, _ = run_annealed_smc!(partition, p88_proposal, measure, sched,
                                            64, 200, rng; path=path, init_steps=1000,
                                            collect_steps=0)

        # weight each surviving particle by exp(logW) and histogram its cut edges
        lw = [p.logW for p in particles]
        mx = maximum(lw)
        w  = exp.(lw .- mx)
        acc = Dict{Int, Float64}()
        for (p, wi) in zip(particles, w)
            ce = get_cut_edge_sum(p.state, column="connections")
            acc[ce] = get(acc, ce, 0.0) + wi
        end
        tot = sum(values(acc))
        truth = Dict(8 => 1/117, 10 => 14/117, 11 => 24/117, 12 => 78/117)
        # every observed cut count must be one the enumeration knows about
        @test issubset(Set(keys(acc)), Set(keys(truth)))
        # 64 particles is a small histogram, so this is a loose shape check: the
        # 12-cut plans carry 2/3 of the mass and the 8-cut plan almost none.
        @test get(acc, 12, 0.0)/tot > 0.4
        @test get(acc, 8, 0.0)/tot < 0.10
        l1 = sum(abs(get(acc, k, 0.0)/tot - truth[k]) for k in keys(truth))
        @test l1 < 0.35
    end

    @testset "schedules are reusable across runs" begin
        # Schedules carry a cursor and finish a run exhausted. Before reset_schedule!
        # the second run with the same object did zero blocks and returned logZ=0.0 --
        # silently, with no error and an empty trace.
        p88_constraints = initialize_constraints()
        add_constraint!(p88_constraints, PopulationConstraint(4, 4))
        p88_proposal = [(0.1, build_lifted_tree_cycle_walk(p88_constraints)),
                        (0.9, build_internal_forest_walk(p88_constraints))]
        measure = Measure()
        push_energy!(measure, get_log_spanning_forests, 1.0)
        sched = FixedSchedule(range(0, 1; length=11); ess_frac=0.5)

        results = map(1:2) do _
            rng = PCG.PCGStateOneseq(UInt64, 4242)
            part = LinkCutPartition(annealed_smc_graph, p88_constraints, 4; rng=rng)
            _, logZ, trace = run_annealed_smc!(part, p88_proposal, measure, sched,
                                               8, 20, rng; init_steps=100)
            (logZ, length(trace))
        end
        # same seed and a reset schedule => the second run must reproduce the first
        @test results[1][2] == results[2][2] == 10
        @test results[1][1] ≈ results[2][1]
        @test results[2][1] != 0.0

        # reset_schedule! is idempotent and returns the schedule
        @test CycleWalk.reset_schedule!(sched) === sched
        @test sched.k == 0
        adaptive = AdaptiveTempering(; ess_target=0.5)
        adaptive.t_prev = 1.0
        @test CycleWalk.reset_schedule!(adaptive).t_prev == 0.0
    end

    @testset "AdaptiveTempering rejects init_steps=0" begin
        # With no burn-in every particle is an identical clone, ESS is identically N,
        # and the bisection jumps straight to t=1 on that false signal.
        measure = make_measure()
        rng = PCG.PCGStateOneseq(UInt64, 5150)
        partition = fresh_partition(5150)
        @test_throws ArgumentError run_annealed_smc!(
            partition, proposal, measure, AdaptiveTempering(; ess_target=0.5),
            8, 10, rng; init_steps=0)
        # FixedSchedule is merely inadvisable there, not rejected
        sched = FixedSchedule(range(0, 1; length=4); ess_frac=0.5)
        @test run_annealed_smc!(partition, proposal, measure, sched, 4, 5, rng;
                                init_steps=0) isa Tuple
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
