using CycleWalk
using RandomNumbers
using Test

# Tests for parallel tempering (src/chains/parallel_tempering*.jl). Follows
# test_annealed_smc.jl's structure: pure-helper unit tests first, then driver
# invariants, then ground truth. Internal (unexported) helpers are reached through
# the CycleWalk namespace. See memories/parallel_tempering_implementation.md §7.

@testset verbose = true "parallel_tempering" begin
    pt_testdir = dirname(@__FILE__)
    pt_json = joinpath(pt_testdir, "test_graphs", "4x4pct_2x2cnty.json")
    pt_node_data = Set(["county", "pct", "pop", "area", "border_length"])
    pt_base_graph = BaseGraph(pt_json, "pop",
                              inc_node_data=pt_node_data,
                              area_col="area",
                              node_border_col="border_length",
                              edge_perimeter_col="length")
    pt_graph = MultiLevelGraph(pt_base_graph, ["pct"])

    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4, 4))
    proposal = [(0.1, build_lifted_tree_cycle_walk(constraints)),
                (0.9, build_internal_forest_walk(constraints))]

    # ---------------------------------------------------------------- BetaLattice

    @testset "BetaLattice validation" begin
        @test CycleWalk.BetaLattice([0.0, 0.5, 1.0]) isa CycleWalk.BetaLattice
        @test_throws ArgumentError CycleWalk.BetaLattice([0.0, 1.0, 0.5])   # unsorted
        @test_throws ArgumentError CycleWalk.BetaLattice([0.0, 0.5, 0.5, 1.0]) # duplicate
        @test_throws ArgumentError CycleWalk.BetaLattice([-0.1, 0.5, 1.0])  # negative
        @test_throws ArgumentError CycleWalk.BetaLattice([0.0, 0.5, 0.9])   # end != 1
        @test_throws ArgumentError CycleWalk.BetaLattice(Float64[])         # empty
    end

    @testset "linear_betas / geometric_betas" begin
        lb = linear_betas(5)
        @test lb.betas == [0.0, 0.25, 0.5, 0.75, 1.0]
        @test length(lb) == 5
        @test lb[1] == 0.0 && lb[5] == 1.0

        gb = geometric_betas(5; beta_min=0.05)
        @test length(gb) == 5
        @test gb.betas[end] == 1.0
        @test issorted(gb.betas)
        @test gb.betas[1] ≈ 0.05

        # geometric_betas reproduces the Java ladder for matching endpoints: that
        # ladder is built cold-to-hot (temperature low -> high), ours hot-to-cold
        # (temperature high -> low), so reverse one side before comparing.
        temps_ours = 1 ./ gb.betas                       # hot -> cold  (high T -> low T)
        java_style_cold_to_hot = reverse(temps_ours)      # cold -> hot (low T -> high T)
        @test issorted(java_style_cold_to_hot)
        @test java_style_cold_to_hot[1] ≈ 1.0
        @test java_style_cold_to_hot[end] ≈ 1/0.05
    end

    # ---------------------------------------------------------------- PTDiagnostics

    @testset "PTDiagnostics reset" begin
        d = CycleWalk.PTDiagnostics(4)
        @test length(d.attempts) == 3
        @test size(d.occupancy) == (4, 4)

        d.attempts .= [1, 2, 3]
        d.accepts .= [1, 1, 1]
        d.accept_prob_sum .= [0.5, 0.5, 0.5]
        d.occupancy[1, 1] = 7
        d.round_trips .= [1, 0, 0, 2]
        d.bath_attempts = 3
        d.bath_accepts = 2
        push!(d.straggler_gap, 0.01)

        CycleWalk.reset_pt_diagnostics!(d)

        @test all(iszero, d.attempts)
        @test all(iszero, d.accepts)
        @test all(iszero, d.accept_prob_sum)
        @test all(iszero, d.occupancy)
        @test all(iszero, d.round_trips)
        @test d.bath_attempts == 0
        @test d.bath_accepts == 0
        @test isempty(d.straggler_gap)
    end

    @testset "swap_rate" begin
        d = CycleWalk.PTDiagnostics(3)
        d.attempts .= [4, 0]
        d.accepts .= [2, 0]
        rates = swap_rate(d)
        @test rates[1] ≈ 0.5
        @test isnan(rates[2])
    end

    # ---------------------------------------------------------------- swap mechanics

    fresh_pt_partition(seed) =
        LinkCutPartition(pt_graph, constraints, 4; rng=PCG.PCGStateOneseq(UInt64, seed))

    pt_scores = (get_log_spanning_forests, get_isoperimetric_score)
    make_pt_measure() = begin
        m = Measure()
        push_energy!(m, get_log_spanning_forests, 0.7)
        push_energy!(m, get_isoperimetric_score, 0.3)
        m
    end

    mkreplica(k::Int, phi::NTuple{2,Float64}) =
        CycleWalk.Replica{2}(fresh_pt_partition(1000 + k), k, phi,
                             PCG.PCGStateOneseq(UInt64, UInt64(1000 + k)),
                             RunDiagnostics(), make_pt_measure(), k, 0, 0)

    function mkensemble(phis::Vector{NTuple{2,Float64}}, path)
        M = length(phis)
        replicas = [mkreplica(k, phis[k]) for k in 1:M]
        return CycleWalk.PTEnsemble{2}(replicas, collect(1:M), linear_betas(M),
                                       path, pt_scores)
    end

    @testset "even_odd_pairs / idle_rungs" begin
        # alternation
        @test CycleWalk.even_odd_pairs(6, 1) == [(1,2), (3,4), (5,6)]
        @test CycleWalk.even_odd_pairs(6, 2) == [(2,3), (4,5)]
        @test CycleWalk.even_odd_pairs(6, 3) == [(1,2), (3,4), (5,6)]   # odd again

        for M in (4, 5, 6, 7)
            for round in (1, 2)
                pairs = CycleWalk.even_odd_pairs(M, round)
                # disjoint
                flat = collect(Iterators.flatten(pairs))
                @test length(flat) == length(unique(flat))
                idle = CycleWalk.idle_rungs(M, round)
                # no idle rung appears in any pair
                @test isempty(intersect(idle, flat))
                # pairs ∪ idle covers 1:M
                @test sort(vcat(flat, idle)) == collect(1:M)
            end
            # over two consecutive rounds every adjacent pair (k,k+1) appears exactly once
            all_pairs = vcat(CycleWalk.even_odd_pairs(M, 1), CycleWalk.even_odd_pairs(M, 2))
            @test sort(all_pairs) == [(k, k+1) for k in 1:(M-1)]
        end

        # hand-checked idle sets from the docstring
        @test CycleWalk.idle_rungs(4, 1) == []
        @test CycleWalk.idle_rungs(4, 2) == [1, 4]
        @test CycleWalk.idle_rungs(5, 1) == [5]
        @test CycleWalk.idle_rungs(5, 2) == [1]
    end

    @testset "swap_logratio" begin
        target_w = (0.7, 0.3)
        path = linear_path(target_w)
        beta_i, beta_j = 0.2, 0.8
        phi_i, phi_j = (2.0, -1.0), (-0.5, 3.0)

        logα = CycleWalk.swap_logratio(path, beta_i, beta_j, phi_i, phi_j)

        # cross-check 1: for a pure β-ladder this collapses to (β_i-β_j)*(E_i-E_j)
        E(phi) = target_w[1]*phi[1] + target_w[2]*phi[2]
        @test logα ≈ (beta_i - beta_j) * (E(phi_i) - E(phi_j))

        # cross-check 2 (direction): hotter rung (smaller β) holding the higher-energy
        # state must be discouraged from swapping (logα < 0).
        beta_hot, beta_cold = 0.1, 0.9
        phi_hot_highE, phi_cold_lowE = (10.0, 10.0), (0.0, 0.0)   # E(hot) > E(cold)
        logα_dir = CycleWalk.swap_logratio(path, beta_hot, beta_cold,
                                           phi_hot_highE, phi_cold_lowE)
        @test logα_dir < 0
    end

    @testset "swap_round! degenerate cases" begin
        target_w = (0.7, 0.3)
        path = linear_path(target_w)
        rng = PCG.PCGStateOneseq(UInt64, 999)

        # "all beta equal" => every rung's weight vector is identical => logα == 0 for
        # every pair => acceptance probability 1. BetaLattice requires distinct rungs
        # (§7.1 test 8), so realize "no separation along the path" with a CONSTANT path
        # (weights_at same at every t) on an ordinary strictly-increasing lattice —
        # physically identical to a degenerate all-equal ladder for this swap-only test.
        @testset "constant path (weights equal at every rung)" begin
            M = 4
            phis = [(1.0, 2.0), (3.0, -1.0), (-2.0, 0.5), (0.0, 4.0)]
            const_path = CycleWalk.LinearPath{2}(t -> (0.0, 0.0))
            ensemble = mkensemble(phis, const_path)
            diag = CycleWalk.PTDiagnostics(M)
            pairs = CycleWalk.even_odd_pairs(M, 1)
            accepted = CycleWalk.swap_round!(ensemble, 1, rng, diag)
            @test accepted == length(pairs)
            @test all(diag.accept_prob_sum[i] ≈ 1.0 for (i, _) in pairs)
            @test CycleWalk.check_ensemble(ensemble)
        end

        # identical phi => logα == 0 for every pair regardless of beta => acceptance 1
        @testset "identical phi" begin
            M = 4
            phi = (1.0, 2.0)
            ensemble = mkensemble(fill(phi, M), path)
            diag = CycleWalk.PTDiagnostics(M)
            pairs = CycleWalk.even_odd_pairs(M, 1)
            accepted = CycleWalk.swap_round!(ensemble, 1, rng, diag)
            @test accepted == length(pairs)
            @test all(diag.accept_prob_sum[i] ≈ 1.0 for (i, _) in pairs)
            @test CycleWalk.check_ensemble(ensemble)
        end
    end

    @testset "record_round!" begin
        M = 3
        phis = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
        target_w = (0.7, 0.3)
        path = linear_path(target_w)
        ensemble = mkensemble(phis, path)
        diag = CycleWalk.PTDiagnostics(M)

        # walker 1 travels M -> 1 -> M by hand, driving beta_index directly
        r = ensemble.replicas[1]

        r.beta_index = M; CycleWalk.record_round!(ensemble, diag)   # start at the cold end
        @test diag.round_trips[1] == 0

        r.beta_index = 1; CycleWalk.record_round!(ensemble, diag)   # visits the hot end
        @test diag.round_trips[1] == 0

        r.beta_index = M; CycleWalk.record_round!(ensemble, diag)   # back to cold: one round trip
        @test diag.round_trips[1] == 1

        # occupancy accumulated one block per replica per round recorded
        @test diag.occupancy[1, M] == 2
        @test diag.occupancy[1, 1] == 1
    end

    # ---------------------------------------------------------------- serial driver

    p88_constraints = initialize_constraints()
    add_constraint!(p88_constraints, PopulationConstraint(4, 4))
    p88_proposal = [(0.1, build_lifted_tree_cycle_walk(p88_constraints)),
                    (0.9, build_internal_forest_walk(p88_constraints))]
    p88_measure = Measure()
    push_energy!(p88_measure, get_log_spanning_forests, 1.0)

    AIO = CycleWalk.AtlasIO
    read_maps(path) = begin
        io = AIO.smartOpen(path, "r")
        atlas = AIO.openAtlas(io)
        maps = AIO.Map[]
        while !eof(io)
            push!(maps, AIO.nextMap(atlas))
        end
        close(io)
        maps
    end
    cut_edge_histogram(maps) = begin
        acc = Dict{Int,Int}()
        for m in maps
            ce = m.data["get_cut_edge_sum"]
            acc[ce] = get(acc, ce, 0) + 1
        end
        acc
    end
    l1_distance(hist, truth) = begin
        tot = sum(values(hist))
        sum(abs(get(hist, k, 0)/tot - v) for (k, v) in truth)
    end
    # γ=0 (uniform on balanced spanning forests): {8,10,11,12} / 654
    truth_hot = Dict(8 => 256/654, 10 => 224/654, 11 => 96/654, 12 => 78/654)
    # γ=1 (the configured target): {8,10,11,12} / 117
    truth_target = Dict(8 => 1/117, 10 => 14/117, 11 => 24/117, 12 => 78/117)

    @testset "M=1 reduces to run_metropolis_hastings!" begin
        swap_interval = 37
        n_rounds = 5
        total_steps = swap_interval * n_rounds
        lattice = CycleWalk.BetaLattice([1.0])
        seed = 4242

        # The driver draws exactly one `rand(rng, UInt64)` (to seed replica 1) before
        # any MH step, since M=1 has no swap pairs to consume the driver rng. Mimic
        # that same single draw on an identically-seeded independent rng to derive
        # the seed a direct run_metropolis_hastings! call must use to match bit for bit.
        driver_rng = PCG.PCGStateOneseq(UInt64, seed)
        mimic_rng  = PCG.PCGStateOneseq(UInt64, seed)
        replica_seed = rand(mimic_rng, UInt64)
        direct_rng = PCG.PCGStateOneseq(UInt64, replica_seed)

        p_direct = fresh_pt_partition(9999)
        run_metropolis_hastings!(p_direct, p88_proposal, p88_measure, total_steps,
                                 direct_rng; output_initial=false)

        p_driver = fresh_pt_partition(9999)
        ensemble, diag = run_parallel_tempering!(p_driver, p88_proposal, p88_measure,
                                                 lattice, swap_interval, n_rounds,
                                                 driver_rng)

        @test ensemble.replicas[1].state.node_to_dist == p_direct.node_to_dist
        @test ensemble.replicas[1].state.identifier == p_direct.identifier
        # n_rounds * swap_interval total steps were taken: the direct call's identifier
        # (accepted-move count) is itself bounded by total_steps, and since both runs
        # draw from bit-identical RNG streams (checked above) they accepted the same
        # moves at the same steps, so this is exactly the driver's step count too.
        @test p_direct.identifier <= total_steps
    end

    @testset "emit_maps!: write_rungs modes and pt/ keys" begin
        M = 4
        lattice = linear_betas(M)
        swap_interval = 5
        n_rounds = 6

        @testset ":target writes only rung M, with all five pt/ keys" begin
            rng = PCG.PCGStateOneseq(UInt64, 555)
            partition = LinkCutPartition(pt_graph, p88_constraints, 4; rng=rng)
            mktempdir() do dir
                path = joinpath(dir, "target.jsonl.gz")
                w = Writer(p88_measure, p88_constraints, partition, path)
                run_parallel_tempering!(partition, p88_proposal, p88_measure, lattice,
                                        swap_interval, n_rounds, rng;
                                        write_rungs=:target, writers=w)
                close_writer(w)

                maps = read_maps(path)
                @test length(maps) == n_rounds
                for m in maps
                    for key in ("pt/replica_id", "pt/beta_index", "pt/beta",
                               "pt/bath_swaps", "pt/round")
                        @test haskey(m.data, key)
                    end
                    @test m.data["pt/beta_index"] == M
                    @test m.data["pt/beta"] == lattice[M]
                end
                @test [m.data["pt/round"] for m in maps] == collect(1:n_rounds)
            end
        end

        @testset ":all writes one file per rung, indexed correctly" begin
            rng = PCG.PCGStateOneseq(UInt64, 777)
            partition = LinkCutPartition(pt_graph, p88_constraints, 4; rng=rng)
            mktempdir() do dir
                writers = Writer[]
                for k in 1:M
                    w = Writer(p88_measure, p88_constraints, partition,
                              joinpath(dir, "beta$(k).jsonl.gz"))
                    push!(writers, w)
                end
                run_parallel_tempering!(partition, p88_proposal, p88_measure, lattice,
                                        swap_interval, n_rounds, rng;
                                        write_rungs=:all, writers=writers)
                foreach(close_writer, writers)

                for k in 1:M
                    maps = read_maps(joinpath(dir, "beta$(k).jsonl.gz"))
                    @test length(maps) == n_rounds
                    @test all(m -> m.data["pt/beta_index"] == k, maps)
                    @test all(m -> m.data["pt/beta"] == lattice[k], maps)
                end
            end
        end

        @testset ":none writes nothing" begin
            rng = PCG.PCGStateOneseq(UInt64, 888)
            partition = LinkCutPartition(pt_graph, p88_constraints, 4; rng=rng)
            ensemble, diag = run_parallel_tempering!(partition, p88_proposal,
                                                     p88_measure, lattice,
                                                     swap_interval, n_rounds, rng;
                                                     write_rungs=:none, writers=nothing)
            @test ensemble isa CycleWalk.PTEnsemble
        end
    end

    @testset "ground truth: target and hot rung distributions" begin
        M = 6
        lattice = linear_betas(M)
        swap_interval = 40
        n_rounds = 300
        rng = PCG.PCGStateOneseq(UInt64, 24601)
        partition = LinkCutPartition(pt_graph, p88_constraints, 4; rng=rng)

        mktempdir() do dir
            writers = Writer[]
            for k in 1:M
                path = joinpath(dir, "rung$(k).jsonl.gz")
                w = Writer(p88_measure, p88_constraints, partition, path)
                push_writer!(w, get_cut_edge_sum)
                push!(writers, w)
            end

            run_parallel_tempering!(partition, p88_proposal, p88_measure, lattice,
                                    swap_interval, n_rounds, rng;
                                    init_steps=500, write_rungs=:all, writers=writers)
            foreach(close_writer, writers)

            target_hist = cut_edge_histogram(read_maps(joinpath(dir, "rung$(M).jsonl.gz")))
            hot_hist    = cut_edge_histogram(read_maps(joinpath(dir, "rung1.jsonl.gz")))

            @test issubset(Set(keys(target_hist)), Set(keys(truth_target)))
            @test l1_distance(target_hist, truth_target) < 0.25

            @test issubset(Set(keys(hot_hist)), Set(keys(truth_hot)))
            @test l1_distance(hot_hist, truth_hot) < 0.25
        end
    end

    @testset "degenerate ladder (all rungs ~= target) is independent chains" begin
        # BetaLattice forbids duplicate rungs (§7.1 test 8), so realize "M rungs, same
        # point" with a strictly-increasing but numerically-indistinguishable-from-1.0
        # ladder: swap_logratio is then ~0 for every pair (=> acceptance ~1, verified
        # below), so each rung is effectively an independent chain at the target.
        M = 4
        degenerate_betas = [nextfloat(1.0, -(M - k)) for k in 1:M]
        lattice = CycleWalk.BetaLattice(degenerate_betas)
        swap_interval = 40
        n_rounds = 200
        rng = PCG.PCGStateOneseq(UInt64, 13131)
        partition = LinkCutPartition(pt_graph, p88_constraints, 4; rng=rng)

        mktempdir() do dir
            writers = Writer[]
            for k in 1:M
                path = joinpath(dir, "rung$(k).jsonl.gz")
                w = Writer(p88_measure, p88_constraints, partition, path)
                push_writer!(w, get_cut_edge_sum)
                push!(writers, w)
            end

            diag = CycleWalk.PTDiagnostics(M)
            run_parallel_tempering!(partition, p88_proposal, p88_measure, lattice,
                                    swap_interval, n_rounds, rng;
                                    init_steps=500, write_rungs=:all, writers=writers,
                                    diagnostics=diag)
            foreach(close_writer, writers)

            # swap acceptance is identically ~1 (logα ~ 0 to within float precision)
            @test all(r -> isapprox(r, 1.0; atol=1e-6), swap_rate(diag))

            # each rung's marginal must still match the γ=1 column; combine all M
            # rungs' maps for a single, better-powered check of that shared claim.
            all_maps = vcat((read_maps(joinpath(dir, "rung$(k).jsonl.gz")) for k in 1:M)...)
            combined_hist = cut_edge_histogram(all_maps)
            @test issubset(Set(keys(combined_hist)), Set(keys(truth_target)))
            @test l1_distance(combined_hist, truth_target) < 0.25
        end
    end

    # ---------------------------------------------------------------- threaded backend

    @testset "ThreadedBackend reproduces SerialBackend bitwise" begin
        # Thread count is fixed for the life of a Julia process, so the only way to
        # exercise both -t1 and -t4 in one test run is a subprocess per count (§7.1
        # test 5). The subprocess script prints "EQUIVALENT" and exits 0 on a match.
        script = joinpath(pt_testdir, "pt_backend_equivalence_check.jl")
        project = Base.active_project()
        for nthreads in (1, 4)
            cmd = `$(Base.julia_cmd()) --project=$(project) --threads=$(nthreads) $(script)`
            io = IOBuffer()
            proc = run(pipeline(cmd; stdout=io, stderr=io); wait=true)
            output = String(take!(io))
            @test success(proc)
            @test occursin("EQUIVALENT", output) && !occursin("NOT EQUIVALENT", output)
        end
    end

    # ---------------------------------------------------------------- heat bath

    @testset "heat bath" begin
        target_scores = (get_log_spanning_forests,)

        @testset "matched bath: high acceptance, rung 1 marginal unchanged" begin
            mktempdir() do dir
                # Bath source: zero-weight (uniform, γ=0) sampling, config embedded so
                # parse_bath_measure can rebuild the SAME zero-weight measure.
                toml_path = joinpath(dir, "bath.toml")
                write(toml_path, "[measure]\n")

                bath_rng = PCG.PCGStateOneseq(UInt64, 909)
                bath_partition = LinkCutPartition(pt_graph, p88_constraints, 4; rng=bath_rng)
                bath_measure = Measure()
                bath_path = joinpath(dir, "bath.jsonl.gz")
                bath_writer = Writer(bath_measure, p88_constraints, bath_partition,
                                     bath_path; config_file=toml_path)
                run_metropolis_hastings!(bath_partition, p88_proposal, bath_measure,
                                         60000, bath_rng; writer=bath_writer,
                                         output_freq=100)
                close_writer(bath_writer)

                hb = HeatBath(bath_path, target_scores, pt_graph; rung=1, burn_in=100,
                             n_samples=200, rng=PCG.PCGStateOneseq(UInt64, 111))
                @test hb.rung == 1
                @test length(hb.samples) == 200
                @test isempty(Set(keys(hb.measure.weights)))   # zero-weight bath measure

                M = 4
                lattice = linear_betas(M)
                pt_rng = PCG.PCGStateOneseq(UInt64, 222)
                partition = LinkCutPartition(pt_graph, p88_constraints, 4; rng=pt_rng)
                diag = CycleWalk.PTDiagnostics(M)

                writers = Writer[]
                for k in 1:M
                    w = Writer(p88_measure, p88_constraints, partition,
                              joinpath(dir, "beta$(k).jsonl.gz"))
                    push_writer!(w, get_cut_edge_sum)
                    push!(writers, w)
                end

                run_parallel_tempering!(partition, p88_proposal, p88_measure, lattice,
                                        40, 300, pt_rng; init_steps=500, heat_bath=hb,
                                        write_rungs=:all, writers=writers,
                                        diagnostics=diag)
                foreach(close_writer, writers)

                # matched bath (zero weight vs. rung 1's own zero weight) -> logα == 0
                # for every attempt -> deterministic acceptance
                @test diag.bath_attempts > 0
                @test diag.bath_accepts == diag.bath_attempts

                hot_hist = cut_edge_histogram(read_maps(joinpath(dir, "beta1.jsonl.gz")))
                @test issubset(Set(keys(hot_hist)), Set(keys(truth_hot)))
                @test l1_distance(hot_hist, truth_hot) < 0.25
            end
        end

        @testset "exhaustion errors" begin
            # (a) HeatBath construction: the source atlas doesn't have enough maps.
            mktempdir() do dir
                toml_path = joinpath(dir, "bath_small.toml")
                write(toml_path, "[measure]\n")
                rng = PCG.PCGStateOneseq(UInt64, 4004)
                partition = LinkCutPartition(pt_graph, p88_constraints, 4; rng=rng)
                measure = Measure()
                path = joinpath(dir, "bath_small.jsonl.gz")
                writer = Writer(measure, p88_constraints, partition, path;
                                config_file=toml_path)
                run_metropolis_hastings!(partition, p88_proposal, measure, 100, rng;
                                         writer=writer, output_freq=10)
                close_writer(writer)   # only ~10 maps

                err = try
                    HeatBath(path, target_scores, pt_graph; rung=1, burn_in=0,
                            n_samples=1000, rng=PCG.PCGStateOneseq(UInt64, 1))
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("1000", err.msg)
            end

            # (b) try_heat_bath! at runtime: the pool itself is too small for how many
            # rounds leave the bath's rung idle.
            begin
                M = 3
                target_w = (0.0,)
                path = linear_path(target_w)
                scores1 = (get_log_spanning_forests,)
                mk1(k) = CycleWalk.Replica{1}(fresh_pt_partition(5000 + k), k, (0.0,),
                                              PCG.PCGStateOneseq(UInt64, UInt64(5000 + k)),
                                              RunDiagnostics(), Measure(), k, 0, 0)
                replicas = [mk1(k) for k in 1:M]
                ensemble = CycleWalk.PTEnsemble{1}(replicas, collect(1:M), linear_betas(M),
                                                   path, scores1)
                st = fresh_pt_partition(5099)
                assignment = CycleWalk.get_node_map(st.node_col, st)
                hb = CycleWalk.HeatBath("<none>", Measure(), [assignment], 1, pt_graph)  # only 1 sample
                diag = CycleWalk.PTDiagnostics(M)
                rng = PCG.PCGStateOneseq(UInt64, 6)

                CycleWalk.try_heat_bath!(ensemble, hb, 2, rng, diag)   # consumes the 1 sample
                @test isempty(hb.samples)
                err = try
                    CycleWalk.try_heat_bath!(ensemble, hb, 2, rng, diag)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("rung 1", err.msg)
            end
        end

        @testset "lineage: accepted bath swap increments bath_swaps, resets last_end_visited" begin
            M = 3
            target_w = (0.0,)
            path = linear_path(target_w)
            scores1 = (get_log_spanning_forests,)
            mk1(k) = CycleWalk.Replica{1}(fresh_pt_partition(6000 + k), k, (0.0,),
                                          PCG.PCGStateOneseq(UInt64, UInt64(6000 + k)),
                                          RunDiagnostics(), Measure(), k, 0, 0)
            replicas = [mk1(k) for k in 1:M]
            ensemble = CycleWalk.PTEnsemble{1}(replicas, collect(1:M), linear_betas(M),
                                               path, scores1)
            st = fresh_pt_partition(6099)
            assignment = CycleWalk.get_node_map(st.node_col, st)
            hb = CycleWalk.HeatBath("<none>", Measure(),
                                    [assignment, assignment, assignment], 1, pt_graph)
            diag = CycleWalk.PTDiagnostics(M)

            replicas[1].last_end_visited = 1   # so the reset below is observable
            # M=3: idle_rungs(3, 2) == [1], so round 2 is when rung 1's bath may fire.
            CycleWalk.try_heat_bath!(ensemble, hb, 2, PCG.PCGStateOneseq(UInt64, 7), diag)

            @test replicas[1].bath_swaps == 1
            @test replicas[1].last_end_visited == 0
            @test diag.bath_attempts == 1
            @test diag.bath_accepts == 1
            @test length(hb.samples) == 2
        end
    end
end
