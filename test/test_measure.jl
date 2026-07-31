# Tests for building a Measure — in particular the zero-weight rule, which decides
# whether an energy an annealing schedule ramps can participate at all.

@testset "measure construction" begin
    @testset "a zero weight is dropped by default" begin
        m = Measure()
        push_energy!(m, get_log_spanning_forests, 0.0)
        @test isempty(m.scores)
        @test isempty(m.weights)
        # the invariant push_energy! asserts holds
        @test keys(m.weights) == m.scores
    end

    @testset "allow_zero keeps a score an anneal will ramp" begin
        m = Measure()
        push_energy!(m, get_log_spanning_forests, 0.0; allow_zero=true)
        @test get_log_spanning_forests in m.scores
        @test m.weights[get_log_spanning_forests] == 0.0
        @test keys(m.weights) == m.scores
        # a later push must not trip the assertion
        push_energy!(m, get_isoperimetric_score, 0.3)
        @test length(m.scores) == 2
    end

    @testset "an annealed weight only counts for a score in the measure" begin
        # This is the shape of the AIS/ASMC bug: an energy annealed down to zero has
        # a zero target weight, so without allow_zero it never enters `scores`, and
        # the schedule's write to `weights` is silently ignored by get_log_energy.
        p = LinkCutPartition(small_square_graph, measure_constraints, 4;
                             rng=PCG.PCGStateOneseq(UInt64, 3))

        dropped = Measure()
        push_energy!(dropped, get_log_spanning_forests, 0.0)
        dropped.weights[get_log_spanning_forests] = 0.25   # what a schedule writes
        @test keys(dropped.weights) != dropped.scores      # invariant now broken
        @test get_log_energy(p, dropped) == 0.0            # and it counts for nothing

        kept = Measure()
        push_energy!(kept, get_log_spanning_forests, 0.0; allow_zero=true)
        kept.weights[get_log_spanning_forests] = 0.25
        @test keys(kept.weights) == kept.scores
        @test get_log_energy(p, kept) != 0.0

        # ramped to the same weight, the two agree with an ordinary measure
        direct = Measure()
        push_energy!(direct, get_log_spanning_forests, 0.25)
        @test get_log_energy(p, kept) == get_log_energy(p, direct)
    end

    @testset "a zero weight still costs nothing to evaluate" begin
        p = LinkCutPartition(small_square_graph, measure_constraints, 4;
                             rng=PCG.PCGStateOneseq(UInt64, 4))
        m = Measure()
        push_energy!(m, get_log_spanning_forests, 0.0; allow_zero=true)
        @test get_log_energy(p, m) == 0.0
    end
end
