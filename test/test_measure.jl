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

# A callable the expression tests can watch: if the evaluator ever invokes a function
# named in an expression, this counter moves.
const EXPRESSION_PROBE = Ref(0)
bump_probe(args...) = (EXPRESSION_PROBE[] += 1; 1.0)

@testset "weight expressions" begin
    knobs = Dict{String, Any}("gamma" => 1.5, "iso_weight" => 0.4, "n" => 3)
    EXPRESSION_PROBE[] = 0

    @testset "arithmetic over named parameters" begin
        @test evaluate_weight_expression("gamma", knobs) == 1.5
        @test evaluate_weight_expression("2*gamma + 1", knobs) == 4.0
        @test evaluate_weight_expression("-gamma", knobs) == -1.5
        @test evaluate_weight_expression("gamma - iso_weight", knobs) == 1.1
        @test evaluate_weight_expression("gamma/2", knobs) == 0.75
        @test evaluate_weight_expression("n^2", knobs) == 9.0
        @test evaluate_weight_expression("1 + 2 + 3", knobs) == 6.0    # n-ary +
        @test evaluate_weight_expression("(gamma + 1)*2", knobs) == 5.0
        @test evaluate_weight_expression("  gamma  ", knobs) == 1.5    # trimmed
        @test evaluate_weight_expression("4", knobs) == 4.0            # bare literal
    end

    @testset "anything that is not arithmetic is refused" begin
        # None of these may run. They are rejected while walking the parsed tree,
        # which is why the expression is never handed to eval.
        for bad in ["run(`ls`)", "include(\"x.jl\")", "Base.exit()", "exit()",
                    "sqrt(gamma)", "x[1]", "f(x)=x", "1+2; run(`ls`)",
                    "\"a string\"", "gamma == 1", "!gamma", "gamma'",
                    "@eval 1", "gamma % 2", "gamma ÷ 2", "gamma & 1",
                    "gamma | 1", "gamma << 1", "gamma > 1", "[gamma]",
                    "(gamma, 1)", "gamma...", "if gamma; 1; end"]
            @test_throws ArgumentError evaluate_weight_expression(bad, knobs)
        end
    end

    @testset "the allowed set is exactly the documented one" begin
        # A guard against the whitelist quietly widening: every operator here is
        # documented as permitted, and nothing else is.
        @test Set(keys(CycleWalk.WEIGHT_EXPRESSION_OPS)) ==
              Set([:+, :-, :*, :/, :^])
        # and each one actually works
        @test evaluate_weight_expression("gamma + 1", knobs) == 2.5
        @test evaluate_weight_expression("gamma - 1", knobs) == 0.5
        @test evaluate_weight_expression("gamma * 2", knobs) == 3.0
        @test evaluate_weight_expression("gamma / 2", knobs) == 0.75
        @test evaluate_weight_expression("n ^ 2", knobs) == 9.0
    end

    @testset "a refused expression is never executed" begin
        # @test_throws alone only proves the call failed — it does not prove the
        # expression did not act first. These would each leave a trace if they ran:
        # evaluating `write(marker, "x")` or `touch(marker)` by hand does create the
        # file, so an empty directory afterwards means the evaluator never acted.
        mktempdir() do dir
            marker = joinpath(dir, "executed")
            m = repr(marker)   # a properly quoted Julia string literal
            for bad in ["write($m, \"x\")",
                        "touch($m)",
                        "run(`touch $marker`)",
                        "(write($m, \"x\"); 1)",
                        "1 + write($m, \"x\")",
                        "gamma * write($m, \"x\")",
                        "open($m, \"w\")"]
                @test_throws ArgumentError evaluate_weight_expression(bad, knobs)
                @test !ispath(marker)          # nothing ran
            end
            @test isempty(readdir(dir))        # and nothing else was created
        end
    end

    @testset "an in-scope callable is still never called" begin
        # The callee whitelist is by name, so even a function that exists, is
        # callable, and would obviously work is not reachable from an expression.
        @test_throws ArgumentError evaluate_weight_expression("bump_probe()", knobs)
        @test EXPRESSION_PROBE[] == 0

        # nor by smuggling the function in as a parameter value
        with_fn = Dict{String, Any}("f" => bump_probe, "gamma" => 1.5)
        @test_throws ArgumentError evaluate_weight_expression("f", with_fn)
        @test_throws ArgumentError evaluate_weight_expression("f()", with_fn)
        @test_throws ArgumentError evaluate_weight_expression("f(gamma)", with_fn)
        @test EXPRESSION_PROBE[] == 0

        # …and a parameter named after an allowed operator is a value, not a call
        @test evaluate_weight_expression("gamma", with_fn) == 1.5
        @test EXPRESSION_PROBE[] == 0
    end

    @testset "unknown names and unusable values are refused" begin
        @test_throws ArgumentError evaluate_weight_expression("gama", knobs)  # typo
        @test_throws ArgumentError evaluate_weight_expression("2*", knobs)    # unparsable
        # not-a-number parameters
        @test_throws ArgumentError evaluate_weight_expression(
            "label", Dict{String, Any}("label" => "text"))
        @test_throws ArgumentError evaluate_weight_expression(
            "flag", Dict{String, Any}("flag" => true))
        @test_throws ArgumentError evaluate_weight_expression("true", knobs)
        # a non-finite result is a config error, not a silent Inf/NaN weight
        @test_throws ArgumentError evaluate_weight_expression("1/0", knobs)
        @test_throws ArgumentError evaluate_weight_expression("0/0", knobs)
        @test_throws ArgumentError evaluate_weight_expression("10.0^400", knobs)
    end

    @testset "the error says what was wrong" begin
        err = try
            evaluate_weight_expression("gama*2", knobs); nothing
        catch e; e end
        @test err isa ArgumentError
        @test occursin("gama", err.msg)
        @test occursin("gamma", err.msg)   # lists the parameters that do exist

        err2 = try
            evaluate_weight_expression("sqrt(gamma)", knobs); nothing
        catch e; e end
        @test occursin("sqrt", err2.msg)
        @test occursin("+ - * / ^", err2.msg)
    end

    @testset "runaway expressions are capped" begin
        # real nesting — redundant parentheses do not create AST nodes, so
        # "((gamma))" parses to the same tree as "gamma"
        deep = "gamma"
        for _ = 1:40
            deep = "(1+" * deep * ")"
        end
        @test_throws ArgumentError evaluate_weight_expression(deep, knobs)

        shallow = "(1+(1+gamma))"
        @test evaluate_weight_expression(shallow, knobs) == 3.5

        wide = join(fill("gamma", 200), "+")     # node count
        @test_throws ArgumentError evaluate_weight_expression(wide, knobs)
    end
end
