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

@testset "measure specs" begin
    short_form = Dict{String, Any}(
        "gamma" => 1.5,
        "iso_weight" => 0.4,
        "measure_scores" => ["get_log_spanning_forests", "get_isoperimetric_score"])

    @testset "parameters are every numeric scalar under [measure]" begin
        cfg = Dict{String, Any}("gamma" => 1.5, "iso_weight" => 0.0, "n" => 3,
                                "label" => "text", "flag" => true,
                                "measure_scores" => ["get_log_spanning_forests"],
                                "weights" => Dict{String, Any}("x" => 1))
        pars = measure_parameters(cfg)
        @test Set(keys(pars)) == Set(["gamma", "iso_weight", "n"])
        @test pars["gamma"] == 1.5
    end

    @testset "the short form is translated, not special-cased" begin
        specs = energy_specs(short_form)
        @test [s.name for s in specs] ==
              ["get_log_spanning_forests", "get_isoperimetric_score"]
        # the two named scores bind to the parameters they always did — as
        # expressions, so nothing downstream needs to know they are special
        @test [s.weight for s in specs] == ["gamma", "iso_weight"]
        @test all(s.weight_start === nothing for s in specs)
    end

    @testset "short and explicit forms produce the same measure" begin
        explicit = Dict{String, Any}(
            "gamma" => 1.5, "iso_weight" => 0.4,
            "energy" => [
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => "gamma"),
                Dict{String, Any}("name" => "get_isoperimetric_score",
                                  "weight" => "iso_weight")])
        a = build_measure(short_form)
        b = build_measure(explicit)
        @test a.scores == b.scores
        @test a.weights == b.weights
        @test a.weights[get_log_spanning_forests] == 1.5
        @test a.weights[get_isoperimetric_score] == 0.4
    end

    @testset "a weight may be a number or an expression" begin
        cfg = Dict{String, Any}(
            "gamma" => 2.0,
            "energy" => [
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => "2*gamma + 1"),
                Dict{String, Any}("name" => "get_isoperimetric_score",
                                  "weight" => 0.25)])
        m = build_measure(cfg)
        @test m.weights[get_log_spanning_forests] == 5.0
        @test m.weights[get_isoperimetric_score] == 0.25
    end

    @testset "weight_start keeps an energy annealed down to zero" begin
        cfg = Dict{String, Any}(
            "gamma" => 0.0, "gamma_start" => 0.5,
            "energy" => [Dict{String, Any}("name" => "get_log_spanning_forests",
                                           "weight" => "gamma",
                                           "weight_start" => "gamma_start")])
        specs = energy_specs(cfg)
        m = build_measure(cfg)
        # zero target weight, but present so a schedule can ramp it
        @test get_log_spanning_forests in m.scores
        @test m.weights[get_log_spanning_forests] == 0.0
        @test keys(m.weights) == m.scores
        @test annealing_weights(specs, measure_parameters(cfg)) ==
              Dict("get_log_spanning_forests" => (0.5, 0.0))

        # without a weight_start the zero-weight energy is dropped as before
        plain = Dict{String, Any}(
            "gamma" => 0.0,
            "energy" => [Dict{String, Any}("name" => "get_log_spanning_forests",
                                           "weight" => "gamma")])
        @test isempty(build_measure(plain).scores)
        @test isempty(annealing_weights(energy_specs(plain),
                                        measure_parameters(plain)))
    end

    @testset "a builder is called with its configured arguments" begin
        cfg = Dict{String, Any}(
            "energy" => [Dict{String, Any}(
                "name" => "build_get_partisan_margins",
                "args" => ["dem", "rep"],
                "weight" => 1.0,
                "desc" => "partisan margins")])
        m = build_measure(cfg)
        @test length(m.scores) == 1
        f = first(m.scores)
        # a built energy is a closure; the desc is what keeps the header readable
        @test m.descriptions[f] == "partisan margins"

        # with no desc, the builder's name is used rather than the closure's
        nodesc = Dict{String, Any}(
            "energy" => [Dict{String, Any}(
                "name" => "build_get_partisan_margins",
                "args" => ["dem", "rep"], "weight" => 1.0)])
        m2 = build_measure(nodesc)
        @test m2.descriptions[first(m2.scores)] == "build_get_partisan_margins"
    end

    @testset "the same builder may appear more than once" begin
        cfg = Dict{String, Any}(
            "energy" => [
                Dict{String, Any}("name" => "build_get_partisan_margins",
                                  "args" => ["dem16", "rep16"], "weight" => 1.0,
                                  "desc" => "2016"),
                Dict{String, Any}("name" => "build_get_partisan_margins",
                                  "args" => ["dem20", "rep20"], "weight" => 2.0,
                                  "desc" => "2020")])
        m = build_measure(cfg)
        @test length(m.scores) == 2          # distinct closures, not a collision
        @test Set(values(m.descriptions)) == Set(["2016", "2020"])
        @test Set(values(m.weights)) == Set([1.0, 2.0])
    end

    @testset "the same plain energy twice is an error" begin
        cfg = Dict{String, Any}(
            "energy" => [
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => 1.0),
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => 2.0)])
        # Measure is keyed by function, so the second would silently replace the first
        @test_throws ArgumentError build_measure(cfg)
    end

    @testset "context supplies what a config cannot write down" begin
        # build_performant_vra_score's first argument is a graph
        cfg = Dict{String, Any}(
            "energy" => [Dict{String, Any}(
                "name" => "build_performant_vra_score",
                "context" => ["graph"],
                "args" => [[(("pop",), ("pop",))]],
                "kwargs" => Dict{String, Any}("target_districts" => 1),
                "weight" => 1.0,
                "desc" => "vra")])
        m = build_measure(cfg; context=(graph=small_square_base_graph,))
        @test length(m.scores) == 1
        @test m.descriptions[first(m.scores)] == "vra"

        # asking for context the run does not provide names what is available
        err = try build_measure(cfg; context=(num_dists=4,)); nothing catch e; e end
        @test err isa ArgumentError
        @test occursin("graph", err.msg)
        @test occursin("num_dists", err.msg)
    end

    @testset "bad configs are refused with a message that says why" begin
        both = Dict{String, Any}("gamma" => 1.0,
                                 "measure_scores" => ["get_log_spanning_forests"],
                                 "energy" => [Dict{String, Any}(
                                     "name" => "get_log_spanning_forests",
                                     "weight" => 1.0)])
        @test_throws ArgumentError energy_specs(both)

        for bad in [
            Dict{String, Any}("energy" => [Dict{String, Any}("weight" => 1.0)]),
            Dict{String, Any}("energy" => [Dict{String, Any}("name" => "x")]),
            Dict{String, Any}("energy" => [Dict{String, Any}(
                "name" => "get_log_spanning_forests", "weight" => 1.0,
                "wieght" => 2.0)]),                      # typo'd key
            Dict{String, Any}("energy" => Dict{String, Any}("name" => "x")),
        ]
            @test_throws ArgumentError energy_specs(bad)
        end

        # a name CycleWalk does not export, including an internal one
        unexported = Dict{String, Any}("energy" => [Dict{String, Any}(
            "name" => "get_district_roots", "weight" => 1.0)])
        @test isdefined(CycleWalk, :get_district_roots)      # it exists…
        @test !(:get_district_roots in names(CycleWalk))     # …but is not exported
        @test_throws ArgumentError build_measure(unexported)

        nonesuch = Dict{String, Any}("energy" => [Dict{String, Any}(
            "name" => "get_nonexistent_score", "weight" => 1.0)])
        @test_throws ArgumentError build_measure(nonesuch)

        # a builder whose configured arguments do not fit
        badargs = Dict{String, Any}("energy" => [Dict{String, Any}(
            "name" => "build_get_partisan_margins",
            "args" => ["only_one"], "weight" => 1.0)])
        @test_throws ArgumentError build_measure(badargs)

        # an unknown parameter in a weight expression
        badweight = Dict{String, Any}("energy" => [Dict{String, Any}(
            "name" => "get_log_spanning_forests", "weight" => "not_a_parameter")])
        @test_throws ArgumentError build_measure(badweight)
    end

    @testset "gamma is the weight on the forest energy, not a config key" begin
        # gamma names something specific — the weight in front of the log
        # spanning-forest energy — and the atlas filename is tagged with it, so it is
        # read off the measure rather than out of a [measure] key that could disagree.
        mismatched = Dict{String, Any}(
            "gamma" => 1.0,                       # a key that says one thing…
            "energy" => [Dict{String, Any}("name" => "get_log_spanning_forests",
                                           "weight" => 0.3)])   # …the measure another
        specs = energy_specs(mismatched)
        pars = measure_parameters(mismatched)
        @test energy_weight(specs, "get_log_spanning_forests", pars) == 0.3
        @test build_measure(mismatched).weights[get_log_spanning_forests] == 0.3

        # the short form agrees with the key, because there the key is what binds
        @test energy_weight(energy_specs(short_form), "get_log_spanning_forests",
                            measure_parameters(short_form)) == 1.5
        @test energy_weight(energy_specs(short_form), "get_isoperimetric_score",
                            measure_parameters(short_form)) == 0.4

        # an energy the measure does not have weighs nothing
        @test energy_weight(specs, "get_isoperimetric_score", pars) == 0.0
        # and it resolves through an expression like any other weight
        expr = Dict{String, Any}(
            "gamma" => 2.0,
            "energy" => [Dict{String, Any}("name" => "get_log_spanning_forests",
                                           "weight" => "3*gamma")])
        @test energy_weight(energy_specs(expr), "get_log_spanning_forests",
                            measure_parameters(expr)) == 6.0
    end

    @testset "build_annealed_measure returns the schedule alongside the measure" begin
        cfg = Dict{String, Any}(
            "gamma" => 1.0, "iso_weight" => 0.3,
            "energy" => [
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => "gamma", "weight_start" => 0.5),
                Dict{String, Any}("name" => "get_isoperimetric_score",
                                  "weight" => "iso_weight",
                                  "weight_start" => "0.1*iso_weight")])
        specs = energy_specs(cfg)
        m, ramp = build_annealed_measure(specs, measure_parameters(cfg))
        @test ramp[get_log_spanning_forests] == (0.5, 1.0)
        @test ramp[get_isoperimetric_score] == (0.03, 0.3)
        # the ramp is keyed by the very functions the measure holds
        @test Set(keys(ramp)) == m.scores

        # an energy with no weight_start starts from default_start (0 for annealing)
        plain = energy_specs(Dict{String, Any}(
            "energy" => [Dict{String, Any}("name" => "get_log_spanning_forests",
                                           "weight" => 2.0)]))
        _, r2 = build_annealed_measure(plain, Dict{String, Any}())
        @test r2[get_log_spanning_forests] == (0.0, 2.0)
        _, r3 = build_annealed_measure(plain, Dict{String, Any}(); default_start=1.0)
        @test r3[get_log_spanning_forests] == (1.0, 2.0)

        # build_measure collects no ramp and is unaffected by weight_start
        @test build_measure(specs, measure_parameters(cfg)).weights ==
              m.weights
    end

    @testset "the short form only ever describes a target" begin
        # It exists for the configs of released runs, which are always plain MCMC
        # targets; an annealing base is a weight_start in the explicit form, and the
        # short form has no way to say one.
        cfg = Dict{String, Any}(
            "gamma" => 1.0, "iso_weight" => 0.3,
            "gamma_start" => 0.5, "iso_start" => 0.1,   # not part of the short form
            "measure_scores" => ["get_log_spanning_forests",
                                 "get_isoperimetric_score"])
        specs = energy_specs(cfg)
        @test all(s.weight_start === nothing for s in specs)
        pars = measure_parameters(cfg)
        @test energy_weight_start(specs, "get_log_spanning_forests", pars) == 0.0
        _, ramp = build_annealed_measure(specs, pars)
        @test ramp[get_log_spanning_forests] == (0.0, 1.0)   # the default base
    end

    @testset "an energy annealed down to zero survives into the ramp" begin
        cfg = Dict{String, Any}(
            "gamma" => 0.0, "iso_weight" => 0.3,
            "energy" => [
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => "gamma", "weight_start" => 0.5),
                Dict{String, Any}("name" => "get_isoperimetric_score",
                                  "weight" => "iso_weight")])
        m, ramp = build_annealed_measure(energy_specs(cfg), measure_parameters(cfg))
        @test get_log_spanning_forests in m.scores
        @test ramp[get_log_spanning_forests] == (0.5, 0.0)
        @test keys(m.weights) == m.scores

        # zero the whole way is still dropped, and so cannot be ramped
        gone = Dict{String, Any}(
            "gamma" => 0.0, "iso_weight" => 0.3,
            "measure_scores" => ["get_log_spanning_forests",
                                 "get_isoperimetric_score"])
        m2, ramp2 = build_annealed_measure(energy_specs(gone),
                                           measure_parameters(gone))
        @test !(get_log_spanning_forests in m2.scores)
        @test !haskey(ramp2, get_log_spanning_forests)
        @test Set(keys(ramp2)) == m2.scores
    end

    @testset "referenced_parameters is what the measure actually reads" begin
        cfg = Dict{String, Any}(
            "gamma" => 1.0, "vra_weight" => 2.0, "unused" => 7.0,
            "energy" => [
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => "gamma"),
                Dict{String, Any}("name" => "get_log_district_trees",
                                  "weight" => "2*vra_weight")])
        # `unused` is defined but no expression names it, so it cannot change the
        # target and must not affect the run's identity
        @test referenced_parameters(energy_specs(cfg)) == ["gamma", "vra_weight"]

        # a literal weight reads nothing
        literal = energy_specs(Dict{String, Any}(
            "energy" => [Dict{String, Any}("name" => "get_log_spanning_forests",
                                           "weight" => 2.0)]))
        @test referenced_parameters(literal) == String[]

        # weight_start counts too — it changes the schedule
        annealed = energy_specs(Dict{String, Any}(
            "gamma" => 1.0, "warmup" => 0.5,
            "energy" => [Dict{String, Any}("name" => "get_log_spanning_forests",
                                           "weight" => "gamma",
                                           "weight_start" => "warmup")]))
        @test referenced_parameters(annealed) == ["gamma", "warmup"]

        # the short form reads gamma/iso_weight, which the runners tag by name
        @test referenced_parameters(energy_specs(short_form)) ==
              ["gamma", "iso_weight"]
    end

    @testset "an empty [measure] gives an empty measure" begin
        @test isempty(energy_specs(Dict{String, Any}()))
        @test isempty(build_measure(Dict{String, Any}()).scores)
    end

    @testset "t is reserved for weight_path expressions" begin
        @test_throws ArgumentError measure_parameters(
            Dict{String, Any}("t" => 0.5, "gamma" => 1.0))
        # only a numeric "t" collides; a non-numeric one is dropped like any other
        # non-numeric key and never reaches the reserved-name check
        @test !haskey(measure_parameters(Dict{String, Any}("t" => "text")), "t")
    end

    @testset "weight_path and weight_start are mutually exclusive per energy" begin
        cfg = Dict{String, Any}("energy" => [Dict{String, Any}(
            "name" => "get_log_spanning_forests", "weight" => 1.0,
            "weight_start" => 0.0, "weight_path" => "t")])
        @test_throws ArgumentError energy_specs(cfg)

        # weight_path must be a string
        bad_type = Dict{String, Any}("energy" => [Dict{String, Any}(
            "name" => "get_log_spanning_forests", "weight" => 1.0,
            "weight_path" => 2.0)])
        @test_throws ArgumentError energy_specs(bad_type)

        # weight_path alone (no weight_start) is fine
        ok = Dict{String, Any}("gamma" => 1.0, "energy" => [Dict{String, Any}(
            "name" => "get_log_spanning_forests", "weight" => "gamma",
            "weight_path" => "gamma*t")])
        specs = energy_specs(ok)
        @test specs[1].weight_path == "gamma*t"
        @test specs[1].weight_start === nothing
    end

    @testset "build_path_measure: measure + tempering path" begin
        cfg = Dict{String, Any}(
            "gamma" => 1.0, "iso_weight" => 0.3,
            "energy" => [
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => "gamma", "weight_path" => "gamma*t"),
                Dict{String, Any}("name" => "get_isoperimetric_score",
                                  "weight" => "iso_weight")])   # no weight_path
        specs = energy_specs(cfg)
        pars = measure_parameters(cfg)
        measure, path = build_path_measure(specs, pars)

        # the measure itself is built from `weight` exactly like build_measure
        @test measure.weights[get_log_spanning_forests] == 1.0
        @test measure.weights[get_isoperimetric_score] == 0.3

        scores, _ = annealed_smc_scores_and_targets(measure)
        @test Set(scores) == measure.scores

        # the tempered energy follows its weight_path...
        w0 = CycleWalk.weights_at(path, 0.0)
        w1 = CycleWalk.weights_at(path, 1.0)
        gamma_idx = findfirst(==(get_log_spanning_forests), scores)
        iso_idx   = findfirst(==(get_isoperimetric_score), scores)
        @test w0[gamma_idx] == 0.0
        @test w1[gamma_idx] == 1.0
        # ...and the untempered energy holds constant at its weight throughout
        @test w0[iso_idx] == 0.3
        @test w1[iso_idx] == 0.3
        @test CycleWalk.weights_at(path, 0.37)[iso_idx] == 0.3
    end

    @testset "build_path_measure: a zero-weight energy with a weight_path is kept" begin
        # mirrors build_annealed_measure's allow_zero rule for weight_start: a
        # weight_path may make an energy nonzero at some t even if its registered
        # `weight` happens to be 0, so it must not be dropped from the measure
        cfg = Dict{String, Any}("energy" => [Dict{String, Any}(
            "name" => "get_log_spanning_forests", "weight" => 0.0,
            "weight_path" => "1 - t")])
        measure, path = build_path_measure(energy_specs(cfg), measure_parameters(cfg))
        @test get_log_spanning_forests in measure.scores
        @test keys(measure.weights) == measure.scores
        scores, _ = annealed_smc_scores_and_targets(measure)
        @test CycleWalk.weights_at(path, 0.0)[1] == 1.0
        @test CycleWalk.weights_at(path, 1.0)[1] == 0.0
    end

    @testset "referenced_parameters scans weight_path, excluding t" begin
        cfg = Dict{String, Any}(
            "gamma" => 1.0, "mcd_weight" => 2.0,
            "energy" => [
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => "gamma", "weight_path" => "gamma*t"),
                Dict{String, Any}("name" => "get_log_district_trees",
                                  "weight" => 0.0,
                                  "weight_path" => "mcd_weight*(1-t)")])
        # mcd_weight is referenced ONLY inside weight_path — must still be tagged,
        # or two runs differing only in it would collide on one output path
        @test referenced_parameters(energy_specs(cfg)) == ["gamma", "mcd_weight"]

        # t itself is never a run parameter and must never appear
        t_only = Dict{String, Any}("energy" => [Dict{String, Any}(
            "name" => "get_log_spanning_forests", "weight" => 1.0,
            "weight_path" => "t")])
        @test referenced_parameters(energy_specs(t_only)) == String[]
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

    @testset "collecting names refuses exactly what evaluating refuses" begin
        # weight_expression_parameters walks the same parsed tree as
        # evaluate_weight_expression, through the same whitelist. It is a second
        # entry point into that walk, so the guarantees have to be re-checked here:
        # a collector that just grabbed every symbol would pass every test above.
        for bad in ["run(`ls`)", "include(\"x.jl\")", "Base.exit()", "exit()",
                    "sqrt(gamma)", "x[1]", "f(x)=x", "1+2; run(`ls`)",
                    "\"a string\"", "gamma == 1", "!gamma", "gamma'",
                    "@eval 1", "gamma % 2", "[gamma]", "(gamma, 1)",
                    "if gamma; 1; end", "2*"]
            @test_throws ArgumentError weight_expression_parameters(bad)
        end

        # the two entry points agree on what *is* allowed
        for good in ["gamma", "2*gamma + 1", "-gamma", "(gamma+1)*2", "4", "n^2"]
            @test weight_expression_parameters(good) isa Vector{String}
            @test evaluate_weight_expression(good, knobs) isa Float64
        end

        @test weight_expression_parameters("2*gamma + gamma/iso_weight") ==
              ["gamma", "iso_weight"]
        @test weight_expression_parameters("4") == String[]
        # collecting deliberately does not require the names to exist yet — it is
        # called on a config before a run's parameters are assembled
        @test weight_expression_parameters("not_a_parameter") == ["not_a_parameter"]

        # and it must not act on what it refuses, any more than evaluating does
        mktempdir() do dir
            marker = joinpath(dir, "executed")
            m = repr(marker)
            for bad in ["write($m, \"x\")", "touch($m)", "run(`touch $marker`)",
                        "(write($m, \"x\"); 1)", "gamma * write($m, \"x\")"]
                @test_throws ArgumentError weight_expression_parameters(bad)
                @test !ispath(marker)
            end
            @test isempty(readdir(dir))
        end
        @test_throws ArgumentError weight_expression_parameters("bump_probe()")
        @test EXPRESSION_PROBE[] == 0

        # the caps apply on this path too
        deep = "gamma"
        for _ = 1:40
            deep = "(1+" * deep * ")"
        end
        @test_throws ArgumentError weight_expression_parameters(deep)
        @test_throws ArgumentError weight_expression_parameters(
            join(fill("gamma", 200), "+"))
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

    @testset "weight_path_closure: parse once, evaluate against t" begin
        f = weight_path_closure("gamma*t", knobs)
        @test f isa Function
        @test f(0.0) == 0.0
        @test f(0.5) == 0.75
        @test f(1.0) == 1.5

        g = weight_path_closure("iso_weight*t^2 + 1", knobs)
        @test g(0.0) == 1.0
        @test g(2.0) == 1.0 + 0.4*4

        # a path that is nonzero at t=0 and fades out — not a monotone 0->target
        # ramp, and that's fine: the grammar has no notion of "target" at all
        h = weight_path_closure("gamma*(1-t)", knobs)
        @test h(0.0) == 1.5
        @test h(1.0) == 0.0

        # constant (no t) is legal — an energy that doesn't move
        c = weight_path_closure("gamma", knobs)
        @test c(0.0) == c(0.5) == c(1.0) == 1.5
    end

    @testset "weight_path_closure validates once, at construction" begin
        # unknown name fails immediately, not on first use
        err = try weight_path_closure("not_a_parameter*t", knobs); nothing
              catch e; e end
        @test err isa ArgumentError
        @test occursin("not_a_parameter", err.msg)

        # bad syntax, same
        @test_throws ArgumentError weight_path_closure("2*", knobs)

        # non-arithmetic is refused exactly as evaluate_weight_expression refuses it
        @test_throws ArgumentError weight_path_closure("sqrt(t)", knobs)
        @test_throws ArgumentError weight_path_closure("run(`ls`)", knobs)
    end

    @testset "weight_path_closure surfaces a bad value at the t that produces it" begin
        # 1/t is fine near t=1 but blows up at t=0; construction (validated at a
        # dummy t=0.0) must therefore fail up front for THIS expression...
        @test_throws ArgumentError weight_path_closure("1/t", knobs)
        # ...but an expression that is finite at t=0 and only misbehaves elsewhere
        # is caught when that t is actually evaluated, not silently
        f = weight_path_closure("1/(1-t)", knobs)
        @test f(0.0) == 1.0
        @test_throws ArgumentError f(1.0)
    end
end
