# Tests for the config plumbing the example runners share (examples/parameterUtils.jl):
# the generic `--set` command-line overrides, the file-name tag that keeps a sweep from
# writing every point to one path, and the guard that refuses to truncate an Atlas that
# is already there.
#
# These live in examples/, which has its own environment that Pkg.test() never runs, so
# the file is included directly here.

using TOML, UnPack
using DataStructures: OrderedDict
include(joinpath(dirname(@__DIR__), "examples", "parameterUtils.jl"))

@testset "config helpers" begin
    @testset "take_flag! removes a flag ArgMacros was not told about" begin
        argv = ["cfg.toml", "--overwrite", "--thread_id", "3"]
        @test take_flag!(argv, "--overwrite")
        @test argv == ["cfg.toml", "--thread_id", "3"]
        @test !take_flag!(argv, "--overwrite")
        @test argv == ["cfg.toml", "--thread_id", "3"]
        # every occurrence, not just the first
        repeated = ["--overwrite", "a", "--overwrite"]
        @test take_flag!(repeated, "--overwrite")
        @test repeated == ["a"]
    end

    @testset "--set is lifted out of ARGS and typed by TOML" begin
        argv = ["cfg.toml", "--set", "measure.vra_weight=2.0",
                "--thread_id", "3", "--set", "plans.num_dists=4"]
        overrides = take_set_overrides!(argv)
        # what ArgMacros goes on to parse no longer contains them
        @test argv == ["cfg.toml", "--thread_id", "3"]
        @test collect(keys(overrides)) == ["measure.vra_weight", "plans.num_dists"]
        # TOML's own types, not strings
        @test overrides["measure.vra_weight"] === 2.0
        @test overrides["plans.num_dists"] === 4
    end

    @testset "values get the types the config file would have given them" begin
        @test parse_toml_value("2.0") === 2.0
        @test parse_toml_value("4") === 4
        @test parse_toml_value("1e6") === 1.0e6
        @test parse_toml_value("true") === true
        @test parse_toml_value("[\"a\", \"b\"]") == ["a", "b"]
        @test parse_toml_value("\"quoted\"") == "quoted"
        # text that is not valid TOML is taken as a plain string, so a description
        # does not have to survive shell quoting
        @test parse_toml_value("hello") == "hello"
        @test parse_toml_value("a description with spaces") ==
              "a description with spaces"
    end

    @testset "--set applies to the parsed config" begin
        params = Dict{String, Any}(
            "measure" => Dict{String, Any}("gamma" => 1.0),
            "plans"   => Dict{String, Any}("num_dists" => 2))
        apply_set_overrides!(params, OrderedDict{String, Any}(
            "measure.gamma"      => 3.0,      # an existing key
            "measure.vra_weight" => 2.0,      # a new one: the point of --set
            "plans.num_dists"    => 4))
        @test params["measure"]["gamma"] == 3.0
        @test params["measure"]["vra_weight"] == 2.0
        @test params["plans"]["num_dists"] == 4
    end

    @testset "--set refuses what it cannot mean" begin
        params = Dict{String, Any}("measure" => Dict{String, Any}("gamma" => 1.0))

        # tables are fixed, so a table that does not exist is a typo, not a section
        err = try
            apply_set_overrides!(params,
                OrderedDict{String, Any}("measur.gamma" => 2.0)); nothing
        catch e; e end
        @test err !== nothing
        @test occursin("measur", err.msg)
        @test occursin("measure", err.msg)      # says what tables there are

        # a value set twice, two ways, is an error rather than a silent winner
        err2 = try
            apply_set_overrides!(params, OrderedDict{String, Any}("measure.gamma" => 2.0);
                named_flags=Dict("measure.gamma" => (3.0, "--gamma"))); nothing
        catch e; e end
        @test err2 !== nothing
        @test occursin("--gamma", err2.msg)

        # …but the same flag left unset is no conflict
        apply_set_overrides!(params, OrderedDict{String, Any}("measure.gamma" => 2.0);
                             named_flags=Dict("measure.gamma" => (nothing, "--gamma")))
        @test params["measure"]["gamma"] == 2.0
    end

    @testset "malformed --set is rejected" begin
        @test_throws ErrorException take_set_overrides!(["--set"])
        @test_throws ErrorException take_set_overrides!(["--set", "novalue"])
        @test_throws ErrorException take_set_overrides!(["--set", "gamma=2.0"])
    end

    @testset "the file name carries every parameter the measure reads" begin
        cfg = Dict{String, Any}(
            "gamma" => 1.0, "vra_weight" => 2.0, "unused" => 7.0,
            "energy" => [
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => "gamma"),
                Dict{String, Any}("name" => "get_log_district_trees",
                                  "weight" => "2*vra_weight")])
        specs = energy_specs(cfg)
        pars = measure_parameters(cfg)

        # gamma is excluded: the runners tag it themselves, from the measure
        @test parameter_tag(specs, pars) == "_vra_weight2.0"
        # a parameter nothing reads cannot change the target, so it is not in the name
        @test !occursin("unused", parameter_tag(specs, pars))

        # two points of a sweep must not produce the same tag
        low  = parameter_tag(specs, merge(pars, Dict("vra_weight" => 2.0)))
        high = parameter_tag(specs, merge(pars, Dict("vra_weight" => 5.0)))
        @test low != high

        # exact formatting: %g would print these the same and let them collide
        a = parameter_tag(specs, merge(pars, Dict("vra_weight" => 0.3333333)))
        b = parameter_tag(specs, merge(pars, Dict("vra_weight" => 0.3333334)))
        @test a != b
        @test parameter_tag(specs, merge(pars, Dict("vra_weight" => 0.1+0.2))) ==
              "_vra_weight0.30000000000000004"

        # a zero weight contributes no energy, so it contributes no tag
        @test parameter_tag(specs, merge(pars, Dict("vra_weight" => 0))) == ""
        # sorted, so one config always gives one path
        multi = energy_specs(Dict{String, Any}(
            "b_param" => 1.0, "a_param" => 2.0,
            "energy" => [
                Dict{String, Any}("name" => "get_log_spanning_forests",
                                  "weight" => "b_param"),
                Dict{String, Any}("name" => "get_log_district_trees",
                                  "weight" => "a_param")]))
        @test parameter_tag(multi, Dict{String, Any}("a_param" => 2.0,
                                                     "b_param" => 1.0)) ==
              "_a_param2.0_b_param1.0"
    end

    @testset "ensure_writable refuses to truncate an existing atlas" begin
        mktempdir() do dir
            path = joinpath(dir, "atlas.jsonl.gz")

            # nothing there yet: fine
            @test ensure_writable(path, "w") == path

            write(path, "existing output")
            err = try ensure_writable(path, "w"); nothing catch e; e end
            @test err !== nothing
            @test occursin("already exists", err.msg)
            @test read(path, String) == "existing output"   # untouched

            # explicitly asked for, or appending, or a different path: allowed
            @test ensure_writable(path, "w"; overwrite=true) == path
            @test ensure_writable(path, "a") == path
            @test ensure_writable(joinpath(dir, "other.jsonl.gz"), "w") ==
                  joinpath(dir, "other.jsonl.gz")
        end
    end
end
