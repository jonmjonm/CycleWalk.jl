## Shared parameter derivation for the TOML-driven example runners.
##
## `runtimeParameters.jl` (used by run_cyclewalk_toml.jl) parses the command line,
## overrides the parsed TOML with it, and then calls `derive_params` here to compute
## everything downstream of the raw config: seeds, step counts, the atlas name and
## output path, and the header's `ad_param` dict. `run_cyclewalk_extend.jl` reuses the
## same function so an extension segment derives its settings exactly as the original
## run did, from the TOML the original run embedded in its atlas header.
##
## Everything derived lives in one NamedTuple so the two callers cannot drift apart.

using UnPack, TOML
using DataStructures: OrderedDict

"""
    derive_params(params, toml_config_file; make_output_dir=true) -> NamedTuple

Compute every quantity the runner scripts derive from a parsed TOML config `params`
(after any command-line overrides have been applied to it): the RNG seed, the total
step count and output frequency implied by `two_cycle_walk_frac`, the atlas name and
output path, the graph path, and the `ad_param` dict recorded in the Atlas header.

`toml_config_file` is recorded in `ad_param` and is otherwise unused. Set
`make_output_dir=false` to skip creating the output directory (a caller that writes
somewhere else, such as an extension segment, does not need it).

Throws if the map file named by the config does not exist, and asserts
`0 ≤ two_cycle_walk_frac ≤ 1`.
"""
function derive_params(
    params::AbstractDict,
    toml_config_file::AbstractString;
    make_output_dir::Bool=true,
    cli_overrides::AbstractDict=Dict{String, Any}()
)
    #load parameters from the (already overridden) TOML dictionary
    @unpack cycle_walk_steps, two_cycle_walk_frac = params["mcmc"]
    @unpack outputDirectory, atlasNameBase = params["run"]
    @unpack thread_id = params["run"]
    @unpack cycle_walk_out_freq = params["run"]
    @unpack map_directory, map_file = params["plans"]
    @unpack num_dists, pop_dev = params["plans"]
    @unpack node_data, pop_col, geo_units = params["plans"]

    # optional parameters (absent keys take defaults)
    measure_scores     = get(params["measure"], "measure_scores", [])
    # The measure itself is read by CycleWalk (see `energy_specs`), which accepts both
    # the [[measure.energy]] form and the older measure_scores/gamma/iso_weight one.
    measure_specs      = energy_specs(params["measure"])
    measure_params     = measure_parameters(params["measure"])
    # gamma is the weight in front of the log spanning-forest energy, and iso_weight
    # the one in front of the Polsby-Popper score — so read them off the measure
    # rather than out of a config key, which could say something the measure does not
    # (they tag the atlas name, and that tag has to be true).
    gamma      = energy_weight(measure_specs, "get_log_spanning_forests",
                               measure_params)
    iso_weight = energy_weight(measure_specs, "get_isoperimetric_score",
                               measure_params)
    # writer_stats belongs to [run] — it says what each recorded map carries, not what
    # a plan is — and that is where every config puts it and where the AIS/SMC runners
    # read it. [plans] is read as a fallback so a config written against the earlier
    # (mis)reading keeps working rather than silently recording nothing.
    writer_stats       = get(params["run"], "writer_stats",
                             get(params["plans"], "writer_stats", []))
    area_col           = get(params["plans"], "area_col", nothing)
    node_border_col    = get(params["plans"], "node_border_col", nothing)
    edge_perimeter_col = get(params["plans"], "edge_perimeter_col", nothing)
    compress = haskey(params["run"], "compress") ? "."*params["run"]["compress"] : ""

    # optional output / run settings (backward compatible: absent keys take defaults)
    output_districting = get(params["run"], "output_districting", true) # write per-map node→district
    io_mode            = get(params["run"], "io_mode", "w")             # "w" (truncate) or "a" (append)
    description        = get(params["run"], "description", "")          # recorded in the Atlas header
    rng_seed_base      = get(params["run"], "rng_seed_base", 454190)    # base RNG seed (see rng_seed below)
    blas_threads       = Int(get(params["run"], "blas_threads", 0))     # 0 = leave BLAS default; else pin
    run_diagnostics_flag = params["run"]["run_diagnostics"]

    node_data = Set(node_data) # change from vector to Set
    @assert 0 ≤ two_cycle_walk_frac ≤ 1

    rng_seed = rng_seed_base + 15123*thread_id
    #set number of total steps needed to have correct expected number of cycle_walk_steps
    steps = Int(ceil(cycle_walk_steps/two_cycle_walk_frac))
    outfreq = Int(floor(cycle_walk_out_freq/two_cycle_walk_frac))

    atlasName = atlasNameBase*"_thread"*string(thread_id)
    atlasName *= "_cyclewalkVS_2treeCycleWalk_"*string(two_cycle_walk_frac)
    if gamma > 0; atlasName *="_gamma"*string(gamma) end
    if iso_weight > 0; atlasName *="_iso"*string(iso_weight) end
    # …and every other parameter the measure reads, so a sweep over a parameter that
    # is not gamma or iso_weight does not write every point to one path
    atlasName *= parameter_tag(measure_specs, measure_params)
    atlasName *= ".jsonl"*compress
    output_file_path = joinpath(outputDirectory... , atlasName)
    make_output_dir && mkpath(dirname(output_file_path))

    pctGraphPath = joinpath(map_directory... , map_file)
    isfile(pctGraphPath) || error("map file not found: $pctGraphPath")

    #data to be added to atlas header
    ad_param = Dict{String, Any}(
        "popdev"            => pop_dev,
        "toml_config_file"  => toml_config_file,
        "map_file"          => pctGraphPath,
        "pop_col"           => pop_col,
        "geo_units"         => geo_units,
        "cycle_walk_steps"  => cycle_walk_steps,
        "cycle_walk_out_freq" => cycle_walk_out_freq,
        "rng_seed_base"     => rng_seed_base,
        "blas_threads"      => blas_threads,
        "run_diagnostics"   => run_diagnostics_flag,
    )
    # The header embeds the config *file*, which says nothing about what the command
    # line changed — so a run overridden on the command line would otherwise carry a
    # config that does not describe it.
    isempty(cli_overrides) || (ad_param["cli_overrides"] = cli_overrides)

    return (; gamma, iso_weight, cycle_walk_steps, two_cycle_walk_frac,
            outputDirectory, atlasNameBase, thread_id, cycle_walk_out_freq,
            map_directory, map_file, num_dists, pop_dev, node_data, pop_col,
            geo_units, measure_scores, measure_specs, measure_params, writer_stats,
            area_col, node_border_col,
            edge_perimeter_col, compress, output_districting, io_mode,
            description, rng_seed_base, blas_threads, run_diagnostics_flag,
            rng_seed, steps, outfreq, atlasName, output_file_path, pctGraphPath,
            ad_param)
end

"""
    take_flag!(argv, flag) -> Bool

Remove every occurrence of `flag` from `argv` and report whether there was one.
ArgMacros parses the global `ARGS`, so a flag it does not declare has to be taken out
before it runs or it is reported as an unrecognized argument.
"""
function take_flag!(argv::AbstractVector{String}, flag::AbstractString)
    found = false
    ii = 1
    while ii <= length(argv)
        if argv[ii] == flag
            deleteat!(argv, ii)
            found = true
        else
            ii += 1
        end
    end
    return found
end

"""
    take_set_overrides!(argv) -> Dict{String, Any}

Remove every `--set <table>.<key>=<value>` from `argv` and return them as a `Dict`
from dotted path to parsed value, in the order given.

Values are parsed by TOML itself, so they get exactly the types they would have had in
the config file — `--set mcmc.cycle_walk_steps=1e6` is a float, `--set
plans.num_dists=4` an integer, `--set plans.geo_units='["node_name"]'` an array. Text
that is not valid TOML is taken as a plain string, so `--set run.description=hello`
does not need quoting through the shell.
"""
function take_set_overrides!(argv::AbstractVector{String})
    overrides = OrderedDict{String, Any}()
    ii = 1
    while ii <= length(argv)
        if argv[ii] == "--set"
            ii + 1 <= length(argv) ||
                error("--set needs an argument, as --set <table>.<key>=<value>")
            spec = argv[ii+1]
            deleteat!(argv, ii:ii+1)
            eq = findfirst('=', spec)
            eq === nothing &&
                error("--set $spec is not of the form <table>.<key>=<value>")
            path = strip(spec[1:eq-1])
            text = strip(spec[eq+1:end])
            occursin('.', path) ||
                error("--set $spec: \"$path\" needs a table, as <table>.<key>")
            overrides[path] = parse_toml_value(text)
        else
            ii += 1
        end
    end
    return overrides
end

"""
    parse_toml_value(text) -> Any

Read one TOML value out of `text`, falling back to `text` itself when it is not valid
TOML. Lets a command line supply config values with the types the config file would
have given them, without quoting every string.
"""
function parse_toml_value(text::AbstractString)
    try
        return TOML.parse("value = " * text)["value"]
    catch
        return String(text)
    end
end

"""
    apply_set_overrides!(params, overrides; named_flags=Dict())

Apply `--set` overrides (see [`take_set_overrides!`](@ref)) to the parsed config
`params`, in place.

The table must already exist in the config: `[plans]`, `[measure]` and friends are
fixed, so `--set measur.gamma=2` is a typo, not a new section. Keys within a table are
open, since supplying a *new* parameter is the point — and a mistyped parameter name
surfaces later anyway, when a weight expression names something the measure does not
have.

`named_flags` maps a dotted path to `(value, flag_name)` for the flags declared
separately; a path set both ways is an error rather than a silent winner.
"""
function apply_set_overrides!(params::AbstractDict, overrides::AbstractDict;
                              named_flags::AbstractDict=Dict{String, Any}())
    for (path, value) in overrides
        table, key = split(path, '.', limit=2)
        haskey(params, table) ||
            error("--set $path: the config has no [$table] table (it has: " *
                  "$(join(sort!(collect(String.(keys(params)))), ", ")))")
        if haskey(named_flags, path)
            flag_value, flag_name = named_flags[path]
            flag_value === nothing ||
                error("$flag_name and --set $path both set the same value; use one")
        end
        params[table][key] = value
    end
    return params
end

"""
    parameter_tag(specs, parameters; exclude=("gamma", "iso_weight")) -> String

The `_name<value>` fragments an atlas file name needs so that two runs whose measures
differ cannot land on the same path: one for every parameter the measure actually
reads (see [`referenced_parameters`](@ref)) that is not already tagged by name, in
sorted order, skipping zeros.

`gamma` and `iso_weight` are excluded by default because the runners tag them
explicitly, from the weights the measure ended up with rather than from a config key.

Values are formatted with `string`, which round-trips exactly. A computed weight can
therefore read `_w0.30000000000000004`; that is the value the run used, and a
shortening format like `%g` would let two genuinely different weights share a name —
the one thing this is here to prevent.
"""
function parameter_tag(specs, parameters::AbstractDict;
                       exclude=("gamma", "iso_weight"))
    tag = ""
    for name in referenced_parameters(specs)
        name in exclude && continue
        value = get(parameters, name, 0)
        value == 0 && continue
        tag *= "_" * name * string(value)
    end
    return tag
end

"""
    ensure_writable(path, io_mode; overwrite=false)

Stop a run that would truncate an Atlas that is already there. Naming a run's
parameters into its file name (see [`parameter_tag`](@ref)) keeps most collisions from
happening, but it cannot catch all of them — a literal weight changed in the config,
or a builder given different arguments, leaves the name untouched — and quietly
destroying a finished run's output is a far worse failure than refusing to start.

Only applies when `io_mode` is `"w"`; appending to an existing file is the point of
`"a"`. Pass `overwrite=true` (the runners take `--overwrite`) to truncate anyway.
"""
function ensure_writable(path::AbstractString, io_mode::AbstractString;
                         overwrite::Bool=false)
    (io_mode == "w" && isfile(path) && !overwrite) &&
        error("$path already exists, and this run would truncate it. Pass " *
              "--overwrite to replace it, set io_mode=\"a\" to append, or move it " *
              "aside.")
    return path
end

"""
    print_params(p)

Echo the headline settings of a [`derive_params`](@ref) NamedTuple, as the runner
scripts have always done on startup.
"""
function print_params(p)
    @show p.thread_id
    @show p.steps, p.outfreq
    @show p.two_cycle_walk_frac, p.cycle_walk_steps
    @show p.num_dists, p.pop_dev
    @show p.atlasName
    @show p.node_data
    @show p.pctGraphPath
    return nothing
end
