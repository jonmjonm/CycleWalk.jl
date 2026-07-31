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
    make_output_dir::Bool=true
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
    writer_stats       = get(params["plans"], "writer_stats", [])
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
