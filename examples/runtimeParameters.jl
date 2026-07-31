## loads command line parameters
## These override the values in the TOML config file
## Everything downstream of the (overridden) TOML is computed by `derive_params` in
## parameterUtils.jl, which run_cyclewalk_extend.jl shares.
args = @dictarguments begin
    @argumentoptional Int thread_id "--thread_id"
    @argumentoptional Number two_cycle_walk_frac "--two_cycle_walk_frac" "-f"
    @argumentoptional Number cycle_walk_steps "--cycle_walk_steps" "-s"
    @argumentoptional Number gamma "--gamma"
    @argumentoptional Number iso_weight "--iso_weight"
    @argumentoptional Int num_dists "--num_dists" "-n"
    @argumentoptional Number pop_dev "--pop_dev" "-p"
    @argumentoptional Bool run_diagnostics "--run_diagnostics" "-d"
    @positionalrequired String toml_config_file
end

toml_config_file=args[:toml_config_file] # set name of TOML config file from arguments
params=TOML.parsefile(toml_config_file) # parse TOML config file



# override TOML config values with command line config values
params["run"]["thread_id"]= args[:thread_id]!=nothing ? args[:thread_id] : params["run"]["thread_id"]
params["mcmc"]["two_cycle_walk_frac"]= args[:two_cycle_walk_frac]!=nothing ?  args[:two_cycle_walk_frac] : params["mcmc"]["two_cycle_walk_frac"]
params["mcmc"]["cycle_walk_steps"]= args[:cycle_walk_steps]!=nothing ?  args[:cycle_walk_steps] : params["mcmc"]["cycle_walk_steps"]
params["measure"]["gamma"]= args[:gamma]!=nothing ? args[:gamma] : get(params["measure"], "gamma", 0.0)
params["measure"]["iso_weight"]= args[:iso_weight]!=nothing ? args[:iso_weight] : get(params["measure"], "iso_weight", 0.0)
params["plans"]["pop_dev"]= args[:pop_dev]!=nothing ? args[:pop_dev] : params["plans"]["pop_dev"]
params["plans"]["num_dists"]= args[:num_dists]!=nothing ? args[:num_dists] : params["plans"]["num_dists"]
params["run"]["run_diagnostics"]= args[:run_diagnostics]!=nothing ? args[:run_diagnostics] : params["run"]["run_diagnostics"]


include("parameterUtils.jl") # derive_params / print_params

derived_params = derive_params(params, toml_config_file)

@unpack gamma, iso_weight, cycle_walk_steps, two_cycle_walk_frac,
        outputDirectory, atlasNameBase, thread_id, cycle_walk_out_freq,
        map_directory, map_file, num_dists, pop_dev, node_data, pop_col,
        geo_units, measure_scores, measure_specs, measure_params, writer_stats,
        area_col, node_border_col,
        edge_perimeter_col, compress, output_districting, io_mode,
        description, rng_seed_base, blas_threads,
        rng_seed, steps, outfreq, atlasName, output_file_path, pctGraphPath,
        ad_param = derived_params

print_params(derived_params)
