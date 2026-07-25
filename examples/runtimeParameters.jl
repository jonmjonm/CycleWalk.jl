## loads command line parameters
## These override the values in the TOML config file
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
params["measure"]["gamma"]= args[:gamma]!=nothing ? args[:gamma] : params["measure"]["gamma"]
params["measure"]["iso_weight"]= args[:iso_weight]!=nothing ? args[:iso_weight] : params["measure"]["iso_weight"]
params["plans"]["pop_dev"]= args[:pop_dev]!=nothing ? args[:pop_dev] : params["plans"]["pop_dev"]
params["plans"]["num_dists"]= args[:num_dists]!=nothing ? args[:num_dists] : params["plans"]["num_dists"]
params["run"]["run_diagnostics"]= args[:run_diagnostics]!=nothing ? args[:run_diagnostics] : params["run"]["run_diagnostics"]


#load parameters from updated TOML dictionary 
@unpack gamma,iso_weight = params["measure"]
@unpack cycle_walk_steps, two_cycle_walk_frac = params["mcmc"]
@unpack outputDirectory,atlasNameBase = params["run"]
@unpack thread_id = params["run"]
@unpack cycle_walk_out_freq = params["run"]
@unpack map_directory,map_file = params["plans"]
@unpack num_dists,pop_dev=params["plans"]
@unpack node_data,pop_col,geo_units=params["plans"]


# load optional parameters from updated TOML dictionary 
measure_scores =   "measure_scores" in  keys(params["measure"]) ? params["measure"]["measure_scores"] : []  
writer_stats =  "writer_stats" in  keys(params["plans"]) ? params["plans"]["writer_stats"] : []
area_col= "area_col" in keys(params["plans"]) ? params["plans"]["area_col"] : nothing 
node_border_col= "node_border_col" in keys(params["plans"]) ? params["plans"]["node_border_col"] : nothing 
edge_perimeter_col= "edge_perimeter_col" in keys(params["plans"]) ? params["plans"]["edge_perimeter_col"] : nothing 
compress= "compress" in keys(params["run"]) ? "."*params["run"]["compress"] : ""

# optional output / run settings (backward compatible: absent keys take defaults)
output_districting = get(params["run"], "output_districting", true) # write per-map node→district
io_mode            = get(params["run"], "io_mode", "w")             # "w" (truncate) or "a" (append)
description        = get(params["run"], "description", "")          # recorded in the Atlas header
rng_seed_base      = get(params["run"], "rng_seed_base", 454190)    # base RNG seed (see rng_seed below)
blas_threads       = Int(get(params["run"], "blas_threads", 0))     # 0 = leave BLAS default; else pin

node_data=Set(node_data) # change from vector to Set
@assert 0 ≤ two_cycle_walk_frac ≤ 1


rng_seed = rng_seed_base + 15123*thread_id
#set number of total steps needed to have correct expected number of cycle_wak_steps
steps = Int(ceil(cycle_walk_steps/two_cycle_walk_frac))  
outfreq = Int(floor(cycle_walk_out_freq/two_cycle_walk_frac))


atlasName = atlasNameBase*"_thread"*string(thread_id)
atlasName *= "_cyclewalkVS_2treeCycleWalk_"*string(two_cycle_walk_frac)
if gamma > 0; atlasName *="_gamma"*string(gamma) end
if iso_weight > 0; atlasName *="_iso"*string(iso_weight) end
atlasName *= ".jsonl"*compress
output_file_path = joinpath(outputDirectory... , atlasName)
mkpath(dirname(output_file_path))

@show thread_id
@show steps, outfreq
@show two_cycle_walk_frac,cycle_walk_steps
@show num_dists,pop_dev
@show atlasName
@show node_data

pctGraphPath = joinpath(map_directory... , map_file)
isfile(pctGraphPath) || error("map file not found: $pctGraphPath")
@show pctGraphPath

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
    "run_diagnostics"   => params["run"]["run_diagnostics"],
)