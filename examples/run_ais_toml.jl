# Run annealed importance sampling (AIS) from a TOML config.
#
#   julia -t 4 run_ais_toml.jl toml/param_ais_ct.toml
#
# A base chain samples the base measure and each retained sample is annealed toward
# the [measure] target while its log importance weight is accumulated. ais.temper
# picks how each energy gets there: "linear" (default) ramps every energy from its
# gamma_start/iso_start (default 0) straight to its target; "path" follows each
# energy's own [[measure.energy]] weight_path expression in t instead (see
# docs/run_pt_toml.md, which documents the shared weight_path mechanism — not
# PT-specific despite the doc's name). Annealing runs execute on `ntasks` concurrent
# tasks — start Julia with threads (-t). See run_ais_ct.jl for the same run
# expressed directly in Julia, and run_cyclewalk_toml.jl for the serial TOML runner.

import Pkg
Pkg.activate(".")
Pkg.instantiate()

using RandomNumbers
using CycleWalk
using TOML, UnPack
using LinearAlgebra

include("parameterUtils.jl") # parameter_tag / ensure_writable / take_flag!

# ---------------------------------------------------------------------------
# parse config
# ---------------------------------------------------------------------------
overwrite_output = take_flag!(ARGS, "--overwrite")
length(ARGS) >= 1 ||
    error("usage: julia -t N run_ais_toml.jl <config.toml> [--overwrite]")
params = TOML.parsefile(ARGS[1])

# [plans]
@unpack num_dists, pop_dev, pop_col, geo_units, node_data = params["plans"]
@unpack map_directory, map_file = params["plans"]
area_col           = get(params["plans"], "area_col", nothing)
node_border_col    = get(params["plans"], "node_border_col", nothing)
edge_perimeter_col = get(params["plans"], "edge_perimeter_col", nothing)
num_dists = Int(num_dists)

# [measure] — each energy's `weight` is its ANNEALING TARGET (t=1) and its
# `weight_start` the BASE measure the base chain samples (t=0); the base chain must be
# able to mix under the base. A start of 0 recovers the spanning-forest base. The older
# gamma/iso_weight + gamma_start/iso_start form is read as the same thing (see
# `energy_specs`), so existing configs are unaffected.
haskey(params, "measure") || (params["measure"] = Dict{String, Any}())
haskey(params["measure"], "measure_scores") ||
    haskey(params["measure"], "energy") ||
    (params["measure"]["measure_scores"] = ["get_log_spanning_forests",
                                            "get_isoperimetric_score"])
measure_specs  = energy_specs(params["measure"])
measure_params = measure_parameters(params["measure"])
# gamma is the weight in front of the log spanning-forest energy and iso_weight the
# one in front of the Polsby-Popper score, wherever the config put them; these tag the
# atlas name, so they are read off the measure rather than out of a config key.
gamma       = energy_weight(measure_specs, "get_log_spanning_forests", measure_params)
iso_weight  = energy_weight(measure_specs, "get_isoperimetric_score",  measure_params)
gamma_start = energy_weight_start(measure_specs, "get_log_spanning_forests",
                                  measure_params)
iso_start   = energy_weight_start(measure_specs, "get_isoperimetric_score",
                                  measure_params)

# [ais]
@unpack total_steps, base_steps_per_sample, steps_per_annealing = params["ais"]
temper_kind = get(params["ais"], "temper", "linear")
temper_kind in ("linear", "path") ||
    error("ais.temper must be \"linear\" or \"path\", got \"$temper_kind\"")
total_steps           = Int(total_steps)
base_steps_per_sample = Int(base_steps_per_sample)
steps_per_annealing   = Int(steps_per_annealing)

# [run]
@unpack atlasNameBase, outputDirectory = params["run"]
two_cycle_walk_frac = get(params["run"], "two_cycle_walk_frac", 0.1)
thread_id           = Int(get(params["run"], "thread_id", 1))
rng_seed_base       = get(params["run"], "rng_seed_base", 454190)
ntasks              = Int(get(params["run"], "ntasks", Threads.nthreads()))
blas_threads        = Int(get(params["run"], "blas_threads", 0))
output_districting  = get(params["run"], "output_districting", true)
io_mode             = get(params["run"], "io_mode", "w")
description         = get(params["run"], "description", "")
compress            = "compress" in keys(params["run"]) ? "."*params["run"]["compress"] : ""
writer_stats        = get(params["run"], "writer_stats", ["get_isoperimetric_scores"])

@assert 0 ≤ two_cycle_walk_frac ≤ 1
@assert ntasks ≥ 1

# AIS auto-pins BLAS to 1 thread for ntasks>1 (the per-district log-det is small and
# each task otherwise spawns its own BLAS pool). blas_threads is an optional override.
blas_threads > 0 && BLAS.set_num_threads(blas_threads)

# ---------------------------------------------------------------------------
# build graph / partition / proposal
# ---------------------------------------------------------------------------
node_data = Set(node_data)
pctGraphPath = joinpath(map_directory..., map_file)
isfile(pctGraphPath) || error("map file not found: $pctGraphPath")
graph = Graph(pctGraphPath, pop_col, geo_units[1]; inc_node_data=node_data,
              area_col=area_col, node_border_col=node_border_col,
              edge_perimeter_col=edge_perimeter_col)

constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))

rng = PCG.PCGStateOneseq(UInt64, rng_seed_base + 15123*thread_id)
partition = LinkCutPartition(graph, constraints, num_dists; rng=rng, verbose=true)

cycle_walk    = build_two_tree_cycle_walk(constraints)
internal_walk = build_one_tree_cycle_walk(constraints)
proposal = [(two_cycle_walk_frac, cycle_walk),
            (1.0 - two_cycle_walk_frac, internal_walk)]

# ---------------------------------------------------------------------------
# target measure + tempering path
# ---------------------------------------------------------------------------
context = (graph=graph, num_dists=num_dists, pop_col=pop_col)
if temper_kind == "linear"
    measure, ramp = build_annealed_measure(measure_specs, measure_params; context=context)
    scores, target_w = annealed_smc_scores_and_targets(measure)
    K = length(scores)
    base_w = ntuple(k -> ramp[scores[k]][1], K)
    path = LinearPath{K}(t -> ntuple(k -> base_w[k] + t * (target_w[k] - base_w[k]), K))
else # "path"
    measure, path = build_path_measure(measure_specs, measure_params; context=context)
end

# ---------------------------------------------------------------------------
# writer (weight_type=Float64 so maps carry log importance weights)
# ---------------------------------------------------------------------------
atlasName  = atlasNameBase * "_thread" * string(thread_id)
atlasName *= "_ais_2cf" * string(two_cycle_walk_frac)
temper_kind == "path" && (atlasName *= "_path")   # "linear" is the default; unflagged
gamma       > 0 && (atlasName *= "_gamma" * string(gamma))
iso_weight  > 0 && (atlasName *= "_iso" * string(iso_weight))
gamma_start > 0 && (atlasName *= "_gammastart" * string(gamma_start))
iso_start   > 0 && (atlasName *= "_isostart" * string(iso_start))
# …and every other parameter the measure reads, so a sweep over one of them does not
# write every point to a single path
atlasName *= parameter_tag(measure_specs, measure_params)
atlasName *= ".jsonl" * compress
output_file_path = joinpath(outputDirectory..., atlasName)
mkpath(dirname(output_file_path))
ensure_writable(output_file_path, io_mode; overwrite=overwrite_output)

ad_param = Dict{String, Any}("popdev" => pop_dev)
writer = Writer(measure, constraints, partition, output_file_path;
                additional_parameters=ad_param, weight_type=Float64,
                output_districting=output_districting,
                io_mode=io_mode, description=description,
                config_file=ARGS[1])
for stat in writer_stats
    push_writer!(writer, getfield(CycleWalk, Symbol(stat)))
end

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
println("running AIS (temper=$temper_kind) on ", ntasks, " task(s); outputting here: ",
        output_file_path)
log_weights = run_annealed_importance_sampling!(
    partition, proposal, measure, total_steps,
    base_steps_per_sample, steps_per_annealing, rng;
    path=path, writer=writer, ntasks=ntasks,
    seed=rng_seed_base + 15123 * thread_id)
close_writer(writer)

# summarize log importance weights (stable: subtract the max before exponentiating)
println("collected ", length(log_weights), " samples")
if !isempty(log_weights)
    max_lw = maximum(log_weights)
    w = exp.(log_weights .- max_lw)
    ess = sum(w)^2 / sum(w .^ 2)
    println("log weights: min=", round(minimum(log_weights), digits=3),
            " max=", round(max_lw, digits=3))
    println("effective sample size: ", round(ess, digits=2), " of ", length(log_weights))
end
