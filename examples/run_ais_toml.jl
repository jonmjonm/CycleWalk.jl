# Run annealed importance sampling (AIS) from a TOML config.
#
#   julia -t 4 run_ais_toml.jl toml/param_ais_ct.toml
#
# A base chain samples the base measure ([measure] gamma_start/iso_start, default 0 =
# spanning-forest) and each retained sample is annealed toward the target measure (the
# [measure] gamma/iso_weight) while its log importance weight is accumulated. Annealing
# runs execute on `ntasks`
# concurrent tasks — start Julia with threads (-t). See run_ais_ct.jl for the same run
# expressed directly in Julia, and run_cyclewalk_toml.jl for the serial TOML runner.

import Pkg
Pkg.activate(".")
Pkg.instantiate()

using RandomNumbers
using CycleWalk
using TOML, UnPack
using LinearAlgebra

# ---------------------------------------------------------------------------
# parse config
# ---------------------------------------------------------------------------
length(ARGS) >= 1 || error("usage: julia -t N run_ais_toml.jl <config.toml>")
params = TOML.parsefile(ARGS[1])

# [plans]
@unpack num_dists, pop_dev, pop_col, geo_units, node_data = params["plans"]
@unpack map_directory, map_file = params["plans"]
area_col           = get(params["plans"], "area_col", nothing)
node_border_col    = get(params["plans"], "node_border_col", nothing)
edge_perimeter_col = get(params["plans"], "edge_perimeter_col", nothing)
num_dists = Int(num_dists)

# [measure] — `gamma`/`iso_weight` are the ANNEALING TARGETS (t=1). `gamma_start`/
# `iso_start` (default 0) are the BASE measure the base chain samples (t=0); the base
# chain must be able to mix under them. Default 0/0 recovers the spanning-forest base.
@unpack gamma, iso_weight = params["measure"]
gamma_start = Float64(get(params["measure"], "gamma_start", 0.0))
iso_start   = Float64(get(params["measure"], "iso_start",   0.0))
measure_scores = get(params["measure"], "measure_scores",
                     ["get_log_spanning_forests", "get_isoperimetric_score"])

# [ais]
@unpack total_steps, base_steps_per_sample, steps_per_annealing = params["ais"]
schedule = get(params["ais"], "schedule", "linear")
schedule == "linear" || error("only schedule=\"linear\" is supported (got \"$schedule\")")
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
# target measure + linear annealing schedule
# ---------------------------------------------------------------------------
target_weight = Dict{String, Float64}("get_log_spanning_forests" => gamma,
                                      "get_isoperimetric_score"   => iso_weight)
base_weight   = Dict{String, Float64}("get_log_spanning_forests" => gamma_start,
                                      "get_isoperimetric_score"   => iso_start)
measure = Measure()
for s in measure_scores
    haskey(target_weight, s) ||
        error("unsupported measure score for AIS: \"$s\" (expected one of $(keys(target_weight)))")
    push_energy!(measure, getfield(CycleWalk, Symbol(s)), target_weight[s])
end

# ramp every energy weight linearly from its base value (step 0) to its target
function modify_measure!(m::Measure, step::Int, total::Int)
    frac = step / total
    for s in measure_scores
        fnct = getfield(CycleWalk, Symbol(s))
        m.weights[fnct] = base_weight[s] + (target_weight[s] - base_weight[s]) * frac
    end
end

# ---------------------------------------------------------------------------
# writer (weight_type=Float64 so maps carry log importance weights)
# ---------------------------------------------------------------------------
atlasName  = atlasNameBase * "_thread" * string(thread_id)
atlasName *= "_ais_2cf" * string(two_cycle_walk_frac)
gamma       > 0 && (atlasName *= "_gamma" * string(gamma))
iso_weight  > 0 && (atlasName *= "_iso" * string(iso_weight))
gamma_start > 0 && (atlasName *= "_gammastart" * string(gamma_start))
iso_start   > 0 && (atlasName *= "_isostart" * string(iso_start))
atlasName *= ".jsonl" * compress
output_file_path = joinpath(outputDirectory..., atlasName)
mkpath(dirname(output_file_path))

ad_param = Dict{String, Any}("popdev" => pop_dev)
writer = Writer(measure, constraints, partition, output_file_path;
                additional_parameters=ad_param, weight_type=Float64,
                output_districting=output_districting,
                io_mode=io_mode, description=description)
for stat in writer_stats
    push_writer!(writer, getfield(CycleWalk, Symbol(stat)))
end

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
println("running AIS on ", ntasks, " task(s); outputting here: ", output_file_path)
log_weights = run_annealed_importance_sampling!(
    partition, proposal, measure, modify_measure!, total_steps,
    base_steps_per_sample, steps_per_annealing, rng;
    writer=writer, ntasks=ntasks,
    seed=rng_seed_base + 15123 * thread_id, schedule=schedule)
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
