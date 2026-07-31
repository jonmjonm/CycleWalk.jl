# Run annealed SMC from a TOML config, with CLI-flag overrides (flags win over TOML).
#
#   julia -t 48 run_asmc_toml.jl toml/param_annealed_smc_nc.toml
#   julia -t 48 run_asmc_toml.jl toml/param_annealed_smc_nc.toml --particles 2048 --rejuv 3000 \
#                               --collect-steps 8000 --collect-every 500
#
# A population of `particles` is annealed from the base measure ([measure]
# gamma_start/iso_start, default 0 = all weights 0) to the [measure] target along
# `blocks` t-steps, resampling when ESS drops and rejuvenating
# `rejuv` MH steps per block. With collect_steps>0, after t=1 the sampler keeps sampling
# the target and emits each particle every collect_every steps (option-1 amplification:
# particles * collect_steps/collect_every samples). See run_asmc.jl for the CLI-only
# variant and run_ais_toml.jl for the AIS TOML runner this mirrors.

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__)))
Pkg.instantiate()

using RandomNumbers
using CycleWalk
using TOML, UnPack, ArgMacros
using LinearAlgebra
using Printf

# ---------------------------------------------------------------------------
# CLI overrides (nothing when absent) + TOML config; CLI wins over TOML
# ---------------------------------------------------------------------------
args = @dictarguments begin
    @argumentoptional String  schedule      "--schedule"
    @argumentoptional Int      particles     "--particles"
    @argumentoptional Int      blocks        "--blocks"
    @argumentoptional Int      rejuv         "--rejuv"
    @argumentoptional Int      init_steps    "--init-steps"
    @argumentoptional Int      collect_steps "--collect-steps"
    @argumentoptional Int      collect_every "--collect-every"
    @argumentflag              no_resample_before_collect "--no-resample-before-collect"
    @argumentoptional Number   ess_frac      "--ess-frac"
    @argumentoptional Number   ess_target    "--ess-target"
    @argumentoptional Number   gamma         "--gamma"
    @argumentoptional Number   iso_weight    "--iso-weight"
    @argumentoptional Number   gamma_start   "--gamma-start"
    @argumentoptional Number   iso_start     "--iso-start"
    @argumentoptional Int      seed          "--seed"
    @positionalrequired String toml_config_file
end
params = TOML.parsefile(args[:toml_config_file])

# apply a CLI override (if given) onto params[section][key]; else keep TOML / default
function cli!(section, key, v)
    haskey(params, section) || (params[section] = Dict{String,Any}())
    v !== nothing && (params[section][key] = v)
end
cli!("smc", "schedule",      args[:schedule])
cli!("smc", "particles",     args[:particles])
cli!("smc", "blocks",        args[:blocks])
cli!("smc", "rejuv",         args[:rejuv])
cli!("smc", "init_steps",    args[:init_steps])
cli!("smc", "collect_steps", args[:collect_steps])
cli!("smc", "collect_every", args[:collect_every])
# CLI --no-resample-before-collect forces it off (else keep TOML value / default true)
args[:no_resample_before_collect] && (params["smc"]["resample_before_collect"] = false)
cli!("smc", "ess_frac",      args[:ess_frac])
cli!("smc", "ess_target",    args[:ess_target])
cli!("measure", "gamma",      args[:gamma])
cli!("measure", "iso_weight", args[:iso_weight])
cli!("measure", "gamma_start", args[:gamma_start])
cli!("measure", "iso_start",   args[:iso_start])
cli!("run", "seed",          args[:seed])

# ---------------------------------------------------------------------------
# unpack (with defaults for optional keys)
# ---------------------------------------------------------------------------
@unpack num_dists, pop_dev, pop_col, geo_units, node_data = params["plans"]
@unpack map_directory, map_file = params["plans"]
area_col           = get(params["plans"], "area_col", nothing)
node_border_col    = get(params["plans"], "node_border_col", nothing)
edge_perimeter_col = get(params["plans"], "edge_perimeter_col", nothing)
num_dists = Int(num_dists)

# Each energy's `weight` is its target (t=1) and its `weight_start` the base (t=0).
# The older gamma/iso_weight + gamma_start/iso_start form reads as the same thing (see
# `energy_specs`), so existing configs are unaffected.
haskey(params, "measure") || (params["measure"] = Dict{String, Any}())
haskey(params["measure"], "measure_scores") ||
    haskey(params["measure"], "energy") ||
    (params["measure"]["measure_scores"] = ["get_log_spanning_forests",
                                            "get_isoperimetric_score"])
measure_specs  = energy_specs(params["measure"])
measure_params = measure_parameters(params["measure"])
# read off the measure, not out of a config key, so the atlas name cannot disagree
gamma       = energy_weight(measure_specs, "get_log_spanning_forests", measure_params)
iso_weight  = energy_weight(measure_specs, "get_isoperimetric_score",  measure_params)
gamma_start = energy_weight_start(measure_specs, "get_log_spanning_forests",
                                  measure_params)
iso_start   = energy_weight_start(measure_specs, "get_isoperimetric_score",
                                  measure_params)

s = params["smc"]
schedule_kind = get(s, "schedule", "fixed")
particles     = Int(get(s, "particles", 512))
blocks        = Int(get(s, "blocks", 100))
rejuv         = Int(get(s, "rejuv", 300))
init_steps    = Int(get(s, "init_steps", 2000))
collect_steps = Int(get(s, "collect_steps", 0))
collect_every = Int(get(s, "collect_every", 100))
resample_before_collect = Bool(get(s, "resample_before_collect", true))
ess_frac      = Float64(get(s, "ess_frac", 0.5))
ess_target    = Float64(get(s, "ess_target", 0.5))
schedule_kind in ("fixed", "adaptive") || error("smc.schedule must be fixed or adaptive")

r = params["run"]
@unpack atlasNameBase, outputDirectory = r
two_cycle_walk_frac = get(r, "two_cycle_walk_frac", 0.1)
writer_stats        = get(r, "writer_stats", ["get_isoperimetric_scores", "get_log_spanning_trees"])
seed                = UInt64(get(r, "seed", 0x5A1D0007))
output_districting  = get(r, "output_districting", true)
description         = get(r, "description", "")
compress            = "compress" in keys(r) ? "." * r["compress"] : ""
nthreads            = Threads.nthreads()

# ---------------------------------------------------------------------------
# graph / partition / proposal
# ---------------------------------------------------------------------------
pctGraphPath = joinpath(map_directory..., map_file)
isfile(pctGraphPath) || error("map file not found: $pctGraphPath")
graph = Graph(pctGraphPath, pop_col, geo_units[1];
              inc_node_data=Set(node_data), area_col=area_col,
              node_border_col=node_border_col, edge_perimeter_col=edge_perimeter_col)
constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))
rng = PCG.PCGStateOneseq(UInt64, seed)
partition = LinkCutPartition(graph, constraints, num_dists; rng=rng, verbose=true)
cycle_walk    = build_two_tree_cycle_walk(constraints)
internal_walk = build_one_tree_cycle_walk(constraints)
proposal = [(two_cycle_walk_frac, cycle_walk), (1.0 - two_cycle_walk_frac, internal_walk)]

# ---------------------------------------------------------------------------
# target measure + base measure. The linear path ramps each term from its base
# weight (gamma_start/iso_start, default 0) at t=0 to its target at t=1. A term is
# annealed if either endpoint is nonzero.
# ---------------------------------------------------------------------------
# The target measure, plus the (base, target) weight of every energy in it. An energy
# that is zero the whole way is dropped by push_energy!; one annealed *down to* zero is
# kept (allow_zero), since the path cannot ramp an energy the measure does not have.
measure, ramp = build_annealed_measure(measure_specs, measure_params;
                                       context=(graph=graph, num_dists=num_dists,
                                                pop_col=pop_col))

# Linear path from base weights (t=0) to target weights (t=1), aligned to the exact
# `scores` order the driver freezes. All-zero base recovers the default linear_path.
# `ramp` is keyed by the energy function, which is what `scores` holds — no lookup
# back through a name, which a built energy would not have.
scores, target_w = annealed_smc_scores_and_targets(measure)
K = length(scores)
base_w  = ntuple(k -> ramp[scores[k]][1], K)
path = LinearPath{K}(t -> ntuple(k -> base_w[k] + t * (target_w[k] - base_w[k]), K))

# ---------------------------------------------------------------------------
# writer: run-tagged atlas name (graph / size / threads) so runs never overwrite
# ---------------------------------------------------------------------------
tag = @sprintf("_smc_p%d_b%d_r%d_t%d", particles, blocks, rejuv, nthreads)
collect_steps > 0 && (tag *= @sprintf("_c%de%d", collect_steps, collect_every))
(gamma_start > 0 || iso_start > 0) && (tag *= @sprintf("_start_g%g_i%g", gamma_start, iso_start))
outdir = get(ENV, "CW_OUTDIR", joinpath(@__DIR__, outputDirectory...))
mkpath(outdir)
output_file_path = joinpath(outdir, atlasNameBase * tag * ".jsonl" * compress)
writer = Writer(measure, constraints, partition, output_file_path;
                weight_type=Float64, output_districting=output_districting,
                description=description, config_file=args[:toml_config_file])
for stat in writer_stats
    push_writer!(writer, getfield(CycleWalk, Symbol(stat)))
end

# ---------------------------------------------------------------------------
# schedule + run
# ---------------------------------------------------------------------------
schedule = schedule_kind == "adaptive" ?
    AdaptiveTempering(; ess_target=ess_target) :
    FixedSchedule(range(0, 1; length=blocks + 1); ess_frac=ess_frac)

n_maps = particles * (collect_steps > 0 ? div(collect_steps, max(collect_every, 1)) : 1)
println("SMC $(atlasNameBase)/$(schedule_kind): N=$particles blocks=$blocks rejuv=$rejuv ",
        "init=$init_steps collect=$collect_steps/$collect_every threads=$nthreads")
println("  base γ=$gamma_start iso=$iso_start  ->  target γ=$gamma iso=$iso_weight")
println("  -> $output_file_path  (~$n_maps maps)")

@time particles_out, logZ, trace =
    run_annealed_smc!(partition, proposal, measure, schedule, particles, rejuv, rng;
                      path=path,
                      init_steps=init_steps, collect_steps=collect_steps,
                      collect_every=collect_every,
                      resample_before_collect=resample_before_collect,
                      writer=writer, seed=seed)
close_writer(writer)

# ---------------------------------------------------------------------------
# report: per-block ESS trace + summary
# ---------------------------------------------------------------------------
println(" block   t        ESS%   resampled")
for (b, rc) in enumerate(trace)
    @printf("  %3d   %.4f   %5.1f%%   %s\n", b, rc.t, 100 * rc.ess / particles,
            rc.resampled ? "yes" : "")
end
final_ess = CycleWalk.ess_from_logw([p.logW for p in particles_out])
min_ess   = minimum(rc.ess for rc in trace)
@printf("wrote ~%d maps | blocks=%d resamples=%d min-block-ESS=%.1f (%.0f%%) final-ESS=%.1f (%.0f%%) logZ=%.4f\n",
        n_maps, length(trace), count(rc -> rc.resampled, trace),
        min_ess, 100 * min_ess / particles, final_ess, 100 * final_ess / particles, logZ)
