# Run annealed SMC from a TOML config, with CLI-flag overrides (flags win over TOML).
#
#   julia -t 48 run_asmc_toml.jl toml/param_annealed_smc_nc.toml
#   julia -t 48 run_asmc_toml.jl toml/param_annealed_smc_nc.toml --particles 2048 --rejuv 3000 \
#                               --collect-steps 8000 --collect-every 500
#
# A population of `particles` is annealed from the base measure (all weights 0) to the
# [measure] target along `blocks` t-steps, resampling when ESS drops and rejuvenating
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

@unpack gamma, iso_weight = params["measure"]
measure_scores = get(params["measure"], "measure_scores",
                     ["get_log_spanning_forests", "get_isoperimetric_score"])

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
graph = Graph(joinpath(map_directory..., map_file), pop_col, geo_units[1];
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
# target measure (SMC's default linear path ramps these from 0 -> target)
# ---------------------------------------------------------------------------
target_weight = Dict{String,Float64}("get_log_spanning_forests" => gamma,
                                     "get_isoperimetric_score"   => iso_weight)
measure = Measure()
for sc in measure_scores
    haskey(target_weight, sc) || error("unsupported measure score \"$sc\"")
    target_weight[sc] > 0 && push_energy!(measure, getfield(CycleWalk, Symbol(sc)), target_weight[sc])
end

# ---------------------------------------------------------------------------
# writer: run-tagged atlas name (graph / size / threads) so runs never overwrite
# ---------------------------------------------------------------------------
tag = @sprintf("_smc_p%d_b%d_r%d_t%d", particles, blocks, rejuv, nthreads)
collect_steps > 0 && (tag *= @sprintf("_c%de%d", collect_steps, collect_every))
outdir = get(ENV, "CW_OUTDIR", joinpath(@__DIR__, outputDirectory...))
mkpath(outdir)
output_file_path = joinpath(outdir, atlasNameBase * tag * ".jsonl" * compress)
writer = Writer(measure, constraints, partition, output_file_path;
                weight_type=Float64, output_districting=output_districting,
                description=description)
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
println("  target γ=$gamma iso=$iso_weight  -> $output_file_path  (~$n_maps maps)")

@time particles_out, logZ, trace =
    run_annealed_smc!(partition, proposal, measure, schedule, particles, rejuv, rng;
                      init_steps=init_steps, collect_steps=collect_steps,
                      collect_every=collect_every,
                      resample_before_collect=resample_before_collect, writer=writer)
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
