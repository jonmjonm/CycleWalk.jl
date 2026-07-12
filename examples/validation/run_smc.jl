# Validation driver for annealed SMC: sample a test case with run_annealed_smc!,
# targeting the SAME measure as run_case.jl, and write an Atlas of the final
# particles (each map's weight = its LOG importance weight, AIS convention) so
# analyze.jl --smc can compare its marginals to the cycle walk.
#
#   julia -t 4 run_smc.jl --case grid --schedule fixed    [--dev]
#   julia -t 4 run_smc.jl --case grid --schedule adaptive [--dev]
#
# Mirrors run_case.jl's graph/partition/proposal/measure setup exactly.

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
Pkg.instantiate()

using CycleWalk
using RandomNumbers
using LinearAlgebra
using Printf

function argval(flag, default=nothing)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    (i == length(ARGS)) && error("missing value after $flag")
    return ARGS[i+1]
end
hasflag(flag) = flag in ARGS

case     = argval("--case", "grid")
sched_kind = argval("--schedule", "fixed")
dev      = hasflag("--dev")
sched_kind in ("fixed", "adaptive") || error("--schedule must be fixed or adaptive")

const EXAMPLES = normpath(joinpath(@__DIR__, ".."))
const REPO     = normpath(joinpath(EXAMPLES, ".."))

# per-case config (subset of run_case.jl case_config; grid/ct/nc)
function case_config(case)
    if case == "grid"
        return (; graph_path=joinpath(EXAMPLES,"data","grid","grid_graph_10_by_10.json"),
                pop_col="population", geo="node_name", area_col="area",
                node_border_col="border_length", edge_perimeter_col="length",
                node_data=["node_name","population","area","border_length","county"],
                num_dists=3, pop_dev=0.02, gamma=0.25, iso_weight=0.3, twocycle_frac=0.1)
    elseif case == "ct"
        return (; graph_path=joinpath(EXAMPLES,"data","ct","CT_pct20.json"),
                pop_col="POP20", geo="NAME", area_col="area",
                node_border_col="border_length", edge_perimeter_col="length",
                node_data=["COUNTY","NAME","POP20","area","border_length"],
                num_dists=5, pop_dev=0.02, gamma=0.25, iso_weight=0.3, twocycle_frac=0.1)
    elseif case == "nc"
        return (; graph_path=joinpath(REPO,"test","test_graphs","NC_pct21.json"),
                pop_col="pop2020cen", geo="prec_id", area_col="area",
                node_border_col="border_length", edge_perimeter_col="length",
                node_data=["county","prec_id","pop2020cen","area","border_length"],
                num_dists=14, pop_dev=0.02, gamma=0.25, iso_weight=0.3, twocycle_frac=0.1)
    else
        error("unknown --case $(repr(case)) (grid, ct, nc)")
    end
end
cfg = case_config(case)

# run sizes: dev picks smaller defaults, but every knob is CLI-overridable so the
# accuracy/cost trade-off (rejuv_steps, ess_target, blocks) can be explored in dev.
d(x) = dev ? x[1] : x[2]
n_particles = parse(Int,     argval("--particles",  string(d((64, 512)))))
n_blocks    = parse(Int,     argval("--blocks",     string(d((40, 100)))))
rejuv_steps = parse(Int,     argval("--rejuv",      string(d((50, 200)))))
init_steps  = parse(Int,     argval("--init-steps", string(d((500, 2000)))))
ess_target  = parse(Float64, argval("--ess-target", string(0.5)))

gamma      = parse(Float64, argval("--gamma", string(cfg.gamma)))
iso_weight = parse(Float64, argval("--iso", string(cfg.iso_weight)))
seed       = parse(UInt64, argval("--seed", "0x5A1D0007"))

# graph / partition / proposal (identical to run_case.jl)
graph = Graph(cfg.graph_path, cfg.pop_col, cfg.geo;
              inc_node_data=Set(cfg.node_data), area_col=cfg.area_col,
              node_border_col=cfg.node_border_col, edge_perimeter_col=cfg.edge_perimeter_col)
constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, cfg.num_dists, cfg.pop_dev))
rng = PCG.PCGStateOneseq(UInt64, seed)
partition = LinkCutPartition(graph, constraints, cfg.num_dists; rng=rng, verbose=true)
cycle_walk    = build_two_tree_cycle_walk(constraints)
internal_walk = build_one_tree_cycle_walk(constraints)
proposal = [(cfg.twocycle_frac, cycle_walk), (1.0 - cfg.twocycle_frac, internal_walk)]

# target measure (identical to run_case.jl)
measure = Measure()
push_energy!(measure, get_log_spanning_forests, gamma)
iso_weight > 0 && push_energy!(measure, get_isoperimetric_score, iso_weight)

# writer: Float64 weights (log importance weights), same observables as run_case AIS
outdir = get(ENV, "CW_OUTDIR", joinpath(EXAMPLES, "output", "validation"))
mkpath(outdir)
suffix = dev ? "_dev" : ""
output_file_path = joinpath(outdir, "$(case)_smc$(suffix).jsonl.gz")
writer = Writer(measure, constraints, partition, output_file_path;
                weight_type=Float64, description="$case SMC validation")
push_writer!(writer, get_isoperimetric_scores)
push_writer!(writer, get_log_spanning_trees)

# schedule
schedule = sched_kind == "fixed" ?
    FixedSchedule(range(0, 1; length = n_blocks + 1); ess_frac = 0.5) :
    AdaptiveTempering(; ess_target = ess_target)

println("SMC $case/$sched_kind: N=$n_particles rejuv=$rejuv_steps init=$init_steps ",
        "(γ=$gamma iso=$iso_weight)\n  -> $output_file_path")
@time particles, logZ, trace =
    run_annealed_smc!(partition, proposal, measure, schedule,
                      n_particles, rejuv_steps, rng;
                      init_steps=init_steps, writer=writer)
close_writer(writer)

final_ess = CycleWalk.ess_from_logw([p.logW for p in particles])
n_resamples = count(r -> r.resampled, trace)
@printf("wrote %d particles | blocks=%d resamples=%d final-ESS=%.1f (%.0f%%) logZ=%.4f\n",
        length(particles), length(trace), n_resamples, final_ess,
        100*final_ess/n_particles, logZ)
println("analyze:  julia analyze.jl --case $case --smc", dev ? " --dev" : "")
