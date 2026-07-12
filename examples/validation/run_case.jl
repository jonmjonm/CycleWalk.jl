# Validation driver: sample one test case with EITHER annealed importance sampling
# (AIS) OR a standard cycle walk, targeting the SAME measure, and write an Atlas that
# records each district's isoperimetric score plus a per-map weight.
#
#   julia -t 4 run_case.jl --case ct   --mode ais        [--dev]
#   julia -t 4 run_case.jl --case ct   --mode cyclewalk  [--dev]
#
# Target measure (both modes): get_log_spanning_forests weight gamma=0.25 and
# get_isoperimetric_score weight 0.3. AIS anneals both linearly from 0 -> target and
# writes the log importance weight as each map's weight; the cycle walk samples the
# target directly and writes weight 1. analyze.jl then compares the two outputs.
#
# --dev shrinks every count to a fast smoke size. Omit it for the production run
# (>=10,000 samples, 4000 annealing steps per sample).

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
Pkg.instantiate()

using CycleWalk
using RandomNumbers
using LinearAlgebra

# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
function argval(flag, default=nothing)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    (i == length(ARGS)) && error("missing value after $flag")
    return ARGS[i+1]
end
hasflag(flag) = flag in ARGS

case    = argval("--case")
mode    = argval("--mode")
dev     = hasflag("--dev")
case === nothing && error("usage: julia -t N run_case.jl --case {small,ct,grid,nc} --mode {ais,cyclewalk} [--dev]")
mode in ("ais", "cyclewalk") || error("--mode must be ais or cyclewalk (got $(repr(mode)))")

# ---------------------------------------------------------------------------
# per-case graph configuration
# ---------------------------------------------------------------------------
const EXAMPLES = normpath(joinpath(@__DIR__, ".."))
const REPO     = normpath(joinpath(EXAMPLES, ".."))

function case_config(case)
    if case == "small"
        # the small test-suite graph the AIS unit test already uses (4x4 pcts,
        # 2x2 counties); 16 units of pop 1, exactly divisible into 4 districts.
        # Target/proposal match the AIS test in test/test_annealed_importance_sampling.jl:
        # spanning-forest weight gamma=0.7, NO isoperimetric penalty, 0.5/0.5 mixture.
        return (; graph_path = joinpath(REPO, "test", "test_graphs", "4x4pct_2x2cnty.json"),
                pop_col="pop", geo="pct", area_col="area",
                node_border_col="border_length", edge_perimeter_col="length",
                node_data=["county","pct","pop","area","border_length"],
                num_dists=4, pop_dev=0.1,
                gamma=0.7, iso_weight=0.0, twocycle_frac=0.5)
    elseif case == "ct"
        return (; graph_path = joinpath(EXAMPLES, "data", "ct", "CT_pct20.json"),
                pop_col="POP20", geo="NAME", area_col="area",
                node_border_col="border_length", edge_perimeter_col="length",
                node_data=["COUNTY","NAME","POP20","area","border_length"],
                num_dists=5, pop_dev=0.02,
                gamma=0.25, iso_weight=0.3, twocycle_frac=0.1)
    elseif case == "grid"
        # 100 units of pop 1 into 3 districts (33/33/34): pop_dev must allow ~+2%
        return (; graph_path = joinpath(EXAMPLES, "data", "grid", "grid_graph_10_by_10.json"),
                pop_col="population", geo="node_name", area_col="area",
                node_border_col="border_length", edge_perimeter_col="length",
                node_data=["node_name","population","area","border_length","county"],
                num_dists=3, pop_dev=0.02,
                gamma=0.25, iso_weight=0.3, twocycle_frac=0.1)
    elseif case == "nc"
        # NC 2020 precincts (2650 nodes) into 14 congressional districts
        return (; graph_path = joinpath(REPO, "test", "test_graphs", "NC_pct21.json"),
                pop_col="pop2020cen", geo="prec_id", area_col="area",
                node_border_col="border_length", edge_perimeter_col="length",
                node_data=["county","prec_id","pop2020cen","area","border_length"],
                num_dists=14, pop_dev=0.02,
                gamma=0.25, iso_weight=0.3, twocycle_frac=0.1)
    else
        error("unknown --case $(repr(case)) (expected small, ct, grid, or nc)")
    end
end
cfg = case_config(case)

# ---------------------------------------------------------------------------
# run sizes (dev = fast smoke; production = the real comparison)
# ---------------------------------------------------------------------------
if dev
    steps_per_annealing   = 200
    n_samples             = 200     # AIS outer samples
    base_steps_per_sample = 50
    n_std                 = 200      # cycle-walk samples
    std_spacing           = 100
else
    steps_per_annealing   = 4000
    n_samples             = 10_000
    base_steps_per_sample = 500
    n_std                 = 10_000
    std_spacing           = 1000
end

# Size overrides (for scaling up on big machines). --anneal-steps is the main AIS
# quality knob (too few steps collapses the importance weights / ESS); the rest let the
# HPC scripts dial sample counts and spacing without editing this file.
steps_per_annealing   = parse(Int, argval("--anneal-steps", string(steps_per_annealing)))
n_samples             = parse(Int, argval("--samples",      string(n_samples)))
base_steps_per_sample = parse(Int, argval("--base-steps",   string(base_steps_per_sample)))
n_std                 = parse(Int, argval("--std-samples",  string(n_std)))
std_spacing           = parse(Int, argval("--std-spacing",  string(std_spacing)))

twocycle_frac = cfg.twocycle_frac
gamma         = cfg.gamma
iso_weight    = cfg.iso_weight   # 0 => no isoperimetric energy in the target
ntasks        = parse(Int, argval("--ntasks", string(Threads.nthreads())))
# independent seeds per mode so the two runs are statistically independent
seed = parse(UInt64, argval("--seed", string(mode == "ais" ? 0x51A5_0001 : 0x0CE1_0002)))

# ---------------------------------------------------------------------------
# graph / partition / proposal
# ---------------------------------------------------------------------------
graph = Graph(cfg.graph_path, cfg.pop_col, cfg.geo;
              inc_node_data=Set(cfg.node_data), area_col=cfg.area_col,
              node_border_col=cfg.node_border_col,
              edge_perimeter_col=cfg.edge_perimeter_col)

constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, cfg.num_dists, cfg.pop_dev))

rng = PCG.PCGStateOneseq(UInt64, seed)
partition = LinkCutPartition(graph, constraints, cfg.num_dists; rng=rng, verbose=true)

cycle_walk    = build_two_tree_cycle_walk(constraints)
internal_walk = build_one_tree_cycle_walk(constraints)
proposal = [(twocycle_frac, cycle_walk), (1.0 - twocycle_frac, internal_walk)]

# ---------------------------------------------------------------------------
# target measure (identical for both modes)
# ---------------------------------------------------------------------------
use_iso = iso_weight > 0
measure = Measure()
push_energy!(measure, get_log_spanning_forests, gamma)
use_iso && push_energy!(measure, get_isoperimetric_score, iso_weight)

# linear anneal of the target's energy weights from 0 (base) to target
function modify_measure!(m::Measure, step::Int, total::Int)
    frac = step / total
    m.weights[get_log_spanning_forests] = gamma * frac
    use_iso && (m.weights[get_isoperimetric_score] = iso_weight * frac)
end

# ---------------------------------------------------------------------------
# writer (per-district isoperimetric scores; Float64 weights only for AIS)
# ---------------------------------------------------------------------------
outdir = joinpath(EXAMPLES, "output", "validation")
mkpath(outdir)
suffix = dev ? "_dev" : ""
output_file_path = joinpath(outdir, "$(case)_$(mode)$(suffix).jsonl.gz")

# --record-path (AIS only) writes the per-sample annealing trajectory into each map:
# running log-weight, schedule fraction, per-step Δlog-weight, and the iso / spanning-
# forest energy along the path. --path-points sets the ~points per sample (default 50).
record_path = hasflag("--record-path")
path_points = parse(Int, argval("--path-points", "50"))

if mode == "ais"
    writer = Writer(measure, constraints, partition, output_file_path;
                    weight_type=Float64, description="$case AIS validation",
                    path_target_points=path_points)
else
    writer = Writer(measure, constraints, partition, output_file_path;
                    description="$case cycle-walk validation")
end
push_writer!(writer, get_isoperimetric_scores)   # per-district vector into map.data

if mode == "ais" && record_path
    push_path_writer!(writer, :log_weight)
    push_path_writer!(writer, :schedule_frac)
    push_path_writer!(writer, :delta_log_weight)
    push_path_writer!(writer, get_isoperimetric_score)
    push_path_writer!(writer, get_log_spanning_forests)
    println("  recording annealing path (~$path_points points/sample)")
end

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
println("case=$case mode=$mode dev=$dev ntasks=$ntasks steps_per_annealing=$steps_per_annealing")
println("  -> $output_file_path")

if mode == "ais"
    total_steps = n_samples * base_steps_per_sample
    log_weights = run_annealed_importance_sampling!(
        partition, proposal, measure, modify_measure!, total_steps,
        base_steps_per_sample, steps_per_annealing, rng;
        writer=writer, ntasks=ntasks)
    close_writer(writer)

    println("collected ", length(log_weights), " AIS samples")
    if !isempty(log_weights)
        max_lw = maximum(log_weights)
        w = exp.(log_weights .- max_lw)
        ess = sum(w)^2 / sum(w .^ 2)
        println("  log weights: min=", round(minimum(log_weights), digits=3),
                " max=", round(max_lw, digits=3))
        println("  effective sample size: ", round(ess, digits=1),
                " of ", length(log_weights))
    end
else
    steps = n_std * std_spacing
    run_metropolis_hastings!(partition, proposal, measure, steps, rng;
                             writer=writer, output_freq=std_spacing)
    close_writer(writer)
    println("collected ~", n_std, " cycle-walk samples (1 per ", std_spacing, " steps)")
end
