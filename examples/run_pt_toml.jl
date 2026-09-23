# Run parallel tempering from a TOML config, with CLI-flag overrides (flags win over
# TOML).
#
#   julia -t 8 run_pt_toml.jl toml/param_pt_grid.toml
#   julia -t 8 run_pt_toml.jl toml/param_pt_grid.toml --n-rungs 16 --swap-interval 300
#
# M replicas ("rungs") are tempered from the base measure to the [measure] target
# along a BetaLattice, swapping adjacent rungs every swap_interval MH steps
# (deterministic even/odd, non-reversible). Give it -t N matching or exceeding
# [pt] n_rungs to actually use the threaded backend's parallelism — advance!
# spawns exactly n_rungs tasks per round, so extra threads beyond that are wasted
# (see docs/pt_profiling_notes.md, which is why n_rungs should be sized to the
# core count you're running with, not left at a small default).

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__)))
Pkg.instantiate()

using RandomNumbers
using CycleWalk
using TOML, UnPack, ArgMacros
using LinearAlgebra
using Printf

include("parameterUtils.jl") # parameter_tag / ensure_writable / take_flag!

# ---------------------------------------------------------------------------
# CLI overrides (nothing when absent) + TOML config; CLI wins over TOML
# ---------------------------------------------------------------------------
overwrite_output = take_flag!(ARGS, "--overwrite")

args = @dictarguments begin
    @argumentoptional Int      n_rungs        "--n-rungs"
    @argumentoptional String   lattice        "--lattice"
    @argumentoptional Number   beta_min       "--beta-min"
    @argumentoptional Int      swap_interval  "--swap-interval"
    @argumentoptional Int      n_rounds       "--n-rounds"
    @argumentoptional Int      init_steps     "--init-steps"
    @argumentoptional String   backend        "--backend"
    @argumentoptional String   temper         "--temper"
    @argumentoptional String   write_rungs    "--write-rungs"
    @argumentoptional Int      output_every   "--output-every"
    @argumentoptional Number   gamma          "--gamma"
    @argumentoptional Number   iso_weight     "--iso-weight"
    @argumentoptional Int      seed           "--seed"
    @positionalrequired String toml_config_file
end
params = TOML.parsefile(args[:toml_config_file])

function cli!(section, key, v)
    haskey(params, section) || (params[section] = Dict{String,Any}())
    v !== nothing && (params[section][key] = v)
end
cli!("pt", "n_rungs",       args[:n_rungs])
cli!("pt", "lattice",       args[:lattice])
cli!("pt", "beta_min",      args[:beta_min])
cli!("pt", "swap_interval", args[:swap_interval])
cli!("pt", "n_rounds",      args[:n_rounds])
cli!("pt", "init_steps",    args[:init_steps])
cli!("pt", "backend",       args[:backend])
cli!("pt", "temper",        args[:temper])
cli!("pt", "write_rungs",   args[:write_rungs])
cli!("pt", "output_every",  args[:output_every])
cli!("measure", "gamma",      args[:gamma])
cli!("measure", "iso_weight", args[:iso_weight])
cli!("run", "seed", args[:seed])

# ---------------------------------------------------------------------------
# [plans]
# ---------------------------------------------------------------------------
@unpack num_dists, pop_dev, pop_col, geo_units, node_data = params["plans"]
@unpack map_directory, map_file = params["plans"]
area_col           = get(params["plans"], "area_col", nothing)
node_border_col    = get(params["plans"], "node_border_col", nothing)
edge_perimeter_col = get(params["plans"], "edge_perimeter_col", nothing)
num_dists = Int(num_dists)

# ---------------------------------------------------------------------------
# [measure] — default to the usual gamma/iso pair if the config names neither form
# ---------------------------------------------------------------------------
haskey(params, "measure") || (params["measure"] = Dict{String, Any}())
haskey(params["measure"], "measure_scores") ||
    haskey(params["measure"], "energy") ||
    (params["measure"]["measure_scores"] = ["get_log_spanning_forests",
                                            "get_isoperimetric_score"])
measure_specs  = energy_specs(params["measure"])
measure_params = measure_parameters(params["measure"])
gamma      = energy_weight(measure_specs, "get_log_spanning_forests", measure_params)
iso_weight = energy_weight(measure_specs, "get_isoperimetric_score",  measure_params)

# ---------------------------------------------------------------------------
# [pt]
# ---------------------------------------------------------------------------
pt = params["pt"]
n_rungs       = Int(get(pt, "n_rungs", 8))
lattice_kind  = get(pt, "lattice", "linear")
beta_min      = Float64(get(pt, "beta_min", 0.05))
swap_interval = Int(get(pt, "swap_interval", 500))
n_rounds      = Int(get(pt, "n_rounds", 2000))
init_steps    = Int(get(pt, "init_steps", 2000))
backend_kind  = get(pt, "backend", "threaded")
temper_kind   = get(pt, "temper", "linear")
write_rungs   = Symbol(get(pt, "write_rungs", "target"))
output_every  = Int(get(pt, "output_every", 1))

backend_kind in ("serial", "threaded") ||
    error("pt.backend must be \"serial\" or \"threaded\" for now (\"distributed\" " *
          "is not implemented yet — see memories/parallel_tempering_implementation.md " *
          "task 9), got \"$backend_kind\"")
temper_kind in ("linear", "path") ||
    error("pt.temper must be \"linear\" or \"path\", got \"$temper_kind\"")
write_rungs in (:target, :all, :none) ||
    error("pt.write_rungs must be \"target\", \"all\", or \"none\", got " *
          "\"$write_rungs\"")

if haskey(pt, "betas")
    lattice = BetaLattice(Float64.(pt["betas"]))
elseif lattice_kind == "linear"
    lattice = linear_betas(n_rungs)
elseif lattice_kind == "geometric"
    lattice = geometric_betas(n_rungs; beta_min=beta_min)
else
    error("pt.lattice must be \"linear\" or \"geometric\" (or supply pt.betas " *
          "explicitly), got \"$lattice_kind\"")
end
n_rungs = length(lattice)   # betas, if given explicitly, is authoritative

# ---------------------------------------------------------------------------
# [run]
# ---------------------------------------------------------------------------
@unpack atlasNameBase, outputDirectory = params["run"]
two_cycle_walk_frac = get(params["run"], "two_cycle_walk_frac", 0.1)
writer_stats        = get(params["run"], "writer_stats", String[])
seed                = UInt64(get(params["run"], "seed", 0x5A1D0007))
output_districting  = get(params["run"], "output_districting", true)
description         = get(params["run"], "description", "")
compress            = "compress" in keys(params["run"]) ? "." * params["run"]["compress"] : ""
nthreads            = Threads.nthreads()

n_rungs > nthreads && backend_kind == "threaded" &&
    @warn "pt.n_rungs=$n_rungs but Julia has $nthreads thread(s); ThreadedBackend " *
          "spawns one task per rung, so this run cannot use more than $nthreads " *
          "of them at once. Start Julia with -t $n_rungs (or more) to use every " *
          "rung's worth of parallelism (see docs/pt_profiling_notes.md)."

# ---------------------------------------------------------------------------
# graph / partition / proposal
# ---------------------------------------------------------------------------
pctGraphPath = joinpath(map_directory..., map_file)
isfile(pctGraphPath) || error("map file not found: $pctGraphPath")
base_graph = BaseGraph(pctGraphPath, pop_col; inc_node_data=Set(node_data),
                       area_col=area_col, node_border_col=node_border_col,
                       edge_perimeter_col=edge_perimeter_col)
# [plans.derive]: named columns computed from existing ones — see
# docs/run_cyclewalk_toml.md. Must run before `Graph` below, since geo_units/pop_col
# can name a derived column, and by the time anything looks a column up it has to
# already exist.
haskey(params["plans"], "derive") &&
    derive_node_columns!(base_graph, params["plans"]["derive"])
graph = Graph(base_graph, geo_units[1])
constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))
rng = PCG.PCGStateOneseq(UInt64, seed)
partition = LinkCutPartition(graph, constraints, num_dists; rng=rng, verbose=true)
cycle_walk    = build_two_tree_cycle_walk(constraints)
internal_walk = build_one_tree_cycle_walk(constraints)
proposal = [(two_cycle_walk_frac, cycle_walk), (1.0 - two_cycle_walk_frac, internal_walk)]

# ---------------------------------------------------------------------------
# target measure + tempering path
# ---------------------------------------------------------------------------
context = (graph=graph, base_graph=base_graph, num_dists=num_dists, pop_col=pop_col)
if temper_kind == "linear"
    measure, ramp = build_annealed_measure(measure_specs, measure_params;
                                           context=context, default_start=0.0)
    scores, target_w = annealed_smc_scores_and_targets(measure)
    K = length(scores)
    base_w = ntuple(k -> ramp[scores[k]][1], K)
    path = LinearPath{K}(t -> ntuple(k -> base_w[k] + t * (target_w[k] - base_w[k]), K))
else # "path"
    measure, path = build_path_measure(measure_specs, measure_params; context=context)
    scores, _ = annealed_smc_scores_and_targets(measure)
end

# ---------------------------------------------------------------------------
# backend
# ---------------------------------------------------------------------------
backend = backend_kind == "serial" ? SerialBackend(proposal) :
          ThreadedBackend(proposal, n_rungs)

# ---------------------------------------------------------------------------
# heat bath (optional)
# ---------------------------------------------------------------------------
heat_bath = nothing
if haskey(pt, "heat_bath")
    hb = pt["heat_bath"]
    haskey(hb, "source") || error("[pt.heat_bath] needs a \"source\" atlas path")
    hb_rung      = Int(get(hb, "rung", 1))
    hb_burn_in   = Int(get(hb, "burn_in", 0))
    hb_n_samples = Int(get(hb, "n_samples", cld(n_rounds, 2) + 20))
    heat_bath = HeatBath(hb["source"], scores, graph; rung=hb_rung,
                         burn_in=hb_burn_in, n_samples=hb_n_samples, rng=rng,
                         context=context)
    println("heat bath: $(hb_n_samples) samples from $(hb["source"]) at rung $hb_rung")
end

# ---------------------------------------------------------------------------
# writer(s): run-tagged atlas name(s) so runs never overwrite
# ---------------------------------------------------------------------------
"""
    insert_tag(path, tag)

Insert `tag` into an atlas file name, before its `.jsonl[.gz]` extension.
"""
function insert_tag(path::AbstractString, tag::AbstractString)
    i = findfirst(".jsonl", path)
    stem, ext = i === nothing ? (path, "") : (path[1:first(i)-1], path[first(i):end])
    return stem * tag * ext
end

tag = @sprintf("_pt_r%d_si%d_nr%d_t%d", n_rungs, swap_interval, n_rounds, nthreads)
tag *= "_" * backend_kind * "_" * temper_kind
gamma      > 0 && (tag *= "_gamma" * string(gamma))
iso_weight > 0 && (tag *= "_iso" * string(iso_weight))
tag *= parameter_tag(measure_specs, measure_params)
outdir = get(ENV, "CW_OUTDIR", joinpath(@__DIR__, outputDirectory...))
mkpath(outdir)
output_file_path = joinpath(outdir, atlasNameBase * tag * ".jsonl" * compress)

function build_writer(path)
    ensure_writable(path, "w"; overwrite=overwrite_output)
    w = Writer(measure, constraints, partition, path;
              output_districting=output_districting, description=description,
              config_file=args[:toml_config_file])
    for stat in writer_stats
        push_writer!(w, getfield(CycleWalk, Symbol(stat)))
    end
    return w
end

writers = if write_rungs == :target
    build_writer(output_file_path)
elseif write_rungs == :all
    [build_writer(insert_tag(output_file_path, "_beta$(k)")) for k in 1:n_rungs]
else
    nothing
end

# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
println("PT $(atlasNameBase): n_rungs=$n_rungs swap_interval=$swap_interval " *
        "n_rounds=$n_rounds init=$init_steps backend=$backend_kind " *
        "temper=$temper_kind threads=$nthreads")
println("  target: γ=$gamma iso=$iso_weight")
println("  -> $output_file_path" * (write_rungs == :all ? " (+ $(n_rungs-1) more, one per rung)" : ""))

diagnostics = PTDiagnostics(n_rungs)
@time ensemble, diag = run_parallel_tempering!(
    partition, proposal, measure, lattice, swap_interval, n_rounds, rng;
    path=path, backend=backend, heat_bath=heat_bath, init_steps=init_steps,
    write_rungs=write_rungs, output_every=output_every, writers=writers,
    diagnostics=diagnostics, seed=seed)

if writers !== nothing
    foreach(close_writer, writers isa Vector ? writers : [writers])
end

# ---------------------------------------------------------------------------
# report: swap rates per pair, round trips, straggler gap
# ---------------------------------------------------------------------------
rates = swap_rate(diag)
println("\nswap rates (rung k <-> k+1), target is roughly EQUAL across pairs:")
for (k, r) in enumerate(rates)
    @printf("  %2d <-> %2d : %s\n", k, k+1, isnan(r) ? "n/a" : @sprintf("%.3f", r))
end
println("round trips per walker: ", diag.round_trips)
mean_gap = isempty(diag.straggler_gap) ? NaN :
          sum(diag.straggler_gap) / length(diag.straggler_gap)
@printf("mean straggler gap: %.4fs (use to size swap_interval)\n", mean_gap)
if heat_bath !== nothing
    @printf("heat bath: %d/%d accepted, %d samples left unused\n",
            diag.bath_accepts, diag.bath_attempts, length(heat_bath.samples))
end
