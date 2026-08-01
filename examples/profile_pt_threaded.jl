# Threaded-backend scaling profile for parallel tempering.
#
#   julia --project=<examples-dir> -t N profile_pt_threaded.jl [backend] [graph] [M]
#
# backend defaults to "threaded" ("serial" for the nthreads=1 reference point).
# graph defaults to "ct" (also: hex, grid, nc, oh).
# M (rung count) defaults to max(nthreads, 8) — ThreadedBackend spawns exactly M
# tasks/round, so M must be >= nthreads or scaling flatlines below the thread count.
# Prints one CSV line to stdout:
#   nthreads,backend,graph,M,elapsed_s,total_mh_steps,steps_per_sec,mean_straggler_gap,mean_swap_rate

const EXAMPLES = normpath(@__DIR__)

import Pkg
Pkg.activate(EXAMPLES)

using CycleWalk
using RandomNumbers

backend_kind = length(ARGS) >= 1 ? ARGS[1] : "threaded"
graph_kind   = length(ARGS) >= 2 ? ARGS[2] : "ct"

gamma      = 0.3
iso_weight = 0.3

if graph_kind == "ct"
    graph_path = joinpath(EXAMPLES, "data", "ct", "CT_pct20.json")
    num_dists  = 5
    pop_dev    = 0.02
    graph = Graph(graph_path, "POP20", "NAME";
                  inc_node_data=Set(["COUNTY", "NAME", "POP20", "area", "border_length"]),
                  area_col="area", node_border_col="border_length",
                  edge_perimeter_col="length")
elseif graph_kind == "hex"
    graph_path = joinpath(EXAMPLES, "data", "hex", "hex_graph_10_by_10.json")
    num_dists  = 5
    pop_dev    = 0.0001
    graph = Graph(graph_path, "population", "node_name";
                  inc_node_data=Set(["county", "node_name", "population", "area",
                                     "border_length"]),
                  area_col="area", node_border_col="border_length",
                  edge_perimeter_col="length")
elseif graph_kind == "nc"
    # "uid" (county_prec_id) is a synthetic column added to the downloaded JSON:
    # prec_id alone is not unique across counties (see the codedoc geographic
    # page), and the raw integer "id" column breaks MultiLevelGraph, which needs
    # a unique STRING level column.
    graph_path = joinpath(EXAMPLES, "data", "nc", "NC_pct21.json")
    num_dists  = 14
    pop_dev    = 0.02
    graph = Graph(graph_path, "pop2020cen", "uid";
                  inc_node_data=Set(["uid", "county", "pop2020cen", "area",
                                     "border_length"]),
                  area_col="area", node_border_col="border_length",
                  edge_perimeter_col="length")
elseif graph_kind == "oh"
    graph_path = joinpath(EXAMPLES, "data", "oh", "OHpct20.json")
    num_dists  = 15
    pop_dev    = 0.02
    graph = Graph(graph_path, "POP20", "NAME";
                  inc_node_data=Set(["COUNTY", "NAME", "POP20", "area",
                                     "border_length"]),
                  area_col="area", node_border_col="border_length",
                  edge_perimeter_col="length")
elseif graph_kind == "grid"
    graph_path = joinpath(EXAMPLES, "data", "grid", "grid_graph_10_by_10.json")
    num_dists  = 3
    pop_dev    = 0.02
    graph = Graph(graph_path, "population", "node_name";
                  inc_node_data=Set(["node_name", "population", "area",
                                     "border_length", "county"]),
                  area_col="area", node_border_col="border_length",
                  edge_perimeter_col="length")
else
    error("unknown graph \"$graph_kind\" (expected \"ct\", \"hex\", \"grid\", \"nc\", or \"oh\")")
end

constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))

cycle_walk    = build_lifted_tree_cycle_walk(constraints)
internal_walk = build_internal_forest_walk(constraints)
proposal = [(0.1, cycle_walk), (0.9, internal_walk)]

measure = Measure()
push_energy!(measure, get_log_spanning_forests, gamma)
push_energy!(measure, get_isoperimetric_score, iso_weight)

# M (rung count) must be >= the thread count to give ThreadedBackend anything to
# scale onto: advance! spawns exactly M tasks per round, one per replica, so with
# M=8 fixed, scaling flatlines at 8 threads regardless of how many cores are
# available. Default to nthreads (or 8, whichever is larger) unless overridden.
M = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : max(Threads.nthreads(), 8)
lattice = linear_betas(M)

function run_once(swap_interval, n_rounds, seed)
    rng = PCG.PCGStateOneseq(UInt64, seed)
    partition = LinkCutPartition(graph, constraints, num_dists; rng=rng, verbose=false)
    backend = backend_kind == "serial" ? SerialBackend(proposal) :
              backend_kind == "threaded" ? ThreadedBackend(proposal, M) :
              error("unknown backend \"$backend_kind\" (expected \"serial\" or \"threaded\")")
    ensemble, diag = run_parallel_tempering!(partition, proposal, measure, lattice,
                                             swap_interval, n_rounds, rng;
                                             init_steps=0, backend=backend,
                                             write_rungs=:none, writers=nothing)
    return ensemble, diag
end

# warm-up: absorb JIT/precompilation so the timed run measures steady-state throughput
run_once(5, 3, 1)

swap_interval = 200
n_rounds = 50
seed = 20260801
elapsed = @elapsed (ensemble, diag) = run_once(swap_interval, n_rounds, seed)

total_steps = M * swap_interval * n_rounds
steps_per_sec = total_steps / elapsed
mean_gap = isempty(diag.straggler_gap) ? NaN : sum(diag.straggler_gap) / length(diag.straggler_gap)
rates = filter(!isnan, swap_rate(diag))
mean_rate = isempty(rates) ? NaN : sum(rates) / length(rates)

println("$(Threads.nthreads()),$(backend_kind),$(graph_kind),$(M),$(elapsed),$(total_steps)," *
        "$(steps_per_sec),$(mean_gap),$(mean_rate)")
