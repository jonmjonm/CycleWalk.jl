# Serial hotspot profile for parallel tempering, via Julia's stdlib sampling
# profiler (`Profile`) — where the time goes within a PT round, not how fast it
# scales with threads (see profile_pt_threaded.jl for that).
#
#   julia --project=<examples-dir> profile_pt_hotspots.jl [graph] [M] [n_rounds]
#
# graph defaults to "ct" (also: hex10/hex20/hex30/hex40/hex50 [hex == hex10], grid,
# nc, oh). M (rung count) defaults to 8, n_rounds to 50 (swap_interval is fixed at
# 200, matching profile_pt_threaded.jl's default point).
#
# Always uses SerialBackend, never ThreadedBackend: Profile samples one call stack
# per tick, so a Threads.@spawn'd run would split its samples across tasks/threads
# and blur the attribution. A serial run's hotspots are the same functions a
# threaded run spends time in per-replica — just measured without that noise.
#
# Prints a flat, count-sorted text hotspot table via Profile.print (stdlib only,
# no extra dependency, works headless over ssh/tmux).

const EXAMPLES = normpath(@__DIR__)

import Pkg
Pkg.activate(EXAMPLES)

using CycleWalk
using RandomNumbers
using DataStructures: OrderedDict
using Profile

graph_kind = length(ARGS) >= 1 ? ARGS[1] : "ct"
M          = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 8
n_rounds   = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 50

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
elseif graph_kind in ("hex", "hex10", "hex20", "hex30", "hex40", "hex50")
    hex_n = graph_kind == "hex" ? 10 : parse(Int, graph_kind[4:end])
    graph_path = joinpath(EXAMPLES, "data", "hex", "hex_graph_$(hex_n)_by_$(hex_n).json")
    num_dists  = 5
    pop_dev    = 0.0001
    graph = Graph(graph_path, "population", "node_name";
                  inc_node_data=Set(["county", "node_name", "population", "area",
                                     "border_length"]),
                  area_col="area", node_border_col="border_length",
                  edge_perimeter_col="length")
elseif graph_kind == "nc"
    # "uid" is a DERIVED column (county * "_" * prec_id) — see profile_pt_threaded.jl
    # for why this needs a BaseGraph + derive_node_columns! step first.
    graph_path = joinpath(EXAMPLES, "data", "nc", "NC_pct21.json")
    num_dists  = 14
    pop_dev    = 0.02
    base_graph = BaseGraph(graph_path, "pop2020cen";
                           inc_node_data=Set(["prec_id", "county", "pop2020cen",
                                              "area", "border_length"]),
                           area_col="area", node_border_col="border_length",
                           edge_perimeter_col="length")
    derive_node_columns!(base_graph, OrderedDict("uid" => "county * \"_\" * prec_id"))
    graph = Graph(base_graph, "uid")
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
    error("unknown graph \"$graph_kind\" (expected \"ct\", \"hex10\"/\"hex20\"/\"hex30\"/\"hex40\"/\"hex50\" (or \"hex\"), \"grid\", \"nc\", or \"oh\")")
end

constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))

cycle_walk    = build_lifted_tree_cycle_walk(constraints)
internal_walk = build_internal_forest_walk(constraints)
proposal = [(0.1, cycle_walk), (0.9, internal_walk)]

measure = Measure()
push_energy!(measure, get_log_spanning_forests, gamma)
push_energy!(measure, get_isoperimetric_score, iso_weight)

lattice = linear_betas(M)
swap_interval = 200

function run_once(n_rounds, seed)
    rng = PCG.PCGStateOneseq(UInt64, seed)
    partition = LinkCutPartition(graph, constraints, num_dists; rng=rng, verbose=false)
    backend = SerialBackend(proposal)
    run_parallel_tempering!(partition, proposal, measure, lattice, swap_interval,
                             n_rounds, rng; init_steps=0, backend=backend,
                             write_rungs=:none, writers=nothing)
end

# warm-up: absorb JIT/precompilation so the profile captures steady-state work,
# not compilation of the proposal/energy functions.
run_once(3, 1)

Profile.clear()
Profile.@profile run_once(n_rounds, 20260801)

println("=== hotspots: graph=$graph_kind, M=$M, swap_interval=$swap_interval, n_rounds=$n_rounds ===")
Profile.print(format=:flat, sortedby=:count, mincount=20)
