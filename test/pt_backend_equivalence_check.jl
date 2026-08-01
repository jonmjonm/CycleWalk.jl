# Standalone script (run as a subprocess under different `-t` counts — see
# test_parallel_tempering.jl's "ThreadedBackend reproduces SerialBackend bitwise"
# test) checking that ThreadedBackend reproduces SerialBackend bit for bit for the
# same seed, regardless of Threads.nthreads(). Prints "EQUIVALENT" and exits 0 on
# success; prints a diagnosis and exits 1 on any mismatch.

using CycleWalk
using RandomNumbers

testdir = dirname(@__FILE__)
json = joinpath(testdir, "test_graphs", "4x4pct_2x2cnty.json")
node_data = Set(["county", "pct", "pop", "area", "border_length"])
base_graph = BaseGraph(json, "pop"; inc_node_data=node_data, area_col="area",
                       node_border_col="border_length", edge_perimeter_col="length")
graph = MultiLevelGraph(base_graph, ["pct"])

constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(4, 4))
proposal = [(0.1, build_lifted_tree_cycle_walk(constraints)),
            (0.9, build_internal_forest_walk(constraints))]

measure = Measure()
push_energy!(measure, get_log_spanning_forests, 1.0)

M = 5
lattice = linear_betas(M)
swap_interval = 15
n_rounds = 10
seed = 20260801

function run_with(backend_ctor)
    rng = PCG.PCGStateOneseq(UInt64, seed)
    partition = LinkCutPartition(graph, constraints, 4; rng=PCG.PCGStateOneseq(UInt64, seed + 1))
    backend = backend_ctor(proposal)
    ensemble, diag = run_parallel_tempering!(partition, proposal, measure, lattice,
                                             swap_interval, n_rounds, rng;
                                             init_steps=20, backend=backend)
    return ensemble, diag
end

ensemble_serial, diag_serial = run_with(SerialBackend)
ensemble_threaded, diag_threaded = run_with(p -> ThreadedBackend(p, M))

ok = true
if ensemble_serial.walker_at_rung != ensemble_threaded.walker_at_rung
    println("MISMATCH: walker_at_rung differs: ",
           ensemble_serial.walker_at_rung, " vs ", ensemble_threaded.walker_at_rung)
    global ok = false
end
if diag_serial.round_trips != diag_threaded.round_trips
    println("MISMATCH: round_trips differs: ",
           diag_serial.round_trips, " vs ", diag_threaded.round_trips)
    global ok = false
end
if diag_serial.accepts != diag_threaded.accepts
    println("MISMATCH: accepts differs: ", diag_serial.accepts, " vs ", diag_threaded.accepts)
    global ok = false
end
for w in 1:M
    if ensemble_serial.replicas[w].state.node_to_dist != ensemble_threaded.replicas[w].state.node_to_dist
        println("MISMATCH: replica $w node_to_dist differs")
        global ok = false
    end
    if ensemble_serial.replicas[w].beta_index != ensemble_threaded.replicas[w].beta_index
        println("MISMATCH: replica $w beta_index differs")
        global ok = false
    end
end

if ok
    println("EQUIVALENT (nthreads=$(Threads.nthreads()))")
    exit(0)
else
    println("NOT EQUIVALENT (nthreads=$(Threads.nthreads()))")
    exit(1)
end
