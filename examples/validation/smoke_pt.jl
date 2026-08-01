# Smoke test for parallel tempering (run_parallel_tempering!) on the `grid` case
# (10x10, 3 districts) — same graph/measure as smoke_annealed_smc.jl. Reports swap
# rates per adjacent pair, round trips per walker, and the mean straggler gap, and
# asserts basic health (not accuracy — that's test/test_parallel_tempering.jl's
# ground-truth tests against the enumerated 4x4 distribution).
#
#   julia -t 8 examples/validation/smoke_pt.jl

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
Pkg.instantiate()

using CycleWalk
using RandomNumbers
using LinearAlgebra
using Printf

const EXAMPLES = normpath(joinpath(@__DIR__, ".."))

# --- grid case config (mirrors run_case.jl case_config("grid")) ---------------
graph_path   = joinpath(EXAMPLES, "data", "grid", "grid_graph_10_by_10.json")
num_dists    = 3
pop_dev      = 0.02
gamma        = 0.25
iso_weight   = 0.3
twocycle_frac = 0.1

graph = Graph(graph_path, "population", "node_name";
              inc_node_data=Set(["node_name","population","area","border_length","county"]),
              area_col="area", node_border_col="border_length",
              edge_perimeter_col="length")

constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))

cycle_walk    = build_two_tree_cycle_walk(constraints)
internal_walk = build_one_tree_cycle_walk(constraints)
proposal = [(twocycle_frac, cycle_walk), (1.0 - twocycle_frac, internal_walk)]

measure = Measure()
push_energy!(measure, get_log_spanning_forests, gamma)
push_energy!(measure, get_isoperimetric_score, iso_weight)

# --- run PT and report ---------------------------------------------------------
n_rungs       = max(Threads.nthreads(), 8)
swap_interval = 100
n_rounds      = 200
init_steps    = 200

function run_and_report(name, backend)
    rng = PCG.PCGStateOneseq(UInt64, 0x5A1D_0007)
    partition = LinkCutPartition(graph, constraints, num_dists; rng=rng, verbose=false)
    lattice = linear_betas(n_rungs)
    diag = PTDiagnostics(n_rungs)

    println("\n", "="^60, "\n", name, " (n_rungs=", n_rungs, ")")
    ensemble, diag = run_parallel_tempering!(
        partition, proposal, measure, lattice, swap_interval, n_rounds, rng;
        backend=backend, init_steps=init_steps, write_rungs=:none,
        diagnostics=diag)

    rates = swap_rate(diag)
    println(" pair    rate")
    for (k, r) in enumerate(rates)
        @printf("  %2d-%2d   %s\n", k, k+1, isnan(r) ? "n/a" : @sprintf("%.3f", r))
    end
    mean_gap = isempty(diag.straggler_gap) ? NaN :
              sum(diag.straggler_gap) / length(diag.straggler_gap)
    @printf("  round trips: %s\n  mean straggler gap: %.4fs\n",
            diag.round_trips, mean_gap)

    # assertions (smoke, not accuracy)
    @assert CycleWalk.check_ensemble(ensemble) "rung/walker bijection must hold"
    @assert all(isfinite, rates[.!isnan.(rates)]) "swap rates must be finite"
    @assert all(r -> 0 <= r <= 1, rates[.!isnan.(rates)]) "swap rates must be in [0,1]"
    @assert isfinite(mean_gap) && mean_gap >= 0 "straggler gap must be finite, >= 0"
    @assert length(diag.round_trips) == n_rungs
    return (; ensemble, diag, rates)
end

println("PT grid smoke: n_rungs=$n_rungs swap_interval=$swap_interval n_rounds=$n_rounds ",
        "(γ=$gamma iso=$iso_weight)")

serial = run_and_report("SerialBackend", SerialBackend(proposal))
threaded = run_and_report("ThreadedBackend (nthreads=$(Threads.nthreads()))",
                          ThreadedBackend(proposal, n_rungs))

# --- cross-check: same seed, same lattice => identical final states -----------
# SerialBackend and ThreadedBackend must produce bit-identical results for the same
# seed (see test/pt_backend_equivalence_check.jl for the full multi-thread-count
# version of this check); here it's an extra smoke assertion on the grid case.
dstates_match = all(serial.ensemble.replicas[k].state.node_to_dist ==
                    threaded.ensemble.replicas[k].state.node_to_dist
                    for k in 1:n_rungs)
println("\n", "="^60)
println("serial vs threaded final states match: ", dstates_match)
@assert dstates_match "SerialBackend and ThreadedBackend must agree for the same seed"

println("\nSMOKE OK — both backends run, agree bit for bit, diagnostics are sane")
