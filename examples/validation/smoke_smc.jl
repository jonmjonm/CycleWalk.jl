# Smoke test for the annealed SMC sampler (run_annealed_smc!) on the `grid` case
# (10x10, 3 districts) — the correctness gate: AIS reaches ~90% ESS here, so a
# working SMC must keep healthy ESS and a finite logZ. Runs BOTH schedule variants
# from the same initial partition and compares:
#   - FixedSchedule     : precomputed t-grid, resample when ESS < ess_frac*N
#   - AdaptiveTempering : pick each t online to hold ESS at ess_target*N, resample
#                         every step (coupled)
# The strong cross-check: both estimate the SAME logZ, so their logZ should agree.
#
#   julia -t 4 examples/validation/smoke_smc.jl

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

rng = PCG.PCGStateOneseq(UInt64, 0x5A1D_0007)
partition = LinkCutPartition(graph, constraints, num_dists; rng=rng, verbose=false)

cycle_walk    = build_two_tree_cycle_walk(constraints)
internal_walk = build_one_tree_cycle_walk(constraints)
proposal = [(twocycle_frac, cycle_walk), (1.0 - twocycle_frac, internal_walk)]

# target measure: identical to run_case.jl (γ spanning-forest + iso)
measure = Measure()
push_energy!(measure, get_log_spanning_forests, gamma)
push_energy!(measure, get_isoperimetric_score, iso_weight)

# --- run one SMC variant and report ------------------------------------------
n_particles = 64
rejuv_steps = 50
init_steps  = 500     # base burn-in: decorrelate the shared initial partition

function run_and_report(name, schedule; show_trace=false)
    rng = PCG.PCGStateOneseq(UInt64, 0x5A1D_0007)   # same seed => same init draw
    println("\n", "="^60, "\n", name)
    particles, logZ, trace =
        run_annealed_smc!(partition, proposal, measure, schedule,
                          n_particles, rejuv_steps, rng;   # path=linear_path default
                          init_steps = init_steps)

    if show_trace
        println(" block   t        ESS      ESS%   resampled")
        for (b, r) in enumerate(trace)
            @printf("  %3d   %.4f   %7.2f   %5.1f%%    %s\n",
                    b, r.t, r.ess, 100 * r.ess / n_particles, r.resampled ? "yes" : "")
        end
    end
    n_resamples = count(r -> r.resampled, trace)
    min_ess   = minimum(r.ess for r in trace)
    final_ess = CycleWalk.ess_from_logw([p.logW for p in particles])
    @printf("  blocks=%d  resamples=%d  min-block-ESS=%.1f (%.0f%%)  final-ESS=%.1f (%.0f%%)  logZ=%.4f\n",
            length(trace), n_resamples, min_ess, 100*min_ess/n_particles,
            final_ess, 100*final_ess/n_particles, logZ)

    # assertions (smoke, not accuracy)
    @assert abs(trace[end].t - 1.0) < 1e-9              "schedule must reach t=1"
    @assert all(r -> isfinite(r.ess) && r.ess > 0, trace) "ESS must be finite/positive"
    @assert isfinite(logZ)                              "logZ must be finite"
    @assert length(particles) == n_particles
    return (; logZ, n_blocks=length(trace), n_resamples, final_ess)
end

println("SMC grid smoke: N=$n_particles rejuv=$rejuv_steps (γ=$gamma iso=$iso_weight)")

fixed = run_and_report("FixedSchedule (40 blocks, ess_frac=0.5)",
                       FixedSchedule(range(0, 1; length = 41); ess_frac = 0.5);
                       show_trace = true)

adapt = run_and_report("AdaptiveTempering (ess_target=0.5, resample every step)",
                       AdaptiveTempering(; ess_target = 0.5);
                       show_trace = true)

# --- cross-check: both estimate the same normalizer --------------------------
dlogZ = abs(fixed.logZ - adapt.logZ)
println("\n", "="^60)
@printf("logZ:  fixed=%.4f   adaptive=%.4f   |Δ|=%.4f\n",
        fixed.logZ, adapt.logZ, dlogZ)
println("adaptive used ", adapt.n_blocks, " blocks vs fixed ", fixed.n_blocks,
        "  (adaptive resamples=", adapt.n_resamples, ")")
# Loose bound: two independent-noise estimators of the same logZ on a small
# population; they should be in the same ballpark, not bit-identical.
@assert dlogZ < 5.0  "fixed and adaptive logZ disagree by $dlogZ (>5); suspect a bug"
println("\nSMOKE OK — both schedules run, logZ estimates agree")
