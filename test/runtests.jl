
using CycleWalk
using RandomNumbers
using Test

const testdir = dirname(@__FILE__)


function is_close(a,b)
    if a > 0.01
        0.9 <= a/b && a/b <= 1.1
    else 
        0.6 <= a/b && a/b <= 1.4
    end 
end


function get_observed_cut_edges(
    graph::MultiLevelGraph, 
    constraints::Constraints,
    num_districts::Int, 
    measure::Measure=Measure();
    cycle_steps::Int=50_000,
    cycle_walk_frac::Float64 = 0.1,
    cut_edge_field = "connections"
)::Dict
    rng = PCG.PCGStateOneseq(UInt64, 1241909)
    partition = LinkCutPartition(graph, constraints, num_districts; rng=rng)

    cycle_walk = build_lifted_tree_cycle_walk(constraints)
    internal_walk = build_internal_forest_walk(constraints)
    proposal = [(cycle_walk_frac, cycle_walk), 
                (1.0-cycle_walk_frac, internal_walk)]
         
    instep = Int(floor(1.0/cycle_walk_frac))
    edge_cut_counts = Dict{Int64, Int64}()
    for ii = 1:cycle_steps
        run_metropolis_hastings!(partition, proposal, measure, instep, rng);
        cut_edges = get_cut_edge_sum(partition, column=cut_edge_field)
        edge_cut_counts[cut_edges] = get(edge_cut_counts, cut_edges, 0) + 1
    end

    return edge_cut_counts
end 



small_square_json = joinpath(testdir, "test_graphs", "4x4pct_2x2cnty.json")
small_square_node_data = Set(["county", "pct", "pop", "area", "border_length"])
small_square_base_graph = BaseGraph(small_square_json, "pop", 
                                    inc_node_data=small_square_node_data,
                                    area_col="area",
                                    node_border_col="border_length", 
                                    edge_perimeter_col="length")
small_square_graph = MultiLevelGraph(small_square_base_graph, ["pct"])


include(joinpath(testdir, "test_linkcuttree.jl"))
include(joinpath(testdir, "test_option_b.jl"))
include(joinpath(testdir, "test_cuttable_edges.jl"))
include(joinpath(testdir, "test_annealed_importance_sampling.jl"))
include(joinpath(testdir, "test_annealed_smc.jl"))
include(joinpath(testdir, "test_run_metadata.jl"))
include(joinpath(testdir, "test_docs_coverage.jl"))
@testset verbose = true "observables_and_diagnostics" begin
    include(joinpath(testdir, "test_observables_diagnostics.jl"))
end

tests = [
    "small_square_p88_unweighted",
    "small_square_p88_weighted",
    "small_square_p88_polsby_popper",
    "test_constraints",
    "test_budgeted_region_constraint",
    "test_budgeted_region_mfr_translation",
    ]
for t in tests
    @testset verbose = true "$t" begin
        tp = joinpath(testdir, "test_cases","$(t).jl")
        include(tp)
    end
end;
