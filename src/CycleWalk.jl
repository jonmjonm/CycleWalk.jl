module CycleWalk
using JSON
using SimpleWeightedGraphs
using DataStructures:
    IntDisjointSets,
    in_same_set,
    SortedDict
using Graphs
using RandomNumbers
using LinearAlgebra
using Hungarian
import Combinatorics
using Dates
using AtlasIO

export AbstractGraph,
    BaseGraph,
    modify_edge_weights!,
    #
    MultiLevelGraph,
    MultiLevelPartition,
    edge_weight,
    build_graph,

    # proposals
    build_lifted_tree_cycle_walk,
    build_cycle_walk,
    build_internal_forest_walk,

    # constraints
    initialize_constraints,
    add_constraint!,
    AbstractConstraint,
    PopulationConstraint,
    ConstrainDiscontinuousTraversals,
    PackNodeConstraint,
    MaxCoarseNodeSplits,
    MaxSharedCoarseNodes,
    MaxSharedNodes,
    AllowedExcessDistsInCoarseNodes,
    MaxHammingDistance,
    MultiScaleCuttableTree,
    
    LinkCutPartition,
    AbstractTreeState,

    # Diagnostics
    RunDiagnostics,
    AcceptanceRatios,
    CycleLengthDiagnostic,
    DeltaNodesDiagnostic,
    DeltaPopDiagnostic,
    CuttableEdgePairsDiagnostic,
    UniqueCuttableEdgesDiagnostic,
    MaxSwappablePopulationDiagnostic,
    AvgSwappablePopulationDiagnostic,
    push_diagnostic!,

    # Writer
    Writer,
    close_writer,
    push_writer!,

    # mcmc
    run_metropolis_hastings!,
    
    # energies/observables
    Measure,
    push_energy!,
    get_log_energy,
    get_log_spanning_trees,
    get_log_spanning_forests,
    get_isoperimetric_score,
    get_isoperimetric_scores,
    get_center_moments,
    get_center_leaves_moments,
    get_average_degrees,
    get_degree_distributions,
    get_neighbor_lists,
    get_diameters,
    # get_split_unit_by_node_count,
    # build_split_unit_by_node_energy,
    # build_split_unit_by_node_count,
    build_performant_vra_score,
    build_performant_vra_report,
    get_target_vra_districts,
    build_get_partisan_margins,
    build_get_partisan_seats,
    get_cut_edge_sum,
    # get_cut_edge_count,
    # get_minofracs,
    # get_node_counts,
    # build_mcd_score,
    # build_get_vra_score,
    # get_vra_score,
    # get_vra_scores,

    # abstract chain
    Chain,
    run_chain!,

    # parallel tempering
    # parallel_tempering!,
    # parse_base_samples,
    # parse_base_measure,

    # cluster graph
    cluster_base_graph

include("auxilary/random_extensions.jl")
include("auxilary/SimpleWeightedGraphs_BugFixes.jl")
include("utilities/array_utils.jl")

include("graph/multi_level_graph.jl")
include("graph/subgraph.jl")
include("graph/node_set.jl")
include("graph/multi_level_subgraph.jl")

#JCM Added. Loads in some simple neighbor list algorithms
include("trees/neighbor_list_tree.jl")

include("trees/tree.jl")
include("trees/splaytrees.jl")
include("trees/linkcuttrees.jl")
include("trees/russo_ust.jl")

# type defs
include("measure/constraints/constraint_types.jl")
include("measure/energy/energy_types.jl")
include("proposals/update.jl")
include("diagnostics/proposal_diagnostics_types.jl")

include("partition/multi_level_partition.jl")
include("partition/balance_multi_level_graph.jl")
include("partition/construct_multi_level_partition.jl")
include("partition/link_cut_partition.jl")

include("cuttable_tree/cuttable_tree.jl")

include("measure/constraints/constraints.jl")

include("measure/measure.jl")
include("measure/energy/defaults.jl")
include("measure/energy/log_forest_count.jl")
include("measure/energy/polsby_popper.jl")
include("measure/energy/split_unit_count_by_node.jl")
include("measure/energy/performant_vra_dists.jl")
# include("measure/energy/mcd_ousted_population.jl")
# include("measure/energy/vap_frac.jl")

include("diagnostics/proposal_diagnostics.jl")
include("diagnostics/lifted_cycle_walk_diagnostics.jl")
include("diagnostics/delta_population.jl")
include("diagnostics/delta_nodes.jl")
include("diagnostics/cuttable_edge_pairs.jl")
include("diagnostics/unique_cuttable_edges.jl")
include("diagnostics/swappable_pop.jl")

include("observables/node_counts.jl")
include("observables/partisan_lean.jl")

#include("io/AtlasIO.jl")
include("io/writer.jl")

include("proposals/lifted_tree_cycle_walk.jl")
include("proposals/internal_forest_walk.jl")
include("chains/chain.jl")
include("chains/mcmc.jl")

# include("parallel_tempering_multiprocessing.jl")
end # module
