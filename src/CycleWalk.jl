module CycleWalk
using JSON
using SimpleWeightedGraphs
using DataStructures:
    IntDisjointSet,
    in_same_set,
    SortedDict,
    OrderedDict
using Graphs
using RandomNumbers
using LinearAlgebra
using SparseArrays: rowvals, nonzeros, nzrange, sparse
using Hungarian
import Combinatorics
using Dates
using AtlasIO
using TOML
using LinkCutTreesAugmented:
    Node,
    LinkCutTree,
    link_cut_tree,
    pop_link_cut_tree,
    subtree_pop,
    link!,
    cut!,
    evert!,
    set_root!,
    find_root!,
    expose!,
    path_children,
    first_path_child,
    next_path_sibling,
    cc,
    nv_cc,
    get_connected_edge_list,
    parents,
    findPath,
    get_diameter,
    get_farthest_node
import MetropolizedForestRecom:
    initialize_constraints as mfr_initialize_constraints,
    add_constraint! as mfr_add_constraint!,
    get_district as mfr_get_district
using MetropolizedForestRecom:
    AbstractGraph,
    AbstractPartition,
    BaseGraph,
    Graph,
    MultiLevelGraph,
    MultiLevelSubGraph,
    MultiLevelPartition,
    Partition,
    AbstractConstraint,
    PopulationConstraint,
    PackNodeConstraint,
    AllowedExcessDistsInCoarseNodes,
    MaxTotalExcessDistsInCoarseNodes,
    MaxTotalMissingPackedDistsInCoarseNodes,
    # WANT ConstrainDiscontinuousTraversals,
    # WANT -- but rename -- MaxCoarseNodeSplits,
    # WANT -- but rename -- MaxSharedCoarseNodes,
    # WANT -- but rename -- AllowedExcessDistsInCoarseNodes,
    # initialize_constraints,
    # add_constraint!,
    cluster_base_graph

export AbstractGraph,
    BaseGraph,
    modify_edge_weights!,
    #
    Graph,
    MultiLevelGraph,
    MultiLevelPartition,
    AbstractPartition,
    edge_weight,
    build_graph,
    multi_level_graph,

    # proposals
    build_one_tree_cycle_walk,
    build_two_tree_cycle_walk,
    build_cycle_walk,
    build_lifted_tree_cycle_walk,
    build_internal_forest_walk,

    # constraints
    initialize_constraints,
    Constraints,
    satisfies_constraint,
    satisfies_constraints,
    add_constraint!,
    push_constraint!,
    AbstractConstraint,
    PopulationConstraint,
    PackRegionConstraint,
    CapRegionDistricts,
    CapRegionDistConstraint,
    BudgetedRegionConstraint,
    # ConstrainDiscontinuousTraversals,
    # MaxCoarseNodeSplits,
    # MaxSharedCoarseNodes,
    # MaxSharedNodes,
    # AllowedExcessDistsInCoarseNodes,
    # MaxHammingDistance,
    
    LinkCutPartition,
    relabel_districts!,

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
    push_path_writer!,
    stamp_execution_metadata!,
    collect_included_sources,
    write_header!,

    # mcmc
    run_metropolis_hastings!,
    run_annealed_importance_sampling!,
    run_annealed_smc!,
    FixedSchedule,
    AdaptiveTempering,
    reset_schedule!,
    LinearPath,
    linear_path,
    annealed_smc_scores_and_targets,
    describe_proposal,
    name_proposal!,
    proposal_name,
    metropolis_hastings_run_metadata,
    annealed_importance_sampling_run_metadata,
    annealed_smc_run_metadata,
    parallel_tempering_run_metadata,

    # energies/observables
    Measure,
    push_energy!,
    evaluate_weight_expression,
    weight_expression_parameters,
    weight_path_closure,
    evaluate_column_expression,
    derive_node_columns!,
    EnergySpec,
    energy_specs,
    measure_parameters,
    annealing_weights,
    referenced_parameters,
    energy_weight,
    energy_weight_start,
    build_annealed_measure,
    build_measure,
    build_path_measure,
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
    get_log_linking_edges,
    get_log_district_trees,
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
    BetaLattice,
    linear_betas,
    geometric_betas,
    HeatBath,
    PTDiagnostics,
    reset_pt_diagnostics!,
    swap_rate,
    run_parallel_tempering!,
    SerialBackend,
    ThreadedBackend,
    # DistributedBackend,
    parse_bath_measure,
    parse_bath_samples,
    try_heat_bath!,

    # cluster graph
    cluster_base_graph

include("./auxilary/random_extensions.jl")
include("./auxilary/SimpleWeightedGraphs_BugFixes.jl")
include("./utilities/array_utils.jl")

include("./graph/graph.jl")
include("./graph/derived_columns.jl")
include("./trees/neighbor_list_tree.jl")
include("./trees/tree.jl")
include("./trees/tree_metrics.jl")
include("./trees/russo_ust.jl")

# type defs
include("./measure/energy/energy_types.jl")
include("./proposals/update.jl")
include("./proposals/named_proposal.jl")
include("./diagnostics/proposal_diagnostics_types.jl")

include("./measure/constraints/constraints.jl")
include("./measure/constraints/pack_region/pack_region_type.jl")
include("./measure/constraints/cap_region_districts/cap_region_districts_type.jl")
include("./measure/constraints/budgeted_region_constraint/budgeted_region_constraint_type.jl")
include("./measure/constraints/mfr_constraints_interpreter.jl")

include("./partition/link_cut_partition.jl")
include("./partition/mfr_partition_ext.jl")

include("./measure/constraints/check_constraints.jl")

include("./measure/data/regional_split_data.jl")
include("./measure/constraints/pack_region/pack_region.jl")
include("./measure/constraints/cap_region_districts/cap_region_districts.jl")
include("./measure/constraints/budgeted_region_constraint/budgeted_region_constraint.jl")
include("./measure/constraints/balance.jl")

include("./measure/measure.jl")
include("./measure/weight_expressions.jl")
include("./measure/measure_spec.jl")
include("./measure/energy/defaults.jl")
include("./measure/energy/log_forest_count.jl")
include("./measure/energy/polsby_popper.jl")
include("./measure/energy/split_unit_count_by_node.jl")
include("./measure/energy/performant_vra_dists.jl")
include("./measure/energy/log_linking_edges.jl")
include("./measure/energy/log_district_trees.jl")
# include("./measure/energy/mcd_ousted_population.jl")
# include("./measure/energy/vap_frac.jl")

include("./diagnostics/proposal_diagnostics.jl")
include("./diagnostics/lifted_cycle_walk_diagnostics.jl")
include("./diagnostics/delta_population.jl")
include("./diagnostics/delta_nodes.jl")
include("./diagnostics/cuttable_edge_pairs.jl")
include("./diagnostics/unique_cuttable_edges.jl")
include("./diagnostics/swappable_pop.jl")

include("./observables/node_counts.jl")
include("./observables/partisan_lean.jl")

#include("./io/AtlasIO.jl")
include("./io/writer.jl")

include("./proposals/lifted_tree_cycle_walk.jl")
include("./proposals/internal_forest_walk.jl")
include("./chains/chain.jl")
include("./chains/mcmc.jl")
include("./chains/anneal_path.jl")
include("./chains/annealed_importance_sampling.jl")
include("./chains/annealed_smc.jl")
include("./chains/parallel_tempering_types.jl")
include("./chains/pt_backends.jl")
include("./chains/parallel_tempering.jl")
include("./chains/pt_heat_bath.jl")
include("./io/run_metadata.jl")

# include("./parallel_tempering_multiprocessing.jl")
end # module
