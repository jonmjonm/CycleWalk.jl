# global information

function edge_weight_func(
    base_graph::BaseGraph, 
    n1::Int, n2::Int;
    weights::Vector{Float64}=Vector{Float64}(undef, 0),
    edge_sets::Vector{Set{Tuple{Int64, Int64}}}=
        Vector{Set{Tuple{Int64, Int64}}}(undef, 0)
)
    edge = (min(n1-1,n2-1), max(n1-1,n2-1))
    for (w, es) in zip(weights, edge_sets)
        if edge ∈ es
            return w
        end
    end
    return 1
end


function weight_small_square_base_graph(weights::Vector{Float64}=[2.0,2.5,3.0])
    # Weight Graph
    outercorners = Set([(0,1), (4,5), (0,2), (5,7), (8,10), (13,15), (10,11), 
                        (14,15)])
    outermiddle = Set([(1,4), (2,8), (7,13), (11,14)])
    innermiddle = Set([(3,6), (6,12), (9,12), (3,9)])

    weighted_base_graph = deepcopy(small_square_base_graph)
    edge_sets = [outercorners, outermiddle, innermiddle]
    weight_func(g, n1, n2) = edge_weight_func(g, n1, n2; weights=weights, 
                                              edge_sets=edge_sets)
    modify_edge_weights!(weighted_base_graph, weight_func);
    for e in keys(weighted_base_graph.edge_attributes)
        weighted_base_graph.edge_attributes[e]["unit"] = 1
    end
    return MultiLevelGraph(weighted_base_graph, ["pct"]);
end


function get_weighted_cut_edge_dist_small_square(
    graph::MultiLevelGraph,
    gamma::Float64
)
    plans = [(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4),
             (1, 1, 3, 1, 2, 2, 2, 2, 3, 1, 3, 3, 4, 4, 4, 4),
             (1, 2, 1, 2, 3, 3, 3, 3, 1, 2, 1, 2, 4, 4, 4, 4),
             (1, 2, 1, 2, 2, 3, 2, 3, 1, 4, 1, 4, 4, 3, 4, 3),
             (1, 2, 1, 2, 2, 3, 2, 3, 1, 4, 1, 4, 3, 3, 4, 4),
             (1, 2, 1, 2, 3, 3, 3, 3, 1, 2, 1, 4, 2, 4, 4, 4),
             (1, 2, 1, 2, 2, 2, 3, 3, 1, 4, 1, 4, 3, 3, 4, 4),
             (1, 2, 1, 3, 2, 2, 3, 2, 1, 3, 1, 4, 3, 4, 4, 4),
             (1, 1, 3, 1, 2, 2, 1, 2, 3, 3, 3, 4, 4, 2, 4, 4),
             (1, 2, 1, 2, 3, 3, 2, 3, 1, 4, 1, 4, 2, 3, 4, 4),
             (1, 2, 1, 3, 2, 2, 4, 2, 1, 3, 1, 3, 4, 4, 3, 4),
             (1, 2, 1, 3, 2, 2, 2, 4, 1, 3, 1, 3, 3, 4, 4, 4),
             (1, 2, 1, 3, 2, 2, 2, 4, 1, 3, 1, 3, 4, 4, 3, 4),
             (1, 1, 4, 1, 2, 3, 2, 3, 4, 1, 4, 4, 2, 3, 2, 3),
             (1, 2, 1, 3, 2, 2, 3, 2, 1, 4, 1, 4, 3, 3, 4, 4),
             (1, 1, 3, 1, 2, 2, 2, 4, 3, 1, 3, 3, 2, 4, 4, 4),
             (1, 1, 3, 1, 2, 2, 4, 2, 3, 1, 3, 3, 4, 2, 4, 4),
             (1, 1, 3, 1, 2, 2, 1, 2, 3, 4, 3, 3, 4, 2, 4, 4),
             (1, 2, 1, 2, 2, 3, 4, 3, 1, 2, 1, 4, 4, 3, 4, 3),
             (1, 2, 1, 2, 2, 2, 3, 3, 1, 1, 4, 4, 4, 3, 4, 3),
             (1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4),
             (1, 2, 1, 1, 2, 2, 2, 3, 1, 4, 4, 4, 3, 3, 4, 3)]

    # I reordered so these aren't right
    symmetries = [1, 8, 4, 2, 8, 8, 4, 4, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 2, 
                  2, 2]
    cut_edges_per_plan = [8, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 
                          12, 12, 12, 12, 12, 12, 12, 12]
    sp_trees = [256, 16, 16, 16, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                1, 1, 1]
    prob_of_cut_edges = Dict{Int64, Float64}()

    node_to_district = Dict{Tuple{Vararg{String}}, Int}()

    for (pind, p) in enumerate(plans)
        for ii = 1:length(p)
            node_to_district[graph.id_to_partitions[1][ii]] = p[ii]
        end
        ml_partition = MultiLevelPartition(graph, node_to_district)
        rng = PCG.PCGStateOneseq(UInt64, 4052159124)
        partition = LinkCutPartition(ml_partition, rng);
        cut_edges_uw = get_cut_edge_sum(partition; column="unit")
        sfs = Int(round(exp(get_log_spanning_forests(partition))))
        sym_count = symmetries[pind]
        prob = sfs^(1.0-gamma)*sym_count
        tot_prob = prob + get(prob_of_cut_edges, cut_edges_uw, 0)
        prob_of_cut_edges[cut_edges_uw] = tot_prob
    end

    n_inv = 1.0/sum(values(prob_of_cut_edges)) # need to normalize
    for ce in keys(prob_of_cut_edges)
        prob_of_cut_edges[ce] *= n_inv
    end
    return prob_of_cut_edges
end

name = "small square test graph (weighted), 4 districts, pop=8, gamma∈{0,1}"
@testset "$name" begin
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4,4))

    weighed_graph = weight_small_square_base_graph()

    series = Int[]
    observed_cuts = get_observed_cut_edges(weighed_graph, constraints, 4,
                                           cycle_steps=200_000, series=series,
                                           cut_edge_field="unit")

    # check that the observed districts cut counts are correct 
    @test length(values(observed_cuts)) == 4

    # test distribution (calculated explicitly)
    ce_to_prob = get_weighted_cut_edge_dist_small_square(weighed_graph, 0.0)
    @test mc_matches(series, identity, ce_to_prob; label=name)

    #######################################

    measure = Measure()
    push_energy!(measure, get_log_spanning_forests, 1.0) 

    series = Int[]
    observed_cuts = get_observed_cut_edges(weighed_graph, constraints, 4,
                                           measure, cycle_steps=100_000, series=series; 
                                           cut_edge_field="unit")

    # check that the observed districts cut counts are correct 
    @test length(values(observed_cuts)) == 4

    # test distribution (calculated explicitly)
    ce_to_prob = get_weighted_cut_edge_dist_small_square(weighed_graph, 1.0)
    @test mc_matches(series, identity, ce_to_prob; label=name)
end

