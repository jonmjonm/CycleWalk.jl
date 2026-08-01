# global information

function modify_perims!(
    base_graph::BaseGraph, 
    col::String,
    weights::Vector{Float64}=Vector{Float64}(undef, 0),
    edge_sets::Vector{Set{Tuple{Int64, Int64}}}=
        Vector{Set{Tuple{Int64, Int64}}}(undef, 0);
    base_weight::Float64=1.0
)
    for edge in keys(base_graph.edge_attributes)    
        e1, e2 = collect(edge).-1
        etup = (min(e1, e2), max(e1, e2))
        found_huh = false
        for (es_ind, es) ∈ enumerate(edge_sets)
            if etup ∈ es
                base_graph.edge_attributes[edge][col] = weights[es_ind]
                found_huh = true
                break
            end
        end
        if !found_huh
            base_graph.edge_attributes[edge][col] = base_weight
        end
    end
end


function modify_edge_len_small_square_base_graph(
    weights::Vector{Float64}=[2.0,2.5,3.0]
)
    # Weight Graph
    outercorners = Set([(0,1), (4,5), (0,2), (5,7), (8,10), (13,15), (10,11), 
                        (14,15)])
    outermiddle = Set([(1,4), (2,8), (7,13), (11,14)])
    innermiddle = Set([(3,6), (6,12), (9,12), (3,9)])

    weighted_base_graph = deepcopy(small_square_base_graph)
    edge_sets = [outercorners, outermiddle, innermiddle]
    modify_perims!(weighted_base_graph, "length", [2.0,2.5,3.0], edge_sets)
    return MultiLevelGraph(weighted_base_graph, ["pct"]);
end

function get_weighted_cut_edge_dist_small_square(
    graph::MultiLevelGraph,
    gamma::Float64,
    iso_weight::Float64
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
        cut_edges = get_cut_edge_sum(partition)
        sfs = Int(round(exp(get_log_spanning_forests(partition))))
        sym_count = symmetries[pind]
        iso_sum = get_isoperimetric_score(partition)
        iso_prob = exp(-iso_weight*iso_sum)
        prob = sfs^(1.0-gamma)*sym_count*iso_prob
        tot_prob = prob + get(prob_of_cut_edges, cut_edges, 0)
        prob_of_cut_edges[cut_edges] = tot_prob
    end

    n_inv = 1.0/sum(values(prob_of_cut_edges)) # need to normalize
    for ce in keys(prob_of_cut_edges)
        prob_of_cut_edges[ce] *= n_inv
    end
    return prob_of_cut_edges
end

name = "small square test graph (polsby-popper), 4 districts, pop=8, gamma∈{0,1}"
@testset "$name" begin
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(4,4))

    modified_graph = modify_edge_len_small_square_base_graph()
    
    iso_weight = 0.02
    measure = Measure()
    push_energy!(measure, get_isoperimetric_score, iso_weight) 

    observed_cuts = get_observed_cut_edges(modified_graph, constraints, 4,
                                           measure, cycle_steps=100_000)

    # check that the observed districts cut counts are correct 
    @test length(values(observed_cuts)) == 4

    # test distribution (calculated explicitly)
    ce_to_prob = get_weighted_cut_edge_dist_small_square(modified_graph, 0.0, 
                                                         iso_weight)
    steps = sum(values(observed_cuts))
    @test all([is_close(v/steps, ce_to_prob[k]) for (k,v) in observed_cuts])

    #######################################

    push_energy!(measure, get_log_spanning_forests, 1.0) 

    observed_cuts = get_observed_cut_edges(modified_graph, constraints, 4,
                                           measure, cycle_steps=100_000)

    # check that the observed districts cut counts are correct 
    @test length(values(observed_cuts)) == 4

    # test distribution (calculated explicitly)
    ce_to_prob = get_weighted_cut_edge_dist_small_square(modified_graph, 1.0,
                                                         iso_weight)
    steps = sum(values(observed_cuts))
    @test all([is_close(v/steps, ce_to_prob[k]) for (k,v) in observed_cuts])

    #######################################
    # AIS sign regression with BOTH energies annealing together. The historical
    # AIS/ASMC sign bug (commit 02c304c) was pinned in
    # test_annealed_importance_sampling.jl, but that test only anneals gamma alone
    # (K=1) — the multi-term dot product (K=2, both energies moving from 0 to a
    # nonzero target on the SAME path) is untested there. This graph already has
    # ground truth for gamma>0 AND iso_weight>0 simultaneously
    # (get_weighted_cut_edge_dist_small_square), so reweight AIS's base-chain
    # samples (uniform, gamma=iso=0) toward that joint target and compare the
    # importance-weighted cut-edge histogram against the enumeration — a flipped
    # sign would push mass toward the WRONG end of the cut-edge distribution.
    @testset "AIS reweights correctly with gamma AND iso_weight both nonzero" begin
        AIO = CycleWalk.AtlasIO
        gamma_target = 0.6
        iso_target = 0.02

        ais_measure = Measure()
        push_energy!(ais_measure, get_log_spanning_forests, gamma_target)
        push_energy!(ais_measure, get_isoperimetric_score, iso_target)

        cycle_walk    = build_lifted_tree_cycle_walk(constraints)
        internal_walk = build_internal_forest_walk(constraints)
        ais_proposal  = [(0.1, cycle_walk), (0.9, internal_walk)]

        scores, target_w = annealed_smc_scores_and_targets(ais_measure)
        path = linear_path(target_w)   # (0,0) -> (gamma_target, iso_target), together

        rng = PCG.PCGStateOneseq(UInt64, 314159265)
        ais_partition = LinkCutPartition(modified_graph, constraints, 4; rng=rng)

        mktempdir() do tmpdir
            output_path = joinpath(tmpdir, "ais_polsby_popper.jsonl.gz")
            writer = Writer(ais_measure, constraints, ais_partition, output_path;
                            weight_type=Float64)
            push_writer!(writer, get_cut_edge_sum)
            log_weights = run_annealed_importance_sampling!(
                ais_partition, ais_proposal, ais_measure, 80_000, 100, 300, rng;
                path=path, writer=writer)
            close_writer(writer)

            io = AIO.smartOpen(output_path, "r")
            maps = AIO.nextMaps(AIO.openAtlas(io))
            close(io)
            @test length(maps) == length(log_weights)

            mx = maximum(log_weights)
            w = exp.(log_weights .- mx)
            acc = Dict{Int, Float64}()
            for (m, wi) in zip(maps, w)
                ce = Int(m.data["get_cut_edge_sum"])
                acc[ce] = get(acc, ce, 0.0) + wi
            end
            tot = sum(values(acc))

            truth = get_weighted_cut_edge_dist_small_square(modified_graph,
                                                             gamma_target, iso_target)
            @test issubset(Set(keys(acc)), Set(keys(truth)))
            l1 = sum(abs(get(acc, k, 0.0)/tot - v) for (k, v) in truth)
            @test l1 < 0.15
            # a flipped sign concentrates mass on the wrong (high-cut) end instead
            @test l1 < 0.5   # loose bound that a flipped sign fails by a wide margin
        end
    end
end

