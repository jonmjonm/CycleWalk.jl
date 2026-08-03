@testset "graph-weighted initialization on NC graph" begin
    graph_path = joinpath(testdir, "test_graphs", "NC_pct21.json")
    node_data = Set(["prec_id", "pop2020cen", "area", "border_length"])
    base_graph = BaseGraph(
        graph_path,
        "pop2020cen";
        inc_node_data=node_data,
        area_col="area",
        node_border_col="border_length",
        edge_perimeter_col="length",
        edge_weights="length",
    )

    graph_weights = base_graph.simple_graph.weights.nzval
    @test base_graph.edge_weights == "length"
    @test all(isfinite, graph_weights)
    @test all(>(0), graph_weights)
    @test length(unique(graph_weights)) > 1

    graph = Graph(base_graph, ["prec_id"])
    num_parts = 14
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(graph, num_parts, 0.02))

    weighted_partition = LinkCutPartition(
        graph,
        constraints,
        num_parts;
        rng=PCG.PCGStateOneseq(UInt64, 454190),
        initializer=GraphWeightedInitializer(),
    )
    uniform_partition = LinkCutPartition(
        graph,
        constraints,
        num_parts;
        rng=PCG.PCGStateOneseq(UInt64, 454190),
    )

    @test sort(unique(weighted_partition.node_to_dist)) == collect(1:num_parts)
    @test satisfies_constraints(
        weighted_partition,
        constraints;
        check_population=true,
    )
    @test weighted_partition.node_to_dist != uniform_partition.node_to_dist
end

@testset "graph-weighted initialization after BaseGraph mutation" begin
    graph_path = joinpath(testdir, "test_graphs", "NC_pct21.json")
    node_data = Set(["prec_id", "pop2020cen", "area", "border_length"])
    base_graph = BaseGraph(
        graph_path,
        "pop2020cen";
        inc_node_data=node_data,
        area_col="area",
        node_border_col="border_length",
        edge_perimeter_col="length",
    )

    @test base_graph.edge_weights == "connections"
    @test all(==(1), base_graph.simple_graph.weights.nzval)

    function weight_from_edge_data(graph, n1, n2)
        edge_data = graph.edge_attributes[Set([n1, n2])]
        return 1.0 + 10.0 * Float64(edge_data["length"])
    end
    modify_edge_weights!(base_graph, weight_from_edge_data)

    graph_weights = base_graph.simple_graph.weights.nzval
    @test all(isfinite, graph_weights)
    @test all(>(0), graph_weights)
    @test length(unique(graph_weights)) > 1
    @test all(
        edge_data[base_graph.edge_weights] ==
        weight_from_edge_data(base_graph, collect(edge)[1], collect(edge)[2])
        for (edge, edge_data) in base_graph.edge_attributes
    )

    graph = MultiLevelGraph(base_graph, ["prec_id"])
    num_parts = 14
    constraints = initialize_constraints()
    add_constraint!(constraints, PopulationConstraint(graph, num_parts, 0.02))
    mfr_constraints, levels = CycleWalk.interpret_constraints(constraints, graph)

    @test graph.num_levels == 1
    @test levels == ["prec_id"]

    rng = PCG.PCGStateOneseq(UInt64, 454190)
    initial_partition = MultiLevelPartition(
        graph,
        mfr_constraints,
        num_parts;
        rng=rng,
        initializer=GraphWeightedInitializer(),
    )
    partition = LinkCutPartition(initial_partition, rng)

    @test initial_partition.num_dists == num_parts
    @test satisfies_constraints(partition, constraints; check_population=true)
end
