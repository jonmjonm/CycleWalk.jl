using Test

# modify_edge_weights! on a MultiLevelGraph must agree with modifying the base graph
# and rebuilding the multilevel graph from it, for one and for several levels.
@testset "modify_edge_weights!(::MultiLevelGraph)" begin
    weight_func(g, u, v) = 1.0 + (7u + 3v) % 5

    for levels in (["pct"], ["county", "pct"])
        @testset "levels = $levels" begin
            graph = MultiLevelGraph(deepcopy(small_square_base_graph), levels)
            old_weights = copy(graph.graphs_by_level[end].simple_graph.weights)
            modify_edge_weights!(graph, weight_func)

            reference_base = deepcopy(small_square_base_graph)
            modify_edge_weights!(reference_base, weight_func)
            reference = MultiLevelGraph(reference_base, levels)

            base = graph.graphs_by_level[end]
            @test base.simple_graph.weights != old_weights
            for level in eachindex(reference.graphs_by_level)
                @test graph.graphs_by_level[level].simple_graph.weights ==
                      reference.graphs_by_level[level].simple_graph.weights
                isassigned(reference.coarse_to_fine_graphs, level) || continue
                for ii in eachindex(reference.coarse_to_fine_graphs[level])
                    got = graph.coarse_to_fine_graphs[level][ii]
                    want = reference.coarse_to_fine_graphs[level][ii]
                    @test got.graph.weights == want.graph.weights
                    @test got.vmap == want.vmap
                end
            end
            @test graph.mixed_nbr_weights == reference.mixed_nbr_weights
            # the edge-attribute table agrees with the adjacency weights
            @test all(base.simple_graph.weights[minimum(k), maximum(k)] ==
                      attrs[base.edge_weights]
                      for (k, attrs) in base.edge_attributes)
        end
    end
end
