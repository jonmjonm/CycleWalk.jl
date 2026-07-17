# Validates the link-cut-tree refactor: the partition's tree is a PopAug tree,
# and the maintained subtree sums reproduce the previous on-demand
# topological_sort exactly, including the collapsed-cycle-weight computation.

@testset "PopAug partition" begin
    json = joinpath(testdir, "test_graphs", "4x4pct_2x2cnty.json")
    nd = Set(["county", "pct", "pop", "area", "border_length"])
    bg = BaseGraph(json, "pop"; inc_node_data=nd, area_col="area",
                   node_border_col="border_length", edge_perimeter_col="length")
    graph = MultiLevelGraph(bg, ["pct"])
    cons = initialize_constraints()
    add_constraint!(cons, PopulationConstraint(graph, 2, 0.5))
    rng = PCG.PCGStateOneseq(UInt64, 12345)
    part = LinkCutPartition(graph, cons, 2; rng=rng)

    @testset "partition lct is a PopAug tree" begin
        @test occursin("PopAug", string(eltype(part.lct.nodes)))
        @test part.lct.nodes[1] isa CycleWalk.Node
    end

    @testset "subtree_pop reproduces topological_sort per vertex" begin
        for r in part.district_roots
            CycleWalk.evert!(part.lct.nodes[r])
            cd = CycleWalk.topological_sort(part.lct.nodes[r], part)   # Dict{Int,Float64}
            CycleWalk.evert!(part.lct.nodes[r])
            for v in CycleWalk.cc(part.lct.nodes[r])
                @test CycleWalk.subtree_pop(part.lct.nodes[v]) == cd[v]
            end
        end
    end

    @testset "collapsed cycle weights: subpop == topological_sort" begin
        # find a district pair with at least two boundary edges
        pair = nothing
        for (k, s) in part.cross_district_edges
            length(s) >= 2 && (pair = (k, collect(s)); break)
        end
        if pair === nothing
            @test_skip "no district pair with >= 2 boundary edges in this build"
        else
            edge_pair = (pair[2][1], pair[2][2])
            uPath, vPath = CycleWalk.get_paths!(part, edge_pair)
            w_sub = CycleWalk.get_collapsed_cycle_weights_subpop(uPath, vPath, part)
            w_topo = CycleWalk.get_collapsed_cycle_weights_topo(uPath, vPath, part)
            @test w_sub == w_topo
        end
    end
end
