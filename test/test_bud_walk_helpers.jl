using Test
using Graphs
using SimpleWeightedGraphs
using RandomNumbers
using Combinatorics

@testset "bud_walk_helpers" begin
    @test CycleWalk.ordered_edge_key(7, 3) == (3, 7)

    cross_edges = Dict{Tuple{Int64, Int64}, Set{SimpleWeightedEdge}}(
        (1, 2) => Set([
            SimpleWeightedEdge(1, 2, 2.0),
            SimpleWeightedEdge(3, 4, 5.0),
        ]),
        (2, 3) => Set([
            SimpleWeightedEdge(5, 6, 1.5),
        ]),
    )

    qg, pair_to_qedge = CycleWalk.build_district_quotient_graph(3, cross_edges)
    @test nv(qg) == 3
    @test ne(qg) == 2
    @test qg.weights[1, 2] == 7.0
    @test qg.weights[2, 3] == 1.5
    @test src(pair_to_qedge[(1, 2)]) == 1
    @test dst(pair_to_qedge[(1, 2)]) == 2

    rng = PCG.PCGStateOneseq(UInt64, 17)
    single_edge_set = Set([
        SimpleWeightedEdge(10, 11, 4.2),
    ])
    sampled = CycleWalk.sample_weighted_edge(single_edge_set, rng)
    @test CycleWalk.ordered_edge_key(sampled) == (10, 11)

    # four equal nodes; the last argument is the cycle's total population
    cycle_weights = [10.0, 10.0, 10.0, 10.0]
    total_pop = sum(cycle_weights)
    # k=2: only the 20|20 split works, from any starting edge
    @test CycleWalk.edge_supports_all_k_splits(cycle_weights, 1, 2, 18.0, 22.0, total_pop)
    @test CycleWalk.edge_supports_all_k_splits(cycle_weights, 2, 2, 18.0, 22.0, total_pop)
    @test !CycleWalk.edge_supports_all_k_splits(cycle_weights, 1, 2, 8.0, 12.0, total_pop)
    # k=4: one node per district is fine with [8, 12] but not with [11, 12]
    @test CycleWalk.edge_supports_all_k_splits(cycle_weights, 1, 4, 8.0, 12.0, total_pop)
    @test !CycleWalk.edge_supports_all_k_splits(cycle_weights, 1, 4, 11.0, 12.0, total_pop)
    # k=3: 40 cannot be split into three districts of at most 12
    @test !CycleWalk.edge_supports_all_k_splits(cycle_weights, 1, 3, 8.0, 12.0, total_pop)
end

@testset "cut_cross_district_edges!" begin
    # relies on `small_square_graph` from runtests.jl
    cons = initialize_constraints()
    add_constraint!(cons, PopulationConstraint(small_square_graph, 4, 0.5))
    rng = PCG.PCGStateOneseq(UInt64, 7)
    part = LinkCutPartition(small_square_graph, cons, 4; rng=rng)

    for _ in 1:20
        qg, _ = CycleWalk.build_district_quotient_graph(part)
        qtree = CycleWalk.wilson_rst(qg, rng, weights(qg))
        qlct = CycleWalk.build_quotient_lct(qtree; safeflag=false)
        qpath = CycleWalk.get_cycle_path_for_edge!(qlct,
                                                   (qtree[1].src, qtree[1].dst))
        specified = Dict{Edge{Int}, Tuple{Int, Int}}()
        CycleWalk.specify_quotient_path!(part, specified, qpath, rng)
        linked = CycleWalk.add_specified_edges!(part, specified, qpath)
        @test length(linked) == length(qpath) - 1
        CycleWalk.cut_cross_district_edges!(part, qpath, specified)

        # one tree per district; root bookkeeping consistent with the lct
        roots, _ = CycleWalk.get_district_roots(part.lct)
        @test length(roots) == part.num_dists
        @test Set(roots) == Set(part.district_roots)
        @test length(unique(part.district_roots)) == part.num_dists
        @test Set(keys(part.roots_to_district)) == Set(roots)
        for d in 1:part.num_dists
            @test part.roots_to_district[part.district_roots[d]] == d
            @test part.node_to_dist[part.district_roots[d]] == d
        end
        for v in 1:nv(part.graph.simple_graph)
            root = CycleWalk.find_root!(part.lct.nodes[v]).vertex
            @test part.node_to_dist[root] == part.node_to_dist[v]
        end
    end
end

@testset "cycle cut selection" begin
    rng = PCG.PCGStateOneseq(UInt64, 11)

    # brute-force: every k-subset of cycle edges whose k arcs have pop in [lo, hi]
    function valid_collections(pops, k, lo, hi)
        m = length(pops)
        function arcs_ok(B)
            for j in 1:k
                a, b = B[j], B[mod1(j + 1, k)]
                nodes = a < b ? (a+1:b) : vcat(a+1:m, 1:b)
                lo <= sum(pops[nodes]) <= hi || return false
            end
            return true
        end
        return [collect(B) for B in Combinatorics.combinations(1:m, k)
                if arcs_ok(collect(B))]
    end
    # every (removed edge, marked edges) route of the valid collections
    routes_of(valid) = [(e, sort!(filter(!=(e), B))) for B in valid for e in B]

    @testset "12 unit nodes, 4 districts of 3" begin
        pops, ew = ones(12), ones(12)
        valid = valid_collections(pops, 4, 3.0, 3.0)
        @test length(valid) == 3
        cut_data = CycleWalk.CycleCutData(pops, ew, collect(1:12), 4, (3.0, 3.0))
        @test cut_data.feasible == collect(1:12)
        @test all(cut_data.Z .== 1.0)
        @test cut_data.D == 12.0
        # 3 collections x 4 possible removed edges = 12 equally likely routes
        for (e, marked) in routes_of(valid)
            @test CycleWalk.log_prob_cut_route(cut_data, e, marked) ≈ log(1 / 12)
        end
        (e, marked), lp = CycleWalk.select_cycle_cuts(cut_data, rng)
        @test sort(vcat(e, marked)) in valid
        @test lp ≈ log(1 / 12)
        # invalid routes
        @test CycleWalk.log_prob_cut_route(cut_data, 3, [6, 8, 12]) == -Inf
        @test CycleWalk.log_prob_cut_route(cut_data, 3, [6, 9]) == -Inf
        @test CycleWalk.log_prob_cut_route(cut_data, 3, [3, 6, 9]) == -Inf
    end

    @testset "infeasible" begin
        cut_data = CycleWalk.CycleCutData(ones(12), ones(12), collect(1:12), 5,
                                          (3.0, 3.0))
        @test isempty(cut_data.feasible)
        @test CycleWalk.select_cycle_cuts(cut_data, rng) === nothing
    end

    pops = Float64[2, 1, 3, 2, 2, 1, 3, 2, 1]
    ew = Float64[1, 2, 1, 3, 1, 1, 2, 1, 1]
    m, k, lo, hi = 9, 3, 4.0, 8.0
    valid = valid_collections(pops, k, lo, hi)
    routes = routes_of(valid)
    @test !isempty(valid)
    # any superset of the true boundary edges is a legitimate candidate set
    cut_data = CycleWalk.CycleCutData(pops, ew, collect(1:m), k, (lo, hi))
    @test Set(cut_data.feasible) == Set(vcat(valid...))

    @testset "route probabilities" begin
        lps = [CycleWalk.log_prob_cut_route(cut_data, e, mk) for (e, mk) in routes]
        @test sum(exp, lps) ≈ 1.0
        # a route's probability is the product of 1/w over its cuts, over D:
        # which edge was removed first does not matter
        for B in valid
            lp = [CycleWalk.log_prob_cut_route(cut_data, e, filter(!=(e), B))
                  for e in B]
            @test all(≈(sum(log(1 / ew[b]) for b in B) - log(cut_data.D)), lp)
        end
    end

    @testset "sampling matches the scored probabilities" begin
        counts = Dict(r => 0 for r in routes)
        nsamp = 20000
        for _ in 1:nsamp
            (e, marked), lp = CycleWalk.select_cycle_cuts(cut_data, rng)
            counts[(e, marked)] += 1
            @test lp ≈ CycleWalk.log_prob_cut_route(cut_data, e, marked)
        end
        for (e, mk) in routes
            p = exp(CycleWalk.log_prob_cut_route(cut_data, e, mk))
            @test abs(counts[(e, mk)] / nsamp - p) < 0.015
        end
    end

    @testset "DP walk steps reproduce P(marked | removed edge)" begin
        inv_w = 1.0 ./ ew
        for (e, mk) in routes
            tab = CycleWalk.rotated_cut_table(pops, inv_w, e, k, (lo, hi))
            rot = sort!([mod1(b - e, m) for b in mk], rev=true)
            log_cond, q = 0.0, m
            for (t, p) in zip(k:-1:1, vcat(rot, 0))
                step = CycleWalk.select_next_cut(tab, q, t, rng; forced=p)
                @test step !== nothing
                log_cond += step[2]
                q = p
            end
            @test log_cond ≈ sum(log(inv_w[b]) for b in mk) - log(cut_data.Z[e])
        end
        # a cut that cannot end the last district is not a valid choice
        tab = CycleWalk.rotated_cut_table(pops, inv_w, first(first(routes)), k,
                                          (lo, hi))
        @test_throws AssertionError CycleWalk.select_next_cut(tab, m, k, rng;
                                                              forced=1)
    end
end

@testset "bud_walk state consistency" begin
    cons = initialize_constraints()
    add_constraint!(cons, PopulationConstraint(small_square_graph, 4, 0.5))
    rng = PCG.PCGStateOneseq(UInt64, 3)
    partition = LinkCutPartition(small_square_graph, cons, 4; rng=rng)
    bud = CycleWalk.build_bud_walk(cons)

    function consistent(partition)
        roots, _ = CycleWalk.get_district_roots(partition.lct)
        ok = length(roots) == partition.num_dists &&
             Set(roots) == Set(partition.district_roots) &&
             length(partition.roots_to_district) == partition.num_dists &&
             all(partition.roots_to_district[partition.district_roots[d]] == d
                 for d in 1:partition.num_dists)
        # labels must match the trees hanging off the recorded roots
        saved = copy(partition.node_to_dist_update)
        CycleWalk.assign_district_map!(partition, update=true)
        ok &= partition.node_to_dist_update == partition.node_to_dist
        partition.node_to_dist_update .= saved
        cross = CycleWalk.find_cross_district_edges!(
            partition, collect(1:partition.num_dists),
            typeof(partition.cross_district_edges)())
        ok &= Set(keys(cross)) == Set(keys(partition.cross_district_edges))
        # same edges and weights, whatever their orientation
        canon(es) = Set((min(e.src, e.dst), max(e.src, e.dst), e.weight) for e in es)
        for (key, edgeset) in cross
            ok &= canon(edgeset) == canon(partition.cross_district_edges[key])
        end
        return ok
    end

    num_proposed = 0
    num_accepted = 0
    for _ in 1:1500
        before = copy(partition.node_to_dist)
        p, update = bud(partition, rng)
        @test consistent(partition)
        @test partition.node_to_dist == before
        if p > 0
            num_proposed += 1
            if rand(rng) < min(p, 1)
                CycleWalk.update_partition!(partition, update)
                num_accepted += 1
                @test consistent(partition)
            end
        end
    end
    @test num_proposed > 0
    @test num_accepted > 0
end

@testset "collapsed cycle weights: subtree sums == topological_sort" begin
    cons = initialize_constraints()
    add_constraint!(cons, PopulationConstraint(small_square_graph, 2, 0.5))
    rng = PCG.PCGStateOneseq(UInt64, 5)
    partition = LinkCutPartition(small_square_graph, cons, 2; rng=rng)
    num_checked = 0
    for _ in 1:200
        # any two nodes of one district tree: the tree path between them
        d = rand(rng, 1:partition.num_dists)
        nodes_d = findall(==(d), partition.node_to_dist)
        length(nodes_d) >= 2 || continue
        u, v = rand(rng, nodes_d, 2)
        u == v && continue
        path = CycleWalk.get_cycle_path_for_edge!(partition, (u, v))
        by_dict = CycleWalk.get_collapsed_cycle_weights_path(
            path, CycleWalk.topological_sort(path[1], partition))
        by_sums = CycleWalk.get_collapsed_cycle_weights_path(path, partition)
        @test by_sums == by_dict
        @test sum(by_sums) == sum(partition.graph.node_attributes[w][partition.graph.pop_col]
                                  for w in nodes_d)
        num_checked += 1
    end
    @test num_checked > 50
end

@testset "LogDistrictTreesData: cached and proposed values match a rebuild" begin
    cons = initialize_constraints()
    add_constraint!(cons, PopulationConstraint(small_square_graph, 4, 0.5))
    rng = PCG.PCGStateOneseq(UInt64, 3)
    partition = LinkCutPartition(small_square_graph, cons, 4; rng=rng)
    bud = CycleWalk.build_bud_walk(cons)
    lifted = CycleWalk.build_lifted_tree_cycle_walk(cons)
    rebuild(p) = CycleWalk.log_nspanning(CycleWalk.build_district_quotient_graph(p)[1])

    num_accepted = 0
    for i in 1:3000
        walk = isodd(i ÷ 500) ? lifted : bud
        p, update = walk(partition, rng)
        # the cache is only sometimes built/staged before an accept, so both the
        # synced and the unsynced commit paths are exercised
        i % 7 == 0 && @test CycleWalk.get_log_district_trees(partition) ≈ rebuild(partition)
        update === nothing && continue
        p > 0 || continue
        proposed = i % 3 == 0 ? CycleWalk.get_log_district_trees(partition; update=update) : NaN
        if rand(rng) < min(p, 1)
            CycleWalk.update_partition!(partition, update)
            num_accepted += 1
            @test CycleWalk.get_log_district_trees(partition) ≈ rebuild(partition)
            isnan(proposed) || @test CycleWalk.get_log_district_trees(partition) ≈ proposed
        end
    end
    @test num_accepted > 100
end
