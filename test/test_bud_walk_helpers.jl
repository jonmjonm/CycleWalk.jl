using Test
using Graphs
using SimpleWeightedGraphs
using RandomNumbers
using Combinatorics

# From-scratch district quotient graph (summed cross-edge weights per district pair).
function rebuilt_quotient_graph(partition)
    g = SimpleWeightedGraph(partition.num_dists)
    for ((a, b), edgeset) in partition.cross_district_edges
        add_edge!(g, a, b, sum(e.weight for e in edgeset))
    end
    return g
end

@testset "bud_walk_helpers" begin
    @test CycleWalk.ordered_edge_key(7, 3) == (3, 7)

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
        # a quotient path of the cached marked tree between two random districts
        tree = CycleWalk.refresh_quotient_tree!(part, rng; updown_steps=0)
        d1, d2 = rand(rng, 1:part.num_dists, 2)
        d1 == d2 && continue
        districts, slots = CycleWalk.quotient_tree_path(tree, d1, d2)
        qpath = [CycleWalk.QuotientPathVertex(d) for d in districts]
        specified = Dict{Edge{Int}, Tuple{Int, Int}}(
            Edge(districts[i], districts[i+1]) => tree.edges[s]
            for (i, s) in enumerate(slots))
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

@testset "collapsed cycle weights: subtree sums == topological_sort differences" begin
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
        below = CycleWalk.topological_sort(path[1], partition)  # subtree weights
        by_dict = [below[path[i].vertex] - (i < length(path) ? below[path[i+1].vertex] : 0.0)
                   for i in eachindex(path)]
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
    rebuild(p) = CycleWalk.log_nspanning(rebuilt_quotient_graph(p))

    num_accepted = 0
    for i in 1:3000
        p, update = bud(partition, rng)
        # the cache is only sometimes built/staged before an accept, so both the
        # synced and the unsynced commit paths are exercised; BUD's internal walk
        # supplies single-district updates and its cross moves multi-district ones
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

@testset "reset_update_buffer!: staged constraint data belongs to the proposal" begin
    # CapRegionDistricts stages RegionalSplitData (keyed on update.identifier) in
    # every BUD constraint check; the buffers are carried across proposals
    for excess_split in (3, 1)
        cons = initialize_constraints()
        add_constraint!(cons, PopulationConstraint(small_square_graph, 4, 0.5))
        add_constraint!(cons, CapRegionDistricts(small_square_graph, "county";
                                                 excess_split=excess_split,
                                                 num_dists=4))
        rng = PCG.PCGStateOneseq(UInt64, 3)
        partition = LinkCutPartition(small_square_graph, cons, 4; rng=rng)
        bud = CycleWalk.build_bud_walk(cons)
        node_attrs = partition.graph.node_attributes
        county(v) = node_attrs[v]["county"]

        # hold the state fixed over many proposals so that proposals with equal
        # update.identifier (same changed districts, links and cuts) recur
        num_checked = 0
        for _ in 1:40
            for _ in 1:300
                p, update = bud(partition, rng)
                update === nothing && continue
                isempty(update.links) && continue  # partition unchanged: nothing staged
                data = partition.energy_data[(CycleWalk.RegionalSplitData,
                                              ("county",))]
                for d in update.changed_districts
                    want = Set(county(v) for v in eachindex(partition.node_to_dist)
                               if partition.node_to_dist_update[v] == d)
                    @test data.dists_to_regions_update[d] == want
                end
                num_checked += 1
            end
            for _ in 1:50
                p, update = bud(partition, rng)
                if p > 0 && rand(rng) < min(p, 1)
                    CycleWalk.update_partition!(partition, update)
                    break
                end
            end
        end
        @test num_checked > 300
    end
end

# Structural invariants of the cached marked tree M.
function marked_tree_valid(partition)
    data = partition.energy_data[CycleWalk.QuotientTreeData]
    D = partition.num_dists
    n2d = partition.node_to_dist
    ok = length(data.edges) == D - 1 && length(data.slot_of) == D - 1
    parent = collect(1:D)
    find(x) = parent[x] == x ? x : (parent[x] = find(parent[x]))
    for (s, (u, v)) in enumerate(data.edges)
        a, b = n2d[u], n2d[v]
        ok &= u < v && a != b && data.slot_of[(u, v)] == s
        ok &= data.ends[s] == (a, b) && s in data.adj[a] && s in data.adj[b]
        # a concrete cross edge of the partition, outside the forest
        pair = haskey(partition.cross_district_edges, (a, b)) ? (a, b) : (b, a)
        ok &= any(e -> CycleWalk.ordered_edge_key(e) == (u, v),
                  partition.cross_district_edges[pair])
        ok &= !CycleWalk.neighbors_huh(partition.lct[u], partition.lct[v])
        ok &= data.edge_weights[s] == Float64(
            CycleWalk.get_weight(partition.graph.simple_graph, u, v))
        ra, rb = find(a), find(b)
        ok &= ra != rb  # no cycle
        parent[ra] = rb
    end
    ok &= length(unique(find(d) for d in 1:D)) == 1  # spanning
    ok &= sum(length, data.adj) == 2 * (D - 1)
    return ok
end

@testset "marked tree stays a valid spanning tree of the quotient graph" begin
    for (wilson_prob, updown_steps) in ((0.0, 0), (0.0, 5), (0.2, 3))
        cons = initialize_constraints()
        add_constraint!(cons, PopulationConstraint(small_square_graph, 4, 0.5))
        rng = PCG.PCGStateOneseq(UInt64, 11)
        partition = LinkCutPartition(small_square_graph, cons, 4; rng=rng)
        bud = CycleWalk.build_bud_walk(cons; wilson_prob=wilson_prob,
                                       updown_steps=updown_steps)
        num_accepted = 0
        num_marked_only = 0
        for _ in 1:3000
            p, update = bud(partition, rng)
            @test marked_tree_valid(partition)
            if p > 0 && rand(rng) < min(p, 1)
                num_marked_only += isempty(update.links)
                CycleWalk.update_partition!(partition, update)
                num_accepted += 1
                data = partition.energy_data[CycleWalk.QuotientTreeData]
                @test data.identifier == partition.identifier  # never went stale
                @test marked_tree_valid(partition)
            end
        end
        @test num_accepted > 300
    end
end

@testset "marked tree is redrawn when the partition moved without it" begin
    cons = initialize_constraints()
    add_constraint!(cons, PopulationConstraint(small_square_graph, 4, 0.5))
    rng = PCG.PCGStateOneseq(UInt64, 5)
    partition = LinkCutPartition(small_square_graph, cons, 4; rng=rng)
    bud = CycleWalk.build_bud_walk(cons)
    for i in 1:2000
        p, update = bud(partition, rng)
        if p > 0 && rand(rng) < min(p, 1)
            CycleWalk.update_partition!(partition, update)
        end
        data = partition.energy_data[CycleWalk.QuotientTreeData]
        @test data.identifier == partition.identifier
        if i % 50 == 0
            # an accepted move by another proposal bumps the identifier and carries
            # no marked change for this cache
            partition.identifier += 1
            for eData in values(partition.energy_data)
                eData isa CycleWalk.QuotientTreeData || continue
                @test eData.identifier != partition.identifier
            end
            bud(partition, rng)
            @test data.identifier == partition.identifier
            @test marked_tree_valid(partition)
        end
    end
end

@testset "marked tree given the forest follows w(M)/τ(Q)" begin
    # unequal edge weights, so that w and 1/w are distinguishable
    weighted_base = deepcopy(small_square_graph.graphs_by_level[end])
    CycleWalk.modify_edge_weights!(weighted_base, (g, u, v) -> 1.0 + (7u + 3v) % 5)
    weighted_graph = MultiLevelGraph(weighted_base, ["pct"])
    cons = initialize_constraints()
    add_constraint!(cons, PopulationConstraint(weighted_graph, 4, 0.5))
    rng = PCG.PCGStateOneseq(UInt64, 21)
    partition = LinkCutPartition(weighted_graph, cons, 4; rng=rng)
    D = partition.num_dists
    n2d = partition.node_to_dist

    # every spanning tree of the cross-edge multigraph, with its weight
    cross = [(CycleWalk.ordered_edge_key(e), Float64(CycleWalk.weight(e)))
             for es in values(partition.cross_district_edges) for e in es]
    exact = Dict{Set{Tuple{Int,Int}}, Float64}()
    for combo in combinations(cross, D - 1)
        parent = collect(1:D)
        find(x) = parent[x] == x ? x : (parent[x] = find(parent[x]))
        is_tree = true
        for ((u, v), _) in combo
            ra, rb = find(n2d[u]), find(n2d[v])
            ra == rb && (is_tree = false; break)
            parent[ra] = rb
        end
        is_tree || continue
        exact[Set(first.(combo))] = prod(last.(combo))
    end
    total = sum(values(exact))
    @test length(exact) > 10
    @test total ≈ exp(CycleWalk.get_log_district_trees(partition))  # τ(Q)

    function tv_distance(draws)
        counts = Dict{Set{Tuple{Int,Int}}, Int}()
        for t in draws
            counts[t] = get(counts, t, 0) + 1
        end
        n = length(draws)
        @test all(t -> haskey(exact, t), keys(counts))
        return sum(abs(get(counts, t, 0) / n - w / total) for (t, w) in exact) / 2
    end

    data = CycleWalk.QuotientTreeData(partition)
    # exact redraw
    wilson_draws = Vector{Set{Tuple{Int,Int}}}()
    for _ in 1:100_000
        CycleWalk.rebuild_quotient_tree!(data, partition, rng)
        push!(wilson_draws, Set(data.edges))
    end
    @test tv_distance(wilson_draws) < 0.03

    # heat-bath chain, started from a Wilson draw
    CycleWalk.rebuild_quotient_tree!(data, partition, rng)
    chain_draws = Vector{Set{Tuple{Int,Int}}}()
    for _ in 1:300_000
        CycleWalk.quotient_updown_step!(data, partition, rng)
        push!(chain_draws, Set(data.edges))
    end
    @test tv_distance(chain_draws) < 0.03

end
