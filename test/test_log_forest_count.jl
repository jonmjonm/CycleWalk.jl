# Correctness check for get_log_spanning_trees's dense/sparse Cholesky dispatch
# (DENSE_CHOLESKY_MAX_M in src/measure/energy/log_forest_count.jl): both branches
# must agree with an independent reference on a small (dense-path) and a large
# (sparse-path) district, and both must return -Inf for a disconnected district.
#
# Builds SimpleWeightedGraphs directly rather than going through the full
# Graph/BaseGraph/partition machinery -- get_log_spanning_trees's low-level kernel
# only needs (node_to_dist, simple_graph, di), so this exercises it in isolation
# with exact, deterministic control over district size relative to the threshold.

using SimpleWeightedGraphs: SimpleWeightedGraph, add_edge!
using Graphs: induced_subgraph, is_connected, laplacian_matrix, nv
using LinearAlgebra: logdet

@testset "get_log_spanning_trees: dense/sparse Cholesky dispatch" begin
    # Independent reference: Graphs.jl's own Laplacian + dense logdet on the
    # induced subgraph -- a different code path from CycleWalk's hand-rolled
    # kernel (which never materializes an induced_subgraph), so this isn't just
    # comparing the two branches against each other.
    function reference_log_spanning_trees(node_to_dist, simple_graph, di)
        nodes = findall(==(di), node_to_dist)
        length(nodes) <= 1 && return 0.0
        sub, _ = induced_subgraph(simple_graph, nodes)
        is_connected(sub) || return -Inf
        L = Matrix(laplacian_matrix(sub))
        return logdet(L[2:end, 2:end])
    end

    function cycle_graph(n)
        g = SimpleWeightedGraph(n)
        for i in 1:(n - 1)
            add_edge!(g, i, i + 1, 1.0)
        end
        add_edge!(g, n, 1, 1.0)
        return g
    end

    function two_disjoint_cycles(half_n)
        n = 2 * half_n
        g = SimpleWeightedGraph(n)
        for i in 1:(half_n - 1)
            add_edge!(g, i, i + 1, 1.0)
        end
        add_edge!(g, half_n, 1, 1.0)
        for i in (half_n + 1):(n - 1)
            add_edge!(g, i, i + 1, 1.0)
        end
        add_edge!(g, n, half_n + 1, 1.0)
        return g
    end

    @testset "small district (dense path, m < threshold)" begin
        n = 50
        g = cycle_graph(n)
        node_to_dist = ones(Int, n)
        @test (n - 1) < CycleWalk.DENSE_CHOLESKY_MAX_M
        got = get_log_spanning_trees(node_to_dist, g, 1)
        want = reference_log_spanning_trees(node_to_dist, g, 1)
        @test isapprox(got, want; rtol=1e-8)
    end

    @testset "large district (sparse path, m >= threshold)" begin
        n = 250
        g = cycle_graph(n)
        node_to_dist = ones(Int, n)
        @test (n - 1) >= CycleWalk.DENSE_CHOLESKY_MAX_M
        got = get_log_spanning_trees(node_to_dist, g, 1)
        want = reference_log_spanning_trees(node_to_dist, g, 1)
        @test isapprox(got, want; rtol=1e-6)
    end

    @testset "disconnected district, small (dense path)" begin
        g = two_disjoint_cycles(30)
        n = nv(g)
        node_to_dist = ones(Int, n)
        @test (n - 1) < CycleWalk.DENSE_CHOLESKY_MAX_M
        @test get_log_spanning_trees(node_to_dist, g, 1) == -Inf
    end

    @testset "disconnected district, large (sparse path)" begin
        g = two_disjoint_cycles(150)
        n = nv(g)
        node_to_dist = ones(Int, n)
        @test (n - 1) >= CycleWalk.DENSE_CHOLESKY_MAX_M
        @test get_log_spanning_trees(node_to_dist, g, 1) == -Inf
    end
end
