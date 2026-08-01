# Tests for derive_node_columns! (src/graph/derived_columns.jl) — computing a new
# node attribute from existing ones, e.g. a unique node name joined from columns
# that aren't unique alone (the motivating case: NC's prec_id repeats across
# counties — see docs/run_cyclewalk_toml.md and examples/toml/param_nc.toml).

@testset "derive_node_columns!" begin
    derived_columns_json = joinpath(testdir, "test_graphs", "4x4pct_2x2cnty.json")
    fresh_base_graph() = BaseGraph(derived_columns_json, "pop";
        inc_node_data=Set(["county", "pct", "pop", "area", "border_length"]),
        area_col="area", node_border_col="border_length", edge_perimeter_col="length")

    @testset "a derived string column is written to every node" begin
        bg = fresh_base_graph()
        derive_node_columns!(bg, Dict("uid" => "county * \"_\" * pct"))
        @test all(attrs -> haskey(attrs, "uid"), bg.node_attributes)
        for attrs in bg.node_attributes
            @test attrs["uid"] == attrs["county"] * "_" * attrs["pct"]
        end
        # values distinct where the raw pair already was (this graph's pct IS
        # unique alone, but the join must still be internally consistent)
        @test length(Set(a["uid"] for a in bg.node_attributes)) ==
              length(Set((a["county"], a["pct"]) for a in bg.node_attributes))
    end

    @testset "a derived numeric column" begin
        bg = fresh_base_graph()
        derive_node_columns!(bg, Dict("half_pop" => "pop / 2"))
        for attrs in bg.node_attributes
            @test attrs["half_pop"] == attrs["pop"] / 2
        end
    end

    @testset "the derived column is usable exactly like a raw one downstream" begin
        bg = fresh_base_graph()
        derive_node_columns!(bg, Dict("uid" => "county * \"_\" * pct"))
        graph = MultiLevelGraph(bg, ["uid"])   # would throw if "uid" weren't a real attr
        @test graph.graphs_by_level[1].num_nodes == bg.num_nodes
    end

    @testset "colliding with an existing column is refused" begin
        bg = fresh_base_graph()
        @test_throws ArgumentError derive_node_columns!(bg, Dict("pct" => "county"))
        # also refused against an EARLIER entry in the same call
        @test_throws ArgumentError derive_node_columns!(bg,
            ["uid" => "county * pct", "uid" => "pct * county"])
    end

    @testset "multiple derived columns in one call" begin
        bg = fresh_base_graph()
        derive_node_columns!(bg, ["uid" => "county * \"_\" * pct",
                                  "half_pop" => "pop / 2"])
        @test haskey(bg.node_attributes[1], "uid")
        @test haskey(bg.node_attributes[1], "half_pop")
    end

    @testset "non-string name/expr are refused" begin
        bg = fresh_base_graph()
        @test_throws ArgumentError derive_node_columns!(bg, Dict(1 => "county"))
        @test_throws ArgumentError derive_node_columns!(bg, Dict("uid" => 5))
    end
end
