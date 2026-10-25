name = "NC graph regional constraints"
@testset "$name" begin
    nc_json = joinpath(testdir, "test_graphs", "NC_pct21.json")
    node_data = Set(["county", "prec_id", "pop2020cen", "area", "border_length"])

    base_graph = BaseGraph(nc_json, "pop2020cen";
                           inc_node_data=node_data,
                           area_col="area",
                           node_border_col="border_length",
                           edge_perimeter_col="length")
    graph = MultiLevelGraph(base_graph, ["prec_id"])

    num_dists = 20
    pop_dev = 0.08

    population_constraints = initialize_constraints()
    add_constraint!(population_constraints,
                    PopulationConstraint(graph, num_dists, pop_dev))

    @testset "partition without county preservation violates strict packing" begin
        strict_pack = PackRegionConstraint(graph, "county";
                                           num_dists=num_dists,
                                           unpack=0)

        rng = PCG.PCGStateOneseq(UInt64, 1001)
        partition = LinkCutPartition(graph, population_constraints, num_dists;
                                     rng=rng)

        @test !satisfies_constraint(partition, strict_pack)
    end

    @testset "pack constraint is preserved during proposals" begin
        # unpack=1 keeps a non-trivial packing requirement while allowing
        # deterministic initialization from this NC graph.
        pack_constraint = PackRegionConstraint(graph, "county";
                                               num_dists=num_dists,
                                               unpack=1)

        constraints = deepcopy(population_constraints)
        add_constraint!(constraints, pack_constraint)

        init_rng = PCG.PCGStateOneseq(UInt64, 7015)
        partition = LinkCutPartition(graph, constraints, num_dists;
                                     rng=init_rng)
        @test satisfies_constraint(partition, pack_constraint)

        constrained = initialize_constraints()
        add_constraint!(constrained, PopulationConstraint(graph, num_dists, pop_dev))
        add_constraint!(constrained, pack_constraint)

        cycle_walk = build_lifted_tree_cycle_walk(constrained)
        internal_walk = build_internal_forest_walk(constrained)
        proposal = [(0.2, cycle_walk), (0.8, internal_walk)]

        measure = Measure()
        chain_rng = PCG.PCGStateOneseq(UInt64, 8888)
        preserved = true
        for _ = 1:1000
            run_metropolis_hastings!(partition, proposal, measure, 1, chain_rng)
            if !satisfies_constraint(partition, pack_constraint)
                preserved = false
                break
            end
        end
        @test preserved
        
        run_metropolis_hastings!(partition, proposal, measure, 1000, chain_rng)
        satisfied1 = satisfies_constraint(partition, pack_constraint)
        partition.identifier += 1 # force update of energy data
        satisfied2 = satisfies_constraint(partition, pack_constraint)
        @test satisfied1 && satisfied2
    end

    @testset "partition without cap constraint violates strict cap" begin
        strict_cap = CapRegionDistricts(graph, "county";
                                        num_dists=num_dists,
                                        excess_split=0)

        rng = PCG.PCGStateOneseq(UInt64, 1001)
        partition = LinkCutPartition(graph, population_constraints, num_dists;
                                     rng=rng)

        @test !satisfies_constraint(partition, strict_cap)
    end

    @testset "cap constraint is preserved during proposals" begin
        cap_constraint = CapRegionDistricts(graph, "county";
                                            num_dists=num_dists,
                                            excess_split=1)
        constraints = deepcopy(population_constraints)
        add_constraint!(constraints, cap_constraint)

        init_rng = PCG.PCGStateOneseq(UInt64, 7015)
        partition = LinkCutPartition(graph, constraints, num_dists;
                                     rng=init_rng)
        @test satisfies_constraint(partition, cap_constraint)

        constrained = initialize_constraints()
        add_constraint!(constrained, PopulationConstraint(graph, num_dists, pop_dev))
        add_constraint!(constrained, cap_constraint)

        cycle_walk = build_lifted_tree_cycle_walk(constrained)
        internal_walk = build_internal_forest_walk(constrained)
        proposal = [(0.2, cycle_walk), (0.8, internal_walk)]

        measure = Measure()
        chain_rng = PCG.PCGStateOneseq(UInt64, 9998)
        preserved = true
        for _ = 1:200
            run_metropolis_hastings!(partition, proposal, measure, 1, chain_rng)
            if !satisfies_constraint(partition, cap_constraint)
                preserved = false
                break
            end
        end
        @test preserved
    end
end