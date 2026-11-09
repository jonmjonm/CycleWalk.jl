## Run:
# julia run_bud_ct_county.jl
#
# BUD walk on Connecticut with edge weights that favor keeping counties together.
# Edges between two different counties have their weight divided by `cnty_scale`, so
# the weighted spanning-tree (and forest) measures put less mass on trees, and hence
# districts, that cross county lines. The weight function lives in this script: it is
# an example of a custom `edge_weight_func`, not part of the package.
#
# The chain is run in blocks so the county splits can be recorded as it goes. Set
# `cnty_scale = 1.0` for the unweighted measure to compare; raising it lowers the
# number of (county, district) pieces, but modestly: the weights tilt the measure,
# they do not forbid splits.

## Activate the CycleWalk environment and load necessary packages
import Pkg
Pkg.activate(".")
Pkg.instantiate()

using RandomNumbers
using CycleWalk
using Statistics

gamma = 1.0 # 0 is spanning forest measure, 1 is partition
cnty_scale = 10.0 # Float64 > 1; cross-county edge weights are divided by this
iso_weight = 0.0 # weight on the sum of isoperimetric ratios; i.e. Polsby-Popper

num_dists = 5
rng = PCG.PCGStateOneseq(UInt64, 4541901234)
pop_dev = 0.02 # population deviation (fraction from ideal)
steps = 10^5
outfreq = 10^3 # steps per output block

## An edge weight function for `modify_edge_weights!`: edges whose endpoints lie in
## different counties are scaled down by `cnty_scale`. It reads the edge's current
## weight, so apply it to a graph only once.
function edge_weight_func(
    base_graph::BaseGraph,
    n1::Int, n2::Int;
    cnty_scale::Float64=1.0,
    cnty_field::Union{String, Nothing}=nothing,
)
    @assert !(cnty_scale != 1.0 && cnty_field === nothing)

    cur_weight = base_graph.simple_graph.weights[n1, n2]
    if cnty_field !== nothing && cnty_scale != 1.0
        @assert cnty_scale > 1
        cnty1 = base_graph.node_attributes[n1][cnty_field]
        cnty2 = base_graph.node_attributes[n2][cnty_field]
        if cnty1 != cnty2
            return cur_weight / cnty_scale
        end
    end

    return cur_weight
end

## build graph, then weight its edges (once)
pctGraphPath = joinpath("data","ct","CT_pct20.json")
isfile(pctGraphPath) || error("map file not found: $pctGraphPath")
nodeData = Set(["COUNTY", "NAME", "POP20", "area", "border_length"]);
graph = Graph(pctGraphPath, "POP20", "NAME"; inc_node_data=nodeData,
              area_col="area", node_border_col="border_length",
              edge_perimeter_col="length")
weight_func(g, n1, n2) = edge_weight_func(g, n1, n2; cnty_scale=cnty_scale,
                                          cnty_field="COUNTY")
modify_edge_weights!(graph, weight_func)
base_graph = graph.graphs_by_level[end] # node attributes, for the county labels

## build partition
constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))
partition = LinkCutPartition(graph, constraints, num_dists; rng=rng,
                             verbose=true);

## build proposal; BUD includes an internal forest walk
proposal = build_bud_walk(constraints)

## build measure
measure = Measure()
push_energy!(measure, get_log_spanning_forests, gamma) # add spanning forests energy
if iso_weight > 0
    push_energy!(measure, get_isoperimetric_score, iso_weight)
end

## establish output name and path
atlasName = "bud_county"*string(cnty_scale)
atlasName *= "_gamma"*string(gamma)
atlasName *= "_iso"*string(iso_weight)
atlasName *= ".jsonl.gz" # or just ".jsonl" for an uncompressed output
output_file_path = joinpath("output","ct", atlasName) # add output directory to path
mkpath(dirname(output_file_path))

## establish writer to which the output will be written
ad_param = Dict{String, Any}("popdev" => pop_dev,
                             "county weight scale" => cnty_scale)
writer = Writer(measure, constraints, partition, output_file_path;
                additional_parameters=ad_param)
push_writer!(writer, get_log_spanning_trees) # add spanning trees count to writer
push_writer!(writer, get_log_spanning_forests) # add spanning forests count to writer
if iso_weight > 0
    push_writer!(writer, get_isoperimetric_scores) # add isoperimetric scores to writer
end

## county splits: the number of counties that intersect more than one district, and
## the number of (county, district) pieces. Every county needs at least one piece, so
## `pieces - number of counties` is the excess; it is a finer measure than the number
## of split counties, which is near the total on CT (8 counties, 5 districts).
county_of = [base_graph.node_attributes[v]["COUNTY"]
             for v in 1:base_graph.num_nodes]
function county_splits(partition)
    districts_of = Dict{String, Set{Int}}()
    for (v, county) in enumerate(county_of)
        push!(get!(districts_of, county, Set{Int}()), partition.node_to_dist[v])
    end
    return count(ds -> length(ds) > 1, values(districts_of)),
           sum(length, values(districts_of))
end

## run MCMC sampler in blocks, recording the county splits after each block
println("running mcmc; outputting here: "* output_file_path)
num_counties = length(unique(county_of))
println("county weight scale: ", cnty_scale, "; counties: ", num_counties,
        "; initially split: ", county_splits(partition)[1],
        " (", county_splits(partition)[2], " pieces)")
split_counties = Int[]
pieces = Int[]
for block in 1:(steps ÷ outfreq)
    block_steps = ((block - 1) * outfreq + 1, block * outfreq)
    run_metropolis_hastings!(partition, proposal, measure, block_steps, rng,
                             writer=writer, output_freq=outfreq,
                             seed=4541901234)
    split, num_pieces = county_splits(partition)
    push!(split_counties, split)
    push!(pieces, num_pieces)
end
close_writer(writer) # close atlas

println("per block: split counties mean ", round(mean(split_counties), digits=2),
        " of ", num_counties, "; (county, district) pieces mean ",
        round(mean(pieces), digits=2), " (", num_counties, " if no county is split)")
