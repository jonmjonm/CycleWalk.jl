## Run:
# julia -t 4 run_ais_ct.jl
#
# Annealed importance sampling (AIS): a base chain samples the spanning-forest
# measure (gamma = 0), and each retained sample is annealed toward the target
# measure (gamma = 1 with an isoperimetric penalty) while its log importance
# weight is accumulated. Annealing runs execute on `ntasks` concurrent tasks —
# start Julia with threads (-t) to run them in parallel. Sample plans land in
# the output Atlas in base-chain order with their log weights as map weights.

## Activate the CycleWalk environment and load necessary packages
import Pkg
Pkg.activate(".")
Pkg.instantiate()

using RandomNumbers
using CycleWalk

twocycle_frac = 0.1
gamma = 1.0 # target spanning-forest weight (base chain uses 0)
iso_weight = 0.3 # target weight on the sum of isoperimetric ratios

@assert 0 ≤ twocycle_frac ≤ 1

num_dists = 5
rng = PCG.PCGStateOneseq(UInt64, 4541901234)
pop_dev = 0.02 # population deviation (fraction from ideal)

total_steps = 10^4          # total base-chain steps
base_steps_per_sample = 500 # base-chain steps between annealed samples
steps_per_annealing = 2000  # annealing steps per sample
ntasks = Threads.nthreads() # concurrent annealing runs

## build graph
pctGraphPath = joinpath("data","ct","CT_pct20.json")
isfile(pctGraphPath) || error("map file not found: $pctGraphPath")
nodeData = Set(["COUNTY", "NAME", "POP20", "area", "border_length"]);
graph = Graph(pctGraphPath, "POP20", "NAME"; inc_node_data=nodeData,
              area_col="area", node_border_col="border_length",
              edge_perimeter_col="length")

## build partition
constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))
partition = LinkCutPartition(graph, constraints, num_dists; rng=rng,
                             verbose=true);

## build proposal
cycle_walk = build_two_tree_cycle_walk(constraints)
internal_walk = build_one_tree_cycle_walk(constraints)
proposal = [(twocycle_frac, cycle_walk),
            (1.0-twocycle_frac, internal_walk)]

## build target measure; the annealing schedule ramps both energy weights
## linearly from 0 (the base measure) to their target values
measure = Measure()
push_energy!(measure, get_log_spanning_forests, gamma)
push_energy!(measure, get_isoperimetric_score, iso_weight)

function modify_measure!(m::Measure, step::Int, total::Int)
    m.weights[get_log_spanning_forests] = gamma*step/total
    m.weights[get_isoperimetric_score] = iso_weight*step/total
end

## establish output name and path
atlasName = "ais_2cyclefrac_"*string(twocycle_frac)
atlasName *= "_gamma"*string(gamma)
atlasName *= "_iso"*string(iso_weight)
atlasName *= ".jsonl.gz" # or just ".jsonl" for an uncompressed output
output_file_path = joinpath("output","ct", atlasName)
mkpath(dirname(output_file_path))

## establish writer; weight_type=Float64 so maps carry log importance weights
ad_param = Dict{String, Any}("popdev" => pop_dev,
                             "steps per annealing" => steps_per_annealing)
writer = Writer(measure, constraints, partition, output_file_path;
                additional_parameters=ad_param, weight_type=Float64)
push_writer!(writer, get_isoperimetric_scores)

## run annealed importance sampling
println("running AIS on ", ntasks, " task(s); outputting here: ",
        output_file_path)
log_weights = run_annealed_importance_sampling!(
    partition, proposal, measure, modify_measure!, total_steps,
    base_steps_per_sample, steps_per_annealing, rng;
    writer=writer, ntasks=ntasks, seed=4541901234)
close_writer(writer) # close atlas

## summarize the log importance weights; subtract the max before
## exponentiating so the weights are numerically stable
println("collected ", length(log_weights), " samples")
max_lw = maximum(log_weights)
w = exp.(log_weights .- max_lw)
ess = sum(w)^2/sum(w.^2) # effective sample size
println("log weights: min=", round(minimum(log_weights), digits=3),
        " max=", round(max_lw, digits=3))
println("effective sample size: ", round(ess, digits=2), " of ",
        length(log_weights))
