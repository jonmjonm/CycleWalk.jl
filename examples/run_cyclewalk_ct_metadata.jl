## Run:
# julia run_cyclewalk_ct_metadata.jl
#
# Same as run_cyclewalk_ct.jl, but focused on ONE thing: the general info that
# lands in the Atlas header (the file's 3rd line, the `atlasParam` dict).
#
# The Writer stamps EXECUTION metadata automatically -- no code needed:
#   * "user"        -- ENV["USER"] / ENV["USERNAME"] of whoever ran the script
#   * "script_name" -- basename of the script being executed (PROGRAM_FILE)
#   * "script"      -- the ENTIRE source of this script, read from disk, appended
#                      as the header's LAST key. Pass include_script=false to omit
#                      it (e.g. for a very large script or to keep files small).
#
# You can also add your OWN info two ways:
#   (1) `additional_parameters` in the `Writer` constructor -- the intended path
#       for anything you know up front. These win over the auto-stamped fields.
#   (2) Mutating `writer.atlas.atlasParam` directly -- for values you only learn
#       after the writer exists. This works because the header is written lazily
#       (by the first map / `run_*!`), so anything added before the run starts is
#       included.

## Activate the CycleWalk environment and load necessary packages
import Pkg
Pkg.activate(".")
Pkg.instantiate()

using Dates
using RandomNumbers
using CycleWalk


twocycle_frac = 0.1
gamma = 1.0 # 0 is spanning forest measure, 1 is partition
iso_weight = 0.3 # weight on the sum of isoperimetric ratios; i.e. Polsby-Popper

@assert 0 ≤ twocycle_frac ≤ 1

num_dists = 5
seed = 4541901234
rng = PCG.PCGStateOneseq(UInt64, seed)
pop_dev = 0.02 # population deviation (fraction from ideal)
cycle_walk_steps = 10^2
steps = Int(cycle_walk_steps/twocycle_frac)
outfreq = Int(1000/twocycle_frac)

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

## build measure
measure = Measure()
push_energy!(measure, get_log_spanning_forests, gamma) # add spanning forests energy
push_energy!(measure, get_isoperimetric_score, iso_weight) # add isoperimetric score energy

## establish output name and path
atlasName = "cycleWalk_ct_metadata.jsonl.gz" # or ".jsonl" for uncompressed output
output_file_path = joinpath("output","ct", atlasName) # add output directory to path
mkpath(dirname(output_file_path))

## (1) Header metadata you know up front: pass it via `additional_parameters`.
##     Every key here lands on the 3rd line of the Atlas file, alongside the
##     built-in fields (energies, energy weights, districts, population bounds,
##     constraints, package.version) and the auto-stamped user/script_name/script.
##     Values must be JSON-serializable.
ad_param = Dict{String, Any}(
    "popdev"      => pop_dev,
    "state"       => "CT",
    "run_label"   => "metadata demo",
    "notes"       => "example showing how to stamp custom info into the header",
    "created_at"  => string(now()),
)
## user / script_name / script are stamped automatically; include_script=true is
## the default (set include_script=false to leave the source blob out).
writer = Writer(measure, constraints, partition, output_file_path;
                description="CycleWalk CT run — header-metadata example",
                additional_parameters=ad_param,
                include_script=true)

## (2) Header metadata you only learn after the writer exists: mutate the
##     `atlasParam` dict directly. This must happen BEFORE the run starts, since
##     the first map (or the `run_*!` sampler) writes the header and freezes it.
writer.atlas.atlasParam["rng_seed"] = seed
writer.atlas.atlasParam["cycle_walk_steps"] = cycle_walk_steps
writer.atlas.atlasParam["twocycle_frac"] = twocycle_frac

push_writer!(writer, get_log_spanning_trees) # add spanning trees count to writer
push_writer!(writer, get_log_spanning_forests) # add spanning forests count to writer
push_writer!(writer, get_isoperimetric_scores) # add isoperimetric scores to writer

## run MCMC sampler
println("running mcmc; outputting here: "* output_file_path)
run_metropolis_hastings!(partition, proposal, measure, steps, rng,
                         writer=writer, output_freq=outfreq, seed=seed)
close_writer(writer) # close atlas

## Peek at the header we just wrote. The 3rd line is the atlasParam dict; because
## it now embeds the full script under "script" it can be long, so we print the
## banner + header record verbatim and only the KEY ORDER of the atlasParam dict
## (note "script" is last).
println("\n--- Atlas header ---")
open(`gzip -dc $output_file_path`) do io
    lines = collect(Iterators.take(eachline(io), 3))
    println(lines[1])   # format banner
    println(lines[2])   # header record (description, date, types)
    keys3 = [m.captures[1] for m in eachmatch(r"\"([^\"]+)\":", lines[3])]
    # de-dup while preserving order; regex may also match inside the script value
    seen = String[]
    for k in keys3
        k in seen && continue
        push!(seen, k)
        k == "script" && break   # stop once we reach the final embedded-source key
    end
    println("atlasParam keys (in order): ", seen)
end
