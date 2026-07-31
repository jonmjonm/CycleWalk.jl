# Extend an existing CycleWalk atlas with more samples, restarting the chain from the
# last plan the atlas recorded.
#
#   julia run_cyclewalk_extend.jl output/grid/<atlas>.jsonl.gz --add_cycle_walk_steps 1e5
#   julia run_cyclewalk_extend.jl <atlas>.jsonl.gz -s 1e5 --burn_in 1e4
#   julia run_cyclewalk_extend.jl <atlas>.jsonl.gz -s 1e5 --mode rewrite
#
# The atlas is expected to have been produced by run_cyclewalk_toml.jl, which embeds
# its TOML config in the header under "toml_config"; this script reads that back and
# rebuilds the graph, constraints and measure from it, so the run is defined by the
# atlas alone. Pass --toml to supply a config file instead (only for atlases written
# before config embedding, or to recover from a lost header).
#
# What is and is not carried across the seam:
#   * the districting of the last recorded map is the starting plan;
#   * the spanning forest is NOT stored in an atlas, so it is redrawn with Wilson's
#     algorithm exactly as a fresh run's initial partition is (a Gibbs refresh of the
#     forest coordinate, recorded in the header as "forest_resampled_at_resume");
#   * the RNG is reseeded deterministically from the segment index, so the extension
#     does not replay the parent's random stream;
#   * step numbering continues from the parent's last map, so map names stay monotone
#     across segments and the output grid stays aligned.
#
# --mode selects where the new maps go:
#   segment (default)  a new file "<atlas>_seg<n>.jsonl[.gz]" beside the parent
#   rewrite            a new copy of the whole atlas (parent maps + new ones) that
#                      replaces the parent when complete
#   append             appended in place to the parent file. Fast and keeps one file,
#                      but the parent's header cannot be rewritten, so the lineage is
#                      written to a "<atlas>.lineage.json" sidecar instead.

import Pkg
Pkg.activate(".")
Pkg.instantiate()

using RandomNumbers
using CycleWalk
using AtlasIO
using LinearAlgebra
using JSON

using UnPack, TOML, ArgMacros
include("parameterUtils.jl") # derive_params / print_params / build_measure

args = @dictarguments begin
    @argumentoptional Number add_cycle_walk_steps "--add_cycle_walk_steps" "-s"
    @argumentdefault Number 0 burn_in "--burn_in" "-b"
    @argumentdefault String "segment" mode "--mode" "-m"
    @argumentoptional String toml_override "--toml"
    @argumentoptional String out_override "--out" "-o"
    @argumentflag force "--force"
    @positionalrequired String atlas_file
end

atlas_file = args[:atlas_file]
mode = Symbol(args[:mode])
mode in (:segment, :rewrite, :append) ||
    error("--mode must be one of segment, rewrite, append (got \"$(args[:mode])\")")
isfile(atlas_file) || error("atlas not found: $atlas_file")

# ---------------------------------------------------------------------------------
# Read the parent atlas: its header, its last map, and how many maps it holds.
# ---------------------------------------------------------------------------------
println("scanning $atlas_file ...")
parent_io = smartOpen(atlas_file, "r")
parent_atlas = openAtlas(parent_io)
parent_ap = parent_atlas.atlasParam
last_line = ""
n_parent_maps = 0
while !eof(parent_io)
    line = readline(parent_io)
    isempty(strip(line)) && continue
    global last_line = line
    global n_parent_maps += 1
end
close(parent_io)
n_parent_maps > 0 || error("$atlas_file contains no maps; nothing to resume from")
last_map = parseBufferToMap(parent_atlas, last_line)

# The parent must have recorded districtings, or there is no plan to restart from.
isempty(last_map.districting) &&
    error("$atlas_file was written with output_districting=false, so it records no " *
          "plans and cannot be extended")

# Map names are "step<N>" (see `output` in src/io/writer.jl); the number is the chain
# step the map was written at, and the extension picks up immediately after it.
step_match = match(r"^step(\d+)$", last_map.name)
step_match === nothing &&
    error("cannot read a step number from the last map's name \"$(last_map.name)\"; " *
          "this atlas does not look like a run_cyclewalk_toml.jl run")
resume_step = parse(Int, step_match.captures[1])

# Either an original run or an earlier segment of one may be extended.
parent_script = get(parent_ap, "script_name", "")
if !(parent_script in ("run_cyclewalk_toml.jl", "run_cyclewalk_extend.jl")) &&
   !args[:force]
    error("$atlas_file was written by \"$parent_script\", not run_cyclewalk_toml.jl; " *
          "pass --force to extend it anyway")
end

# This script re-derives the run's settings with today's parameterUtils.jl, not the
# copy the parent actually ran. When the parent embedded its includes, compare them
# against what is on disk now and say so if they have drifted. The recorded sources
# are never executed — an atlas is a file that gets passed around, and running code
# out of one would make opening someone else's atlas a way to run their code.
parent_includes = get(parent_ap, "script_includes", nothing)
if parent_includes !== nothing
    drifted = String[]
    for key in keys(parent_includes)
        rel = String(key)
        current = joinpath(@__DIR__, rel)
        isfile(current) || continue
        read(current, String) == String(parent_includes[key]) || push!(drifted, rel)
    end
    isempty(drifted) ||
        @warn "these files have changed since $(basename(atlas_file)) was written, " *
              "so this extension may not derive its settings the way the parent " *
              "run did" drifted
end

# ---------------------------------------------------------------------------------
# Rebuild the run's parameters from the config the parent run embedded in its header.
# ---------------------------------------------------------------------------------
toml_source = ""
if args[:toml_override] !== nothing
    isfile(args[:toml_override]) || error("config not found: $(args[:toml_override])")
    params = TOML.parsefile(args[:toml_override])
    toml_text = read(args[:toml_override], String)
    toml_source = args[:toml_override]
    println("using config from $toml_source (overriding the atlas header)")
elseif haskey(parent_ap, "toml_config")
    toml_text = String(parent_ap["toml_config"])
    params = TOML.parse(toml_text)
    toml_source = String(get(parent_ap, "toml_config_file", "<embedded in atlas>"))
    println("using the config embedded in the atlas header ($toml_source)")
else
    error("$atlas_file has no \"toml_config\" in its header (it predates config " *
          "embedding); pass --toml <config.toml> to supply one")
end

# The extension runs the same target with the same graph — only the length changes.
add_cycle_walk_steps = args[:add_cycle_walk_steps] !== nothing ?
                       args[:add_cycle_walk_steps] : params["mcmc"]["cycle_walk_steps"]
params["mcmc"]["cycle_walk_steps"] = add_cycle_walk_steps

# The output directory of the parent config is irrelevant here: every mode writes
# beside the parent atlas, so skip creating it.
p = derive_params(params, toml_source; make_output_dir=false)
@unpack gamma, iso_weight, two_cycle_walk_frac, num_dists, pop_dev, node_data,
        pop_col, geo_units, measure_scores, measure_weights, writer_stats, area_col,
        node_border_col, edge_perimeter_col, output_districting, description,
        rng_seed_base, blas_threads, thread_id, steps, outfreq, pctGraphPath,
        ad_param = p

burn_in_cycle_walk_steps = args[:burn_in]
burn_in_steps = Int(ceil(burn_in_cycle_walk_steps/two_cycle_walk_frac))

# An append cannot update the parent's header, so after the first append the header
# no longer describes the file: the sidecar written by the previous append is the
# authority on where the chain had got to.
lineage_sidecar = atlas_file*".lineage.json"
parent_lineage = (mode == :append && isfile(lineage_sidecar)) ?
                 JSON.parsefile(lineage_sidecar) : nothing
parent_lineage === nothing ||
    println("reading the lineage of previous appends from $lineage_sidecar")

# Segment index and the seed derived from it: every segment of a chain draws from a
# different stream, so an extension continues the chain rather than replaying it.
parent_segment = parent_lineage !== nothing ?
                 Int(parent_lineage["segment"]) : Int(get(parent_ap, "segment", 1))
segment = parent_segment + 1
rng_seed = rng_seed_base + 15123*thread_id + 7919*segment

first_step = resume_step + burn_in_steps + 1
final_step = first_step + steps - 1

blas_threads > 0 && BLAS.set_num_threads(blas_threads)

# ---------------------------------------------------------------------------------
# Where the new maps go.
# ---------------------------------------------------------------------------------
"""
    insert_tag(path, tag; strip_seg=false)

Insert `tag` into an atlas file name, before its `.jsonl[.gz]` extension — the
extension has to be preserved verbatim, since `smartOpen` picks its codec from it.
With `strip_seg`, an existing `_seg<n>` tag is dropped first.
"""
function insert_tag(path::AbstractString, tag::AbstractString; strip_seg::Bool=false)
    dir, file = dirname(path), basename(path)
    i = findfirst(".jsonl", file)
    stem, ext = i === nothing ? (file, "") : (file[1:first(i)-1], file[first(i):end])
    strip_seg && (stem = replace(stem, r"_seg\d+$" => ""))
    return joinpath(dir, stem*tag*ext)
end

"""
    segment_path(path, n)

Name of the `n`th segment of the chain `path` belongs to. An existing `_seg<m>` tag is
replaced rather than appended to, so a chain of segments is named `<base>_seg2`,
`<base>_seg3`, … rather than accreting suffixes.
"""
segment_path(path::AbstractString, n::Integer) =
    insert_tag(path, "_seg$(n)"; strip_seg=true)

rewrite_target = nothing   # set in :rewrite mode: the file the finished copy replaces
if args[:out_override] !== nothing
    output_file_path = args[:out_override]
elseif mode == :segment
    output_file_path = segment_path(atlas_file, segment)
elseif mode == :rewrite
    output_file_path = insert_tag(atlas_file, "_extending")
    rewrite_target = atlas_file
else # :append
    output_file_path = atlas_file
end
if mode == :rewrite && args[:out_override] === nothing
    isfile(output_file_path) &&
        error("$output_file_path already exists (a previous rewrite may have died); " *
              "remove it before retrying")
end
if mode != :append && output_file_path != atlas_file
    isfile(output_file_path) && mode == :segment &&
        error("$output_file_path already exists; segment $segment appears to have " *
              "been written already")
end

# ---------------------------------------------------------------------------------
# Provenance: this segment's own record, plus the chain's history back to segment 1.
# ---------------------------------------------------------------------------------
if parent_lineage !== nothing
    # Continuing a chain of appends: the previous append's own record.
    parent_seed = get(parent_lineage, "rng_seed", nothing)
    parent_steps = get(parent_lineage, "appended_cycle_walk_steps", nothing)
    prior_history = Any[h for h in get(parent_lineage, "chain.history", [])]
else
    parent_seed = get(parent_ap, "rng_seed", nothing)
    if parent_seed === nothing
        cp = get(parent_ap, "chain.parameters", nothing)
        parent_seed = cp isa AbstractDict ? get(cp, "seed", nothing) : nothing
    end
    parent_steps = get(parent_ap, "cycle_walk_steps", nothing)
    prior_history = Any[h for h in get(parent_ap, "chain.history", [])]
end

original_atlas = isempty(prior_history) ? basename(atlas_file) :
                 String(prior_history[1]["atlas"])
parent_record = Dict{String, Any}(
    "atlas"               => basename(atlas_file),
    "segment"             => parent_segment,
    "date"                => parent_atlas.date,
    "maps"                => n_parent_maps,
    "final_map"           => last_map.name,
    "final_recorded_step" => resume_step,
    "cycle_walk_steps"    => parent_steps,
    "rng_seed"            => parent_seed,
)
chain_history = vcat(prior_history, Any[parent_record])

cumulative_cycle_walk_steps = add_cycle_walk_steps
for h in chain_history
    s = get(h, "cycle_walk_steps", nothing)
    s isa Number && (global cumulative_cycle_walk_steps += s)
end

merge!(ad_param, Dict{String, Any}(
    "segment"                     => segment,
    "original_atlas"              => original_atlas,
    "extends_atlas"               => basename(atlas_file),
    "extends_atlas_path"          => atlas_file,
    "extends_atlas_date"          => parent_atlas.date,
    "resume_step"                 => resume_step,
    "resume_map"                  => last_map.name,
    "burn_in_cycle_walk_steps"    => burn_in_cycle_walk_steps,
    "burn_in_steps"               => burn_in_steps,
    "burn_in_step_range"          => burn_in_steps > 0 ?
                                     [resume_step+1, resume_step+burn_in_steps] : [],
    "step_range"                  => [first_step, final_step],
    "rng_seed"                    => rng_seed,
    "write_mode"                  => String(mode),
    "forest_resampled_at_resume"  => true,
    "cumulative_cycle_walk_steps" => cumulative_cycle_walk_steps,
    "chain.history"               => chain_history,
))

# ---------------------------------------------------------------------------------
# Rebuild the chain and restart it from the parent's last plan.
# ---------------------------------------------------------------------------------
base_graph = BaseGraph(pctGraphPath, pop_col, inc_node_data=node_data,
                       area_col=area_col, node_border_col=node_border_col,
                       edge_perimeter_col=edge_perimeter_col)
graph = Graph(base_graph, geo_units)

constraints = initialize_constraints()
add_constraint!(constraints, PopulationConstraint(graph, num_dists, pop_dev))

rng = PCG.PCGStateOneseq(UInt64, rng_seed)

# The atlas records exactly the assignment type the partition constructor takes: a
# Dict from each node's one-tuple key to its district (see `get_node_map` in
# src/io/writer.jl and `flatten_assignment` in src/partition/mfr_partition_ext.jl).
seed_assignment = Dict{Tuple{Vararg{String}}, Int}(last_map.districting)
partition = LinkCutPartition(MultiLevelPartition(graph, seed_assignment), rng)
partition.num_dists == num_dists ||
    error("the atlas' last map has $(partition.num_dists) districts but the config " *
          "asks for $num_dists")
# The constructor labels districts by the link-cut tree's scan order, so restore the
# labels the parent recorded: district d in this segment is then the same district d
# the parent wrote out.
relabel_districts!(partition, seed_assignment)
satisfies_constraints(partition, constraints; check_population=true) ||
    error("the atlas' last map does not satisfy the config's constraints; the atlas " *
          "and the config do not describe the same run")

cycle_walk = build_two_tree_cycle_walk(constraints)
internal_walk = build_one_tree_cycle_walk(constraints)
proposal = [(two_cycle_walk_frac, cycle_walk),
            (1.0-two_cycle_walk_frac, internal_walk)]

measure = build_measure(measure_scores, gamma, iso_weight; weights=measure_weights)

# Precompute the run metadata rather than letting run_metropolis_hastings! stamp it,
# so the header is complete before it is written (:rewrite forces it out early, to
# get the parent's maps in behind it).
merge!(ad_param, metropolis_hastings_run_metadata(proposal, (first_step, final_step);
                                                  seed=rng_seed, output_freq=outfreq))

println("\nextending  $atlas_file")
println("  segment            $parent_segment -> $segment")
println("  mode               $mode -> $output_file_path")
println("  resuming from      $(last_map.name) ($n_parent_maps maps in the parent)")
println("  burn in            $burn_in_cycle_walk_steps cycle-walk steps " *
        "($burn_in_steps steps, unwritten)")
println("  adding             $add_cycle_walk_steps cycle-walk steps " *
        "($steps steps), output every $outfreq")
println("  steps              $first_step - $final_step")
println("  rng seed           $rng_seed")
println()

# Burn in before the writer exists, so nothing from the burn-in period is recorded.
if burn_in_steps > 0
    println("burning in $burn_in_steps steps ...")
    run_metropolis_hastings!(partition, proposal, measure,
                             (resume_step+1, resume_step+burn_in_steps), rng)
end

# The embedded config is re-embedded in the new segment. When it came from the parent's
# header rather than from a file on disk, stage it in a temp file for the writer.
config_file = args[:toml_override]
temp_config = nothing
if config_file === nothing
    temp_config = tempname()*".toml"
    write(temp_config, toml_text)
    config_file = temp_config
end

writer = Writer(measure, constraints, partition, output_file_path;
                additional_parameters=ad_param,
                output_districting=output_districting,
                io_mode=(mode == :append ? "a" : "w"),
                write_header=(mode != :append),
                description=description,
                config_file=config_file)
temp_config === nothing || rm(temp_config, force=true)
for stat in writer_stats
    fnct = getfield(CycleWalk, Symbol(stat))
    push_writer!(writer, fnct)
end

# :rewrite copies the parent's maps in behind the freshly written header, so the
# result is one atlas holding every segment under a header that describes them all.
if mode == :rewrite
    write_header!(writer)
    println("copying $n_parent_maps maps from the parent ...")
    copy_io = smartOpen(atlas_file, "r")
    for _ = 1:3            # skip the parent's own header
        readline(copy_io)
    end
    while !eof(copy_io)
        line = readline(copy_io)
        isempty(strip(line)) && continue
        write(writer.atlas.io, line, "\n")
    end
    close(copy_io)
end

run_diagnostics = RunDiagnostics()
if params["run"]["run_diagnostics"] == true
    push_diagnostic!(run_diagnostics, cycle_walk, AcceptanceRatios(),
                     desc = "cycle_walk")
    push_diagnostic!(run_diagnostics, cycle_walk, CycleLengthDiagnostic())
    push_diagnostic!(run_diagnostics, cycle_walk, DeltaNodesDiagnostic())
end

run_metropolis_hastings!(partition, proposal, measure, (first_step, final_step), rng,
                         writer=writer, output_freq=outfreq,
                         run_diagnostics=run_diagnostics, seed=rng_seed);

close_writer(writer)

if mode == :rewrite && rewrite_target !== nothing
    mv(output_file_path, rewrite_target; force=true)
    global output_file_path = rewrite_target
end

# An append cannot rewrite the parent's header, so the lineage goes in a sidecar.
if mode == :append
    sidecar = lineage_sidecar
    lineage = Dict{String, Any}(k => ad_param[k] for k in
        ("segment", "original_atlas", "extends_atlas", "resume_step", "resume_map",
         "burn_in_cycle_walk_steps", "burn_in_steps", "burn_in_step_range",
         "step_range", "rng_seed", "write_mode", "forest_resampled_at_resume",
         "cumulative_cycle_walk_steps", "chain.history"))
    lineage["appended_cycle_walk_steps"] = add_cycle_walk_steps
    open(sidecar, "w") do io
        JSON.print(io, lineage, 2)
    end
    @warn "appended in place: the header of $atlas_file still describes only its " *
          "first segment. The lineage of this extension is recorded in $sidecar."
end

println("CycleWalk extension completed. Output written to $output_file_path")
