# Dump per-rank, per-copy marginal histograms of the per-district log spanning-tree count
# to a compact CSV, for plotting the cross-copy convergence overlay.
#
#   julia dump_marginals.jl --case grid --tags t0.000000,t1.000000 [--bins 50] > marginals.csv
#
# CSV columns: point_tag,rank,copy,bin_center,density   (density = normalized count).
# Bin edges are SHARED across the copies of a given (point,rank) so the overlaid curves
# are directly comparable.

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using CycleWalk
const AIO = CycleWalk.AtlasIO

argval(flag, default=nothing) = (i = findfirst(==(flag), ARGS);
    i === nothing ? default : ARGS[i+1])

case = argval("--case"); case === nothing && error("--case required")
tags = split(argval("--tags", ""), ","; keepempty=false)
isempty(tags) && error("--tags required (comma-separated point tags)")
nbins = parse(Int, argval("--bins", "50"))

const EXAMPLES = normpath(joinpath(@__DIR__, ".."))
outdir = joinpath(EXAMPLES, "output", "validation")

function read_copy(path)
    io = AIO.smartOpen(path, "r"); atlas = AIO.openAtlas(io)
    maps = AIO.nextMaps(atlas); close(io)
    K = length(maps[1].data["get_log_spanning_trees"]); N = length(maps)
    ranks = Matrix{Float64}(undef, K, N)
    for (i, m) in enumerate(maps)
        ranks[:, i] = sort!(Float64.(m.data["get_log_spanning_trees"]))
    end
    return ranks
end

println("point_tag,rank,copy,bin_center,density")
for tag in tags
    prefix = "$(case)_cyclewalk_$(tag)_c"
    files = sort(filter(f -> startswith(f, prefix) && endswith(f, ".jsonl.gz"), readdir(outdir)))
    copies = [read_copy(joinpath(outdir, f)) for f in files]
    K = size(copies[1], 1)
    for k in 1:K
        xs = [c[k, :] for c in copies]
        lo = minimum(minimum.(xs)); hi = maximum(maximum.(xs)); hi <= lo && (hi = lo + 1.0)
        edges = range(lo, hi; length = nbins + 1)
        centers = [(edges[b] + edges[b+1]) / 2 for b in 1:nbins]
        for (ci, x) in enumerate(xs)
            h = zeros(nbins)
            for xi in x
                b = clamp(searchsortedlast(edges, xi), 1, nbins); h[b] += 1
            end
            s = sum(h); s > 0 && (h ./= s)
            for b in 1:nbins
                println("$tag,$k,$ci,$(round(centers[b], digits=5)),$(round(h[b], digits=6))")
            end
        end
    end
end
