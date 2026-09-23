# Cross-copy convergence check for the standard cycle walk at ONE point on the (gamma,iso)
# line. Given C independent copies (same target, different seeds), ask: do the copies agree
# on the RANK-ORDERED marginals of the per-district log spanning-tree count?
#
#   julia analyze_convergence.jl --case grid --tag t0.500 [--bins 40]
#
# Reads every output/validation/<case>_cyclewalk_<tag>_c*.jsonl.gz. For each copy we read
# the per-district get_log_spanning_trees vector of each plan and SORT it ascending, turning
# every plan into order statistics: rank 1 = fewest spanning trees ... rank K = most. For
# each rank we compare the copies:
#   - max pairwise total-variation distance between binned marginals (the disagreement),
#   - a split-half noise floor (max over copies of the TV between the two halves of one copy),
#   - max pairwise Kolmogorov-Smirnov distance, and the spread (min->max) of the copy means.
# Verdict (noise-floor gate, same rule as analyze.jl): CONVERGED iff for EVERY rank the max
# pairwise TV is within 2*noise + 0.02. Emits a machine-readable line the bisection driver
# greps:  VERDICT case=<c> tag=<tag> t=<t> : CONVERGED|NOT

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using CycleWalk
using Printf
using Statistics

const AIO = CycleWalk.AtlasIO

function argval(flag, default=nothing)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    (i == length(ARGS)) && error("missing value after $flag")
    return ARGS[i+1]
end

case = argval("--case")
tag  = argval("--tag")
case === nothing && error("usage: julia analyze_convergence.jl --case {grid,ct,...} --tag <point-tag> [--bins N]")
tag  === nothing && error("--tag is required (the per-point tag, e.g. t0.500)")
nbins = parse(Int, argval("--bins", "40"))
# t is only used for the reported VERDICT line; default parses it out of a "t<...>" tag.
tstr = argval("--t", startswith(tag, "t") ? tag[2:end] : "?")

const EXAMPLES = normpath(joinpath(@__DIR__, ".."))
outdir = joinpath(EXAMPLES, "output", "validation")

# ---------------------------------------------------------------------------
# locate the copies: <case>_cyclewalk_<tag>_c*.jsonl.gz
# ---------------------------------------------------------------------------
prefix = "$(case)_cyclewalk_$(tag)_c"
copy_files = sort(filter(f -> startswith(f, prefix) && endswith(f, ".jsonl.gz"),
                         readdir(outdir)))
isempty(copy_files) && error("no copies found matching $(joinpath(outdir, prefix * "*.jsonl.gz"))")

# ---------------------------------------------------------------------------
# read one copy -> K x N sorted log-spanning-tree matrix (rank k = k-th smallest)
# ---------------------------------------------------------------------------
function read_copy(path)
    io = AIO.smartOpen(path, "r")
    atlas = AIO.openAtlas(io)
    maps = AIO.nextMaps(atlas)
    close(io)
    isempty(maps) && error("no maps in $path")
    K = length(maps[1].data["get_log_spanning_trees"])
    N = length(maps)
    ranks = Matrix{Float64}(undef, K, N)
    for (i, m) in enumerate(maps)
        v = sort!(Float64.(m.data["get_log_spanning_trees"]))
        @assert length(v) == K "ragged district count in $path"
        ranks[:, i] = v
    end
    return ranks
end

# ---------------------------------------------------------------------------
# stats helpers (unweighted: cycle-walk maps all carry weight 1)
# ---------------------------------------------------------------------------
function hist(x, edges)
    nb = length(edges) - 1
    h = zeros(nb)
    for xi in x
        b = clamp(searchsortedlast(edges, xi), 1, nb)
        h[b] += 1
    end
    s = sum(h)
    s == 0 ? h : h ./ s
end
tv(p, q) = 0.5 * sum(abs.(p .- q))

function split_half_tv(x, edges)
    n = length(x)
    n < 4 && return NaN
    h = n ÷ 2
    tv(hist(x[1:h], edges), hist(x[h+1:end], edges))
end

# two-sample Kolmogorov-Smirnov distance: max over x of |F_a(x) - F_b(x)|.
function ks_2samp(a, b)
    sa = sort(a); sb = sort(b)
    na = length(sa); nb = length(sb)
    (na == 0 || nb == 0) && return NaN
    ia = ib = 1; d = 0.0
    # sweep the merged support; at each distinct value advance past all ties in both,
    # THEN compare the (now-updated) empirical CDFs F_a = (ia-1)/na, F_b = (ib-1)/nb.
    while ia <= na || ib <= nb
        x = min(ia <= na ? sa[ia] : Inf, ib <= nb ? sb[ib] : Inf)
        while ia <= na && sa[ia] <= x; ia += 1; end
        while ib <= nb && sb[ib] <= x; ib += 1; end
        d = max(d, abs((ia - 1) / na - (ib - 1) / nb))
    end
    return d
end

# ---------------------------------------------------------------------------
# read all copies
# ---------------------------------------------------------------------------
copies = [read_copy(joinpath(outdir, f)) for f in copy_files]
K = size(copies[1], 1)
for (f, c) in zip(copy_files, copies)
    @assert size(c, 1) == K "district count differs in $f ($(size(c,1)) vs $K)"
end
C = length(copies)

println("="^100)
println("convergence check  case=$case  tag=$tag  bins=$nbins")
println("copies ($C):")
for (f, c) in zip(copy_files, copies)
    @printf("  %-48s  %d samples\n", f, size(c, 2))
end
println("rank 1 = fewest spanning trees ... rank K = most   (per-district log spanning-tree count)")
println("="^100)
@printf("%-5s | %-11s | %-8s | %-8s | %-8s | %-8s | %s\n",
        "rank", "mean spread", "maxTV", "2*noise", "maxKS", "meanΔ", "verdict")
println("-"^100)

function compare_copies(copies, C, K, nbins)
    converged = true
    for k in 1:K
        xs = [c[k, :] for c in copies]                # C vectors of this rank's values
        lo = minimum(minimum.(xs)); hi = maximum(maximum.(xs))
        hi <= lo && (hi = lo + 1.0)
        edges = range(lo, hi; length = nbins + 1)

        hs = [hist(x, edges) for x in xs]
        maxtv = 0.0
        maxks = 0.0
        for i in 1:C, j in (i+1):C
            maxtv = max(maxtv, tv(hs[i], hs[j]))
            maxks = max(maxks, ks_2samp(xs[i], xs[j]))
        end
        noise = maximum(split_half_tv(x, edges) for x in xs)
        means = [mean(x) for x in xs]
        mean_lo, mean_hi = minimum(means), maximum(means)

        ok = isnan(noise) ? true : maxtv <= 2 * noise + 0.02
        converged &= ok
        @printf("%-5d | %5.2f→%5.2f | %-8.4f | %-8.4f | %-8.4f | %-8.4f | %s\n",
                k, mean_lo, mean_hi, maxtv, 2 * noise, maxks, mean_hi - mean_lo,
                ok ? "ok" : "**DIFFERS**")
    end
    return converged
end
converged = compare_copies(copies, C, K, nbins)
println("-"^100)
verdict = converged ? "CONVERGED" : "NOT"
println(converged ?
        "all ranks' max pairwise TV within the split-half noise floor" :
        "one or more ranks exceed the noise floor -- copies disagree")
# machine-readable line for the bisection driver
@printf("VERDICT case=%s tag=%s t=%s : %s\n", case, tag, tstr, verdict)
