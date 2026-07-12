# Multi-chain convergence at ONE point on the (gamma,iso) line via rank-normalized split-R-hat
# (Gelman-Rubin, Vehtari et al. 2021). Given C independent cycle-walk copies (chains) that
# targeted the SAME measure, ask: have they converged to the same distribution?
#
#   julia analyze_rhat.jl --case nc --tag t0.500000 [--t 0.5] [--rhat 1.01]
#
# For each district RANK (the sorted per-district log spanning-tree count) we treat the
# C chains' values as C MCMC chains and compute rank-normalized split-R-hat plus the folded
# (tail) version; R-hat = max(bulk, tail). Verdict CONVERGED iff max over ranks R-hat < thresh.
# Also reports bulk-ESS. Emits a machine-readable line the driver greps:
#   VERDICT case=<c> tag=<tag> t=<t> rhat=<maxRhat> : CONVERGED|NOT
#
# Dependency-free: inverse-normal-CDF via Acklam's rational approximation; no Distributions.

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using CycleWalk
using Printf, Statistics
const AIO = CycleWalk.AtlasIO

argval(flag, default=nothing) = (i = findfirst(==(flag), ARGS);
    i === nothing ? default : ARGS[i+1])

case = argval("--case"); case === nothing && error("--case required")
tag  = argval("--tag");  tag  === nothing && error("--tag required (per-point tag, e.g. t0.500000)")
thresh = parse(Float64, argval("--rhat", "1.01"))
tstr = argval("--t", startswith(tag, "t") ? tag[2:end] : "?")

const EXAMPLES = normpath(joinpath(@__DIR__, ".."))
outdir = joinpath(EXAMPLES, "output", "validation")

# --- inverse normal CDF (Acklam), dependency-free ---------------------------
function norminv(p::Float64)
    (p <= 0) && return -Inf
    (p >= 1) && return Inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow = 0.02425; phigh = 1 - plow
    if p < plow
        q = sqrt(-2 * log(p))
        return (((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
               ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    elseif p <= phigh
        q = p - 0.5; r = q * q
        return (((((a[1]*r + a[2])*r + a[3])*r + a[4])*r + a[5])*r + a[6]) * q /
               (((((b[1]*r + b[2])*r + b[3])*r + b[4])*r + b[5])*r + 1)
    else
        q = sqrt(-2 * log(1 - p))
        return -(((((c[1]*q + c[2])*q + c[3])*q + c[4])*q + c[5])*q + c[6]) /
                ((((d[1]*q + d[2])*q + d[3])*q + d[4])*q + 1)
    end
end

# --- average (fractional) ranks with tie handling ---------------------------
function tiedrank(x::Vector{Float64})
    n = length(x); p = sortperm(x); r = Vector{Float64}(undef, n)
    i = 1
    while i <= n
        j = i
        while j < n && x[p[j+1]] == x[p[i]]; j += 1; end
        avg = (i + j) / 2
        for k in i:j; r[p[k]] = avg; end
        i = j + 1
    end
    return r
end

# rank-normalize a pooled vector -> z-scores (Blom transform)
function rank_normalize(pooled::Vector{Float64})
    S = length(pooled)
    r = tiedrank(pooled)
    return [norminv((r[i] - 3/8) / (S - 1/4 + 0)) for i in 1:S]
end

# R-hat and bulk-ESS from a matrix of split sub-chains (n draws x M chains)
function rhat_ess(mat::Matrix{Float64})
    n, M = size(mat)
    means = vec(mean(mat; dims=1))
    vars  = vec(var(mat; dims=1))            # sample var, /(n-1)
    W = mean(vars)
    B = n * var(means)                       # var over M chains, /(M-1)
    W <= 0 && return (1.0, Float64(n * M))
    var_plus = (n - 1) / n * W + B / n
    rhat = sqrt(var_plus / W)
    # crude bulk-ESS (no autocov term across split halves; lower bound-ish)
    ess = M * n * var_plus / B
    return (rhat, isfinite(ess) ? ess : Float64(n * M))
end

# split each chain in half -> 2C sub-chains, rank-normalize pooled, reshape, R-hat
function split_rhat(chains::Vector{Vector{Float64}}; fold::Bool=false)
    C = length(chains)
    n = minimum(length.(chains)) ÷ 2
    n < 2 && return (NaN, NaN)
    subs = Vector{Vector{Float64}}()
    for ch in chains
        push!(subs, ch[1:n]); push!(subs, ch[n+1:2n])
    end
    pooled = reduce(vcat, subs)
    if fold
        med = median(pooled)
        pooled = abs.(pooled .- med)
    end
    z = rank_normalize(pooled)
    M = 2C
    mat = Matrix{Float64}(undef, n, M)
    for m in 1:M; mat[:, m] = z[(m-1)*n+1 : m*n]; end
    return rhat_ess(mat)
end

# --- read a copy -> K x N sorted log-spanning-tree matrix --------------------
function read_copy(path)
    io = AIO.smartOpen(path, "r"); atlas = AIO.openAtlas(io)
    maps = AIO.nextMaps(atlas); close(io)
    isempty(maps) && error("no maps in $path")
    K = length(maps[1].data["get_log_spanning_trees"]); N = length(maps)
    ranks = Matrix{Float64}(undef, K, N)
    for (i, m) in enumerate(maps)
        ranks[:, i] = sort!(Float64.(m.data["get_log_spanning_trees"]))
    end
    return ranks
end

prefix = "$(case)_cyclewalk_$(tag)_c"
files = sort(filter(f -> startswith(f, prefix) && endswith(f, ".jsonl.gz"), readdir(outdir)))
isempty(files) && error("no copies matching $(joinpath(outdir, prefix * "*.jsonl.gz"))")
copies = [read_copy(joinpath(outdir, f)) for f in files]
K = size(copies[1], 1)
for (f, c) in zip(files, copies)
    @assert size(c, 1) == K "district count differs in $f"
end
C = length(copies)

println("="^92)
println("split-R̂ convergence  case=$case  tag=$tag  chains=$C  threshold R̂<$thresh")
for (f, c) in zip(files, copies)
    @printf("  %-46s  %d samples\n", f, size(c, 2))
end
println("rank 1 = fewest spanning trees ... rank K = most")
println("="^92)
@printf("%-5s | %-10s | %-10s | %-10s | %-12s | %s\n",
        "rank", "bulk-R̂", "tail-R̂", "R̂", "bulk-ESS", "verdict")
println("-"^92)

function report(copies, C, K, thresh)
    maxr = 0.0
    for k in 1:K
        chains = [copies[c][k, :] for c in 1:C]
        br, ess = split_rhat(chains; fold=false)
        tr, _   = split_rhat(chains; fold=true)
        r = max(br, tr)
        maxr = max(maxr, r)
        ok = r < thresh
        @printf("%-5d | %-10.4f | %-10.4f | %-10.4f | %-12.0f | %s\n",
                k, br, tr, r, ess, ok ? "ok" : "**HIGH**")
    end
    return maxr
end
maxr = report(copies, C, K, thresh)
println("-"^92)
verdict = maxr < thresh ? "CONVERGED" : "NOT"
println(maxr < thresh ?
        @sprintf("max R̂ = %.4f < %.2f  → chains have converged", maxr, thresh) :
        @sprintf("max R̂ = %.4f ≥ %.2f  → chains have NOT converged", maxr, thresh))
@printf("VERDICT case=%s tag=%s t=%s rhat=%.4f : %s\n", case, tag, tstr, maxr, verdict)
