# Gelman-Rubin convergence diagnostics for the 16 independent sequential CycleWalk
# chains that build an NC t=1 reference (gamma, iso=0.3). A districting is UNORDERED,
# so we compare the RANK-ORDERED marginals of each saved per-district statistic
# (get_isoperimetric_scores, get_log_spanning_trees -- both length-14 vectors): sort
# each plan's 14 values ascending, rank k = k-th smallest.
#
# For every rank-marginal (14 ranks x 2 stats) we compute, across the 16 chains:
#   * classic split Gelman-Rubin R-hat,
#   * split rank-normalized R-hat (Vehtari et al. 2021) -- bulk + folded tail,
#   * bulk / tail ESS via consistent batch means (O(N), no FFT dependency).
# Plus two plan-level scalars (mean compactness, sum log-spanning-trees).
#
# Emits CSVs consumed by report/baseline_convergence.qmd (CairoMakie).
#
#   julia -t auto --project=examples \
#       examples/validation/baseline_convergence_compute.jl \
#       --gamma-tag gamma0.25 --label g0.25
#
# Run once per gamma benchmark; each writes report/data_baseline/<label>/.

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))

using CycleWalk
using Statistics
using Printf

const AIO = CycleWalk.AtlasIO

# ---- args -----------------------------------------------------------------
argval(flag, default=nothing) = (i = findfirst(==(flag), ARGS);
    i === nothing ? default : ARGS[i+1])

const GAMMA_TAG = argval("--gamma-tag", "gamma0.25")   # substring identifying the benchmark
const PREFIX    = argval("--prefix", "nc_cyclewalk_long")  # atlas base name of the run
const LABEL     = argval("--label", replace(GAMMA_TAG, "gamma" => "g"))
const EXAMPLES  = normpath(joinpath(@__DIR__, ".."))
const OUTDIR    = joinpath(EXAMPLES, "output", "validation")
const DATADIR   = joinpath(@__DIR__, "report", "data_baseline", LABEL)
const STATS     = ["get_isoperimetric_scores", "get_log_spanning_trees"]
const STAT_LABEL = Dict("get_isoperimetric_scores" => "isoperimetric",
                        "get_log_spanning_trees"   => "log_spanning_trees")

# ---- numerics -------------------------------------------------------------
# Acklam inverse normal CDF (no SpecialFunctions dependency).
function norminv(p::Float64)
    a = (-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
          1.383577518672690e2, -3.066479806614716e1, 2.506628277459239e0)
    b = (-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
          6.680131188771972e1, -1.328068155288572e1)
    c = (-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0,
         -2.549732539343734e0, 4.374664141464968e0, 2.938163982698783e0)
    d = (7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0,
         3.754408661907416e0)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow
        q = sqrt(-2log(p))
        return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
               ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p <= phigh
        q = p - 0.5; r = q*q
        return (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q /
               (((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
    else
        q = sqrt(-2log(1-p))
        return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6]) /
                ((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
end

# split each of M chains (rows of M×N) into 2M chains of length N÷2
function splithalf(mat::AbstractMatrix)
    M, N = size(mat); n = N ÷ 2
    vcat(mat[:, 1:n], mat[:, n+1:2n])
end

function split_rhat(mat::AbstractMatrix)
    C = splithalf(mat); m, n = size(C)
    n < 2 && return NaN
    means = vec(mean(C; dims=2)); vars = vec(var(C; dims=2))  # var: ddof=1
    W = mean(vars)
    W <= 0 && return 1.0
    B = n * var(means)
    sqrt(((n-1)/n * W + B/n) / W)
end

# pooled rank-normalization (Blom), preserving M×N shape (column-major safe)
function rank_normalize(mat::AbstractMatrix)
    flat = vec(mat); S = length(flat)
    r = Vector{Float64}(undef, S)
    r[sortperm(flat)] = collect(1.0:S)
    reshape(norminv.((r .- 3/8) ./ (S - 1/4)), size(mat))
end

foldabs(mat) = abs.(mat .- median(vec(mat)))

rhat_ranknorm(mat) = max(split_rhat(rank_normalize(mat)),
                         split_rhat(rank_normalize(foldabs(mat))))

# consistent batch-means ESS for a single chain
function bm_ess(x::AbstractVector)
    n = length(x); n < 8 && return float(n)
    b = max(1, floor(Int, sqrt(n))); a = n ÷ b
    a < 2 && return float(n)
    xt = @view x[1:a*b]
    means = [mean(@view xt[(k-1)*b+1:k*b]) for k in 1:a]
    g = mean(xt)
    sig2 = b/(a-1) * sum((means .- g).^2)     # BM asymptotic variance
    s2 = var(xt)
    (sig2 <= 0 || s2 <= 0) && return float(n)
    n * s2 / sig2
end

# total ESS across independent chains = sum of per-chain ESS (rank-normalized)
function ess_bulk(mat::AbstractMatrix)
    z = rank_normalize(mat)
    sum(bm_ess(view(z, i, :)) for i in 1:size(z, 1))
end
function ess_tail(mat::AbstractMatrix)
    z = rank_normalize(foldabs(mat))
    sum(bm_ess(view(z, i, :)) for i in 1:size(z, 1))
end

function diagnose(mat::AbstractMatrix)
    C = splithalf(mat); n = size(C, 2)
    means = vec(mean(C; dims=2)); vars = vec(var(C; dims=2))
    (mean = mean(mat), B = n * var(means), W = mean(vars),
     rhat_classic = split_rhat(mat), rhat_rn = rhat_ranknorm(mat),
     bulk_ess = ess_bulk(mat), tail_ess = ess_tail(mat))
end

# ---- load chains ----------------------------------------------------------
function chain_files()
    pat = Regex(PREFIX * raw"_thread\d+_")
    fs = filter(readdir(OUTDIR)) do f
        occursin(pat, f) && occursin("_" * GAMMA_TAG * "_", f) &&
            endswith(f, ".jsonl.gz")
    end
    sort!(fs; by = f -> parse(Int, match(r"thread(\d+)", f).captures[1]))
    joinpath.(OUTDIR, fs)
end

# read one atlas -> Dict stat => Matrix [K ranks × N samples] (sorted ascending).
# We only need the two length-14 marginal vectors per map, so stream map-by-map and
# keep just the compact Float64 order-statistics — never hold all Map objects at once.
# Serial per file (no internal threading) so the 16 files can be read in parallel by
# the caller: 16 independent gzip streams overlap decompression + parse across threads.
function read_chain(path)
    io = AIO.smartOpen(path, "r"); atlas = AIO.openAtlas(io)
    acc = Dict(s => Vector{Vector{Float64}}() for s in STATS)
    K = 0
    while !eof(atlas)
        m = AIO.nextMap(atlas)
        for s in STATS
            v = sort!(Float64.(m.data[s]))
            K == 0 && (K = length(v))
            length(v) == K || error("ragged district count in $path")
            push!(acc[s], v)
        end
        # m is dropped next iteration → GC'd; only the compact Float64 vectors persist
    end
    close(io)
    isempty(acc[STATS[1]]) && error("no maps in $path")
    Dict(s => reduce(hcat, acc[s]) for s in STATS)   # K × N
end

# ---- CSV helpers ----------------------------------------------------------
function writecsv(path, header, rows)
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows; println(io, join(r, ",")); end
    end
end

# ---- main -----------------------------------------------------------------
files = chain_files()
isempty(files) && error("no chain files for $GAMMA_TAG in $OUTDIR")
const M = length(files)
@info "reading $M chains for $LABEL ($GAMMA_TAG) on $(Threads.nthreads()) threads"

# Parallel read: 16 independent gzip streams, one thread each (read_chain is serial),
# so decompression + JSON parse overlap across chains — the dominant cost.
chains = Vector{Dict{String,Matrix{Float64}}}(undef, M)
Threads.@threads for i in 1:M
    chains[i] = read_chain(files[i])
end
K = size(chains[1][STATS[1]], 1)
nmin = minimum(size(c[STATS[1]], 2) for c in chains)
@info "M=$M chains, nmin=$nmin samples, K=$K ranks"

# M×nmin matrix of the rank-k order statistic for one stat (rows = chains), on demand.
# Avoids materializing a second full [M,nmin,K] copy of the data.
rankmat(s, k) = Float64[chains[c][s][k, t] for c in 1:M, t in 1:nmin]

mkpath(DATADIR)

# meta
writecsv(joinpath(DATADIR, "meta.csv"), ["key", "value"],
    [["label", LABEL], ["gamma_tag", GAMMA_TAG], ["M", M],
     ["nsamples", nmin], ["K", K], ["total", M*nmin]])

# summary (per stat × rank) — each marginal is independent → compute in parallel
pairs = [(s, k) for s in STATS for k in 1:K]
srows = Vector{Any}(undef, length(pairs))
Threads.@threads for idx in 1:length(pairs)
    s, k = pairs[idx]
    d = diagnose(rankmat(s, k))
    srows[idx] = [STAT_LABEL[s], k, round(d.mean, digits=5), round(d.B, sigdigits=5),
        round(d.W, sigdigits=5), round(d.rhat_classic, digits=5),
        round(d.rhat_rn, digits=5), round(Int, d.bulk_ess), round(Int, d.tail_ess)]
end
writecsv(joinpath(DATADIR, "summary.csv"),
    ["stat","rank","mean","B","W","rhat_classic","rhat_rn","bulk_ess","tail_ess"], srows)

# plan-level scalars (2, cheap → serial)
scal = ["mean_compactness"       => [mean(chains[c]["get_isoperimetric_scores"][:, t]) for c in 1:M, t in 1:nmin],
        "sum_log_spanning_trees" => [sum(chains[c]["get_log_spanning_trees"][:, t])    for c in 1:M, t in 1:nmin]]
scrows = Any[]
for (name, mat) in scal
    d = diagnose(mat)
    push!(scrows, [name, round(d.mean, digits=5), round(d.rhat_classic, digits=5),
        round(d.rhat_rn, digits=5), round(Int, d.bulk_ess), round(Int, d.tail_ess)])
end
writecsv(joinpath(DATADIR, "scalars.csv"),
    ["name","mean","rhat_classic","rhat_rn","bulk_ess","tail_ess"], scrows)

# ECDF overlays for representative ranks (downsample to <=200 pts/chain)
showranks = unique([1, cld(K, 2), K])
erows = Any[]
for s in STATS, k in showranks
    RM = rankmat(s, k)
    for c in 1:M
        x = sort(view(RM, c, :)); n = length(x); npts = min(200, n)
        for j in 1:npts
            idx = clamp(round(Int, (j/npts) * n), 1, n)
            push!(erows, [STAT_LABEL[s], k, c, round(x[idx], sigdigits=6), round(j/npts, digits=5)])
        end
    end
end
writecsv(joinpath(DATADIR, "ecdf.csv"), ["stat","rank","chain","x","cdf"], erows)

# cumulative-mean traces at the least-compact rank K (downsample to <=300 pts/chain)
trows = Any[]
for s in STATS
    RM = rankmat(s, K)
    for c in 1:M
        x = view(RM, c, :); run = cumsum(x) ./ (1:nmin); npts = min(300, nmin)
        for j in 1:npts
            idx = clamp(round(Int, (j/npts) * nmin), 1, nmin)
            push!(trows, [STAT_LABEL[s], c, round(idx/nmin, digits=5), round(run[idx], sigdigits=6)])
        end
    end
end
writecsv(joinpath(DATADIR, "trace.csv"), ["stat","chain","frac","runmean"], trows)

# console summary (reuse the already-computed R̂ in srows)
@info "wrote $DATADIR"
for s in STATS
    sl = STAT_LABEL[s]
    m = maximum(Float64(srows[i][7]) for i in 1:length(srows) if srows[i][1] == sl)
    @printf("  %-22s  max rank-norm R-hat = %.4f  (%s)\n", sl, m,
            m < 1.01 ? "CONVERGED" : "review")
end
