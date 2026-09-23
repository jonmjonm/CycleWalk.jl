# Weight diagnostics for the AIS runs. Reads the log importance weight of every map in
# each AIS Atlas (weight field only -- the districting is skipped, so this is fast even
# on the 37 MB CT files) and reports, per run:
#
#   * log-weight statistics (mean, sd, min, max, range in nats)
#   * effective sample size ESS = (Σw)²/Σw² and ESS fraction
#   * concentration: largest single normalized weight, and how few samples hold
#     50% / 90% of the total weight
#   * an ASCII histogram of the log weights
#
# and a one-line-per-run summary table. Usage:
#
#   julia -t 10 validation/analyze_weights.jl                # default run set
#   julia -t 10 validation/analyze_weights.jl a.jsonl.gz b…  # explicit files (label=basename)

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using CycleWalk
using Printf
using Statistics

const AIO = CycleWalk.AtlasIO
const OUTDIR = normpath(joinpath(@__DIR__, "..", "output", "validation"))

# read ONLY the per-map log weight (regex on each map line; no districting parse)
function read_log_weights(path)
    io = AIO.smartOpen(path, "r")
    AIO.openAtlas(io)                # consumes the 3 header lines, positions at map 1
    ws = Float64[]
    rx = r"\"weight\":\s*(-?[0-9.eE+]+)"
    while !eof(io)
        line = readline(io)
        m = match(rx, line)
        m === nothing && continue
        push!(ws, parse(Float64, m.captures[1]))
    end
    close(io)
    return ws
end

# how many of the largest normalized weights are needed to reach fraction `frac`
function n_for_mass(Wsorted_desc, frac)
    c = 0.0
    for (i, w) in enumerate(Wsorted_desc)
        c += w
        c >= frac && return i
    end
    return length(Wsorted_desc)
end

struct WeightStats
    label::String; n::Int
    lmean::Float64; lsd::Float64; lmin::Float64; lmax::Float64
    ess::Float64; wmax::Float64; n50::Int; n90::Int
    lw::Vector{Float64}
end

function weight_stats(label, lw)
    n = length(lw)
    W = exp.(lw .- maximum(lw)); W ./= sum(W)         # normalized importance weights
    ess = 1 / sum(abs2, W)
    Wd = sort(W; rev = true)
    return WeightStats(label, n, mean(lw), std(lw), minimum(lw), maximum(lw),
                       ess, Wd[1], n_for_mass(Wd, 0.5), n_for_mass(Wd, 0.9), lw)
end

function ascii_hist(lw; nbins = 30, width = 56)
    lo, hi = minimum(lw), maximum(lw)
    hi <= lo && (hi = lo + 1)
    edges = range(lo, hi; length = nbins + 1)
    counts = zeros(Int, nbins)
    for x in lw
        b = clamp(searchsortedlast(edges, x), 1, nbins)
        counts[b] += 1
    end
    cmax = maximum(counts)
    for i in 1:nbins
        c = edges[i]
        bar = repeat("█", round(Int, width * counts[i] / max(cmax, 1)))
        @printf("  %8.2f | %-56s %d\n", c, bar, counts[i])
    end
end

# ---------------------------------------------------------------------------
default_runs = [
    ("small @4k",  "small_ais_a4000.jsonl.gz"),
    ("grid @4k",   "grid_ais_a4000.jsonl.gz"),
    ("grid @16k",  "grid_ais_a16000.jsonl.gz"),
    ("ct @4k",     "ct_ais_a4000.jsonl.gz"),
    ("ct @16k",    "ct_ais_a16000.jsonl.gz"),
]
runs = isempty(ARGS) ? [(basename(p), joinpath(OUTDIR, p)) for (_, p) in default_runs] :
                       [(basename(a), a) for a in ARGS]

allstats = WeightStats[]
for (label, path) in runs
    if !isfile(path)
        println("(skip $label: $path not found)")
        continue
    end
    push!(allstats, weight_stats(label, read_log_weights(path)))
end

println("\n" * "="^92)
println("AIS importance-weight diagnostics")
println("="^92)
@printf("%-11s | %6s | %8s %7s | %7s %7s %7s | %7s %6s | %5s %5s\n",
        "run", "n", "logμ", "logσ", "logmin", "logmax", "range",
        "ESS", "ESS%", "n50", "n90")
println("-"^92)
for s in allstats
    @printf("%-11s | %6d | %8.2f %7.3f | %7.2f %7.2f %7.2f | %7.1f %5.1f%% | %5d %5d\n",
            s.label, s.n, s.lmean, s.lsd, s.lmin, s.lmax, s.lmax - s.lmin,
            s.ess, 100 * s.ess / s.n, s.n50, s.n90)
end
println("-"^92)
println("range = spread of log weights in nats (larger => more variable weights)")
println("ESS%  = ESS / n  (higher is better; low means a few samples dominate)")
println("n50/n90 = number of largest-weight samples holding 50% / 90% of total weight")

for s in allstats
    println("\n" * "-"^92)
    @printf("%s   (n=%d, ESS=%.1f = %.1f%%, log-weight range=%.2f nats)\n",
            s.label, s.n, s.ess, 100 * s.ess / s.n, s.lmax - s.lmin)
    println("-"^92)
    ascii_hist(s.lw)
end
