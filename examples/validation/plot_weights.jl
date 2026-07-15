# Generate self-contained SVG figures of the AIS importance weights (no plotting deps).
# Writes three SVGs into output/validation/ that RESULTS.md embeds:
#   weights_hist.svg    overlaid density of (log-weight - mean) for all runs
#   weights_lorenz.svg  weight-concentration curves (cumulative weight vs sample frac)
#   weights_ess.svg     ESS% vs log-weight sigma with the exp(-sigma^2) prediction
#
#   julia validation/plot_weights.jl

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using CycleWalk, Statistics, Printf
const AIO = CycleWalk.AtlasIO
const OUTDIR = normpath(joinpath(@__DIR__, "..", "output", "validation"))

function read_log_weights(path)
    io = AIO.smartOpen(path, "r"); AIO.openAtlas(io)
    ws = Float64[]; rx = r"\"weight\":\s*(-?[0-9.eE+]+)"
    while !eof(io)
        m = match(rx, readline(io)); m !== nothing && push!(ws, parse(Float64, m.captures[1]))
    end
    close(io); return ws
end

runs = [  # label, file, color
    ("small @4k",  "small_ais_a4000.jsonl.gz",  "#2563eb"),
    ("grid @4k",   "grid_ais_a4000.jsonl.gz",   "#f59e0b"),
    ("grid @16k",  "grid_ais_a16000.jsonl.gz",  "#16a34a"),
    ("ct @4k",     "ct_ais_a4000.jsonl.gz",     "#dc2626"),
    ("ct @16k",    "ct_ais_a16000.jsonl.gz",    "#9333ea"),
]
data = [(lbl, col, read_log_weights(joinpath(OUTDIR, f))) for (lbl, f, col) in runs
        if isfile(joinpath(OUTDIR, f))]

# --- tiny SVG scaffolding -------------------------------------------------
const AX = "#888"          # axis / text: legible on light and dark backgrounds
esc(s) = replace(string(s), "&"=>"&amp;", "<"=>"&lt;", ">"=>"&gt;")
poly(pts, col; w=2.0, fill="none") =
    "<polyline fill=\"$fill\" stroke=\"$col\" stroke-width=\"$w\" points=\"" *
    join(("$(round(x,digits=2)),$(round(y,digits=2))" for (x,y) in pts), " ") * "\"/>"
line(x1,y1,x2,y2; col=AX, w=1.0) =
    "<line x1=\"$x1\" y1=\"$y1\" x2=\"$x2\" y2=\"$y2\" stroke=\"$col\" stroke-width=\"$w\"/>"
text(x,y,s; col=AX, size=13, anchor="middle") =
    "<text x=\"$x\" y=\"$y\" fill=\"$col\" font-size=\"$size\" font-family=\"sans-serif\" text-anchor=\"$anchor\">$(esc(s))</text>"

function svg_open(w, h; title="")
    io = IOBuffer()
    print(io, "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 $w $h\" width=\"$w\" height=\"$h\" font-family=\"sans-serif\">")
    isempty(title) || print(io, text(w/2, 22, title; col=AX, size=15))
    return io
end
svg_close(io, path) = (print(io, "</svg>"); write(path, String(take!(io))); println("wrote ", path))

function legend(io, entries, x, y; dy=20)
    for (i, (lbl, col)) in enumerate(entries)
        yy = y + (i-1)*dy
        print(io, line(x, yy, x+22, yy; col=col, w=3))
        print(io, text(x+28, yy+4, lbl; col=AX, size=12, anchor="start"))
    end
end

# --- Figure 1: overlaid density of (log-weight - mean) --------------------
function fig_hist(data)
    W, H = 760, 440; L, R, T, B = 60, 20, 44, 56
    xlo, xhi = -16.0, 16.0; nb = 96
    edges = range(xlo, xhi; length=nb+1); bw = step(edges)
    sx(x) = L + (x - xlo)/(xhi - xlo)*(W-L-R)
    dens = Tuple{String,String,Vector{Float64}}[]
    ymax = 0.0
    for (lbl, col, lw) in data
        c = lw .- mean(lw); h = zeros(nb)
        for v in c
            b = clamp(searchsortedlast(edges, v), 1, nb); h[b] += 1
        end
        h ./= (length(lw)*bw); ymax = max(ymax, maximum(h))
        push!(dens, (lbl, col, h))
    end
    ymax *= 1.08
    sy(y) = (H-B) - y/ymax*(H-B-T)
    io = svg_open(W, H; title="AIS log-weight distributions, centered (density vs log-weight − mean)")
    print(io, line(L, H-B, W-R, H-B))                       # x axis
    for xt in -15:5:15
        print(io, line(sx(xt), H-B, sx(xt), H-B+4))
        print(io, text(sx(xt), H-B+18, string(xt)))
    end
    print(io, text((L+W-R)/2, H-14, "log-weight − mean  (nats)"))
    for (lbl, col, h) in dens                               # step outlines
        pts = Tuple{Float64,Float64}[]
        for i in 1:nb
            push!(pts, (sx(edges[i]), sy(h[i]))); push!(pts, (sx(edges[i+1]), sy(h[i])))
        end
        print(io, poly(pts, col; w=1.8))
    end
    legend(io, [(l,c) for (l,c,_) in dens], W-R-150, T+6)
    svg_close(io, joinpath(OUTDIR, "weights_hist.svg"))
end

# --- Figure 2: weight-concentration (Lorenz-style) curves -----------------
function fig_lorenz(data)
    W, H = 620, 460; L, R, T, B = 60, 20, 44, 56
    sx(x) = L + x*(W-L-R); sy(y) = (H-B) - y*(H-B-T)
    io = svg_open(W, H; title="Weight concentration: cumulative weight vs sample fraction")
    print(io, line(L, H-B, W-R, H-B)); print(io, line(L, T, L, H-B))
    for t in 0:0.25:1
        print(io, line(sx(t), H-B, sx(t), H-B+4)); print(io, text(sx(t), H-B+18, string(t)))
        print(io, line(L-4, sy(t), L, sy(t))); print(io, text(L-8, sy(t)+4, string(t); anchor="end"))
    end
    print(io, poly([(sx(0),sy(0)),(sx(1),sy(1))], AX; w=1.0))   # uniform reference
    print(io, text(sx(0.62), sy(0.30), "uniform weights"; col=AX, size=11))
    for (lbl, col, lw) in data
        w = exp.(lw .- maximum(lw)); w ./= sum(w); sort!(w; rev=true)
        cum = cumsum(w); n = length(cum)
        pts = [(sx(0.0), sy(0.0))]
        step_i = max(1, n ÷ 200)
        for i in 1:step_i:n; push!(pts, (sx(i/n), sy(cum[i]))); end
        push!(pts, (sx(1.0), sy(1.0)))
        print(io, poly(pts, col; w=2.2))
    end
    print(io, text((L+W-R)/2, H-14, "fraction of samples (largest weight first)"))
    print(io, "<text x=\"18\" y=\"$((T+H-B)/2)\" fill=\"$AX\" font-size=\"12\" font-family=\"sans-serif\" text-anchor=\"middle\" transform=\"rotate(-90 18 $((T+H-B)/2))\">cumulative fraction of total weight</text>")
    legend(io, [(l,c) for (l,c,_) in data], L+18, T+10)
    svg_close(io, joinpath(OUTDIR, "weights_lorenz.svg"))
end

# --- Figure 3: ESS% vs sigma with exp(-sigma^2) prediction ----------------
function fig_ess(data)
    W, H = 620, 440; L, R, T, B = 56, 16, 44, 56
    xhi = 4.2
    sx(x) = L + x/xhi*(W-L-R); sy(y) = (H-B) - y/100*(H-B-T)
    io = svg_open(W, H; title="ESS fraction vs log-weight σ  (points) and exp(−σ²) law (curve)")
    print(io, line(L, H-B, W-R, H-B)); print(io, line(L, T, L, H-B))
    for xt in 0:1:4
        print(io, line(sx(xt), H-B, sx(xt), H-B+4)); print(io, text(sx(xt), H-B+18, string(xt)))
    end
    for yt in 0:25:100
        print(io, line(L-4, sy(yt), L, sy(yt))); print(io, text(L-8, sy(yt)+4, string(yt); anchor="end"))
    end
    pts = [(sx(s), sy(100*exp(-s^2))) for s in range(0, xhi; length=120)]  # theory curve
    print(io, poly(pts, AX; w=1.6))
    for (lbl, col, lw) in data
        s = std(lw); w = exp.(lw .- maximum(lw)); w ./= sum(w); ess = 100/(sum(abs2, w)*length(w))
        cx, cy = sx(s), sy(ess)
        print(io, "<circle cx=\"$(round(cx,digits=1))\" cy=\"$(round(cy,digits=1))\" r=\"5\" fill=\"$col\"/>")
        print(io, text(cx, cy-10, lbl; col=col, size=11))
    end
    print(io, text((L+W-R)/2, H-14, "log-weight standard deviation σ  (nats)"))
    print(io, "<text x=\"16\" y=\"$((T+H-B)/2)\" fill=\"$AX\" font-size=\"12\" font-family=\"sans-serif\" text-anchor=\"middle\" transform=\"rotate(-90 16 $((T+H-B)/2))\">ESS as % of n</text>")
    svg_close(io, joinpath(OUTDIR, "weights_ess.svg"))
end

fig_hist(data); fig_lorenz(data); fig_ess(data)
println("done")
