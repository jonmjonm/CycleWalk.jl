# Analyze the per-sample annealing PATH recorded by --record-path: where along the
# schedule does the importance-weight variance get injected? Reads one or more
# path-recorded AIS Atlases and, at each schedule position, computes across samples the
# mean and variance of the running log-weight (variance at the end sets the ESS), plus
# the variance-accumulation curve. Writes SVG+PNG figures and a short text summary.
#
#   julia validation/analyze_path.jl grid_ais_path_a4000.jsonl.gz grid_ais_path_a16000.jsonl.gz
# (bare filenames are resolved under output/validation/)

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using CycleWalk, Statistics, Printf
const AIO = CycleWalk.AtlasIO
const OUTDIR = normpath(joinpath(@__DIR__, "..", "output", "validation"))

# --case grid|ct picks the default path-recorded files and output naming; explicit
# filenames may still be passed positionally.
argval(flag, d=nothing) = (i = findfirst(==(flag), ARGS);
    i === nothing ? d : ARGS[i+1])
case = argval("--case", "grid")
steps_of(f) = (m = match(r"_a(\d+)\.jsonl", basename(f)); m === nothing ? 0 : parse(Int, m.captures[1]))
posfiles = filter(a -> endswith(a, ".jsonl.gz"), ARGS)
# default: every recorded anneal length for this case, ordered by step count
files = isempty(posfiles) ?
    sort(filter(f -> occursin(Regex("^$(case)_ais_path_a\\d+"), f), readdir(OUTDIR)); by=steps_of) :
    posfiles
isempty(files) && error("no path-recorded atlases for case=$case in $OUTDIR " *
                        "(run run_case.jl --record-path first)")
function pathlabel(f)
    b = basename(f); s = steps_of(f)
    c = occursin("ct_", b) ? "ct" : occursin("nc_", b) ? "nc" :
        occursin("grid_", b) ? "grid" : occursin("small_", b) ? "small" : case
    strip("$c " * (s == 0 ? "" : "@$(s ÷ 1000)k"))
end
labels = pathlabel.(files)
colors = ["#f59e0b", "#16a34a", "#2563eb", "#dc2626", "#9333ea", "#0891b2"]

# read path arrays -> (x schedule frac vector, matrix[npoints x nsamples] of running lw,
#                      matrix of iso-along-path, matrix of logspanningforest-along-path)
function read_path(path)
    isfile(path) || error("missing $path")
    io = AIO.smartOpen(path, "r"); atlas = AIO.openAtlas(io)
    maps = AIO.nextMaps(atlas); close(io)
    haskey(maps[1].data, "path/log_weight") ||
        error("$path has no path data -- run run_case.jl with --record-path")
    npts = minimum(length(m.data["path/log_weight"]) for m in maps)
    n = length(maps)
    lw  = Matrix{Float64}(undef, npts, n)
    iso = Matrix{Float64}(undef, npts, n)
    lsf = Matrix{Float64}(undef, npts, n)
    for (j, m) in enumerate(maps)
        lw[:, j]  = Float64.(m.data["path/log_weight"])[1:npts]
        iso[:, j] = Float64.(m.data["path/get_isoperimetric_score"])[1:npts]
        lsf[:, j] = Float64.(m.data["path/get_log_spanning_forests"])[1:npts]
    end
    x = Float64.(maps[1].data["path/schedule_frac"])[1:npts]
    return x, lw, iso, lsf
end

runs = [(labels[i], colors[i], read_path(joinpath(OUTDIR, f))...) for (i, f) in enumerate(files)]

# --- tiny SVG scaffolding (shared style with plot_weights.jl) --------------
const AX = "#888"
esc(s) = replace(string(s), "&"=>"&amp;", "<"=>"&lt;", ">"=>"&gt;")
poly(pts, col; w=2.0, dash="") =
    "<polyline fill=\"none\" stroke=\"$col\" stroke-width=\"$w\"" *
    (isempty(dash) ? "" : " stroke-dasharray=\"$dash\"") * " points=\"" *
    join(("$(round(x,digits=2)),$(round(y,digits=2))" for (x,y) in pts), " ") * "\"/>"
line(x1,y1,x2,y2;col=AX,w=1.0)="<line x1=\"$x1\" y1=\"$y1\" x2=\"$x2\" y2=\"$y2\" stroke=\"$col\" stroke-width=\"$w\"/>"
txt(x,y,s;col=AX,size=13,anchor="middle")="<text x=\"$x\" y=\"$y\" fill=\"$col\" font-size=\"$size\" font-family=\"sans-serif\" text-anchor=\"$anchor\">$(esc(s))</text>"
function svg_open(w,h;title="")
    io=IOBuffer(); print(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 $w $h\" width=\"$w\" height=\"$h\" font-family=\"sans-serif\">")
    isempty(title)||print(io,txt(w/2,22,title;size=15)); io
end
svg_close(io,path)=(print(io,"</svg>"); write(path,String(take!(io))); println("wrote ",path))
function legend(io,entries,x,y;dy=20)
    for (i,(l,c)) in enumerate(entries)
        print(io,line(x,y+(i-1)*dy,x+22,y+(i-1)*dy;col=c,w=3)); print(io,txt(x+28,y+(i-1)*dy+4,l;size=12,anchor="start"))
    end
end
function axes(io,W,H,L,R,T,B,xlo,xhi,ylo,yhi,xlabel,ylabel;xr16=16)
    sx(x)=L+(x-xlo)/(xhi-xlo)*(W-L-R); sy(y)=(H-B)-(y-ylo)/(yhi-ylo)*(H-B-T)
    print(io,line(L,H-B,W-R,H-B)); print(io,line(L,T,L,H-B))
    for k in 0:4
        xt=xlo+(xhi-xlo)*k/4; print(io,line(sx(xt),H-B,sx(xt),H-B+4)); print(io,txt(sx(xt),H-B+18,string(round(xt,digits=2))))
        yt=ylo+(yhi-ylo)*k/4; print(io,line(L-4,sy(yt),L,sy(yt))); print(io,txt(L-8,sy(yt)+4,string(round(yt,digits=2));anchor="end"))
    end
    print(io,txt((L+W-R)/2,H-14,xlabel))
    print(io,"<text x=\"16\" y=\"$((T+H-B)/2)\" fill=\"$AX\" font-size=\"12\" font-family=\"sans-serif\" text-anchor=\"middle\" transform=\"rotate(-90 16 $((T+H-B)/2))\">$(esc(ylabel))</text>")
    return sx,sy
end

# per-run stats along the path
stats = map(runs) do (lbl, col, x, lw, iso, lsf)
    meanlw = vec(mean(lw; dims=2)); varlw = vec(var(lw; dims=2))
    (lbl=lbl, col=col, x=x, meanlw=meanlw, varlw=varlw,
     meaniso=vec(mean(iso; dims=2)))
end

# --- Figure 1: variance of running log-weight vs schedule fraction ---------
let W=680,H=440,L=64,R=20,T=44,B=56
    ymax = maximum(maximum(s.varlw) for s in stats)*1.08
    io=svg_open(W,H;title="Where the weight variance is injected: Var(running log-weight) vs schedule")
    sx,sy=axes(io,W,H,L,R,T,B,0.0,1.0,0.0,ymax,"annealing schedule fraction  (γ ramps 0→target)","variance of running log-weight")
    for s in stats; print(io,poly([(sx(s.x[i]),sy(s.varlw[i])) for i in eachindex(s.x)],s.col;w=2.4)); end
    legend(io,[(s.lbl,s.col) for s in stats],L+18,T+8)
    svg_close(io,joinpath(OUTDIR,"path_$(case)_variance.svg"))
end

# --- Figure 2: mean running log-weight (the free-energy path) ---------------
let W=680,H=440,L=64,R=20,T=44,B=56
    ymax = maximum(maximum(s.meanlw) for s in stats)*1.08
    io=svg_open(W,H;title="Mean running log-weight along the annealing path")
    sx,sy=axes(io,W,H,L,R,T,B,0.0,1.0,0.0,ymax,"annealing schedule fraction","mean running log-weight (nats)")
    for s in stats; print(io,poly([(sx(s.x[i]),sy(s.meanlw[i])) for i in eachindex(s.x)],s.col;w=2.4)); end
    legend(io,[(s.lbl,s.col) for s in stats],L+18,T+8)
    svg_close(io,joinpath(OUTDIR,"path_$(case)_mean.svg"))
end

# --- Figure 3: normalized variance-accumulation curve ----------------------
let W=680,H=440,L=64,R=20,T=44,B=56
    io=svg_open(W,H;title="Fraction of final weight-variance accumulated by each schedule point")
    sx,sy=axes(io,W,H,L,R,T,B,0.0,1.0,0.0,1.05,"annealing schedule fraction","cumulative fraction of final variance")
    print(io,poly([(sx(0),sy(0)),(sx(1),sy(1))],AX;w=1.0,dash="4,4"))
    for s in stats
        vend=s.varlw[end]; print(io,poly([(sx(s.x[i]),sy(s.varlw[i]/vend)) for i in eachindex(s.x)],s.col;w=2.4))
    end
    legend(io,[(s.lbl,s.col) for s in stats],L+18,T+8)
    print(io,txt(sx(0.7),sy(0.62),"diagonal = variance injected evenly";col=AX,size=11))
    svg_close(io,joinpath(OUTDIR,"path_$(case)_varaccum.svg"))
end

println("\n=== annealing-path summary ===")
for s in stats
    # schedule fraction at which half the final variance has accumulated
    vend=s.varlw[end]; half=findfirst(v->v>=0.5*vend, s.varlw)
    @printf("%-9s  final Var(logw)=%.3f  (σ=%.3f)   ½-variance reached by schedule frac %.2f\n",
            s.lbl, vend, sqrt(vend), s.x[half])
end

# --- suggested reschedule from the longest-annealing run -------------------
# Under the LINEAR schedule the local variance-injection rate r(f)=dVar/df is ∝ the
# thermodynamic susceptibility χ(f) of the annealed measure. For a fixed step budget the
# total weight variance ∫r df is minimized (Cauchy–Schwarz) by spending steps ∝ √r, i.e.
# a schedule whose cumulative step position is S(f)=∫₀^f √r / ∫₀^1 √r and whose γ-fraction
# is the inverse f(u). Predicted optimal variance = (∫√r df)²  ≤  ∫r df = current.
sref = stats[end]                                    # longest schedule ≈ best χ estimate
f = sref.x; V = sref.varlw
r = similar(V)
r[1] = (V[2]-V[1])/(f[2]-f[1])
for i in 2:length(V)-1; r[i] = (V[i+1]-V[i-1])/(f[i+1]-f[i-1]); end
r[end] = (V[end]-V[end-1])/(f[end]-f[end-1])
r = max.(r, 0.0); sq = sqrt.(r)
Sc = zeros(length(f))
for i in 2:length(f); Sc[i] = Sc[i-1] + 0.5*(sq[i]+sq[i-1])*(f[i]-f[i-1]); end
Stot = Sc[end]; u = Stot == 0 ? f : Sc ./ Stot       # u(f): step fraction to reach f
Vtot = V[end]; Vopt = Stot^2                          # predicted variance under optimal warp
n = size(read_path(joinpath(OUTDIR, files[end]))[2], 2)

# invert u(f) -> f(u) on an even new-schedule grid
function invert(ug)
    ug <= 0 && return 0.0; ug >= 1 && return 1.0
    k = findfirst(>=(ug), u); (k === nothing || k == 1) && return f[max(k===nothing ? length(f) : 1,1)]
    f[k-1] + (f[k]-f[k-1])*(ug-u[k-1])/(u[k]-u[k-1])
end
ugrid = 0:0.05:1.0; fofu = invert.(ugrid)

println("\n=== suggested reschedule (from $(sref.lbl)) ===")
@printf("current linear: Var=%.3f  σ=%.3f  ESS≈%.1f%%\n", Vtot, sqrt(Vtot), 100*exp(-Vtot))
@printf("optimal warp:   Var=%.3f  σ=%.3f  ESS≈%.1f%%   (variance ×%.2f)\n",
        Vopt, sqrt(Vopt), 100*exp(-Vopt), Vopt/max(Vtot,1e-12))
println("γ-fraction to target at even new-schedule points (u -> γ/target):")
for i in 1:2:length(ugrid)
    @printf("  u=%.2f -> γ/target=%.3f\n", ugrid[i], fofu[i])
end
open(joinpath(OUTDIR, "path_$(case)_reschedule.csv"), "w") do io
    println(io, "u,gamma_fraction")
    for (uu, ff) in zip(ugrid, fofu); @printf(io, "%.4f,%.4f\n", uu, ff); end
end
println("wrote ", joinpath(OUTDIR, "path_$(case)_reschedule.csv"),
        "  (load in run_case.jl to drive a non-linear AnnealPath)")

# warp plot: recommended γ-fraction vs new (even) schedule position
let W=620,H=460,L=60,R=20,T=44,B=56
    io=svg_open(W,H;title="Suggested schedule warp: γ-fraction vs even step position")
    sx,sy=axes(io,W,H,L,R,T,B,0.0,1.0,0.0,1.0,"new schedule position (uniform steps)","γ as fraction of target")
    print(io,poly([(sx(0),sy(0)),(sx(1),sy(1))],AX;w=1.0,dash="4,4"))
    print(io,poly([(sx(ugrid[i]),sy(fofu[i])) for i in eachindex(ugrid)],"#dc2626";w=2.6))
    print(io,txt(sx(0.62),sy(0.30),"dashed = current linear schedule";col=AX,size=11))
    svg_close(io,joinpath(OUTDIR,"path_$(case)_reschedule.svg"))
end
