#!/usr/bin/env python3
"""Self-contained HTML figure for a bisection run: R-hat-vs-t transition curve + marginal
overlays at the two bracketing points. Reads <res>/summary.tsv and <res>/marginals.csv,
writes <res>/report.html. Pure stdlib (no matplotlib)."""
import csv, os, argparse, collections, math

ap = argparse.ArgumentParser()
ap.add_argument("--case", required=True)
ap.add_argument("--res", required=True)
ap.add_argument("--thr", type=float, default=1.01)
A = ap.parse_args()

def tagof(t):        # t string -> "t%.6f" tag matching the driver
    return "t%.6f" % float(t)

# --- read summary ----------------------------------------------------------
rows = []
with open(os.path.join(A.res, "summary.tsv")) as f:
    for r in csv.DictReader(f, delimiter="\t"):
        try: t = float(r["t"])
        except ValueError: continue
        rh = None
        try: rh = float(r["rhat"])
        except (ValueError, KeyError): pass
        rows.append({"t": t, "verdict": r["verdict"], "rhat": rh, "tag": tagof(r["t"])})
rows.sort(key=lambda d: d["t"])

conv_ts = [d["t"] for d in rows if d["verdict"] == "CONVERGED"]
not_ts  = [d["t"] for d in rows if d["verdict"] != "CONVERGED"]
lo = max(conv_ts) if conv_ts else None
hi = min([t for t in not_ts if lo is None or t > lo], default=None)

# --- read marginals --------------------------------------------------------
marg = collections.defaultdict(lambda: collections.defaultdict(list))  # (tag,rank)->copy->[(x,d)]
mpath = os.path.join(A.res, "marginals.csv")
if os.path.exists(mpath):
    with open(mpath) as f:
        for r in csv.DictReader(f):
            marg[(r["point_tag"], int(r["rank"]))][int(r["copy"])].append(
                (float(r["bin_center"]), float(r["density"])))

# --- R-hat-vs-t curve SVG --------------------------------------------------
def rhat_curve():
    pts = [(d["t"], d["rhat"], d["verdict"]) for d in rows if d["rhat"] is not None]
    if not pts:
        return "<p>No R-hat values (criterion was not rhat).</p>"
    W, H = 720, 340; ml, mr, mt, mb = 58, 18, 16, 40
    ymax = max(max(p[1] for p in pts), A.thr) * 1.06
    ymin = min(min(p[1] for p in pts), 1.0) - 0.002
    X = lambda t: ml + t * (W - ml - mr)
    Y = lambda y: H - mb - (y - ymin) / (ymax - ymin) * (H - mt - mb)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="R-hat versus t">']
    # y grid + labels
    for gi in range(5):
        yv = ymin + (ymax - ymin) * gi / 4
        s.append(f'<line x1="{ml}" y1="{Y(yv):.1f}" x2="{W-mr}" y2="{Y(yv):.1f}" class="grid"/>')
        s.append(f'<text x="{ml-6}" y="{Y(yv)+3:.1f}" class="tick" text-anchor="end">{yv:.3f}</text>')
    # threshold line
    s.append(f'<line x1="{ml}" y1="{Y(A.thr):.1f}" x2="{W-mr}" y2="{Y(A.thr):.1f}" class="thr"/>')
    s.append(f'<text x="{W-mr}" y="{Y(A.thr)-5:.1f}" class="thrlab" text-anchor="end">R̂ = {A.thr:g}</text>')
    # x ticks
    for xv in (0, 0.25, 0.5, 0.75, 1.0):
        s.append(f'<text x="{X(xv):.1f}" y="{H-mb+16:.1f}" class="tick" text-anchor="middle">{xv:g}</text>')
    s.append(f'<text x="{(ml+W-mr)/2:.0f}" y="{H-4}" class="axlab" text-anchor="middle">t   (γ = t, iso = {A.case and ""}0.3·t)</text>')
    # bracket shading
    if lo is not None and hi is not None:
        s.append(f'<rect x="{X(lo):.1f}" y="{mt}" width="{X(hi)-X(lo):.1f}" height="{H-mt-mb}" class="brk"/>')
    # connecting line
    line = " ".join(f"{X(t):.1f},{Y(r):.1f}" for t, r, _ in pts)
    s.append(f'<polyline points="{line}" class="curve"/>')
    # points
    for t, r, v in pts:
        cls = "cok" if v == "CONVERGED" else "cno"
        s.append(f'<circle cx="{X(t):.1f}" cy="{Y(r):.1f}" r="4.5" class="{cls}"/>')
    s.append("</svg>")
    return "".join(s)

# --- overlay panels for one point ------------------------------------------
def overlay(tag, hue):
    ranks = sorted({rk for (tg, rk) in marg if tg == tag})
    if not ranks:
        return f'<p class="muted">no marginals for {tag}</p>'
    PW, PH, ML, MR, MT, MB = 260, 150, 30, 8, 10, 24
    cells = []
    for rank in ranks:
        copies = marg[(tag, rank)]
        xs_all = [x for c in copies.values() for x, _ in c]
        ys_all = [y for c in copies.values() for _, y in c]
        xmin, xmax = min(xs_all), max(xs_all); ymax = max(ys_all) * 1.08 or 1.0
        X = lambda x: ML + (x - xmin) / ((xmax - xmin) or 1) * (PW - ML - MR)
        Yf = lambda y: PH - MB - y / ymax * (PH - MT - MB)
        p = [f'<line x1="{ML}" y1="{Yf(0):.1f}" x2="{PW-MR}" y2="{Yf(0):.1f}" class="axis"/>']
        n = len(copies); mean = None
        for ci in sorted(copies):
            pts = sorted(copies[ci]); d = " ".join(f"{X(x):.1f},{Yf(y):.1f}" for x, y in pts)
            p.append(f'<polyline points="{d}" class="copy {hue}"/>')
            if mean is None: mean = [[x, 0.0] for x, _ in pts]
            for i, (_, y) in enumerate(pts): mean[i][1] += y / n
        dm = " ".join(f"{X(x):.1f},{Yf(y):.1f}" for x, y in mean)
        p.append(f'<polyline points="{dm}" class="mean {hue}"/>')
        for xv in (xmin, xmax):
            p.append(f'<text x="{X(xv):.1f}" y="{PH-MB+14:.1f}" class="tick" text-anchor="middle">{xv:.0f}</text>')
        cells.append(f'<figure class="panel"><figcaption>Rank {rank}</figcaption>'
                     f'<svg viewBox="0 0 {PW} {PH}">{"".join(p)}</svg></figure>')
    return f'<div class="panels">{"".join(cells)}</div>'

# --- assemble --------------------------------------------------------------
brk = (f"convergence breaks at <b>t∗ ∈ ({lo:g}, {hi:g}]</b>"
       if (lo is not None and hi is not None) else
       ("all tested points CONVERGED — no break in [0,1]" if not not_ts
        else "no CONVERGED point — breaks at or below the lowest t tested"))

sections = [f'<section><h2>R̂ transition</h2>{rhat_curve()}'
            f'<p class="cap">Each point is one t on the line; green = CONVERGED '
            f'(max split-R̂ &lt; {A.thr:g}), red = NOT. Shaded band = the final bracket. '
            f'<b>{brk}</b></p></section>']
if lo is not None:
    sections.append(f'<section><h2>Marginals at the converged edge (t = {lo:g})</h2>{overlay(tagof(lo), "teal")}</section>')
if hi is not None:
    sections.append(f'<section><h2>Marginals just past the break (t = {hi:g})</h2>{overlay(tagof(hi), "amber")}</section>')

STYLE = """
:root{--paper:#f5f7f6;--panel:#fff;--ink:#171b1d;--ink2:#586066;--hair:#d7dedb;--grid:#eef2f0;
 --teal:#0e7c72;--teal-soft:rgba(14,124,114,.30);--amber:#b3630a;--amber-soft:rgba(179,99,10,.30);
 --ok:#0e7c72;--no:#b3300a;--thr:#8a6d3b;}
@media(prefers-color-scheme:dark){:root{--paper:#101413;--panel:#171c1b;--ink:#e9edeb;--ink2:#93a09c;
 --hair:#28312e;--grid:#1e2523;--teal:#4bd6c4;--teal-soft:rgba(75,214,196,.28);--amber:#e8a24d;
 --amber-soft:rgba(232,162,77,.30);--ok:#4bd6c4;--no:#f0885a;--thr:#c8ab6a;}}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
 font:400 15.5px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
.mono,.tick,.thrlab,figcaption{font-family:ui-monospace,Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;}
main{max-width:1000px;margin:0 auto;padding:40px 24px 64px;}
h1{font-size:28px;letter-spacing:-.02em;margin:0 0 6px;} h2{font-size:18px;margin:30px 0 10px;}
.sub{color:var(--ink2);margin:0 0 8px;} .cap{color:var(--ink2);font-size:13.5px;max-width:70ch;} .cap b{color:var(--ink);}
.muted{color:var(--ink2);} svg{display:block;width:100%;height:auto;background:var(--panel);border:1px solid var(--hair);border-radius:8px;}
.grid{stroke:var(--grid);stroke-width:1;} .axis{stroke:var(--hair);stroke-width:1;} .tick{fill:var(--ink2);font-size:10px;}
.thr{stroke:var(--thr);stroke-width:1.4;stroke-dasharray:5 4;} .thrlab{fill:var(--thr);font-size:11px;}
.axlab{fill:var(--ink2);font-size:12px;} .brk{fill:rgba(179,99,10,.10);}
.curve{fill:none;stroke:var(--ink2);stroke-width:1.4;} .cok{fill:var(--ok);} .cno{fill:var(--no);}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:4px;}
.panel{margin:0;} .panel figcaption{font-size:11px;color:var(--ink2);margin:0 0 3px 2px;}
.panel svg{border-radius:8px;} .copy{fill:none;stroke-width:1.05;} .copy.teal{stroke:var(--teal-soft);} .copy.amber{stroke:var(--amber-soft);}
.mean{fill:none;stroke-width:2;} .mean.teal{stroke:var(--teal);} .mean.amber{stroke:var(--amber);}
"""
CASE = A.case.upper()
html = (f"<style>{STYLE}</style><main>"
        f"<h1>{CASE}: cycle-walk convergence bisection</h1>"
        f'<p class="sub">Standard cycle walk on the line γ = t, iso = 0.3·t. '
        f'{len(rows)} points, {sum(1 for d in rows if d["verdict"]=="CONVERGED")} converged. '
        f'Criterion: rank-normalized split-R̂ &lt; {A.thr:g}.</p>'
        + "".join(sections) + "</main>")
open(os.path.join(A.res, "report.html"), "w").write(html)
print("plot_bisection: wrote", os.path.join(A.res, "report.html"), len(html), "bytes")
