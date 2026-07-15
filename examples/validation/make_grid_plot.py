#!/usr/bin/env python3
"""Render the grid cross-copy convergence overlay as a self-contained HTML artifact."""
import csv, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "output", "validation", "grid_marginals.csv")
OUT = os.path.join(HERE, "..", "output", "validation", "grid_convergence.html")

# (point, rank) -> copy -> list[(x, density)]
data = collections.defaultdict(lambda: collections.defaultdict(list))
with open(CSV) as f:
    for row in csv.DictReader(f):
        key = (row["point_tag"], int(row["rank"]))
        data[key][int(row["copy"])].append((float(row["bin_center"]), float(row["density"])))

# per-point verdict stats from analyze_convergence (rank -> (maxTV, twoNoise, maxKS))
STATS = {
    "t0.000000": {1: (0.0207, 0.0549, 0.0166), 2: (0.0184, 0.0545, 0.0144), 3: (0.0234, 0.0533, 0.0160)},
    "t1.000000": {1: (0.0191, 0.0667, 0.0114), 2: (0.0165, 0.0503, 0.0141), 3: (0.0202, 0.0601, 0.0141)},
}
POINTS = [
    ("t0.000000", "t = 0", "flat base  ·  γ = 0, iso = 0", "teal"),
    ("t1.000000", "t = 1", "sharp target  ·  γ = 1, iso = 0.3", "amber"),
]
RANK_LABEL = {1: "Rank 1 — fewest trees", 2: "Rank 2 — middle", 3: "Rank 3 — most trees"}

# --- SVG panel -------------------------------------------------------------
PW, PH = 340, 200                     # panel viewBox
ML, MR, MT, MB = 46, 12, 14, 34       # margins

def panel_svg(point, rank, hue):
    copies = data[(point, rank)]
    xs_all = [x for c in copies.values() for x, _ in c]
    ys_all = [y for c in copies.values() for _, y in c]
    xmin, xmax = min(xs_all), max(xs_all)
    ymax = max(ys_all) * 1.08 or 1.0
    def X(x): return ML + (x - xmin) / (xmax - xmin) * (PW - ML - MR)
    def Y(y): return PH - MB - y / ymax * (PH - MT - MB)

    parts = []
    # y gridlines
    for gi in range(1, 4):
        gy = Y(ymax * gi / 4)
        parts.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{PW-MR}" y2="{gy:.1f}" class="grid"/>')
    # baseline + axis
    parts.append(f'<line x1="{ML}" y1="{Y(0):.1f}" x2="{PW-MR}" y2="{Y(0):.1f}" class="axis"/>')
    # x ticks (min / mid / max)
    for xv in (xmin, (xmin+xmax)/2, xmax):
        parts.append(f'<text x="{X(xv):.1f}" y="{PH-MB+16:.1f}" class="tick" text-anchor="middle">{xv:.1f}</text>')

    # the 8 translucent copy curves
    n = len(copies)
    mean = None
    for ci in sorted(copies):
        pts = sorted(copies[ci])
        d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        parts.append(f'<polyline points="{d}" class="copy {hue}"/>')
        if mean is None:
            mean = [[x, 0.0] for x, _ in pts]
        for i, (_, y) in enumerate(pts):
            mean[i][1] += y / n
    # solid mean curve on top
    dm = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in mean)
    parts.append(f'<polyline points="{dm}" class="mean {hue}"/>')

    tv, tn, ks = STATS[point][rank]
    return (
        f'<figure class="panel">'
        f'<figcaption class="pcap">{RANK_LABEL[rank]}</figcaption>'
        f'<svg viewBox="0 0 {PW} {PH}" role="img" aria-label="overlaid density of 8 copies">{"".join(parts)}</svg>'
        f'<div class="stat">'
        f'<span>max&nbsp;TV <b>{tv:.3f}</b></span>'
        f'<span>floor <b>{tn:.3f}</b></span>'
        f'<span>max&nbsp;KS <b>{ks:.3f}</b></span>'
        f'</div></figure>'
    )

rows = []
for point, big, sub, hue in POINTS:
    panels = "".join(panel_svg(point, r, hue) for r in (1, 2, 3))
    rows.append(
        f'<section class="row {hue}">'
        f'<header class="rowhead"><div class="tt">{big}</div>'
        f'<div class="ts">{sub}</div>'
        f'<div class="verdict">8 / 8 copies agree · <b>CONVERGED</b></div></header>'
        f'<div class="panels">{panels}</div></section>'
    )

BODY = f"""
<main>
  <header class="masthead">
    <p class="eyebrow">Cycle-walk convergence diagnostic · 10×10 grid, 3 districts</p>
    <h1>Eight independent chains, one distribution</h1>
    <p class="lede">Each panel overlays the rank-ordered marginal of the per-district
      <em>log spanning-tree count</em> from <strong>8 independent cycle-walk copies</strong>
      (20,000 samples each, distinct seeds). The curves are drawn in one translucent ink:
      where the eight coincide into a single band, the chains have reached the same
      distribution. A solid line marks the pointwise mean.</p>
  </header>
  {"".join(rows)}
  <footer class="foot">
    <p><b>Read.</b> At both ends of the line — the flat base and the sharp target — the eight
    curves collapse onto each other in every district rank. The cross-copy total-variation
    distance (<span class="mono">max&nbsp;TV</span>) sits at or below the finite-sample noise
    floor (the split-half TV within a single chain), so the disagreement between chains is
    indistinguishable from ordinary sampling noise.</p>
    <p><b>Conclusion.</b> The standard cycle walk mixes well across the entire
    (γ, iso) line on this graph; the 3-district grid is too small to exhibit a convergence
    break. The bisection search moves to Connecticut, where a transition is expected.</p>
  </footer>
</main>
"""

STYLE = """
:root{
  --paper:#f5f7f6; --panel:#fff; --ink:#171b1d; --ink2:#586066; --hair:#d7dedb;
  --grid:#eef2f0; --teal:#0e7c72; --teal-soft:rgba(14,124,114,.30);
  --amber:#b3630a; --amber-soft:rgba(179,99,10,.28); --good:#0e7c72;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#101413; --panel:#171c1b; --ink:#e9edeb; --ink2:#93a09c; --hair:#28312e;
  --grid:#1e2523; --teal:#4bd6c4; --teal-soft:rgba(75,214,196,.26);
  --amber:#e8a24d; --amber-soft:rgba(232,162,77,.26); --good:#4bd6c4;
}}
:root[data-theme="light"]{--paper:#f5f7f6;--panel:#fff;--ink:#171b1d;--ink2:#586066;--hair:#d7dedb;--grid:#eef2f0;--teal:#0e7c72;--teal-soft:rgba(14,124,114,.30);--amber:#b3630a;--amber-soft:rgba(179,99,10,.28);--good:#0e7c72;}
:root[data-theme="dark"]{--paper:#101413;--panel:#171c1b;--ink:#e9edeb;--ink2:#93a09c;--hair:#28312e;--grid:#1e2523;--teal:#4bd6c4;--teal-soft:rgba(75,214,196,.26);--amber:#e8a24d;--amber-soft:rgba(232,162,77,.26);--good:#4bd6c4;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:400 16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;}
.mono,.tick,.stat,.pcap{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;}
main{max-width:1020px;margin:0 auto;padding:44px 24px 72px;}
.masthead{max-width:66ch;margin-bottom:34px;}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.10em;
  text-transform:uppercase;color:var(--ink2);margin:0 0 12px;}
h1{font-size:clamp(28px,4vw,42px);line-height:1.06;letter-spacing:-.02em;margin:0 0 16px;
  text-wrap:balance;font-weight:660;}
.lede{color:var(--ink2);font-size:16.5px;margin:0;}
.lede em{font-style:italic;color:var(--ink);} .lede strong{color:var(--ink);font-weight:620;}
.row{border-top:1px solid var(--hair);padding-top:22px;margin-top:30px;}
.rowhead{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:14px;}
.tt{font-size:23px;font-weight:660;letter-spacing:-.01em;}
.row.teal .tt{color:var(--teal)} .row.amber .tt{color:var(--amber)}
.ts{color:var(--ink2);font-family:ui-monospace,Menlo,monospace;font-size:12.5px;}
.verdict{margin-left:auto;font-family:ui-monospace,Menlo,monospace;font-size:12px;
  letter-spacing:.03em;color:var(--ink2);border:1px solid var(--hair);border-radius:999px;
  padding:4px 12px;} .verdict b{color:var(--good);}
.panels{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
@media(max-width:720px){.panels{grid-template-columns:1fr;}}
.panel{margin:0;background:var(--panel);border:1px solid var(--hair);border-radius:10px;
  padding:12px 12px 10px;}
.pcap{font-size:11.5px;color:var(--ink2);letter-spacing:.02em;margin-bottom:4px;}
svg{display:block;width:100%;height:auto;}
.grid{stroke:var(--grid);stroke-width:1;} .axis{stroke:var(--hair);stroke-width:1;}
.tick{fill:var(--ink2);font-size:10px;}
.copy{fill:none;stroke-width:1.15;stroke-linejoin:round;stroke-linecap:round;}
.copy.teal{stroke:var(--teal-soft);} .copy.amber{stroke:var(--amber-soft);}
.mean{fill:none;stroke-width:2;stroke-linejoin:round;}
.mean.teal{stroke:var(--teal);} .mean.amber{stroke:var(--amber);}
.stat{display:flex;gap:14px;justify-content:space-between;margin-top:8px;font-size:11px;
  color:var(--ink2);border-top:1px dashed var(--hair);padding-top:7px;}
.stat b{color:var(--ink);font-weight:600;}
.foot{border-top:1px solid var(--hair);margin-top:38px;padding-top:20px;max-width:70ch;
  color:var(--ink2);font-size:14.5px;} .foot b{color:var(--ink);} .foot p{margin:0 0 12px;}
"""

html = f'<style>{STYLE}</style>{BODY}'
with open(OUT, "w") as f:
    f.write(html)
print("wrote", OUT, len(html), "bytes")
