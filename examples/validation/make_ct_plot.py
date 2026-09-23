#!/usr/bin/env python3
"""Render the CT cross-copy convergence overlay (5 districts) as a self-contained artifact."""
import csv, collections, os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "output", "validation", "ct_marginals.csv")
OUT = os.path.join(HERE, "..", "output", "validation", "ct_convergence.html")

data = collections.defaultdict(lambda: collections.defaultdict(list))
with open(CSV) as f:
    for row in csv.DictReader(f):
        data[(row["point_tag"], int(row["rank"]))][int(row["copy"])].append(
            (float(row["bin_center"]), float(row["density"])))

# analyze_convergence stats: rank -> (maxTV, twoNoise, maxKS, meanDelta)
STATS = {
    "t0.000000": {1:(0.0219,0.0557,0.0151,0.174),2:(0.0229,0.0681,0.0126,0.092),
                  3:(0.0243,0.0657,0.0133,0.131),4:(0.0250,0.0661,0.0140,0.141),
                  5:(0.0235,0.0703,0.0197,0.333)},
    "t1.000000": {1:(0.1602,0.5367,0.1615,2.655),2:(0.1713,0.4619,0.1726,2.190),
                  3:(0.1512,0.3722,0.1534,1.777),4:(0.1090,0.2246,0.1094,1.369),
                  5:(0.0726,0.1619,0.0739,1.260)},
}
POINTS = [
    ("t0.000000", "t = 0", "flat base · γ = 0, iso = 0", "teal"),
    ("t1.000000", "t = 1", "sharp target · γ = 1, iso = 0.3", "amber"),
]
RANKS = sorted({r for (_, r) in data})
RANK_LABEL = {r: f"Rank {r}" for r in RANKS}
RANK_LABEL[min(RANKS)] += " · fewest trees"
RANK_LABEL[max(RANKS)] += " · most trees"

PW, PH = 300, 190
ML, MR, MT, MB = 40, 10, 12, 32

def panel_svg(point, rank, hue):
    copies = data[(point, rank)]
    xs_all = [x for c in copies.values() for x, _ in c]
    ys_all = [y for c in copies.values() for _, y in c]
    xmin, xmax = min(xs_all), max(xs_all)
    ymax = max(ys_all) * 1.08 or 1.0
    X = lambda x: ML + (x - xmin) / (xmax - xmin) * (PW - ML - MR)
    Y = lambda y: PH - MB - y / ymax * (PH - MT - MB)
    parts = []
    for gi in range(1, 4):
        gy = Y(ymax * gi / 4)
        parts.append(f'<line x1="{ML}" y1="{gy:.1f}" x2="{PW-MR}" y2="{gy:.1f}" class="grid"/>')
    parts.append(f'<line x1="{ML}" y1="{Y(0):.1f}" x2="{PW-MR}" y2="{Y(0):.1f}" class="axis"/>')
    for xv in (xmin, (xmin+xmax)/2, xmax):
        parts.append(f'<text x="{X(xv):.1f}" y="{PH-MB+15:.1f}" class="tick" text-anchor="middle">{xv:.0f}</text>')
    n = len(copies); mean = None
    for ci in sorted(copies):
        pts = sorted(copies[ci])
        d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        parts.append(f'<polyline points="{d}" class="copy {hue}"/>')
        if mean is None: mean = [[x, 0.0] for x, _ in pts]
        for i, (_, y) in enumerate(pts): mean[i][1] += y / n
    dm = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in mean)
    parts.append(f'<polyline points="{dm}" class="mean {hue}"/>')
    tv, tn, ks, md = STATS[point][rank]
    hot = " hot" if md > 1.0 else ""
    return (f'<figure class="panel"><figcaption class="pcap">{RANK_LABEL[rank]}</figcaption>'
            f'<svg viewBox="0 0 {PW} {PH}" role="img" aria-label="overlaid density of 8 copies">{"".join(parts)}</svg>'
            f'<div class="stat"><span>TV <b>{tv:.3f}</b></span>'
            f'<span>floor <b>{tn:.3f}</b></span>'
            f'<span class="sp{hot}">spread <b>{md:.2f}</b></span></div></figure>')

rows = []
for point, big, sub, hue in POINTS:
    panels = "".join(panel_svg(point, r, hue) for r in RANKS)
    rows.append(f'<section class="row {hue}"><header class="rowhead"><div class="tt">{big}</div>'
                f'<div class="ts">{sub}</div>'
                f'<div class="verdict">8 / 8 agree · <b>CONVERGED</b></div></header>'
                f'<div class="panels">{panels}</div></section>')

BODY = f"""
<main>
  <header class="masthead">
    <p class="eyebrow">Cycle-walk convergence diagnostic · Connecticut, 5 districts</p>
    <h1>Converges at both ends — but the sharp target strains it</h1>
    <p class="lede">Each panel overlays the rank-ordered marginal of the per-district
      <em>log spanning-tree count</em> from <strong>8 independent cycle-walk copies</strong>
      (20,000 samples each). Curves are drawn in one translucent ink: a single tight band
      means the chains reached the same distribution; visible separation means they are
      drifting apart. The solid line is the pointwise mean. <b>spread</b> is the gap between
      the highest and lowest copy means, in nats.</p>
  </header>
  {"".join(rows)}
  <footer class="foot">
    <p><b>Both rows pass the verdict</b> — every rank's between-copy total variation
    (<span class="mono">TV</span>) stays under the within-chain noise <span class="mono">floor</span>,
    so no chain is provably sampling a different distribution. That is why the bisection never
    triggered: neither endpoint fails.</p>
    <p><b>But the sharp target is visibly harder.</b> From t=0 to t=1 the noise floor grows
    roughly <b>10×</b> and the copy-mean <span class="mono">spread</span> grows from a few
    tenths of a nat to <b>1.3–2.7 nats</b> (highlighted). The chains are mixing far more slowly
    at γ=1 — the bands widen and the copies start to pull apart — they simply have not separated
    beyond their own sampling noise at 20k samples.</p>
    <p><b>Takeaway.</b> The standard cycle walk stays self-consistent across the whole
    (γ, iso) line; the earlier AIS weight collapse was an importance-weighting problem, not a
    failure of the walk to mix. The graceful slow-down here marks where a harder case
    (NC, 14 districts) or a longer run would push it past breaking.</p>
  </footer>
</main>
"""

STYLE = """
:root{--paper:#f5f7f6;--panel:#fff;--ink:#171b1d;--ink2:#586066;--hair:#d7dedb;--grid:#eef2f0;
  --teal:#0e7c72;--teal-soft:rgba(14,124,114,.30);--amber:#b3630a;--amber-soft:rgba(179,99,10,.30);
  --good:#0e7c72;--hot:#b3300a;}
@media(prefers-color-scheme:dark){:root{--paper:#101413;--panel:#171c1b;--ink:#e9edeb;--ink2:#93a09c;
  --hair:#28312e;--grid:#1e2523;--teal:#4bd6c4;--teal-soft:rgba(75,214,196,.26);--amber:#e8a24d;
  --amber-soft:rgba(232,162,77,.28);--good:#4bd6c4;--hot:#f0885a;}}
:root[data-theme="light"]{--paper:#f5f7f6;--panel:#fff;--ink:#171b1d;--ink2:#586066;--hair:#d7dedb;--grid:#eef2f0;--teal:#0e7c72;--teal-soft:rgba(14,124,114,.30);--amber:#b3630a;--amber-soft:rgba(179,99,10,.30);--good:#0e7c72;--hot:#b3300a;}
:root[data-theme="dark"]{--paper:#101413;--panel:#171c1b;--ink:#e9edeb;--ink2:#93a09c;--hair:#28312e;--grid:#1e2523;--teal:#4bd6c4;--teal-soft:rgba(75,214,196,.26);--amber:#e8a24d;--amber-soft:rgba(232,162,77,.28);--good:#4bd6c4;--hot:#f0885a;}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:400 16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;-webkit-font-smoothing:antialiased;}
.mono,.tick,.stat,.pcap{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-variant-numeric:tabular-nums;}
main{max-width:1080px;margin:0 auto;padding:44px 24px 72px;}
.masthead{max-width:68ch;margin-bottom:34px;}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.10em;text-transform:uppercase;color:var(--ink2);margin:0 0 12px;}
h1{font-size:clamp(26px,3.6vw,40px);line-height:1.07;letter-spacing:-.02em;margin:0 0 16px;text-wrap:balance;font-weight:660;}
.lede{color:var(--ink2);font-size:16px;margin:0;} .lede em{font-style:italic;color:var(--ink);} .lede b,.lede strong{color:var(--ink);font-weight:620;}
.row{border-top:1px solid var(--hair);padding-top:22px;margin-top:30px;}
.rowhead{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:14px;}
.tt{font-size:22px;font-weight:660;letter-spacing:-.01em;} .row.teal .tt{color:var(--teal)} .row.amber .tt{color:var(--amber)}
.ts{color:var(--ink2);font-family:ui-monospace,Menlo,monospace;font-size:12.5px;}
.verdict{margin-left:auto;font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.03em;color:var(--ink2);border:1px solid var(--hair);border-radius:999px;padding:4px 12px;} .verdict b{color:var(--good);}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(188px,1fr));gap:14px;}
.panel{margin:0;background:var(--panel);border:1px solid var(--hair);border-radius:10px;padding:11px 11px 9px;}
.pcap{font-size:11px;color:var(--ink2);letter-spacing:.01em;margin-bottom:4px;}
svg{display:block;width:100%;height:auto;}
.grid{stroke:var(--grid);stroke-width:1;} .axis{stroke:var(--hair);stroke-width:1;} .tick{fill:var(--ink2);font-size:9.5px;}
.copy{fill:none;stroke-width:1.1;stroke-linejoin:round;stroke-linecap:round;} .copy.teal{stroke:var(--teal-soft);} .copy.amber{stroke:var(--amber-soft);}
.mean{fill:none;stroke-width:2;stroke-linejoin:round;} .mean.teal{stroke:var(--teal);} .mean.amber{stroke:var(--amber);}
.stat{display:flex;gap:10px;justify-content:space-between;margin-top:8px;font-size:10.5px;color:var(--ink2);border-top:1px dashed var(--hair);padding-top:6px;}
.stat b{color:var(--ink);font-weight:600;} .stat .sp.hot{color:var(--hot);} .stat .sp.hot b{color:var(--hot);}
.foot{border-top:1px solid var(--hair);margin-top:38px;padding-top:20px;max-width:72ch;color:var(--ink2);font-size:14.5px;} .foot b{color:var(--ink);} .foot p{margin:0 0 12px;}
"""

html = f"<style>{STYLE}</style>{BODY}"
open(OUT, "w").write(html)
print("wrote", OUT, len(html), "bytes; ranks:", RANKS)
