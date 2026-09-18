import json, math
R = json.load(open("layer_hit.json"))
mr = [r["miss_rate"]*100 for r in R]
h1 = [r["miss_h1"]*100 for r in R]
h2 = [r["miss_h2"]*100 for r in R]
worst = sorted(range(40), key=lambda i: -mr[i])[:4]

W, H = 940, 340
L, Rg, T, B = 52, 14, 18, 44
pw, ph = W-L-Rg, H-T-B
ymax = 2.0
def x(i): return L + (i+0.5)*pw/40
def y(v): return T + ph*(1 - v/ymax)
bw = pw/40*0.56

bars, marks = [], []
for i in range(40):
    cls = "bar flag" if i in worst else "bar"
    bars.append(f'<rect class="{cls}" x="{x(i)-bw/2:.1f}" y="{y(mr[i]):.1f}" '
                f'width="{bw:.1f}" height="{max(0.6,y(0)-y(mr[i])):.1f}" rx="1.5"><title>'
                f'L{i}: {mr[i]:.2f}% miss ({R[i]["misses"]} of {R[i]["acc"]:,})</title></rect>')
    marks.append(f'<line class="hh" x1="{x(i)-bw/2-1.5:.1f}" x2="{x(i)+bw/2+1.5:.1f}" '
                 f'y1="{y(h1[i]):.1f}" y2="{y(h1[i]):.1f}"/>')
    marks.append(f'<line class="hh" x1="{x(i)-bw/2-1.5:.1f}" x2="{x(i)+bw/2+1.5:.1f}" '
                 f'y1="{y(h2[i]):.1f}" y2="{y(h2[i]):.1f}"/>')
grid = []
for v in (0.5, 1.0, 1.5, 2.0):
    grid.append(f'<line class="g" x1="{L}" x2="{W-Rg}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
    grid.append(f'<text class="ax" x="{L-8}" y="{y(v)+3.5:.1f}" text-anchor="end">{v:.1f}%</text>')
xlab = []
for i in (0, 5, 10, 15, 19, 20, 25, 30, 35, 39):
    xlab.append(f'<text class="ax" x="{x(i):.1f}" y="{H-B+16:.1f}" text-anchor="middle">{i}</text>')
seam = (f'<line class="seam" x1="{x(19.5):.1f}" x2="{x(19.5):.1f}" y1="{T}" y2="{y(0):.1f}"/>'
        f'<text class="seamlab" x="{x(19.5)+6:.1f}" y="{T+12}">CED seam · decoder starts</text>')
mean = sum(r["misses"] for r in R)/sum(r["acc"] for r in R)*100
meanline = (f'<line class="mean" x1="{L}" x2="{W-Rg}" y1="{y(mean):.1f}" y2="{y(mean):.1f}"/>'
            f'<text class="meanlab" x="{W-Rg}" y="{y(mean)-6:.1f}" text-anchor="end">mean {mean:.2f}%</text>')
chart1 = (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Per-layer decode miss rate">'
          + "".join(grid) + seam + "".join(bars) + "".join(marks) + meanline
          + "".join(xlab)
          + f'<text class="ax axt" x="{L+pw/2:.0f}" y="{H-6}" text-anchor="middle">layer</text>'
          + '</svg>')

# --- ruled-out correlations ---
CO = [("depth (layer index)", 0.037), ("experts box 1 owns", 0.005),
      ("routing entropy", -0.129), ("top-16 expert share", 0.174),
      ("median reuse gap", -0.262), ("distinct experts touched", -0.077),
      ("SPLIT-HALF replication", 0.803)]
W2 = 940; rowh = 34; H2 = rowh*len(CO)+26
cx = 430; sc = 300
rows2 = []
for i,(name,r) in enumerate(CO):
    yy = 20 + i*rowh
    cls = "cbar strong" if abs(r) > 0.5 else "cbar"
    x0 = cx + (0 if r >= 0 else r*sc); w = abs(r)*sc
    rows2.append(f'<text class="cn" x="{cx-sc-14}" y="{yy+13}">{name}</text>')
    rows2.append(f'<rect class="{cls}" x="{x0:.1f}" y="{yy}" width="{max(1.5,w):.1f}" height="19" rx="2"/>')
    rows2.append(f'<text class="cv" x="{cx+sc+16}" y="{yy+13}">{r:+.3f}</text>')
rows2.append(f'<line class="zero" x1="{cx}" x2="{cx}" y1="8" y2="{H2-10}"/>')
chart2 = f'<svg viewBox="0 0 {W2} {H2}" role="img" aria-label="Correlations">' + "".join(rows2) + '</svg>'

tbl = "".join(
    f"<tr><td>L{i}</td><td class='n'>{mr[i]:.2f}</td><td class='n'>{R[i]['misses']}</td>"
    f"<td class='n'>{R[i]['acc']:,}</td><td class='n'>{100*R[i]['cold_share']:.1f}</td></tr>"
    for i in sorted(range(40), key=lambda i: -mr[i])[:8])

html = f"""<title>Expert Cache Miss by Layer</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#F7F6F1;--card:#FFFFFF;--ink:#191C21;--mute:#6C7076;--grid:#E2DED4;
--accent:#0B6E63;--flag:#B0402F;--rule:#DED9CD;--seam:#B9AE97;}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--paper:#11141A;--card:#181C23;
--ink:#E9E6E0;--mute:#949AA3;--grid:#262C35;--accent:#4FB3A4;--flag:#E07A64;--rule:#2A313A;--seam:#4A5360;}}}}
:root[data-theme="dark"]{{--paper:#11141A;--card:#181C23;--ink:#E9E6E0;--mute:#949AA3;--grid:#262C35;
--accent:#4FB3A4;--flag:#E07A64;--rule:#2A313A;--seam:#4A5360;}}
*{{box-sizing:border-box}}
body{{background:var(--paper);color:var(--ink);font:16px/1.6 "Source Sans 3",system-ui,sans-serif;
margin:0;padding:40px 24px 72px;}}
main{{max-width:980px;margin:0 auto;display:flex;flex-direction:column;gap:34px}}
.eyebrow{{font:500 12px/1 "IBM Plex Mono",monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--mute)}}
h1{{font:700 clamp(30px,4.4vw,46px)/1.08 Fraunces,Georgia,serif;margin:10px 0 0;text-wrap:balance;letter-spacing:-.015em}}
.sub{{color:var(--mute);max-width:64ch;margin:12px 0 0}}
.panel{{background:var(--card);border:1px solid var(--rule);border-radius:10px;padding:22px 20px}}
.panel h2{{font:700 19px/1.3 Fraunces,Georgia,serif;margin:0 0 4px}}
.panel p.cap{{color:var(--mute);font-size:14.5px;margin:0 0 16px;max-width:70ch}}
.scroll{{overflow-x:auto}}
svg{{display:block;width:100%;height:auto;min-width:640px}}
.g{{stroke:var(--grid);stroke-width:1}}
.ax{{fill:var(--mute);font:400 11px "IBM Plex Mono",monospace}}
.axt{{font-size:12px}}
.bar{{fill:var(--accent);opacity:.85}}
.bar.flag{{fill:var(--flag);opacity:1}}
.hh{{stroke:var(--ink);stroke-width:1.3;opacity:.45}}
.mean{{stroke:var(--ink);stroke-width:1;stroke-dasharray:5 4;opacity:.5}}
.meanlab{{fill:var(--mute);font:400 11px "IBM Plex Mono",monospace}}
.seam{{stroke:var(--seam);stroke-width:1;stroke-dasharray:3 3}}
.seamlab{{fill:var(--seam);font:400 10.5px "IBM Plex Mono",monospace}}
.cn{{fill:var(--ink);font:400 14px "Source Sans 3",sans-serif;text-anchor:end}}
.cv{{fill:var(--mute);font:400 13px "IBM Plex Mono",monospace;text-anchor:end}}
.cbar{{fill:var(--mute);opacity:.55}}
.cbar.strong{{fill:var(--accent);opacity:1}}
.zero{{stroke:var(--rule);stroke-width:1.5}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:1px;background:var(--rule);
border:1px solid var(--rule);border-radius:10px;overflow:hidden}}
.stat{{background:var(--card);padding:16px 18px}}
.stat .v{{font:500 26px/1.1 "IBM Plex Mono",monospace;color:var(--accent);font-variant-numeric:tabular-nums}}
.stat .k{{font-size:13px;color:var(--mute);margin-top:5px;line-height:1.35}}
table{{border-collapse:collapse;width:100%;font-size:14.5px}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule)}}
th{{font:500 11.5px "IBM Plex Mono",monospace;letter-spacing:.06em;text-transform:uppercase;color:var(--mute)}}
td.n{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;text-align:right}}
.key{{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;color:var(--mute);margin-top:14px}}
.key i{{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.verdict{{border-left:3px solid var(--accent);padding:2px 0 2px 18px}}
.verdict strong{{color:var(--ink)}}
footer{{color:var(--mute);font-size:13px;border-top:1px solid var(--rule);padding-top:18px}}
code{{font-family:"IBM Plex Mono",monospace;font-size:.92em;background:var(--grid);padding:1px 5px;border-radius:3px}}
</style>
<main>
<header>
<div class="eyebrow">DeepSeek V4.1-Flash · decode · 19,139 tokens</div>
<h1>Expert cache miss, layer by layer</h1>
<p class="sub">Box 1's routed-expert LRU, replayed against a real pick trace. Plotted as
<em>miss</em> rate, because miss rate is what costs: every one is an 8.85&nbsp;ms blocking read
from dm-crypt. On a hit-rate axis this whole chart is the span 98.2–99.2% and looks flat.</p>
</header>

<div class="stats">
<div class="stat"><div class="v">2.19&times;</div><div class="k">spread between best and worst layer<br>0.83% (L1) → 1.83% (L31)</div></div>
<div class="stat"><div class="v">614</div><div class="k">chi-square vs a constant rate<br>on 39 dof — ~39 if it were noise</div></div>
<div class="stat"><div class="v">+0.80</div><div class="k">split-half correlation<br>the pattern replicates</div></div>
<div class="stat"><div class="v">+0.04</div><div class="k">correlation with depth<br>it is not a depth trend</div></div>
</div>

<section class="panel">
<h2>Per-layer miss rate</h2>
<p class="cap">Bars are the full trace. The two horizontal ticks on each bar are the first and
second half simulated independently — how tightly they straddle the bar is the replication.
Highlighted layers are the worst four.</p>
<div class="scroll">{chart1}</div>
<div class="key">
<span><i style="background:var(--accent)"></i>miss rate, full trace</span>
<span><i style="background:var(--flag)"></i>worst four (L{', L'.join(str(w) for w in sorted(worst))})</span>
<span><i style="background:var(--ink);opacity:.45"></i>first / second half, simulated separately</span>
</div>
</section>

<section class="panel">
<h2>What explains it — nothing obvious</h2>
<p class="cap">Correlation of each candidate against the 40 per-layer miss rates. Everything
mechanical comes back near zero, including our own hash partition and layer depth. Only the
split-half check is strong, which says the pattern is real but unexplained.</p>
<div class="scroll">{chart2}</div>
</section>

<section class="panel">
<h2>The worst layers</h2>
<p class="cap">Cold share is the fraction of misses that are a first-ever touch of that
(layer, expert) pair — irreducible at any capacity.</p>
<table><thead><tr><th>layer</th><th class="n">miss %</th><th class="n">misses</th>
<th class="n">accesses</th><th class="n">cold %</th></tr></thead><tbody>{tbl}</tbody></table>
</section>

<section class="panel verdict">
<h2>What it is not good for</h2>
<p style="margin:8px 0 0"><strong>Rebalancing the partition does not work.</strong> Giving box 2 a
larger share of the layers box 1 misses on was tested with per-layer thresholds fitted on the
first half and scored on the second: box-1 misses got <strong>4.6% worse</strong>, total 0.2% worse
— and worse in-sample too, so it is the wrong mechanism rather than overfitting. Moving an expert
across the link does not change its reuse distance, and both boxes run LRU over their own streams,
so a layer with poor locality misses wherever it lives.</p>
<p style="margin:14px 0 0"><strong>Capacity is the untested one.</strong> Reserving slots per layer
inside box 1 is a different mechanism from reserving <em>ownership</em>, and LRU equalises recency
rather than marginal miss-reduction, so there is room in principle. Nobody has measured it.</p>
</section>

<footer>Simulated from <code>picks-20260918-1501.trace</code> against box 1's production geometry:
4,070 decode slots, hash partition at 39.7%, first 2,000 tokens discarded as warm-up.
Per-layer accesses are equal by construction in decode (6 picks every token), which is why these
rates are comparable — the server's live <code>by_layer</code> histogram mixes prefill, where CED
gives layers 0–19 about 80&times; the rows of 20–39, and should not be read as a per-layer signal.</footer>
</main>"""
open("layer_miss.html","w").write(html)
print("wrote", len(html), "bytes")
