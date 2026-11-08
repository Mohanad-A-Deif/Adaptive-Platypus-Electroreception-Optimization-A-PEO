"""Figures for the improvement paper.
Strict Nature/IEEE greyscale: white background, serif, inward ticks, dotted grid,
series separated by line style and marker only, 600 dpi, tight_layout."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import benchmarks as B
from shifted import CENTRED

plt.rcParams.update({
    "font.family": "cmr10", "axes.unicode_minus": False, "font.size": 11,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.labelsize": 11,
    "axes.titlesize": 11, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4, "ytick.major.size": 4, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "legend.fontsize": 9, "legend.frameon": False,
    "grid.alpha": 0.25, "grid.linestyle": ":", "grid.linewidth": 0.6, "grid.color": "black",
    "figure.dpi": 600, "savefig.dpi": 600, "figure.facecolor": "white",
    "axes.facecolor": "white", "mathtext.fontset": "cm"})
BOX = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0)

R = json.load(open("../results/compare.json"))
ALGS = ["A-PEO", "PSO", "DE", "GWO", "SMA", "LSHADE"]
LAB = {"A-PEO": "A-PEO", "PSO": "PSO",
       "DE": "DE", "GWO": "GWO", "SMA": "SMA", "LSHADE": "L-SHADE"}
NAMES = [s[0] for s in B.SUITE]
SHIFT = {"F15 Foxholes": 0.001996}
rn = lambda su, fn, a: np.array(R[f"{su}|{fn}|{a}"]["runs"]) + SHIFT.get(fn, 0.0)

# ---------- figure 3: mean rank, centred vs translated -----------------------
def ranks(su):
    M = np.array([[rn(su, f, a).mean() for a in ALGS] for f in NAMES])
    return (np.argsort(np.argsort(M, 1), 1) + 1).mean(0)
rc, rs = ranks("classic"), ranks("shifted")
o = np.argsort(rs)
x = np.arange(len(ALGS)); w = 0.38
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.bar(x - w/2, rc[o], w, facecolor="0.80", edgecolor="black", lw=0.6, label="centred suite")
ax.bar(x + w/2, rs[o], w, facecolor="white", edgecolor="black", lw=0.6, hatch="///",
       label="translated suite")
for k in range(len(ALGS)):
    d = rs[o][k] - rc[o][k]
    if abs(d) >= 0.4:
        ax.annotate("%+.1f" % d, (x[k] + w/2, rs[o][k]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([LAB[ALGS[i]] for i in o])
ax.set_ylabel("mean Friedman rank"); ax.grid(True, axis="y"); ax.set_ylim(0, 7.4)
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig("../figures/fig3_ranks.png", dpi=600, bbox_inches="tight"); plt.close(fig)
print("fig3 written")

# ---------- figure 4: per-function degradation ratio -------------------------
sel = [NAMES[i] for i in sorted(CENTRED)]
FLOOR = 1e-30
M = np.array([[max(rn("shifted", f, a).mean(), FLOOR) / max(rn("classic", f, a).mean(), FLOOR)
               for f in sel] for a in ALGS])
M = np.clip(M, 1e-2, 1e12)                      # cap so the panel stays readable
order = np.argsort(np.median(M, 1))
sty = [dict(marker="o", ls="-"), dict(marker="s", ls="--"), dict(marker="^", ls=":"),
       dict(marker="d", ls="-."), dict(marker="v", ls=(0, (3, 1, 1, 1))),
       dict(marker="<", ls=(0, (1, 1))), dict(marker=">", ls=(0, (5, 1)))]
grey = ["black", "0.25", "0.4", "0.5", "0.6", "0.7", "0.8"]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
xx = np.arange(len(sel))
for rank, i in enumerate(order):
    ax.plot(xx, M[i], color=grey[rank], lw=1.0, ms=4.0, mfc="none", mew=0.8,
            label=LAB[ALGS[i]], **sty[rank])
ax.axhline(1.0, color="0.3", lw=0.7, ls=(0, (1, 3)))
ax.set_yscale("log"); ax.grid(True)
ax.set_xticks(xx); ax.set_xticklabels([s.split()[0] for s in sel], rotation=0)
ax.set_ylabel("error ratio  (translated / centred)")
ax.text(0.02, 0.95, "values above the dotted line indicate degradation", transform=ax.transAxes,
        va="top", fontsize=8, bbox=BOX)
fig.legend(loc="upper center", bbox_to_anchor=(0.54, 1.10), ncol=7)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("../figures/fig4_shift_ratio.png", dpi=600, bbox_inches="tight"); plt.close(fig)
print("fig4 written")
