"""Figure 4 as a greyscale heat map: degradation ratio per algorithm and function."""
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
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 3,
    "ytick.major.size": 3, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 600, "savefig.dpi": 600, "figure.facecolor": "white",
    "axes.facecolor": "white", "mathtext.fontset": "cm"})

R = json.load(open("../results/compare.json"))
ALGS = ["LSHADE", "A-PEO", "DE", "PSO", "SMA", "GWO"]
LAB = {"A-PEO": "A-PEO", "PSO": "PSO",
       "DE": "DE", "GWO": "GWO", "SMA": "SMA", "LSHADE": "L-SHADE"}
NAMES = [s[0] for s in B.SUITE]
sel = [NAMES[i] for i in sorted(CENTRED)]
rn = lambda su, fn, a: np.array(R[f"{su}|{fn}|{a}"]["runs"]).mean()
FLOOR = 1e-30

M = np.array([[max(rn("shifted", f, a), FLOOR) / max(rn("classic", f, a), FLOOR)
               for f in sel] for a in ALGS])
Lg = np.log10(M)
order = np.argsort(np.median(Lg, 1))
Lg, labs = Lg[order], [LAB[ALGS[i]] for i in order]

fig, ax = plt.subplots(figsize=(7.2, 3.0))
vmax = 12
im = ax.imshow(np.clip(Lg, 0, vmax), cmap="Greys", vmin=0, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(sel))); ax.set_xticklabels([s.split()[0] for s in sel])
ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs)
for i in range(len(labs)):
    for j in range(len(sel)):
        v = Lg[i, j]
        txt = "$\\leq$1" if v <= 0.05 else (">12" if v > vmax else "%.0f" % v)
        ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                color="white" if min(v, vmax) > 0.55 * vmax else "black")
ax.set_xticks(np.arange(-.5, len(sel), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(labs), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.8); ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
cb.set_label("$\\log_{10}$ (error ratio)", fontsize=9); cb.ax.tick_params(labelsize=8)
ax.set_title("degradation when the optimum is moved off the centre of the box", fontsize=10, pad=6)
fig.tight_layout()
fig.savefig("../figures/fig4_shift_ratio.png", dpi=600, bbox_inches="tight")
print("fig4 rewritten")
