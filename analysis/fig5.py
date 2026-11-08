"""Figure 5: behaviour of the published switching rule as a function of theta."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "cmr10", "axes.unicode_minus": False, "font.size": 11,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.labelsize": 11,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 4,
    "ytick.major.size": 4, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False, "grid.alpha": 0.25,
    "grid.linestyle": ":", "grid.linewidth": 0.6, "grid.color": "black",
    "figure.dpi": 600, "savefig.dpi": 600, "figure.facecolor": "white",
    "axes.facecolor": "white", "mathtext.fontset": "cm"})
BOX = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.0)

d = json.load(open("../results/theta_sens.json"))
TH = [1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]
FN = ["F1  Sphere", "F9  Rastrigin", "F10 Ackley"]
sty = [dict(marker="o", ls="-", color="black"),
       dict(marker="s", ls="--", color="0.45"),
       dict(marker="^", ls=":", color="0.65")]

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
for k, fn in enumerate(FN):
    ax[0].plot(TH, [100 * d[f"{fn}|{t}"]["frac"] for t in TH], lw=1.0, ms=3.5,
               mfc="none", mew=0.8, label=fn.split()[0], **sty[k])
    ax[1].plot(TH, [d[f"{fn}|{t}"]["mean"] for t in TH], lw=1.0, ms=3.5,
               mfc="none", mew=0.8, **sty[k])
ax[0].axvline(1e-4, color="0.4", lw=0.7, ls=(0, (1, 3)))
ax[1].axvline(1e-4, color="0.4", lw=0.7, ls=(0, (1, 3)))
for a in ax:
    a.annotate(r"published $\theta$", (1e-4, 0.5), xycoords=("data", "axes fraction"),
               xytext=(5, 0), textcoords="offset points", fontsize=8, rotation=90, va="center")
for a in ax:
    a.set_xscale("log"); a.grid(True); a.set_xlabel(r"switching threshold $\theta$")
ax[0].set_ylabel("agents in exploitation (%)"); ax[0].set_ylim(-5, 112)
ax[0].legend(loc="lower left", bbox_to_anchor=(0.02, 0.06))
ax[1].set_yscale("log"); ax[1].set_ylabel("mean error  $f-f^{*}$")
ax[0].text(0.97, 0.93, "(a)", transform=ax[0].transAxes, ha="right", va="top", bbox=BOX)
ax[1].text(0.97, 0.93, "(b)", transform=ax[1].transAxes, ha="right", va="top", bbox=BOX)
fig.tight_layout()
fig.savefig("../figures/fig5_theta.png", dpi=600, bbox_inches="tight")
print("fig5 written")
