"""Table and figure for the engineering design problems."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import sys, os
import json
import numpy as np
from scipy.stats import friedmanchisquare
import engineering as E
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("../results/engineering.json"))
ALGS = ["A-PEO", "PSO", "DE", "GWO", "SMA", "LSHADE"]
S = {"A-PEO": r"\textbf{A-PEO}",
     "PSO": "PSO", "DE": "DE", "GWO": "GWO", "SMA": "SMA", "LSHADE": "L-SHADE"}
PLAIN = {k: v.replace(r"\textbf{", "").replace("}", "") for k, v in S.items()}
BIG = 1e30

def kr(nm, a):
    d = R[f"{nm}|{a}"]; f = np.array(d["runs"]); v = np.array(d["viol"])
    return np.where(v <= 0, f, BIG + v)

# ---------- LaTeX table ----------
L = [r"\begin{table*}[htbp]", r"\centering",
     r"\caption{Constrained engineering design problems: best and mean objective over the "
     r"feasible runs of 30, and the number of feasible runs. All algorithms use the same "
     r"feasibility-first selection rule and the same evaluation budget.}",
     r"\label{tab:engineering}", r"\resizebox{\textwidth}{!}{%",
     r"\begin{tabular}{llrrrr}", r"\toprule",
     r"Problem & Algorithm & Feasible & Best & Mean & Std \\", r"\midrule"]
for nm, fn in E.PROBLEMS:
    bests = [R[f"{nm}|{a}"]["best"] for a in ALGS]
    bb = min(b for b in bests if b is not None)
    for i, a in enumerate(ALGS):
        d = R[f"{nm}|{a}"]
        pn = nm if i == 0 else ""
        if d["best"] is None:
            L.append(f"{pn} & {S[a]} & 0/30 & --- & --- & --- \\\\")
            continue
        bs = "%.6g" % d["best"]
        if abs(d["best"] - bb) < 1e-9 * max(1, abs(bb)):
            bs = r"\textbf{%s}" % bs
        L.append("%s & %s & %d/30 & %s & %.6g & %.2g \\\\"
                 % (pn, S[a], d["feasible"], bs, d["mean"],
                    d["std"] if d["std"] is not None else float("nan")))
    L.append(r"\multicolumn{3}{l}{\textit{best known}} & \multicolumn{3}{l}{%.6g} \\" % fn.ref)
    L.append(r"\midrule")
L[-1] = r"\bottomrule"
L += [r"\end{tabular}}", r"\end{table*}"]
open("../paper/tables_engineering.tex", "w").write("\n".join(L))

# ---------- statistics ----------
ranks = np.array([[np.median(kr(nm, a)) for a in ALGS] for nm, _ in E.PROBLEMS])
ranks = np.argsort(np.argsort(ranks, 1), 1) + 1
chi = friedmanchisquare(*[ranks[:, i] for i in range(len(ALGS))])
feas = {a: sum(R[f"{nm}|{a}"]["feasible"] for nm, _ in E.PROBLEMS) for a in ALGS}
print("mean rank:", {PLAIN[a]: round(ranks[:, i].mean(), 2) for i, a in enumerate(ALGS)})
print("Friedman chi2=%.2f p=%.3g (n=5, low power)" % (chi.statistic, chi.pvalue))
print("feasible runs of 150:", {PLAIN[a]: feas[a] for a in ALGS})

# ---------- figure ----------
plt.rcParams.update({
    "font.family": "cmr10", "axes.unicode_minus": False, "font.size": 11,
    "axes.edgecolor": "black", "axes.linewidth": 0.8, "axes.labelsize": 11,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.size": 4,
    "ytick.major.size": 4, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False, "grid.alpha": 0.25,
    "grid.linestyle": ":", "grid.linewidth": 0.6, "grid.color": "black",
    "figure.dpi": 600, "savefig.dpi": 600, "figure.facecolor": "white",
    "axes.facecolor": "white", "mathtext.fontset": "cm"})
order = np.argsort([ranks[:, i].mean() for i in range(len(ALGS))])
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
xx = np.arange(len(ALGS))
bars = ax[0].bar(xx, [100 * feas[ALGS[i]] / 150 for i in order], 0.6,
                 facecolor="0.80", edgecolor="black", lw=0.6)
for b, i in zip(bars, order):
    ax[0].annotate("%d" % feas[ALGS[i]], (b.get_x() + b.get_width() / 2, b.get_height()),
                   xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8)
ax[0].set_ylabel("feasible runs (%)"); ax[0].set_ylim(0, 108)
ax[0].set_xticks(xx); ax[0].set_xticklabels([PLAIN[ALGS[i]] for i in order], rotation=45,
                                            ha="right")
ax[0].grid(True, axis="y")
# excess over the best known value, median run, log scale
exc = []
for i in order:
    e = []
    for nm, fn in E.PROBLEMS:
        d = R[f"{nm}|{ALGS[i]}"]
        f = np.array(d["runs"]); v = np.array(d["viol"])
        ok = v <= 0
        e.append(max(np.median(f[ok]) / fn.ref - 1, 1e-9) if ok.any() else np.nan)
    exc.append(e)
exc = np.array(exc)
mk = ["o", "s", "^", "d", "v", "<", ">"]
gr = ["black", "0.25", "0.4", "0.5", "0.6", "0.7", "0.8"]
ls = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 1))]
for k in range(len(ALGS)):
    ax[1].plot(np.arange(5), exc[k], color=gr[k], ls=ls[k], marker=mk[k], lw=1.0,
               ms=4.0, mfc="none", mew=0.8, label=PLAIN[ALGS[order[k]]])
ax[1].set_ylim(3e-8, 1.0)
ax[1].set_yscale("log"); ax[1].grid(True)
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["PV", "WB", "TBT", "SR", "TCS"])
ax[1].set_ylabel("relative excess over best known")
fig.legend(loc="upper center", bbox_to_anchor=(0.54, 1.11), ncol=7)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig("../figures/fig6_engineering.png", dpi=600, bbox_inches="tight")
print("fig6 written; tables_engineering.tex written")
