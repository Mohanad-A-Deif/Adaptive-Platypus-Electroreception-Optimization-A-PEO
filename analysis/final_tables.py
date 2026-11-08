"""LaTeX tables for the improvement paper."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import json
import numpy as np
from scipy.stats import ranksums, friedmanchisquare
import benchmarks as B
from shifted import CENTRED

R = json.load(open("../results/compare.json"))
ALGS = ["A-PEO", "PSO", "DE", "GWO", "SMA", "LSHADE"]
HEAD = {"A-PEO": r"\textbf{A-PEO}",
        "PSO": "PSO", "DE": "DE", "GWO": "GWO", "SMA": "SMA", "LSHADE": "L-SHADE"}
NAMES = [s[0] for s in B.SUITE]
SHIFT = {"F15 Foxholes": 0.001996}
tidy = lambda n: n.replace("  ", " ").replace("_", " ")

def runs(su, fn, a): return np.array(R[f"{su}|{fn}|{a}"]["runs"]) + SHIFT.get(fn, 0.0)
def num(v): return "0" if v <= 0 else "%.2e" % v

def holm(p):
    o = np.argsort(p); adj = np.empty(len(p)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(run, 1.0)
    return adj

def main_table(su, label, cap):
    L = [r"\begin{table*}[htbp]", r"\centering", r"\caption{%s}" % cap,
         r"\label{%s}" % label, r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "r" * len(ALGS) + "}", r"\toprule",
         "Function & " + " & ".join(HEAD[a] for a in ALGS) + r" \\", r"\midrule"]
    ranks = []
    for fn in NAMES:
        m = [runs(su, fn, a).mean() for a in ALGS]
        ranks.append(np.argsort(np.argsort(m)) + 1)
        b = int(np.argmin(m))
        L.append(tidy(fn) + " & " + " & ".join(
            (r"\textbf{%s}" % num(v) if k == b else num(v)) for k, v in enumerate(m)) + r" \\")
    ranks = np.array(ranks)
    L += [r"\midrule", "Mean rank & " + " & ".join("%.2f" % ranks[:, i].mean()
          for i in range(len(ALGS))) + r" \\"]
    # win/loss/tie of A-PEO against each competitor
    ps, sg = [], []
    for a in ALGS[1:]:
        for fn in NAMES:
            x, y = runs(su, fn, ALGS[0]), runs(su, fn, a)
            ps.append(ranksums(x, y).pvalue); sg.append(x.mean() < y.mean())
    adj = holm(ps).reshape(len(ALGS) - 1, len(NAMES)); sg = np.array(sg).reshape(adj.shape)
    wl = ["---"]
    for k in range(len(ALGS) - 1):
        w = int(((adj[k] < .05) & sg[k]).sum()); l = int(((adj[k] < .05) & ~sg[k]).sum())
        wl.append("%d/%d/%d" % (w, l, 18 - w - l))
    L += [r"$+/-/\approx$ vs A-PEO & " + " & ".join(wl) + r" \\", r"\bottomrule",
          r"\end{tabular}}", r"\end{table*}"]
    chi = friedmanchisquare(*[ranks[:, i] for i in range(len(ALGS))])
    return "\n".join(L), ranks, chi

def sens_table():
    sel = [NAMES[i] for i in sorted(CENTRED)]
    FLOOR = 1e-30
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Degradation under translation of the optimum, over the ten benchmark "
         r"functions whose optimum lies at the centre of a symmetric box. "
         r"Ratio $=$ shifted mean error $/$ centred mean error; "
         r"``worsened'' counts functions with a ratio above ten.}",
         r"\label{tab:shift}", r"\begin{tabular}{lrrr}", r"\toprule",
         r"Algorithm & Median ratio & Max ratio & Worsened \\", r"\midrule"]
    rows = []
    for a in ALGS:
        rr = np.array([max(runs("shifted", f, a).mean(), FLOOR) /
                       max(runs("classic", f, a).mean(), FLOOR) for f in sel])
        rows.append((a, np.median(rr), rr.max(), int((rr > 10).sum())))
    for a, med, mx, w in sorted(rows, key=lambda r: r[1]):
        fmt = (lambda v: "%.2f" % v) if med < 1e4 else (lambda v: "%.1e" % v)
        L.append("%s & %s & %s & %d/10 \\\\" % (HEAD[a], fmt(med), "%.1e" % mx, w))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)

def ablation_table():
    """Component ablation on the analytical suite, same protocol, 30 runs."""
    r2 = json.load(open("../results/results.json"))
    cols = [("literal", "base"), ("decay", "$+$C2"),
            ("A-PEO no-pmem", "$+$C1,C2,C3,C4"), ("A-PEO no-E", r"$-\,\mathbf{E}$"),
            ("A-PEO", r"\textbf{A-PEO}")]
    L = [r"\begin{table*}[htbp]", r"\centering",
         r"\caption{Component ablation on the analytical suite (mean error over 30 runs). "
         r"Column ``base'' disables every adaptive component; the remaining columns enable them "
         r"cumulatively. Column $-\,\mathbf{E}$ removes the directional-sensing operator from "
         r"the complete configuration, all other components retained.}",
         r"\label{tab:ablation}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "r" * len(cols) + "}",
         r"\toprule", "Function & " + " & ".join(t for _, t in cols) + r" \\", r"\midrule"]
    for fn in NAMES:
        v = [max(r2[f"{c}|{fn}"]["mean"] + SHIFT.get(fn, 0.0), 0.0) for c, _ in cols]
        L.append(tidy(fn) + " & " + " & ".join(num(x) for x in v) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(L)

t1, r1, c1 = main_table("classic", "tab:classic",
    "Mean error over 30 independent runs on the centred benchmark suite "
    "($N=30$, $\\mathrm{MaxFEs}=10^4 D$). Best value per function in bold. "
    "The last row gives wins/losses/ties of A-PEO against each competitor "
    "(Wilcoxon rank-sum, Holm-corrected over 108 comparisons).")
t2, r2_, c2 = main_table("shifted", "tab:shifted",
    "Mean error over 30 independent runs on the translated suite, in which the ten "
    "functions with a centred optimum are evaluated at $z=x-o$. Notation as in "
    "Table~\\ref{tab:classic}.")
open("../paper/tables_paper.tex", "w").write("\n\n".join([t1, t2, sens_table(), ablation_table()]))
print("centred  Friedman chi2=%.2f p=%.3g" % (c1.statistic, c1.pvalue))
print("shifted  Friedman chi2=%.2f p=%.3g" % (c2.statistic, c2.pvalue))
print("ranks centred:", dict(zip([HEAD[a] for a in ALGS], np.round(r1.mean(0), 2))))
print("ranks shifted:", dict(zip([HEAD[a] for a in ALGS], np.round(r2_.mean(0), 2))))
print("written tables_paper.tex")
