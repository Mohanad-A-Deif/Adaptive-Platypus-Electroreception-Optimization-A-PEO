"""Tables and statistics for the seven-algorithm comparison."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import json
import numpy as np
from scipy.stats import ranksums, friedmanchisquare
import benchmarks as B

R = json.load(open("../results/compare.json"))
ALGS = ["A-PEO", "PEO (base formulation)", "PSO", "DE", "GWO", "SMA", "LSHADE"]
SHORT = {"A-PEO": "A-PEO", "PEO (base formulation)": "PEO", "PSO": "PSO",
         "DE": "DE", "GWO": "GWO", "SMA": "SMA", "LSHADE": "LSHADE"}
NAMES = [s[0] for s in B.SUITE]
SHIFT = {"F15 Foxholes": 0.001996}          # the paper lists f_min = 1; the true value is 0.998004

def runs(suite, fn, alg):
    return np.array(R[f"{suite}|{fn}|{alg}"]["runs"]) + SHIFT.get(fn, 0.0)

def holm(p):
    o = np.argsort(p); adj = np.empty_like(np.asarray(p, float)); run = 0.0
    for k, i in enumerate(o):
        run = max(run, (len(p) - k) * p[i]); adj[i] = min(run, 1.0)
    return adj

def table(suite):
    print(f"\n==================  {suite.upper()} SUITE  —  mean error over 30 runs  ==================")
    print("%-18s" % "function" + "".join("%12s" % SHORT[a] for a in ALGS))
    ranks = []
    for fn in NAMES:
        m = [runs(suite, fn, a).mean() for a in ALGS]
        ranks.append(np.argsort(np.argsort(m)) + 1)
        best = int(np.argmin(m))
        cells = "".join(("*%11.3g" if k == best else "%12.3g") % v for k, v in enumerate(m))
        print("%-18s" % fn + cells)
    ranks = np.array(ranks)
    print("%-18s" % "mean rank" + "".join("%12.2f" % ranks[:, i].mean() for i in range(len(ALGS))))
    chi = friedmanchisquare(*[ranks[:, i] for i in range(len(ALGS))])
    print("Friedman: chi2 = %.2f, p = %.3g" % (chi.statistic, chi.pvalue))

    # Wilcoxon rank-sum of A-PEO against each competitor, Holm-corrected over 6x18 comparisons
    ps, sgn = [], []
    for a in ALGS[1:]:
        for fn in NAMES:
            x, y = runs(suite, fn, ALGS[0]), runs(suite, fn, a)
            ps.append(ranksums(x, y).pvalue); sgn.append(np.mean(x) < np.mean(y))
    adj = holm(ps).reshape(len(ALGS) - 1, len(NAMES))
    sgn = np.array(sgn).reshape(len(ALGS) - 1, len(NAMES))
    print("\nPEO-R vs each competitor (Wilcoxon rank-sum, Holm over %d comparisons):" % len(ps))
    print("%-22s %6s %6s %6s" % ("competitor", "win", "loss", "tie"))
    for k, a in enumerate(ALGS[1:]):
        w = int(((adj[k] < .05) & sgn[k]).sum()); l = int(((adj[k] < .05) & ~sgn[k]).sum())
        print("%-22s %6d %6d %6d" % (a, w, l, 18 - w - l))
    return ranks

def shift_sensitivity():
    """Degradation under translation, computed ONLY over the ten functions that were
    actually shifted; the other eight are identical in both suites and would otherwise
    pin the median at exactly 1."""
    from shifted import CENTRED
    sel = [NAMES[i] for i in sorted(CENTRED)]
    print("\n==================  SHIFT SENSITIVITY (the %d translated functions)  =================="
          % len(sel))
    print("ratio = shifted mean error / classic mean error;  a floor of 1e-30 is applied to both")
    print("%-22s %10s %10s %10s %10s" % ("algorithm", "median", "geo-mean", "max", "worsened"))
    FLOOR = 1e-30
    rows = []
    for a in ALGS:
        rr = []
        for fn in sel:
            c = max(runs("classic", fn, a).mean(), FLOOR)
            s_ = max(runs("shifted", fn, a).mean(), FLOOR)
            rr.append(s_ / c)
        rr = np.array(rr)
        wors = int((rr > 10).sum())
        rows.append((a, np.median(rr), float(np.exp(np.mean(np.log(rr)))), rr.max(), wors, rr))
    for a, med, geo, mx, w, _ in sorted(rows, key=lambda r: r[1]):
        print("%-22s %10.3g %10.3g %10.3g %7d/%d" % (a, med, geo, mx, w, len(sel)))
    print("  median and 'worsened' (ratio > 10) are the robust summaries; the geometric mean and")
    print("  max are reported for completeness but are dominated by functions where an algorithm")
    print("  scores exactly 0 on the centred version, which is the artefact under study.")
    print("\n  per-function ratios")
    print("%-18s" % "function" + "".join("%12s" % SHORT[a] for a in ALGS))
    for k, fn in enumerate(sel):
        print("%-18s" % fn + "".join("%12.3g" % r[5][k] for r in rows if True) )
    return rows

if __name__ == "__main__":
    have = {k.split("|")[0] for k in R}
    for s in ("classic", "shifted"):
        if s in have and sum(1 for k in R if k.startswith(s)) == 18 * len(ALGS):
            table(s)
    if {"classic", "shifted"} <= have and len(R) == 2 * 18 * len(ALGS):
        shift_sensitivity()
