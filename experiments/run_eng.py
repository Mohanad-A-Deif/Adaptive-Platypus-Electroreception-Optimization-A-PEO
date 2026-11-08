"""Five engineering design problems x 7 algorithms x 30 runs, feasibility-first."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import json, os, time
import numpy as np
import engineering as E
from constrained import peo_c, pso_c, de_c, gwo_c, sma_c, lshade_c

ALGS = {
    "A-PEO":     lambda p, L, U, D, FE, R, s: peo_c(p, L, U, D, FE=FE, R=R, seed=s),
    "PEO (base formulation)": lambda p, L, U, D, FE, R, s: peo_c(p, L, U, D, FE=FE, R=R, seed=s,
                                                               adaptive=False),
    "PSO":    lambda p, L, U, D, FE, R, s: pso_c(p, L, U, D, FE=FE, R=R, seed=s),
    "DE":     lambda p, L, U, D, FE, R, s: de_c(p, L, U, D, FE=FE, R=R, seed=s),
    "GWO":    lambda p, L, U, D, FE, R, s: gwo_c(p, L, U, D, FE=FE, R=R, seed=s),
    "SMA":    lambda p, L, U, D, FE, R, s: sma_c(p, L, U, D, FE=FE, R=R, seed=s),
    "LSHADE": lambda p, L, U, D, FE, R, s: lshade_c(p, L, U, D, FE=FE, R=R, seed=s),
}
OUT = "engineering.json"
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for i, (name, fn) in enumerate(E.PROBLEMS):
    L, U = fn.bounds; D = len(L)
    for an, alg in ALGS.items():
        key = f"{name}|{an}"
        if key in res:
            continue
        t0 = time.time()
        f, v = alg(fn, L, U, D, 10_000 * D, 30, 6000 + i)
        ok = v <= 0
        res[key] = dict(feasible=int(ok.sum()), runs=[float(x) for x in f],
                        viol=[float(x) for x in v],
                        best=float(f[ok].min()) if ok.any() else None,
                        mean=float(f[ok].mean()) if ok.any() else None,
                        std=float(f[ok].std(ddof=1)) if ok.sum() > 1 else None,
                        ref=fn.ref, secs=round(time.time() - t0, 1))
        json.dump(res, open(OUT, "w"))
        print("%-28s %-20s feas %2d/30  best %12.6g  mean %12.6g  (%.0fs)"
              % (name, an, ok.sum(),
                 res[key]["best"] if res[key]["best"] is not None else float("nan"),
                 res[key]["mean"] if res[key]["mean"] is not None else float("nan"),
                 time.time() - t0), flush=True)
print("DONE")
