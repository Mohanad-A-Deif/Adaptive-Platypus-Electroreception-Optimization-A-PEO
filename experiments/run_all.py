"""Full comparison: 7 algorithms x 18 functions x {classic, shifted} x 30 runs.
Checkpointed after every cell, so the job can be interrupted and restarted."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import json, os, sys, time
import numpy as np
import benchmarks as B
from shifted import SUITE_S
from peo import peo
from baselines import pso, de, gwo, sma, lshade

REP = dict(relative_scale=True, decay="exp", decay_floor=1e-6,
           esens="normalised", switch="median", mu=0.9)
ALGS = {
    "PEO (base formulation)": lambda f, L, U, D, FE, R, s: peo(f, L, U, D, FE=FE, R=R, seed=s),
    "A-PEO":     lambda f, L, U, D, FE, R, s: peo(f, L, U, D, FE=FE, R=R, seed=s, **REP),
    "PSO":    lambda f, L, U, D, FE, R, s: pso(f, L, U, D, FE=FE, R=R, seed=s),
    "DE":     lambda f, L, U, D, FE, R, s: de(f, L, U, D, FE=FE, R=R, seed=s),
    "GWO":    lambda f, L, U, D, FE, R, s: gwo(f, L, U, D, FE=FE, R=R, seed=s),
    "SMA":    lambda f, L, U, D, FE, R, s: sma(f, L, U, D, FE=FE, R=R, seed=s),
    "LSHADE": lambda f, L, U, D, FE, R, s: lshade(f, L, U, D, FE=FE, R=R, seed=s),
}
OUT = "compare.json"

def main(which):
    suite = B.SUITE if which == "classic" else SUITE_S
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for i, (name, f, L, U, D, fmin) in enumerate(suite):
        for an, fn in ALGS.items():
            key = f"{which}|{name}|{an}"
            if key in res:
                continue
            t0 = time.time()
            G = fn(f, L, U, D, 10_000 * D, 30, 5000 + i)
            e = np.asarray(G) - fmin
            res[key] = dict(best=float(e.min()), mean=float(e.mean()), median=float(np.median(e)),
                            worst=float(e.max()), std=float(e.std(ddof=1)),
                            runs=[float(v) for v in e], secs=round(time.time() - t0, 1))
            json.dump(res, open(OUT, "w"))
            print("%-8s %-18s %-20s mean %10.4g  (%.0fs)" % (which, name, an, e.mean(),
                                                             time.time() - t0), flush=True)
    print("DONE", which)

if __name__ == "__main__":
    main(sys.argv[1])
