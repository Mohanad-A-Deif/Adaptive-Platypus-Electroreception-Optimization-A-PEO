"""Run one PEO configuration over the whole 18-function suite and store the statistics."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import json, os, sys, time
import numpy as np
import benchmarks as B
from peo import peo

# PEO means reported in the source manuscript (Tables tab:bench_unimodal / multimodal / fixed)
PAPER_MEAN = [0.019476, 0.016588, 0.013038, 0.006088, 0.002819, 0.008374, 0.011008,
              2.041272, 1.914360, 1.728408, 1.656244, 2.043152, 1.010664, 1.358251,
              0.084871, 0.197703, 0.108536, 0.166781]
PAPER_BEST = [0.000437, 0.000973, 0.000128, 0.000917, 0.000678, 0.000365, 0.000147,
              0.256537, 0.060531, 0.390802, 0.352099, 0.264880, 0.092778, 0.052591,
              0.027097, 0.006728, 0.035655, 0.004287]

CONFIGS = {
    "literal":      dict(),
    "relative":     dict(relative_scale=True),
    "decay":        dict(decay="exp"),
    "rel+decay":    dict(relative_scale=True, decay="exp"),
    "rel+decay+gr": dict(relative_scale=True, decay="exp", greedy=True),
    "decay+greedy": dict(decay="exp", greedy=True),
    "greedy":       dict(greedy=True),
    # the adaptive variant and the ablation that isolates the per-agent memory
    "A-PEO":        dict(relative_scale=True, decay="exp", decay_floor=1e-6,
                         esens="normalised", switch="median", mu=0.9),
    "A-PEO no-pmem": dict(relative_scale=True, decay="exp", decay_floor=1e-6,
                          esens="normalised", switch="median", mu=0.0),
    # ablation that decides whether any of PEO's own identity survives the adaptive components:
    # alpha = 0 deletes the directional-sensing operator E of Eq. (6) entirely.
    "A-PEO no-E":   dict(relative_scale=True, decay="exp", decay_floor=1e-6,
                         esens="normalised", switch="median", mu=0.9, alpha=0.0),
}

def main(cfg_name, out="results.json", R=30):
    kw = CONFIGS[cfg_name]
    res = json.load(open(out)) if os.path.exists(out) else {}
    for idx, (name, f, L, U, D, fmin) in enumerate(B.SUITE):
        key = f"{cfg_name}|{name}"
        if key in res:
            continue
        FE = 10_000 * D                      # paper: MaxFEs = 1e4 x D
        t0 = time.time()
        G = peo(f, L, U, D, N=30, FE=FE, R=R, seed=1000 + idx, **kw)
        err = G - fmin                       # the paper's tables report error, not raw f
        res[key] = dict(best=float(err.min()), mean=float(err.mean()),
                        worst=float(err.max()), std=float(err.std(ddof=1)),
                        secs=round(time.time() - t0, 1), paper_mean=PAPER_MEAN[idx])
        json.dump(res, open(out, "w"), indent=1)
        print("%-14s %-18s best %10.4g  mean %10.4g  worst %10.4g   paper %8.4g  (%.0fs)"
              % (cfg_name, name, err.min(), err.mean(), err.max(), PAPER_MEAN[idx],
                 time.time() - t0), flush=True)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "literal")
