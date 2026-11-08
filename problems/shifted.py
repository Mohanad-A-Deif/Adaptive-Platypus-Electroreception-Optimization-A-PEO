"""Shifted variants of the suite — centre-bias control.

Ten of the eighteen functions place the global optimum exactly at the centre of a symmetric
box (F1-F4, F6, F7, F9-F11, F14).  Any operator that contracts toward the population centroid
is rewarded there for a reason unrelated to search quality, so a comparison restricted to
this suite cannot separate the two effects.

Only those ten functions are translated, by z = x - o with o drawn once per function.  The
remaining eight already have an off-centre optimum (F5 at 1, F8 at 420.97, F12 at -1, F13 at
+1, F15-F18 at irregular points) and are left untouched; translating them would be pointless
and, for F8, invalid, because Schwefel 2.26 is unbounded below outside its box so f_min would
no longer be the global minimum of the translated problem.
"""
import numpy as np
import benchmarks as B

CENTRED = {0, 1, 2, 3, 5, 6, 8, 9, 10, 13}        # indices into B.SUITE whose optimum is at 0

def make_shifted(seed=20261026, frac=0.5):
    rng = np.random.default_rng(seed)
    out = []
    for i, (name, f, L, U, D, fmin) in enumerate(B.SUITE):
        if i not in CENTRED:
            out.append((name, f, L, U, D, fmin))
            continue
        o = rng.uniform(-frac, frac, D) * (U - L) / 2
        def g(X, r=None, f=f, o=o):
            return f(X - o, r) if getattr(f, "__name__", "") == "F7" else f(X - o)
        g.__name__ = f.__name__
        out.append((name, g, L, U, D, fmin))
    return out

SUITE_S = make_shifted()
