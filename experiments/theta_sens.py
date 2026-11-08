"""Sensitivity of the published switching rule to theta.

Eq. (13) compares |dF| with an absolute threshold.  Because |dF| spans many orders of
magnitude across the suite, the behaviour of the rule depends on the ratio of theta to the
scale of the objective rather than on theta alone.  This script measures, for a grid of
theta, both the fraction of agents that end up in the exploitation branch and the resulting
error, using the published formulation otherwise unchanged."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import json
import numpy as np
import benchmarks as B
import peo as _p

def run(f, L, U, D, theta, N=30, FE=None, R=10, seed=77):
    FE = FE or 10_000 * D
    rng = np.random.default_rng(seed); span = float(U - L); T = FE // N
    alpha, beta, gamma, lam, zeta, eps = 1.5, 1.0, 0.5, 0.8, 0.9, 1e-6
    X = L + rng.random((R, N, D)) * span
    Fv = _p._noisy(f) and f(X, rng) or f(X)
    ri = np.arange(R); k = Fv.argmin(1); g = X[ri, k].copy(); G = Fv[ri, k].copy()
    Fprev = None; frac = []
    for t in range(T):
        s = np.zeros((R, N), bool) if Fprev is None else (np.abs(Fv - Fprev) > theta)
        frac.append(s.mean())
        j = (np.arange(N) + rng.integers(1, N, (R, N))) % N
        Xj = np.take_along_axis(X, j[..., None], 1); Fj = np.take_along_axis(Fv, j, 1)
        dX = Xj - X
        E = -alpha * ((Fj - Fv) / (np.linalg.norm(dX, axis=-1) + eps))[..., None] * dX
        Xn = np.clip(np.where(s[..., None],
                              X + lam * (g[:, None] - X) + zeta * rng.standard_normal((R, N, D)),
                              X + E + beta * (2 * rng.random((R, N, D)) - 1) + gamma * (g[:, None] - X)),
                     L, U)
        Fn = f(Xn, rng) if _p._noisy(f) else f(Xn)
        Fprev, X, Fv = Fv, Xn, Fn
        k = Fv.argmin(1); b = Fv[ri, k] < G; g[b] = X[ri, k][b]; G[b] = Fv[ri, k][b]
    return float(np.mean(frac)), G

TH = [1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6, 1e8]
out = {}
print("%-18s %8s %12s %12s" % ("function", "theta", "exploit frac", "mean error"))
for i in (0, 8, 9):
    nm, f, L, U, D, fmin = B.SUITE[i]
    for th in TH:
        fr, G = run(f, L, U, D, th)
        out[f"{nm}|{th}"] = dict(frac=fr, mean=float((G - fmin).mean()))
        print("%-18s %8.0e %11.1f%% %12.4g" % (nm, th, 100 * fr, (G - fmin).mean()), flush=True)
json.dump(out, open("theta_sens.json", "w"), indent=1)
