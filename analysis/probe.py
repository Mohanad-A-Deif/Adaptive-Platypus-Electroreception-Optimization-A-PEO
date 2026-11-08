"""Short diagnostic probe: population diversity and exploitation fraction over a run."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import numpy as np, benchmarks as B
from peo import peo
import peo as _p

def instrumented(f, L, U, D, N=30, FE=300_000, R=5, seed=7, **kw):
    """Re-runs the loop of peo() with two extra traces."""
    rng = np.random.default_rng(seed); span = float(U - L); T = FE // N
    rel = kw.get("relative_scale", False); dec = kw.get("decay"); flo = kw.get("decay_floor", 1e-3)
    alpha, beta, gamma, lam, zeta, eps, theta = 1.5, 1.0, 0.5, 0.8, 0.9, 1e-6, 1e-4
    b0 = beta * span / 2 if rel else beta; z0 = zeta * span / 2 if rel else zeta
    X = L + rng.random((R, N, D)) * span
    Fv = f(X, rng) if _p._noisy(f) else f(X)
    ri = np.arange(R); k = Fv.argmin(1); g = X[ri, k].copy(); G = Fv[ri, k].copy()
    Fprev = None; trace = []
    for t in range(T):
        m = (max(1 - t / T, flo) if dec == "linear" else max(flo ** (t / T), flo) if dec == "exp" else 1.0)
        b, z = b0 * m, z0 * m
        s = np.zeros((R, N), bool) if Fprev is None else (np.abs(Fv - Fprev) > theta)
        j = (np.arange(N) + rng.integers(1, N, (R, N))) % N
        Xj = np.take_along_axis(X, j[..., None], 1); Fj = np.take_along_axis(Fv, j, 1)
        dX = Xj - X
        E = -alpha * ((Fj - Fv) / (np.linalg.norm(dX, axis=-1) + eps))[..., None] * dX
        Xn = np.clip(np.where(s[..., None],
                              X + lam * (g[:, None] - X) + z * rng.standard_normal((R, N, D)),
                              X + E + b * (2 * rng.random((R, N, D)) - 1) + gamma * (g[:, None] - X)), L, U)
        Fn = f(Xn, rng) if _p._noisy(f) else f(Xn)
        Fprev, X, Fv = Fv, Xn, Fn
        k = Fv.argmin(1); bt = Fv[ri, k] < G; g[bt] = X[ri, k][bt]; G[bt] = Fv[ri, k][bt]
        if t % max(1, T // 10) == 0:
            trace.append((t / T, float(s.mean()), float(np.mean(X.std(1))), float(np.median(G))))
    return trace

for nm, f, L, U, D, fmin in [B.SUITE[i] for i in (0, 5, 7, 8, 13)]:
    print("\n%s   (domain width %g)" % (nm, U - L))
    print("  t/T   exploit%   mean pop. std   median best-so-far")
    for a, b, c, d in instrumented(f, L, U, D):
        print("  %.1f    %6.1f%%   %12.4g   %18.6g" % (a, 100 * b, c, d - fmin))
