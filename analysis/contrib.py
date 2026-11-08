"""How large is the directional-sensing term E relative to the other two terms of Eq. (9),
inside the REPAIRED algorithm?  Measured over a full run."""
import sys, os; sys.path[:0] = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", d) for d in ("", "problems")]
import numpy as np, benchmarks as B, peo as _p

def trace(f, L, U, D, R=5, seed=11, mu=0.9, flo=1e-6):
    rng = np.random.default_rng(seed); span = float(U - L); FE = 10_000 * D; T = FE // N if False else FE // 30
    N = 30; alpha, beta, gamma, lam, zeta, eps = 1.5, 1.0, 0.5, 0.8, 0.9, 1e-6
    b0, z0 = beta * span / 2, zeta * span / 2
    X = L + rng.random((R, N, D)) * span
    Fv = f(X, rng) if _p._noisy(f) else f(X)
    P = X.copy(); FP = Fv.copy()
    ri = np.arange(R); k = Fv.argmin(1); g = X[ri, k].copy(); G = Fv[ri, k].copy()
    Fprev = None; acc = []
    for t in range(T):
        m = max(flo ** (t / T), flo); b, z = b0 * m, z0 * m
        if Fprev is None: s = np.zeros((R, N), bool)
        else:
            d = np.abs(Fv - Fprev); s = d > np.median(d, 1, keepdims=True)
        j = (np.arange(N) + rng.integers(1, N, (R, N))) % N
        Xj = np.take_along_axis(X, j[..., None], 1); Fj = np.take_along_axis(Fv, j, 1)
        dX = Xj - X
        spread = Fv.max(1, keepdims=True) - Fv.min(1, keepdims=True) + eps
        E = -alpha * ((Fj - Fv) / spread)[..., None] * dX
        M = b * (2 * rng.random((R, N, D)) - 1)
        tgt = (1 - mu) * g[:, None] + mu * P
        Rt = gamma * (tgt - X)
        ne, nm, nr = (np.linalg.norm(E, axis=-1), np.linalg.norm(M, axis=-1), np.linalg.norm(Rt, axis=-1))
        expl = ~s
        if expl.any():
            acc.append([ne[expl].mean(), nm[expl].mean(), nr[expl].mean()])
        Xn = np.clip(np.where(s[..., None], X + lam * (tgt - X) + z * rng.standard_normal((R, N, D)),
                              X + E + M + Rt), L, U)
        Fn = f(Xn, rng) if _p._noisy(f) else f(Xn)
        imp = Fn < FP; P = np.where(imp[..., None], Xn, P); FP = np.where(imp, Fn, FP)
        Fprev, X, Fv = Fv, Xn, Fn
        k = Fv.argmin(1); bt = Fv[ri, k] < G; g[bt] = X[ri, k][bt]; G[bt] = Fv[ri, k][bt]
    a = np.array(acc)
    return a.mean(0)

print("%-18s %10s %10s %10s %10s" % ("function", "||E||", "||M||", "||R||", "E share"))
for i in (0, 4, 8, 9, 13):
    nm, f, L, U, D, fmin = B.SUITE[i]
    e, m, r = trace(f, L, U, D)
    print("%-18s %10.4g %10.4g %10.4g %9.1f%%" % (nm, e, m, r, 100 * e / (e + m + r)), flush=True)
