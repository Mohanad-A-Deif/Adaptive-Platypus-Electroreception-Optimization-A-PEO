"""Reference optimisers, re-implemented under one protocol and vectorised over runs.
Every algorithm has the same signature as peo(): (f, L, U, D, N, FE, R, seed) -> (R,) best values.
Settings follow the original descriptions; none was tuned on the benchmark suite."""
import numpy as np
import peo as _p


def _ev(f, X, rng):
    return f(X, rng) if _p._noisy(f) else f(X)


def pso(f, L, U, D, N=30, FE=300_000, R=30, seed=0, c1=2.0, c2=2.0, w0=0.9, w1=0.2):
    """Standard PSO with linearly decreasing inertia and velocity clamped to 20% of the range."""
    rng = np.random.default_rng(seed); span = float(U - L); T = FE // N
    X = L + rng.random((R, N, D)) * span
    V = np.zeros((R, N, D)); vmax = 0.2 * span
    Fv = _ev(f, X, rng); P, FP = X.copy(), Fv.copy()
    ri = np.arange(R); k = FP.argmin(1); g = P[ri, k].copy(); G = FP[ri, k].copy()
    for t in range(T):
        w = w0 + (w1 - w0) * t / T
        V = (w * V + c1 * rng.random((R, N, D)) * (P - X)
             + c2 * rng.random((R, N, D)) * (g[:, None] - X))
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, L, U)
        Fv = _ev(f, X, rng)
        imp = Fv < FP
        P = np.where(imp[..., None], X, P); FP = np.where(imp, Fv, FP)
        k = FP.argmin(1); b = FP[ri, k] < G
        g[b] = P[ri, k][b]; G[b] = FP[ri, k][b]
    return G


def de(f, L, U, D, N=30, FE=300_000, R=30, seed=0, F=0.5, CR=0.9):
    """DE/rand/1/bin."""
    rng = np.random.default_rng(seed); span = float(U - L); T = FE // N
    X = L + rng.random((R, N, D)) * span
    Fv = _ev(f, X, rng)
    for t in range(T):
        key = rng.random((R, N, N))                                    # donors r1,r2,r3
        key[:, np.arange(N), np.arange(N)] = np.inf                    # all distinct and != i
        idx = np.argsort(key, axis=-1)[..., :3]
        pick = lambda i: np.take_along_axis(X, idx[..., i][..., None] * np.ones((1, 1, D), int), 1)
        a, b, c = pick(0), pick(1), pick(2)
        Vt = a + F * (b - c)
        j = rng.integers(0, D, (R, N, 1))
        mask = (rng.random((R, N, D)) < CR) | (np.arange(D) == j)
        Ut = np.clip(np.where(mask, Vt, X), L, U)
        FU = _ev(f, Ut, rng)
        keep = FU <= Fv
        X = np.where(keep[..., None], Ut, X); Fv = np.where(keep, FU, Fv)
    return Fv.min(1)


def gwo(f, L, U, D, N=30, FE=300_000, R=30, seed=0):
    """Grey Wolf Optimizer (Mirjalili et al., 2014)."""
    rng = np.random.default_rng(seed); span = float(U - L); T = FE // N
    X = L + rng.random((R, N, D)) * span
    Fv = _ev(f, X, rng)
    for t in range(T):
        o = np.argsort(Fv, 1)[:, :3]
        lead = np.take_along_axis(X, o[..., None] * np.ones((1, 1, D), int), 1)   # (R,3,D)
        a = 2 - 2 * t / T
        Xn = 0.0
        for i in range(3):
            A = a * (2 * rng.random((R, N, D)) - 1)
            C = 2 * rng.random((R, N, D))
            Dl = np.abs(C * lead[:, i:i + 1] - X)
            Xn = Xn + (lead[:, i:i + 1] - A * Dl)
        X = np.clip(Xn / 3.0, L, U)
        Fv = _ev(f, X, rng)
    return Fv.min(1)


def sma(f, L, U, D, N=30, FE=300_000, R=30, seed=0, z=0.03):
    """Slime Mould Algorithm (Li et al., 2020)."""
    rng = np.random.default_rng(seed); span = float(U - L); T = FE // N
    X = L + rng.random((R, N, D)) * span
    Fv = _ev(f, X, rng)
    ri = np.arange(R); k = Fv.argmin(1); g = X[ri, k].copy(); G = Fv[ri, k].copy()
    for t in range(T):
        o = np.argsort(Fv, 1)
        rank = np.empty_like(o); np.put_along_axis(rank, o, np.arange(N), 1)
        bF = Fv.min(1, keepdims=True); wF = Fv.max(1, keepdims=True)
        S = (bF - Fv) / (bF - wF - 1e-12)
        half = rank < N / 2
        W = 1 + rng.random((R, N)) * np.log1p(S) * np.where(half, 1.0, -1.0)
        a = np.arctanh(np.clip(-(t / T) + 1, -1 + 1e-12, 1 - 1e-12))
        b = 1 - t / T
        vb = rng.uniform(-a, a, (R, N, D)); vc = rng.uniform(-b, b, (R, N, D))
        p = np.tanh(np.abs(Fv - G[:, None]))
        A = np.take_along_axis(X, rng.integers(0, N, (R, N))[..., None] * np.ones((1, 1, D), int), 1)
        Bv = np.take_along_axis(X, rng.integers(0, N, (R, N))[..., None] * np.ones((1, 1, D), int), 1)
        upd = np.where((rng.random((R, N, D)) < p[..., None]),
                       g[:, None] + vb * (W[..., None] * A - Bv), vc * X)
        X = np.where(rng.random((R, N, D)) < z, L + rng.random((R, N, D)) * span, upd)
        X = np.clip(X, L, U)
        Fv = _ev(f, X, rng)
        k = Fv.argmin(1); bt = Fv[ri, k] < G
        g[bt] = X[ri, k][bt]; G[bt] = Fv[ri, k][bt]
    return G


def lshade(f, L, U, D, FE=300_000, R=30, seed=0, N=None, H=6, p=0.11):
    """L-SHADE (Tanabe and Fukunaga, 2014): current-to-pbest/1/bin with success-history
    adaptation of F and CR, an external archive, and linear population-size reduction."""
    rng = np.random.default_rng(seed); span = float(U - L)
    Nini = 18 * D; Nmin = 4
    X = L + rng.random((R, Nini, D)) * span
    Fv = _ev(f, X, rng); fe = Nini
    MF = np.full((R, H), 0.5); MCR = np.full((R, H), 0.5); hk = np.zeros(R, int)
    Ar = [np.zeros((0, D)) for _ in range(R)]
    Ncur = Nini
    while fe + Ncur <= FE:
        Nnext = max(Nmin, round(Nini + (Nmin - Nini) * fe / FE))
        r = rng.integers(0, H, (R, Ncur))
        cr = np.clip(np.take_along_axis(MCR, r, 1) + 0.1 * rng.standard_normal((R, Ncur)), 0, 1)
        mf = np.take_along_axis(MF, r, 1)
        Fc = mf + 0.1 * np.tan(np.pi * (rng.random((R, Ncur)) - 0.5))
        while (Fc <= 0).any():
            Fc = np.where(Fc <= 0, mf + 0.1 * np.tan(np.pi * (rng.random((R, Ncur)) - 0.5)), Fc)
        Fc = np.minimum(Fc, 1.0)
        npb = max(2, int(p * Ncur))
        order = np.argsort(Fv, 1)
        pbi = np.take_along_axis(order, rng.integers(0, npb, (R, Ncur)), 1)   # a random p-best
        pb = np.take_along_axis(X, pbi[..., None] * np.ones((1, 1, D), int), 1)
        r1 = rng.integers(0, Ncur, (R, Ncur))
        x1 = np.take_along_axis(X, r1[..., None] * np.ones((1, 1, D), int), 1)
        x2 = np.empty_like(x1)
        for q in range(R):                                   # union of population and archive
            pool = np.vstack([X[q], Ar[q]])
            x2[q] = pool[rng.integers(0, len(pool), Ncur)]
        V = X + Fc[..., None] * (pb - X) + Fc[..., None] * (x1 - x2)
        V = np.where(V < L, (L + X) / 2, np.where(V > U, (U + X) / 2, V))   # LSHADE repair
        j = rng.integers(0, D, (R, Ncur, 1))
        mask = (rng.random((R, Ncur, D)) < cr[..., None]) | (np.arange(D) == j)
        Ut = np.where(mask, V, X)
        FU = _ev(f, Ut, rng); fe += Ncur
        succ = FU < Fv
        for q in range(R):
            if succ[q].any():
                Ar[q] = np.vstack([Ar[q], X[q][succ[q]]])
                if len(Ar[q]) > Ncur:
                    Ar[q] = Ar[q][rng.choice(len(Ar[q]), Ncur, replace=False)]
                d = (Fv[q][succ[q]] - FU[q][succ[q]]); w = d / d.sum()
                MCR[q, hk[q]] = (w * cr[q][succ[q]] ** 2).sum() / max((w * cr[q][succ[q]]).sum(), 1e-12)
                MF[q, hk[q]] = (w * Fc[q][succ[q]] ** 2).sum() / max((w * Fc[q][succ[q]]).sum(), 1e-12)
                hk[q] = (hk[q] + 1) % H
        X = np.where(succ[..., None], Ut, X); Fv = np.where(succ, FU, Fv)
        if Nnext < Ncur:                                     # linear population reduction
            keep = np.argsort(Fv, 1)[:, :Nnext]
            X = np.take_along_axis(X, keep[..., None] * np.ones((1, 1, D), int), 1)
            Fv = np.take_along_axis(Fv, keep, 1); Ncur = Nnext
    return Fv.min(1)
