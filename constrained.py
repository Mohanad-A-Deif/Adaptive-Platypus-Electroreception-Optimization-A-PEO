"""Constrained versions of the seven algorithms, all under one feasibility-first rule.

Selection (Sec. 5.4, Eq. 44 of the source manuscript): a feasible point beats an infeasible
one; two feasible points are ordered by f; two infeasible points by the aggregate violation.
It is implemented as a comparison key, so no penalty coefficient enters anywhere.
"""
import numpy as np

BIG = 1e30

def key(f, v):
    """Scalar surrogate implementing feasibility-first order exactly: every infeasible point
    maps above every feasible one, and ties never cross the boundary."""
    return np.where(v <= 0, f, BIG + v)

def _ev(prob, X):
    f, v = prob(X)
    return f, v, key(f, v)


def peo_c(prob, L, U, D, N=30, FE=300_000, R=30, seed=0, adaptive=True,
          alpha=1.5, beta=1.0, gamma=0.5, lam=0.8, zeta=0.9, eps=1e-6, theta=1e-4,
          decay_floor=1e-6, mu=0.9):
    rng = np.random.default_rng(seed); span = U - L; T = FE // N
    b0 = beta * span / 2 if adaptive else beta * np.ones(D)
    z0 = zeta * span / 2 if adaptive else zeta * np.ones(D)
    X = L + rng.random((R, N, D)) * span
    Fv, Vv, K = _ev(prob, X)
    P, KP = X.copy(), K.copy()
    ri = np.arange(R); k = K.argmin(1)
    g = X[ri, k].copy(); G = K[ri, k].copy()
    gf, gv = Fv[ri, k].copy(), Vv[ri, k].copy()
    Kprev = None
    for t in range(T):
        m = max(decay_floor ** (t / T), decay_floor) if adaptive else 1.0
        b, z = b0 * m, z0 * m
        if Kprev is None:
            s = np.zeros((R, N), bool)
        elif adaptive:
            d = np.abs(K - Kprev); s = d > np.median(d, 1, keepdims=True)
        else:
            s = np.abs(K - Kprev) > theta
        j = (np.arange(N) + rng.integers(1, N, (R, N))) % N
        Xj = np.take_along_axis(X, j[..., None], 1)
        Kj = np.take_along_axis(K, j, 1)
        dX = Xj - X
        if adaptive:
            spread = K.max(1, keepdims=True) - K.min(1, keepdims=True) + eps
            E = -alpha * ((Kj - K) / spread)[..., None] * dX
        else:
            E = -alpha * ((Kj - K) / (np.linalg.norm(dX, axis=-1) + eps))[..., None] * dX
        tgt = (1 - mu) * g[:, None] + mu * P if adaptive else g[:, None]
        Xn = np.clip(np.where(s[..., None],
                              X + lam * (tgt - X) + z * rng.standard_normal((R, N, D)),
                              X + E + b * (2 * rng.random((R, N, D)) - 1) + gamma * (tgt - X)),
                     L, U)
        Fn, Vn, Kn = _ev(prob, Xn)
        if adaptive:
            imp = Kn < KP
            P = np.where(imp[..., None], Xn, P); KP = np.where(imp, Kn, KP)
        Kprev, X, Fv, Vv, K = K, Xn, Fn, Vn, Kn
        k = K.argmin(1); bt = K[ri, k] < G
        g[bt] = X[ri, k][bt]; G[bt] = K[ri, k][bt]
        gf[bt] = Fv[ri, k][bt]; gv[bt] = Vv[ri, k][bt]
    return gf, gv


def pso_c(prob, L, U, D, N=30, FE=300_000, R=30, seed=0, c1=2.0, c2=2.0, w0=0.9, w1=0.2):
    rng = np.random.default_rng(seed); span = U - L; T = FE // N
    X = L + rng.random((R, N, D)) * span; V = np.zeros((R, N, D)); vmax = 0.2 * span
    Fv, Vv, K = _ev(prob, X)
    P, KP, PF, PV = X.copy(), K.copy(), Fv.copy(), Vv.copy()
    ri = np.arange(N * 0 + np.arange(R).shape[0]); ri = np.arange(R)
    k = KP.argmin(1); g = P[ri, k].copy(); G = KP[ri, k].copy()
    gf, gv = PF[ri, k].copy(), PV[ri, k].copy()
    for t in range(T):
        w = w0 + (w1 - w0) * t / T
        V = np.clip(w * V + c1 * rng.random((R, N, D)) * (P - X)
                    + c2 * rng.random((R, N, D)) * (g[:, None] - X), -vmax, vmax)
        X = np.clip(X + V, L, U)
        Fv, Vv, K = _ev(prob, X)
        imp = K < KP
        P = np.where(imp[..., None], X, P); KP = np.where(imp, K, KP)
        PF = np.where(imp, Fv, PF); PV = np.where(imp, Vv, PV)
        k = KP.argmin(1); b = KP[ri, k] < G
        g[b] = P[ri, k][b]; G[b] = KP[ri, k][b]; gf[b] = PF[ri, k][b]; gv[b] = PV[ri, k][b]
    return gf, gv


def de_c(prob, L, U, D, N=30, FE=300_000, R=30, seed=0, F=0.5, CR=0.9):
    rng = np.random.default_rng(seed); span = U - L; T = FE // N
    X = L + rng.random((R, N, D)) * span
    Fv, Vv, K = _ev(prob, X)
    for t in range(T):
        kk = rng.random((R, N, N)); kk[:, np.arange(N), np.arange(N)] = np.inf
        idx = np.argsort(kk, -1)[..., :3]
        pick = lambda i: np.take_along_axis(X, idx[..., i][..., None] * np.ones((1, 1, D), int), 1)
        a, b, c = pick(0), pick(1), pick(2)
        Vt = a + F * (b - c)
        j = rng.integers(0, D, (R, N, 1))
        msk = (rng.random((R, N, D)) < CR) | (np.arange(D) == j)
        Ut = np.clip(np.where(msk, Vt, X), L, U)
        FU, VU, KU = _ev(prob, Ut)
        keep = KU <= K
        X = np.where(keep[..., None], Ut, X)
        Fv = np.where(keep, FU, Fv); Vv = np.where(keep, VU, Vv); K = np.where(keep, KU, K)
    k = K.argmin(1); ri = np.arange(X.shape[0])
    return Fv[ri, k], Vv[ri, k]


def gwo_c(prob, L, U, D, N=30, FE=300_000, R=30, seed=0):
    rng = np.random.default_rng(seed); span = U - L; T = FE // N
    X = L + rng.random((R, N, D)) * span
    Fv, Vv, K = _ev(prob, X)
    ri = np.arange(R); best_f, best_v, best_k = None, None, np.full(R, np.inf)
    for t in range(T):
        o = np.argsort(K, 1)[:, :3]
        lead = np.take_along_axis(X, o[..., None] * np.ones((1, 1, D), int), 1)
        a = 2 - 2 * t / T; acc = 0.0
        for i in range(3):
            A = a * (2 * rng.random((R, N, D)) - 1); C = 2 * rng.random((R, N, D))
            acc = acc + (lead[:, i:i + 1] - A * np.abs(C * lead[:, i:i + 1] - X))
        X = np.clip(acc / 3.0, L, U)
        Fv, Vv, K = _ev(prob, X)
        k = K.argmin(1); bt = K[ri, k] < best_k
        if best_f is None:
            best_f, best_v = Fv[ri, k].copy(), Vv[ri, k].copy(); best_k = K[ri, k].copy()
        else:
            best_f[bt] = Fv[ri, k][bt]; best_v[bt] = Vv[ri, k][bt]; best_k[bt] = K[ri, k][bt]
    return best_f, best_v


def sma_c(prob, L, U, D, N=30, FE=300_000, R=30, seed=0, z=0.03):
    rng = np.random.default_rng(seed); span = U - L; T = FE // N
    X = L + rng.random((R, N, D)) * span
    Fv, Vv, K = _ev(prob, X)
    ri = np.arange(R); k = K.argmin(1)
    g = X[ri, k].copy(); G = K[ri, k].copy(); gf, gv = Fv[ri, k].copy(), Vv[ri, k].copy()
    for t in range(T):
        Kc = np.minimum(K, 1e12)                      # rank on a finite surrogate
        o = np.argsort(Kc, 1); rank = np.empty_like(o)
        np.put_along_axis(rank, o, np.arange(N), 1)
        bF, wF = Kc.min(1, keepdims=True), Kc.max(1, keepdims=True)
        S = (bF - Kc) / (bF - wF - 1e-12)
        W = 1 + rng.random((R, N)) * np.log1p(np.abs(S)) * np.where(rank < N / 2, 1.0, -1.0)
        aa = np.arctanh(np.clip(-(t / T) + 1, -1 + 1e-12, 1 - 1e-12)); bb = 1 - t / T
        vb = rng.uniform(-aa, aa, (R, N, D)); vc = rng.uniform(-bb, bb, (R, N, D))
        p = np.tanh(np.abs(Kc - np.minimum(G, 1e12)[:, None]))
        A = np.take_along_axis(X, rng.integers(0, N, (R, N))[..., None] * np.ones((1, 1, D), int), 1)
        Bv = np.take_along_axis(X, rng.integers(0, N, (R, N))[..., None] * np.ones((1, 1, D), int), 1)
        upd = np.where(rng.random((R, N, D)) < p[..., None],
                       g[:, None] + vb * (W[..., None] * A - Bv), vc * X)
        X = np.clip(np.where(rng.random((R, N, D)) < z, L + rng.random((R, N, D)) * span, upd), L, U)
        Fv, Vv, K = _ev(prob, X)
        k = K.argmin(1); bt = K[ri, k] < G
        g[bt] = X[ri, k][bt]; G[bt] = K[ri, k][bt]; gf[bt] = Fv[ri, k][bt]; gv[bt] = Vv[ri, k][bt]
    return gf, gv


def lshade_c(prob, L, U, D, FE=300_000, R=30, seed=0, H=6, p=0.11):
    rng = np.random.default_rng(seed); span = U - L
    Nini, Nmin = 18 * D, 4
    X = L + rng.random((R, Nini, D)) * span
    Fv, Vv, K = _ev(prob, X); fe = Nini
    MF = np.full((R, H), .5); MCR = np.full((R, H), .5); hk = np.zeros(R, int)
    Ar = [np.zeros((0, D)) for _ in range(R)]; Nc = Nini
    while fe + Nc <= FE:
        Nn = max(Nmin, round(Nini + (Nmin - Nini) * fe / FE))
        r = rng.integers(0, H, (R, Nc))
        cr = np.clip(np.take_along_axis(MCR, r, 1) + .1 * rng.standard_normal((R, Nc)), 0, 1)
        mf = np.take_along_axis(MF, r, 1)
        Fc = mf + .1 * np.tan(np.pi * (rng.random((R, Nc)) - .5))
        while (Fc <= 0).any():
            Fc = np.where(Fc <= 0, mf + .1 * np.tan(np.pi * (rng.random((R, Nc)) - .5)), Fc)
        Fc = np.minimum(Fc, 1.0)
        npb = max(2, int(p * Nc)); order = np.argsort(K, 1)
        pbi = np.take_along_axis(order, rng.integers(0, npb, (R, Nc)), 1)
        pb = np.take_along_axis(X, pbi[..., None] * np.ones((1, 1, D), int), 1)
        x1 = np.take_along_axis(X, rng.integers(0, Nc, (R, Nc))[..., None] * np.ones((1, 1, D), int), 1)
        x2 = np.empty_like(x1)
        for q in range(R):
            pool = np.vstack([X[q], Ar[q]]); x2[q] = pool[rng.integers(0, len(pool), Nc)]
        V = X + Fc[..., None] * (pb - X) + Fc[..., None] * (x1 - x2)
        V = np.where(V < L, (L + X) / 2, np.where(V > U, (U + X) / 2, V))
        j = rng.integers(0, D, (R, Nc, 1))
        msk = (rng.random((R, Nc, D)) < cr[..., None]) | (np.arange(D) == j)
        Ut = np.where(msk, V, X)
        FU, VU, KU = _ev(prob, Ut); fe += Nc
        su = KU < K
        for q in range(R):
            if su[q].any():
                Ar[q] = np.vstack([Ar[q], X[q][su[q]]])
                if len(Ar[q]) > Nc:
                    Ar[q] = Ar[q][rng.choice(len(Ar[q]), Nc, replace=False)]
                d = np.minimum(K[q][su[q]] - KU[q][su[q]], 1e12); w = d / max(d.sum(), 1e-12)
                MCR[q, hk[q]] = (w * cr[q][su[q]] ** 2).sum() / max((w * cr[q][su[q]]).sum(), 1e-12)
                MF[q, hk[q]] = (w * Fc[q][su[q]] ** 2).sum() / max((w * Fc[q][su[q]]).sum(), 1e-12)
                hk[q] = (hk[q] + 1) % H
        X = np.where(su[..., None], Ut, X)
        Fv = np.where(su, FU, Fv); Vv = np.where(su, VU, Vv); K = np.where(su, KU, K)
        if Nn < Nc:
            kp = np.argsort(K, 1)[:, :Nn]
            X = np.take_along_axis(X, kp[..., None] * np.ones((1, 1, D), int), 1)
            Fv = np.take_along_axis(Fv, kp, 1); Vv = np.take_along_axis(Vv, kp, 1)
            K = np.take_along_axis(K, kp, 1); Nc = Nn
    k = K.argmin(1); ri = np.arange(R)
    return Fv[ri, k], Vv[ri, k]
