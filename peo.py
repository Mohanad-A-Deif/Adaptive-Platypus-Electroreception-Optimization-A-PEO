"""Platypus Electroreception Optimization (PEO).

`variant="literal"` implements Eqs. (4)-(14) of the source manuscript exactly as written.
The other variants switch on individually-documented adaptive components; each is off by default so
that the literal algorithm is always recoverable.

All runs of one experiment are executed in parallel: the population tensor has shape
(R, N, D) for R independent runs, so the per-run random streams are independent but the
arithmetic is fully vectorised.
"""
import numpy as np


def peo(f, L, U, D, N=30, FE=300_000, R=30, seed=0,
        alpha=1.5, beta=1.0, gamma=0.5, lam=0.8, zeta=0.9, eps=1e-6, theta=1e-4,
        relative_scale=False,     # adaptive component 1: beta, zeta expressed as a fraction of (U-L)
        decay=None,               # adaptive component 2: None | "linear" | "exp"  amplitude schedule
        greedy=False,             # adaptive component 3: keep the agent's previous point if the move is worse
        mu=0.0,                   # adaptive component 4: weight of the agent's OWN best in the memory term
        switch="absolute",        # adaptive component 5: "absolute" = Eq.(13) as published | "median" = scale-free
        esens="published",        # adaptive component 6: "published" = Eq.(6) | "normalised" = scale-free secant
        decay_floor=1e-3,         # smallest amplitude multiplier reached by the schedule
        track=False):             # return the median best-so-far curve as well
    rng = np.random.default_rng(seed)
    span = float(U - L)
    T = FE // N
    # amplitude bases
    b0 = beta * span / 2.0 if relative_scale else beta
    z0 = zeta * span / 2.0 if relative_scale else zeta

    X = L + rng.random((R, N, D)) * span
    Fv = f(X, rng) if _noisy(f) else f(X)
    fe = N
    k = Fv.argmin(1)
    r_ix = np.arange(R)
    g = X[r_ix, k].copy()
    G = Fv[r_ix, k].copy()
    P = X.copy(); FP = Fv.copy()          # per-agent memory (component 4); unused when mu == 0
    Fprev = None
    curve = []

    for t in range(T):
        # --- amplitude schedule (component 2) -----------------------------------
        if decay == "linear":
            m = max(1.0 - t / T, decay_floor)
        elif decay == "exp":
            m = max(decay_floor ** (t / T), decay_floor)
        else:
            m = 1.0
        b, z = b0 * m, z0 * m

        # --- phase switching, Eqs. (12)-(13) ---------------------------------
        if Fprev is None:
            s = np.zeros((R, N), bool)
        elif switch == "median":
            # scale-free rule: refine only if this agent's recent fitness contrast exceeds
            # the population median contrast; removes theta and its dependence on |F|.
            d = np.abs(Fv - Fprev)
            s = d > np.median(d, axis=1, keepdims=True)
        else:
            s = np.abs(Fv - Fprev) > theta                     # Eq. (13) as published

        # --- directional sensing, Eq. (6): a peer j != i, drawn uniformly ------
        j = (np.arange(N) + rng.integers(1, N, (R, N))) % N
        Xj = np.take_along_axis(X, j[..., None], 1)
        Fj = np.take_along_axis(Fv, j, 1)
        dX = Xj - X
        if esens == "normalised":
            # Eq. (6) gives ||E|| ~ alpha*|dF|, i.e. a LENGTH proportional to a FITNESS value.
            # Dividing the contrast by the current population spread restores the units and
            # bounds the step by alpha*||dx||, while keeping the sign convention of Eq. (6).
            spread = Fv.max(1, keepdims=True) - Fv.min(1, keepdims=True) + eps
            E = -alpha * ((Fj - Fv) / spread)[..., None] * dX
        else:
            E = -alpha * ((Fj - Fv) / (np.linalg.norm(dX, axis=-1) + eps))[..., None] * dX
        # --- mechanoreceptive perturbation, Eq. (7) ---------------------------
        M = b * (2 * rng.random((R, N, D)) - 1)
        # --- memory-guided attraction, Eq. (8) --------------------------------
        # mu = 0 reproduces Eq. (8) exactly; mu > 0 blends in the agent's own record
        tgt = g[:, None, :] if mu == 0.0 else (1 - mu) * g[:, None, :] + mu * P
        Rt = gamma * (tgt - X)

        X_expl = X + E + M + Rt                                   # Eq.  (9)
        X_expt = X + lam * (tgt - X) + z * rng.standard_normal((R, N, D))  # Eq. (10)
        Xn = np.clip(np.where(s[..., None], X_expt, X_expl), L, U)  # Eq. (11), projection

        Fn = f(Xn, rng) if _noisy(f) else f(Xn)
        fe += N

        if greedy:                                                # adaptive component 3
            keep = Fn <= Fv
            Xn = np.where(keep[..., None], Xn, X)
            Fn = np.where(keep, Fn, Fv)

        if mu > 0.0:                                          # refresh per-agent memory
            imp = Fn < FP
            P = np.where(imp[..., None], Xn, P); FP = np.where(imp, Fn, FP)

        Fprev, X, Fv = Fv, Xn, Fn

        # --- elitist record, Eq. (14) ----------------------------------------
        k = Fv.argmin(1)
        better = Fv[r_ix, k] < G
        g[better] = X[r_ix, k][better]
        G[better] = Fv[r_ix, k][better]
        if track and (t % max(1, T // 200) == 0 or t == T - 1):
            curve.append((fe, float(np.median(G))))

    return (G, curve) if track else G


def _noisy(f):
    return getattr(f, "__name__", "") == "F7"
