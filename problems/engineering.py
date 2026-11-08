"""The five constrained engineering design problems of the source manuscript
(Eqs. 10-42), vectorised over runs: X has shape (..., D) and every function returns
(objective, aggregate violation) with the same leading shape.

Constraint handling follows Sec. 5.4 of the source manuscript: the aggregate violation is
phi(x) = sum_i max(0, g_i(x)), and selection is feasibility-first (Eq. 44).  Discrete
variables are rounded after every update, as the manuscript specifies.
"""
import numpy as np

_pos = lambda v: np.maximum(v, 0.0)


# ---------------- 1. Pressure vessel, Eqs. (10)-(14) ----------------
def pressure_vessel(X):
    x1, x2, x3, x4 = (X[..., i] for i in range(4))
    x1 = np.round(x1 / 0.0625) * 0.0625          # plate thicknesses: multiples of 1/16 in
    x2 = np.round(x2 / 0.0625) * 0.0625
    f = (0.6224 * x1 * x3 * x4 + 1.7781 * x2 * x3 ** 2
         + 3.1661 * x1 ** 2 * x4 + 19.84 * x1 ** 2 * x3)
    g = [0.0193 * x3 - x1,
         0.00954 * x3 - x2,
         -(np.pi * x3 ** 2 * x4 + 4 / 3 * np.pi * x3 ** 3) + 1_296_000,
         x4 - 240]
    return f, sum(_pos(v) for v in g)

pressure_vessel.bounds = (np.array([0.0625, 0.0625, 10.0, 10.0]),
                          np.array([99.0, 99.0, 200.0, 200.0]))
# Best-known feasible value under the DISCRETE thickness rule the manuscript imposes
# (x1, x2 multiples of 0.0625).  5885.33 is the continuous-relaxation optimum and is not
# attainable here; the discrete optimum is 6059.714.
pressure_vessel.ref = 6059.714


# ---------------- 2. Welded beam, Eqs. (15)-(22) ----------------
def welded_beam(X):
    h, l, t, b = (X[..., i] for i in range(4))
    P, L, E, G = 6000.0, 14.0, 30e6, 12e6
    f = 1.10471 * h ** 2 * l + 0.04811 * t * b * (14.0 + l)
    M = P * (L + l / 2)
    R = np.sqrt(l ** 2 / 4 + ((h + t) / 2) ** 2)
    J = 2 * (np.sqrt(2) * h * l * (l ** 2 / 12 + ((h + t) / 2) ** 2))
    tau1 = P / (np.sqrt(2) * h * l)
    tau2 = M * R / J
    tau = np.sqrt(tau1 ** 2 + 2 * tau1 * tau2 * l / (2 * R) + tau2 ** 2)
    sigma = 6 * P * L / (b * t ** 2)
    delta = 4 * P * L ** 3 / (E * b * t ** 3)
    Pc = (4.013 * E * np.sqrt(t ** 2 * b ** 6 / 36) / L ** 2) * (1 - t / (2 * L) * np.sqrt(E / (4 * G)))
    g = [tau - 13600, sigma - 30000, h - b,
         0.10471 * h ** 2 + 0.04811 * t * b * (14 + l) - 5.0,
         0.125 - h, delta - 0.25, P - Pc]
    return f, sum(_pos(v) for v in g)

welded_beam.bounds = (np.array([0.1, 0.1, 0.1, 0.1]), np.array([2.0, 10.0, 10.0, 2.0]))
welded_beam.ref = 1.724852


# ---------------- 3. Three-bar truss, Eqs. (23)-(26) ----------------
def three_bar(X):
    x1, x2 = X[..., 0], X[..., 1]
    P, s, L = 2.0, 2.0, 100.0
    f = (2 * np.sqrt(2) * x1 + x2) * L
    d = np.sqrt(2) * x1 ** 2 + 2 * x1 * x2
    g = [(np.sqrt(2) * x1 + x2) / np.maximum(d, 1e-12) * P - s,
         x2 / np.maximum(d, 1e-12) * P - s,
         1.0 / np.maximum(np.sqrt(2) * x2 + x1, 1e-12) * P - s]
    return f, sum(_pos(v) for v in g)

three_bar.bounds = (np.array([1e-4, 1e-4]), np.array([1.0, 1.0]))
three_bar.ref = 263.8958


# ---------------- 4. Speed reducer, Eqs. (27)-(35) ----------------
def speed_reducer(X):
    x1, x2, x3, x4, x5, x6, x7 = (X[..., i] for i in range(7))
    x3 = np.round(x3)                                    # number of teeth: integer
    f = (0.7854 * x1 * x2 ** 2 * (3.3333 * x3 ** 2 + 14.9334 * x3 - 43.0934)
         - 1.508 * x1 * (x6 ** 2 + x7 ** 2) + 7.4777 * (x6 ** 3 + x7 ** 3)
         + 0.7854 * (x4 * x6 ** 2 + x5 * x7 ** 2))
    g = [27 / (x1 * x2 ** 2 * x3) - 1,
         397.5 / (x1 * x2 ** 2 * x3 ** 2) - 1,
         1.93 * x4 ** 3 / (x2 * x3 * x6 ** 4) - 1,
         1.93 * x5 ** 3 / (x2 * x3 * x7 ** 4) - 1,
         np.sqrt((745 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (110 * x6 ** 3) - 1,
         np.sqrt((745 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (85 * x7 ** 3) - 1,
         x2 * x3 / 40 - 1, 5 * x2 / x1 - 1, x1 / (12 * x2) - 1,
         (1.5 * x6 + 1.9) / x4 - 1, (1.1 * x7 + 1.9) / x5 - 1]
    return f, sum(_pos(v) for v in g)

speed_reducer.bounds = (np.array([2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0]),
                        np.array([3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]))
# 2994.4245 is the continuous optimum; with x3 rounded to an integer, as the manuscript
# requires, the attainable optimum is 2996.348.
speed_reducer.ref = 2996.348


# ---------------- 5. Tension / compression spring, Eqs. (36)-(42) ----------------
def spring(X):
    d, D, N = X[..., 0], X[..., 1], X[..., 2]
    f = (N + 2) * D * d ** 2
    g = [1 - D ** 3 * N / (71785 * d ** 4),
         (4 * D ** 2 - d * D) / (12566 * (D * d ** 3 - d ** 4)) + 1 / (5108 * d ** 2) - 1,
         1 - 140.45 * d / (D ** 2 * N),
         (D + d) / 1.5 - 1]
    return f, sum(_pos(v) for v in g)

spring.bounds = (np.array([0.05, 0.25, 2.0]), np.array([2.0, 1.3, 15.0]))
spring.ref = 0.012665

PROBLEMS = [("Pressure vessel", pressure_vessel), ("Welded beam", welded_beam),
            ("Three-bar truss", three_bar), ("Speed reducer", speed_reducer),
            ("Tension/compression spring", spring)]

if __name__ == "__main__":
    # each problem evaluated at its best-known design: violation must be ~0 and f ~ ref
    best = {
        "Pressure vessel": [0.8125, 0.4375, 42.098446, 176.636596],
        "Welded beam": [0.205730, 3.470489, 9.036624, 0.205730],
        "Three-bar truss": [0.788675, 0.408248],
        "Speed reducer": [3.5, 0.7, 17, 7.3, 7.8, 3.350215, 5.286683],
        "Tension/compression spring": [0.051689, 0.356718, 11.288966],
    }
    for name, fn in PROBLEMS:
        x = np.array(best[name])
        f, v = fn(x)
        print("%-28s f = %12.6f   (ref %12.6f)   violation = %.3e"
              % (name, f, fn.ref, v))
