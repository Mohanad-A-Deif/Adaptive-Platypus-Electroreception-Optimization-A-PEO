"""The 18 analytical benchmark functions of the PEO paper (Table: tab:cec2022).
All functions are vectorised: X has shape (..., D) and the return has shape (...)."""
import numpy as np

def _u(x, a=10, k=100, m=4):          # penalty helper for F12/F13
    return k * (np.maximum(x - a, 0) ** m + np.maximum(-x - a, 0) ** m)

def F1(X):  return (X ** 2).sum(-1)                                   # Sphere
def F2(X):  return np.abs(X).sum(-1) + np.abs(X).prod(-1)             # Schwefel 2.22
def F3(X):  return (np.cumsum(X, axis=-1) ** 2).sum(-1)               # Schwefel 1.2
def F4(X):  return np.abs(X).max(-1)                                  # Schwefel 2.21
def F5(X):                                                            # Rosenbrock
    return (100 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2).sum(-1)
def F6(X):  return X[..., 0] ** 2 + 1e6 * (X[..., 1:] ** 2).sum(-1)   # Cigar
def F7(X, rng=None):                                                  # Quartic + noise
    i = np.arange(1, X.shape[-1] + 1)
    r = (rng.random(X.shape[:-1]) if rng is not None else np.random.random(X.shape[:-1]))
    return (i * X ** 4).sum(-1) + r
def F8(X):  return (-X * np.sin(np.sqrt(np.abs(X)))).sum(-1)          # Schwefel 2.26
def F9(X):  return (X ** 2 - 10 * np.cos(2 * np.pi * X) + 10).sum(-1) # Rastrigin
def F10(X):                                                           # Ackley
    D = X.shape[-1]
    return (-20 * np.exp(-0.2 * np.sqrt((X ** 2).sum(-1) / D))
            - np.exp(np.cos(2 * np.pi * X).sum(-1) / D) + 20 + np.e)
def F11(X):                                                           # Griewank
    i = np.sqrt(np.arange(1, X.shape[-1] + 1))
    return 1 + (X ** 2).sum(-1) / 4000 - np.cos(X / i).prod(-1)
def F12(X):                                                           # Penalized 1
    D = X.shape[-1]; y = 1 + (X + 1) / 4
    t = (10 * np.sin(np.pi * y[..., 0]) ** 2
         + ((y[..., :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[..., 1:]) ** 2)).sum(-1)
         + (y[..., -1] - 1) ** 2)
    return np.pi / D * t + _u(X).sum(-1)
def F13(X):                                                           # Penalized 2
    t = (np.sin(3 * np.pi * X[..., 0]) ** 2
         + ((X[..., :-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * X[..., 1:]) ** 2)).sum(-1)
         + (X[..., -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * X[..., -1]) ** 2))
    return 0.1 * t + _u(X, a=5, k=100, m=4).sum(-1)
def F14(X, a=0.5, b=3.0, kmax=20):                                    # Weierstrass
    D = X.shape[-1]; k = np.arange(kmax + 1)
    ak, bk = a ** k, b ** k
    s1 = (ak * np.cos(2 * np.pi * bk * (X[..., None] + 0.5))).sum(-1).sum(-1)
    s2 = (ak * np.cos(np.pi * bk)).sum()
    return s1 - D * s2

_A15 = np.array([
    [-32, -16, 0, 16, 32] * 5,
    [-32] * 5 + [-16] * 5 + [0] * 5 + [16] * 5 + [32] * 5], float)
def F15(X):                                                           # Shekel's Foxholes
    j = np.arange(1, 26)
    d = ((X[..., None, :] - _A15.T) ** 6).sum(-1)
    return (1 / 500 + (1 / (j + d)).sum(-1)) ** -1

_a16 = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
_b16 = 1 / np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
def F16(X):                                                           # Kowalik
    x1, x2, x3, x4 = X[..., 0:1], X[..., 1:2], X[..., 2:3], X[..., 3:4]
    return ((_a16 - x1 * (_b16 ** 2 + _b16 * x2) / (_b16 ** 2 + _b16 * x3 + x4)) ** 2).sum(-1)
def F17(X):                                                           # Six-hump camel back
    x1, x2 = X[..., 0], X[..., 1]
    return (4 - 2.1 * x1 ** 2 + x1 ** 4 / 3) * x1 ** 2 + x1 * x2 + (-4 + 4 * x2 ** 2) * x2 ** 2
def F18(X):                                                           # Branin
    x1, x2 = X[..., 0], X[..., 1]
    return ((x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10)

# name, function, lower, upper, D, f_min   (ranges and dimensions from the paper)
SUITE = [
    ("F1  Sphere",        F1,  -100, 100,   30, 0.0),
    ("F2  Schwefel2.22",  F2,   -10,  10,   30, 0.0),
    ("F3  Schwefel1.2",   F3,  -100, 100,   30, 0.0),
    ("F4  Schwefel2.21",  F4,  -100, 100,   30, 0.0),
    ("F5  Rosenbrock",    F5,   -30,  30,   30, 0.0),
    ("F6  Cigar",         F6,  -100, 100,   30, 0.0),
    ("F7  Quartic",       F7, -1.28,1.28,   30, 0.0),
    ("F8  Schwefel2.26",  F8,  -500, 500,   30, -418.9829 * 30),
    ("F9  Rastrigin",     F9,    -5,   5,   30, 0.0),
    ("F10 Ackley",       F10,   -32,  32,   30, 0.0),
    ("F11 Griewank",     F11,  -600, 600,   30, 0.0),
    ("F12 Penalized1",   F12,   -50,  50,   30, 0.0),
    ("F13 Penalized2",   F13,   -50,  50,   30, 0.0),
    ("F14 Weierstrass",  F14,    -1,   1,   30, 0.0),
    ("F15 Foxholes",     F15,   -65,  65,    2, 1.0),
    ("F16 Kowalik",      F16,    -5,   5,    4, 0.0003075),
    ("F17 SixHumpCamel", F17,    -5,   5,    2, -1.0316285),
    ("F18 Branin",       F18,    -5,   5,    2, 0.397887),
]
