"""Figure 1: structural comparison of the base PEO update and the A-PEO update.

Nature/IEEE greyscale style: white background, Computer Modern, no colour, 600 dpi.
The two panels carry the SAME flow; only the five annotated quantities differ,
which is the claim the figure exists to make.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams.update({
    "font.family": "cmr10",            # Computer Modern, matching the manuscript
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 9,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

GREY = "0.58"          # unchanged structure
BLACK = "black"        # the five adaptive quantities

XMAX, YMAX = 11.6, 13.0
LOOP_X = 11.05


def box(ax, x, y, w, h, text, lw=0.7, ec=GREY, fs=8.0):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.010,rounding_size=0.08",
                                linewidth=lw, edgecolor=ec, facecolor="white", zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.6)


def arrow(ax, x0, y0, x1, y1, lw=0.7, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=lw, linestyle=ls, color="black",
                                 shrinkA=0, shrinkB=0, zorder=1))


def tag(ax, x, y, s):
    """Component marker C1..C5, on the top edge of the box it refers to."""
    ax.add_patch(Circle((x, y), 0.27, facecolor="white", edgecolor="black",
                        linewidth=0.8, zorder=5))
    ax.text(x, y, s, ha="center", va="center", fontsize=6.4, zorder=6)


def panel(ax, adaptive):
    ax.set_xlim(0, XMAX)
    ax.set_ylim(0, YMAX)
    ax.axis("off")

    e = BLACK if adaptive else GREY
    w = 1.1 if adaptive else 0.7

    CX = 5.1                      # centre of the column
    WIDE = 9.4                    # width of the full-width boxes

    # ---- population --------------------------------------------------
    y_pop = 12.35
    box(ax, CX, y_pop, WIDE, 0.80,
        r"population $\{\mathbf{x}_i^{t}\}_{i=1}^{N}$,   objective values $F_i^{t}$", fs=8.8)

    # ---- the three fusion operators -----------------------------------
    y_op = 10.60
    centres = (1.90, 5.10, 8.30)
    wbox = 3.05                   # 0.40 clearance between neighbours

    if adaptive:
        texts = (
            "directional sensing\n"
            r"$\mathbf{E}=-\alpha\,(\Delta F/\rho^{t})\,\Delta\mathbf{x}$",
            "perturbation\n"
            r"$\mathbf{M}=\beta_{0}\,m(t)\,\boldsymbol{\eta}$",
            "memory attraction\n"
            r"$\mathbf{R}=\gamma\,(\mathbf{c}_i-\mathbf{x}_i)$")
    else:
        texts = (
            "directional sensing\n"
            r"$\mathbf{E}=-\alpha\,(\Delta F/\|\Delta\mathbf{x}\|)\,\Delta\mathbf{x}$",
            "perturbation\n"
            r"$\mathbf{M}=\beta\,\boldsymbol{\eta}$",
            "memory attraction\n"
            r"$\mathbf{R}=\gamma\,(\mathbf{g}-\mathbf{x}_i)$")

    hbox = 1.35
    for cx, txt in zip(centres, texts):
        box(ax, cx, y_op, wbox, hbox, txt, lw=w, ec=e, fs=7.0)
        arrow(ax, cx, y_pop - 0.40, cx, y_op + hbox / 2 + 0.04)
        arrow(ax, cx, y_op - hbox / 2 - 0.04, cx, 8.78)

    if adaptive:
        ytop = y_op + hbox / 2
        tag(ax, centres[0] + 1.10, ytop, "C4")
        tag(ax, centres[1] + 0.78, ytop, "C1")
        tag(ax, centres[1] + 1.42, ytop, "C2")
        tag(ax, centres[2] + 1.10, ytop, "C5")

    # ---- phase switch ---------------------------------------------------
    y_sw = 8.15
    tsw = (r"phase switch:   $s_i^{t}=\mathbb{1}\{\delta_i^{t}>\tilde{\delta}^{t}\}$" "\n"
           r"population median, invariant to the scale of $f$"
           if adaptive else
           r"phase switch:   $s_i^{t}=\mathbb{1}\{\delta_i^{t}>\theta\}$" "\n"
           r"absolute threshold $\theta$, tied to the scale of $f$")
    box(ax, CX, y_sw, WIDE, 1.05, tsw, lw=w, ec=e, fs=8.0)
    if adaptive:
        tag(ax, CX + WIDE / 2 - 0.55, y_sw + 0.525, "C3")

    # ---- the two branches ------------------------------------------------
    y_br = 5.85
    tx = "exploration branch\n" + r"$\mathbf{x}_i+\mathbf{E}+\mathbf{M}+\mathbf{R}$"
    tt = ("exploitation branch\n"
          r"$\mathbf{x}_i+\lambda(\mathbf{c}_i-\mathbf{x}_i)+\zeta_{0}m(t)\,\boldsymbol{\xi}$"
          if adaptive else
          "exploitation branch\n"
          r"$\mathbf{x}_i+\lambda(\mathbf{g}-\mathbf{x}_i)+\zeta\,\boldsymbol{\xi}$")
    box(ax, 2.70, y_br, 4.05, 1.20, tx, lw=0.7, ec=GREY, fs=7.8)
    box(ax, 7.50, y_br, 4.05, 1.20, tt, lw=w, ec=e, fs=7.4)
    arrow(ax, 3.70, y_sw - 0.53, 2.70, y_br + 0.62)
    arrow(ax, 6.50, y_sw - 0.53, 7.50, y_br + 0.62)
    ax.text(2.95, 7.00, r"$s_i^{t}=0$", fontsize=8.0, ha="right")
    ax.text(7.25, 7.00, r"$s_i^{t}=1$", fontsize=8.0, ha="left")
    if adaptive:
        tag(ax, 7.50 + 4.05 / 2 - 0.50, y_br + 0.60, "C2")

    # ---- projection and elitist record ------------------------------------
    y_pr, y_rc = 3.85, 2.30
    box(ax, CX, y_pr, WIDE, 0.80,
        r"projection onto the box:   $\mathbf{x}_i^{t+1}=\Pi_{\Omega}(\,\cdot\,)$", fs=8.8)
    arrow(ax, 2.70, y_br - 0.62, 4.20, y_pr + 0.42)
    arrow(ax, 7.50, y_br - 0.62, 6.00, y_pr + 0.42)

    box(ax, CX, y_rc, WIDE, 0.80,
        r"elitist record:   $G^{t+1}=\min\{G^{t},\,\min_i F_i^{t+1}\}$", fs=8.8)
    arrow(ax, CX, y_pr - 0.42, CX, y_rc + 0.42)

    # ---- feedback loop, drawn clear of every box ---------------------------
    ax.plot([CX + WIDE / 2, LOOP_X], [y_rc, y_rc], lw=0.7, color="black", zorder=1)
    ax.plot([LOOP_X, LOOP_X], [y_rc, y_pop], lw=0.7, ls=(0, (3, 2)), color="black", zorder=1)
    arrow(ax, LOOP_X, y_pop, CX + WIDE / 2 + 0.02, y_pop, ls=(0, (3, 2)))
    ax.text(LOOP_X + 0.30, (y_rc + y_pop) / 2, "next iteration", fontsize=7.0,
            rotation=90, va="center", ha="center")

    # ---- the one-line verdict for the panel ---------------------------------
    note = ("every amplitude and every threshold is estimated from\n"
            r"the current population; the parameter $\theta$ is removed"
            if adaptive else
            "four quantities are fixed in absolute units, so the\n"
            "calibration is tied to one problem scale")
    ax.text(CX, 0.85, note, fontsize=8.2, ha="center", va="center")


fig, axes = plt.subplots(1, 2, figsize=(6.5, 4.6))
panel(axes[0], adaptive=False)
panel(axes[1], adaptive=True)
axes[0].set_title("(a)   base PEO formulation", fontsize=9.6, pad=1)
axes[1].set_title("(b)   A-PEO", fontsize=9.6, pad=1)

fig.subplots_adjust(wspace=0.05)
fig.tight_layout()
fig.savefig("../figures/fig1_schematic.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("written ../figures/fig1_schematic.png")
