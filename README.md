# Adaptive Platypus Electroreception Optimization (A-PEO)

Reference implementation and complete experimental code for **A-PEO**, an adaptive
formulation of the Platypus Electroreception Optimization algorithm.

A-PEO builds on the PEO formulation introduced in

> M. A. Deif and M. Khishe, *Platypus Electroreception Optimization (PEO): A Bio-Inspired
> Metaheuristic Based on Sensor Fusion and Memory Behavior*, Cluster Computing, 2026.

and replaces five fixed, scale-dependent quantities of that formulation with scale-free and
time-varying counterparts. Every number reported here — for A-PEO, for the base formulation,
and for all five reference optimizers — was produced by the code in this repository under a
single protocol with fixed seeds.

---

## The adaptive components

The base formulation is recovered exactly by leaving every flag at its default; each
component below is enabled independently, so its individual effect can be measured.

| | Component | Base formulation | A-PEO | Motivation |
|---|---|---|---|---|
| C1 | perturbation scale | β, ζ absolute | fraction of (U−L)/2 | the benchmark domains span [−1,1] to [−600,600], a factor of 600 |
| C2 | amplitude schedule | constant | annealed 1 → 10⁻⁶ | with constant amplitudes the stationary spread about **g** does not vanish; this is the second-moment term of Eq. (17) of the base paper |
| C3 | phase switching | \|ΔF\| > θ | \|ΔF_i\| > median_j \|ΔF_j\| | makes the rule invariant to the scale of the objective and **removes** the parameter θ |
| C4 | directional sensing | ‖E‖ ∝ \|ΔF\| | ΔF normalised by the current fitness range | gives the step the units of a length, bounded by α‖Δx‖ |
| C5 | memory | one swarm-wide record **g** | (1−μ)·**g** + μ·**p**_i | matches the per-individual revisitation described in Section 3 of the base paper |

C3 removes a parameter rather than adding one. A sixth component, greedy acceptance, was
tested and **not adopted**: it helps on two functions and hurts on two others.

### Why the directional-sensing operator is kept

Setting α = 0 in the adaptive variant — deleting **E** and changing nothing else — costs:
Weierstrass 0 → 16.0, Shekel's Foxholes 0 → 0.133, Griewank 2.1e-07 → 1.7e-03 (8036×),
Rastrigin 18.6 → 33.1. Measured over a full run, ‖E‖ is 31–40% of the exploration step
length. The operator is decisive on rugged multimodal landscapes and neutral elsewhere.

---

## Results at a glance

**Analytical suite (18 functions, 30 runs, MaxFEs = 10⁴·D).** Mean Friedman rank, lower is
better. On the standard, centred suite A-PEO ranks 3.78 of six. When the ten functions
whose optimum sits at the centre of a symmetric box are translated — a control for the
centre bias of that suite — A-PEO moves to **2.94**, second overall.

| suite | L-SHADE | A-PEO | SMA | PSO | DE | GWO |
|---|---|---|---|---|---|---|
| centred | 2.33 | 3.78 | 2.78 | 3.89 | 4.83 | 3.39 |
| translated | 1.28 | **2.94** | 2.94 | 4.06 | 4.11 | 5.67 |

**Translation robustness.** Every A-PEO operator is a function of vector differences, so the
search is translation-equivariant by construction. Median degradation ratio over the ten
translated functions: A-PEO 1.01, DE 1.09, PSO 112, SMA 3.6e+28, GWO 3.3e+31.

**Engineering design (5 problems, 30 runs, feasibility-first selection).** A-PEO is feasible
in **150 of 150** runs and reaches the best known design in four of the five problems with
near-zero dispersion. Feasible runs out of 150: A-PEO, DE, GWO and L-SHADE 150; PSO 129;
SMA 94.

---

## Layout

```
peo.py              PEO and A-PEO; every adaptive component behind its own flag
baselines.py        PSO, DE, GWO, SMA, L-SHADE, vectorised over runs
constrained.py      the same seven algorithms under feasibility-first selection
problems/
  benchmarks.py     the 18 analytical functions, each verified at its analytic optimum
  shifted.py        the centre-bias control and the reason each function is or is not shifted
  engineering.py    the five design problems, verified at their best known designs
experiments/
  run_all.py        7 algorithms x 18 functions x 2 suites x 30 runs
  run_eng.py        7 algorithms x 5 design problems x 30 runs
  run_suite.py      the component ablation
  theta_sens.py     sensitivity of the switching rule to theta
analysis/
  analyse.py        ranks, Friedman, Wilcoxon with Holm correction
  final_tables.py   LaTeX tables for the analytical suites
  eng_tables.py     LaTeX table and figure for the design problems
  fig1_schematic.py the base-formulation vs A-PEO schematic
  probe.py          instrumentation: exploitation fraction and population diversity
  contrib.py        relative magnitude of E, M and R during a run
results/            every run, raw
figures/            all figures, greyscale, 600 dpi
```

## Reproducing

```bash
pip install -r requirements.txt
cd experiments
python run_all.py classic      # ~40 min on one core
python run_all.py shifted
python run_eng.py
python run_suite.py literal    # ablation
cd ../analysis
python analyse.py && python final_tables.py && python eng_tables.py
```

Seeds are fixed (5000 + problem index for the comparison, 6000 + index for the design
problems), so a rerun reproduces the stored numbers exactly. Every experiment checkpoints
after each cell and skips completed work, so a run can be interrupted and restarted.

Requires numpy, scipy and matplotlib only.

## Using A-PEO

```python
from peo import peo
from problems.benchmarks import F9          # Rastrigin, D = 30

best = peo(F9, L=-5, U=5, D=30, N=30, FE=300_000, R=30, seed=0,
           relative_scale=True, decay="exp", decay_floor=1e-6,
           esens="normalised", switch="median", mu=0.9)   # the A-PEO configuration
```

Leaving the five keyword arguments out gives the base formulation. `R` is the number of
independent runs, executed in parallel as one tensor.

## Citation

```bibtex
@article{peo2026,
  title   = {Platypus Electroreception Optimization (PEO): A Bio-Inspired Metaheuristic
             Based on Sensor Fusion and Memory Behavior},
  author  = {Deif, Mohanad A. and Khishe, Mohammad},
  journal = {Cluster Computing},
  year    = {2026}
}
```

## License

MIT, see `LICENSE`.
