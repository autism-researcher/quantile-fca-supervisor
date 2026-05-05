# Reproducibility Guide

This document provides step-by-step instructions for reproducing every
numerical result, table, and figure in the paper from the released data.

## Prerequisites

- Python 3.9 or later
- Approximately 8 GB free disk space (after extracting Zenodo archive)
- The released dataset from Zenodo: [10.5281/zenodo.20036794](https://doi.org/10.5281/zenodo.20036794)

## 1. Clone and set up

```bash
git clone https://github.com/autism-researcher/quantile-fca-supervisor.git
cd quantile-fca-supervisor
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Download the dataset

Download the Zenodo archive and extract it into `./data/`:

```
data/
├── stageA_calibration/
│   ├── seed_1000/  (240 NORMAL episodes)
│   ├── seed_2000/
│   └── seed_3000/
├── stageB_evaluation/
│   ├── seed_1000/  (360 episodes × 8 controllers)
│   ├── seed_2000/
│   └── seed_3000/
├── pilot_stageA/                     (999 NORMAL pilot episodes)
├── pilot_stageB/                     (94 paired pilot episodes)
├── calibrations/                     (per-seed B_τ and δ JSON)
├── combined_master.csv               (8,640 controller-runs aggregated)
└── analysis_provenance.json          (data integrity hashes)
```

## 3. Verify data integrity (optional)

The Zenodo archive ships with `analysis_provenance.json`, which lists
SHA-256 hashes for every released CSV. To verify the download manually:

```bash
# Linux / macOS
cd ./data
python -c "import json,hashlib,sys
prov=json.load(open('analysis_provenance.json'))
ok=True
for fn,want in prov.items():
    try:
        got=hashlib.sha256(open(fn,'rb').read()).hexdigest()
        if got!=want: print('MISMATCH',fn); ok=False
    except FileNotFoundError: print('MISSING',fn); ok=False
print('OK' if ok else 'FAIL')"
```

## 4. Reproduce paper tables

```bash
python src/analyze_framework.py ./data
```

Outputs are written to `./data/analysis/` as CSVs:
`combined_master.csv`, `per_controller_bootstrap_ci.csv`,
`h1_calibration_convergence.csv`, `h2_safety_noninferiority.csv`,
`h3_pareto_tradeoff.csv`, `h3_spearman_per_seed.csv`,
`h3b_pareto_dominance.csv`, `h4_reproducibility.csv`,
`h5_vs_normal_mcnemar.csv`, `h6_comfort_noninferiority.csv`.

These reproduce:
- **Table IX** (H1 calibration convergence per seed × τ)
- **Table X** (per-controller collision rate, NMR, NMR_2, TTC_min, J_max)
- **Table XI** (hypothesis test verdicts H1–H7)
- **Table XII** (H3b Pareto-dominance bootstrap)
- **Table XIII** (H4 reproducibility coefficients of variation)
- **Table XIV** (mission efficiency η_v, e_v, IR)

## 5. Reproduce paper figures

The two figure-regeneration scripts read the analysis CSVs produced
by step 4 from the **current working directory**, so `cd` into the
analysis output folder first:

```bash
cd ./data/analysis
python ../../analysis/regenerate_pareto.py
python ../../analysis/cross_seed_figure.py
cd ../..
```

Produces `figure_pareto_bw.png`, `figure_collision_bars_bw.png`,
`figure_per_seed_bw.png`, `figure_cross_seed_bw.png` — these are
Figures 3, 4, 5, 6 in the paper.

## 6. Reproduce sensitivity analysis

```bash
python analysis/sensitivity_analysis.py ./data
```

Reproduces the leave-one-feature-out sensitivity results reported in
Sections V-I and VI-K of the paper (mean |ΔB_{N,τ}| = 0.030–0.033;
maximum 0.049 at τ = 0.10).

## 7. Re-run CARLA simulation from scratch (optional)

This step is only required if you want to verify the simulation pipeline
end-to-end. It is **not** required to reproduce the paper's numerical
claims, which can be reproduced from the per-episode summary CSVs in
the Zenodo archive alone.

Requirements:
- CARLA 0.9.13 installed (https://carla.readthedocs.io/en/0.9.13/)
- Approximately 8–12 hours per replication on a workstation with one GPU

```bash
# In one terminal, start CARLA server
$CARLA_ROOT/CarlaUE4.sh -opengl -RenderOffScreen

# In another terminal, run a replication
python src/batch_run_framework_v4.py \
  --seed 1000 \
  --output-dir ./runs/seed_1000 \
  --calibration-stage \
  --evaluation-stage
```

Repeat for seeds 2000 and 3000. Each replication produces 600 episodes:
240 Stage A calibration + 360 Stage B evaluation, with each Stage B
episode re-run under all 8 controllers.

## Expected results

After running the analysis pipeline you should reproduce, to within
floating-point tolerance:

| Quantity | Expected value |
|---|---|
| H1: 9/9 cells within tolerance | ✓ |
| P_{0.20} collision rate | 3.61% [2.50, 4.81] |
| T_{3.5} collision rate | 4.17% [3.06, 5.37] |
| NORMAL collision rate | 21.57% [19.07, 24.07] |
| H2 p-value (P_{0.20} vs T_{3.5}) | 0.031 |
| H3b dominance fraction (P_{0.10}) | 99.9% |
| H3b dominance fraction (P_{0.15}) | 100.0% |
| H3b dominance fraction (P_{0.20}) | 99.9% |
| Cross-seed Spearman ρ on collision-rate ordering | {0.896, 1.000, 0.896} |
| Sign-of-effect agreement (28 controller pairs) | 23/28 |

If you obtain different values, please open a GitHub issue with a
description of your environment (Python version, OS, package versions
from `pip freeze`) and the contents of any failing CSV.

## Pre-registered deviations

All deviations from the signed pre-registration of 2026-05-01 are
recorded in Table V of the paper (10 entries). The OSF entry for
the pre-registration is preserved in its original form.

## Contact

Issues, bugs, reproducibility problems: please open a GitHub issue.
General questions about the methodology: m.hossain@upm.edu.sa.
