# Quantile-Calibrated FCA Supervisor

Implementation and analysis code accompanying:

> **Hossain, M.B., Kamal, M.A.S., Rahman, S.S., Tayan, O., Mehedi, I.M., Showail, A.**
> "Real-Time Risk-Aware Supervisory Control for Autonomous Driving with
> Quantile-Calibrated Safety Envelopes,"
> *IEEE Transactions on Intelligent Transportation Systems* (under review, 2026).

This repository implements the calibration framework, the eight evaluated
controllers (NORMAL, four TTC tuning grid points, three calibrated
\\(P_{\tau}\\) controllers), and the statistical analysis pipeline described
in the paper.

## Pre-registration

The hypotheses, analysis plan, exclusion criteria, and disconfirmation
conditions were registered on the Open Science Framework on 2026-05-01,
**before** any Stage B evaluation data was analysed.

- **Pre-registration:** https://osf.io/snpbg
- **Companion theoretical manuscript:** Hossain, Kamal, Mehedi (2026),
  "Quantile-calibrated invariant sets for data-driven safety in
  control-affine systems," *Automatica* (under review).

## Data

The full Stage A calibration dataset (720 NORMAL episodes pooled across
three seeds) and Stage B evaluation dataset (1,080 paired episodes × 8
controllers = 8,640 controller-runs) are archived on Zenodo:

> **Zenodo DOI:** [10.5281/zenodo.20036794](https://doi.org/10.5281/zenodo.20036794)

The 94 pilot Stage B paired episodes referenced in pre-registration §6 are
included in the same archive under `pilot_stageB/`.

## Repository structure

```
.
├── README.md                          (this file)
├── LICENSE
├── PATENTS.md                         (patent notice for USPTO App 19/533,330)
├── CITATION.cff                       (citation metadata)
├── requirements.txt                   (Python dependencies)
├── .gitignore
│
├── src/                               (CARLA simulation framework)
│   ├── batch_run_framework_v4.py        — episode orchestrator
│   ├── carla_episode_logger_PUB_v3.py   — per-tick CARLA logger
│   └── analyze_framework.py             — main analysis pipeline
│
├── analysis/                          (paper-specific analysis scripts)
│   ├── sensitivity_analysis.py          — Section V-I, VI-K
│   ├── regenerate_pareto.py             — Figure 4
│   └── cross_seed_figure.py             — Figure 6
│
└── docs/
    └── reproducibility.md             — step-by-step reproducibility guide
```

## Reproducibility

To reproduce the headline numerical results in the paper:

### 1. Environment setup

```bash
git clone https://github.com/autism-researcher/quantile-fca-supervisor.git
cd quantile-fca-supervisor
python -m venv venv
source venv/bin/activate     # Linux/macOS
# venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

### 2. CARLA installation (only required if re-running simulation)

The simulation framework requires CARLA 0.9.13. The released datasets
(Zenodo, above) contain the per-episode summaries needed to reproduce
all paper tables and figures *without* re-running CARLA.

For instructions on CARLA installation, see:
https://carla.readthedocs.io/en/0.9.13/

### 3. Reproduce paper results from released data

```bash
# Download the Zenodo archive into ./data/ so that ./data/seed_1000/eval_master.csv etc.
# exist. Then:
python src/analyze_framework.py ./data
# This produces ./data/analysis/combined_master.csv,
# per_controller_bootstrap_ci.csv, h1..h6 result CSVs, and the
# figure-3/figure-5/etc PNGs.

# The two figure-regeneration scripts read the analysis outputs from
# the current working directory, so cd into the analysis folder first:
cd ./data/analysis
python ../../analysis/regenerate_pareto.py
python ../../analysis/cross_seed_figure.py
cd ../..

# Sensitivity analysis (Section V-I, VI-K) takes the simulation
# output directory as a positional argument:
python analysis/sensitivity_analysis.py ./data
```

This regenerates Tables IX, X, XI, XII, XIII, XIV and Figures 3, 4, 5, 6
from the paper, using only the released data.

### 4. Re-run simulation from scratch (optional, requires CARLA)

```bash
# Each replication takes approximately 8-12 hours on a single workstation.
python src/batch_run_framework_v4.py --seed 1000 --output-dir ./runs/seed_1000
python src/batch_run_framework_v4.py --seed 2000 --output-dir ./runs/seed_2000
python src/batch_run_framework_v4.py --seed 3000 --output-dir ./runs/seed_3000
```

See `docs/reproducibility.md` for detailed step-by-step instructions.

## Pre-registered hypotheses (summary)

| ID | Hypothesis | Result |
|---|---|---|
| H1 | Calibration convergence within τ ± 0.03 in ≥ 2/3 seeds | **pass** (9/9 cells) |
| H2 | Safety non-inferiority of at least one P_τ vs best TTC | **pass** (P_{0.20}, p=0.031) |
| H3a | Monotonic Spearman τ vs (IR, NMR) | **pass** (ρ = ±1) |
| H3b | Pareto-dominance bootstrap ≥ 50% | **pass** (99.9–100%) |
| H4 | Across-seed CV < 0.50 for primary effects | **partial** (rare-event CV up to 0.63) |
| H5 | Every FCA controller beats NORMAL | **pass** (all p < 10⁻³⁹) |
| H6 | Comfort non-inferiority vs T_{2.5} | **pass** (6/6) |
| H7 | Strict tail-suppression vs best TTC (post-hoc) | **does not pass** |

See paper Tables V (deviations) and XI (hypothesis verdicts) for detail.

## License

[See LICENSE file]

## Patent notice

This work relates to USPTO Application No. 19/533,330. See PATENTS.md.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{hossain2026fca,
  author  = {Hossain, M. B. and Kamal, M. A. S. and Rahman, S. S. and
             Tayan, O. and Mehedi, I. M. and Showail, A.},
  title   = {Real-Time Risk-Aware Supervisory Control for Autonomous
             Driving with Quantile-Calibrated Safety Envelopes},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  year    = {2026},
  note    = {Under review}
}
```

## Contact

Mohammad Belayet Hossain (corresponding author):
m.hossain@upm.edu.sa
