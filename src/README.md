# `src/` — CARLA simulation and analysis pipeline

This directory contains the four Python scripts that constitute the
end-to-end pipeline used to produce the results in the paper.

| Script | Role | Cited in paper |
|---|---|---|
| `batch_run_framework_v4.py` | Top-level batch orchestrator. Generates the 600-episode-per-seed schedule, drives Stage A calibration and Stage B paired evaluation, calibrates per-seed boundaries `B_τ` and softness `δ` for each `τ ∈ {0.10, 0.15, 0.20}`, and writes per-seed `eval_master.csv`. **This is the entry point for re-running the simulation from scratch.** | Data Availability §; Algorithm 1 reference |
| `carla_episode_logger_PUB_v3.py` | Per-episode CARLA worker. Connects to CARLA 0.9.13, manages actors, applies the selected controller (NORMAL / TTC-grid / Innov-τ) through the Traffic Manager, logs per-tick telemetry (8 risk features, intervention factor, TM setpoints, jerk, lane offset, etc.), and writes one CSV + one summary CSV per episode. Invoked once per `(episode, controller)` pair by the orchestrator. | §V-A, §V-D, Algorithms 1 & 2 |
| `analyze_framework.py` | Statistical analysis pipeline. Reads the per-seed `eval_master.csv` files, combines them, and produces every paper table (IX, X, XI, XII, XIII, XIV) and Figures 3, 4, 5 with bootstrap CIs (5,000 resamples). Runs hypothesis tests H1–H7 and writes both CSV and TeX outputs. | Tables IX–XIV, Figs. 3–5 |
| `batch_run_calibrate_PUB_v3.py` | **Predecessor** of `batch_run_framework_v4.py`. Single-`τ` calibration runner used during the pilot study (cf. paper §VI-A and pre-registration §6). Retained for traceability of the pilot data; **not used by the main study.** New users should run `batch_run_framework_v4.py` instead. | Pilot Stage A reference |

## Typical invocation

```bash
# (assuming CARLA 0.9.13 server is running on localhost:2000)
python src/batch_run_framework_v4.py --seed0 1000 --out_dir ./runs/seed_1000
python src/batch_run_framework_v4.py --seed0 2000 --out_dir ./runs/seed_2000
python src/batch_run_framework_v4.py --seed0 3000 --out_dir ./runs/seed_3000
python src/analyze_framework.py ./runs
```

For the analysis-only path (no CARLA required, using the released
Zenodo archive as input), see [`../docs/reproducibility.md`](../docs/reproducibility.md).

## Notes

- The `_PUB_v3` suffix on the logger and pilot orchestrator denotes the
  publication-grade revision that fixes the anti-brake-paradox issue
  documented in the logger header (see top of
  `carla_episode_logger_PUB_v3.py`).
- All four scripts target Python 3.7+ for compatibility with the Python
  interpreter shipped inside CARLA 0.9.13 (the analysis script also
  works on 3.9+).

