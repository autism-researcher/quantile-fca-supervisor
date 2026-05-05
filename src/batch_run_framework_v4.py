#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner v4 - calibration framework edition.

Differences from v3:
  - Drops Town04 from the experimental design (justified by pilot:
    45.6% low-interaction, 2.4% baseline collision rate vs 20%+ elsewhere).
  - Calibrates at MULTIPLE target violation rates (default 0.10, 0.15, 0.20),
    producing a Pareto trade-off curve rather than a single point.
  - Evaluates a TTC-tuning grid (default thresholds 2.0, 2.5, 3.0, 3.5 s)
    so the framework comparison is "principled calibration vs manual TTC grid".
  - Per-seed output directories so multi-seed replications don't clobber each other.
  - Resume logic: skips any (episode_id, controller_label) pair whose output exists.
  - Atomic-ish writes: completed = BOTH tick CSV and summary CSV present.

Output layout (per seed):
  <out_dir>/
    seed_<seed0>/
      episodes.csv
      stageA_normal/
        ep_NNNNN_normal.csv
        ep_NNNNN_normal_episode_summary.csv
      calibrations/
        calibration_010.json
        calibration_015.json
        calibration_020.json
      stageB_eval/
        ep_NNNNN_<label>.csv
        ep_NNNNN_<label>_episode_summary.csv
        (where <label> in {normal, ttc_20, ttc_25, ttc_30, ttc_35,
                            innov_010, innov_015, innov_020})
      eval_master.csv

  <out_dir>/
    failures.csv              (cross-seed failures log, appended)

Usage (typical 3-seed run):
  python batch_run_framework_v4.py --seed0 1000 --out_dir results_framework ...
  python batch_run_framework_v4.py --seed0 2000 --out_dir results_framework ...
  python batch_run_framework_v4.py --seed0 3000 --out_dir results_framework ...
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import carla


# Town04 is excluded from the design (see pilot analysis).
DEFAULT_TOWNS = ["Town03", "Town05"]
SCENARIOS = ["stopped_vehicle", "sudden_brake_lead", "cut_in"]
WEATHERS = ["rain_fog", "clear"]

# Calibration target rates.
DEFAULT_TARGET_RATES = [0.10, 0.15, 0.20]

# TTC tuning grid (thresholds in seconds).
DEFAULT_TTC_GRID = [2.0, 2.5, 3.0, 3.5]


def get_available_towns(host, port, timeout_s=10.0):
    client = carla.Client(host, int(port))
    client.set_timeout(float(timeout_s))
    maps = client.get_available_maps()
    out = set()
    for m in maps:
        s = str(m)
        if "Town" in s:
            out.add(s.split("/")[-1])
    return out


def episode_param_sampler(rng, town):
    """Same sampler as v3 (anti-brake-paradox, no tailgater)."""
    scenario = rng.choice(
        ["sudden_brake_lead", "cut_in", "stopped_vehicle"],
        p=[0.4, 0.4, 0.2]
    ).item()

    traffic = int(rng.integers(40, 71))
    lead_ahead_m = float(rng.uniform(12.0, 22.0))
    lead_brake_time_s = float(rng.uniform(4.0, 8.0))
    cut_in_time_s = float(rng.uniform(3.0, 6.0))
    cut_in_force = 1
    tailgate_dist_m = 0.0
    tailgate_speedup_pct = 0.0
    n_events = 1
    ego_speedup_pct = float(rng.uniform(-9.0, -3.0))
    global_headway_m = float(rng.uniform(0.8, 1.3))
    weather = str(rng.choice(["clear", "rain_fog"], p=[0.80, 0.20]).item())
    spawn_index = int(rng.integers(0, 120))

    return dict(
        town=str(town),
        traffic=traffic,
        scenario=str(scenario),
        lead_ahead_m=lead_ahead_m,
        lead_brake_time_s=lead_brake_time_s,
        cut_in_time_s=cut_in_time_s,
        cut_in_force=cut_in_force,
        tailgate_dist_m=tailgate_dist_m,
        tailgate_speedup_pct=tailgate_speedup_pct,
        n_events=n_events,
        ego_speedup_pct=ego_speedup_pct,
        global_headway_m=global_headway_m,
        weather=str(weather),
        spawn_index=spawn_index,
    )


def run_one(logger_py, python_exe, out_csv, logger_mode, label,
            ep, ep_id, common,
            B_use, softness_use,
            ttc_thresh_override=None, ttc_slow_pct_override=None,
            max_retries=2, retry_sleep_s=3.0):
    """
    Run a single episode in a specific controller configuration.

    logger_mode: one of {"normal", "ttc_baseline", "innov"} (logger constraint)
    label:       filename tag - "normal", "ttc_25", "innov_015", etc.
    """
    ttc_thresh = ttc_thresh_override if ttc_thresh_override is not None else common["ttc_thresh"]
    ttc_slow_pct = ttc_slow_pct_override if ttc_slow_pct_override is not None else common["ttc_slow_pct"]

    cmd = [
        python_exe, logger_py,
        "--mode", logger_mode,
        "--out_csv", str(out_csv),
        "--episode_id", str(ep_id),

        "--town", ep["town"],
        "--fps", str(common["fps"]),
        "--duration", str(common["duration"]),
        "--warmup_s", str(common["warmup_s"]),
        "--host", common["host"],
        "--port", str(common["port"]),
        "--tm_port", str(common["tm_port"]),
        "--client_timeout_s", str(common["client_timeout_s"]),
        "--connect_retry_s", str(common["connect_retry_s"]),

        "--seed", str(common["seed0"] + ep_id),
        "--tm_seed", str(common["tm_seed"]),
        "--profile", common["profile"],

        "--traffic", str(ep["traffic"]),
        "--weather", ep["weather"],
        "--scenario", ep["scenario"],
        "--lead_ahead_m", str(ep["lead_ahead_m"]),
        "--lead_brake_time_s", str(ep["lead_brake_time_s"]),
        "--lead_brake_hold_s", str(common["lead_brake_hold_s"]),
        "--cut_in_time_s", str(ep["cut_in_time_s"]),
        "--cut_in_force", str(ep["cut_in_force"]),
        "--tailgate_dist_m", str(ep["tailgate_dist_m"]),
        "--tailgate_speedup_pct", str(ep["tailgate_speedup_pct"]),
        "--n_events", str(ep["n_events"]),
        "--ego_speedup_pct", str(ep["ego_speedup_pct"]),
        "--global_headway_m", str(ep["global_headway_m"]),
        "--spawn_index", str(ep["spawn_index"]),

        "--ego_speed_cap_kmh", str(common["ego_speed_cap_kmh"]),
        "--traffic_speed_cap_kmh", str(common["traffic_speed_cap_kmh"]),

        "--ttc_thresh", str(ttc_thresh),
        "--ttc_slow_pct", str(ttc_slow_pct),
        "--ttc_headway_m", str(common["ttc_headway_m"]),

        "--near_miss_ttc_s", str(common["near_miss_ttc_s"]),
        "--ttc_metric_max_s", str(common["ttc_metric_max_s"]),
        "--ego_max_delta_pct_per_s", str(common["ego_max_delta_pct_per_s"]),

        "--innov_ema_tau_s", str(common["innov_ema_tau_s"]),
        "--innov_max_delta_pct_per_s", str(common["innov_max_delta_pct_per_s"]),
        "--innov_max_slow_pct", str(common["innov_max_slow_pct"]),
        "--innov_headway_gain_m", str(common["innov_headway_gain_m"]),
        "--innov_lane_freeze_u", str(common["innov_lane_freeze_u"]),
        "--innov_u_gamma", str(common["innov_u_gamma"]),

        "--speed_ema_tau_s", str(common["speed_ema_tau_s"]),

        "--risk_profile", common["risk_profile"],
        "--hybrid_physics", str(common["hybrid_physics"]),
        "--spawn_bias_close", "1" if common["spawn_bias_close"] else "0",
        "--spawn_ring_min_m", str(common["spawn_ring_min_m"]),
        "--spawn_ring_max_m", str(common["spawn_ring_max_m"]),

        "--sensor_noise_level", str(common["sensor_noise_level"]),
        "--sensor_pos_noise_m", str(common["sensor_pos_noise_m"]),
        "--sensor_vel_noise_mps", str(common["sensor_vel_noise_mps"]),
        "--control_latency_s", str(common["control_latency_s"]),
        "--measure_compute_time", str(common["measure_compute_time"]),

        "--B_override", str(B_use),
        "--debug_overlay", "1" if common["debug_overlay"] else "0",
    ]
    if softness_use is not None:
        cmd += ["--softness_override", str(softness_use)]

    print("[RUN]", " ".join(cmd))
    import time
    last_rc = 0
    for attempt in range(int(max_retries)):
        try:
            subprocess.run(cmd, check=True)
            last_rc = 0
            break
        except subprocess.CalledProcessError as e:
            last_rc = int(getattr(e, 'returncode', -1))
            print("[WARN] logger failed (attempt {}/{}) rc={}".format(
                attempt+1, int(max_retries), last_rc))
            time.sleep(float(retry_sleep_s))
    if last_rc != 0:
        raise RuntimeError("logger failed after retries rc={}".format(last_rc))


def calibrate_from_normal(stageA_dir, episodes_df, target_violation_rate, out_path):
    max_Rs = []
    for ep_id in episodes_df["episode_id"].tolist():
        tick_csv = stageA_dir / "ep_{:05d}_normal.csv".format(int(ep_id))
        if not tick_csv.exists():
            continue
        df = pd.read_csv(tick_csv)
        if "R" not in df.columns or df.empty:
            continue
        max_Rs.append(float(df["R"].max()))

    if not max_Rs:
        raise RuntimeError("Calibration failed: no NORMAL per-tick CSVs found or R missing.")

    x = np.array(max_Rs, dtype=float)
    q = float(1.0 - target_violation_rate)
    B = float(np.quantile(x, q))

    q_lo = float(np.quantile(x, 0.25))
    q_hi = float(np.quantile(x, 0.75))
    spread = float(max(1e-6, q_hi - q_lo))
    softness = float(np.clip(0.5 * spread, 0.06, 0.18))

    payload = dict(
        target_violation_rate=float(target_violation_rate),
        q=q,
        B_calibrated=B,
        softness_calibrated=softness,
        meta=dict(q_lo=q_lo, q_hi=q_hi, spread=spread, n=int(len(x))),
        calibration_from="NORMAL stageA: per-episode max_R",
    )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return B, softness, payload


def episode_complete(stage_dir, ep_id, label):
    """Both tick CSV and summary CSV must exist to count as done."""
    tick = stage_dir / "ep_{:05d}_{}.csv".format(int(ep_id), label)
    summ = stage_dir / "ep_{:05d}_{}_episode_summary.csv".format(int(ep_id), label)
    return tick.exists() and summ.exists()


def cleanup_partial(stage_dir, ep_id, label):
    """Delete partial files for an ep/label so it re-runs cleanly."""
    tick = stage_dir / "ep_{:05d}_{}.csv".format(int(ep_id), label)
    summ = stage_dir / "ep_{:05d}_{}_episode_summary.csv".format(int(ep_id), label)
    for p in (tick, summ):
        if p.exists():
            print("[CLEAN] removing partial {}".format(p.name))
            p.unlink()


def load_and_merge_eval(stageB_dir, episodes_df, controller_labels):
    rows = []
    for ep_id in episodes_df["episode_id"].tolist():
        for label in controller_labels:
            summ = stageB_dir / "ep_{:05d}_{}_episode_summary.csv".format(int(ep_id), label)
            if not summ.exists():
                continue
            df = pd.read_csv(summ)
            if df.empty:
                continue
            df["episode_id"] = int(ep_id)
            df["controller_label"] = label
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    summ_df = pd.concat(rows, ignore_index=True)
    # Build plan-column subset: episode_id (for merge) plus all columns
    # that are NOT already in the summary (avoids _x/_y suffix collisions).
    plan_subset_cols = ["episode_id"] + [
        c for c in episodes_df.columns
        if c not in summ_df.columns and c != "episode_id"
    ]
    plan_subset = episodes_df[plan_subset_cols].copy()
    return summ_df.merge(plan_subset, on="episode_id", how="left", validate="many_to_one")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--out_dir", required=True,
                    help="Top-level output dir. Per-seed subdirs created inside.")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed0", type=int, required=True,
                    help="Seed for episode plan; one seed per replication.")
    ap.add_argument("--calibration_fraction", type=float, default=0.40)

    ap.add_argument("--target_violation_rates", nargs="+", type=float,
                    default=DEFAULT_TARGET_RATES)
    ap.add_argument("--ttc_grid", nargs="+", type=float,
                    default=DEFAULT_TTC_GRID)

    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--logger_py", default="carla_episode_logger_PUB_v3.py")

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=8000)
    ap.add_argument("--client_timeout_s", type=float, default=60.0)
    ap.add_argument("--connect_retry_s", type=float, default=120.0)

    ap.add_argument("--towns", nargs="+", default=DEFAULT_TOWNS,
                    help="Towns to use. Default excludes Town04.")
    ap.add_argument("--severity", type=float, default=1.10)
    ap.add_argument("--lead_brake_hold_s", type=float, default=1.2)

    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup_s", type=float, default=3.0)
    ap.add_argument("--tm_seed", type=int, default=0)
    ap.add_argument("--profile", choices=["normal", "sensitive"], default="normal")

    ap.add_argument("--ego_speed_cap_kmh", type=float, default=65.0)
    ap.add_argument("--traffic_speed_cap_kmh", type=float, default=60.0)

    ap.add_argument("--ttc_thresh", type=float, default=2.5,
                    help="Default TTC thresh (used by ttc_25 in the grid).")
    ap.add_argument("--ttc_slow_pct", type=float, default=28.0)
    ap.add_argument("--ttc_headway_m", type=float, default=12.0)
    ap.add_argument("--near_miss_ttc_s", type=float, default=3.0)
    ap.add_argument("--ttc_metric_max_s", type=float, default=6.0)
    ap.add_argument("--ego_max_delta_pct_per_s", type=float, default=25.0)

    ap.add_argument("--B_init", type=float, default=0.32)
    ap.add_argument("--innov_ema_tau_s", type=float, default=0.55)
    ap.add_argument("--innov_max_delta_pct_per_s", type=float, default=40.0)
    ap.add_argument("--innov_max_slow_pct", type=float, default=35.0)
    ap.add_argument("--innov_headway_gain_m", type=float, default=8.0)
    ap.add_argument("--innov_lane_freeze_u", type=float, default=0.45)
    ap.add_argument("--innov_u_gamma", type=float, default=0.75)
    ap.add_argument("--speed_ema_tau_s", type=float, default=0.40)

    ap.add_argument("--risk_profile", choices=["normal", "aggressive"], default="aggressive")
    ap.add_argument("--hybrid_physics", type=int, default=0)
    ap.add_argument("--spawn_bias_close", type=int, default=1)
    ap.add_argument("--spawn_ring_min_m", type=float, default=8.0)
    ap.add_argument("--spawn_ring_max_m", type=float, default=30.0)

    ap.add_argument("--sensor_noise_level",
                    choices=["none", "low", "medium", "high"], default="none")
    ap.add_argument("--sensor_pos_noise_m", type=float, default=-1.0)
    ap.add_argument("--sensor_vel_noise_mps", type=float, default=-1.0)
    ap.add_argument("--control_latency_s", type=float, default=0.10)
    ap.add_argument("--measure_compute_time", type=int, default=1)

    ap.add_argument("--max_retries", type=int, default=3)
    ap.add_argument("--retry_sleep_s", type=float, default=3.0)
    ap.add_argument("--continue_on_fail", type=int, default=1)
    ap.add_argument("--debug_overlay", type=int, default=0)

    args = ap.parse_args()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    seed_dir = out_root / "seed_{}".format(int(args.seed0))
    seed_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        host=args.host, port=args.port, tm_port=args.tm_port,
        client_timeout_s=args.client_timeout_s, connect_retry_s=args.connect_retry_s,
        fps=int(args.fps), duration=float(args.duration), warmup_s=float(args.warmup_s),
        seed0=int(args.seed0), tm_seed=int(args.tm_seed), profile=args.profile,
        ego_speed_cap_kmh=float(args.ego_speed_cap_kmh),
        traffic_speed_cap_kmh=float(args.traffic_speed_cap_kmh),
        lead_brake_hold_s=float(args.lead_brake_hold_s),
        ttc_thresh=float(args.ttc_thresh),
        ttc_slow_pct=float(args.ttc_slow_pct),
        ttc_headway_m=float(args.ttc_headway_m),
        near_miss_ttc_s=float(args.near_miss_ttc_s),
        ttc_metric_max_s=float(args.ttc_metric_max_s),
        ego_max_delta_pct_per_s=float(args.ego_max_delta_pct_per_s),
        innov_ema_tau_s=float(args.innov_ema_tau_s),
        innov_max_delta_pct_per_s=float(args.innov_max_delta_pct_per_s),
        innov_max_slow_pct=float(args.innov_max_slow_pct),
        innov_headway_gain_m=float(args.innov_headway_gain_m),
        innov_lane_freeze_u=float(args.innov_lane_freeze_u),
        innov_u_gamma=float(args.innov_u_gamma),
        speed_ema_tau_s=float(args.speed_ema_tau_s),
        risk_profile=args.risk_profile,
        hybrid_physics=int(args.hybrid_physics),
        spawn_bias_close=int(args.spawn_bias_close),
        spawn_ring_min_m=float(args.spawn_ring_min_m),
        spawn_ring_max_m=float(args.spawn_ring_max_m),
        sensor_noise_level=args.sensor_noise_level,
        sensor_pos_noise_m=float(args.sensor_pos_noise_m),
        sensor_vel_noise_mps=float(args.sensor_vel_noise_mps),
        control_latency_s=float(args.control_latency_s),
        measure_compute_time=int(args.measure_compute_time),
        debug_overlay=int(args.debug_overlay),
    )

    # Episodes plan (resume-aware).
    rng = np.random.default_rng(int(args.seed0))
    episodes = []
    for ep_id in range(int(args.episodes)):
        town = str(args.towns[int(ep_id) % len(args.towns)])
        ep = episode_param_sampler(rng, town)
        sev = float(max(0.90, min(1.25, args.severity)))
        ep["lead_ahead_m"] = float(np.clip(ep["lead_ahead_m"] / (1.0 + 0.15 * (sev - 1.0)), 10.0, 24.0))
        ep["lead_brake_time_s"] = float(np.clip(ep["lead_brake_time_s"] / (1.0 + 0.15 * (sev - 1.0)), 3.5, 9.0))
        ep["cut_in_time_s"] = float(np.clip(ep["cut_in_time_s"] / (1.0 + 0.15 * (sev - 1.0)), 2.5, 7.0))
        ep["tailgate_dist_m"] = float(ep["tailgate_dist_m"])
        ep["ego_speedup_pct"] = float(np.clip(ep["ego_speedup_pct"] * min(sev, 1.05), -10.0, 0.0))
        ep["episode_id"] = int(ep_id)
        episodes.append(ep)

    episodes_csv = seed_dir / "episodes.csv"
    if episodes_csv.exists():
        print("[RESUME] reusing existing episode plan {}".format(episodes_csv))
        episodes_df = pd.read_csv(episodes_csv)
    else:
        episodes_df = pd.DataFrame(episodes)
        episodes_df.to_csv(episodes_csv, index=False)
        print("[OK] wrote episode plan {} (n={})".format(episodes_csv, len(episodes_df)))

    # Stage A
    stageA = seed_dir / "stageA_normal"
    stageA.mkdir(parents=True, exist_ok=True)

    failures = []
    print("\n=== STAGE A: calibration data (NORMAL controller) ===")
    for _, ep in episodes_df.iterrows():
        ep_id = int(ep["episode_id"])
        if episode_complete(stageA, ep_id, "normal"):
            continue
        cleanup_partial(stageA, ep_id, "normal")
        out_csv = stageA / "ep_{:05d}_normal.csv".format(ep_id)
        try:
            run_one(args.logger_py, args.python_exe, out_csv, "normal", "normal",
                    ep.to_dict(), ep_id, common,
                    B_use=float(args.B_init), softness_use=None,
                    max_retries=args.max_retries, retry_sleep_s=args.retry_sleep_s)
        except Exception as e:
            failures.append(dict(seed0=args.seed0, stage="stageA_normal",
                                 episode_id=ep_id, label="normal", error=str(e)))
            if not int(args.continue_on_fail):
                raise
            print("[WARN] skipping failed stageA episode", ep_id, e)

    # Episodes that produced a Stage A tick CSV (used for both calibration and eval).
    have_normal = set()
    for p in stageA.glob("ep_*_normal.csv"):
        if p.name.endswith("_episode_summary.csv"):
            continue
        try:
            have_normal.add(int(p.stem.split("_")[1]))
        except Exception:
            pass
    ok_stageA = episodes_df[episodes_df["episode_id"].isin(have_normal)].copy()
    if len(ok_stageA) == 0:
        raise RuntimeError("Stage A produced no completed episodes; cannot calibrate.")

    n_total = len(ok_stageA)
    n_cal = max(1, int(round(float(args.calibration_fraction) * n_total)))
    cal_set  = ok_stageA.iloc[:n_cal].copy()
    eval_set = ok_stageA.iloc[n_cal:].copy() if n_cal < n_total else ok_stageA.copy()
    print("[SPLIT] calibration n={}, evaluation n={}".format(len(cal_set), len(eval_set)))

    # Multi-tau calibration.
    cal_dir = seed_dir / "calibrations"
    cal_dir.mkdir(parents=True, exist_ok=True)
    calibrations = {}  # tau_str -> (B, softness, tau_float)
    print("\n=== CALIBRATION (multi-tau) ===")
    for tau in args.target_violation_rates:
        tau_str = "{:03d}".format(int(round(tau * 100)))
        cal_path = cal_dir / "calibration_{}.json".format(tau_str)
        if cal_path.exists():
            print("[RESUME] reusing {}".format(cal_path.name))
            payload = json.loads(cal_path.read_text(encoding="utf-8"))
            B = float(payload["B_calibrated"])
            soft = float(payload["softness_calibrated"])
        else:
            B, soft, payload = calibrate_from_normal(stageA, cal_set, float(tau), cal_path)
            print("[OK] tau={:.2f}  B={:.4f}  softness={:.4f}  n={}".format(
                tau, B, soft, payload['meta']['n']))
        calibrations[tau_str] = (B, soft, float(tau))

    # Stage B controller plan.
    # (label, logger_mode, B_use, softness_use, ttc_thresh_override, ttc_slow_pct_override)
    controller_plan = []
    controller_plan.append(("normal", "normal", float(args.B_init), None, None, None))
    for thr in args.ttc_grid:
        thr_str = "{:02d}".format(int(round(thr * 10)))
        label = "ttc_{}".format(thr_str)
        controller_plan.append((label, "ttc_baseline", float(args.B_init), None,
                                float(thr), float(args.ttc_slow_pct)))
    for tau_str, (B, soft, _tau) in calibrations.items():
        label = "innov_{}".format(tau_str)
        controller_plan.append((label, "innov", B, soft, None, None))

    print("\n=== STAGE B: evaluation ===")
    print("  Controllers ({}): {}".format(
        len(controller_plan), [p[0] for p in controller_plan]))
    print("  Episodes: {}".format(len(eval_set)))
    print("  Total runs: {}".format(len(eval_set) * len(controller_plan)))

    stageB = seed_dir / "stageB_eval"
    stageB.mkdir(parents=True, exist_ok=True)

    for _, ep in eval_set.iterrows():
        ep_id = int(ep["episode_id"])
        for label, logger_mode, B_use, soft_use, ttc_thr_ov, ttc_slow_ov in controller_plan:
            if episode_complete(stageB, ep_id, label):
                continue
            cleanup_partial(stageB, ep_id, label)
            out_csv = stageB / "ep_{:05d}_{}.csv".format(ep_id, label)
            try:
                run_one(args.logger_py, args.python_exe, out_csv, logger_mode, label,
                        ep.to_dict(), ep_id, common,
                        B_use=B_use, softness_use=soft_use,
                        ttc_thresh_override=ttc_thr_ov,
                        ttc_slow_pct_override=ttc_slow_ov,
                        max_retries=args.max_retries, retry_sleep_s=args.retry_sleep_s)
            except Exception as e:
                failures.append(dict(seed0=args.seed0, stage="stageB_eval",
                                     episode_id=ep_id, label=label, error=str(e)))
                if not int(args.continue_on_fail):
                    raise
                print("[WARN] failed stageB ep={} label={} err={}".format(ep_id, label, e))

    if failures:
        fpath = out_root / "failures.csv"
        df_new = pd.DataFrame(failures)
        if fpath.exists():
            df_old = pd.read_csv(fpath)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        df_new.to_csv(fpath, index=False)
        print("[INFO] wrote {} new failures to {}".format(len(failures), fpath))

    labels = [p[0] for p in controller_plan]
    eval_df = load_and_merge_eval(stageB, eval_set, labels)
    if eval_df.empty:
        print("[WARN] no episode summaries found in stageB_eval. Logger probably failed.")
        return

    cal_lookup = {}
    for tau_str, (B, soft, tau) in calibrations.items():
        cal_lookup["innov_{}".format(tau_str)] = (B, soft, tau)
    eval_df["B_used"] = eval_df["controller_label"].map(
        lambda lbl: cal_lookup[lbl][0] if lbl in cal_lookup else float("nan"))
    eval_df["softness_used"] = eval_df["controller_label"].map(
        lambda lbl: cal_lookup[lbl][1] if lbl in cal_lookup else float("nan"))
    eval_df["target_tau"] = eval_df["controller_label"].map(
        lambda lbl: cal_lookup[lbl][2] if lbl in cal_lookup else float("nan"))
    eval_df["seed0"] = int(args.seed0)

    eval_master = seed_dir / "eval_master.csv"
    eval_df.to_csv(eval_master, index=False)
    print("\n[OK] wrote {} (n={} controller-runs)".format(eval_master, len(eval_df)))
    print("[DONE] seed {} complete.".format(args.seed0))


if __name__ == "__main__":
    main()
