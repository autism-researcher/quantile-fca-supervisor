#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch runner — Python 3.7 compatible.

This version is tuned for:
- Realistic (not racing) speeds + smooth motion
- Still enough risk to differentiate:
  normal vs TTC baseline vs your innov controller

What creates "risk" here (without making CARLA unstable):
- close lead/stopped vehicle (7–12m)
- early braking (0.8–1.8s)
- cut-in events (higher probability)
- locally packed traffic (spawn ring)

What avoids "abnormal high speed" & jitter:
- ego_speedup_pct near normal (-12..+6)
- traffic speed capped in logger (defaults: 75 km/h)
- global random lane changes disabled in logger
- warmup period before logging

Outputs:
- episodes.csv
- calibration.json
- eval_master.csv
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


SCENARIOS = ["stopped_vehicle", "sudden_brake_lead", "cut_in", "tailgater", "mixed"]
WEATHERS = ["rain_fog", "clear"]


def get_available_towns(host, port, timeout_s=10.0):
    """
    Returns a set of short map names installed on this CARLA server, e.g. {"Town03","Town10HD"}.
    """
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
    """
    Publication-grade scenario sampler v9.2 / v3 (2026-04-29, third iteration).

    Changes vs v9.1:
      - Lead distance increased to 12-22 m. The previous 3.5-6.0 m made
        the ego's autopilot refuse to accelerate (it saw a slow car right
        in front and slammed the brakes), producing 0 m/s ego speed for
        the entire episode. With 12-22 m and the lead driving at ~10%
        slower than limit, the ego catches up over ~5-8 s.
      - Trigger time pushed to 4-8 s so it lands AFTER the ego has caught
        up to the lead (rather than firing while the gap is still 15 m).
      - Tailgater dropped (kept from v9.1).
      - Geometric forward cone TTC (kept from v9.1, in the logger).
      - Ego configured to ignore lights (in the logger) so red-light traps
        no longer trap the ego at 0 m/s.
    """

    scenario = rng.choice(
        ["sudden_brake_lead", "cut_in", "stopped_vehicle"],
        p=[0.4, 0.4, 0.2]
    ).item()

    traffic = int(rng.integers(40, 71))

    # Larger lead distance avoids brake-paradox.
    lead_ahead_m = float(rng.uniform(12.0, 22.0))
    # Trigger time pushed back so the ego has time to close on the lead first.
    lead_brake_time_s = float(rng.uniform(4.0, 8.0))
    cut_in_time_s = float(rng.uniform(3.0, 6.0))
    cut_in_force = 1

    tailgate_dist_m = 0.0
    tailgate_speedup_pct = 0.0
    n_events = 1

    # Ego slightly faster than limit so it catches up to the moderate lead.
    ego_speedup_pct = float(rng.uniform(-8.0, -3.0))

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
def run_one(logger_py, python_exe, out_csv, mode,
            ep, ep_id, common,
            B_use, softness_use, max_retries=2, retry_sleep_s=3.0):
    cmd = [
        python_exe, logger_py,
        "--mode", mode,
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

        "--ttc_thresh", str(common["ttc_thresh"]),
        "--ttc_slow_pct", str(common["ttc_slow_pct"]),
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
            print(f"[WARN] logger failed (attempt {attempt+1}/{int(max_retries)}) rc={last_rc}")
            time.sleep(float(retry_sleep_s))
    if last_rc != 0:
        raise RuntimeError(f"logger failed after retries rc={last_rc}")
def calibrate_from_normal(out_dir, episodes_df, target_violation_rate):
    max_Rs = []
    for ep_id in episodes_df["episode_id"].tolist():
        tick_csv = out_dir / "stageA_normal" / "ep_{:05d}_normal.csv".format(int(ep_id))
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
        calibration_from="NORMAL stageA: per-episode max_R in realistic-risk environment",
    )
    (out_dir / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return B, softness, payload


def load_and_merge_eval(out_dir, episodes_df, stage_name, modes):
    rows = []
    for ep_id in episodes_df["episode_id"].tolist():
        for mode in modes:
            summ = out_dir / stage_name / "ep_{:05d}_{}_episode_summary.csv".format(int(ep_id), mode)
            if not summ.exists():
                continue
            df = pd.read_csv(summ)
            if df.empty:
                continue
            df["episode_id"] = int(ep_id)
            df["mode"] = mode
            rows.append(df)

    if not rows:
        return pd.DataFrame()

    summ_df = pd.concat(rows, ignore_index=True)
    merged = summ_df.merge(episodes_df, on="episode_id", how="left", validate="many_to_one")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="batch_out_validated_realistic")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--severity", type=float, default=1.15,
                    help="Risk severity multiplier ~[0.7..1.3]. Multiplies closeness/timing (batch only).")
    ap.add_argument("--python_exe", default=sys.executable)
    ap.add_argument("--logger_py", default="carla_episode_logger_q1d_science.py")

    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm_port", type=int, default=8000)

    ap.add_argument("--towns", nargs="+", default=["Town03", "Town04", "Town05"])
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--warmup_s", type=float, default=3.0)

    ap.add_argument("--seed0", type=int, default=100)
    ap.add_argument("--tm_seed", type=int, default=0)
    ap.add_argument("--profile", choices=["normal","sensitive"], default="normal")

    ap.add_argument("--B_init", type=float, default=0.32)
    ap.add_argument("--lead_brake_hold_s", type=float, default=1.2,
                    help="Lead vehicle hard-braking duration forwarded to the episode logger.")

    # Speed caps (km/h). Set <=0 to disable a cap.
    ap.add_argument("--ego_speed_cap_kmh", type=float, default=72.0)
    ap.add_argument("--traffic_speed_cap_kmh", type=float, default=68.0)

    ap.add_argument("--ttc_thresh", type=float, default=3.0)
    ap.add_argument("--ttc_slow_pct", type=float, default=28.0)
    ap.add_argument("--ttc_headway_m", type=float, default=12.0)

    ap.add_argument("--innov_ema_tau_s", type=float, default=0.55)
    ap.add_argument("--innov_max_delta_pct_per_s", type=float, default=40.0)
    ap.add_argument("--innov_max_slow_pct", type=float, default=38.0)
    ap.add_argument("--innov_headway_gain_m", type=float, default=8.0)
    ap.add_argument("--innov_lane_freeze_u", type=float, default=0.45)

    ap.add_argument("--speed_ema_tau_s", type=float, default=0.40)

    ap.add_argument("--client_timeout_s", type=float, default=60.0)
    ap.add_argument("--connect_retry_s", type=float, default=120.0)
    ap.add_argument("--max_retries", type=int, default=2,
                    help="Retry a failed episode logger call this many times before skipping the episode.")
    ap.add_argument("--retry_sleep_s", type=float, default=3.0,
                    help="Seconds to wait between retries.")
    ap.add_argument("--continue_on_fail", type=int, default=1,
                    help="If 1, continue batch even if some episodes fail; writes failures.csv.")
    ap.add_argument("--debug_overlay", type=int, default=0)

    ap.add_argument("--target_violation_rate", type=float, default=0.15)

    ap.add_argument("--risk_profile", choices=["normal","aggressive"], default="aggressive")
    ap.add_argument("--hybrid_physics", type=int, default=0)

    # Local packing (riskier interactions)
    ap.add_argument("--spawn_bias_close", type=int, default=1)
    ap.add_argument("--spawn_ring_min_m", type=float, default=8.0)
    ap.add_argument("--spawn_ring_max_m", type=float, default=30.0)

    # ----------------------------------------------------------
    # PUBLICATION REALISM PATCHES
    # ----------------------------------------------------------
    ap.add_argument("--sensor_noise_level",
                    choices=["none", "light", "medium", "heavy"],
                    default="none",
                    help="Preset sensor-noise level for perception inputs.")
    ap.add_argument("--sensor_pos_noise_m", type=float, default=-1.0,
                    help="Override sigma of headway noise (m). <0 = preset.")
    ap.add_argument("--sensor_vel_noise_mps", type=float, default=-1.0,
                    help="Override sigma of relative-speed noise (m/s). <0 = preset.")
    ap.add_argument("--control_latency_s", type=float, default=0.10,
                    help="End-to-end control latency in seconds.")
    ap.add_argument("--measure_compute_time", type=int, default=1,
                    help="If 1, measure controller compute time per tick.")

    # Calibration / evaluation split (no leakage).
    ap.add_argument("--calibration_fraction", type=float, default=0.40,
                    help="Fraction of OK Stage A episodes used to fit B; rest used for evaluation.")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter towns to those actually installed on this CARLA server
    available_towns = get_available_towns(args.host, args.port, timeout_s=min(30.0, args.client_timeout_s))
    requested_towns = [str(t) for t in args.towns]
    valid_towns = [t for t in requested_towns if t in available_towns]
    invalid_towns = [t for t in requested_towns if t not in available_towns]
    if invalid_towns:
        print("[WARN] These towns are not installed and will be skipped:", ", ".join(invalid_towns))
        print("[INFO] Available towns:", ", ".join(sorted(available_towns)))
    if not valid_towns:
        raise RuntimeError("None of the requested towns are installed. Available: " + ", ".join(sorted(available_towns)))
    args.towns = valid_towns

    # Normalize caps
    ego_cap = None if float(args.ego_speed_cap_kmh) <= 0 else float(args.ego_speed_cap_kmh)
    traffic_cap = None if float(args.traffic_speed_cap_kmh) <= 0 else float(args.traffic_speed_cap_kmh)

    rng = np.random.default_rng(int(args.seed0))

    episodes = []
    for ep_id in range(int(args.episodes)):
        # Balanced multi-town design: for 999 episodes and 3 towns, this gives 333 per town.
        town = str(args.towns[int(ep_id) % len(args.towns)])
        ep = episode_param_sampler(rng, town)

        # Severity scales criticality. Clamps aligned with v9.2 sampler.
        sev = float(max(0.90, min(1.25, args.severity)))
        ep["lead_ahead_m"] = float(np.clip(ep["lead_ahead_m"] / (1.0 + 0.15 * (sev - 1.0)), 10.0, 24.0))
        ep["lead_brake_time_s"] = float(np.clip(ep["lead_brake_time_s"] / (1.0 + 0.15 * (sev - 1.0)), 3.5, 9.0))
        ep["cut_in_time_s"] = float(np.clip(ep["cut_in_time_s"] / (1.0 + 0.15 * (sev - 1.0)), 2.5, 7.0))
        ep["tailgate_dist_m"] = float(ep["tailgate_dist_m"])  # placeholder (scenario dropped)
        ep["ego_speedup_pct"] = float(np.clip(ep["ego_speedup_pct"] * min(sev, 1.05), -10.0, 0.0))

        ep["episode_id"] = int(ep_id)
        episodes.append(ep)

    episodes_df = pd.DataFrame(episodes)
    episodes_df.to_csv(out_dir / "episodes.csv", index=False)

    common = dict(
        host=args.host, port=args.port, tm_port=args.tm_port,
        fps=args.fps, duration=args.duration, warmup_s=args.warmup_s,
        seed0=args.seed0, tm_seed=args.tm_seed,
        profile=args.profile,
        lead_brake_hold_s=float(args.lead_brake_hold_s),

        ego_speed_cap_kmh=ego_cap if ego_cap is not None else -1,
        traffic_speed_cap_kmh=traffic_cap if traffic_cap is not None else -1,

        ttc_thresh=args.ttc_thresh, ttc_slow_pct=args.ttc_slow_pct, ttc_headway_m=args.ttc_headway_m,
        near_miss_ttc_s=3.0,
        ttc_metric_max_s=6.0,
        ego_max_delta_pct_per_s=25.0,

        innov_ema_tau_s=args.innov_ema_tau_s,
        innov_max_delta_pct_per_s=args.innov_max_delta_pct_per_s,
        innov_max_slow_pct=min(args.innov_max_slow_pct, 35.0),
        innov_headway_gain_m=args.innov_headway_gain_m,
        innov_lane_freeze_u=args.innov_lane_freeze_u,
        innov_u_gamma=0.75,

        speed_ema_tau_s=args.speed_ema_tau_s,

        client_timeout_s=args.client_timeout_s,
        connect_retry_s=args.connect_retry_s,
        debug_overlay=bool(args.debug_overlay),

        risk_profile=args.risk_profile,
        spawn_bias_close=bool(args.spawn_bias_close),
        spawn_ring_min_m=float(args.spawn_ring_min_m),
        spawn_ring_max_m=float(args.spawn_ring_max_m),
        hybrid_physics=int(args.hybrid_physics),

        sensor_noise_level=str(args.sensor_noise_level),
        sensor_pos_noise_m=float(args.sensor_pos_noise_m),
        sensor_vel_noise_mps=float(args.sensor_vel_noise_mps),
        control_latency_s=float(args.control_latency_s),
        measure_compute_time=int(args.measure_compute_time),
    )

    # ---------- Stage A (NORMAL only) ----------
    stageA = out_dir / "stageA_normal"
    stageA.mkdir(parents=True, exist_ok=True)

    failures = []

    for _, ep in episodes_df.iterrows():
        ep_id = int(ep["episode_id"])
        out_csv = stageA / "ep_{:05d}_normal.csv".format(ep_id)
        try:
            run_one(args.logger_py, args.python_exe, out_csv, "normal",
                    ep.to_dict(), ep_id, common,
                    B_use=float(args.B_init), softness_use=None,
                    max_retries=args.max_retries, retry_sleep_s=args.retry_sleep_s)
        except Exception as e:
            failures.append(dict(stage="stageA_normal", episode_id=ep_id, mode="normal", error=str(e)))
            if not int(args.continue_on_fail):
                raise
            print("[WARN] skipping failed stageA episode", ep_id, e)

    ok_stageA = episodes_df[episodes_df["episode_id"].isin(
        [int(p.stem.split("_")[1]) for p in stageA.glob("ep_*_normal.csv")]
    )].copy()

    # ---------- Calibration / evaluation split (no leakage) ----------
    # Use the first `calibration_fraction` of OK episodes to fit B and softness.
    # The remaining episodes are used in Stage B for evaluation.
    # This addresses reviewer concerns about calibrating and evaluating on
    # the same seeds.
    n_total = len(ok_stageA)
    n_cal = max(1, int(round(float(args.calibration_fraction) * n_total)))
    cal_set  = ok_stageA.iloc[:n_cal].copy()
    eval_set = ok_stageA.iloc[n_cal:].copy() if n_cal < n_total else ok_stageA.copy()
    print(f"[SPLIT] calibration n={len(cal_set)}, evaluation n={len(eval_set)}")

    B_cal, softness_cal, payload = calibrate_from_normal(out_dir, cal_set, float(args.target_violation_rate))
    print("[OK] calibration:", payload)

    # ---------- Stage B (all modes with calibrated B/softness) ----------
    stageB = out_dir / "stageB_eval"
    stageB.mkdir(parents=True, exist_ok=True)

    modes = ["normal", "ttc_baseline", "innov"]
    for _, ep in eval_set.iterrows():
        ep_id = int(ep["episode_id"])
        for mode in modes:
            out_csv = stageB / "ep_{:05d}_{}.csv".format(ep_id, mode)
            try:
                run_one(args.logger_py, args.python_exe, out_csv, mode,
                        ep.to_dict(), ep_id, common,
                        B_use=float(B_cal), softness_use=float(softness_cal),
                        max_retries=args.max_retries, retry_sleep_s=args.retry_sleep_s)
            except Exception as e:
                failures.append(dict(stage="stageB_eval", episode_id=ep_id, mode=mode, error=str(e)))
                if not int(args.continue_on_fail):
                    raise
                print("[WARN] skipping failed stageB episode={} mode={} err={}".format(ep_id, mode, e))

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "failures.csv", index=False)

    eval_df = load_and_merge_eval(out_dir, eval_set, "stageB_eval", modes)
    if eval_df.empty:
        raise RuntimeError("No episode summaries found in stageB_eval. Check logger outputs.")

    eval_df["B_used"] = float(B_cal)
    eval_df["softness_used"] = float(softness_cal)
    eval_df.to_csv(out_dir / "eval_master.csv", index=False)
    print("[OK] wrote", out_dir / "eval_master.csv")

    # Q1: Effect-size + CI report (journal-ready) — all episodes
    try:
        eff_path = out_dir / "stats_effect_sizes.csv"
        if eff_path.exists():
            eff_path.unlink()  # fresh run
        write_effect_size_report(eval_df, eff_path, label="all")
        print("[OK] wrote", eff_path, "(label=all)")
    except Exception as e:
        print("[WARN] effect-size report (all) failed:", e)

    # Q2: Interaction-only subset — episodes where TTC was valid for at least
    # one tick. This isolates the controller's effect from pure cruise
    # episodes and is the most defensible comparison for the paper.
    try:
        # Per-episode "had any interaction" flag from the NORMAL run.
        had_interaction = (
            eval_df.loc[eval_df["mode"]=="normal", :]
                   .set_index("episode_id")["ttc_valid_ratio"]
                   > 0.0
        )
        interactive_ids = had_interaction[had_interaction].index.tolist()
        eval_df_int = eval_df[eval_df["episode_id"].isin(interactive_ids)].copy()
        if len(eval_df_int) > 0:
            write_effect_size_report(eval_df_int, eff_path, label="interaction_only")
            print(f"[OK] wrote interaction-only stats (n_episodes={len(interactive_ids)})")
            eval_df_int.to_csv(out_dir / "eval_master_interaction_only.csv", index=False)
        else:
            print("[WARN] no interaction-only episodes found")
    except Exception as e:
        print("[WARN] effect-size report (interaction_only) failed:", e)


def cohen_d(x, y):
    """Cohen's d effect size for continuous episode-level metrics."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    sx = float(np.var(x, ddof=1))
    sy = float(np.var(y, ddof=1))
    pooled = math.sqrt(((len(x) - 1) * sx + (len(y) - 1) * sy) / float(len(x) + len(y) - 2))
    if pooled <= 1e-12:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def cliff_delta(x, y):
    """Cliff's delta effect size (nonparametric). Returns delta in [-1,1]."""
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    # O(n^2) is fine for 300 episodes; keep simple and Py3.7-safe.
    gt = 0
    lt = 0
    for xi in x:
        gt += int(np.sum(xi > y))
        lt += int(np.sum(xi < y))
    return float((gt - lt) / float(len(x) * len(y)))

def bootstrap_ci_mean(x, n_boot=5000, ci=0.95, seed=123):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    means = []
    n = len(x)
    for _ in range(int(n_boot)):
        samp = rng.choice(x, size=n, replace=True)
        means.append(float(np.mean(samp)))
    lo = float(np.percentile(means, (1-ci)/2*100))
    hi = float(np.percentile(means, (1+ci)/2*100))
    return lo, hi




def write_effect_size_report(eval_df, out_path, label="all"):
    """
    Compute mean + 95 % bootstrap CI per (metric, mode), and Cliff's delta /
    Cohen's d for innov vs ttc_baseline and innov vs normal.

    `label` is appended to the rows under a 'subset' column so a single
    consolidated CSV can hold both the all-episodes and interaction-only
    statistics for direct comparison.
    """
    metrics = [
        "collision_any",
        "collision_rate",
        "min_ttc",
        "min_ttc_any",
        "ttc_lt_2_ratio",
        "ttc_lt_3_ratio",
        "ttc_any_lt_3_ratio",
        "max_abs_jerk_f",
        "intervention_rate",
        "ttc_valid_ratio",
        "low_interaction_flag",
        "travel_time_s",
        "distance_travelled_m",
        "mean_speed_mps",
        "speed_efficiency_ratio",
        "velocity_tracking_error_mps",
        # Realism diagnostics (only present in PUB run output)
        "ctrl_compute_mean_ms",
        "ctrl_compute_p95_ms",
        "ctrl_compute_max_ms",
    ]

    rows = []

    for metric in metrics:
        if metric not in eval_df.columns:
            continue

        for mode in ["normal","ttc_baseline","innov"]:
            x = eval_df.loc[eval_df["mode"] == mode, metric].dropna().values

            if len(x) == 0:
                continue

            mean = float(np.mean(x))
            lo, hi = bootstrap_ci_mean(x)

            rows.append({
                "subset": label,
                "metric": metric,
                "mode": mode,
                "n": int(len(x)),
                "mean": mean,
                "ci95_lo": lo,
                "ci95_hi": hi
            })

        # Effect sizes
        innov = eval_df.loc[eval_df["mode"] == "innov", metric].dropna().values
        base  = eval_df.loc[eval_df["mode"] == "ttc_baseline", metric].dropna().values
        norm  = eval_df.loc[eval_df["mode"] == "normal", metric].dropna().values

        rows.append({
            "subset": label,
            "metric": metric,
            "mode": "effect",
            "cliff_delta_innov_vs_baseline": cliff_delta(innov, base),
            "cliff_delta_innov_vs_normal": cliff_delta(innov, norm),
            "cohen_d_innov_vs_baseline": cohen_d(innov, base),
            "cohen_d_innov_vs_normal": cohen_d(innov, norm)
        })

    df_out = pd.DataFrame(rows)
    # Append mode if file exists (so all subsets share one CSV).
    if out_path.exists():
        df_out.to_csv(out_path, index=False, mode="a", header=False)
    else:
        df_out.to_csv(out_path, index=False)

    print(f"[Effect Size + CI] Saved to {out_path}")

if __name__ == "__main__":
    main()
