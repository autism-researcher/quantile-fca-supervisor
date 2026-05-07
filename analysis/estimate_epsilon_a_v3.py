"""
estimate_epsilon_a_v3.py

Fixed: groups by (seed, mode) at the SEED level (not episode), pools all
boundary-crossing windows across episodes within each seed, and reports
multiple percentiles of the response-coefficient distribution so the user
can pick a defensible point estimate.

Key change vs v2: the per-(seed, mode) groupby now correctly aggregates
hundreds of episodes per cell rather than one episode per cell.
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd


def find_log_files(root, include_stage_a=False):
    pattern = os.path.join(root, "seed_*", "**", "ep_*.csv")
    eps = [p for p in glob.glob(pattern, recursive=True)
           if not p.endswith("_episode_summary.csv")]
    if not include_stage_a:
        eps = [p for p in eps if "stageA_normal" not in p.replace("\\", "/")]
    return sorted(eps)


def load_episodes(paths, max_files=None):
    needed = ["episode_id", "seed", "tick", "R", "u_smooth", "B", "mode"]
    rows = []
    for i, p in enumerate(paths):
        if max_files is not None and i >= max_files:
            break
        try:
            df = pd.read_csv(p, usecols=needed)
        except ValueError:
            df = pd.read_csv(p)
            missing = set(needed) - set(df.columns)
            if missing:
                print(f"  [skip] {os.path.basename(p)} missing {missing}",
                      file=sys.stderr)
                continue
            df = df[needed]
        # Add a synthetic file-level identifier in case episode_id repeats across files
        df["__file_idx"] = i
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_response_coefficients(df, boundary_band=0.02, window_half=5,
                                  du_floor=1e-3):
    """Pool all boundary-crossing windows from df. Returns array of |dR/du|.

    df can be ANY subset (single seed+mode, single episode, etc.) -- we group
    internally by (file_idx, episode_id) to keep windows from different
    episodes from blending across episode boundaries.
    """
    coefs = []
    n_boundary = 0
    n_kept = 0

    grouper_cols = ["__file_idx", "episode_id"] if "__file_idx" in df.columns else ["episode_id"]
    for _, ep in df.groupby(grouper_cols, sort=False):
        ep = ep.sort_values("tick").reset_index(drop=True)
        R = ep["R"].values.astype(float)
        u = ep["u_smooth"].values.astype(float)
        B = ep["B"].values.astype(float)
        n = len(ep)

        idxs = np.where(np.abs(R - B) <= boundary_band)[0]
        idxs = idxs[(idxs >= window_half) & (idxs < n - window_half)]
        n_boundary += len(idxs)

        for t in idxs:
            sl = slice(t - window_half, t + window_half + 1)
            R_w = R[sl]
            u_w = u[sl]
            if u_w.max() - u_w.min() < du_floor:
                continue
            A = np.vstack([u_w, np.ones_like(u_w)]).T
            try:
                slope, _ = np.linalg.lstsq(A, R_w, rcond=None)[0]
            except np.linalg.LinAlgError:
                continue
            c = abs(slope)
            if np.isfinite(c) and c > 0:
                coefs.append(c)
                n_kept += 1
    return np.array(coefs), n_boundary, n_kept


def summarize(coefs):
    if len(coefs) == 0:
        return {p: np.nan for p in ("p1", "p5", "p25", "p50", "p75")}
    return dict(
        p1=float(np.percentile(coefs, 1)),
        p5=float(np.percentile(coefs, 5)),
        p25=float(np.percentile(coefs, 25)),
        p50=float(np.percentile(coefs, 50)),
        p75=float(np.percentile(coefs, 75)),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results_framework")
    parser.add_argument("--boundary_band", type=float, default=0.02)
    parser.add_argument("--window_half", type=int, default=5)
    parser.add_argument("--du_floor", type=float, default=1e-3)
    parser.add_argument("--include_stageA", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    args = parser.parse_args()

    print(f"Root: {os.path.abspath(args.root)}")
    paths = find_log_files(args.root, include_stage_a=args.include_stageA)
    print(f"Found {len(paths)} episode log files.")
    if not paths:
        print("No episode logs found.")
        return

    print("Loading...")
    df = load_episodes(paths, max_files=args.max_files)
    if df.empty:
        print("No usable rows.")
        return

    print(f"Loaded {len(df):,} ticks across {df['__file_idx'].nunique()} files "
          f"and {df['seed'].nunique()} unique seed values.")
    print(f"Modes: {sorted(df['mode'].unique())}")
    print(f"Distinct seeds: {sorted(df['seed'].unique())}")

    # ---- Per-seed-and-mode pooled estimate ----
    print()
    print("=" * 90)
    print("PER-SEED, PER-MODE POOLED RESPONSE-COEFFICIENT DISTRIBUTION")
    print("=" * 90)
    print(f"{'seed':>6} {'mode':>14} {'B med':>7} {'n_files':>8} {'n_bnd':>7} "
          f"{'n_kept':>7} {'p5':>8} {'p25':>8} {'p50':>8} {'p75':>8}")
    print("-" * 90)

    cell_summaries = []
    for (seed, mode), grp in df.groupby(["seed", "mode"], sort=True):
        coefs, n_bnd, n_kept = compute_response_coefficients(
            grp,
            boundary_band=args.boundary_band,
            window_half=args.window_half,
            du_floor=args.du_floor,
        )
        s = summarize(coefs)
        n_files = grp["__file_idx"].nunique()
        B_med = float(grp["B"].median())
        if n_kept > 0:
            print(f"{int(seed):>6} {mode:>14} {B_med:>7.4f} {n_files:>8} "
                  f"{n_bnd:>7} {n_kept:>7} "
                  f"{s['p5']:>8.4f} {s['p25']:>8.4f} {s['p50']:>8.4f} {s['p75']:>8.4f}")
        else:
            print(f"{int(seed):>6} {mode:>14} {B_med:>7.4f} {n_files:>8} "
                  f"{n_bnd:>7} {n_kept:>7} {'n/a':>8} {'n/a':>8} {'n/a':>8} {'n/a':>8}")
        cell_summaries.append((int(seed), mode, B_med, n_kept, s))

    # ---- Mode-pooled (across all seeds) ----
    print()
    print("=" * 90)
    print("MODE-POOLED ACROSS ALL SEEDS (most relevant for Theorem 1)")
    print("=" * 90)
    print(f"{'mode':>14} {'B range':>16} {'n_kept':>8} "
          f"{'p1':>8} {'p5':>8} {'p25':>8} {'p50':>8} {'p75':>8}")
    print("-" * 90)

    mode_results = {}
    for mode, grp in df.groupby("mode", sort=True):
        coefs, n_bnd, n_kept = compute_response_coefficients(
            grp,
            boundary_band=args.boundary_band,
            window_half=args.window_half,
            du_floor=args.du_floor,
        )
        s = summarize(coefs)
        B_min, B_max = float(grp["B"].min()), float(grp["B"].max())
        if n_kept > 0:
            print(f"{mode:>14} [{B_min:.4f},{B_max:.4f}] {n_kept:>8} "
                  f"{s['p1']:>8.4f} {s['p5']:>8.4f} {s['p25']:>8.4f} "
                  f"{s['p50']:>8.4f} {s['p75']:>8.4f}")
        else:
            print(f"{mode:>14} [{B_min:.4f},{B_max:.4f}] {n_kept:>8} (no boundary samples)")
        mode_results[mode] = (s, n_kept)

    # ---- Theorem 1 verdict using innov mode (the calibrated controllers) ----
    print()
    print("=" * 90)
    print("THEOREM 1 NUMERICAL INSTANTIATION")
    print("=" * 90)
    if "innov" in mode_results and mode_results["innov"][1] > 0:
        s_innov = mode_results["innov"][0]
        print(f"  innov-mode boundary-crossing windows: {mode_results['innov'][1]} pooled")
        print(f"  Response coefficient distribution:")
        print(f"    1st  percentile = {s_innov['p1']:.4f}")
        print(f"    5th  percentile = {s_innov['p5']:.4f}  <-- conservative report value")
        print(f"    25th percentile = {s_innov['p25']:.4f}")
        print(f"    50th percentile (median) = {s_innov['p50']:.4f}")
        print(f"    75th percentile = {s_innov['p75']:.4f}")

        mu = 0.35
        s_lower = 0.5
        lam = 0.18

        for label, eps_a in (("p5  (conservative)", s_innov['p5']),
                             ("p25 (typical-low)",  s_innov['p25']),
                             ("p50 (median)",       s_innov['p50'])):
            lhs = mu * s_lower * eps_a ** 2
            verdict = "YES (Thm 1 holds)" if lhs > lam else "NO  (Thm 1 fails)"
            print(f"  Using eps_a = {eps_a:.4f} ({label}):  "
                  f"LHS = {lhs:.4g}  vs  lambda = {lam:.2f}  -->  {verdict}")
    else:
        print("  No innov-mode windows recovered. Cannot instantiate Theorem 1.")


if __name__ == "__main__":
    main()
