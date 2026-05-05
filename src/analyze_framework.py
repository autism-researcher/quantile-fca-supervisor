#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analysis script for the calibration-framework study (pre-registered).

Reads:
    <out_dir>/seed_1000/eval_master.csv
    <out_dir>/seed_2000/eval_master.csv
    <out_dir>/seed_3000/eval_master.csv

Produces (in <out_dir>/analysis/):
    - combined_master.csv               raw combined data, all seeds
    - per_controller_summary.csv        mean / SD / count per controller, per metric
    - per_controller_bootstrap_ci.csv   bootstrap 95% CIs (5000 resamples) per metric
    - h1_calibration_convergence.csv    realized vs target violation rate
    - h2_safety_noninferiority.csv      innov-tau vs best-performing TTC (collision rate)
    - h3_pareto_tradeoff.csv            intervention_rate vs ttc_lt_3_ratio (per seed/tau)
    - h3_spearman_per_seed.csv          monotonicity test
    - h3b_pareto_dominance.csv          bootstrap-based dominance test vs TTC frontier
    - h4_reproducibility.csv            per-seed effect sizes, CV across seeds
    - h5_vs_normal_mcnemar.csv          paired McNemar tests, every controller vs Normal
    - h6_comfort_noninferiority.csv     max_abs_jerk / max_abs_accel vs ttc_25 (Bonferroni)
    - figure_pareto.png                 trade-off curve plot with bootstrap error bars
    - figure_collision_bars.png         collision rate by controller with 95% CIs
    - figure_per_seed_consistency.png   effect sizes by seed

This script is designed to be run AFTER all three seeds finish.
It will run on whatever subset of seeds is available, but H4
(reproducibility) needs at least 2 seeds to be meaningful.

Usage:
    python analyze_framework.py results_framework
    python analyze_framework.py results_framework --bootstrap_n 5000 --rng_seed 12345
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_combined(out_dir: Path):
    rows = []
    for seed_dir in sorted(out_dir.glob("seed_*")):
        em = seed_dir / "eval_master.csv"
        if not em.exists():
            print("[SKIP] {} (no eval_master.csv yet)".format(seed_dir.name))
            continue
        df = pd.read_csv(em)
        if "seed0" not in df.columns:
            df["seed0"] = int(seed_dir.name.split("_")[1])
        rows.append(df)
        print("[OK] loaded {}: {} rows, controllers={}".format(
            seed_dir.name, len(df), sorted(df['controller_label'].unique())))
    if not rows:
        raise RuntimeError("No seed data found.")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Statistical primitives
# ---------------------------------------------------------------------------

def wilson_ci(k, n, alpha=0.05):
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def bootstrap_ci(values, n_resamples=5000, alpha=0.05, rng=None):
    """Percentile bootstrap 95% CI of the mean of a 1-D array."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng()
    n = len(arr)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1 - alpha / 2.0))
    return (lo, hi)


def cohens_d(x, y):
    """Cohen's d (pooled SD)."""
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    y = np.asarray(y, dtype=float); y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    sp = np.sqrt(((nx - 1) * sx * sx + (ny - 1) * sy * sy) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / sp)


def mcnemar_paired(a, b):
    """
    McNemar's test on a paired binary outcome.
    a, b : 1-D arrays of 0/1 of equal length, episode-aligned.
    Returns (b01, b10, statistic, p-value).
    b01 = "a=0 and b=1" (controller A safer, controller B crashed)
    b10 = "a=1 and b=0" (controller A crashed, controller B safer)
    Uses an exact binomial test (no continuity correction issues).
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    b01 = int(((a == 0) & (b == 1)).sum())
    b10 = int(((a == 1) & (b == 0)).sum())
    n = b01 + b10
    if n == 0:
        return b01, b10, 0.0, 1.0
    # Two-sided exact binomial test on min(b01, b10) vs n at p=0.5
    k = min(b01, b10)
    pval = float(stats.binomtest(k, n, p=0.5, alternative="two-sided").pvalue)
    stat = float((b01 - b10) ** 2 / (b01 + b10))  # asymptotic chi-square statistic
    return b01, b10, stat, pval


# ---------------------------------------------------------------------------
# Per-controller summary
# ---------------------------------------------------------------------------

DEFAULT_METRICS = [
    "collision_any", "min_ttc",
    "ttc_lt_2_ratio", "ttc_lt_3_ratio",
    "max_abs_accel", "max_abs_jerk",
    "max_abs_accel_f", "max_abs_jerk_f",
    "mean_speed_mps", "speed_efficiency_ratio",
    "velocity_tracking_error_mps",
    "intervention_rate", "mean_pct_speed_diff",
    "throughput_mps",
]


def per_controller_summary(df: pd.DataFrame):
    metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    g = df.groupby("controller_label")[metrics].agg(["mean", "std", "count"])
    return g


def per_controller_bootstrap_ci(df: pd.DataFrame, n_resamples=5000, rng=None):
    """5000-resample bootstrap 95% CIs on per-controller means."""
    if rng is None:
        rng = np.random.default_rng(12345)
    metrics = [m for m in DEFAULT_METRICS if m in df.columns]
    rows = []
    for ctrl, grp in df.groupby("controller_label"):
        for m in metrics:
            vals = grp[m].dropna().values
            if len(vals) == 0:
                lo, hi, mu = float("nan"), float("nan"), float("nan")
            else:
                lo, hi = bootstrap_ci(vals, n_resamples=n_resamples, rng=rng)
                mu = float(np.mean(vals))
            rows.append(dict(controller=ctrl, metric=m,
                             n=len(vals), mean=mu,
                             ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# H1 — Calibration convergence
# ---------------------------------------------------------------------------

def h1_calibration_convergence(df: pd.DataFrame, out_dir: Path):
    """
    For each (seed, tau) calibration, did the realized violation rate
    on the eval set fall within tau +/- 0.03?

    A 'violation' here means: the per-episode max_R exceeded the
    calibrated B for that controller.
    """
    rows = []
    innov_rows = df[df["controller_label"].str.startswith("innov_")].copy()
    if innov_rows.empty:
        print("[H1] no innov rows found; skipping")
        return pd.DataFrame()

    has_maxR = "max_R" in innov_rows.columns
    if not has_maxR:
        print("[H1] WARNING: 'max_R' not in summary CSV; H1 realized rate will be NaN. "
              "Patch carla_episode_logger to emit max_R per episode.")

    for (seed, lbl), grp in innov_rows.groupby(["seed0", "controller_label"]):
        tau = float(grp["target_tau"].iloc[0])
        B = float(grp["B_used"].iloc[0])
        if has_maxR:
            n_violations = int((grp["max_R"] > B).sum())
        else:
            n_violations = float("nan")
        n = int(len(grp))
        realized = (n_violations / n) if (has_maxR and n > 0) else float("nan")
        delta = abs(realized - tau) if not np.isnan(realized) else float("nan")
        rows.append(dict(
            seed=seed, controller=lbl, target_tau=tau, B=B,
            n_eval=n, n_violations=n_violations,
            realized_rate=realized, delta_from_target=delta,
            within_tolerance=(delta <= 0.03) if not np.isnan(delta) else None,
        ))
    out = pd.DataFrame(rows)

    # Pre-reg passes: at least 2 of 3 seeds within tolerance for each tau
    if not out.empty and out["within_tolerance"].notna().any():
        per_tau = out.groupby("controller").agg(
            n_seeds=("seed", "count"),
            n_within=("within_tolerance", "sum"),
        ).reset_index()
        per_tau["h1_pass"] = per_tau["n_within"] >= 2
        per_tau.to_csv(out_dir / "h1_pass_per_tau.csv", index=False)

    out.to_csv(out_dir / "h1_calibration_convergence.csv", index=False)
    print("[H1] wrote h1_calibration_convergence.csv")
    return out


# ---------------------------------------------------------------------------
# H2 — Safety non-inferiority vs the best-performing TTC threshold
# ---------------------------------------------------------------------------

def h2_safety_noninferiority(df: pd.DataFrame, out_dir: Path,
                             reference_label=None, margin=0.01):
    """
    Per pre-reg §3 / §8.1: H2 reference is the BEST-performing TTC threshold
    in the tuning grid (lowest collision rate). If reference_label is None,
    the best TTC is selected dynamically here. Combined across all seeds.
    """
    rows = []
    pivot = df.pivot_table(
        index=["seed0", "episode_id"], columns="controller_label",
        values="collision_any", aggfunc="first")
    ttc_cols = [c for c in pivot.columns if c.startswith("ttc_")]
    if not ttc_cols:
        print("[H2] no TTC controllers found; skipping")
        return pd.DataFrame()

    if reference_label is None:
        ttc_means = pivot[ttc_cols].mean()
        reference_label = str(ttc_means.idxmin())
        print("[H2] dynamic reference TTC = {} (collision rate {:.4f})".format(
            reference_label, float(ttc_means.min())))

    if reference_label not in pivot.columns:
        print("[H2] reference label {} not found".format(reference_label))
        return pd.DataFrame()

    ttc_rate = float(pivot[reference_label].mean())
    ttc_k = int(pivot[reference_label].sum())
    ttc_n = int(pivot[reference_label].notna().sum())

    for col in pivot.columns:
        if not col.startswith("innov_"):
            continue
        sub = pivot[[col, reference_label]].dropna()
        n = int(len(sub))
        if n == 0:
            continue
        innov_rate = float(sub[col].mean())
        innov_k = int(sub[col].sum())
        diff = innov_rate - ttc_rate
        innov_ci = wilson_ci(innov_k, n)
        # one-sided non-inferiority via two-proportion z (pooled variance)
        if (innov_k + ttc_k) > 0 and (n + ttc_n) > 0:
            p_pool = (innov_k + ttc_k) / (n + ttc_n)
            if 0 < p_pool < 1:
                se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / n + 1.0 / ttc_n))
                z = (diff - margin) / se if se > 0 else float("nan")
                pval = float(stats.norm.cdf(z)) if not np.isnan(z) else float("nan")
            else:
                pval = float("nan")
        else:
            pval = float("nan")

        non_inferior = (not np.isnan(pval)) and (pval < 0.05)
        rows.append(dict(
            controller=col, reference=reference_label,
            n=n,
            innov_collision_rate=innov_rate,
            ttc_collision_rate=ttc_rate,
            diff=diff, margin=margin,
            innov_ci_lo=innov_ci[0], innov_ci_hi=innov_ci[1],
            p_noninferior=pval,
            non_inferior_at_alpha_05=non_inferior,
        ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["h2_pass_any"] = bool(out["non_inferior_at_alpha_05"].any())
    out.to_csv(out_dir / "h2_safety_noninferiority.csv", index=False)
    print("[H2] wrote h2_safety_noninferiority.csv (ref = {})".format(reference_label))
    return out


# ---------------------------------------------------------------------------
# H3 — Pareto trade-off
# ---------------------------------------------------------------------------

def _pareto_dominates_or_lies_below(point, frontier_points):
    """
    Return True if `point` lies on or below the empirical Pareto frontier
    defined by `frontier_points` in a 2-D space where smaller-is-better
    on both axes (intervention_rate, ttc_lt_3_ratio).

    A point P=(x,y) is dominated by a frontier point F=(fx,fy) if
    fx <= x and fy <= y (with at least one strict).  P is "on or below
    the frontier" if no frontier point strictly dominates it -- i.e.,
    P itself is non-dominated relative to the frontier.
    """
    px, py = point
    for fx, fy in frontier_points:
        if (fx <= px) and (fy <= py) and ((fx < px) or (fy < py)):
            return False
    return True


def h3_pareto(df: pd.DataFrame, out_dir: Path):
    """
    For each seed, compute mean intervention_rate and mean ttc_lt_3_ratio
    for each innov_<tau>. Spearman rho between tau and each metric per seed,
    Fisher-z combined.
    """
    innov = df[df["controller_label"].str.startswith("innov_")].copy()
    if innov.empty:
        print("[H3] no innov data")
        return pd.DataFrame()
    g = innov.groupby(["seed0", "target_tau"]).agg(
        intervention_rate=("intervention_rate", "mean"),
        ttc_lt_3_ratio=("ttc_lt_3_ratio", "mean"),
        collision_rate=("collision_any", "mean"),
        max_abs_jerk=("max_abs_jerk", "mean"),
        n=("episode_id", "count"),
    ).reset_index()
    g.to_csv(out_dir / "h3_pareto_tradeoff.csv", index=False)
    print("[H3] wrote h3_pareto_tradeoff.csv")

    rho_rows = []
    fisher_z_iv = []; fisher_z_ttc = []
    for seed, grp in g.groupby("seed0"):
        if len(grp) < 3:
            continue
        rho_iv,  p_iv  = stats.spearmanr(grp["target_tau"], grp["intervention_rate"])
        rho_ttc, p_ttc = stats.spearmanr(grp["target_tau"], grp["ttc_lt_3_ratio"])
        rho_rows.append(dict(seed=seed, n_taus=len(grp),
                             rho_tau_vs_intervention=rho_iv, p_tau_vs_intervention=p_iv,
                             rho_tau_vs_ttc_lt_3=rho_ttc, p_tau_vs_ttc_lt_3=p_ttc))
        if abs(rho_iv) < 0.999:
            fisher_z_iv.append(np.arctanh(rho_iv))
        if abs(rho_ttc) < 0.999:
            fisher_z_ttc.append(np.arctanh(rho_ttc))

    rho_df = pd.DataFrame(rho_rows)
    rho_df.to_csv(out_dir / "h3_spearman_per_seed.csv", index=False)

    # Fisher-z combination across seeds (each seed has 3 tau points => SE=1/sqrt(0))
    # Use simple unweighted mean of z-transformed rhos.
    if fisher_z_iv:
        rho_iv_combined = float(np.tanh(np.mean(fisher_z_iv)))
    else:
        rho_iv_combined = float("nan")
    if fisher_z_ttc:
        rho_ttc_combined = float(np.tanh(np.mean(fisher_z_ttc)))
    else:
        rho_ttc_combined = float("nan")
    pd.DataFrame([dict(rho_tau_vs_intervention_combined=rho_iv_combined,
                       rho_tau_vs_ttc_lt_3_combined=rho_ttc_combined)]) \
      .to_csv(out_dir / "h3_spearman_combined.csv", index=False)
    return g


def h3b_pareto_dominance(df: pd.DataFrame, out_dir: Path,
                         n_resamples=1000, dominance_threshold=0.5,
                         rng=None):
    """
    Pre-reg §8.1 H3b: bootstrap (n=1000) the (intervention_rate, ttc_lt_3_ratio)
    coordinates of each Innov-tau and ask whether the bootstrap distribution
    lies on/below the TTC-grid Pareto frontier in at least 50% of resamples.

    The frontier is the *non-dominated subset* of the four TTC controllers
    (where smaller is better on both axes).
    """
    if rng is None:
        rng = np.random.default_rng(67890)

    eligible = df[df["controller_label"].str.startswith(("ttc_", "innov_"))].copy()
    if eligible.empty:
        print("[H3b] no innov/ttc data")
        return pd.DataFrame()

    # Per-controller paired (intervention_rate, ttc_lt_3_ratio) at episode level.
    by_ctrl = {ctrl: g for ctrl, g in eligible.groupby("controller_label")}

    def _mean_resample(arr_iv, arr_tt, n):
        idx = rng.integers(0, n, size=n)
        return float(arr_iv[idx].mean()), float(arr_tt[idx].mean())

    # Build TTC frontier from observed means (point estimates).
    ttc_pts = []
    for ctrl in [c for c in by_ctrl if c.startswith("ttc_")]:
        g = by_ctrl[ctrl].dropna(subset=["intervention_rate", "ttc_lt_3_ratio"])
        ttc_pts.append((ctrl,
                        float(g["intervention_rate"].mean()),
                        float(g["ttc_lt_3_ratio"].mean())))

    # Non-dominated TTC subset (smaller-is-better on both axes).
    frontier = []
    for name_i, xi, yi in ttc_pts:
        dominated = False
        for name_j, xj, yj in ttc_pts:
            if name_j == name_i:
                continue
            if (xj <= xi) and (yj <= yi) and ((xj < xi) or (yj < yi)):
                dominated = True
                break
        if not dominated:
            frontier.append((xi, yi))

    rows = []
    for ctrl in [c for c in by_ctrl if c.startswith("innov_")]:
        g = by_ctrl[ctrl].dropna(subset=["intervention_rate", "ttc_lt_3_ratio"])
        n = int(len(g))
        if n == 0:
            continue
        arr_iv = g["intervention_rate"].values.astype(float)
        arr_tt = g["ttc_lt_3_ratio"].values.astype(float)
        on_or_below = 0
        boot_iv = np.empty(n_resamples, dtype=float)
        boot_tt = np.empty(n_resamples, dtype=float)
        for k in range(n_resamples):
            mu_iv, mu_tt = _mean_resample(arr_iv, arr_tt, n)
            boot_iv[k] = mu_iv
            boot_tt[k] = mu_tt
            if _pareto_dominates_or_lies_below((mu_iv, mu_tt), frontier):
                on_or_below += 1
        frac = on_or_below / float(n_resamples) if n_resamples > 0 else float("nan")
        rows.append(dict(
            controller=ctrl, n_paired=n,
            mean_intervention_rate=float(arr_iv.mean()),
            mean_ttc_lt_3_ratio=float(arr_tt.mean()),
            boot_iv_ci_lo=float(np.quantile(boot_iv, 0.025)),
            boot_iv_ci_hi=float(np.quantile(boot_iv, 0.975)),
            boot_tt_ci_lo=float(np.quantile(boot_tt, 0.025)),
            boot_tt_ci_hi=float(np.quantile(boot_tt, 0.975)),
            n_resamples=n_resamples,
            frac_on_or_below_frontier=frac,
            h3b_pass=(frac >= dominance_threshold),
        ))
    out = pd.DataFrame(rows)
    if not out.empty:
        out["h3b_pass_any"] = bool(out["h3b_pass"].any())
    out.to_csv(out_dir / "h3b_pareto_dominance.csv", index=False)
    print("[H3b] wrote h3b_pareto_dominance.csv (frontier size = {})".format(len(frontier)))
    return out


# ---------------------------------------------------------------------------
# H4 — Reproducibility across seeds
# ---------------------------------------------------------------------------

def h4_reproducibility(df: pd.DataFrame, out_dir: Path):
    metrics = ["collision_any", "ttc_lt_3_ratio", "intervention_rate",
               "max_abs_jerk", "mean_pct_speed_diff", "min_ttc"]
    metrics = [m for m in metrics if m in df.columns]
    rows = []
    for ctrl, grp in df.groupby("controller_label"):
        for m in metrics:
            per_seed = grp.groupby("seed0")[m].mean()
            if len(per_seed) < 2:
                cv = float("nan")
                sd = float("nan")
            else:
                mu = per_seed.mean()
                sd = per_seed.std(ddof=1)
                cv = sd / abs(mu) if mu != 0 else float("nan")
            rows.append(dict(
                controller=ctrl, metric=m,
                n_seeds=len(per_seed),
                mean_across_seeds=per_seed.mean(),
                sd_across_seeds=sd,
                cv=cv,
                stable=(not np.isnan(cv)) and (cv < 0.20),
            ))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "h4_reproducibility.csv", index=False)
    print("[H4] wrote h4_reproducibility.csv")
    return out


# ---------------------------------------------------------------------------
# H5 — Every controller vs Normal (paired McNemar, p<0.001)
# ---------------------------------------------------------------------------

def h5_vs_normal(df: pd.DataFrame, out_dir: Path):
    pivot = df.pivot_table(
        index=["seed0", "episode_id"], columns="controller_label",
        values="collision_any", aggfunc="first")
    if "normal" not in pivot.columns:
        print("[H5] no 'normal' controller in data; skipping")
        return pd.DataFrame()

    rows = []
    for col in pivot.columns:
        if col == "normal":
            continue
        sub = pivot[["normal", col]].dropna()
        n = int(len(sub))
        if n == 0:
            continue
        a = sub["normal"].astype(int).values
        b = sub[col].astype(int).values
        b01, b10, stat, pval = mcnemar_paired(a, b)
        normal_rate = float(a.mean())
        ctrl_rate = float(b.mean())
        rows.append(dict(
            controller=col, n_paired=n,
            normal_collision_rate=normal_rate,
            controller_collision_rate=ctrl_rate,
            diff=ctrl_rate - normal_rate,
            b_normal_safe_ctrl_crash=b01,
            b_normal_crash_ctrl_safe=b10,
            mcnemar_chi2=stat,
            p_mcnemar=pval,
            h5_pass=(pval < 0.001 and ctrl_rate < normal_rate),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "h5_vs_normal_mcnemar.csv", index=False)
    print("[H5] wrote h5_vs_normal_mcnemar.csv")
    return out


# ---------------------------------------------------------------------------
# H6 — Comfort non-inferiority vs ttc_25 (Bonferroni for 2 tests)
# ---------------------------------------------------------------------------

def h6_comfort_noninferiority(df: pd.DataFrame, out_dir: Path,
                              reference_label="ttc_25",
                              metrics=("max_abs_jerk", "max_abs_accel"),
                              margin_sd_units=0.2,
                              n_resamples=5000, rng=None):
    """
    Pre-reg §3 H6: calibrated controllers do not produce worse comfort than
    the median TTC grid point (ttc_25). Margin Δ = 0.2σ of TTC distribution.
    Bonferroni-adjusted alpha = 0.05 / 2 tests = 0.025 per test.

    Test logic: for each Innov-tau and each comfort metric, compute the
    paired mean difference (innov - ttc_25) per episode. Bootstrap a 95% CI
    (Bonferroni: 97.5% CI) of the mean difference, and declare non-inferior
    if the upper CI < margin (where margin is +0.2 * SD of ttc_25).
    """
    if rng is None:
        rng = np.random.default_rng(54321)
    rows = []
    available = [m for m in metrics if m in df.columns]
    if not available:
        print("[H6] none of the requested comfort metrics are in the data")
        return pd.DataFrame()

    alpha_bonf = 0.05 / max(1, len(available))  # = 0.025 if both metrics
    ci_alpha = alpha_bonf  # one Bonferroni-adjusted CI per metric

    for metric in available:
        pivot = df.pivot_table(
            index=["seed0", "episode_id"], columns="controller_label",
            values=metric, aggfunc="first")
        if reference_label not in pivot.columns:
            print("[H6] reference {} not found for metric {}".format(reference_label, metric))
            continue
        ref_vals = pivot[reference_label].dropna().values.astype(float)
        if len(ref_vals) == 0:
            continue
        ref_sd = float(np.std(ref_vals, ddof=1))
        margin = margin_sd_units * ref_sd
        for col in pivot.columns:
            if not col.startswith("innov_"):
                continue
            sub = pivot[[col, reference_label]].dropna()
            n = int(len(sub))
            if n == 0:
                continue
            diffs = (sub[col] - sub[reference_label]).values.astype(float)
            mean_diff = float(diffs.mean())
            # Bonferroni-adjusted bootstrap CI (1 - alpha_bonf) on the mean diff
            idx = rng.integers(0, n, size=(n_resamples, n))
            boot_means = diffs[idx].mean(axis=1)
            ci_lo = float(np.quantile(boot_means, ci_alpha / 2.0))
            ci_hi = float(np.quantile(boot_means, 1 - ci_alpha / 2.0))
            non_inferior = ci_hi < margin
            rows.append(dict(
                controller=col, reference=reference_label, metric=metric,
                n_paired=n,
                mean_diff=mean_diff, ttc_sd=ref_sd, margin=margin,
                bonf_ci_lo=ci_lo, bonf_ci_hi=ci_hi,
                bonferroni_alpha=alpha_bonf,
                non_inferior=non_inferior,
            ))
    out = pd.DataFrame(rows)
    if not out.empty:
        # H6 passes overall if BOTH metrics are non-inferior for at least one
        # innov controller.
        per_ctrl = out.groupby("controller")["non_inferior"].all()
        out["h6_pass_for_this_controller"] = out["controller"].map(per_ctrl)
        out["h6_pass_any"] = bool(per_ctrl.any())
    out.to_csv(out_dir / "h6_comfort_noninferiority.csv", index=False)
    print("[H6] wrote h6_comfort_noninferiority.csv (Bonferroni alpha = {:.4f})".format(alpha_bonf))
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(df: pd.DataFrame, out_dir: Path,
                 ci_table: pd.DataFrame = None,
                 n_resamples=5000, rng=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[FIG] matplotlib not installed; skipping figures")
        return

    if rng is None:
        rng = np.random.default_rng(11111)

    # --- Fig 1: Pareto trade-off with bootstrap error bars ------------------
    summary = df.groupby("controller_label").agg(
        intervention_rate=("intervention_rate", "mean"),
        ttc_lt_3_ratio=("ttc_lt_3_ratio", "mean"),
        collision_rate=("collision_any", "mean"),
    ).reset_index()
    summary["family"] = summary["controller_label"].apply(
        lambda l: "innov" if l.startswith("innov_") else
                  ("ttc" if l.startswith("ttc_") else "normal"))

    # Bootstrap CIs on the two Pareto axes per controller.
    iv_ci, tt_ci = {}, {}
    for ctrl, grp in df.groupby("controller_label"):
        iv_vals = grp["intervention_rate"].dropna().values.astype(float)
        tt_vals = grp["ttc_lt_3_ratio"].dropna().values.astype(float)
        iv_ci[ctrl] = bootstrap_ci(iv_vals, n_resamples=n_resamples, rng=rng) \
                      if len(iv_vals) else (np.nan, np.nan)
        tt_ci[ctrl] = bootstrap_ci(tt_vals, n_resamples=n_resamples, rng=rng) \
                      if len(tt_vals) else (np.nan, np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    style = {"innov":  ("o", "tab:blue",   "Innov (calibrated)"),
             "ttc":    ("s", "tab:orange", "TTC tuning grid"),
             "normal": ("x", "tab:gray",   "Normal (no FCA)")}
    for fam in ["innov", "ttc", "normal"]:
        s = summary[summary["family"] == fam].sort_values("intervention_rate")
        if len(s) == 0:
            continue
        marker, color, label = style[fam]
        x = s["intervention_rate"].values
        y = s["ttc_lt_3_ratio"].values
        xerr = np.array([[x[i] - iv_ci[c][0] for i, c in enumerate(s["controller_label"])],
                         [iv_ci[c][1] - x[i] for i, c in enumerate(s["controller_label"])]])
        yerr = np.array([[y[i] - tt_ci[c][0] for i, c in enumerate(s["controller_label"])],
                         [tt_ci[c][1] - y[i] for i, c in enumerate(s["controller_label"])]])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                    marker=marker, color=color, label=label,
                    linestyle="-" if fam in ("innov", "ttc") else "",
                    markersize=10, capsize=3, alpha=0.85)
        for i, r in enumerate(s.itertuples(index=False)):
            ax.annotate(r.controller_label.replace("innov_", "τ=0.").replace("ttc_", "TTC="),
                        (x[i], y[i]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Mean intervention rate")
    ax.set_ylabel("Mean TTC<3s exposure ratio")
    ax.set_title("Pareto trade-off: safety vs intervention\n"
                 "(Innov framework vs TTC tuning grid, error bars: 95% bootstrap CI)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure_pareto.png", dpi=150)
    plt.close(fig)
    print("[FIG] wrote figure_pareto.png")

    # --- Fig 2: Collision-rate bar chart with Wilson 95% CIs ---------------
    fig, ax = plt.subplots(figsize=(8, 4))
    summary_sorted = summary.sort_values("collision_rate")
    colors = ["tab:gray" if f == "normal" else
              "tab:orange" if f == "ttc" else "tab:blue"
              for f in summary_sorted["family"]]
    # Wilson CIs for each controller
    cis = []
    for ctrl in summary_sorted["controller_label"]:
        g = df[df["controller_label"] == ctrl]
        k = int(g["collision_any"].sum())
        n = int(g["collision_any"].notna().sum())
        lo, hi = wilson_ci(k, n)
        cis.append((lo, hi))
    yerr_lo = [r * 100 - lo * 100 for r, (lo, _) in zip(summary_sorted["collision_rate"], cis)]
    yerr_hi = [hi * 100 - r * 100 for r, (_, hi) in zip(summary_sorted["collision_rate"], cis)]
    ax.bar(summary_sorted["controller_label"],
           summary_sorted["collision_rate"] * 100,
           color=colors,
           yerr=[yerr_lo, yerr_hi], capsize=3,
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Collision rate (%)")
    ax.set_title("Collision rate by controller (combined across seeds, 95% Wilson CI)")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "figure_collision_bars.png", dpi=150)
    plt.close(fig)
    print("[FIG] wrote figure_collision_bars.png")

    # --- Fig 3: Per-seed reproducibility -----------------------------------
    fig, ax = plt.subplots(figsize=(9, 4))
    per_seed = df.groupby(["seed0", "controller_label"])["collision_any"].mean().reset_index()
    seeds = sorted(per_seed["seed0"].unique())
    ctrls = sorted(per_seed["controller_label"].unique())
    width = 0.8 / max(1, len(seeds))
    x_base = np.arange(len(ctrls))
    for i, s in enumerate(seeds):
        sub = per_seed[per_seed["seed0"] == s].set_index("controller_label").reindex(ctrls)
        ax.bar(x_base + i * width, sub["collision_any"] * 100,
               width=width, label="seed {}".format(int(s)))
    ax.set_xticks(x_base + width * (len(seeds) - 1) / 2)
    ax.set_xticklabels(ctrls, rotation=45, ha="right")
    ax.set_ylabel("Collision rate (%)")
    ax.set_title("Per-seed collision rate (H4 reproducibility)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "figure_per_seed_consistency.png", dpi=150)
    plt.close(fig)
    print("[FIG] wrote figure_per_seed_consistency.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", help="Top-level output dir (with seed_* subdirs)")
    ap.add_argument("--bootstrap_n", type=int, default=5000,
                    help="Bootstrap resamples for CIs (default 5000)")
    ap.add_argument("--pareto_bootstrap_n", type=int, default=1000,
                    help="Bootstrap resamples for H3b Pareto dominance (pre-reg: 1000)")
    ap.add_argument("--rng_seed", type=int, default=12345,
                    help="Seed for the bootstrap RNG (reproducibility)")
    ap.add_argument("--h2_reference", default="best",
                    help="'best' (pre-reg, dynamic) or a specific TTC label like 'ttc_25'")
    ap.add_argument("--h2_margin", type=float, default=0.01,
                    help="Non-inferiority margin (collision-rate diff) for H2")
    ap.add_argument("--h6_reference", default="ttc_25",
                    help="Reference TTC for H6 comfort (pre-reg: ttc_25)")
    ap.add_argument("--h6_margin_sd", type=float, default=0.2,
                    help="H6 margin in SD units of the reference TTC distribution")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        raise SystemExit("out_dir {} does not exist".format(out_dir))

    analysis_dir = out_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.rng_seed))

    # Load.
    df = load_combined(out_dir)
    df.to_csv(analysis_dir / "combined_master.csv", index=False)
    print("[OK] combined_master.csv: n_rows={}, seeds={}".format(
        len(df), sorted(df['seed0'].unique())))

    # Per-controller summary + bootstrap CIs.
    summary = per_controller_summary(df)
    summary.to_csv(analysis_dir / "per_controller_summary.csv")
    print("[OK] per_controller_summary.csv")

    ci_table = per_controller_bootstrap_ci(df, n_resamples=args.bootstrap_n, rng=rng)
    ci_table.to_csv(analysis_dir / "per_controller_bootstrap_ci.csv", index=False)
    print("[OK] per_controller_bootstrap_ci.csv ({} resamples)".format(args.bootstrap_n))

    # Hypotheses.
    h1_calibration_convergence(df, analysis_dir)

    h2_ref = None if str(args.h2_reference).lower() == "best" else str(args.h2_reference)
    h2_safety_noninferiority(df, analysis_dir,
                             reference_label=h2_ref, margin=float(args.h2_margin))

    h3_pareto(df, analysis_dir)
    h3b_pareto_dominance(df, analysis_dir,
                         n_resamples=args.pareto_bootstrap_n,
                         rng=rng)

    h4_reproducibility(df, analysis_dir)
    h5_vs_normal(df, analysis_dir)
    h6_comfort_noninferiority(df, analysis_dir,
                              reference_label=args.h6_reference,
                              margin_sd_units=float(args.h6_margin_sd),
                              n_resamples=args.bootstrap_n,
                              rng=rng)

    make_figures(df, analysis_dir, ci_table=ci_table,
                 n_resamples=args.bootstrap_n, rng=rng)

    # Provenance: dump the actual analysis settings used.
    provenance = dict(
        bootstrap_n=int(args.bootstrap_n),
        pareto_bootstrap_n=int(args.pareto_bootstrap_n),
        rng_seed=int(args.rng_seed),
        h2_reference=("best (dynamic)" if h2_ref is None else h2_ref),
        h2_margin=float(args.h2_margin),
        h6_reference=str(args.h6_reference),
        h6_margin_sd_units=float(args.h6_margin_sd),
        n_episodes=int(len(df)),
        seeds=[int(s) for s in sorted(df['seed0'].unique())],
        controllers=sorted(df['controller_label'].unique().tolist()),
    )
    (analysis_dir / "analysis_provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print("[OK] analysis_provenance.json")

    print("\n[DONE] all outputs in {}".format(analysis_dir))


if __name__ == "__main__":
    main()
