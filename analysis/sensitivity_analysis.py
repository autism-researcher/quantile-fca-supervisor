"""
Sensitivity analysis for the target-violation-rate calibration framework.

This script answers the reviewer question: "Are your conclusions robust to
the specific feature weights you chose?"

Procedure (one-at-a-time / leave-one-feature-out perturbation):
  For each of the 8 risk features:
    1. Perturb its weight by +/-20% (relative)
    2. Renormalize all 8 weights to sum to 1
    3. Recompute R(x) on every Stage A NORMAL tick
    4. Recompute per-episode peak R_max
    5. Recompute the calibrated boundary B at each tau in {0.10, 0.15, 0.20}
    6. Report Delta_B = |B_perturbed - B_original|

Output:
  results_framework/analysis/sensitivity_table.csv
  results_framework/analysis/sensitivity_summary.txt

Usage:
  python sensitivity_analysis.py results_framework
  python sensitivity_analysis.py results_framework --seed 1000

Run AFTER the main simulation completes (do not run while CARLA is active).
"""

from __future__ import print_function
import argparse
import json
import os
import sys
import glob
import numpy as np
import pandas as pd


# Original weights used in the live simulation (must match
# carla_episode_logger_PUB_v3.py exactly).
ORIGINAL_WEIGHTS = {
    'speed':       0.08,
    'accel':       0.10,
    'jerk':        0.10,
    'ttc':         0.25,
    'headway':     0.15,
    'lane_offset': 0.12,
    'steer_var':   0.08,
    'density':     0.06,
}

# Per-feature normalization caps (must match logger).
FEATURE_CAPS = {
    'speed':       28.0,
    'accel':       7.0,
    'jerk':        12.0,
    'ttc_min':     0.5,
    'ttc_max':     8.0,
    'headway_max': 60.0,
    'lane_offset': 1.5,
    'steer_var':   0.35,
    'density':     25.0,
}


def normalize_features(df):
    """Pass-through: per-tick CSVs already contain pre-normalized x_* columns
    written by the live simulation logger. Using them directly guarantees
    that the sensitivity analysis re-weights the EXACT same normalized
    features the controllers saw at runtime (no risk of drift between
    logger and post-hoc re-normalization).

    Expects per-tick CSV columns:
      x_speed, x_accel, x_jerk, x_ttc, x_headway,
      x_lane_offset, x_steer_var, x_density

    Returns a DataFrame with 8 columns named xbar_* (renamed from x_* for
    compatibility with compute_R()).
    """
    rename_map = {
        'x_speed':       'xbar_speed',
        'x_accel':       'xbar_accel',
        'x_jerk':        'xbar_jerk',
        'x_ttc':         'xbar_ttc',
        'x_headway':     'xbar_headway',
        'x_lane_offset': 'xbar_lane_offset',
        'x_steer_var':   'xbar_steer_var',
        'x_density':     'xbar_density',
    }
    missing = [c for c in rename_map if c not in df.columns]
    if missing:
        raise KeyError('Missing pre-normalized columns: {}'.format(missing))
    return df[list(rename_map.keys())].rename(columns=rename_map).clip(0, 1)


def compute_R(normalized_df, weights):
    """Compute composite risk R(x) for every row.

    weights: dict with keys matching ORIGINAL_WEIGHTS.
    Returns: pandas Series of R values in [0, 1].
    """
    feature_to_col = {
        'speed':       'xbar_speed',
        'accel':       'xbar_accel',
        'jerk':        'xbar_jerk',
        'ttc':         'xbar_ttc',
        'headway':     'xbar_headway',
        'lane_offset': 'xbar_lane_offset',
        'steer_var':   'xbar_steer_var',
        'density':     'xbar_density',
    }

    R = pd.Series(0.0, index=normalized_df.index)
    for feature_name, weight in weights.items():
        col = feature_to_col[feature_name]
        R = R + weight * normalized_df[col]

    return R.clip(0, 1)


def perturb_weights(original, feature_to_perturb, factor):
    """Perturb a single feature's weight by `factor` (e.g., 1.2 for +20%).

    Renormalizes all weights to sum to 1 after perturbation.
    """
    perturbed = dict(original)
    perturbed[feature_to_perturb] = original[feature_to_perturb] * factor
    total = sum(perturbed.values())
    return {k: v / total for k, v in perturbed.items()}


def calibrate_boundary(R_max_array, tau):
    """Return the (1 - tau)-quantile boundary B from per-episode R_max values."""
    return float(np.quantile(R_max_array, 1.0 - tau))


def load_stage_a_tick_csvs(seed_dir):
    """Load all Stage A NORMAL per-tick CSVs from a single seed directory.

    Returns a list of DataFrames, one per episode. Each DataFrame must
    contain the raw feature columns expected by normalize_features().
    """
    stage_a_dir = os.path.join(seed_dir, 'stageA_normal')
    if not os.path.isdir(stage_a_dir):
        print('  WARNING: directory not found: ' + stage_a_dir)
        return []

    # Tick CSVs are named ep_NNNNN_normal.csv (the per-episode summary
    # files end in _episode_summary.csv and are excluded by the trailing
    # _normal.csv anchor).
    pattern = os.path.join(stage_a_dir, 'ep_[0-9]*_normal.csv')
    paths = sorted(glob.glob(pattern))

    print('  Loading {} Stage A tick CSVs from {}'.format(len(paths), stage_a_dir))

    episodes = []
    for path in paths:
        try:
            df = pd.read_csv(path)
            episodes.append(df)
        except Exception as e:
            print('    skipped (read error): {} -- {}'.format(path, e))

    return episodes


def compute_R_max_per_episode(episodes, weights):
    """For each episode DataFrame, compute the per-tick R, then the per-episode max.

    Returns a numpy array of length n_episodes containing R_max values.
    """
    R_max_values = []
    for ep in episodes:
        try:
            normalized = normalize_features(ep)
            R = compute_R(normalized, weights)
            R_max_values.append(float(R.max()))
        except KeyError as e:
            print('    skipped episode (missing column): {}'.format(e))
            continue

    return np.array(R_max_values)


def run_sensitivity_analysis(seed_dirs, taus=(0.10, 0.15, 0.20), perturbation=0.20):
    """Run the leave-one-feature-out sensitivity analysis across seeds.

    For each seed, for each feature, for each perturbation direction,
    compute the change in calibrated boundary B at each tau.

    Returns a pandas DataFrame summarizing the sensitivity.
    """
    rows = []

    for seed_dir in seed_dirs:
        seed_label = os.path.basename(seed_dir.rstrip('/').rstrip('\\'))
        print('\n--- Processing {} ---'.format(seed_label))

        episodes = load_stage_a_tick_csvs(seed_dir)
        if len(episodes) == 0:
            print('  No episodes loaded; skipping.')
            continue

        # Compute baseline R_max with original weights
        print('  Computing baseline R_max with original weights...')
        R_max_orig = compute_R_max_per_episode(episodes, ORIGINAL_WEIGHTS)
        print('    n_episodes = {}, R_max range = [{:.4f}, {:.4f}]'.format(
            len(R_max_orig), R_max_orig.min(), R_max_orig.max()))

        # Sanity check: recomputed R per tick should match the 'R' column
        # logged at runtime (within float tolerance), confirming that
        # ORIGINAL_WEIGHTS in this script matches the live simulation.
        try:
            sample = episodes[0]
            if 'R' in sample.columns:
                normalized = normalize_features(sample)
                R_recomp = compute_R(normalized, ORIGINAL_WEIGHTS)
                R_logged = sample['R'].astype(float)
                max_abs_err = float((R_recomp - R_logged).abs().max())
                mean_abs_err = float((R_recomp - R_logged).abs().mean())
                print('    sanity (ep 0): max|R_recomp - R_logged| = {:.6f}, '
                      'mean = {:.6f}'.format(max_abs_err, mean_abs_err))
                if max_abs_err > 1e-3:
                    print('    WARNING: recomputed R differs from logged R '
                          'by > 1e-3. ORIGINAL_WEIGHTS in this script may '
                          'not match the live simulation. Sensitivity '
                          'results will still be valid as a self-consistent '
                          're-weighting study, but baseline B may differ '
                          'slightly from the value reported elsewhere.')
        except Exception as e:
            print('    sanity check skipped: {}'.format(e))

        # Original boundaries at each tau
        B_orig = {}
        for tau in taus:
            B_orig[tau] = calibrate_boundary(R_max_orig, tau)
        print('    B_original: {}'.format(
            {('tau=%.2f' % tau): round(B, 4) for tau, B in B_orig.items()}))

        # For each feature, perturb +/-20% and recompute
        for feature in ORIGINAL_WEIGHTS.keys():
            for direction, factor in [('plus20pct', 1.0 + perturbation),
                                       ('minus20pct', 1.0 - perturbation)]:
                perturbed_weights = perturb_weights(ORIGINAL_WEIGHTS, feature, factor)
                R_max_pert = compute_R_max_per_episode(episodes, perturbed_weights)

                row = {
                    'seed': seed_label,
                    'feature_perturbed': feature,
                    'perturbation_direction': direction,
                    'w_original': ORIGINAL_WEIGHTS[feature],
                    'w_perturbed': perturbed_weights[feature],
                }

                # Compare boundaries at each tau
                for tau in taus:
                    B_pert = calibrate_boundary(R_max_pert, tau)
                    delta_B = B_pert - B_orig[tau]
                    row['B_orig_tau{:.2f}'.format(tau)] = B_orig[tau]
                    row['B_pert_tau{:.2f}'.format(tau)] = B_pert
                    row['delta_B_tau{:.2f}'.format(tau)] = delta_B
                    row['abs_delta_B_tau{:.2f}'.format(tau)] = abs(delta_B)

                rows.append(row)

    if len(rows) == 0:
        print('\nNo data processed. Aborting.')
        return None

    return pd.DataFrame(rows)


def write_summary(df, out_path, taus=(0.10, 0.15, 0.20), threshold=0.02):
    """Write a human-readable summary of the sensitivity analysis."""
    lines = []
    lines.append('=' * 70)
    lines.append('SENSITIVITY ANALYSIS SUMMARY')
    lines.append('=' * 70)
    lines.append('')
    lines.append('Procedure: For each of 8 risk features, perturb its weight by')
    lines.append('+/-20% (relative), renormalize all weights to sum to 1,')
    lines.append('recompute the per-episode peak risk R_max on the Stage A')
    lines.append('NORMAL calibration set, and recompute the calibrated boundary')
    lines.append('B at each target violation rate tau. Report |Delta B|.')
    lines.append('')
    lines.append('Threshold for "robust" claim: |Delta B| < {:.2f}'.format(threshold))
    lines.append('')

    # Per-tau summary
    for tau in taus:
        col = 'abs_delta_B_tau{:.2f}'.format(tau)
        if col not in df.columns:
            continue

        max_delta = df[col].max()
        mean_delta = df[col].mean()

        # Worst feature/direction at this tau
        worst_idx = df[col].idxmax()
        worst_feature = df.loc[worst_idx, 'feature_perturbed']
        worst_dir = df.loc[worst_idx, 'perturbation_direction']
        worst_seed = df.loc[worst_idx, 'seed']

        n_violations = int((df[col] >= threshold).sum())
        n_total = len(df)

        lines.append('--- tau = {:.2f} ---'.format(tau))
        lines.append('  Max |Delta B|:  {:.4f}  (worst: {}/{} on {})'.format(
            max_delta, worst_feature, worst_dir, worst_seed))
        lines.append('  Mean |Delta B|: {:.4f}'.format(mean_delta))
        lines.append('  Perturbations exceeding threshold: {}/{} ({:.1f}%)'.format(
            n_violations, n_total,
            100.0 * n_violations / max(n_total, 1)))
        lines.append('')

    # Overall conclusion
    overall_max = max(df[c].max() for c in df.columns
                      if c.startswith('abs_delta_B_tau'))
    lines.append('--- Overall ---')
    lines.append('  Max |Delta B| across all (feature, direction, tau, seed): {:.4f}'.format(
        overall_max))

    if overall_max < threshold:
        lines.append('  CONCLUSION: Framework is ROBUST to weight specification.')
        lines.append('  A +/-20% perturbation in any single feature weight changes')
        lines.append('  the calibrated boundary B by less than {:.2f} in all cases.'.format(threshold))
    else:
        lines.append('  CONCLUSION: Some weight perturbations exceed the {:.2f} threshold.'.format(threshold))
        lines.append('  See per-feature breakdown in sensitivity_table.csv.')

    lines.append('')
    lines.append('Suggested paper text (Section III-B):')
    lines.append('-' * 70)

    if overall_max < threshold:
        lines.append(
            '"A leave-one-feature-out sensitivity analysis was conducted on')
        lines.append(
            'the Stage A NORMAL calibration episodes. Perturbing each w_i by')
        lines.append(
            '+/-20% with renormalization produced changes in the calibrated')
        lines.append(
            'boundary B below {:.3f} in all cases (max |Delta B| = {:.4f}),'.format(
                threshold, overall_max))
        lines.append(
            'indicating that the framework is robust to weight specification."')
    else:
        # Find max delta for inline reporting
        lines.append(
            '"A leave-one-feature-out sensitivity analysis was conducted on')
        lines.append(
            'the Stage A NORMAL calibration episodes. Perturbing each w_i by')
        lines.append(
            '+/-20% with renormalization produced a maximum change in the')
        lines.append(
            'calibrated boundary B of {:.4f} (most sensitive feature: {}).'.format(
                overall_max,
                df.loc[df[[c for c in df.columns
                          if c.startswith('abs_delta_B_tau')]].max(axis=1).idxmax(),
                       'feature_perturbed']))
        lines.append('See sensitivity_table.csv for full breakdown."')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_dir',
                        help='Path to results_framework directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Run on a single seed only (e.g., 1000). '
                             'If not specified, run on all seed_* subdirs.')
    parser.add_argument('--perturbation', type=float, default=0.20,
                        help='Relative weight perturbation magnitude (default: 0.20 = +/-20%%)')
    parser.add_argument('--threshold', type=float, default=0.02,
                        help='Threshold for "robust" claim on |Delta B| (default: 0.02)')
    args = parser.parse_args()

    # Find seed directories
    if args.seed is not None:
        seed_dirs = [os.path.join(args.results_dir, 'seed_{}'.format(args.seed))]
    else:
        seed_dirs = sorted(glob.glob(os.path.join(args.results_dir, 'seed_*')))

    seed_dirs = [d for d in seed_dirs if os.path.isdir(d)]

    if len(seed_dirs) == 0:
        print('ERROR: no seed directories found under {}'.format(args.results_dir))
        sys.exit(1)

    print('Sensitivity analysis')
    print('=' * 70)
    print('Seed directories: {}'.format(seed_dirs))
    print('Perturbation magnitude: +/-{}%'.format(int(args.perturbation * 100)))
    print('Robustness threshold: |Delta B| < {}'.format(args.threshold))

    # Run analysis
    df = run_sensitivity_analysis(seed_dirs, perturbation=args.perturbation)

    if df is None:
        sys.exit(1)

    # Write outputs
    out_dir = os.path.join(args.results_dir, 'analysis')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    csv_path = os.path.join(out_dir, 'sensitivity_table.csv')
    df.to_csv(csv_path, index=False)
    print('\nWrote: {}'.format(csv_path))

    summary_path = os.path.join(out_dir, 'sensitivity_summary.txt')
    write_summary(df, summary_path, threshold=args.threshold)
    print('\nWrote: {}'.format(summary_path))

    print('\nDone.')


if __name__ == '__main__':
    main()
