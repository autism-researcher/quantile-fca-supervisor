"""
regenerate_fig_h2_forest.py  (v2)

Fixes the v92 typesetting glitches in the in-panel labels:
  - "Δ=1\,pp"            ->  "Δ = 1 pp"   (thin space rendered properly)
  - "∗\,formal pass..."  ->  "∗ formal pass..."

Otherwise identical to the previous version.

Usage:
    pip install pandas numpy scipy matplotlib
    python regenerate_fig_h2_forest.py eval_master_combined_with_maxR.csv
"""

import sys
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl


DELTA = 1.0     # pp; pre-registered non-inferiority margin
ALPHA = 0.05

# Unicode thin space (U+2009).  Matplotlib renders this directly,
# whereas the LaTeX macro "\," is only honoured under usetex=True.
THIN = "\u2009"


def load_paired(path):
    df = pd.read_csv(path)
    needed = {"seed0", "episode_id", "controller_label", "collision_any"}
    missing = needed - set(df.columns)
    if missing:
        sys.exit(f"Missing columns: {missing}")

    wide = df.pivot_table(
        index=["seed0", "episode_id"],
        columns="controller_label",
        values="collision_any",
        aggfunc="first",
    ).reset_index()

    expected = {"normal", "ttc_20", "ttc_25", "ttc_30", "ttc_35",
                "innov_010", "innov_015", "innov_020"}
    miss_ctrl = expected - set(wide.columns)
    if miss_ctrl:
        sys.exit(f"Missing controllers: {miss_ctrl}")

    if len(wide) != 1080:
        print(f"Warning: expected 1080 paired episodes, got {len(wide)}.",
              file=sys.stderr)
    return wide


def paired_NI_test(p_col, t_col, delta_pp=DELTA):
    p = np.asarray(p_col, dtype=int)
    t = np.asarray(t_col, dtype=int)
    n = len(p)
    d = p - t
    diff_pp = 100.0 * d.mean()
    se_pp   = 100.0 * d.std(ddof=1) / math.sqrt(n)
    z_ni    = (diff_pp - delta_pp) / se_pp if se_pp > 0 else 0.0
    p_one   = stats.norm.cdf(z_ni)
    ci_lo   = diff_pp - 1.96 * se_pp
    ci_hi   = diff_pp + 1.96 * se_pp
    b = int(((p == 1) & (t == 0)).sum())
    c = int(((p == 0) & (t == 1)).sum())
    return dict(n=n, diff_pp=diff_pp, se_pp=se_pp,
                ci_lo_pp=ci_lo, ci_hi_pp=ci_hi,
                z_NI=z_ni, p_NI_one_sided=p_one,
                b_P_only=b, c_T_only=c, n_discordant=b + c)


def cross_check(s):
    issues = []
    if abs(s["diff_pp"] - (-0.56)) > 0.05:
        issues.append(f"diff = {s['diff_pp']:.3f} pp != -0.56 pp")
    if abs(s["se_pp"] - 0.32) > 0.02:
        issues.append(f"SE = {s['se_pp']:.3f} pp != 0.32 pp")
    if abs(s["ci_lo_pp"] - (-1.18)) > 0.05 or abs(s["ci_hi_pp"] - 0.07) > 0.05:
        issues.append(f"CI mismatch")
    if abs(math.log10(max(s["p_NI_one_sided"], 1e-300)) -
           math.log10(6.0e-7)) > 0.3:
        issues.append(f"p_NI = {s['p_NI_one_sided']:.2e} != ~6.0e-7")
    if (s["b_P_only"], s["c_T_only"]) != (3, 9):
        issues.append(f"discordant b={s['b_P_only']} c={s['c_T_only']} != (3,9)")
    if issues:
        print("CROSS-CHECK FAILED:", file=sys.stderr)
        for ii in issues:
            print(f"  - {ii}", file=sys.stderr)
        sys.exit(1)
    print("Cross-check passed: paired statistics match Section VI-C.")


def render_forest(rows, outfile="fig_h2_forest.pdf"):
    mpl.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size":        10,
        "xtick.labelsize":  9,
        "ytick.labelsize":  10,
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    y_pos = [3, 2, 1]   # top-to-bottom: P_{0.10}, P_{0.15}, P_{0.20}

    # shaded non-inferiority region
    ax.axvspan(-3.5, DELTA, alpha=0.18, color="#8FBF8F", zorder=0)
    ax.text(DELTA - 0.06, 3.55, "non-inferiority region",
            fontsize=7.5, ha="right", va="bottom",
            style="italic", color="#3a6e3a")

    ax.axvline(0,     color="0.45", lw=0.6, linestyle=":")
    ax.axvline(DELTA, color="0.20", lw=0.9, linestyle="--")

    # ---- FIX 1: Delta label uses Unicode thin space, not LaTeX "\," ----
    ax.text(DELTA + 0.05, 3.55, f"$\\Delta = 1${THIN}pp",
            fontsize=8, ha="left", va="bottom")

    for y, (label, m, se, p) in zip(y_pos, rows):
        lo, hi  = m - 1.96 * se, m + 1.96 * se
        is_pass = p < ALPHA
        col     = "#1a1a1a" if is_pass else "#777777"
        face    = col       if is_pass else "white"

        ax.plot([lo, hi], [y, y], color=col, lw=1.2, solid_capstyle="round")
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.13, y + 0.13], color=col, lw=1.2)
        ax.plot(m, y, marker="D", markersize=6.5,
                markerfacecolor=face, markeredgecolor=col,
                markeredgewidth=1.0)

        # p-value annotation, with a thin space before the star
        if p < 1e-3:
            mant, exp = f"{p:.1e}".split("e")
            p_str = rf"$p = {mant}\times10^{{{int(exp)}}}$"
        else:
            p_str = rf"$p = {p:.2g}$"
        suffix = f"{THIN}*" if is_pass else ""
        ax.annotate(p_str + suffix, xy=(hi + 0.10, y),
                    fontsize=8, va="center", ha="left", color=col)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_ylim(0.4, 3.95)
    ax.set_xlim(-3.0, 3.0)
    ax.set_xlabel(r"$P_\tau - T_{3.5}$ collision rate (pp)", labelpad=2)
    ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("0.4")
    ax.spines["bottom"].set_color("0.4")
    ax.tick_params(axis="both", which="both", length=3, color="0.4")

    # ---- FIX 2: footnote uses Unicode thin space, not LaTeX "\," ----
    ax.text(0.99, 1.04, f"*{THIN}formal pass at $\\alpha = 0.05$",
            transform=ax.transAxes, fontsize=7.5, ha="right",
            va="bottom", color="0.25")

    plt.tight_layout(pad=0.4)
    plt.savefig(outfile, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {outfile}")


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)

    wide = load_paired(sys.argv[1])

    s10 = paired_NI_test(wide["innov_010"], wide["ttc_35"])
    s15 = paired_NI_test(wide["innov_015"], wide["ttc_35"])
    s20 = paired_NI_test(wide["innov_020"], wide["ttc_35"])

    print("\nP_{0.20} vs T_{3.5} (the H2 primary comparator):")
    for k, v in s20.items():
        print(f"  {k:18s} = {v}")
    cross_check(s20)

    rows = [
        ("$P_{0.10}$", s10["diff_pp"], s10["se_pp"], s10["p_NI_one_sided"]),
        ("$P_{0.15}$", s15["diff_pp"], s15["se_pp"], s15["p_NI_one_sided"]),
        ("$P_{0.20}$", s20["diff_pp"], s20["se_pp"], s20["p_NI_one_sided"]),
    ]
    render_forest(rows, outfile="fig_h2_forest.pdf")


if __name__ == "__main__":
    main()