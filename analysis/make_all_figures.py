"""
Build all six figures for the IEEE T-ITS paper from real CARLA logs.

Inputs (relative to script directory):
    data/eval_master_combined_with_maxR.csv     (per-episode logs, 8640 rows)
    data/analysis_outputs/per_controller_bootstrap_ci.csv
    data/analysis_outputs/h1_calibration_convergence.csv
    data/analysis_outputs/h2_safety_noninferiority.csv
    data/analysis_outputs/h4_reproducibility.csv
    data/analysis_outputs/h6_comfort_noninferiority.csv

Outputs (PDF + PNG):
    fig_collision_rate.pdf  -- Fig. 3: collision-rate bar chart with 95% bootstrap CIs
    fig_pareto.pdf          -- Fig. 4: IR vs NMR Pareto trade-off
    fig_h1_calibration.pdf  -- Fig. 3 (renumbered as needed): target tau vs realised tau
    fig_h2_forest.pdf       -- Non-inferiority forest plot (P_tau - T_3.5)
    fig_h4_heatmap.pdf      -- Cross-seed CV heatmap (controllers x metrics)
    fig_h6_comfort.pdf      -- Comfort non-inferiority intervals (J_max, A_max)

IEEE journal column width = 3.5 in. Run twice if any caching issue arises.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------- IEEE journal style ----------------
rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "axes.linewidth":   0.5,
    "xtick.major.width":0.5,
    "ytick.major.width":0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth":  1.0,
    "patch.linewidth":  0.5,
    "axes.grid":        False,
    "savefig.dpi":      400,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "text.usetex":      False,
})

# Load real data
ci = pd.read_csv("data/analysis_outputs/per_controller_bootstrap_ci.csv")
master = pd.read_csv("data/eval_master_combined_with_maxR.csv")
master["replication"] = (master["seed"] // 1000).astype(int)

# Display labels
LBL = {
    "normal":     "NORMAL",
    "ttc_20":     r"$T_{2.0}$",
    "ttc_25":     r"$T_{2.5}$",
    "ttc_30":     r"$T_{3.0}$",
    "ttc_35":     r"$T_{3.5}$",
    "innov_010":  r"$P_{0.10}$",
    "innov_015":  r"$P_{0.15}$",
    "innov_020":  r"$P_{0.20}$",
}

# ====================================================================
# Fig. 3 — Collision rate bar chart with bootstrap 95% CIs
# ====================================================================
order = ["innov_010","innov_015","innov_020",
         "ttc_20","ttc_25","ttc_30","ttc_35",
         "normal"]

cr = (ci[ci.metric == "collision_any"]
      .set_index("controller")
      .loc[order, ["mean","ci_lo","ci_hi"]]
      * 100)  # to percent

fig, ax = plt.subplots(figsize=(3.5, 2.2))

x = np.arange(len(order), dtype=float)
# add extra spacing between ttc_35 and normal so labels don't collide
x[7] += 0.7
COL_INNOV  = "#1f4e9a"
COL_TTC    = "#999999"
COL_NORMAL = "#c1272d"

bar_colors = ([COL_INNOV]*3) + ([COL_TTC]*4) + [COL_NORMAL]
bar_alphas = ([1.0]*3) + ([0.85]*4) + [0.95]

for xi, ctrl, color, alpha in zip(x, order, bar_colors, bar_alphas):
    val   = cr.loc[ctrl,"mean"]
    lo,hi = cr.loc[ctrl,"ci_lo"], cr.loc[ctrl,"ci_hi"]
    bar = ax.bar(xi, val, width=0.7, color=color, alpha=alpha,
                 edgecolor="black", linewidth=0.4, zorder=2)
    ax.errorbar(xi, val, yerr=[[val-lo],[hi-val]], fmt="none",
                ecolor="black", elinewidth=0.7, capsize=2.0,
                capthick=0.6, zorder=3)
    # value label on top of CI cap
    label_y = hi + 0.6
    fontw   = "bold" if ctrl == "innov_020" else "normal"
    ax.text(xi, label_y, f"{val:.2f}", ha="center", va="bottom",
            fontsize=6.5, fontweight=fontw, zorder=4)

ax.set_xticks(x)
ax.set_xticklabels([LBL[c] for c in order], rotation=0)
ax.set_ylabel("Collision rate (\%)" if rcParams["text.usetex"] else "Collision rate (%)")
ax.set_ylim(0, 26)
ax.yaxis.set_ticks(np.arange(0,26,5))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", length=0)
ax.axhline(0, color="black", linewidth=0.5, zorder=1)

# Group separators
ax.axvline(2.5, color="0.7", linewidth=0.4, linestyle=":", zorder=1)
ax.axvline(7.1, color="0.7", linewidth=0.4, linestyle=":", zorder=1)

# Group legends
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COL_INNOV,  edgecolor="black", linewidth=0.4, label="Calibrated"),
    Patch(facecolor=COL_TTC,    edgecolor="black", linewidth=0.4, label="TTC grid"),
    Patch(facecolor=COL_NORMAL, edgecolor="black", linewidth=0.4, label="No FCA"),
]
ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(0.005, 0.99),
          frameon=False, handlelength=1.2, handleheight=0.8, borderpad=0.2,
          labelspacing=0.3)

plt.savefig("fig_collision_rate.pdf", bbox_inches="tight", pad_inches=0.02)
plt.savefig("fig_collision_rate.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
plt.close()
print("Wrote fig_collision_rate.pdf")

# ====================================================================
# Fig. 4 — Pareto trade-off  (IR vs NMR), with per-replication scatter
# ====================================================================
# Per-replication means
per_rep = (master.groupby(["replication","controller_label"])
           [["intervention_ratio","ttc_lt_3_ratio"]].mean()
           .rename(columns={"intervention_ratio":"IR","ttc_lt_3_ratio":"NMR"})
           .reset_index())
# Pooled means
pooled = (master.groupby("controller_label")
          [["intervention_ratio","ttc_lt_3_ratio"]].mean()
          .rename(columns={"intervention_ratio":"IR","ttc_lt_3_ratio":"NMR"}))

ttc_order   = ["ttc_20","ttc_25","ttc_30","ttc_35"]
innov_order = ["innov_010","innov_015","innov_020"]

fig, ax = plt.subplots(figsize=(3.5, 2.7))

# TTC frontier (pooled means)
ttc_x = pooled.loc[ttc_order,"IR"].values
ttc_y = pooled.loc[ttc_order,"NMR"].values
ax.plot(ttc_x, ttc_y, "--", color="#666666", linewidth=0.9, zorder=2,
        label="TTC tuning grid")

# Per-replication TTC scatter (small dots)
for ctrl in ttc_order:
    sub = per_rep[per_rep.controller_label == ctrl]
    ax.scatter(sub.IR, sub.NMR, s=6, marker="s", facecolor="white",
               edgecolor="#999999", linewidth=0.4, alpha=0.5, zorder=3)

# Pooled TTC markers (open squares)
ax.scatter(ttc_x, ttc_y, s=42, marker="s", facecolor="white",
           edgecolor="#444444", linewidth=1.0, zorder=4)

# Innov frontier (pooled means)
in_x = pooled.loc[innov_order,"IR"].values
in_y = pooled.loc[innov_order,"NMR"].values
ax.plot(in_x, in_y, "-", color=COL_INNOV, linewidth=1.6, zorder=5,
        label="Calibrated")

# Per-replication innov scatter
for ctrl in innov_order:
    sub = per_rep[per_rep.controller_label == ctrl]
    ax.scatter(sub.IR, sub.NMR, s=8, marker="o", facecolor=COL_INNOV,
               edgecolor="white", linewidth=0.3, alpha=0.30, zorder=6)
ax.scatter(in_x, in_y, s=42, marker="o", facecolor=COL_INNOV,
           edgecolor="white", linewidth=0.8, zorder=7)

# NORMAL marker
nx = pooled.loc["normal","IR"]
ny = pooled.loc["normal","NMR"]
ax.scatter([nx],[ny], s=42, marker="x", color=COL_NORMAL,
           linewidth=1.4, zorder=6, label="No FCA")
ax.annotate("NORMAL", (nx, ny), xytext=(4, -1), textcoords="offset points",
            color=COL_NORMAL, fontsize=7, va="center")

# TTC labels
ttc_lbl_offsets = {
    "ttc_20": (-2,  6, "right"),
    "ttc_25": ( 4, -8, "left"),
    "ttc_30": ( 5,  4, "left"),
    "ttc_35": ( 4,  4, "left"),
}
for ctrl in ttc_order:
    px = pooled.loc[ctrl,"IR"]; py = pooled.loc[ctrl,"NMR"]
    dx,dy,ha = ttc_lbl_offsets[ctrl]
    ax.annotate(LBL[ctrl], (px,py), xytext=(dx,dy), textcoords="offset points",
                ha=ha, va="center", fontsize=7, color="#333333")

# Innov labels
innov_lbl_offsets = {
    "innov_010": (-4,  6, "right"),
    "innov_015": ( 0,  7, "center"),
    "innov_020": ( 5, -5, "left"),
}
for ctrl in innov_order:
    px = pooled.loc[ctrl,"IR"]; py = pooled.loc[ctrl,"NMR"]
    dx,dy,ha = innov_lbl_offsets[ctrl]
    tau_str = LBL[ctrl]
    ax.annotate(tau_str, (px,py), xytext=(dx,dy), textcoords="offset points",
                ha=ha, va="center", fontsize=7, color=COL_INNOV,
                fontweight="bold")

# Annotate the dominance gap at IR≈0.10
# At matched IR, the calibrated frontier sits ~1pp below the TTC one
p20_x, p20_y = pooled.loc["innov_020","IR"], pooled.loc["innov_020","NMR"]
t30_x, t30_y = pooled.loc["ttc_30","IR"], pooled.loc["ttc_30","NMR"]
gap_pp = (t30_y - p20_y) * 100  # percentage points
# Vertical bracket at IR ~ 0.103 showing the gap
gap_x = 0.102
ax.annotate("", xy=(gap_x, t30_y - 0.0005), xytext=(gap_x, p20_y + 0.0005),
            arrowprops=dict(arrowstyle="<->", color="#c1272d", lw=0.8))
ax.text(gap_x + 0.002, (p20_y + t30_y) / 2,
        f"$\\Delta$NMR\n$\\approx{gap_pp:.2f}$ pp\n($\\sim$10\\% rel.)" if rcParams["text.usetex"]
        else f"\u0394NMR\n\u2248 {gap_pp:.2f} pp\n(~10% rel.)",
        ha="left", va="center", fontsize=6.5, color="#c1272d")

ax.set_xlabel("Mean intervention rate (IR)")
ax.set_ylabel("Mean TTC$<$3 s exposure (NMR)")
ax.set_xlim(-0.005, 0.135)
ax.set_ylim(0.078, 0.110)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", frameon=False, handlelength=1.4,
          borderpad=0.2, labelspacing=0.3)
ax.grid(True, linestyle=":", linewidth=0.4, color="0.85", zorder=0)

plt.savefig("fig_pareto.pdf", bbox_inches="tight", pad_inches=0.02)
plt.savefig("fig_pareto.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
plt.close()
print("Wrote fig_pareto.pdf")

# Print verification numbers
print("\n=== Verification: pooled means used in figs ===")
print(pooled.round(4))


# ===== EXTRA FIGURES =====


COL_INNOV  = "#1f4e9a"
COL_TTC    = "#999999"
COL_NORMAL = "#c1272d"
COL_PASS   = "#2a8a3a"
COL_FAIL   = "#c1272d"

# ========================================================================
# Fig A — H1 calibration convergence (target vs realised, 9 cells)
# ========================================================================
h1 = pd.read_csv("data/analysis_outputs/h1_calibration_convergence.csv")
fig, ax = plt.subplots(figsize=(3.5, 2.2))

# Identity diagonal
ax.plot([0.05, 0.25], [0.05, 0.25], "-", color="black", linewidth=0.7, zorder=1, label=r"$\hat\tau = \tau$")
# Pre-registered ±0.03 tolerance band
xs = np.linspace(0.05, 0.25, 50)
ax.fill_between(xs, xs - 0.03, xs + 0.03, color=COL_PASS, alpha=0.10, zorder=0,
                label=r"Pre-reg tolerance $\pm 0.03$")

# 9 dots, colour by seed
seed_colors = {1000: "#1f4e9a", 2000: "#5798d2", 3000: "#a5c8e1"}
seed_markers = {1000: "o", 2000: "s", 3000: "^"}
for seed in [1000, 2000, 3000]:
    sub = h1[h1.seed == seed]
    ax.scatter(sub.target_tau, sub.realized_rate, s=42,
               marker=seed_markers[seed], facecolor=seed_colors[seed],
               edgecolor="white", linewidth=0.8,
               label=f"seed {seed}", zorder=4)

# Annotate the worst case (largest |diff|)
worst = h1.iloc[h1.delta_from_target.abs().idxmax()]
ax.annotate(f"max |$\\hat\\tau-\\tau$| = {worst.delta_from_target:+.3f}",
            xy=(worst.target_tau, worst.realized_rate),
            xytext=(0.105, 0.232),
            fontsize=6.5, color="#444444",
            arrowprops=dict(arrowstyle="-", lw=0.4, color="#888888"))

ax.set_xlabel(r"Target violation rate $\tau$")
ax.set_ylabel(r"Realised rate $\hat\tau$")
ax.set_xlim(0.07, 0.23); ax.set_ylim(0.07, 0.235)
ax.set_xticks([0.10, 0.15, 0.20])
ax.set_yticks([0.08, 0.12, 0.16, 0.20])
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", frameon=False, handlelength=1.2,
          borderpad=0.2, labelspacing=0.3)
ax.grid(True, linestyle=":", linewidth=0.4, color="0.85", zorder=0)
plt.savefig("fig_h1_calibration.pdf"); plt.savefig("fig_h1_calibration.png", dpi=300)
plt.close()
print("Wrote fig_h1_calibration.pdf")

# ========================================================================
# Fig B — H2 non-inferiority forest plot
# ========================================================================
h2 = pd.read_csv("data/analysis_outputs/h2_safety_noninferiority.csv")
fig, ax = plt.subplots(figsize=(3.5, 1.8))

# We have CIs on innov rate; need CI on the *difference*.
# Wald 95% CI on diff = (p1-p2) ± 1.96 sqrt(p1(1-p1)/n + p2(1-p2)/n)
# We have p1 (innov), p2 (ttc_35 = 0.0417), and n.
n = 1080
p2 = 0.04166666666666666
labels = []
diffs = []
ci_los = []
ci_his = []
pvals = []
for _, row in h2.iterrows():
    p1 = row.innov_collision_rate
    se = np.sqrt(p1*(1-p1)/n + p2*(1-p2)/n)
    diff = (p1 - p2) * 100  # in pp
    ci_lo = (p1 - p2 - 1.96*se) * 100
    ci_hi = (p1 - p2 + 1.96*se) * 100
    name = row.controller.replace("innov_0","P_{0.").replace("0","0") + "}"
    # nicer label
    name_map = {"innov_010":r"$P_{0.10}$",
                "innov_015":r"$P_{0.15}$",
                "innov_020":r"$P_{0.20}$"}
    labels.append(name_map[row.controller])
    diffs.append(diff); ci_los.append(ci_lo); ci_his.append(ci_hi)
    pvals.append(row.p_noninferior)

# horizontal forest
y = np.arange(len(labels))[::-1]
margin_pp = 1.0  # pre-registered Δ = 0.01 in proportion = 1.0 pp

# Margin band  (non-inferior region: diff <= margin_pp)
ax.axvspan(-3.0, margin_pp, color=COL_PASS, alpha=0.06, zorder=0)
ax.axvline(0, color="black", linewidth=0.5, zorder=1)
ax.axvline(margin_pp, color=COL_PASS, linewidth=0.7, linestyle="--", zorder=1, label=r"Non-inf. margin $\Delta=1$\,pp")

for yi, lo, hi, d, p in zip(y, ci_los, ci_his, diffs, pvals):
    pass_ni = p < 0.05  # one-sided test result, matches h2 CSV
    color = COL_PASS if pass_ni else "#888888"
    ax.plot([lo, hi], [yi, yi], color=color, lw=1.2, zorder=3)
    ax.plot([lo, lo], [yi-0.12, yi+0.12], color=color, lw=1.0, zorder=3)
    ax.plot([hi, hi], [yi-0.12, yi+0.12], color=color, lw=1.0, zorder=3)
    ax.scatter([d], [yi], s=44, marker="D", facecolor=color, edgecolor="white",
               linewidth=0.7, zorder=4)
    star = " *" if pass_ni else ""
    ax.text(hi + 0.18, yi, f"$p={p:.3f}${star}", fontsize=6.5, va="center", color=color)

ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel(r"$P_\tau$ minus $T_{3.5}$ collision rate (pp)")
ax.set_xlim(-3.0, 3.5)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
# Annotate the margin line directly (cleaner than a legend that fights for space)
ax.text(margin_pp + 0.04, -0.5, r"$\Delta=1$ pp", color=COL_PASS, fontsize=6.5,
        ha="left", va="top", rotation=0)
ax.grid(True, axis="x", linestyle=":", linewidth=0.4, color="0.85", zorder=0)
ax.set_ylim(-0.7, 2.7)
ax.invert_yaxis()
plt.savefig("fig_h2_forest.pdf"); plt.savefig("fig_h2_forest.png", dpi=300)
plt.close()
print("Wrote fig_h2_forest.pdf")

# ========================================================================
# Fig C — H4 reproducibility heatmap (controller × metric)
# ========================================================================
h4 = pd.read_csv("data/analysis_outputs/h4_reproducibility.csv")

# Pivot: rows=controllers, cols=metrics, values=CV
ctrl_order = ["innov_010","innov_015","innov_020",
              "ttc_20","ttc_25","ttc_30","ttc_35","normal"]
metric_order = ["collision_any","ttc_lt_3_ratio","intervention_rate",
                "max_abs_jerk","mean_pct_speed_diff","min_ttc"]
metric_labels = ["CR","NMR","IR",r"$J_{\max}$",r"$\Delta v\%$",r"$T_{\min}$"]
ctrl_labels = [r"$P_{0.10}$",r"$P_{0.15}$",r"$P_{0.20}$",
               r"$T_{2.0}$",r"$T_{2.5}$",r"$T_{3.0}$",r"$T_{3.5}$","NORMAL"]

cv = h4.pivot(index="controller", columns="metric", values="cv").reindex(
    index=ctrl_order, columns=metric_order)

fig, ax = plt.subplots(figsize=(3.5, 2.4))
# Custom colour scale: green <0.20, yellow 0.20-0.50, red >0.50
import matplotlib.colors as mcolors
cmap = mcolors.LinearSegmentedColormap.from_list(
    "h4cv", ["#2a8a3a","#bfd14a","#f5c84a","#e07a3a","#c1272d"])
im = ax.imshow(cv.values, cmap=cmap, vmin=0.0, vmax=0.65, aspect="auto")

# Annotate cells with CV value, marking pass/fail
for i, ctrl in enumerate(ctrl_order):
    for j, metric in enumerate(metric_order):
        v = cv.iloc[i, j]
        if pd.isna(v):
            ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="white")
        else:
            txt_color = "white" if v > 0.30 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color)

ax.set_xticks(range(len(metric_order))); ax.set_xticklabels(metric_labels)
ax.set_yticks(range(len(ctrl_order))); ax.set_yticklabels(ctrl_labels)
ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", length=0)
ax.set_xlabel("Metric")

# Colour bar with thresholds marked
cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04, aspect=18)
cbar.set_label("Cross-seed CV", fontsize=7)
cbar.ax.tick_params(labelsize=6.5)
# Threshold lines on the colorbar
cbar.ax.axhline(0.20, color="black", linewidth=0.7)
cbar.ax.axhline(0.50, color="black", linewidth=0.7)
# Threshold legend below the colorbar (in figure coords, anchored to cbar.ax)
cbar.ax.annotate("0.20: stable", xy=(0.5, -0.04), xycoords="axes fraction",
                 ha="center", va="top", fontsize=6, color="black")
cbar.ax.annotate("0.50: fail",   xy=(0.5, -0.10), xycoords="axes fraction",
                 ha="center", va="top", fontsize=6, color=COL_FAIL)

plt.savefig("fig_h4_heatmap.pdf"); plt.savefig("fig_h4_heatmap.png", dpi=300)
plt.close()
print("Wrote fig_h4_heatmap.pdf")

# ========================================================================
# Fig D — H6 comfort non-inferiority (J_max and A_max)
# ========================================================================
h6 = pd.read_csv("data/analysis_outputs/h6_comfort_noninferiority.csv")

fig, ax = plt.subplots(figsize=(3.5, 2.0))

# Two metric panels side by side via twin x-axes? Simpler: single panel,
# normalise each interval to "fraction of margin" so both metrics share an axis.
# normalised diff = (mean_diff / margin),  CIs likewise.

h6 = h6.copy()
h6["nd"]      = h6.mean_diff / h6.margin
h6["nd_lo"]   = h6.bonf_ci_lo / h6.margin
h6["nd_hi"]   = h6.bonf_ci_hi / h6.margin
h6["pretty_ctrl"] = h6.controller.map({"innov_010":r"$P_{0.10}$",
                                       "innov_015":r"$P_{0.15}$",
                                       "innov_020":r"$P_{0.20}$"})
h6["pretty_metric"] = h6.metric.map({"max_abs_jerk":r"$J_{\max}$",
                                     "max_abs_accel":r"$A_{\max}$"})

# Layout: y position by (controller, metric)
labels = []; ys = []
metric_groups = ["max_abs_jerk","max_abs_accel"]
ctrls = ["innov_010","innov_015","innov_020"]
yi = 0
y_positions = {}
for m in metric_groups:
    for c in ctrls:
        labels.append(f"{ {'innov_010':'$P_{0.10}$','innov_015':'$P_{0.15}$','innov_020':'$P_{0.20}$'}[c] }")
        y_positions[(m,c)] = yi
        yi += 1
    yi += 0.6  # gap between metric groups

# Margin envelope (normalized to 1)
ax.axvspan(-2.5, 1.0, color=COL_PASS, alpha=0.06, zorder=0)
ax.axvline(0,   color="black", linewidth=0.5, zorder=1)
ax.axvline(1.0, color=COL_PASS, linewidth=0.7, linestyle="--", zorder=1,
           label=r"$0.2\sigma$ margin")
ax.axvline(-1.0, color="0.7", linewidth=0.5, linestyle=":", zorder=1)

for _, row in h6.iterrows():
    yv = y_positions[(row.metric, row.controller)]
    pass_ni = row.non_inferior
    color = COL_PASS if pass_ni else COL_FAIL
    ax.plot([row.nd_lo, row.nd_hi], [yv, yv], color=color, lw=1.3, zorder=3)
    ax.plot([row.nd_lo, row.nd_lo], [yv-0.13, yv+0.13], color=color, lw=1.0, zorder=3)
    ax.plot([row.nd_hi, row.nd_hi], [yv-0.13, yv+0.13], color=color, lw=1.0, zorder=3)
    ax.scatter([row.nd], [yv], s=42, marker="D", facecolor=color,
               edgecolor="white", linewidth=0.7, zorder=4)

# y labels (just the controllers, repeated for each metric group)
yticks = []; ylabels = []
for m in metric_groups:
    for c in ctrls:
        yticks.append(y_positions[(m,c)])
        ylabels.append({"innov_010":r"$P_{0.10}$","innov_015":r"$P_{0.15}$","innov_020":r"$P_{0.20}$"}[c])
ax.set_yticks(yticks); ax.set_yticklabels(ylabels)

# Group-band labels on the right side, inside the plot area
jmax_y = np.mean([y_positions[("max_abs_jerk",c)] for c in ctrls])
amax_y = np.mean([y_positions[("max_abs_accel",c)] for c in ctrls])
ax.text(1.42, jmax_y, r"$J_{\max}$", fontsize=8, ha="right", va="center",
        fontweight="bold", color="#444444")
ax.text(1.42, amax_y, r"$A_{\max}$", fontsize=8, ha="right", va="center",
        fontweight="bold", color="#444444")
# Light separator between metric groups
sep_y = (jmax_y + amax_y) / 2
ax.axhline(sep_y, color="0.85", linewidth=0.5, linestyle="-", zorder=1)

ax.invert_yaxis()
ax.set_xlim(-2.5, 1.5)
ax.set_xlabel(r"Mean difference vs. $T_{2.5}$, normalised to $0.2\sigma$ margin")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(loc="lower left", frameon=False, handlelength=1.4, borderpad=0.2, fontsize=6.5)
ax.grid(True, axis="x", linestyle=":", linewidth=0.4, color="0.85", zorder=0)
plt.savefig("fig_h6_comfort.pdf"); plt.savefig("fig_h6_comfort.png", dpi=300)
plt.close()
print("Wrote fig_h6_comfort.pdf")
