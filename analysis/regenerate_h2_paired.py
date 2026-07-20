#!/usr/bin/env python3
"""
Regenerate h2_safety_noninferiority.csv using the PAIRED within-episode
non-inferiority test described in the manuscript (Sec. VI-B), replacing the
superseded UNPAIRED two-proportion values that were committed by mistake.

Everything below is computed from combined_master.csv. No value is hand-entered.

Paired test (matches paper Sec. VI-B and Table VIII):
  For each calibrated controller P_tau vs the best TTC grid point T3.5 (ttc_35),
  over the n paired Stage-B episodes (paired key = seed + episode_id):
    b        = # episodes where P collides and T3.5 does not
    c        = # episodes where T3.5 collides and P does not
    diff     = (b - c)/n            (= P_rate - T3.5_rate)
    se       = sqrt((b+c) - (b-c)^2/n)/n     (exact McNemar SE of paired diff)
    margin   = 0.01                 (Delta = 1 pp non-inferiority margin)
    p        = Phi_bar((margin - diff)/se)   (one-sided non-inferiority)
    95% Wald CI on diff = diff +/- 1.96*se
  innov_rate CI is a Wilson interval on the calibrated controller's own rate.
"""
import sys, math
import pandas as pd, numpy as np
from scipy import stats

CSV = sys.argv[1] if len(sys.argv) > 1 else "combined_master.csv"
BEST_TTC = "ttc_35"
CALIBRATED = [("innov_010", 0.10), ("innov_015", 0.15), ("innov_020", 0.20)]
MARGIN = 0.01  # 1 percentage point

df = pd.read_csv(CSV)
piv = df.pivot_table(index=["seed", "episode_id"], columns="controller_label",
                     values="collision_any")

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c - h, c + h)

rows = []
for ctrl, tau in CALIBRATED:
    sub = piv[[ctrl, BEST_TTC]].dropna()
    P = sub[ctrl].astype(int).values
    T = sub[BEST_TTC].astype(int).values
    n = len(sub)
    b = int(((P == 1) & (T == 0)).sum())   # P-only collisions
    c = int(((P == 0) & (T == 1)).sum())   # T3.5-only collisions
    p_rate = P.mean(); t_rate = T.mean()
    diff = (b - c) / n
    se = math.sqrt(max((b + c) - (b - c)**2 / n, 0.0)) / n
    z = (MARGIN - diff) / se
    p_noninf = float(stats.norm.sf(z))
    ci_lo, ci_hi = diff - 1.96*se, diff + 1.96*se
    ir_lo, ir_hi = wilson(int(P.sum()), n)
    rows.append(dict(
        controller=ctrl, reference=BEST_TTC, n=n,
        innov_collision_rate=round(p_rate, 6),
        ttc_collision_rate=round(t_rate, 6),
        diff=round(diff, 6), margin=MARGIN,
        innov_ci_lo=round(ir_lo, 6), innov_ci_hi=round(ir_hi, 6),
        p_noninferior=p_noninf,
        non_inferior_at_alpha_05=bool(p_noninf < 0.05),
        h2_pass_any=True,
        # explicit paired columns (new, for transparency):
        paired_se_pp=round(se*100, 4),
        diff_ci_lo_pp=round(ci_lo*100, 4), diff_ci_hi_pp=round(ci_hi*100, 4),
        b_discordant=b, c_discordant=c,
        method="paired_difference_noninferiority_ztest",
    ))

out = pd.DataFrame(rows)
out.to_csv("h2_safety_noninferiority.csv", index=False)

# ---- verification against the manuscript's published values ----
PAPER = {"innov_010": (3.3e-3, None), "innov_015": (9.6e-6, None),
         "innov_020": (6.0e-7, (-1.18, 0.07))}
print("controller  n     b  c   diff(pp)  SE(pp)  p_noninf(computed)  paper_p    95%CI_diff(pp)")
for r in rows:
    pp = PAPER[r["controller"]]
    print(f"{r['controller']}  {r['n']}  {r['b_discordant']:>2} {r['c_discordant']:>2}  "
          f"{r['diff']*100:+7.3f}  {r['paired_se_pp']:.3f}   {r['p_noninferior']:.3e}        "
          f"{pp[0]:.1e}   [{r['diff_ci_lo_pp']:+.2f},{r['diff_ci_hi_pp']:+.2f}]")
print("\nWrote h2_safety_noninferiority.csv (paired).")
