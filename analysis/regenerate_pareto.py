"""Regenerate the Pareto trade-off figure with innov_020 vs ttc_30 annotated.

The annotation highlights the strongest 'Innov below TTC frontier' comparison:
at nearly identical intervention rates (0.107 vs 0.098), innov_020 achieves
~10% lower TTC<3s exposure than ttc_30.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

b = pd.read_csv('per_controller_bootstrap_ci.csv')

points = {}
for ctrl in b['controller'].unique():
    sub = b[b['controller']==ctrl]
    ir = sub[sub['metric']=='intervention_rate'].iloc[0]
    tt3 = sub[sub['metric']=='ttc_lt_3_ratio'].iloc[0]
    points[ctrl] = {
        'ir_mean': ir['mean'], 'ir_lo': ir['ci_lo'], 'ir_hi': ir['ci_hi'],
        'tt3_mean': tt3['mean'], 'tt3_lo': tt3['ci_lo'], 'tt3_hi': tt3['ci_hi'],
    }

innov = ['innov_010', 'innov_015', 'innov_020']
ttc = ['ttc_20', 'ttc_25', 'ttc_30', 'ttc_35']

fig, ax = plt.subplots(figsize=(8.0, 6.0))

# Innov line
ix = [points[c]['ir_mean'] for c in innov]
iy = [points[c]['tt3_mean'] for c in innov]
ix_err = [[points[c]['ir_mean']-points[c]['ir_lo'] for c in innov],
          [points[c]['ir_hi']-points[c]['ir_mean'] for c in innov]]
iy_err = [[points[c]['tt3_mean']-points[c]['tt3_lo'] for c in innov],
          [points[c]['tt3_hi']-points[c]['tt3_mean'] for c in innov]]
ax.errorbar(ix, iy, xerr=ix_err, yerr=iy_err, fmt='o-', color='#1f77b4',
            markersize=10, linewidth=1.8, capsize=3, label='Innov (calibrated)')

# TTC line
tx = [points[c]['ir_mean'] for c in ttc]
ty = [points[c]['tt3_mean'] for c in ttc]
tx_err = [[points[c]['ir_mean']-points[c]['ir_lo'] for c in ttc],
          [points[c]['ir_hi']-points[c]['ir_mean'] for c in ttc]]
ty_err = [[points[c]['tt3_mean']-points[c]['tt3_lo'] for c in ttc],
          [points[c]['tt3_hi']-points[c]['tt3_mean'] for c in ttc]]
ax.errorbar(tx, ty, xerr=tx_err, yerr=ty_err, fmt='s-', color='#ff7f0e',
            markersize=10, linewidth=1.8, capsize=3, label='TTC tuning grid')

# Normal point
n = points['normal']
ax.errorbar([n['ir_mean']], [n['tt3_mean']],
            xerr=[[0],[0]],
            yerr=[[n['tt3_mean']-n['tt3_lo']], [n['tt3_hi']-n['tt3_mean']]],
            fmt='x', color='gray', markersize=12, capsize=3,
            label='Normal (no FCA)')

# Innov labels
innov_labels = {'innov_010': r'$\tau$=0.10', 'innov_015': r'$\tau$=0.15',
                'innov_020': r'$\tau$=0.20'}
for c in innov:
    ax.annotate(innov_labels[c],
                (points[c]['ir_mean'], points[c]['tt3_mean']),
                xytext=(8, 4), textcoords='offset points', fontsize=10)

# TTC labels
ttc_labels = {'ttc_20': 'TTC=20', 'ttc_25': 'TTC=25',
              'ttc_30': 'TTC=30', 'ttc_35': 'TTC=35'}
for c in ttc:
    ax.annotate(ttc_labels[c],
                (points[c]['ir_mean'], points[c]['tt3_mean']),
                xytext=(8, 4), textcoords='offset points', fontsize=10)

# Normal label
ax.annotate('normal',
            (n['ir_mean'], n['tt3_mean']),
            xytext=(8, 0), textcoords='offset points', fontsize=10)

# === Annotation: innov_020 vs ttc_30 ===
i20 = points['innov_020']
t30 = points['ttc_30']

# Vertical dashed line at IR = midpoint between the two
mid_ir = (i20['ir_mean'] + t30['ir_mean']) / 2
ax.axvline(mid_ir, ymin=0.05, ymax=0.95, linestyle='--',
           color='black', alpha=0.35, linewidth=1.0)

# Bracket annotation showing exposure gap
gap_x = mid_ir + 0.005  # offset right of the dashed line
ax.annotate('', xy=(gap_x, t30['tt3_mean']), xytext=(gap_x, i20['tt3_mean']),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.4))
gap_pct = 100.0 * (t30['tt3_mean'] - i20['tt3_mean']) / t30['tt3_mean']
ax.text(gap_x + 0.003, (i20['tt3_mean']+t30['tt3_mean'])/2,
        f'{(t30["tt3_mean"]-i20["tt3_mean"])*100:.2f} pp\n'
        f'({gap_pct:.0f}% relative)',
        fontsize=9, va='center',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='black', alpha=0.85))

# Subtitle text in upper-right area, away from legend
ax.text(0.97, 0.97,
        r'At matched intervention ($\sim$0.10):'
        '\n'
        r'Innov $\tau$=0.20 has $\sim$10% lower'
        '\n'
        r'TTC<3s exposure than TTC=30'
        '\n'
        '(CIs nearly disjoint).',
        transform=ax.transAxes, fontsize=9, style='italic',
        ha='right', va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffbe6',
                  edgecolor='#888', alpha=0.95))

ax.set_xlabel('Mean intervention rate', fontsize=11)
ax.set_ylabel('Mean TTC<3s exposure ratio', fontsize=11)
ax.set_title('Pareto trade-off: safety vs intervention\n'
             '(Innov framework vs TTC tuning grid, 95% bootstrap CI)',
             fontsize=11)
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.01, 0.135)
ax.set_ylim(0.073, 0.112)

plt.tight_layout()
plt.savefig('figure_pareto_annotated.png', dpi=150, bbox_inches='tight')
print('Saved figure_pareto_annotated.png')

# Print the comparison numbers used in the annotation
print()
print(f'innov_020:  IR={i20["ir_mean"]:.4f} [{i20["ir_lo"]:.4f}, {i20["ir_hi"]:.4f}]  '
      f'TTC<3={i20["tt3_mean"]:.4f} [{i20["tt3_lo"]:.4f}, {i20["tt3_hi"]:.4f}]')
print(f'ttc_30:     IR={t30["ir_mean"]:.4f} [{t30["ir_lo"]:.4f}, {t30["ir_hi"]:.4f}]  '
      f'TTC<3={t30["tt3_mean"]:.4f} [{t30["tt3_lo"]:.4f}, {t30["tt3_hi"]:.4f}]')
print(f'TTC<3 absolute gap: {(t30["tt3_mean"]-i20["tt3_mean"])*100:.2f} pp')
print(f'TTC<3 relative gap: {gap_pct:.1f}%')
print(f'CI overlap on TTC<3: '
      f'innov_020 upper {i20["tt3_hi"]:.4f}  vs  ttc_30 lower {t30["tt3_lo"]:.4f}  '
      f'-> {"NEARLY DISJOINT" if i20["tt3_hi"] < t30["tt3_lo"] + 0.005 else "OVERLAPS"}')
