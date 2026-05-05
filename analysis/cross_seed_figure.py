"""Cross-seed reproducibility scatterplot in grayscale.

Three panels: seed1000 vs seed2000, seed1000 vs seed3000, seed2000 vs seed3000.
Each point is one of the 8 controllers; x = collision rate in seed X,
y = collision rate in seed Y. Diagonal y=x indicates perfect reproducibility.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('combined_master.csv')
g = df.groupby(['seed0', 'controller_label'])['collision_any'].mean().unstack()
g_pct = g * 100  # to percent

# Order
order = ['innov_010', 'innov_015', 'innov_020',
         'ttc_20', 'ttc_25', 'ttc_30', 'ttc_35', 'normal']
g_pct = g_pct[order]

# Marker styles per controller family
def marker_style(ctrl):
    if ctrl.startswith('innov'):
        return dict(marker='o', s=80, facecolor='black', edgecolor='black', linewidths=1.0)
    if ctrl.startswith('ttc'):
        return dict(marker='s', s=80, facecolor='white', edgecolor='black', linewidths=1.4)
    return dict(marker='X', s=110, facecolor='#aaaaaa', edgecolor='black', linewidths=1.0)

label_for = {
    'innov_010': r'$\tau{=}0.10$',
    'innov_015': r'$\tau{=}0.15$',
    'innov_020': r'$\tau{=}0.20$',
    'ttc_20': r'$T_{2.0}$',
    'ttc_25': r'$T_{2.5}$',
    'ttc_30': r'$T_{3.0}$',
    'ttc_35': r'$T_{3.5}$',
    'normal': 'N',
}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

pairs = [(1000, 2000), (1000, 3000), (2000, 3000)]
from scipy.stats import spearmanr

for ax, (s1, s2) in zip(axes, pairs):
    x = g_pct.loc[s1].values
    y = g_pct.loc[s2].values
    rho, p = spearmanr(x, y)
    
    # Identity line
    lim_max = max(x.max(), y.max()) * 1.08
    ax.plot([0, lim_max], [0, lim_max], 'k--', linewidth=0.8, alpha=0.5,
            label='$y=x$ (perfect reproducibility)')
    
    # Plot each controller
    for ctrl in order:
        s = marker_style(ctrl)
        ax.scatter([g_pct.loc[s1, ctrl]], [g_pct.loc[s2, ctrl]], **s, zorder=3)
        ax.annotate(label_for[ctrl],
                    (g_pct.loc[s1, ctrl], g_pct.loc[s2, ctrl]),
                    xytext=(7, 4), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel(f'Collision rate, seed {s1} (%)', fontsize=11)
    ax.set_ylabel(f'Collision rate, seed {s2} (%)', fontsize=11)
    ax.set_title(rf'Seed {s1} vs seed {s2}: $\rho={rho:.3f}$ ($p={p:.0e}$)',
                 fontsize=11)
    ax.set_xlim(0, lim_max)
    ax.set_ylim(0, lim_max)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')

# Single legend, top-most subplot only
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='black', markerfacecolor='black',
           markersize=9, linestyle='None', label='Innov (calibrated)'),
    Line2D([0], [0], marker='s', color='black', markerfacecolor='white',
           markersize=9, linestyle='None', markeredgewidth=1.4, label='TTC tuning grid'),
    Line2D([0], [0], marker='X', color='black', markerfacecolor='#aaaaaa',
           markersize=11, linestyle='None', label='Normal (no FCA)'),
    Line2D([0], [0], color='black', linestyle='--', linewidth=0.8,
           alpha=0.6, label='$y=x$'),
]
axes[0].legend(handles=legend_elements, loc='upper left', fontsize=9,
               framealpha=0.95)

plt.suptitle('Cross-seed reproducibility of collision rate (8 controllers per panel)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('figure_cross_seed_bw.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved figure_cross_seed_bw.png')

# Print correlations for use in caption / paper
print()
for s1, s2 in pairs:
    rho, p = spearmanr(g_pct.loc[s1].values, g_pct.loc[s2].values)
    print(f'  {s1} vs {s2}: rho={rho:.4f}, p={p:.2e}')
