#!/usr/bin/env python3
"""
Reproduce Extended Data Figure 8 from source CSVs.

Panels:
  a - Mean age span by trajectory
  b - Cumulative distribution of transition timing
  c - Cross-sectional analysis (pooled population coupling vs age)
  d - Variance decomposition (R² comparison)
  e - Pairwise effect sizes and significance

Requirements: numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- load data ----------
script_dir = Path(__file__).resolve().parent
XLSX = script_dir / 'Supplementary_data.xlsx'
df_age = pd.read_excel(XLSX, sheet_name='Trajectory age span')
df_trans = pd.read_excel(XLSX, sheet_name='Transition timing')
df_cross = pd.read_excel(XLSX, sheet_name='Cross-sectional binned')
for col in df_cross.columns:
    df_cross[col] = pd.to_numeric(df_cross[col], errors='coerce')
df_var = pd.read_excel(XLSX, sheet_name='Variance decomposition')
df_eff = pd.read_excel(XLSX, sheet_name='Pairwise effect sizes')

# Colors
COLOR_SOLOIST = '#3498DB'
COLOR_CHORISTER = '#E74C3C'
COLOR_REFINER = '#9DCA24'
TRAJ_COLORS = {
    'Stable Soloist': COLOR_SOLOIST,
    'Stable Chorister': COLOR_CHORISTER,
    'Chorister-to-Soloist': COLOR_REFINER,
}

trajectories = ['Stable Soloist', 'Stable Chorister', 'Chorister-to-Soloist']

# ---------- create figure ----------
fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))

# ===== Panel a: Mean Age Span by Trajectory =====
ax = axes[0]
means = []
sems = []
for traj in trajectories:
    subset = df_age[df_age['trajectory'] == traj]['age_span']
    means.append(subset.mean())
    sems.append(subset.sem())

x_pos = np.arange(len(trajectories))
colors = [TRAJ_COLORS[t] for t in trajectories]

ax.bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black',
       yerr=sems, capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5})
ax.set_xticks(x_pos)
ax.set_xticklabels(trajectories, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Mean Age Span (days)', fontweight='bold')
ax.set_title('Mean Age Span\nby Trajectory', fontweight='bold', fontsize=11)
ax.set_ylim(0, max(np.array(means) + np.array(sems)) * 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ===== Panel b: Cumulative Distribution of Transition Timing =====
ax = axes[1]
transition_days = df_trans['transition_day'].values
sorted_days = np.sort(transition_days)
cdf = np.arange(1, len(sorted_days) + 1) / len(sorted_days) * 100

ax.plot(sorted_days, cdf, color=COLOR_REFINER, linewidth=3,
        marker='o', markersize=6, markerfacecolor='white',
        markeredgewidth=2, markeredgecolor=COLOR_REFINER)

# Critical period shading
ax.axvspan(21, 35, alpha=0.15, color='red', zorder=0)
ax.axvline(21, color='red', linestyle='--', linewidth=2, alpha=0.7,
           label='Critical Period (P21-P35)')
ax.axvline(35, color='red', linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Postnatal Day (P)', fontweight='bold', fontsize=11)
ax.set_ylabel('Cumulative Percentage (%)', fontweight='bold', fontsize=11)
ax.set_title('Cumulative Distribution of\nTransition Timing', fontweight='bold',
             fontsize=11)
ax.legend(fontsize=9, loc='upper left')
ax.set_xlim(10, 45)
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ===== Panel c: Cross-Sectional Analysis =====
ax = axes[2]

ax.errorbar(df_cross['age_bin_center'], df_cross['mean'],
            yerr=df_cross['sem'], fmt='o-', color='#7f7f7f',
            linewidth=2, markersize=6, capsize=3,
            label='All neurons (pooled)')
ax.fill_between(df_cross['age_bin_center'].values.astype(float),
                (df_cross['mean'] - df_cross['sem']).values.astype(float),
                (df_cross['mean'] + df_cross['sem']).values.astype(float),
                alpha=0.3, color='#7f7f7f')

# Regression line
slope = df_cross['regression_slope'].iloc[0]
intercept = df_cross['regression_intercept'].iloc[0]
r_val = df_cross['regression_r'].iloc[0]
p_val = df_cross['regression_p'].iloc[0]

x_line = np.array([10, 45])
ax.plot(x_line, intercept + slope * x_line, '--', color='black',
        alpha=0.5, linewidth=1)

ax.set_xlabel('Age (postnatal day)', fontweight='bold', fontsize=11)
ax.set_ylabel('Population Coupling', fontweight='bold', fontsize=11)
ax.set_title('Cross-Sectional\nAnalysis', fontweight='bold', fontsize=11)
ax.text(0.05, 0.95, f'r = {r_val:.3f}, p = {p_val:.2e}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top')
ax.set_xlim(10, 45)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ===== Panel d: Variance Decomposition =====
ax = axes[3]

models = df_var['model'].tolist()
r2_pct = df_var['r_squared'].values * 100
bar_colors = ['#888888', '#4477AA', '#44AA77']

bars = ax.bar(range(len(models)), r2_pct, color=bar_colors,
              edgecolor='black', linewidth=1)

for bar, val in zip(bars, r2_pct):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{val:.1f}%', ha='center', fontsize=9)

ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Variance Explained (R\u00b2)', fontweight='bold')
ax.set_title('Variance\nDecomposition', fontweight='bold', fontsize=11)
ax.set_ylim(0, max(r2_pct) * 1.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ===== Panel e: Pairwise Effect Sizes and Significance =====
ax = axes[4]

labels = ['Stable\nSoloist', 'Stable\nChorister', 'Chorister-to-\nSoloist']
traj_order = ['Stable Soloist', 'Stable Chorister', 'Chorister-to-Soloist']

# Build 3x3 matrix
matrix = np.full((3, 3), np.nan)
p_matrix = np.full((3, 3), np.nan)

for _, row in df_eff.iterrows():
    i = traj_order.index(row['group1'])
    j = traj_order.index(row['group2'])
    matrix[i, j] = row['cohens_d']
    matrix[j, i] = row['cohens_d']
    p_matrix[i, j] = row['p_value']
    p_matrix[j, i] = row['p_value']

im = ax.imshow(np.abs(matrix), cmap='RdYlGn', vmin=0, vmax=3, aspect='auto')

for i in range(3):
    for j in range(3):
        if i != j:
            pv = p_matrix[i, j]
            if pv < 0.001:
                sig = '***'
            elif pv < 0.01:
                sig = '**'
            elif pv < 0.05:
                sig = '*'
            else:
                sig = 'n.s.'
            ax.text(j, i, f'd={matrix[i, j]:.2f}\n{sig}',
                    ha='center', va='center', fontsize=9, fontweight='bold')

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(labels, fontsize=9)
ax.set_yticklabels(labels, fontsize=9)
ax.set_title('Pairwise Effect Sizes\nand Significance', fontweight='bold',
             fontsize=11)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("|Cohen's d|", fontweight='bold', fontsize=10)

# ---------- save ----------
plt.tight_layout()
plt.savefig(script_dir / 'trajectory_characterisation.png', dpi=300, bbox_inches='tight')
plt.savefig(script_dir / 'trajectory_characterisation.svg', bbox_inches='tight')
plt.close()
print('Saved trajectory_characterisation.png and trajectory_characterisation.svg')
