#!/usr/bin/env python3
"""
Reproduce Extended Data Figure 9 (Regional Differences) from source CSVs.

Panels:
  a - Population coupling histogram by region (V1 vs mPFC)
  b - Regional comparison violin plot
  c - Cluster distribution by region (stacked bar)
  d - Trajectory proportions by region (grouped bar + SEM)

Requirements: numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ---------- configuration ----------
COLOR_V1 = '#C37AB2'      # pink/purple
COLOR_mPFC = '#F9A11E'    # orange
COLOR_C0 = '#3498DB'      # blue  (low coupling)
COLOR_C1 = '#E74C3C'      # red   (high coupling)

TRAJ_COLORS = {
    'Stable Soloist': '#3498DB',
    'Stable Chorister': '#E74C3C',
    'Chorister-to-Soloist': '#9DCA24',
}
TRAJ_ORDER = ['Stable Soloist', 'Stable Chorister', 'Chorister-to-Soloist']

# ---------- load data ----------
script_dir = Path(__file__).resolve().parent
XLSX = script_dir / 'Supplementary_data.xlsx'
df_reg = pd.read_excel(XLSX, sheet_name='Regional clustering')
df_prop = pd.read_excel(XLSX, sheet_name='Trajectory proportions')

# ---------- create figure ----------
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# ===== Panel a: Population coupling histogram by region =====
ax = axes[0]
v1_vals = df_reg[df_reg['region'] == 'V1']['pop_coupling'].values
mpfc_vals = df_reg[df_reg['region'] == 'mPFC']['pop_coupling'].values

all_vals = np.concatenate([v1_vals, mpfc_vals])
bin_edges = np.linspace(all_vals.min(), all_vals.max(), 41)

ax.hist(v1_vals, bins=bin_edges, alpha=0.6, color=COLOR_V1, density=True, label='V1')
ax.hist(mpfc_vals, bins=bin_edges, alpha=0.6, color=COLOR_mPFC, density=True, label='mPFC')

ax.axvline(np.median(v1_vals), color=COLOR_V1, linestyle='--', linewidth=2.5,
           label=f'V1 median={np.median(v1_vals):.3f}')
ax.axvline(np.median(mpfc_vals), color=COLOR_mPFC, linestyle='--', linewidth=2.5,
           label=f'mPFC median={np.median(mpfc_vals):.3f}')

u_stat, p_mw = stats.mannwhitneyu(v1_vals, mpfc_vals)
ax.text(0.98, 0.98, f'Mann-Whitney U\np = {p_mw:.2e}',
        transform=ax.transAxes, fontsize=9, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlabel('Population Coupling', fontweight='bold', fontsize=11)
ax.set_ylabel('Density', fontweight='bold', fontsize=11)
ax.set_title('Pop Coupling Distribution by Region', fontweight='bold', fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ===== Panel b: Violin plot by region =====
ax = axes[1]
parts = ax.violinplot([v1_vals, mpfc_vals], positions=[0, 1], widths=0.7,
                       showmeans=True, showmedians=True)
for pc, color in zip(parts['bodies'], [COLOR_V1, COLOR_mPFC]):
    pc.set_facecolor(color)
    pc.set_alpha(0.6)

ax.set_xticks([0, 1])
ax.set_xticklabels(['V1\n(Visual Cortex)', 'mPFC\n(Medial PFC)'], fontsize=10)
ax.set_ylabel('Population Coupling', fontweight='bold', fontsize=11)
ax.set_title('Regional Comparison', fontweight='bold', fontsize=12)
ax.grid(alpha=0.3, axis='y')

# ===== Panel c: Cluster distribution by region (stacked bar) =====
ax = axes[2]
v1_clusters = df_reg[df_reg['region'] == 'V1']['cluster'].values
mpfc_clusters = df_reg[df_reg['region'] == 'mPFC']['cluster'].values

v1_c0 = np.sum(v1_clusters == 0) / len(v1_clusters)
v1_c1 = np.sum(v1_clusters == 1) / len(v1_clusters)
mpfc_c0 = np.sum(mpfc_clusters == 0) / len(mpfc_clusters)
mpfc_c1 = np.sum(mpfc_clusters == 1) / len(mpfc_clusters)

x_pos = np.array([0, 1.5])
width = 0.6

ax.bar(x_pos, [v1_c0, mpfc_c0], width, label='C0 (Low)',
       color=COLOR_C0, alpha=0.7, edgecolor='black')
ax.bar(x_pos, [v1_c1, mpfc_c1], width,
       bottom=[v1_c0, mpfc_c0], label='C1 (High)',
       color=COLOR_C1, alpha=0.7, edgecolor='black')

# percentage labels
ax.text(x_pos[0], v1_c0 / 2, f'{v1_c0 * 100:.1f}%',
        ha='center', va='center', fontweight='bold', fontsize=11)
ax.text(x_pos[0], v1_c0 + v1_c1 / 2, f'{v1_c1 * 100:.1f}%',
        ha='center', va='center', fontweight='bold', fontsize=11)
ax.text(x_pos[1], mpfc_c0 / 2, f'{mpfc_c0 * 100:.1f}%',
        ha='center', va='center', fontweight='bold', fontsize=11)
ax.text(x_pos[1], mpfc_c0 + mpfc_c1 / 2, f'{mpfc_c1 * 100:.1f}%',
        ha='center', va='center', fontweight='bold', fontsize=11)

contingency = pd.crosstab(df_reg['region'], df_reg['cluster'])
chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
ax.text(0.02, 0.98, f'\u03c7\u00b2 = {chi2:.2f}\np = {p_chi:.2e}',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xticks(x_pos)
ax.set_xticklabels(['V1', 'mPFC'], fontsize=11)
ax.set_ylabel('Proportion', fontweight='bold', fontsize=11)
ax.set_title('Cluster Distribution by Region', fontweight='bold', fontsize=12)
ax.set_ylim([0, 1])
ax.axhline(0.5, color='black', linestyle='--', alpha=0.5)
ax.legend(fontsize=10, loc='upper right')
ax.grid(alpha=0.3, axis='y')

# ===== Panel d: Trajectory proportions by region =====
ax = axes[3]

# Mean and SEM across animals for each region x trajectory
summary_rows = []
for region in ['V1', 'mPFC']:
    rdf = df_prop[df_prop['region'] == region]
    for traj in TRAJ_ORDER:
        tdf = rdf[rdf['trajectory'] == traj]
        mean_pct = tdf['percentage'].mean()
        sem_pct = tdf['percentage'].std() / np.sqrt(len(tdf)) if len(tdf) > 1 else 0
        total_count = tdf['count'].sum()
        summary_rows.append({
            'region': region, 'trajectory': traj,
            'mean_pct': mean_pct, 'sem': sem_pct,
            'count': total_count
        })

summary = pd.DataFrame(summary_rows)

x = np.arange(2)  # V1, mPFC
bar_width = 0.25

for i, traj in enumerate(TRAJ_ORDER):
    tdata = summary[summary['trajectory'] == traj]
    bars = ax.bar(x + (i - 1) * bar_width,
                  tdata['mean_pct'].values, bar_width,
                  yerr=tdata['sem'].values, capsize=4,
                  label=traj, color=TRAJ_COLORS[traj], alpha=0.8,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5})

ax.set_ylabel('Percentage of Neurons (%)', fontweight='bold')
ax.set_xlabel('Brain Region', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['V1', 'mPFC'], fontsize=11)
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_linewidth(0.75)

# Chi-square test between regions (on total counts)
# Build contingency from the per-trajectory totals
cont_traj = pd.DataFrame({
    'V1': [summary[(summary['region'] == 'V1') & (summary['trajectory'] == t)]['count'].values[0]
           for t in TRAJ_ORDER],
    'mPFC': [summary[(summary['region'] == 'mPFC') & (summary['trajectory'] == t)]['count'].values[0]
             for t in TRAJ_ORDER],
}, index=TRAJ_ORDER)
chi2_t, p_t, _, _ = stats.chi2_contingency(cont_traj.values)
ax.text(0.98, 0.02, f'\u03c7\u00b2 (between regions): p={p_t:.3f}',
        transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Within-region pairwise paired t-tests + significance bars
from itertools import combinations

sig_results = {}
for region in ['V1', 'mPFC']:
    rdf = df_prop[df_prop['region'] == region]
    sig_results[region] = {}
    for t1, t2 in combinations(TRAJ_ORDER, 2):
        d1 = rdf[rdf['trajectory'] == t1].set_index('animal')['percentage']
        d2 = rdf[rdf['trajectory'] == t2].set_index('animal')['percentage']
        common = d1.index.intersection(d2.index)
        if len(common) >= 3:
            _, p_val = stats.ttest_rel(d1.loc[common].values, d2.loc[common].values)
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        else:
            p_val, sig = 1.0, 'ns'
        sig_results[region][(t1, t2)] = sig

# Draw significance bars
traj_idx = {t: i for i, t in enumerate(TRAJ_ORDER)}
region_x_map = {'V1': 0, 'mPFC': 1}

max_h = {}
for region in ['V1', 'mPFC']:
    sdf = summary[summary['region'] == region]
    max_h[region] = (sdf['mean_pct'] + sdf['sem']).max()

y_step = max(max_h.values()) * 0.08
for region in ['V1', 'mPFC']:
    rx = region_x_map[region]
    y_base = max_h[region] + max(max_h.values()) * 0.05
    comps = [('Stable Soloist', 'Stable Chorister'),
             ('Stable Chorister', 'Chorister-to-Soloist'),
             ('Stable Soloist', 'Chorister-to-Soloist')]
    for ci, (t1, t2) in enumerate(comps):
        sig = sig_results[region].get((t1, t2), 'ns')
        if sig == 'ns':
            continue
        x1 = (traj_idx[t1] - 1) * bar_width + rx
        x2 = (traj_idx[t2] - 1) * bar_width + rx
        y = y_base + ci * y_step
        bh = y * 0.02
        ax.plot([x1, x1, x2, x2], [y, y + bh, y + bh, y], 'k-', linewidth=1)
        ax.text((x1 + x2) / 2, y + bh, sig,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylim(0, 80)

# ---------- save ----------
plt.tight_layout()
plt.savefig(script_dir / 'regional_clustering.png', dpi=300, bbox_inches='tight')
plt.savefig(script_dir / 'regional_clustering.svg', bbox_inches='tight')
plt.close()
print('Saved regional_clustering.png and regional_clustering.svg')
