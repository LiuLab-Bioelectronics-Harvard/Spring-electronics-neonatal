#!/usr/bin/env python3
"""
ED10 — Three-panel Extended Data figure:
  (a) Trajectory heatmap of 14 electrophysiological metrics (z-scored)
  (b) LD1 & LD2 mean scores over developmental age per trajectory
  (c) LDA loadings (horizontal bar charts for LD1 and LD2)

Input:  ed10a_heatmap.csv, ed10b_ld_means_over_age.csv, ed10c_lda_loadings.csv
Output: ed10.png, ed10.svg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 10, 'axes.linewidth': 0.75, 'figure.dpi': 300,
    'svg.fonttype': 'none'
})

TRAJ_COLORS = {
    'Stable Soloist': '#3498DB',
    'Stable Chorister': '#E74C3C',
    'Chorister-to-Soloist': '#2ECC71',
}
TRAJ_ORDER = ['Stable Soloist', 'Stable Chorister', 'Chorister-to-Soloist']

CAT_COLORS = {
    'Waveform': '#E74C3C', 'Firing': '#3498DB', 'Variability': '#2ECC71',
    'Phase-Locking': '#9B59B6', 'Autocorrelation': '#F39C12'
}

METRIC_ORDER = [
    'Spike Width', 'Trough-to-Peak', 'Repol. Slope', 'Peak-to-Peak',
    'Asymmetry', 'Firing Rate', 'Burst Index', 'Median ISI',
    'LV', 'Fano Factor', 'Gamma MVL', 'Beta MVL', 'Theta MVL', 'ACG Depth'
]

TIMEPOINTS = ['P12', 'P17', 'P21', 'P26', 'P31', 'P35', 'P40', 'P44']

# ── load data ────────────────────────────────────────────────────────────
XLSX = HERE / 'Supplementary_data.xlsx'
heatmap_df = pd.read_excel(XLSX, sheet_name='14 metrics heatmap')
ld_means_df = pd.read_excel(XLSX, sheet_name='Trajectory LDA means over age')
loadings_df = pd.read_excel(XLSX, sheet_name='Trajectory LDA loadings')

# =========================================================================
# Panel (a): Heatmap
# =========================================================================

age_bins = sorted(heatmap_df['age_bin_center'].unique())
n_metrics = len(METRIC_ORDER)
n_traj = len(TRAJ_ORDER)
n_ages = len(age_bins)

# Build 2-D array: rows = metrics, columns = [traj0_age0 .. traj0_ageN | traj1_age0 .. ]
heatmap_z = np.full((n_metrics, n_traj * n_ages), np.nan)
for m_idx, metric in enumerate(METRIC_ORDER):
    for t_idx, traj in enumerate(TRAJ_ORDER):
        sub = heatmap_df[(heatmap_df['metric'] == metric) & (heatmap_df['trajectory'] == traj)]
        for _, row in sub.iterrows():
            a_idx = age_bins.index(row['age_bin_center'])
            heatmap_z[m_idx, t_idx * n_ages + a_idx] = row['z_score']

# =========================================================================
# Panel (c): LDA loadings — pre-sort
# =========================================================================

ld1_sorted = loadings_df.sort_values(by='LD1', key=abs, ascending=True).reset_index(drop=True)
ld2_sorted = loadings_df.sort_values(by='LD2', key=abs, ascending=True).reset_index(drop=True)

# =========================================================================
# Figure layout: top row = heatmap (wide), bottom row = LD means + loadings
# =========================================================================

fig = plt.figure(figsize=(26, 16))

# --- (a) Heatmap — full width, top half ---
ax_heat = fig.add_axes([0.06, 0.55, 0.82, 0.38])

cmap = LinearSegmentedColormap.from_list('custom_div',
    ['#2166AC', '#4393C3', '#92C5DE', '#D1E5F0', '#F7F7F7',
     '#FDDBC7', '#F4A582', '#D6604D', '#B2182B'])

im = ax_heat.imshow(heatmap_z, aspect='auto', cmap=cmap, vmin=-2.5, vmax=2.5)

# Y-axis: metric names coloured by category
ax_heat.set_yticks(range(n_metrics))
ax_heat.set_yticklabels(METRIC_ORDER, fontsize=11)
# Look up category from the dataframe
metric_to_cat = dict(zip(heatmap_df['metric'], heatmap_df['category']))
for i, m in enumerate(METRIC_ORDER):
    cat = metric_to_cat.get(m, 'Waveform')
    ax_heat.get_yticklabels()[i].set_color(CAT_COLORS[cat])
    ax_heat.get_yticklabels()[i].set_fontweight('bold')

ax_heat.set_xticks([])

# Trajectory labels below
for t_idx, traj in enumerate(TRAJ_ORDER):
    center = t_idx * n_ages + n_ages / 2 - 0.5
    ax_heat.text(center, n_metrics + 0.8, traj, ha='center', va='top',
                 fontsize=13, fontweight='bold', color=TRAJ_COLORS[traj])

# Age tick marks
for t_idx in range(n_traj):
    for a_idx, age in enumerate(age_bins):
        if a_idx % 4 == 0:
            x = t_idx * n_ages + a_idx
            ax_heat.text(x, n_metrics + 0.2, f'P{age}', ha='center', va='top',
                         fontsize=8, rotation=45)

# Trajectory separators
for t_idx in range(1, n_traj):
    ax_heat.axvline(x=t_idx * n_ages - 0.5, color='white', linewidth=3)

# Category legend
cat_patches = [mpatches.Patch(color=CAT_COLORS[c], label=c)
               for c in ['Waveform', 'Firing', 'Variability', 'Phase-Locking', 'Autocorrelation']]
ax_heat.legend(handles=cat_patches, loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10,
               title='Metric Categories')

cbar = plt.colorbar(im, ax=ax_heat, shrink=0.6, pad=0.12)
cbar.set_label('Z-Score', fontsize=12)

ax_heat.set_title('a   Trajectory Heatmap (14 Selected Metrics)\n'
                   'Extrapolated & Gaussian Smoothed',
                   fontsize=14, fontweight='bold', pad=15, loc='left')

# --- (b) LD1 & LD2 means over age — bottom-left ---
ax_ld1 = fig.add_axes([0.06, 0.08, 0.28, 0.36])
ax_ld2 = fig.add_axes([0.38, 0.08, 0.28, 0.36])

for ax, ld_col, ld_sem, ld_label, var_pct in [
    (ax_ld1, 'LD1_mean', 'LD1_sem', 'LD1', 68),
    (ax_ld2, 'LD2_mean', 'LD2_sem', 'LD2', 32),
]:
    for traj in TRAJ_ORDER:
        sub = ld_means_df[ld_means_df['trajectory'] == traj].sort_values('age')
        ax.plot(sub['age'], sub[ld_col], 'o-', color=TRAJ_COLORS[traj], linewidth=3,
                markersize=10, label=traj, markeredgecolor='white', markeredgewidth=1.5)
        ax.fill_between(sub['age'].values.astype(float),
                        (sub[ld_col] - sub[ld_sem]).values.astype(float),
                        (sub[ld_col] + sub[ld_sem]).values.astype(float),
                        color=TRAJ_COLORS[traj], alpha=0.2)
    ax.set_xlabel('Age (Postnatal Days)', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'{ld_col.replace("_mean","")} Score', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([tp[1] for tp in zip(TIMEPOINTS, [12,17,21,26,31,35,40,44])])
    ax.set_xticklabels(TIMEPOINTS)

ax_ld1.set_title('b   LD1 (68% of between-group variance)',
                  fontsize=12, fontweight='bold', loc='left')
ax_ld2.set_title('    LD2 (32% of between-group variance)',
                  fontsize=12, fontweight='bold', loc='left')

# --- (c) LDA loadings — bottom-right, two sub-panels ---
ax_l1 = fig.add_axes([0.72, 0.08, 0.13, 0.36])
ax_l2 = fig.add_axes([0.87, 0.08, 0.13, 0.36])

for ax, sorted_df, ld_col, var_pct in [
    (ax_l1, ld1_sorted, 'LD1', 68),
    (ax_l2, ld2_sorted, 'LD2', 32),
]:
    colors = [CAT_COLORS[row['category']] for _, row in sorted_df.iterrows()]
    labels = sorted_df['metric_display'].values
    values = sorted_df[ld_col].values

    bars = ax.barh(range(len(values)), values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(f'{ld_col} Loading', fontsize=10)
    ax.axvline(x=0, color='black', linewidth=0.8)

    for i, (v, bar) in enumerate(zip(values, bars)):
        if abs(v) > 0.05:
            ha = 'left' if v > 0 else 'right'
            off = 0.03 if v > 0 else -0.03
            ax.text(v + off, i, f'{v:.2f}', va='center', ha=ha, fontsize=7, color='#333')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax_l1.set_title('c   LD1 (68%)', fontsize=11, fontweight='bold', loc='left')
ax_l2.set_title('    LD2 (32%)', fontsize=11, fontweight='bold', loc='left')

# Shared category legend for loadings
cat_patches2 = [mpatches.Patch(color=CAT_COLORS[c], label=c)
                for c in ['Waveform', 'Firing', 'Variability', 'Phase-Locking', 'Autocorrelation']]
ax_l2.legend(handles=cat_patches2, fontsize=7, loc='lower right')

# ── save ─────────────────────────────────────────────────────────────────
for ext in ['png', 'svg']:
    out = HERE / f'metrics_heatmap_lda.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')

plt.close()
