#!/usr/bin/env python3
"""
Reproduce Figure 4i-j-k (Population Coupling Developmental Trajectories)
from source CSV.

Panels:
  i - V1 + mPFC combined
  j - V1 only
  k - mPFC only

Each panel shows mean +/- 95% CI of population coupling in 5-day age bins,
grouped by trajectory type.

Requirements: numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- configuration ----------
TRAJ_COLORS = {
    'Stable Soloist': '#3498DB',
    'Stable Chorister': '#E74C3C',
    'Chorister-to-Soloist': '#9DCA24',
}
TRAJ_ORDER = ['Stable Soloist', 'Stable Chorister', 'Chorister-to-Soloist']
AGE_BINS = np.arange(10, 50, 5)  # 10-15, 15-20, ..., 40-45

# ---------- load data ----------
script_dir = Path(__file__).resolve().parent
XLSX = script_dir / 'Supplementary_data.xlsx'
df = pd.read_excel(XLSX, sheet_name='Populationcoupling trajectories')

# ---------- plotting function ----------
def plot_panel(ax, data, title):
    """Plot one developmental trajectory panel."""
    for traj in TRAJ_ORDER:
        traj_data = data[data['trajectory'] == traj]
        if len(traj_data) < 3:
            continue

        bin_centers = []
        bin_means = []
        bin_sems = []
        for i in range(len(AGE_BINS) - 1):
            bdata = traj_data[(traj_data['age_days'] >= AGE_BINS[i]) &
                              (traj_data['age_days'] < AGE_BINS[i + 1])]
            if len(bdata) >= 2:
                bin_centers.append((AGE_BINS[i] + AGE_BINS[i + 1]) / 2)
                bin_means.append(bdata['pop_coupling'].mean())
                bin_sems.append(bdata['pop_coupling'].sem())

        if not bin_centers:
            continue

        centers = np.array(bin_centers)
        means = np.array(bin_means)
        sems = np.array(bin_sems)
        color = TRAJ_COLORS[traj]

        # 95% CI band
        ax.fill_between(centers, means - 1.96 * sems, means + 1.96 * sems,
                         color=color, alpha=0.2, zorder=2)
        # line
        ax.plot(centers, means, '-', color=color, linewidth=1.5, zorder=3)
        # dots
        ax.plot(centers, means, 'o', color=color, markersize=7, label=traj,
                markeredgecolor='white', markeredgewidth=1.5, zorder=4)

    # formatting
    ax.set_xlabel('Age (Postnatal Day)', fontweight='bold')
    ax.set_ylabel('Population Coupling', fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim(10, 45)

    # pad y-axis
    ylo, yhi = ax.get_ylim()
    yr = yhi - ylo
    ax.set_ylim(ylo - 0.05 * yr, yhi + 0.05 * yr)

    # developmental period shading
    ax.axvspan(10, 20, alpha=0.05, color='blue', zorder=0)
    ax.axvspan(35, 45, alpha=0.05, color='orange', zorder=0)
    ax.text(15, ax.get_ylim()[1] * 0.95, 'Early', ha='center',
            fontsize=8, style='italic', color='blue', alpha=0.7)
    ax.text(40, ax.get_ylim()[1] * 0.95, 'Late', ha='center',
            fontsize=8, style='italic', color='orange', alpha=0.7)


# ---------- create figure ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

plot_panel(axes[0], df, 'Population Coupling (V1 + mPFC)')
plot_panel(axes[1], df[df['region'] == 'V1'], 'Population Coupling (V1)')
plot_panel(axes[2], df[df['region'] == 'mPFC'], 'Population Coupling (mPFC)')

plt.tight_layout()
plt.savefig(script_dir / 'populationcoupling_trajectories.png', dpi=300, bbox_inches='tight')
plt.savefig(script_dir / 'populationcoupling_trajectories.svg', bbox_inches='tight')
plt.close()
print('Saved populationcoupling_trajectories.png and populationcoupling_trajectories.svg')
