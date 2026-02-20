#!/usr/bin/env python3
"""
Reproduce Figure 4f-g-h (Example Neuron Trajectories) from source CSV.

Panels:
  f - Stable Soloist trajectories       (consistently low pop coupling)
  g - Stable Chorister trajectories     (consistently high pop coupling)
  h - Chorister-to-Soloist trajectories (high-to-low transition)

Requirements: numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ---------- configuration ----------
CLUSTER_COLORS = {0: '#3498DB', 1: '#E74C3C'}
TRAJ_COLORS = {
    'Stable Soloist': '#3498DB',
    'Stable Chorister': '#E74C3C',
    'Chorister-to-Soloist': '#9DCA24',
}
TRAJ_ORDER = ['Stable Soloist', 'Stable Chorister', 'Chorister-to-Soloist']

# ---------- load data ----------
script_dir = Path(__file__).resolve().parent
XLSX = script_dir / 'Supplementary_data.xlsx'
df = pd.read_excel(XLSX, sheet_name='Example trajectories')

# global y-limits (consistent across panels)
y_min = df['pop_coupling'].min()
y_max = df['pop_coupling'].max()
y_pad = (y_max - y_min) * 0.1
global_ylim = (y_min - y_pad, y_max + y_pad)

# total unit counts per trajectory (for panel titles)
total_counts = df[['animal', 'unit_id', 'trajectory']].drop_duplicates() \
                 .groupby('trajectory').size().to_dict()

# ---------- plot ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, traj_type in enumerate(TRAJ_ORDER):
    ax = axes[idx]
    traj_shown = df[df['trajectory'] == traj_type]
    units = traj_shown[['animal', 'unit_id']].drop_duplicates()
    n_shown = len(units)
    n_total = total_counts.get(traj_type, 0)

    for _, uid in units.iterrows():
        unit_data = traj_shown[(traj_shown['animal'] == uid['animal']) &
                               (traj_shown['unit_id'] == uid['unit_id'])] \
                    .sort_values('age_days')

        ages = unit_data['age_days'].values
        vals = unit_data['pop_coupling'].values
        clus = unit_data['cluster'].values

        if traj_type == 'Chorister-to-Soloist':
            # connected line colored by cluster of the later point
            for i in range(len(ages) - 1):
                color = CLUSTER_COLORS[clus[i + 1]]
                ax.plot(ages[i:i + 2], vals[i:i + 2],
                        color=color, alpha=0.6, linewidth=1.5)
            # markers on top
            for c in [0, 1]:
                mask = clus == c
                if mask.any():
                    ax.scatter(ages[mask], vals[mask],
                               color=CLUSTER_COLORS[c], s=20, alpha=0.8, zorder=3)
        else:
            # Stable Soloist / Stable Chorister: line per cluster
            for c in [0, 1]:
                mask = clus == c
                if mask.any():
                    ax.plot(ages[mask], vals[mask],
                            marker='o', linestyle='-', alpha=0.6,
                            color=CLUSTER_COLORS[c], linewidth=1.5, markersize=4)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Postnatal Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Population Coupling', fontsize=12, fontweight='bold')
    ax.set_title(f'{traj_type}\n(n={n_total} units, {n_shown} shown)',
                 fontsize=14, fontweight='bold', color=TRAJ_COLORS[traj_type])
    ax.set_xlim(8, 46)
    ax.set_ylim(global_ylim)

    if idx == 0:
        legend_elements = [
            Patch(facecolor=CLUSTER_COLORS[0], label='Cluster 0 (Sparse)'),
            Patch(facecolor=CLUSTER_COLORS[1], label='Cluster 1 (Ensemble)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig(script_dir / 'example_trajectories.png', dpi=300, bbox_inches='tight')
plt.savefig(script_dir / 'example_trajectories.svg', bbox_inches='tight')
plt.close()
print('Saved example_trajectories.png and example_trajectories.svg')
