#!/usr/bin/env python3
"""
Fig 4L — LDA Space Stacked by Age (14 Metrics)

Reproduces the 3D scatter plot where:
  X = LD1 (68% of between-class variance)
  Y = LD2 (32%)
  Z = Age (postnatal day, stacked)

Individual unit-timepoints are shown as small transparent dots.
Trajectory centroids per age are connected by lines; sphere size
decreases from young (large/transparent) to old (small/opaque).

Input:  fig4l_data.csv  (in the same directory)
Output: fig4l.png, fig4l.svg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
XLSX = HERE / 'Supplementary_data.xlsx'

# ── configuration ────────────────────────────────────────────────────────
TRAJECTORIES = ['Stable Soloist', 'Stable Chorister', 'Chorister-to-Soloist']
COLORS = {
    'Stable Soloist':       '#3498DB',
    'Stable Chorister':     '#E74C3C',
    'Chorister-to-Soloist': '#2ECC71',
}
TIMEPOINTS = ['P12', 'P17', 'P21', 'P26', 'P31', 'P35', 'P40', 'P44']
AGE_TO_Z = {tp: i for i, tp in enumerate(TIMEPOINTS)}

# ── load data ────────────────────────────────────────────────────────────
df = pd.read_excel(XLSX, sheet_name='Population coupling LDA visual')
df['z'] = df['timepoint'].map(AGE_TO_Z)

# ── plot ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

for traj in TRAJECTORIES:
    traj_data = df[df['trajectory'] == traj]
    color = COLORS[traj]

    # Individual unit-timepoint dots
    ax.scatter(traj_data['LD1'], traj_data['LD2'], traj_data['z'],
               c=color, s=40, alpha=0.3)

    # Centroids per age
    centroids = (traj_data.groupby('timepoint')
                 .agg(LD1=('LD1', 'mean'), LD2=('LD2', 'mean'), z=('z', 'first'))
                 .reset_index()
                 .sort_values('z'))

    # Connecting line
    ax.plot(centroids['LD1'], centroids['LD2'], centroids['z'],
            color=color, linewidth=2.5, alpha=0.6)

    # Centroid spheres (young = large/transparent, old = small/opaque)
    for i, (_, row) in enumerate(centroids.iterrows()):
        size = 400 - i * 45
        alpha = 0.45 + i * 0.07
        ax.scatter([row['LD1']], [row['LD2']], [row['z']],
                   c=color, s=size, alpha=alpha,
                   edgecolors='black', linewidths=1)

# ── axes & labels ────────────────────────────────────────────────────────
ax.set_xlabel('LD1 (68%)', fontsize=12, labelpad=10)
ax.set_ylabel('LD2 (32%)', fontsize=12, labelpad=10)
ax.set_zlabel('Age', fontsize=12, labelpad=10)
ax.set_zticks(range(len(TIMEPOINTS)))
ax.set_zticklabels(TIMEPOINTS)

ax.view_init(elev=20, azim=72)

# Remove background panes and gridlines
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('none')
ax.yaxis.pane.set_edgecolor('none')
ax.zaxis.pane.set_edgecolor('none')
ax.grid(False)

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLORS[t], markersize=10, label=t)
    for t in TRAJECTORIES
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11)

ax.set_title('LDA Space Stacked by Age (14 Metrics)\n'
             'Large/Transparent = Young, Small/Opaque = Old',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()

for ext in ['png', 'svg']:
    out = HERE / f'population_coupling_lda_visual.{ext}'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')

plt.close()
