#!/usr/bin/env python3
"""
ED5ab — Burst Index developmental trajectories for two example
Chorister-to-Soloist units.

Each unit is shown as a single panel with a linear regression line.

Output: individual_neuron_trajectories.png, individual_neuron_trajectories.svg
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

HERE = Path(__file__).resolve().parent
XLSX = HERE / 'Supplementary_data.xlsx'

# ── style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.linewidth': 0.75, 'figure.dpi': 300,
    'svg.fonttype': 'none'
})

LINE_COLOR = '#2E86AB'
REG_COLOR = '#E74C3C'

# ── load data ────────────────────────────────────────────────────────────
df = pd.read_excel(XLSX, sheet_name='Individual neuron trajectories')
units = sorted(df['aligned_unit'].unique(), reverse=True)  # ['b', 'a']

# ── figure: 2 rows (units) × 1 col (burst index) ────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(7, 8))

for row_idx, unit_id in enumerate(units):
    ax = axes[row_idx]
    unit_df = df[df['aligned_unit'] == unit_id].sort_values('age')
    traj = 'Chorister-to-Soloist'
    n_sessions = len(unit_df)
    age_range = f"P{int(unit_df['age'].min())}-P{int(unit_df['age'].max())}"

    ages = unit_df['age'].values
    vals = unit_df['burst_index'].values

    # Data line
    ax.plot(ages, vals, 'o-', color=LINE_COLOR, linewidth=2,
            markersize=8, markeredgecolor='white', markeredgewidth=1)

    # Linear regression line
    valid = ~np.isnan(vals)
    if valid.sum() >= 3:
        slope, intercept, _, _, _ = stats.linregress(ages[valid], vals[valid])
        x_fit = np.linspace(ages.min(), ages.max(), 100)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, '--', color=REG_COLOR, linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Age (days)', fontsize=11)
    ax.set_ylabel('Burst Index', fontsize=11)
    ax.set_title(f'Unit {unit_id} — Burst Index\n'
                 f'({traj}, {n_sessions} sessions, {age_range})',
                 fontsize=12, fontweight='bold')

plt.tight_layout()

for ext in ['png', 'svg']:
    out = HERE / f'individual_neuron_trajectories.{ext}'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved: {out}')

plt.close()
