#!/usr/bin/env python3
"""
Reproduce the GMM Fit (column 1) and Age vs Pop Coupling (column 4) panels
from the per-animal GMM clustering figure using only the provided CSV files.

Requirements: numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

# ---------- load source data ----------
script_dir = Path(__file__).resolve().parent
XLSX = script_dir / 'Supplementary_data.xlsx'
gmm_fit = pd.read_excel(XLSX, sheet_name='GMM fitting')
gmm_params = pd.read_excel(XLSX, sheet_name='GMM fitting parameters')
age_coupling = pd.read_excel(XLSX, sheet_name='Population coupling with age')

animals = gmm_fit['animal'].unique()
n_animals = len(animals)

# ---------- figure setup ----------
fig, axes = plt.subplots(n_animals, 2, figsize=(10, 4 * n_animals))

# Consistent axis limits for the age vs pop coupling column
global_age_min = age_coupling['age_days'].min()
global_age_max = age_coupling['age_days'].max()
global_coup_min = age_coupling['pop_coupling'].min()
global_coup_max = age_coupling['pop_coupling'].max()
age_pad = (global_age_max - global_age_min) * 0.05
coup_pad = (global_coup_max - global_coup_min) * 0.05

COLORS = {0: '#3498DB', 1: '#E74C3C'}

for i, animal in enumerate(animals):
    # ---- data subsets ----
    fit_sub = gmm_fit[gmm_fit['animal'] == animal]
    params_sub = gmm_params[gmm_params['animal'] == animal].sort_values('component')
    age_sub = age_coupling[age_coupling['animal'] == animal]

    zscores = fit_sub['pop_coupling_zscore'].values
    clusters = fit_sub['cluster'].values

    means = params_sub['mean_zscore'].values
    stds = params_sub['std_zscore'].values
    weights = params_sub['weight'].values

    # ======== Column 1: GMM Fit ========
    ax1 = axes[i, 0]

    bins = np.linspace(zscores.min(), zscores.max(), 31)
    bin_w = bins[1] - bins[0]
    bin_c = (bins[:-1] + bins[1:]) / 2

    counts0, _ = np.histogram(zscores[clusters == 0], bins=bins, density=True)
    counts1, _ = np.histogram(zscores[clusters == 1], bins=bins, density=True)
    max_density = max(counts0.max(), counts1.max())

    ax1.bar(bin_c, counts0 / max_density, width=bin_w, alpha=0.5,
            color=COLORS[0], label=f'C0 (n={int((clusters == 0).sum())})')
    ax1.bar(bin_c, counts1 / max_density, width=bin_w, alpha=0.5,
            color=COLORS[1], label=f'C1 (n={int((clusters == 1).sum())})')

    x = np.linspace(zscores.min(), zscores.max(), 1000)
    gmm_pdf = weights[0] * norm.pdf(x, means[0], stds[0]) + \
              weights[1] * norm.pdf(x, means[1], stds[1])
    ax1.plot(x, gmm_pdf / max_density, 'r-', lw=2, alpha=0.8, label='GMM fit')
    ax1.plot(x, weights[0] * norm.pdf(x, means[0], stds[0]) / max_density,
             '--', color=COLORS[0], lw=1.5, alpha=0.6, label='C0 Gaussian')
    ax1.plot(x, weights[1] * norm.pdf(x, means[1], stds[1]) / max_density,
             '--', color=COLORS[1], lw=1.5, alpha=0.6, label='C1 Gaussian')

    ax1.set_ylim(0, 1)
    ax1.set_xlabel('Pop Coupling (z-score)')
    ax1.set_ylabel('Normalized Density')
    ax1.set_title(f'{animal} — GMM Fit')
    ax1.legend(fontsize=7)
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.75)

    # ======== Column 4: Age vs Pop Coupling ========
    ax4 = axes[i, 1]
    for c in [0, 1]:
        mask = age_sub['cluster'] == c
        ax4.scatter(age_sub.loc[mask, 'age_days'],
                    age_sub.loc[mask, 'pop_coupling'],
                    s=30, alpha=0.6, c=COLORS[c], label=f'C{c}')

    ax4.set_xlim(global_age_min - age_pad, global_age_max + age_pad)
    ax4.set_ylim(global_coup_min - coup_pad, global_coup_max + coup_pad)
    ax4.set_xlabel('Age (days)')
    ax4.set_ylabel('Pop Coupling')
    ax4.set_title(f'{animal} — Age vs Pop Coupling')
    ax4.legend(fontsize=7)
    ax4.grid(False)
    for spine in ax4.spines.values():
        spine.set_linewidth(0.75)

plt.suptitle('GMM k=2 Clustering (Per Animal)', fontsize=14, fontweight='bold', y=1.0)
plt.tight_layout()
plt.savefig(script_dir / 'gmm_fitting.png', dpi=300, bbox_inches='tight')
plt.savefig(script_dir / 'gmm_fitting.svg', bbox_inches='tight')
plt.close()
print('Saved gmm_fitting.png and gmm_fitting.svg')
