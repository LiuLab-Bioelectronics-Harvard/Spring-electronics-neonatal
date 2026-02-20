#!/usr/bin/env python3
"""
Reproduce Extended Data Figure 7 (GMM Normality & Residual Analysis)
from source CSVs.

Panels (4 rows):
  Row 1 (a): Per-animal Q-Q plots, cluster 1 (5 animals)
  Row 2 (a): Per-animal Q-Q plots, cluster 0 (5 animals)
  Row 3 (b): Per-animal histograms + Gaussian fit, cluster 1 (5 animals)
  Row 4 (b): Per-animal histograms + Gaussian fit, cluster 0 (5 animals)
  Row 5 (c-f): Pooled residual analysis (4 panels)

Requirements: numpy, pandas, matplotlib, scipy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, probplot
from pathlib import Path

# ---------- configuration ----------
COLOR_C0 = '#3498DB'  # blue
COLOR_C1 = '#E74C3C'  # red

# ---------- load data ----------
script_dir = Path(__file__).resolve().parent
XLSX = script_dir / 'Supplementary_data.xlsx'
df_pa = pd.read_excel(XLSX, sheet_name='Per-animal cluster normality')
df_res = pd.read_excel(XLSX, sheet_name='Residual analysis')

animals = sorted(df_pa['animal'].unique())

# ---------- create figure ----------
fig = plt.figure(figsize=(25, 22))
gs = fig.add_gridspec(5, 5, hspace=0.45, wspace=0.35)

# ===================================================================
# Rows 1-2: Per-animal Q-Q plots  (row 0 = C1, row 1 = C0)
# ===================================================================
for row_idx, cluster in enumerate([1, 0]):
    color = [COLOR_C0, COLOR_C1][cluster]
    for col_idx, animal in enumerate(animals):
        ax = fig.add_subplot(gs[row_idx, col_idx])

        vals = df_pa[(df_pa['animal'] == animal) &
                     (df_pa['cluster'] == cluster)]['pop_coupling_zscore'].values

        if len(vals) < 5:
            ax.text(0.5, 0.5, 'Insufficient\ndata', ha='center',
                    va='center', transform=ax.transAxes)
            ax.set_title(f'{animal} C{cluster}', fontsize=9)
            continue

        (osm, osr), (slope, intercept, r) = probplot(vals, dist='norm',
                                                      plot=None)
        ax.scatter(osm, osr, alpha=0.6, s=15, color=color)

        x_line = np.array([osm.min(), osm.max()])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', linewidth=1.5)

        ax.set_xlabel('Theoretical Q', fontsize=8)
        ax.set_ylabel('Sample Q', fontsize=8)
        ax.set_title(f'{animal} C{cluster}  (n={len(vals)}, '
                     f'R\u00b2={r**2:.3f})',
                     fontsize=9, fontweight='bold')
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
        ax.tick_params(labelsize=7)

# ===================================================================
# Rows 3-4: Per-animal histograms + Gaussian fit  (row 2 = C1, row 3 = C0)
# ===================================================================
for row_idx, cluster in enumerate([1, 0], start=2):
    color = [COLOR_C0, COLOR_C1][cluster]
    for col_idx, animal in enumerate(animals):
        ax = fig.add_subplot(gs[row_idx, col_idx])

        vals = df_pa[(df_pa['animal'] == animal) &
                     (df_pa['cluster'] == cluster)]['pop_coupling_zscore'].values

        if len(vals) < 5:
            ax.text(0.5, 0.5, 'Insufficient\ndata', ha='center',
                    va='center', transform=ax.transAxes)
            ax.set_title(f'{animal} C{cluster}', fontsize=9)
            ax.set_ylim(0, 1)
            continue

        bins = np.linspace(vals.min(), vals.max(), 21)
        bin_width = bins[1] - bins[0]
        bin_centers = (bins[:-1] + bins[1:]) / 2
        counts, _ = np.histogram(vals, bins=bins, density=True)
        max_density = counts.max()
        counts_norm = counts / max_density

        ax.bar(bin_centers, counts_norm, width=bin_width, alpha=0.6,
               color=color, edgecolor='black')

        mu, std = vals.mean(), vals.std()
        x = np.linspace(vals.min(), vals.max(), 200)
        gaussian_pdf = norm.pdf(x, mu, std)
        gaussian_norm = gaussian_pdf / max_density
        ax.plot(x, gaussian_norm, 'r-', linewidth=2)

        ax.set_xlabel('z-score', fontsize=8)
        ax.set_ylabel('Norm. Density', fontsize=8)
        ax.set_title(f'{animal} C{cluster}  (n={len(vals)})',
                     fontsize=9, fontweight='bold')
        ax.grid(False)
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
        ax.tick_params(labelsize=7)

# ===================================================================
# Row 5: Pooled residual analysis (4 panels)
# ===================================================================
features = df_res['pop_coupling_zscore'].values
clusters = df_res['cluster'].values.astype(int)
expected = df_res['expected_value'].values
residuals = df_res['residual'].values

# Panel c: Residual histogram
ax = fig.add_subplot(gs[4, 0])
ax.hist(residuals, bins=50, density=True, alpha=0.6, color='steelblue',
        edgecolor='black')
x = np.linspace(residuals.min(), residuals.max(), 200)
ax.plot(x, norm.pdf(x, residuals.mean(), residuals.std()), 'r-',
        linewidth=2)
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Residual (Observed \u2212 Expected)', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Residual Distribution', fontweight='bold')
ax.grid(False)
for spine in ax.spines.values():
    spine.set_linewidth(0.75)

# Panel d: Residuals by cluster
ax = fig.add_subplot(gs[4, 1])
bins_res = np.linspace(residuals.min(), residuals.max(), 31)
for c in [0, 1]:
    mask = clusters == c
    ax.hist(residuals[mask], bins=bins_res, alpha=0.6, density=True,
            label=f'C{c}', color=[COLOR_C0, COLOR_C1][c])
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Residual', fontweight='bold')
ax.set_ylabel('Density', fontweight='bold')
ax.set_title('Residuals by Cluster', fontweight='bold')
ax.legend()
ax.grid(False)
for spine in ax.spines.values():
    spine.set_linewidth(0.75)

# Panel e: Residual Q-Q plot
ax = fig.add_subplot(gs[4, 2])
(osm, osr), (slope, intercept, r) = probplot(residuals, dist='norm',
                                              plot=None)
sorted_idx = np.argsort(residuals)
sorted_clusters = clusters[sorted_idx]
colors = np.where(sorted_clusters == 0, COLOR_C0, COLOR_C1)
ax.scatter(osm, osr, alpha=0.5, s=20, c=colors)
x_line = np.array([osm.min(), osm.max()])
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'R\u00b2={r**2:.4f}')
ax.set_xlabel('Theoretical Quantiles', fontweight='bold')
ax.set_ylabel('Sample Quantiles', fontweight='bold')
ax.set_title('Residual Q-Q Plot', fontweight='bold')
ax.legend()
ax.grid(False)
for spine in ax.spines.values():
    spine.set_linewidth(0.75)

# Panel f: Residuals vs fitted
ax = fig.add_subplot(gs[4, 3])
point_colors = np.where(clusters == 0, COLOR_C0, COLOR_C1)
ax.scatter(expected, residuals, alpha=0.4, s=20, c=point_colors)
ax.axhline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Expected Value (Cluster Mean)', fontweight='bold')
ax.set_ylabel('Residual', fontweight='bold')
ax.set_title('Residuals vs Fitted Values', fontweight='bold')
ax.grid(False)
for spine in ax.spines.values():
    spine.set_linewidth(0.75)

# ---------- save ----------
plt.savefig(script_dir / 'per_animal_cluster_normality.png', dpi=300, bbox_inches='tight')
plt.savefig(script_dir / 'per_animal_cluster_normality.svg', bbox_inches='tight')
plt.close()
print('Saved per_animal_cluster_normality.png and per_animal_cluster_normality.svg')
