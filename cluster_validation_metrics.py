#!/usr/bin/env python3
"""
Reproduce Extended Data Figure 6 (GMM Model Validation) from source CSVs.

Panels (1 row, 4 columns):
  a - Optimal K selection (delta from K=2)
  b - Silhouette scores per animal
  c - Predictive accuracy (CV AUROC) per animal
  d - Component separation per animal

Requirements: numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- load data ----------
script_dir = Path(__file__).resolve().parent
XLSX = script_dir / 'Supplementary_data.xlsx'
df_met = pd.read_excel(XLSX, sheet_name='Cluster validation metrics')
df_k = pd.read_excel(XLSX, sheet_name='Optimal K selection')

animals = df_met['animal'].tolist()
n_animals = len(animals)
x_pos = np.arange(n_animals)

# ---------- create figure ----------
fig, axes = plt.subplots(1, 4, figsize=(22, 5))

# ===== Panel a: Optimal K selection (delta from K=2) =====
ax = axes[0]
k_vals = df_k['k'].values
bics_k = df_k['bic'].values
aics_k = df_k['aic'].values
x_k = np.arange(len(k_vals))

# Delta from K=2 (first entry)
bic_delta = bics_k - bics_k[0]
aic_delta = aics_k - aics_k[0]

ax.plot(x_k, bic_delta, 'o-', linewidth=2, markersize=8, label='BIC',
        color='#e74c3c')
ax.plot(x_k, aic_delta, 's-', linewidth=2, markersize=8, label='AIC',
        color='#3498db')

# Mark minima
min_bic = np.argmin(bic_delta)
min_aic = np.argmin(aic_delta)
ax.scatter([min_bic], [bic_delta[min_bic]], s=200, marker='*',
           color='#e74c3c', edgecolor='black', linewidth=1.5, zorder=10)
ax.scatter([min_aic], [aic_delta[min_aic]], s=200, marker='*',
           color='#3498db', edgecolor='black', linewidth=1.5, zorder=10)

ax.axhline(0, color='black', linestyle='--', alpha=0.3)
ax.set_xticks(x_k)
ax.set_xticklabels(k_vals)
ax.set_xlabel('Number of Clusters (K)', fontweight='bold', fontsize=11)
ax.set_ylabel('\u0394 Information Criterion\n(relative to K=2)', fontweight='bold',
              fontsize=11)
ax.set_title('Optimal K Selection', fontweight='bold', fontsize=11)
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ===== Panel b: Silhouette scores =====
ax = axes[1]
sils = df_met['silhouette'].values
ax.bar(x_pos, sils, color='gray', alpha=0.7, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(animals, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Silhouette Score', fontweight='bold')
ax.set_title('Clustering Quality\n(Silhouette)', fontweight='bold', fontsize=11)
ax.set_ylim([0, 0.7])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ===== Panel c: CV AUROC =====
ax = axes[2]
aurocs = df_met['cv_auroc'].values
ax.bar(x_pos, aurocs, color='gray', alpha=0.7, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(animals, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('CV AUROC', fontweight='bold')
ax.set_title('Predictive Accuracy\n(Cross-Validation)', fontweight='bold', fontsize=11)
ax.set_ylim([0, 1])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ===== Panel d: Component separation =====
ax = axes[3]
seps = df_met['separation'].values
ax.bar(x_pos, seps, color='gray', alpha=0.7, edgecolor='black')
ax.set_xticks(x_pos)
ax.set_xticklabels(animals, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Separation (\u03c3)', fontweight='bold')
ax.set_title('Component Separation\n(Standard Deviations)', fontweight='bold',
             fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------- save ----------
plt.tight_layout()
plt.savefig(script_dir / 'cluster_validation_metrics.png', dpi=300, bbox_inches='tight')
plt.savefig(script_dir / 'cluster_validation_metrics.svg', bbox_inches='tight')
plt.close()
print('Saved cluster_validation_metrics.png and cluster_validation_metrics.svg')
