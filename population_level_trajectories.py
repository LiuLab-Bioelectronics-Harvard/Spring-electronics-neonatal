"""
Extended Data Figure 5 (panels c–i) — Plotting script

Reproduces all panels from the source data:
  - Panels c–h: Population developmental trajectories for burst index,
    ACG trough depth, CV2, LV, Fano factor, and mean pairwise
    correlation coefficient.
  - Panel i: Pairwise correlation matrix heatmaps for P14, P16, P19,
    P23, P26, P30, P37, P44.

Requirements: Python 3, numpy, pandas, matplotlib
Usage:        python ed5_plot.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Paths (relative to the directory that contains this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(SCRIPT_DIR, 'Supplementary_data.xlsx')
OUT_DIR = SCRIPT_DIR

# ---------------------------------------------------------------------------
# Load summary data
# ---------------------------------------------------------------------------
df = pd.read_excel(XLSX, sheet_name='Population level trajectories')
days = df["day"].values

# ---------------------------------------------------------------------------
# Metrics to plot (panels c–h)
# ---------------------------------------------------------------------------
metrics = [
    ("burst_index",      "Burst index",                "c"),
    ("acg_trough_depth", "ACG trough depth",           "d"),
    ("cv2",              "CV2",                        "e"),
    ("lv",               "LV",                         "f"),
    ("fano",             "Fano factor",                "g"),
    ("mean_pair_corr",   "Mean pairwise\ncorrelation", "h"),
]

# ---------------------------------------------------------------------------
# Correlation matrix ages (panel i)
# ---------------------------------------------------------------------------
matrix_ages = [14, 16, 19, 23, 26, 30, 37, 44]

# ---------------------------------------------------------------------------
# Build combined figure
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 14))

# Top section: 2 rows x 3 cols for line plots (panels c–h, 6 panels)
gs_top = gridspec.GridSpec(2, 3, top=0.95, bottom=0.52, left=0.06,
                           right=0.97, hspace=0.45, wspace=0.35)

# Bottom section: 2 rows x 4 cols for correlation matrices (panel i)
gs_bot = gridspec.GridSpec(2, 4, top=0.44, bottom=0.03, left=0.06,
                           right=0.92, hspace=0.35, wspace=0.35)

# ---------------------------------------------------------------------------
# Panels c–h: metric line plots
# ---------------------------------------------------------------------------
for idx, (prefix, ylabel, panel_label) in enumerate(metrics):
    row, col = idx // 3, idx % 3
    ax = fig.add_subplot(gs_top[row, col])

    mean_vals = df[f"{prefix}_mean"].values
    ci95_vals = df[f"{prefix}_ci95"].values

    ax.errorbar(days, mean_vals, yerr=ci95_vals, fmt="o-", color="red",
                markersize=5, capsize=3, linewidth=1.5)
    ax.set_xlabel("Postnatal day")
    ax.set_ylabel(ylabel)
    ax.set_xticks(days)
    ax.set_xticklabels([f"P{d}" for d in days], rotation=45, ha="right",
                       fontsize=7)
    ax.text(-0.15, 1.08, panel_label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="top")

# ---------------------------------------------------------------------------
# Panel i: correlation-matrix heatmaps
# ---------------------------------------------------------------------------
# Load all matrices and determine shared colour range
all_vals = []
matrices = {}
for age in matrix_ages:
    mat = pd.read_excel(XLSX, sheet_name=f'Pairwise correlation matrix P{age}',
                        header=0).values.astype(float)
    matrices[age] = mat
    all_vals.append(mat[~np.isnan(mat)])

all_vals = np.concatenate(all_vals)
vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)
vlim = max(abs(vmin), abs(vmax))

for idx, age in enumerate(matrix_ages):
    row, col = idx // 4, idx % 4
    ax = fig.add_subplot(gs_bot[row, col])
    mat = matrices[age]
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                   aspect="equal", interpolation="none")
    ax.set_title(f"P{age}", fontsize=11)
    ax.set_xlabel("Unit number")
    ax.set_ylabel("Unit number")
    if idx == 0 and row == 0:
        ax.text(-0.15, 1.08, "i", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

# Colorbar for correlation matrices
cbar_ax = fig.add_axes([0.94, 0.03, 0.015, 0.41])
fig.colorbar(im, cax=cbar_ax, label="Normalized correlation coefficient")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
fig.savefig(os.path.join(OUT_DIR, "population_level_trajectories.svg"), dpi=150)
fig.savefig(os.path.join(OUT_DIR, "population_level_trajectories.png"), dpi=150)
plt.close(fig)

print("Done — figures saved to", OUT_DIR)
