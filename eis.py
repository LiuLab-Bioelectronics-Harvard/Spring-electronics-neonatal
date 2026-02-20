#!/usr/bin/env python3
"""
Reproduce Extended Data Figure 3 panels a–d from source CSVs.

Panels (1 row, 4 columns):
  a - Bode magnitude plot: Impedance (Ω) vs Frequency (Hz)
  b - Bode phase plot: Phase (°) vs Frequency (Hz)
  c - Impedance @ 1 kHz over incubation time (days)
  d - Impedance @ 1 kHz for Released / Implanted / Stretched conditions

Requirements: numpy, pandas, matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- load data ----------
script_dir = Path(__file__).resolve().parent
XLSX = script_dir / 'Supplementary_data.xlsx'

df_bode = pd.read_excel(XLSX, sheet_name='EIS processed data')
df_raw = pd.read_excel(XLSX, sheet_name='EIS raw data')
df_inc = pd.read_excel(XLSX, sheet_name='Impedance PBS incubation')
df_imp = pd.read_excel(XLSX, sheet_name='Impedance device conditions')

# ---------- create figure ----------
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

# ===== Panel a: Bode magnitude plot =====
ax = axes[0]
freq = df_bode['frequency_Hz'].values
z_mean = df_bode['Z_mean_Ohm'].values
z_std = df_bode['Z_std_Ohm'].values

ax.plot(freq, z_mean, 'o-', color='#d62728', markersize=3, linewidth=1.5)
ax.fill_between(freq, z_mean - z_std, z_mean + z_std, color='#d62728', alpha=0.2)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency (Hz)', fontweight='bold')
ax.set_ylabel('Impedance (Ω)', fontweight='bold')
ax.set_title('a', fontweight='bold', fontsize=14, loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ===== Panel b: Bode phase plot =====
ax = axes[1]
phase_mean = df_bode['phase_mean_deg'].values
phase_std = df_bode['phase_std_deg'].values

ax.plot(freq, phase_mean, 'o-', color='#1f77b4', markersize=3, linewidth=1.5)
ax.fill_between(freq, phase_mean - phase_std, phase_mean + phase_std,
                color='#1f77b4', alpha=0.2)
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)', fontweight='bold')
ax.set_ylabel('Phase (°)', fontweight='bold')
ax.set_title('b', fontweight='bold', fontsize=14, loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

def log_space_stats(vals):
    """Geometric mean with asymmetric ±1 SD error bars in log10 space."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v) & (v > 0)]
    logv = np.log10(v)
    m = logv.mean()
    s = logv.std(ddof=1) if len(v) > 1 else 0.0
    center = 10**m
    return center, center - 10**(m - s), 10**(m + s) - center

# ===== Panel c: Impedance @ 1 kHz over incubation time =====
ax = axes[2]
days = [0, 7, 14, 21, 28, 35]
inc_data = df_inc.iloc[:, 1:]  # skip first column (labels)

# gray circle data points (no jitter)
centers_c, ylo_c, yhi_c = [], [], []
for j, d in enumerate(days):
    vals = inc_data.iloc[:, j].values
    ax.scatter(np.full(len(vals), d), vals, color='gray', alpha=0.4, s=40,
               edgecolors='none', zorder=1)
    c, el, eu = log_space_stats(vals)
    centers_c.append(c); ylo_c.append(el); yhi_c.append(eu)

yerr_c = np.vstack([ylo_c, yhi_c])
ax.errorbar(days, centers_c, yerr=yerr_c, fmt='o-', color='#e74c3c',
            markersize=8, linewidth=2, capsize=4, capthick=1.5,
            markeredgecolor='#e74c3c', markerfacecolor='#e74c3c',
            ecolor='#e74c3c', zorder=3)
ax.set_yscale('log')
ax.set_ylim(1e3, 1e9)
ax.set_xlabel('Incubation time (days)')
ax.set_ylabel('Impedance\n@ 1 kHz (Ω)')
ax.set_xticks(days)
ax.set_title('c', fontweight='bold', fontsize=14, loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ===== Panel d: Impedance @ 1 kHz by condition =====
ax = axes[3]
conditions = ['Released', 'Implanted', 'Stretched']
xpos = np.arange(len(conditions))

centers_d, ylo_d, yhi_d = [], [], []
for i, cond in enumerate(conditions):
    vals = df_imp.loc[df_imp['condition'] == cond, 'impedance_ohm'].values.astype(float)
    # gray circle data points (no jitter)
    ax.scatter(np.full(len(vals), xpos[i]), vals, color='gray', alpha=0.4, s=40,
               edgecolors='none', zorder=1)
    c, el, eu = log_space_stats(vals)
    centers_d.append(c); ylo_d.append(el); yhi_d.append(eu)

yerr_d = np.vstack([ylo_d, yhi_d])
ax.errorbar(xpos, centers_d, yerr=yerr_d, fmt='o', color='#e74c3c',
            markersize=8, linewidth=2, capsize=5, capthick=2,
            markeredgecolor='#e74c3c', markerfacecolor='#e74c3c',
            ecolor='#e74c3c', zorder=3)
ax.set_yscale('log')
ax.set_ylim(1e3, 1e9)
ax.set_xticks(xpos)
ax.set_xticklabels(conditions, rotation=45, ha='right', fontstyle='italic')
ax.set_ylabel('Impedance\n@ 1 kHz (Ω)')
ax.set_title('d', fontweight='bold', fontsize=14, loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# ---------- save ----------
plt.tight_layout()
plt.savefig(script_dir / 'eis.png', dpi=300, bbox_inches='tight')
plt.savefig(script_dir / 'eis.svg', bbox_inches='tight')
plt.close()
print('Saved eis.png and eis.svg')
