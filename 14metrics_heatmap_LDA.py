#!/usr/bin/env python3
"""
Export anonymized CSV data for ED10 (three panels):
  a) Trajectory heatmap (14 metrics, z-scored, smoothed)
  b) LD1 & LD2 means over age per trajectory
  c) LDA loadings for LD1 and LD2

Mirrors the exact computations in the original plotting scripts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

# ── paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path('/media/liulab/Ariel14T/neonatal_laptop')
CSV_DIR = BASE_DIR / 'pipeline' / 'csv_data_1005'
MERGED_DIR = BASE_DIR / 'cluster_popcoupling_gmm' / 'results' / 'merged_metrics'
TRAJ_FILE = BASE_DIR / 'cluster_popcoupling_gmm' / 'results' / 'trajectory_analysis' / 'all_units_trajectory_assignments.csv'
LDA_COORD_FILE = BASE_DIR / 'cluster_popcoupling_gmm' / 'results' / 'umap_updated' / '14metrics' / 'lda_coordinates_14metrics.csv'
LDA_LOADING_FILE = BASE_DIR / 'cluster_popcoupling_gmm' / 'results' / 'systematic_analysis_14metrics' / 'lda_loadings.csv'
OUTPUT_DIR = BASE_DIR / 'cluster_popcoupling_gmm' / 'results' / 'umap_updated' / '14metrics' / 'ED10'

# ── constants ────────────────────────────────────────────────────────────
ANIMAL_MAP = {
    'round5_rat2': 'animal1',
    'round5_rat5': 'animal2',
    'round6_rat3': 'animal3',
    'round6_rat4': 'animal4',
    'round6_rat5': 'animal5',
}

TRAJ_MAP = {
    'Specialist': 'Stable Soloist',
    'Generalist': 'Stable Chorister',
    'Refiner': 'Chorister-to-Soloist',
}

SELECTED_14_METRICS = [
    'width_ms', 'trough_to_peak_ms', 'repol_slope', 'p2p', 'asymmetry',
    'firing_rate', 'burst_index', 'isi_median', 'lv', 'fano',
    'mvl_gamma', 'mvl_beta', 'mvl_theta', 'acg_trough_depth'
]

METRIC_DISPLAY = {
    'width_ms': 'Spike Width', 'trough_to_peak_ms': 'Trough-to-Peak',
    'repol_slope': 'Repol. Slope', 'p2p': 'Peak-to-Peak', 'asymmetry': 'Asymmetry',
    'firing_rate': 'Firing Rate', 'burst_index': 'Burst Index', 'isi_median': 'Median ISI',
    'lv': 'LV', 'fano': 'Fano Factor',
    'mvl_gamma': 'Gamma MVL', 'mvl_beta': 'Beta MVL', 'mvl_theta': 'Theta MVL',
    'acg_trough_depth': 'ACG Depth'
}

METRIC_CATEGORIES = {
    'width_ms': 'Waveform', 'trough_to_peak_ms': 'Waveform', 'repol_slope': 'Waveform',
    'p2p': 'Waveform', 'asymmetry': 'Waveform',
    'firing_rate': 'Firing', 'burst_index': 'Firing', 'isi_median': 'Firing',
    'lv': 'Variability', 'fano': 'Variability',
    'mvl_gamma': 'Phase-Locking', 'mvl_beta': 'Phase-Locking', 'mvl_theta': 'Phase-Locking',
    'acg_trough_depth': 'Autocorrelation'
}

WAVEFORM_METRICS = ['width_ms', 'trough_to_peak_ms', 'repol_slope', 'p2p', 'asymmetry']
MERGED_METRICS_LIST = ['mvl_gamma', 'mvl_beta', 'mvl_theta']

TRAJECTORIES_ORIG = ['Specialist', 'Generalist', 'Refiner']
AGE_BINS = list(range(11, 45, 2))

TIMEPOINTS = [
    ('P12', 10, 14, 12), ('P17', 15, 19, 17), ('P21', 19, 23, 21), ('P26', 24, 28, 26),
    ('P31', 29, 33, 31), ('P35', 33, 37, 35), ('P40', 38, 42, 40), ('P44', 42, 46, 44)
]


# ── helpers ──────────────────────────────────────────────────────────────
def get_epoch(metric):
    return 'entire_session' if metric in WAVEFORM_METRICS else 'spontaneous_baseline_10min'


def load_metric_data(metric_name):
    if metric_name in MERGED_METRICS_LIST:
        merged_file = MERGED_DIR / f'{metric_name}.csv'
        metric_file = merged_file if merged_file.exists() else CSV_DIR / f'{metric_name}.csv'
    else:
        metric_file = CSV_DIR / f'{metric_name}.csv'
    if metric_name == 'firing_rate':
        metric_file = CSV_DIR / 'fr.csv'
    if not metric_file.exists():
        return pd.DataFrame()
    df = pd.read_csv(metric_file)
    value_col = None
    for col in ['value', 'fr_value', metric_name]:
        if col in df.columns:
            value_col = col
            break
    if value_col is None:
        for col in df.columns:
            if col not in ['animal', 'aligned_unit', 'age', 'session', 'phase_name', 'metric', 'unit', 'epoch']:
                value_col = col
                break
    if value_col is None:
        return pd.DataFrame()
    df['value'] = df[value_col]
    epoch = get_epoch(metric_name)
    if 'phase_name' in df.columns:
        df = df[df['phase_name'] == epoch].copy()
    return df[['animal', 'aligned_unit', 'age', 'value']]


# =========================================================================
# CSV 1: Heatmap data (z-scored, extrapolated, Gaussian-smoothed)
# =========================================================================
def export_heatmap():
    print("Exporting heatmap data...")
    df_traj = pd.read_csv(TRAJ_FILE)
    df_traj = df_traj[df_traj['trajectory'].isin(TRAJECTORIES_ORIG)].copy()

    n_metrics = len(SELECTED_14_METRICS)
    n_traj = len(TRAJECTORIES_ORIG)
    n_ages = len(AGE_BINS)

    heatmap_data = np.full((n_metrics, n_traj * n_ages), np.nan)

    for m_idx, metric in enumerate(SELECTED_14_METRICS):
        df_metric = load_metric_data(metric)
        if df_metric.empty:
            continue
        df_merged = df_metric.merge(df_traj[['animal', 'aligned_unit', 'trajectory']],
                                     on=['animal', 'aligned_unit'], how='inner')
        for t_idx, traj in enumerate(TRAJECTORIES_ORIG):
            traj_data = df_merged[df_merged['trajectory'] == traj]
            for a_idx, age_center in enumerate(AGE_BINS):
                age_data = traj_data[(traj_data['age'] >= age_center - 1) & (traj_data['age'] <= age_center + 1)]
                if len(age_data) > 0:
                    heatmap_data[m_idx, t_idx * n_ages + a_idx] = age_data['value'].mean()

    # Extrapolate
    for i in range(n_metrics):
        for t in range(n_traj):
            s, e = t * n_ages, (t + 1) * n_ages
            seg = heatmap_data[i, s:e]
            vi = np.where(~np.isnan(seg))[0]
            if len(vi) >= 2:
                vv = seg[vi]
                for j in range(n_ages):
                    if np.isnan(seg[j]):
                        seg[j] = np.interp(j, vi, vv)
                heatmap_data[i, s:e] = seg

    # Gaussian smoothing
    sigma = 1.5
    for i in range(n_metrics):
        for t in range(n_traj):
            s, e = t * n_ages, (t + 1) * n_ages
            seg = heatmap_data[i, s:e]
            if not np.all(np.isnan(seg)):
                vm = ~np.isnan(seg)
                if vm.sum() > 2:
                    filled = np.interp(np.arange(n_ages), np.where(vm)[0], seg[vm])
                    heatmap_data[i, s:e] = gaussian_filter1d(filled, sigma=sigma)

    # Z-score per metric
    heatmap_z = np.full_like(heatmap_data, np.nan)
    for i in range(n_metrics):
        row = heatmap_data[i, :]
        valid = row[~np.isnan(row)]
        if len(valid) > 1 and np.std(valid) > 0:
            heatmap_z[i, :] = (row - np.mean(valid)) / np.std(valid)

    # Build long-form CSV
    rows = []
    for m_idx, metric in enumerate(SELECTED_14_METRICS):
        for t_idx, traj in enumerate(TRAJECTORIES_ORIG):
            for a_idx, age_center in enumerate(AGE_BINS):
                col_idx = t_idx * n_ages + a_idx
                z_val = heatmap_z[m_idx, col_idx]
                if not np.isnan(z_val):
                    rows.append({
                        'metric': METRIC_DISPLAY[metric],
                        'metric_id': metric,
                        'category': METRIC_CATEGORIES[metric],
                        'trajectory': TRAJ_MAP[traj],
                        'age_bin_center': age_center,
                        'z_score': z_val
                    })

    df_out = pd.DataFrame(rows)
    path = OUTPUT_DIR / 'ed10a_heatmap.csv'
    df_out.to_csv(path, index=False)
    print(f"  {len(df_out)} rows -> {path}")


# =========================================================================
# CSV 2: LD1 & LD2 means ± SEM per trajectory per timepoint
# =========================================================================
def export_ld_means():
    print("Exporting LD means over age...")
    df = pd.read_csv(LDA_COORD_FILE)

    rows = []
    for traj in TRAJECTORIES_ORIG:
        traj_data = df[df['trajectory'] == traj]
        for tp_name, _, _, age_center in TIMEPOINTS:
            tp_data = traj_data[traj_data['timepoint'] == tp_name]
            if len(tp_data) > 0:
                rows.append({
                    'trajectory': TRAJ_MAP[traj],
                    'timepoint': tp_name,
                    'age': age_center,
                    'n': len(tp_data),
                    'LD1_mean': tp_data['LD1'].mean(),
                    'LD1_sem': tp_data['LD1'].std() / np.sqrt(len(tp_data)),
                    'LD2_mean': tp_data['LD2'].mean(),
                    'LD2_sem': tp_data['LD2'].std() / np.sqrt(len(tp_data)),
                })

    df_out = pd.DataFrame(rows)
    path = OUTPUT_DIR / 'ed10b_ld_means_over_age.csv'
    df_out.to_csv(path, index=False)
    print(f"  {len(df_out)} rows -> {path}")


# =========================================================================
# CSV 3: LDA loadings
# =========================================================================
def export_loadings():
    print("Exporting LDA loadings...")
    df = pd.read_csv(LDA_LOADING_FILE)
    df['metric_display'] = df['metric'].map(METRIC_DISPLAY)
    df['category'] = df['metric'].map(METRIC_CATEGORIES)
    df_out = df[['metric', 'metric_display', 'category', 'LD1', 'LD2']]
    path = OUTPUT_DIR / 'ed10c_lda_loadings.csv'
    df_out.to_csv(path, index=False)
    print(f"  {len(df_out)} rows -> {path}")


# =========================================================================
if __name__ == '__main__':
    export_heatmap()
    export_ld_means()
    export_loadings()
    print("\nDone!")
