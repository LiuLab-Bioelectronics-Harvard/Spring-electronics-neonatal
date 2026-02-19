# Supplementary Data and Code

Source data, plotting scripts, and analysis code accompanying the manuscript on **Growth-adaptive spring electronics for long-term, same-neuron mapping in the developing rat brain**.

This repository provides code for each of the supplementary and main figures, along with a pipeline for automated neuron tracking across chronic recording sessions.

---

## Overview

We tracked neural populations in neonatal rats (P10-P45) across chronic recording sessions to characterize how individual neurons transition between population coupling states during early postnatal development. Neurons were classified into three developmental trajectory types using Gaussian Mixture Model (GMM) clustering:

- **Stable Soloist** - consistently low population coupling
- **Stable Chorister** - consistently high population coupling
- **Chorister-to-Soloist** - transitions from high to low coupling, primarily during P21-P35

Population coupling is defined as the Pearson correlation between an individual neuron's spike train and the summed population firing rate.

---

## Repository Structure

```
Supplementary/
├── ED3/                     Electrode impedance characterization (EIS)
├── ED5/
│   ├── ED5ab/               Example unit developmental trajectories
│   └── ED5c_h/              Population-level metric trajectories
├── ED6/                     GMM model validation and optimal k selection
├── ED7/                     GMM normality and residual analysis
├── ED8/                     Trajectory characterization and statistics
├── ED9/                     Regional comparison (V1 vs mPFC)
├── ED10/                    Multi-metric LDA and feature profiles
├── Fig3g-k/                 Neuron tracking quality metrics (data only)
├── Fig4/
│   ├── fig4de/              GMM clustering visualization
│   ├── fig4fgh/             Example neuron trajectories
│   ├── fig4ijk/             Regional trajectory comparison
│   └── fig4l/               LDA space visualization
└── vlm_neuron_tracking/     VLM-based neuron matching pipeline
```

---

## Extended Data Figures

### ED3 -- Electrochemical Impedance Spectroscopy

Characterizes electrode impedance properties across frequency, incubation time, and mechanical conditions.

| File | Description |
|------|-------------|
| `ed3ab.csv` | Mean and SD of impedance magnitude and phase across 8 devices (frequencies up to 100 kHz) |
| `ed3ab_raw.csv` | Per-device raw EIS measurements |
| `ed3c.csv` | Impedance at 1 kHz over 35 days of incubation (50 devices, 6 time points) |
| `ed3d.csv` | Impedance at 1 kHz by condition (Released, Implanted, Stretched) |
| `ed3.py` | Plots panels a-d (Bode plots, incubation time course, condition comparison) |

### ED5 -- Electrophysiological Metric Trajectories

**ED5ab**: Burst index and ACG trough depth trajectories for two example Chorister-to-Soloist units, with linear regression fits.

**ED5c--i**: Population-level developmental trajectories for six metrics (burst index, ACG trough depth, CV2, LV, Fano factor, mean pairwise correlation) and pairwise correlation matrices across postnatal ages.

### ED6 -- GMM Validation

Validates the two-component GMM used for population coupling clustering: optimal k selection (BIC/AIC), silhouette coefficients, cross-validated AUROC, and component separation across animals.

### ED7 -- Normality and Residual Analysis

Assesses GMM assumptions through Q-Q plots, residual distributions, and per-animal cluster normality.

### ED8 -- Trajectory Statistics

Validates developmental trajectory definitions with age span distributions, transition timing estimates, cross-sectional binned analysis, variance decomposition (age vs trajectory), and pairwise effect sizes (Cohen's d).

### ED9 -- Regional Differences

Compares population coupling dynamics between visual cortex (V1) and medial prefrontal cortex (mPFC), including cluster distributions and trajectory proportions by region.

### ED10 -- Multi-Metric Feature Profiles

Visualizes 14 electrophysiological metrics across trajectory classes using z-scored heatmaps and Linear Discriminant Analysis (LDA) projections.

---

## Main Figure Panels

### Fig 3g--k -- Neuron Tracking Quality

Source data for panels characterizing chronic neuron tracking quality: spatial displacement rates, waveform similarity, spatial drift, amplitude stability, and waveform correlation over time. Data files only (no plotting script).

### Fig 4d--l -- Population Coupling Analysis

| Subdirectory | Panels | Description |
|-------------|--------|-------------|
| `fig4de/` | d--e | GMM k=2 clustering of population coupling with fitted Gaussian components |
| `fig4fgh/` | f--h | Example trajectories for one neuron per trajectory class |
| `fig4ijk/` | i--k | Population coupling trajectories split by brain region (V1/mPFC) |
| `fig4l/` | l | 14-metric LDA projection colored by trajectory class across 8 developmental ages |

---

## VLM Neuron Tracking Pipeline

The `vlm_neuron_tracking/` directory contains a Vision Language Model pipeline for automated neuron matching across chronic recording sessions. The pipeline uses multimodal LLMs (GPT-4o, Claude, Gemini) to compare waveform morphology, spike location, and firing rate between sessions.

See `vlm_neuron_tracking/README.md` for full documentation and usage instructions.

---

## Requirements

For the plotting scripts:

```
numpy
pandas
matplotlib
scipy
```

Install with:

```bash
pip install numpy pandas matplotlib scipy
```

The VLM neuron tracking pipeline has additional dependencies listed in `vlm_neuron_tracking/requirements.txt`.

---

## Reproducing Figures

Each directory contains a Python script that reads the local CSV files and generates the corresponding figure panels. To reproduce all figures:

```bash
# Extended Data Figures
python ED3/ed3.py
python ED5/ED5ab/ed5ab.py
python ED5/ED5c_h/ed5c_i_plot.py
python ED6/ed6.py
python ED7_/ed7.py
python ED8/ed8.py
python ED9/ed9.py
python ED10/ed10.py

# Main Figure 4
python Fig4/fig4de/fig4de_code.py
python Fig4/fig4fgh/fig4fgh.py
python Fig4/fig4ijk/fig4ijk.py
python Fig4/fig4l/fig4l.py
```

Each script outputs both PNG (300 dpi) and SVG files in the same directory.
