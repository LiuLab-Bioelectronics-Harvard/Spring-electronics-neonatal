# Supplementary Code

Plotting scripts and analysis code accompanying the manuscript on **Growth-adaptive spring electronics for long-term, same-neuron mapping in the developing rat brain**.

---

## Overview

We tracked neural populations in neonatal rats (P10-P45) across chronic recording sessions to characterize how individual neurons transition between population coupling states during early postnatal development. Neurons were classified into three developmental trajectory types using Gaussian Mixture Model (GMM) clustering:

- **Stable Soloist** - consistently low population coupling
- **Stable Chorister** - consistently high population coupling
- **Chorister-to-Soloist** - transitions from high to low coupling, primarily during P21-P35

Population coupling is defined as the Pearson correlation between an individual neuron's spike train and the summed population firing rate.

---

### Extended Data Figures

| Script | Figure | Description |
|--------|--------|-------------|
| `eis.py` | ED3 | Electrode impedance spectroscopy (Bode plots, incubation, conditions) |
| `individual_neuron_trajectories.py` | ED5 a-b | Burst index trajectories for two example units |
| `population_level_trajectories.py` | ED5 c-i | Population-level metric trajectories and correlation matrices |
| `cluster_validation_metrics.py` | ED6 | GMM validation (optimal k, silhouette, AUROC, separation) |
| `per_animal_cluster_normality.py` | ED7 | Per-animal cluster normality and residual analysis |
| `trajectory_characterisation.py` | ED8 | Trajectory statistics (age span, timing, variance, effect sizes) |
| `regional_clustering.py` | ED9 | Regional comparison of V1 and mPFC |
| `metrics_heatmap_lda.py` | ED10 | 14-metric z-scored heatmaps and LDA projections |

### Main Figure 4 Panels

| Script | Panels | Description |
|--------|--------|-------------|
| `gmm_fitting.py` | d-e | GMM k=2 clustering with fitted Gaussian components |
| `example_trajectories.py` | f-h | Example trajectories for one neuron per trajectory class |
| `populationcoupling_trajectories.py` | i-k | Population coupling trajectories split by brain region |
| `population_coupling_lda_visual.py` | l | 14-metric LDA projection across 8 developmental ages |

---

## VLM Neuron Tracking Pipeline

The `vlm_neuron_tracking/` directory contains a Vision Language Model pipeline for automated neuron matching across chronic recording sessions. The pipeline uses multimodal LLMs (Claude 4, GPT-5.2, and others) to compare waveform morphology, spike location, and firing rate between sessions.

See [`vlm_neuron_tracking/README.md`](vlm_neuron_tracking/README.md) for full documentation and usage instructions.

---

## Requirements

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

