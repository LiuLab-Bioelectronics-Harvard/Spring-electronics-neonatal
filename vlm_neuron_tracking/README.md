# VLM Neuron Tracking

Vision Language Model pipeline for tracking neurons across chronic recording sessions in developing rat brains. The pipeline matches neurons between consecutive days by comparing waveform morphology, spike location, and firing rate using a VLM as a visual evaluator.

> **Important:** VLM-based matching is a preliminary screening step and does not replace expert confirmation. Large language models can make errors in visual assessment. All matched pairs produced by this pipeline should be reviewed and validated by a trained electrophysiologist before being used in downstream analyses.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up API key
cp .env.example .env
# Edit .env and add your OpenAI API key (or Anthropic/Google for other models)

# 3. Run the pipeline
python run_demo.py
```

Results are saved to `output/`:
- `results.csv` — scores for all candidate pairs
- `matched_pairs/*.png` — visual comparison for each matched pair

Pre-computed results are included in `output/` so you can inspect the pipeline output without running the VLM. Note that these results are automated VLM predictions prior to expert review.

## How It Works

The pipeline operates in four stages:

1. **Feature Extraction** — For each unit in both sessions, extract waveform template, peak channel, amplitude profile, spatial location (center of mass), and firing rate.

2. **Candidate Pre-screening** — For each Day 1 unit, compute pairwise similarity to all Day 2 units and select the top-K most similar candidates (default K=3). Similarity is a weighted combination of spatial proximity, waveform correlation, amplitude cosine similarity, and firing rate ratio.

3. **VLM Comparative Ranking** — Generate a multi-panel comparison image showing the reference unit alongside K candidates. Submit to a VLM (default: GPT-5.2) with a structured prompt requesting per-candidate confidence scores. Repeat with N independent "reviewers" (default N=3) and aggregate scores.

4. **Score Fusion & Assignment** — Combine pre-screening similarity (30%) and VLM confidence (70%) into a fused score. Apply the Hungarian algorithm for optimal 1-to-1 matching across all units.

## Configuration

Edit parameters at the top of `run_demo.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `gpt_5_2` | VLM model to use |
| `K_CANDIDATES` | `3` | Candidates per reference unit |
| `N_REVIEWERS` | `3` | Independent VLM reviews per unit |
| `TEMPERATURE` | `0.5` | VLM sampling temperature |
| `MIN_SCORE` | `0.40` | Minimum fused score for a valid match |
| `SCORE_FUSION_ALPHA` | `0.3` | Weight for pre-screening score |
| `SCORE_FUSION_BETA` | `0.7` | Weight for VLM confidence |

Supported models: `claude_4`, `gpt_5_2`, `gpt-4o`, `gpt-4-turbo`, `gpt-4o-mini`, `claude_3_7`, `gemini_2_0_flash`, `gemini_1_5_flash`, `gemini_1_5_pro`.

## Data Format

Each session directory should contain:

```
session_dayN/
├── waveform/
│   ├── templates_average.npy    # (n_units, n_samples, n_channels)
│   ├── templates_std.npy        # same shape
│   ├── params.json
│   ├── recording.json
│   ├── sorting.json
│   └── recording_info/
│       ├── recording_attributes.json   # sampling_frequency, channel_ids
│       └── probegroup.json
└── sorting/
    └── sorter_output/
        └── firings.npz          # SpikeInterface NpzSortingExtractor format
```

These are standard SpikeInterface waveform extractor outputs from MountainSort4.
