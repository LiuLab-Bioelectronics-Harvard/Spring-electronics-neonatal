"""
VLM Neuron Tracking — Demo Script

Loads two recording sessions from data/round5_rat5/ and runs the VLM
comparative alignment pipeline to match neurons across days.

Usage:
    python run_demo.py

Requires an OpenAI API key in .env (see .env.example).
"""
import base64
from pathlib import Path

from vlm.candidate_selector import load_session_data
from vlm.VLM import process_day_pair_comparative
from vlm.image_generator import generate_comparison_image

# ---------------------------------------------------------------------------
# Probe geometry (64 channels, 8 shanks)
# ---------------------------------------------------------------------------
CHANNEL_INDICES = [
    40, 63, 42, 61, 44, 59, 46, 57,
    48, 55, 49, 54, 50, 53, 51, 52,
    11, 12, 10, 13, 9, 14, 8, 15,
    7, 16, 5, 18, 3, 20, 1, 22,
    56, 47, 58, 45, 60, 43, 62, 41,
    24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39,
    23, 0, 21, 2, 19, 4, 17, 6
]
POSITIONS = [
    [61.72735618, 83.22196011], [99.31506102, 69.54115438],
    [48.04655045, 45.63425528], [85.63425528, 31.95344955],
    [34.36574472, 8.04655045], [71.95344955, -5.63425528],
    [20.68493898, -29.54115438], [58.27264382, -43.22196011],
    [-68.03847577, 7.32050808], [-48.03847577, -27.32050808],
    [-102.67949192, -12.67949192], [-82.67949192, -47.32050808],
    [-137.32050808, -32.67949192], [-117.32050808, -67.32050808],
    [-171.96152423, -52.67949192], [-151.96152423, -87.32050808],
    [61.72735618, -416.77803989], [99.31506102, -430.45884562],
    [48.04655045, -454.36574472], [85.63425528, -468.04655045],
    [34.36574472, -491.95344955], [71.95344955, -505.63425528],
    [20.68493898, -529.54115438], [58.27264382, -543.22196011],
    [-68.03847577, -492.67949192], [-48.03847577, -527.32050808],
    [-102.67949192, -512.67949192], [-82.67949192, -547.32050808],
    [-137.32050808, -532.67949192], [-117.32050808, -567.32050808],
    [-171.96152423, -552.67949192], [-151.96152423, -587.32050808],
    [220.68493898, 219.54115438], [258.27264382, 233.22196011],
    [234.36574472, 181.95344955], [271.95344955, 195.63425528],
    [248.04655045, 144.36574472], [285.63425528, 158.04655045],
    [261.72735618, 106.77803989], [299.31506102, 120.45884562],
    [175.3054506, 71.51479674], [215.15323853, 68.02856703],
    [171.81922089, 31.66700882], [211.66700882, 28.18077911],
    [168.33299118, -8.18077911], [208.18077911, -11.66700882],
    [164.84676147, -48.02856703], [204.6945494, -51.51479674],
    [220.68493898, -280.45884562], [258.27264382, -266.77803989],
    [234.36574472, -318.04655045], [271.95344955, -304.36574472],
    [248.04655045, -355.63425528], [285.63425528, -341.95344955],
    [261.72735618, -393.22196011], [299.31506102, -379.54115438],
    [175.3054506, -428.48520326], [215.15323853, -431.97143297],
    [171.81922089, -468.33299118], [211.66700882, -471.81922089],
    [168.33299118, -508.18077911], [208.18077911, -511.66700882],
    [164.84676147, -548.02856703], [204.6945494, -551.51479674]
]
SHANK_IDS = [
    '0','0','0','0','0','0','0','0',
    '1','1','1','1','1','1','1','1',
    '2','2','2','2','2','2','2','2',
    '3','3','3','3','3','3','3','3',
    '4','4','4','4','4','4','4','4',
    '5','5','5','5','5','5','5','5',
    '6','6','6','6','6','6','6','6',
    '7','7','7','7','7','7','7','7'
]

# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
MODEL_NAME = 'gpt_5_2'
K_CANDIDATES = 3
N_REVIEWERS = 3
TEMPERATURE = 0.5
CANDIDATE_WEIGHTS = {'spatial': 1.0, 'waveform': 1.0, 'amplitude': 1.0, 'firing_rate': 1.0}
SCORE_FUSION_ALPHA = 0.3   # pre-screening weight
SCORE_FUSION_BETA = 0.7    # VLM confidence weight
MIN_SCORE = 0.40


def main():
    root = Path(__file__).parent
    data_dir = root / 'data' / 'example_animal'
    output_dir = root / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'matched_pairs').mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load session data
    # ------------------------------------------------------------------
    print("Loading session data...")
    day1_data = load_session_data(str(data_dir / 'session_day1'))
    day2_data = load_session_data(str(data_dir / 'session_day2'))

    n1 = len(day1_data['sorting'].get_unit_ids())
    n2 = len(day2_data['sorting'].get_unit_ids())
    print(f"  Day 1: {n1} units")
    print(f"  Day 2: {n2} units")

    # ------------------------------------------------------------------
    # 2. Run VLM comparative alignment
    # ------------------------------------------------------------------
    print(f"\nRunning {MODEL_NAME} comparative alignment "
          f"(k={K_CANDIDATES}, reviewers={N_REVIEWERS})...\n")

    results_df, best_matches = process_day_pair_comparative(
        model_name=MODEL_NAME,
        day1_data=day1_data,
        day2_data=day2_data,
        channel_indices=CHANNEL_INDICES,
        positions=POSITIONS,
        shank_ids=SHANK_IDS,
        day1_label='Day1',
        day2_label='Day2',
        candidate_weights=CANDIDATE_WEIGHTS,
        k_candidates=K_CANDIDATES,
        n_reviewers=N_REVIEWERS,
        temperature=TEMPERATURE,
        min_score=MIN_SCORE,
        score_fusion_alpha=SCORE_FUSION_ALPHA,
        score_fusion_beta=SCORE_FUSION_BETA,
        example_pair_images=None,
    )

    # ------------------------------------------------------------------
    # 3. Save results CSV
    # ------------------------------------------------------------------
    csv_path = output_dir / 'results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # ------------------------------------------------------------------
    # 4. Generate comparison images for matched pairs
    # ------------------------------------------------------------------
    matched = {ref: comp for ref, comp in best_matches.items() if comp is not None}
    print(f"\nGenerating comparison images for {len(matched)} matched pairs...")

    for ref_uid, comp_uid in sorted(matched.items()):
        match_row = results_df[
            (results_df['ref_unit_id'] == ref_uid) &
            (results_df['comp_unit_id'] == comp_uid)
        ]
        fused = match_row['fused_score'].values[0] if len(match_row) > 0 else 0.0

        img_b64 = generate_comparison_image(
            ref_unit_id=ref_uid,
            comp_unit_id=comp_uid,
            ref_we=day1_data['we'],
            comp_we=day2_data['we'],
            ref_sorting=day1_data['sorting'],
            comp_sorting=day2_data['sorting'],
            ref_sampling_frequency=day1_data['sampling_frequency'],
            comp_sampling_frequency=day2_data['sampling_frequency'],
            ref_channel_ids=day1_data['channel_ids'],
            comp_channel_ids=day2_data['channel_ids'],
            channel_indices=CHANNEL_INDICES,
            positions=POSITIONS,
            shank_ids=SHANK_IDS,
            ref_day_label='Day1',
            comp_day_label='Day2',
        )
        img_path = output_dir / 'matched_pairs' / f'unit{ref_uid}_to_unit{comp_uid}.png'
        with open(img_path, 'wb') as f:
            f.write(base64.b64decode(img_b64))

        print(f"  Unit {ref_uid:3d} -> Unit {comp_uid:3d}  (fused_score={fused:.3f})")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    n_matched = len(matched)
    n_unmatched = len(best_matches) - n_matched
    print(f"\n{'=' * 50}")
    print(f"SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Day 1 units:    {n1}")
    print(f"  Day 2 units:    {n2}")
    print(f"  Matched pairs:  {n_matched}")
    print(f"  Unmatched:      {n_unmatched}")
    print(f"  Results CSV:    {csv_path}")
    print(f"  Match images:   {output_dir / 'matched_pairs'}/")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
