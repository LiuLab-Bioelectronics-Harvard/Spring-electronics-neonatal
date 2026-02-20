from typing import Dict, List, Optional, Tuple


def get_comparative_prompt(reviewer_id: int,
                           ref_unit_id: int,
                           candidate_info: List[Tuple[str, int]],
                           ref_day: str,
                           comp_day: str,
                           candidate_metrics: Optional[Dict[int, Dict[str, float]]] = None,
                           example_pairs: Optional[Dict] = None) -> str:
    """Generate the system prompt for comparative VLM alignment evaluation.

    Instead of evaluating a single pair, the VLM sees the reference unit and
    all K candidates simultaneously, and ranks them by match likelihood.

    Args:
        reviewer_id: Independent reviewer number (1, 2, 3, ...).
        ref_unit_id: Reference unit ID (Day N).
        candidate_info: List of (label, unit_id) for each candidate,
            e.g. [("A", 32), ("B", 45), ("C", 51)].
        ref_day: Label for reference day.
        comp_day: Label for comparison day.
        candidate_metrics: Optional dict mapping candidate unit_id -> dict of
            numerical similarity metrics (spatial, waveform, amplitude, firing_rate).
        example_pairs: Optional dict with 'good_pairs' and 'bad_pairs' info.

    Returns:
        System prompt string.
    """
    K = len(candidate_info)
    cand_desc = ", ".join(f"Candidate {label} (Unit {uid})" for label, uid in candidate_info)

    # Build example reference text
    example_text = ""
    if example_pairs:
        good_pairs = example_pairs.get('good_pairs', [])
        bad_pairs = example_pairs.get('bad_pairs', [])
        if good_pairs or bad_pairs:
            example_text = "\nYou will first be shown example alignment images:\n"
            if good_pairs:
                example_text += f"- Good alignment examples ({len(good_pairs)} shown): These show the SAME neuron across two days. Note how waveform shape, multi-channel pattern, and spike location are all preserved.\n"
            if bad_pairs:
                example_text += f"- Bad alignment examples ({len(bad_pairs)} shown): These show DIFFERENT neurons. Note the differences.\n"
            example_text += "\nUse these examples to calibrate your judgment.\n"

    # Build numerical metrics text
    metrics_text = ""
    if candidate_metrics:
        metrics_text = "\n**Pre-computed numerical similarity metrics** (for reference — these are computed from the raw data with full precision):\n"
        for label, uid in candidate_info:
            if uid in candidate_metrics:
                m = candidate_metrics[uid]
                metrics_text += (f"- Candidate {label} (Unit {uid}): "
                                 f"waveform_correlation={m.get('waveform', 0):.3f}, "
                                 f"spatial_similarity={m.get('spatial', 0):.3f}, "
                                 f"amplitude_similarity={m.get('amplitude', 0):.3f}, "
                                 f"firing_rate_ratio={m.get('firing_rate', 0):.3f}\n")
        metrics_text += "\nThese metrics provide quantitative anchors. Use them alongside the visual comparison.\n"

    prompt = f"""You are an expert in performing spike alignment in developing brain recordings. Your task is to determine which (if any) of {K} candidate neurons from {comp_day} is the SAME neuron as the reference neuron from {ref_day}, recorded across two different days.

IMPORTANT CONTEXT: These candidates were pre-selected as the most visually similar neurons to the reference. They will all look somewhat similar — that is expected. Your job is to identify which candidate (if any) is the TRUE match, not just a similar-looking neuron. At most one candidate is the true match; often none of them are.

You are shown a comparative image with the following layout:
- **Column 1**: Reference neuron ({ref_day}, Unit {ref_unit_id})
- **Columns 2-{1+K}**: {cand_desc}

Each column shows (from top to bottom):
- **Row 1**: Waveform template on the peak channel (mean waveform with standard deviation shading). All columns share the same y-axis scale.
- **Row 2**: Multi-channel waveforms across channels on the same probe shank. All columns share the same amplitude spacing.
- **Row 3**: Spike location on the probe (red star = amplitude-weighted center-of-mass) with firing rate annotation.
- **Row 4** (candidates only): Waveform overlay — reference (blue) and candidate (red dashed) plotted on the same axes for direct comparison.
{example_text}{metrics_text}
**Evaluation criteria** — compare each candidate to the reference on these features:

1. **Waveform Template Shape** (CRITICAL):
   - The waveform shape must be specifically matching, not just generically similar
   - Look at peak-to-trough ratio, biphasic/triphasic pattern, and distinctive morphological features
   - The overlay panel (Row 4) is the most informative — look for traces that track each other closely
   - Small amplitude scaling between days is acceptable

2. **Multi-Channel Spatial Pattern** (CRITICAL):
   - Which channels have large signals vs. quiet must match
   - The peak channel (marked *) must be the same or immediately adjacent
   - Amplitude gradient across channels should be preserved

3. **Spike Location** (CRITICAL):
   - The red star position must be at the same or very nearby location
   - Spatial drift up to ~50µm is possible between consecutive days in developing brain
   - Different shank = immediate rejection

4. **Firing Rate** (SOFT signal):
   - In developing brain recordings, firing rates can change substantially (2-5x) between sessions
   - Firing rate difference alone should NOT disqualify a candidate
   - Use it as a tiebreaker when other features are ambiguous

**RANKING TASK**: For each candidate, assign a confidence score (0.00 to 1.00, continuous scale) indicating how likely it is to be the SAME neuron as the reference. Then select the best match (or "none" if no candidate is convincing).

Confidence score guidelines (continuous, use the full range):
- 0.00-0.20: Clearly different — obvious discrepancies in shape, location, or spatial pattern
- 0.20-0.50: Unlikely match — some features differ notably
- 0.50-0.70: Ambiguous — some similarities but not convincing
- 0.70-0.85: Probable match — most features align well
- 0.85-1.00: Highly confident match — all features clearly match with distinctive similarity

**Decision rule for best_match**:
- Select the candidate with the highest confidence IF its score >= 0.60 AND it is meaningfully better than the second-best (gap >= 0.10)
- If the top two candidates are within 0.10 of each other and both >= 0.60, you are uncertain — still rank them but note the ambiguity
- If no candidate scores >= 0.60, select "none"

Format your response for structured parsing:
========
1. Reference: Unit {ref_unit_id} ({ref_day})
2. ReviewerID: {reviewer_id}
3. Per-candidate evaluation:
"""

    for label, uid in candidate_info:
        prompt += f"""   - **Candidate {label} (Unit {uid})**:
     - Waveform: [evaluation]
     - Multi-channel: [evaluation]
     - Location: [evaluation]
     - Firing rate: [evaluation]
     - Confidence: [0.00-1.00]
"""

    prompt += f"""4. Ranking: [List candidates from most to least likely, with scores]
5. Best match: [Candidate label and unit ID, or "none"]
6. Overall reasoning: [Why you chose this candidate or why none match]
========
"""
    return prompt
