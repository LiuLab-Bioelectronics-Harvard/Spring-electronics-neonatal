"""
VLM Neuron Tracking Pipeline — Comparative Mode

Matches neurons across recording sessions using VLM (Vision Language Model)
comparative ranking. For each reference unit, the top-K most similar candidates
are shown simultaneously and the VLM ranks them by match likelihood.
"""
import asyncio
import os
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional, Tuple
from statistics import mean
from collections import defaultdict
import nest_asyncio
from tqdm.asyncio import tqdm
from pathlib import Path
import sys

from .VLM_prompt import get_comparative_prompt
from .candidate_selector import compute_all_unit_features, get_all_candidates_for_day_pair
from .image_generator import generate_comparative_image
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / '.env')

nest_asyncio.apply()


# ---------------------------------------------------------------------------
# Model registry (lazy initialization)
# ---------------------------------------------------------------------------
_model_cache = {}

N_SEMAPHORE = {
    "gpt_5_2": 50,
    "gpt-4o": 50,
    "gpt-4-turbo": 20,
    "gpt-4o-mini": 10,
    "claude_4": 50,
    "claude_3_7": 50,
    "claude_3_5_sonnet": 5,
    "claude_3_opus": 1,
    "claude_3_haiku": 5,
    "gemini_2_0_flash": 1,
    "gemini_1_5_flash": 20,
    "gemini_1_5_pro": 20,
}


def get_model(model_name: str):
    """Lazily create and cache a model instance."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    if model_name == "gpt_5_2":
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model='gpt-5.2', temperature=0.7, request_timeout=60)
    elif model_name in ("gpt-4o", "gpt-4-turbo", "gpt-4o-mini"):
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=model_name, temperature=0.7, request_timeout=60)
    elif model_name == "claude_3_7":
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model='claude-3-7-sonnet-20250219', temperature=0.7)
    elif model_name == "claude_4":
        from langchain_anthropic import ChatAnthropic
        model = ChatAnthropic(model='claude-sonnet-4-20250514', temperature=0.7)
    elif model_name in ("claude_3_5_sonnet", "claude_3_opus", "claude_3_haiku"):
        from .custom_class import ChatAnthropicCustom
        model_ids = {
            "claude_3_5_sonnet": "claude-3-5-sonnet-20240620-v1",
            "claude_3_opus": "claude-3-opus-20240229-v1",
            "claude_3_haiku": "claude-3-haiku-20240307-v1",
        }
        model = ChatAnthropicCustom(model_name=model_ids[model_name], temperature=0.7)
    elif model_name.startswith("gemini"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_ids = {
            "gemini_2_0_flash": "gemini-2.0-flash-exp",
            "gemini_1_5_flash": "gemini-1.5-flash",
            "gemini_1_5_pro": "gemini-1.5-pro",
        }
        model = ChatGoogleGenerativeAI(model=gemini_ids[model_name], temperature=0.7)
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: "
                         f"{list(N_SEMAPHORE.keys())}")

    _model_cache[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class CandidateRanking(BaseModel):
    """A single candidate's evaluation in comparative mode."""
    candidate_label: str = Field(description="Candidate label (A, B, C, etc.)")
    comp_unit_id: int = Field(description="The candidate unit ID.")
    confidence: float = Field(description="Confidence score 0-1 that this candidate matches the reference.")
    reasoning: str = Field(description="Brief reasoning for this candidate's score.")


class ComparativeClassification(BaseModel):
    """Structured output for comparative alignment (1 ref vs K candidates)."""
    ref_unit_id: int = Field(description="Reference unit ID.")
    reviewer_id: int = Field(description="Reviewer ID.")
    rankings: List[CandidateRanking] = Field(
        description="Per-candidate evaluations, ordered by confidence descending."
    )
    best_match_label: Optional[str] = Field(
        description="Label of best match (A/B/C) or null if no match."
    )
    overall_reasoning: str = Field(description="Overall reasoning for the ranking decision.")


# ---------------------------------------------------------------------------
# Content assembly
# ---------------------------------------------------------------------------
def get_comparative_content(example_pair_images: Optional[Dict[str, List[str]]],
                            comparative_img_b64: str):
    """Assemble multimodal content blocks for comparative VLM evaluation."""
    contents = []

    if example_pair_images:
        good_imgs = example_pair_images.get('good', [])
        if good_imgs:
            block = [{"type": "text", "text": "These are GOOD alignment examples — the same neuron recorded across two consecutive days:"}]
            for img in good_imgs:
                block.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
            contents.append(block)

        bad_imgs = example_pair_images.get('bad', [])
        if bad_imgs:
            block = [{"type": "text", "text": "These are BAD alignment examples — different neurons that should NOT be aligned:"}]
            for img in bad_imgs:
                block.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
            contents.append(block)

    target_block = [
        {"type": "text", "text": "Now evaluate this reference unit against the candidates shown. Rank the candidates and determine the best match:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{comparative_img_b64}"}}
    ]
    contents.append(target_block)

    return contents


# ---------------------------------------------------------------------------
# Single reviewer evaluation
# ---------------------------------------------------------------------------
async def evaluate_comparative_single(ref_unit_id, candidate_info, comparative_img_b64,
                                       reviewer_id, model, example_pair_images,
                                       ref_day, comp_day, candidate_metrics=None):
    """Single async VLM call for one reviewer in comparative mode."""
    await asyncio.sleep(0.5)

    structured_llm = model.with_structured_output(ComparativeClassification)

    sys_prompt = get_comparative_prompt(
        reviewer_id, ref_unit_id, candidate_info, ref_day, comp_day,
        candidate_metrics=candidate_metrics,
        example_pairs={'good_pairs': example_pair_images.get('good', []),
                       'bad_pairs': example_pair_images.get('bad', [])} if example_pair_images else None
    )
    sys_message = SystemMessage(content=sys_prompt)
    contents = get_comparative_content(example_pair_images, comparative_img_b64)
    messages = [sys_message] + [HumanMessage(content=block) for block in contents]

    max_retries = 10
    attempt = 0
    while attempt < max_retries:
        try:
            response = await structured_llm.ainvoke(messages)
            return {"reviewer_id": reviewer_id, "response": response}
        except Exception as e:
            attempt += 1
            print(f"Error for ref {ref_unit_id} reviewer {reviewer_id}, "
                  f"attempt {attempt}: {e}")
            if attempt == max_retries:
                print(f"Skipping ref {ref_unit_id} reviewer {reviewer_id} "
                      f"after {max_retries} failures.")
                fallback_rankings = [
                    CandidateRanking(
                        candidate_label=label, comp_unit_id=uid,
                        confidence=0.0,
                        reasoning=f"Failed after {max_retries} retries"
                    )
                    for label, uid in candidate_info
                ]
                return {
                    "reviewer_id": reviewer_id,
                    "response": ComparativeClassification(
                        ref_unit_id=ref_unit_id, reviewer_id=reviewer_id,
                        rankings=fallback_rankings, best_match_label=None,
                        overall_reasoning=f"Failed after {max_retries} retries: {e}"
                    )
                }
            await asyncio.sleep(1)


# ---------------------------------------------------------------------------
# Ensemble aggregation
# ---------------------------------------------------------------------------
def aggregate_comparative_reviews(reviews, ref_unit_id, candidate_info):
    """Aggregate comparative reviews from multiple reviewers into per-candidate results."""
    candidate_scores = defaultdict(list)
    candidate_reasonings = defaultdict(list)
    for review in reviews:
        resp = review["response"]
        for ranking in resp.rankings:
            candidate_scores[ranking.comp_unit_id].append(ranking.confidence)
            candidate_reasonings[ranking.comp_unit_id].append(
                f"Reviewer {review['reviewer_id']}: {ranking.reasoning}"
            )

    results = []
    for label, uid in candidate_info:
        scores = candidate_scores.get(uid, [0.0])
        avg_score = round(mean(scores), 2)
        classification = 'Match' if avg_score >= 0.60 else 'No_match'

        individual_scores = {}
        for i, s in enumerate(scores):
            individual_scores[f"reviewer_{i+1}_score"] = s

        results.append({
            "ref_unit_id": ref_unit_id,
            "comp_unit_id": uid,
            "average_score": avg_score,
            "final_classification": classification,
            "match_votes": sum(1 for s in scores if s >= 0.60),
            "n_reviewers": len(scores),
            "combined_reasoning": "\n".join(candidate_reasonings.get(uid, [])),
            **individual_scores
        })

    return results


async def process_ref_unit_comparative(ref_unit_id, candidate_info, comparative_img_b64,
                                        model, example_pair_images, ref_day, comp_day,
                                        n_reviewers=3, candidate_metrics=None):
    """Run n reviewers on one ref unit and aggregate."""
    tasks = [
        evaluate_comparative_single(
            ref_unit_id, candidate_info, comparative_img_b64,
            reviewer_id, model, example_pair_images, ref_day, comp_day,
            candidate_metrics
        )
        for reviewer_id in range(1, n_reviewers + 1)
    ]
    reviews = await asyncio.gather(*tasks)
    return aggregate_comparative_reviews(reviews, ref_unit_id, candidate_info)


# ---------------------------------------------------------------------------
# Batch processing with concurrency control
# ---------------------------------------------------------------------------
async def run_comparative_in_batch(ref_units_data, model, example_pair_images,
                                    ref_day, comp_day, n_reviewers=3, n_semaphore=5):
    """Process all reference units with concurrency-limited API calls."""
    all_results = []
    semaphore = asyncio.Semaphore(n_semaphore)

    async def limited_task(unit_data, progress_bar):
        async with semaphore:
            result = await process_ref_unit_comparative(
                unit_data['ref_unit_id'],
                unit_data['candidate_info'],
                unit_data['comparative_img_b64'],
                model, example_pair_images, ref_day, comp_day,
                n_reviewers,
                unit_data.get('candidate_metrics')
            )
            progress_bar.update(1)
            return result

    with tqdm(total=len(ref_units_data), desc="Comparative Alignment",
              unit="unit", file=sys.stdout, leave=True) as progress_bar:
        tasks = [limited_task(ud, progress_bar) for ud in ref_units_data]
        batch_size = max(1, n_semaphore // n_reviewers)
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            for per_cand_results in batch_results:
                all_results.extend(per_cand_results)

    return all_results


# ---------------------------------------------------------------------------
# Score fusion
# ---------------------------------------------------------------------------
def fuse_scores(all_results, prescreen_scores, alpha=0.3, beta=0.7):
    """Combine pre-screening similarity and VLM confidence into a fused score."""
    for r in all_results:
        key = (r['ref_unit_id'], r['comp_unit_id'])
        ps = prescreen_scores.get(key, 0.0)
        r['prescreen_score'] = round(ps, 3)
        r['fused_score'] = round(alpha * ps + beta * r['average_score'], 3)
    return all_results


# ---------------------------------------------------------------------------
# Hungarian assignment (1-to-1 optimal matching)
# ---------------------------------------------------------------------------
def _enforce_unique_assignment(all_results, ref_unit_ids, min_score=0.40):
    """Enforce 1-to-1 assignment using the Hungarian algorithm.

    Builds a score matrix from VLM results and finds the optimal assignment
    that maximizes total score while ensuring each comp unit is assigned
    to at most one ref unit.
    """
    valid_scores = {}
    for r in all_results:
        if r['average_score'] >= min_score:
            key = (r['ref_unit_id'], r['comp_unit_id'])
            if key not in valid_scores or r['average_score'] > valid_scores[key]:
                valid_scores[key] = r['average_score']

    comp_unit_ids = sorted(set(comp for _, comp in valid_scores))

    if not comp_unit_ids:
        return {ref_uid: None for ref_uid in ref_unit_ids}

    ref_list = sorted(ref_unit_ids)
    n_ref = len(ref_list)
    n_comp = len(comp_unit_ids)

    ref_idx = {uid: i for i, uid in enumerate(ref_list)}
    comp_idx = {uid: i for i, uid in enumerate(comp_unit_ids)}

    # Extra dummy columns for "no-match" assignments
    n_cols = n_comp + n_ref
    score_matrix = np.zeros((n_ref, n_cols))

    for (ref_uid, comp_uid), score in valid_scores.items():
        if ref_uid in ref_idx and comp_uid in comp_idx:
            score_matrix[ref_idx[ref_uid], comp_idx[comp_uid]] = score

    for i in range(n_ref):
        score_matrix[i, n_comp + i] = min_score

    cost_matrix = 1.0 - score_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    best_matches = {}
    for r_i, c_i in zip(row_ind, col_ind):
        ref_uid = ref_list[r_i]
        if c_i < n_comp:
            best_matches[ref_uid] = comp_unit_ids[c_i]
        else:
            best_matches[ref_uid] = None

    for ref_uid in ref_unit_ids:
        if ref_uid not in best_matches:
            best_matches[ref_uid] = None

    return best_matches


# ---------------------------------------------------------------------------
# Main entry point: comparative mode (single day pair)
# ---------------------------------------------------------------------------
def process_day_pair_comparative(model_name: str,
                                  day1_data: Dict,
                                  day2_data: Dict,
                                  channel_indices: List[int],
                                  positions: List[List[float]],
                                  shank_ids: List[str],
                                  day1_label: str,
                                  day2_label: str,
                                  candidate_weights: Dict[str, float],
                                  k_candidates: int = 3,
                                  n_reviewers: int = 3,
                                  temperature: float = 0.5,
                                  n_semaphore: Optional[int] = None,
                                  min_score: float = 0.40,
                                  score_fusion_alpha: float = 0.3,
                                  score_fusion_beta: float = 0.7,
                                  example_pair_images: Optional[Dict] = None,
                                  ) -> Tuple[pd.DataFrame, Dict]:
    """Run comparative VLM alignment for a single day pair.

    Shows the VLM all K candidates for each reference unit simultaneously
    and asks it to rank them.

    Args:
        model_name: Name of VLM model (e.g., 'gpt-4o').
        day1_data: Dict with 'we', 'sorting', 'sampling_frequency', 'channel_ids'.
        day2_data: Same structure as day1_data.
        channel_indices: Probe channel-to-position mapping.
        positions: Physical (x, y) coordinates per probe site.
        shank_ids: Shank assignment per probe site.
        day1_label, day2_label: Human-readable labels.
        candidate_weights: Weights for pre-screening similarity.
        k_candidates: Number of candidates per reference unit.
        n_reviewers: Number of independent VLM reviewers.
        temperature: VLM temperature.
        n_semaphore: Max concurrent API calls.
        min_score: Minimum fused score for valid match.
        score_fusion_alpha: Weight for pre-screening similarity.
        score_fusion_beta: Weight for VLM confidence.
        example_pair_images: Optional few-shot example images.

    Returns:
        (results_df, best_matches) where results_df has all candidate scores
        and best_matches maps ref_unit_id -> matched comp_unit_id (or None).
    """
    if n_semaphore is None:
        n_semaphore = N_SEMAPHORE.get(model_name, 5)

    model = get_model(model_name)
    model.temperature = temperature

    # Step 1: Extract features
    print(f"Extracting features for {day1_label}...")
    day1_features = compute_all_unit_features(
        day1_data['we'], day1_data['sorting'],
        day1_data['sampling_frequency'], day1_data['channel_ids'],
        unit_ids=day1_data.get('unit_ids')
    )
    print(f"  {len(day1_features)} units")

    print(f"Extracting features for {day2_label}...")
    day2_features = compute_all_unit_features(
        day2_data['we'], day2_data['sorting'],
        day2_data['sampling_frequency'], day2_data['channel_ids'],
        unit_ids=day2_data.get('unit_ids')
    )
    print(f"  {len(day2_features)} units")

    # Step 2: Find top-K candidates
    print(f"Finding top-{k_candidates} candidates per unit...")
    candidates = get_all_candidates_for_day_pair(
        day1_features, day2_features, candidate_weights, k_candidates
    )

    # Step 3: Generate comparative images
    print("Generating comparative images...")
    ref_units_data = []
    prescreen_scores = {}

    for ref_uid, cands in candidates.items():
        cand_unit_ids = [c[0] for c in cands]
        cand_labels = [chr(ord('A') + i) for i in range(len(cands))]
        candidate_info = list(zip(cand_labels, cand_unit_ids))

        cand_metrics = {}
        comp_firing_rates = {}
        for comp_uid, sim_score, metrics in cands:
            prescreen_scores[(ref_uid, comp_uid)] = sim_score
            cand_metrics[comp_uid] = metrics
            comp_firing_rates[comp_uid] = day2_features[comp_uid]['firing_rate']

        try:
            img_b64 = generate_comparative_image(
                ref_uid, cand_unit_ids,
                day1_data['we'], day2_data['we'],
                day1_data['sorting'], day2_data['sorting'],
                day1_data['sampling_frequency'], day2_data['sampling_frequency'],
                day1_data['channel_ids'], day2_data['channel_ids'],
                channel_indices, positions, shank_ids,
                day1_label, day2_label,
                ref_firing_rate=day1_features[ref_uid]['firing_rate'],
                comp_firing_rates=comp_firing_rates,
                candidate_metrics=cand_metrics
            )
            ref_units_data.append({
                'ref_unit_id': ref_uid,
                'candidate_info': candidate_info,
                'comparative_img_b64': img_b64,
                'candidate_metrics': cand_metrics
            })
        except Exception as e:
            print(f"Warning: Comparative image failed for ref {ref_uid}: {e}")

    print(f"Generated {len(ref_units_data)} comparative images")

    # Step 4: Run comparative VLM alignment
    print(f"{model_name} - Starting comparative alignment for {len(ref_units_data)} units "
          f"({n_reviewers} reviewers each)")
    all_results = asyncio.run(run_comparative_in_batch(
        ref_units_data, model, example_pair_images,
        day1_label, day2_label, n_reviewers, n_semaphore
    ))

    # Step 5: Score fusion
    all_results = fuse_scores(all_results, prescreen_scores,
                               alpha=score_fusion_alpha, beta=score_fusion_beta)

    # Step 6: Hungarian 1-to-1 assignment
    results_df = pd.DataFrame(all_results)
    fused_results = [dict(r, average_score=r['fused_score']) for r in all_results]
    best_matches = _enforce_unique_assignment(
        fused_results, list(candidates.keys()), min_score=min_score
    )

    results_df['is_best_match'] = results_df.apply(
        lambda row: best_matches.get(row['ref_unit_id']) == row['comp_unit_id'],
        axis=1
    )

    print(f"Found {sum(1 for v in best_matches.values() if v is not None)} matches "
          f"out of {len(best_matches)} reference units")

    return results_df, best_matches


# ---------------------------------------------------------------------------
# Multi-day alignment
# ---------------------------------------------------------------------------
def process_multi_day_alignment(model_name: str,
                                sessions_data: List[Dict],
                                session_labels: List[str],
                                channel_indices: List[int],
                                positions: List[List[float]],
                                shank_ids: List[str],
                                candidate_weights: Dict[str, float],
                                k_candidates: int = 3,
                                n_reviewers: int = 3,
                                temperature: float = 0.5,
                                n_semaphore: Optional[int] = None,
                                save_folder: Optional[str] = None,
                                min_score: float = 0.40,
                                example_pair_images: Optional[Dict] = None) -> Dict:
    """Run alignment across all consecutive day pairs and build trajectories.

    Args:
        sessions_data: List of dicts (chronological), each with keys
            'we', 'sorting', 'sampling_frequency', 'channel_ids'.
        session_labels: Human-readable labels per session.

    Returns:
        Dict with 'pairwise_results', 'best_matches', 'trajectories'.
    """
    if save_folder:
        Path(save_folder).mkdir(parents=True, exist_ok=True)

    all_pairwise_results = {}
    all_best_matches = {}

    for i in range(len(sessions_data) - 1):
        day1_label = session_labels[i]
        day2_label = session_labels[i + 1]
        pair_key = f"{day1_label}_vs_{day2_label}"
        print(f"\n{'=' * 60}")
        print(f"Aligning {day1_label} -> {day2_label}")
        print(f"{'=' * 60}")

        results_df, best_matches = process_day_pair_comparative(
            model_name=model_name,
            day1_data=sessions_data[i],
            day2_data=sessions_data[i + 1],
            channel_indices=channel_indices,
            positions=positions,
            shank_ids=shank_ids,
            day1_label=day1_label,
            day2_label=day2_label,
            candidate_weights=candidate_weights,
            k_candidates=k_candidates,
            n_reviewers=n_reviewers,
            temperature=temperature,
            n_semaphore=n_semaphore,
            min_score=min_score,
            example_pair_images=example_pair_images
        )

        all_pairwise_results[pair_key] = results_df
        all_best_matches[pair_key] = best_matches

        if save_folder:
            csv_path = Path(save_folder) / f"alignment_{pair_key}.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")

    trajectories = _build_trajectories(all_best_matches, session_labels)

    if save_folder:
        traj_df = _trajectories_to_dataframe(trajectories, session_labels)
        traj_path = Path(save_folder) / "trajectories.csv"
        traj_df.to_csv(traj_path, index=False)
        print(f"\nSaved trajectories to {traj_path}")

    return {
        'pairwise_results': all_pairwise_results,
        'best_matches': all_best_matches,
        'trajectories': trajectories
    }


# ---------------------------------------------------------------------------
# Trajectory building
# ---------------------------------------------------------------------------
def _build_trajectories(all_best_matches, session_labels):
    """Build neuron trajectories from pairwise match results via forward chaining."""
    forward_edges = {}
    reverse_edges = {}

    for i in range(len(session_labels) - 1):
        pair_key = f"{session_labels[i]}_vs_{session_labels[i + 1]}"
        matches = all_best_matches.get(pair_key, {})
        for ref_uid, comp_uid in matches.items():
            if comp_uid is not None:
                node1 = (i, ref_uid)
                node2 = (i + 1, comp_uid)
                forward_edges[node1] = node2
                reverse_edges[node2] = node1

    all_nodes = set(forward_edges.keys()) | set(forward_edges.values())
    starts = [n for n in all_nodes if n not in reverse_edges]

    for node in forward_edges:
        if node not in reverse_edges and node not in starts:
            starts.append(node)

    trajectories = []
    visited = set()

    for start in sorted(starts):
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        current = start
        while current in forward_edges:
            nxt = forward_edges[current]
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt

        if len(chain) >= 2:
            trajectories.append({
                'trajectory_id': len(trajectories),
                'n_sessions': len(chain),
                'sessions': [session_labels[day_idx] for day_idx, _ in chain],
                'unit_ids': [uid for _, uid in chain],
                'day_indices': [day_idx for day_idx, _ in chain]
            })

    print(f"\nBuilt {len(trajectories)} trajectories spanning 2+ sessions")
    return trajectories


def _trajectories_to_dataframe(trajectories, session_labels):
    """Convert trajectories list to a flat DataFrame."""
    rows = []
    for traj in trajectories:
        for i, (session, unit_id) in enumerate(zip(traj['sessions'], traj['unit_ids'])):
            rows.append({
                'trajectory_id': traj['trajectory_id'],
                'session': session,
                'unit_id': unit_id,
                'position_in_trajectory': i,
                'n_sessions': traj['n_sessions']
            })
    return pd.DataFrame(rows)
