import json
import numpy as np
import spikeinterface.core as sc
import spikeinterface.extractors as se
from spikeinterface.core.template_tools import get_template_extremum_channel
from scipy.spatial.distance import cosine
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def load_session_data(session_path: str) -> Dict:
    """Load SpikeInterface data for a single recording session.

    Loads the waveform extractor in template-only mode (no raw recording or
    raw waveform files needed). Extracts sampling frequency and channel IDs
    from the saved metadata.

    Args:
        session_path: Path to session directory containing waveform/ and sorting/.

    Returns:
        Dict with keys: 'we', 'sorting', 'sampling_frequency', 'channel_ids'.
    """
    session_path = Path(session_path)
    waveform_path = session_path / "waveform"
    sorting_path = session_path / "sorting" / "sorter_output" / "firings.npz"

    sorting = se.NpzSortingExtractor(str(sorting_path))
    we = sc.load_waveforms(folder=str(waveform_path),
                           with_recording=False, sorting=sorting)

    # Extract sampling_frequency and channel_ids from saved metadata
    rec_attrs_path = waveform_path / "recording_info" / "recording_attributes.json"
    with open(rec_attrs_path) as f:
        rec_attrs = json.load(f)
    sampling_frequency = float(rec_attrs['sampling_frequency'])
    channel_ids = [int(ch) for ch in rec_attrs['channel_ids']]

    return {
        'we': we,
        'sorting': sorting,
        'sampling_frequency': sampling_frequency,
        'channel_ids': channel_ids,
    }


def compute_unit_features(unit_id: int,
                          we,
                          sorting,
                          sampling_frequency: float,
                          channel_ids: List[int],
                          channel_locations: Optional[np.ndarray] = None) -> Dict:
    """Extract alignment features for a single unit.

    Returns dict with keys: extremum_channel, location, amplitude_profile,
    mean_template, firing_rate.
    """
    features = {}

    # Extremum channel
    extremum_channels = get_template_extremum_channel(we, peak_sign='neg')
    extremum_ch = extremum_channels[unit_id]
    features['extremum_channel'] = int(extremum_ch)

    # Mean waveform template (n_timepoints, n_channels)
    template = we.get_template(unit_id)
    features['mean_template'] = template

    # Channel amplitude profile (peak amplitude per channel, normalized)
    peak_amplitudes = np.max(np.abs(template), axis=0)
    norm = np.linalg.norm(peak_amplitudes)
    if norm > 1e-10:
        amplitude_profile = peak_amplitudes / norm
    else:
        amplitude_profile = peak_amplitudes
    features['amplitude_profile'] = amplitude_profile

    # Spike location (center of mass weighted by amplitude)
    if channel_locations is not None:
        locs = np.array(channel_locations)
    else:
        locs = None

    if locs is not None and len(locs) >= len(peak_amplitudes):
        weights = peak_amplitudes / (np.sum(peak_amplitudes) + 1e-10)
        com = np.average(locs[:len(peak_amplitudes)], weights=weights, axis=0)
        features['location'] = com
    else:
        # Fallback: use extremum channel index as location proxy
        if extremum_ch in channel_ids:
            ch_idx = channel_ids.index(extremum_ch)
        else:
            ch_idx = 0
        features['location'] = np.array([float(ch_idx), 0.0])

    # Firing rate
    spike_train = sorting.get_unit_spike_train(unit_id)
    if len(spike_train) > 1:
        duration = (spike_train[-1] - spike_train[0]) / sampling_frequency
        features['firing_rate'] = float(len(spike_train) / duration) if duration > 0 else 0.0
    else:
        features['firing_rate'] = 0.0

    return features


def compute_all_unit_features(we,
                              sorting,
                              sampling_frequency: float,
                              channel_ids: List[int],
                              unit_ids: Optional[List[int]] = None,
                              channel_locations: Optional[np.ndarray] = None) -> Dict[int, Dict]:
    """Compute features for all (or specified) units in a session.

    Returns dict mapping unit_id -> features dict.
    """
    if unit_ids is None:
        unit_ids = list(sorting.get_unit_ids())

    all_features = {}
    for uid in unit_ids:
        try:
            all_features[uid] = compute_unit_features(
                uid, we, sorting, sampling_frequency, channel_ids, channel_locations
            )
        except Exception as e:
            print(f"Warning: Failed to compute features for unit {uid}: {e}")

    return all_features


def compute_pairwise_similarity(unit1_features: Dict,
                                unit2_features: Dict,
                                weights: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """Compute weighted similarity between two units.

    Args:
        unit1_features: Features dict for reference unit (Day N).
        unit2_features: Features dict for candidate unit (Day N+1).
        weights: Dict with keys 'spatial', 'waveform', 'amplitude', 'firing_rate'.

    Returns:
        Tuple of (combined_similarity, individual_metrics).
    """
    metrics = {}

    # Spatial similarity: 1 / (1 + euclidean_distance)
    loc1 = unit1_features['location']
    loc2 = unit2_features['location']
    spatial_dist = np.linalg.norm(loc1 - loc2)
    metrics['spatial'] = 1.0 / (1.0 + spatial_dist)

    # Waveform similarity: Pearson correlation of flattened templates
    t1 = unit1_features['mean_template'].flatten()
    t2 = unit2_features['mean_template'].flatten()
    if t1.std() > 1e-6 and t2.std() > 1e-6 and len(t1) == len(t2):
        waveform_corr = np.corrcoef(t1, t2)[0, 1]
        metrics['waveform'] = max(0.0, waveform_corr)
    else:
        metrics['waveform'] = 0.0

    # Amplitude profile cosine similarity
    a1 = unit1_features['amplitude_profile']
    a2 = unit2_features['amplitude_profile']
    if np.linalg.norm(a1) > 1e-6 and np.linalg.norm(a2) > 1e-6 and len(a1) == len(a2):
        cos_sim = 1.0 - cosine(a1, a2)
        metrics['amplitude'] = max(0.0, cos_sim)
    else:
        metrics['amplitude'] = 0.0

    # Firing rate similarity: min/max ratio
    r1 = unit1_features['firing_rate']
    r2 = unit2_features['firing_rate']
    if r1 > 0 and r2 > 0:
        metrics['firing_rate'] = min(r1, r2) / max(r1, r2)
    elif r1 == 0 and r2 == 0:
        metrics['firing_rate'] = 1.0
    else:
        metrics['firing_rate'] = 0.0

    # Weighted combination
    w_total = sum(weights.values())
    if w_total <= 0:
        return 0.0, metrics

    combined = sum(weights[k] / w_total * metrics[k] for k in weights if k in metrics)

    return combined, metrics


def find_top_k_candidates(ref_unit_id: int,
                          day1_features: Dict[int, Dict],
                          day2_features: Dict[int, Dict],
                          weights: Dict[str, float],
                          k: int = 3) -> List[Tuple[int, float, Dict[str, float]]]:
    """Find the top-K most similar candidates on Day N+1 for a reference unit.

    Returns list of (comp_unit_id, combined_similarity, individual_metrics)
    sorted by similarity descending.
    """
    if ref_unit_id not in day1_features:
        return []

    ref_features = day1_features[ref_unit_id]
    scores = []

    for comp_uid, comp_features in day2_features.items():
        try:
            sim, metrics = compute_pairwise_similarity(ref_features, comp_features, weights)
            scores.append((comp_uid, sim, metrics))
        except Exception as e:
            print(f"Warning: Similarity computation failed for {ref_unit_id} vs {comp_uid}: {e}")

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]


def get_all_candidates_for_day_pair(day1_features: Dict[int, Dict],
                                    day2_features: Dict[int, Dict],
                                    weights: Dict[str, float],
                                    k: int = 3) -> Dict[int, List[Tuple[int, float, Dict[str, float]]]]:
    """Find top-K candidates for every unit in Day N.

    Returns dict mapping ref_unit_id -> list of (comp_unit_id, similarity, metrics).
    """
    candidates = {}
    for ref_uid in day1_features:
        candidates[ref_uid] = find_top_k_candidates(
            ref_uid, day1_features, day2_features, weights, k
        )
    return candidates
