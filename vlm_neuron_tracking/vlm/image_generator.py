import io
import base64
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from spikeinterface.core.template_tools import get_template_extremum_channel
from typing import Dict, List, Optional, Tuple


def _get_waveform_template_data(unit_id, we, sampling_frequency, channel_ids):
    """Extract waveform template data for a unit. Returns (time_axis, mean, std)."""
    mean_wf = we.get_template(unit_id=unit_id)

    unit_ids_list = list(we.sorting.get_unit_ids())
    unit_index = unit_ids_list.index(unit_id)
    all_std = we.get_all_templates(mode='std')
    std_wf = all_std[unit_index]

    peak_ch = get_template_extremum_channel(we, peak_sign='neg')[unit_id]
    ch_idx = list(channel_ids).index(peak_ch)

    n_samples = mean_wf.shape[0]
    time_axis = np.arange(n_samples) / sampling_frequency * 1000  # ms

    return time_axis, mean_wf[:, ch_idx], std_wf[:, ch_idx]


def _plot_waveform_template(ax, time_axis, mean, std, ylim=None, color='black'):
    """Plot mean waveform on the peak channel with std shading."""
    ax.fill_between(time_axis, mean - std, mean + std, alpha=0.2, color='gray')
    ax.plot(time_axis, mean, color=color, lw=1.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel('Time (ms)', fontsize=7)
    ax.set_ylabel('Amp (µV)', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(f'Waveform Template', fontsize=8)


def _build_channel_to_position_map(channel_indices, positions, shank_ids):
    """Build a mapping from recording channel ID to (position, shank_id)."""
    ch_to_pos = {}
    ch_to_shank = {}
    for i, ch in enumerate(channel_indices):
        ch_to_pos[ch] = np.array(positions[i])
        ch_to_shank[ch] = int(shank_ids[i])
    return ch_to_pos, ch_to_shank


def _get_unit_shank_and_channels(unit_id, we, channel_ids, channel_indices, shank_ids):
    """Get the shank ID and shank channel list for a unit based on its peak channel."""
    template = we.get_template(unit_id)
    peak_amplitudes = np.max(np.abs(template), axis=0)
    peak_ch = list(channel_ids)[np.argmax(peak_amplitudes)]

    _, ch_to_shank = _build_channel_to_position_map(
        channel_indices, [[0, 0]] * len(channel_indices), shank_ids)

    if peak_ch in ch_to_shank:
        shank_id = ch_to_shank[peak_ch]
    else:
        ext_chs = get_template_extremum_channel(we, peak_sign='neg')
        peak_ch = ext_chs[unit_id]
        shank_id = ch_to_shank.get(peak_ch, 0)

    shank_channels = [channel_indices[i] for i in range(len(channel_indices))
                      if int(shank_ids[i]) == shank_id]

    return peak_ch, shank_id, shank_channels


def _get_multichannel_max_amp(unit_id, we, channel_ids, channel_indices, shank_ids):
    """Get the max amplitude across shank channels for a unit (for shared scaling)."""
    ch_idx_map = {ch: i for i, ch in enumerate(channel_ids)}

    template = we.get_template(unit_id)
    peak_amplitudes = np.max(np.abs(template), axis=0)

    peak_ch, shank_id, shank_channels = _get_unit_shank_and_channels(
        unit_id, we, channel_ids, channel_indices, shank_ids)

    max_amp = 0
    for ch in shank_channels:
        if ch in ch_idx_map:
            max_amp = max(max_amp, peak_amplitudes[ch_idx_map[ch]])
    return max_amp


def _plot_multichannel_waveforms(ax, unit_id, we, sampling_frequency, channel_ids,
                                 channel_indices, positions, shank_ids,
                                 shared_spacing: Optional[float] = None):
    """Plot waveform templates across the 8 channels on the same shank."""
    mean_wf = we.get_template(unit_id=unit_id)

    ch_idx_map = {ch: i for i, ch in enumerate(channel_ids)}

    peak_ch, shank_id, shank_channels = _get_unit_shank_and_channels(
        unit_id, we, channel_ids, channel_indices, shank_ids)

    n_samples = mean_wf.shape[0]
    t = np.arange(n_samples) / sampling_frequency * 1000

    valid_channels = [ch for ch in shank_channels if ch in ch_idx_map]

    if len(valid_channels) == 0:
        ax.text(0.5, 0.5, 'No channels', transform=ax.transAxes, ha='center', fontsize=7)
        ax.axis('off')
        return

    if shared_spacing is not None:
        spacing = shared_spacing
    else:
        max_amp = 0
        for ch in valid_channels:
            max_amp = max(max_amp, np.max(np.abs(mean_wf[:, ch_idx_map[ch]])))
        spacing = max_amp * 2.5 if max_amp > 0 else 1.0

    for idx, ch in enumerate(valid_channels):
        offset = -idx * spacing
        wf = mean_wf[:, ch_idx_map[ch]]
        ax.plot(t, wf + offset, color='black', lw=0.8)

        label = f'Ch{ch}'
        if ch == peak_ch:
            label += '*'
        ax.text(-0.3, offset, label, fontsize=5, va='center', ha='right')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Multi-Channel Waveforms', fontsize=8)


def _plot_spike_location(ax, unit_id, we, channel_ids,
                         channel_indices, positions, shank_ids,
                         firing_rate: float):
    """Plot probe schematic with spike location highlighted, plus firing rate text."""
    ch_idx_map = {ch: i for i, ch in enumerate(channel_ids)}
    ch_to_pos, ch_to_shank = _build_channel_to_position_map(
        channel_indices, positions, shank_ids)

    template = we.get_template(unit_id)
    peak_amplitudes = np.max(np.abs(template), axis=0)

    peak_ch, shank_id, shank_channels = _get_unit_shank_and_channels(
        unit_id, we, channel_ids, channel_indices, shank_ids)

    # Plot all channels on the shank as small gray dots
    shank_positions = np.array([ch_to_pos[ch] for ch in shank_channels if ch in ch_to_pos])
    ax.scatter(shank_positions[:, 0], shank_positions[:, 1],
               c='lightgray', s=30, edgecolors='gray', linewidths=0.5, zorder=1)

    # Compute center-of-mass spike location (amplitude-weighted across shank channels)
    com_weights = []
    com_positions = []
    for ch in shank_channels:
        if ch in ch_idx_map and ch in ch_to_pos:
            amp = peak_amplitudes[ch_idx_map[ch]]
            com_weights.append(amp)
            com_positions.append(ch_to_pos[ch])

    if len(com_weights) > 0 and sum(com_weights) > 0:
        com_weights = np.array(com_weights)
        com_positions = np.array(com_positions)
        com = np.average(com_positions, weights=com_weights, axis=0)
    else:
        com = ch_to_pos.get(peak_ch, np.array([0, 0]))

    ax.scatter([com[0]], [com[1]],
               c='red', s=120, edgecolors='darkred', linewidths=1.5, zorder=3, marker='*')

    ax.set_aspect('equal')
    ax.set_title('Spike Location', fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.5, -0.15, f'Firing Rate: {firing_rate:.1f} Hz',
            transform=ax.transAxes, ha='center', fontsize=8, fontweight='bold')


def _compute_firing_rate(sorting, unit_id, sampling_frequency):
    """Compute firing rate for a unit."""
    spike_train = sorting.get_unit_spike_train(unit_id)
    if len(spike_train) > 1:
        dur = (spike_train[-1] - spike_train[0]) / sampling_frequency
        return len(spike_train) / dur if dur > 0 else 0.0
    return 0.0


def generate_comparison_image(ref_unit_id: int,
                              comp_unit_id: int,
                              ref_we, comp_we,
                              ref_sorting, comp_sorting,
                              ref_sampling_frequency: float,
                              comp_sampling_frequency: float,
                              ref_channel_ids: List[int],
                              comp_channel_ids: List[int],
                              channel_indices: List[int],
                              positions: List[List[float]],
                              shank_ids: List[str],
                              ref_day_label: str = 'Day N',
                              comp_day_label: str = 'Day N+1',
                              ref_firing_rate: Optional[float] = None,
                              comp_firing_rate: Optional[float] = None,
                              dpi: int = 200) -> str:
    """Generate a composite comparison image for a unit pair.

    Layout: 3 rows x 2 columns
        Row 1: Waveform template (peak channel)
        Row 2: Multi-channel waveforms (8ch on same shank)
        Row 3: Spike location on probe + firing rate

    Returns base64-encoded PNG string.
    """
    if ref_firing_rate is None:
        ref_firing_rate = _compute_firing_rate(
            ref_sorting, ref_unit_id, ref_sampling_frequency)
    if comp_firing_rate is None:
        comp_firing_rate = _compute_firing_rate(
            comp_sorting, comp_unit_id, comp_sampling_frequency)

    fig = plt.figure(figsize=(10, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           width_ratios=[1, 1],
                           height_ratios=[1, 1.2, 1],
                           hspace=0.35, wspace=0.3)
    gs_mc = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :],
                                             width_ratios=[1, 1], wspace=0.5)

    ax_t1 = fig.add_subplot(gs[0, 0])
    ax_t2 = fig.add_subplot(gs[0, 1])
    ax_mc1 = fig.add_subplot(gs_mc[0, 0])
    ax_mc2 = fig.add_subplot(gs_mc[0, 1])
    ax_loc1 = fig.add_subplot(gs[2, 0])
    ax_loc2 = fig.add_subplot(gs[2, 1])

    # Row 1: Waveform templates
    ref_t, ref_mean, ref_std = _get_waveform_template_data(
        ref_unit_id, ref_we, ref_sampling_frequency, ref_channel_ids)
    comp_t, comp_mean, comp_std = _get_waveform_template_data(
        comp_unit_id, comp_we, comp_sampling_frequency, comp_channel_ids)

    shared_lo = min(np.min(ref_mean - ref_std), np.min(comp_mean - comp_std))
    shared_hi = max(np.max(ref_mean + ref_std), np.max(comp_mean + comp_std))
    margin = (shared_hi - shared_lo) * 0.1
    shared_ylim = (shared_lo - margin, shared_hi + margin)

    ax_t1.set_title(f'{ref_day_label} (Unit {ref_unit_id})\nWaveform Template', fontsize=9)
    ax_t2.set_title(f'{comp_day_label} (Unit {comp_unit_id})\nWaveform Template', fontsize=9)
    _plot_waveform_template(ax_t1, ref_t, ref_mean, ref_std, ylim=shared_ylim)
    _plot_waveform_template(ax_t2, comp_t, comp_mean, comp_std, ylim=shared_ylim)

    # Row 2: Multi-channel waveforms
    for ax_mc in [ax_mc1, ax_mc2]:
        pos = ax_mc.get_position()
        shrink = 0.7
        new_w = pos.width * shrink
        offset = (pos.width - new_w) / 2
        ax_mc.set_position([pos.x0 + offset, pos.y0, new_w, pos.height])

    ref_max_amp = _get_multichannel_max_amp(
        ref_unit_id, ref_we, ref_channel_ids, channel_indices, shank_ids)
    comp_max_amp = _get_multichannel_max_amp(
        comp_unit_id, comp_we, comp_channel_ids, channel_indices, shank_ids)
    shared_mc_spacing = max(ref_max_amp, comp_max_amp) * 2.5
    if shared_mc_spacing == 0:
        shared_mc_spacing = 1.0

    _plot_multichannel_waveforms(ax_mc1, ref_unit_id, ref_we, ref_sampling_frequency,
                                ref_channel_ids, channel_indices, positions, shank_ids,
                                shared_spacing=shared_mc_spacing)
    _plot_multichannel_waveforms(ax_mc2, comp_unit_id, comp_we, comp_sampling_frequency,
                                comp_channel_ids, channel_indices, positions, shank_ids,
                                shared_spacing=shared_mc_spacing)

    # Row 3: Spike location
    _plot_spike_location(ax_loc1, ref_unit_id, ref_we, ref_channel_ids,
                         channel_indices, positions, shank_ids, ref_firing_rate)
    _plot_spike_location(ax_loc2, comp_unit_id, comp_we, comp_channel_ids,
                         channel_indices, positions, shank_ids, comp_firing_rate)

    fig.suptitle(f'Alignment Comparison: {ref_day_label} Unit {ref_unit_id} vs '
                 f'{comp_day_label} Unit {comp_unit_id}',
                 fontsize=11, fontweight='bold', y=0.98)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def generate_comparative_image(ref_unit_id: int,
                                candidate_unit_ids: List[int],
                                ref_we, comp_we,
                                ref_sorting, comp_sorting,
                                ref_sampling_frequency: float,
                                comp_sampling_frequency: float,
                                ref_channel_ids: List[int],
                                comp_channel_ids: List[int],
                                channel_indices: List[int],
                                positions: List[List[float]],
                                shank_ids: List[str],
                                ref_day_label: str = 'Day N',
                                comp_day_label: str = 'Day N+1',
                                ref_firing_rate: Optional[float] = None,
                                comp_firing_rates: Optional[Dict[int, float]] = None,
                                candidate_metrics: Optional[Dict[int, Dict[str, float]]] = None,
                                dpi: int = 200) -> str:
    """Generate a comparative image: 1 reference vs K candidates side-by-side.

    Layout: 4 rows x (1 + K) columns
        Col 0: Reference unit
        Col 1..K: Candidate units (labeled A, B, C, ...)
        Row 1: Waveform template (shared y-axis)
        Row 2: Multi-channel waveforms (shared spacing)
        Row 3: Spike location + firing rate
        Row 4: Waveform overlay (ref blue, candidate red)

    Returns base64-encoded PNG string.
    """
    K = len(candidate_unit_ids)
    cand_labels = [chr(ord('A') + i) for i in range(K)]

    if ref_firing_rate is None:
        ref_firing_rate = _compute_firing_rate(
            ref_sorting, ref_unit_id, ref_sampling_frequency)

    if comp_firing_rates is None:
        comp_firing_rates = {}
    for cuid in candidate_unit_ids:
        if cuid not in comp_firing_rates:
            comp_firing_rates[cuid] = _compute_firing_rate(
                comp_sorting, cuid, comp_sampling_frequency)

    n_cols = 1 + K
    fig = plt.figure(figsize=(4 * n_cols, 14))
    gs = gridspec.GridSpec(4, n_cols, figure=fig,
                           width_ratios=[1] * n_cols,
                           height_ratios=[1, 1.2, 1, 1],
                           hspace=0.35, wspace=0.3)

    # Gather waveform data
    ref_t, ref_mean, ref_std = _get_waveform_template_data(
        ref_unit_id, ref_we, ref_sampling_frequency, ref_channel_ids)
    cand_data = {}
    for cuid in candidate_unit_ids:
        cand_data[cuid] = _get_waveform_template_data(
            cuid, comp_we, comp_sampling_frequency, comp_channel_ids)

    # Shared y-limits
    all_lo = [np.min(ref_mean - ref_std)]
    all_hi = [np.max(ref_mean + ref_std)]
    for cuid in candidate_unit_ids:
        ct, cm, cs = cand_data[cuid]
        all_lo.append(np.min(cm - cs))
        all_hi.append(np.max(cm + cs))
    shared_lo = min(all_lo)
    shared_hi = max(all_hi)
    margin = (shared_hi - shared_lo) * 0.1
    shared_ylim = (shared_lo - margin, shared_hi + margin)

    # Shared multi-channel spacing
    all_max_amps = [_get_multichannel_max_amp(
        ref_unit_id, ref_we, ref_channel_ids, channel_indices, shank_ids)]
    for cuid in candidate_unit_ids:
        all_max_amps.append(_get_multichannel_max_amp(
            cuid, comp_we, comp_channel_ids, channel_indices, shank_ids))
    shared_mc_spacing = max(all_max_amps) * 2.5
    if shared_mc_spacing == 0:
        shared_mc_spacing = 1.0

    # Row 1: Waveform templates
    ax_ref_t = fig.add_subplot(gs[0, 0])
    ax_ref_t.set_title(f'Reference\n{ref_day_label} Unit {ref_unit_id}', fontsize=9,
                       fontweight='bold', color='blue')
    _plot_waveform_template(ax_ref_t, ref_t, ref_mean, ref_std, ylim=shared_ylim, color='blue')

    for i, cuid in enumerate(candidate_unit_ids):
        ax = fig.add_subplot(gs[0, 1 + i])
        ct, cm, cs = cand_data[cuid]
        ax.set_title(f'Candidate {cand_labels[i]}\n{comp_day_label} Unit {cuid}', fontsize=9,
                     fontweight='bold', color='red')
        _plot_waveform_template(ax, ct, cm, cs, ylim=shared_ylim, color='red')

    # Row 2: Multi-channel waveforms
    ax_ref_mc = fig.add_subplot(gs[1, 0])
    _plot_multichannel_waveforms(ax_ref_mc, ref_unit_id, ref_we, ref_sampling_frequency,
                                 ref_channel_ids, channel_indices, positions, shank_ids,
                                 shared_spacing=shared_mc_spacing)

    for i, cuid in enumerate(candidate_unit_ids):
        ax = fig.add_subplot(gs[1, 1 + i])
        _plot_multichannel_waveforms(ax, cuid, comp_we, comp_sampling_frequency,
                                     comp_channel_ids, channel_indices, positions, shank_ids,
                                     shared_spacing=shared_mc_spacing)

    # Row 3: Spike location
    ax_ref_loc = fig.add_subplot(gs[2, 0])
    _plot_spike_location(ax_ref_loc, ref_unit_id, ref_we, ref_channel_ids,
                         channel_indices, positions, shank_ids, ref_firing_rate)

    for i, cuid in enumerate(candidate_unit_ids):
        ax = fig.add_subplot(gs[2, 1 + i])
        _plot_spike_location(ax, cuid, comp_we, comp_channel_ids,
                             channel_indices, positions, shank_ids,
                             comp_firing_rates[cuid])

    # Row 4: Overlay
    for i, cuid in enumerate(candidate_unit_ids):
        ax = fig.add_subplot(gs[3, 1 + i])
        ct, cm, cs = cand_data[cuid]
        t_len = min(len(ref_t), len(ct))
        ax.fill_between(ref_t[:t_len], (ref_mean - ref_std)[:t_len],
                         (ref_mean + ref_std)[:t_len], alpha=0.15, color='blue')
        ax.plot(ref_t[:t_len], ref_mean[:t_len], color='blue', lw=1.5,
                label=f'Ref (Unit {ref_unit_id})')
        ax.fill_between(ct[:t_len], (cm - cs)[:t_len], (cm + cs)[:t_len],
                         alpha=0.15, color='red')
        ax.plot(ct[:t_len], cm[:t_len], color='red', lw=1.5, ls='--',
                label=f'Cand {cand_labels[i]} (Unit {cuid})')
        ax.set_ylim(shared_ylim)
        ax.set_xlabel('Time (ms)', fontsize=7)
        ax.set_ylabel('Amp (µV)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='lower right')
        ax.set_title(f'Overlay: Ref vs {cand_labels[i]}', fontsize=8)

        if candidate_metrics and cuid in candidate_metrics:
            m = candidate_metrics[cuid]
            metrics_text = (f"wf_corr={m.get('waveform', 0):.2f}  "
                            f"spat={m.get('spatial', 0):.2f}  "
                            f"amp={m.get('amplitude', 0):.2f}  "
                            f"fr={m.get('firing_rate', 0):.2f}")
            ax.text(0.5, -0.18, metrics_text, transform=ax.transAxes,
                    ha='center', fontsize=6, color='gray')

    ax_empty = fig.add_subplot(gs[3, 0])
    ax_empty.text(0.5, 0.5, 'Overlay panels →\n(Ref blue, Cand red)',
                  transform=ax_empty.transAxes, ha='center', va='center',
                  fontsize=9, color='gray')
    ax_empty.axis('off')

    fig.suptitle(f'Comparative Alignment: {ref_day_label} Unit {ref_unit_id} vs '
                 f'{K} Candidates from {comp_day_label}',
                 fontsize=12, fontweight='bold', y=0.99)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')
