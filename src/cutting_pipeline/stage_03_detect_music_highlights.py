from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .audio_features import (
    frame_metric as _frame_metric,
    merge_peak_indices as _merge_highlight_indices,
    moving_average as _moving_average,
    normalize_robust as _normalize,
    pick_peaks as _pick_peaks,
)
from .config import PipelineConfig
from .ffmpeg_tools import decode_audio_mono, get_media_duration
from .json_io import write_json
from .models import MusicHighlightRecord, MusicTrackRecord
from .progress import StageReporter


def _relative(path: Path, project_root: Path) -> str:
    return str(path.relative_to(project_root))


def _records_from_indices(
    indices: list[int],
    seconds_per_frame: float,
    scores: np.ndarray,
    energy: np.ndarray,
    accent: np.ndarray,
) -> list[MusicHighlightRecord]:
    records: list[MusicHighlightRecord] = []
    for peak_index in indices:
        records.append(
            MusicHighlightRecord(
                time=round(peak_index * seconds_per_frame, 3),
                score=round(float(scores[peak_index]), 6),
                energy=round(float(energy[peak_index]), 6),
                accent=round(float(accent[peak_index]), 6),
            )
        )
    return records


def _sparsify_emphasis_beats(
    beat_indices: list[int],
    beat_scores: np.ndarray,
    beat_threshold: float,
    base_min_distance_frames: int,
    config: PipelineConfig,
) -> list[int]:
    if not beat_indices:
        return []

    sparse_threshold = max(
        float(np.quantile(beat_scores, config.audio.beat_sparse_threshold_quantile)),
        beat_threshold,
    )
    sparse_min_distance = max(1, int(round(base_min_distance_frames * config.audio.beat_sparse_distance_scale)))

    kept: list[int] = []
    for index in sorted(beat_indices, key=lambda item: beat_scores[item], reverse=True):
        min_distance = base_min_distance_frames
        if float(beat_scores[index]) >= sparse_threshold:
            min_distance = sparse_min_distance

        if all(abs(index - existing) >= min_distance for existing in kept):
            kept.append(index)
        if len(kept) >= config.audio.beat_top_candidates:
            break

    return sorted(kept)


def _analyze_track(path: Path, config: PipelineConfig) -> MusicTrackRecord:
    samples = decode_audio_mono(path, sample_rate=config.audio.sample_rate)
    accent_samples = np.diff(samples, prepend=samples[0])

    energy = _frame_metric(samples, config.audio.frame_length, config.audio.hop_length)
    accent = _frame_metric(accent_samples, config.audio.frame_length, config.audio.hop_length)

    energy_focus = np.maximum(
        energy - _moving_average(energy, max(3, int(round(config.audio.sample_rate * 3.0 / config.audio.hop_length)))),
        0.0,
    )
    accent_focus = np.maximum(
        accent - _moving_average(accent, max(3, int(round(config.audio.sample_rate * 1.0 / config.audio.hop_length)))),
        0.0,
    )

    normalized_energy = _normalize(energy_focus).astype(np.float32)
    normalized_accent = _normalize(accent_focus).astype(np.float32)
    combined = (normalized_energy * 0.4) + (normalized_accent * 0.6)
    combined = _moving_average(combined.astype(np.float32), 5)

    threshold = max(
        float(np.quantile(combined, config.audio.peak_threshold_quantile)),
        float(combined.mean() + combined.std() * 0.7),
    )
    support_scores = _moving_average(((normalized_accent * 0.7) + (normalized_energy * 0.3)).astype(np.float32), 3)
    support_threshold = max(
        float(np.quantile(support_scores, min(config.audio.peak_threshold_quantile, 0.84))),
        float(support_scores.mean() + support_scores.std() * 0.35),
    )
    min_distance_frames = max(
        1,
        int(round(config.audio.min_peak_distance_seconds * config.audio.sample_rate / config.audio.hop_length)),
    )
    primary_indices = _pick_peaks(
        combined,
        min_distance_frames=min_distance_frames,
        limit=config.audio.top_highlights,
        threshold=threshold,
    )
    secondary_indices = _pick_peaks(
        support_scores,
        min_distance_frames=max(1, min_distance_frames // 2),
        limit=config.audio.top_highlights * 2,
        threshold=support_threshold,
    )
    peak_indices = _merge_highlight_indices(
        primary_indices=primary_indices,
        secondary_indices=secondary_indices,
        scores=np.maximum(combined, support_scores),
        min_distance_frames=min_distance_frames,
        limit=config.audio.top_highlights,
    )

    seconds_per_frame = config.audio.hop_length / config.audio.sample_rate
    highlights = _records_from_indices(
        peak_indices,
        seconds_per_frame,
        combined,
        energy,
        accent,
    )

    beat_scores = _moving_average(((normalized_accent * 0.82) + (normalized_energy * 0.18)).astype(np.float32), 3)
    beat_threshold = max(
        float(np.quantile(beat_scores, config.audio.beat_threshold_quantile)),
        float(beat_scores.mean() + beat_scores.std() * 0.12),
    )
    beat_distance_frames = max(
        1,
        int(round(config.audio.beat_min_distance_seconds * config.audio.sample_rate / config.audio.hop_length)),
    )
    beat_indices = _pick_peaks(
        beat_scores,
        min_distance_frames=beat_distance_frames,
        limit=config.audio.beat_top_candidates,
        threshold=beat_threshold,
    )
    beat_indices = _sparsify_emphasis_beats(
        beat_indices,
        beat_scores,
        beat_threshold,
        beat_distance_frames,
        config,
    )
    beats = _records_from_indices(
        beat_indices,
        seconds_per_frame,
        beat_scores,
        energy,
        accent,
    )

    return MusicTrackRecord(
        music_path=_relative(path, config.paths.project_root),
        duration=round(get_media_duration(path), 3),
        highlights=highlights,
        beats=beats,
    )


def run(config: PipelineConfig, reporter: StageReporter) -> dict:
    music_files = sorted(config.paths.music_source_dir.glob("*"))
    reporter.start(f"Analyzing {len(music_files)} music tracks.")

    tracks: list[MusicTrackRecord] = []
    for index, music_path in enumerate(music_files, start=1):
        progress = (index - 1) / max(len(music_files), 1)
        reporter.update(progress, f"Detecting highlights in {music_path.name} ({index}/{len(music_files)}).")
        tracks.append(_analyze_track(music_path, config))

    payload = {
        "stage": "stage_03_detect_music_highlights",
        "tracks": [asdict(track) for track in tracks],
    }
    output_path = config.paths.build_dir / "stage_03_music_highlights.json"
    write_json(output_path, payload)

    reporter.complete(f"Saved highlight metadata for {len(tracks)} tracks.")
    return payload
