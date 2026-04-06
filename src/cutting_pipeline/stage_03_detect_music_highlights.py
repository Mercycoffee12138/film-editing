from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio_features import (
    frame_metric as _frame_metric,
    merge_peak_indices as _merge_highlight_indices,
    moving_average as _moving_average,
    normalize_robust as _normalize,
    pick_peaks as _pick_peaks,
)
from .config import PipelineConfig
from .ffmpeg_tools import decode_audio_mono, get_media_duration, run_command
from .json_io import write_json
from .models import MusicHighlightRecord, MusicTrackRecord
from .progress import StageReporter
from .qwen_vision import analyze_images, load_config_from_env


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


def _music_review_prompt(candidate_payload: list[dict[str, Any]], config: PipelineConfig) -> str:
    candidate_lines = "\n".join(
        (
            f"{item['candidate_index']}. {item['time']:.3f}s, "
            f"规则提示={item['rule_hint']}, score={item['score']:.3f}"
        )
        for item in candidate_payload
    ) or "无候选点。"

    dense_rhythm_instruction = ""
    if config.audio.beat_ai_dense_rhythm_bias:
        dense_rhythm_instruction = (
            "如果某几个候选点能组成短促、连续、重复的节奏切点，请尽量保留到 rhythmic_group。"
            "整体风格可以明显更密一点，宁可多保留可剪辑的连续鼓点，也不要轻易漏掉。"
        )

    return (
        "你在做音乐剪辑卡点识别。"
        "第 1 张图是整首音乐的概览图，上半部分是波形，下半部分是频谱。"
        "后续图片按顺序对应候选点局部窗口图，同样上半部分是波形，下半部分是频谱。"
        "你的任务是把候选点分成两类："
        "1. single_impact：突然出现的单个重音、爆点、强击打点；"
        "2. rhythmic_group：处在一组重复重音、连续鼓点、连续节奏推进里的点。"
        "如果某个候选点不明显、不稳定、或者不适合做卡点，就不要选。"
        "优先选择清晰、干净、适合剪辑切点的候选。"
        f"{dense_rhythm_instruction}"
        "请只返回 JSON，不要输出 markdown 代码块，不要补充解释。"
        'JSON 格式必须是: {"single_impact_indices":[1,2], "rhythmic_group_indices":[3,4], "summary":"..."}。'
        "\n候选点列表:\n"
        f"{candidate_lines}"
    )


def _parse_music_review_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Music review model did not return JSON: {text}")

    parsed = json.loads(cleaned[start : end + 1])

    def _indices(key: str) -> list[int]:
        values: list[int] = []
        for value in parsed.get(key) or []:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index > 0 and index not in values:
                values.append(index)
        return values

    return {
        "single_impact_indices": _indices("single_impact_indices"),
        "rhythmic_group_indices": _indices("rhythmic_group_indices"),
        "summary": str(parsed.get("summary", "")).strip(),
    }


def _export_audio_image(path: Path, start: float, duration: float, output_path: Path, mode: str) -> None:
    if mode == "wave":
        filter_name = "showwavespic=s=1280x240:colors=white"
    else:
        filter_name = "showspectrumpic=s=1280x360:legend=disabled"

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{max(start, 0.0):.3f}",
        "-t",
        f"{max(duration, 0.1):.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-lavfi",
        filter_name,
        str(output_path),
    ]
    run_command(command)


def _stack_audio_review_image(wave_path: Path, spectrum_path: Path, output_path: Path) -> None:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for AI music review image generation. Install it with `pip install Pillow`.") from exc

    wave = Image.open(wave_path).convert("RGB")
    spectrum = Image.open(spectrum_path).convert("RGB")
    width = max(wave.width, spectrum.width)
    height = wave.height + spectrum.height
    canvas = Image.new("RGB", (width, height), color=(12, 12, 12))
    canvas.paste(wave, ((width - wave.width) // 2, 0))
    canvas.paste(spectrum, ((width - spectrum.width) // 2, wave.height))
    canvas.save(output_path)


def _build_music_review_images(
    music_path: Path,
    review_dir: Path,
    duration: float,
    candidate_payload: list[dict[str, Any]],
) -> list[Path]:
    review_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []

    overview_wave = review_dir / "overview_wave.png"
    overview_spectrum = review_dir / "overview_spectrum.png"
    overview_image = review_dir / "overview.png"
    _export_audio_image(music_path, 0.0, duration, overview_wave, "wave")
    _export_audio_image(music_path, 0.0, duration, overview_spectrum, "spectrum")
    _stack_audio_review_image(overview_wave, overview_spectrum, overview_image)
    image_paths.append(overview_image)

    for candidate in candidate_payload:
        event_time = float(candidate["time"])
        window_duration = 3.2 if candidate["rule_hint"] == "rhythmic_group" else 2.2
        start = max(0.0, min(event_time - (window_duration * 0.5), max(duration - window_duration, 0.0)))
        stem = review_dir / f"candidate_{int(candidate['candidate_index']):03d}"
        wave_path = stem.with_name(f"{stem.name}_wave.png")
        spectrum_path = stem.with_name(f"{stem.name}_spectrum.png")
        output_path = stem.with_name(f"{stem.name}.png")
        _export_audio_image(music_path, start, window_duration, wave_path, "wave")
        _export_audio_image(music_path, start, window_duration, spectrum_path, "spectrum")
        _stack_audio_review_image(wave_path, spectrum_path, output_path)
        image_paths.append(output_path)

    return image_paths


def _merge_review_candidates(
    highlights: list[MusicHighlightRecord],
    beats: list[MusicHighlightRecord],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    def _attach(record: MusicHighlightRecord, hint: str) -> None:
        event_time = round(float(record.time), 3)
        for existing in merged:
            if abs(float(existing["time"]) - event_time) <= 0.18:
                existing["score"] = max(float(existing["score"]), float(record.score))
                existing["energy"] = max(float(existing["energy"]), float(record.energy))
                existing["accent"] = max(float(existing["accent"]), float(record.accent))
                hints = set(str(existing["rule_hint"]).split("+"))
                hints.add(hint)
                existing["rule_hint"] = "+".join(sorted(hints))
                return
        merged.append(
            {
                "time": event_time,
                "score": round(float(record.score), 6),
                "energy": round(float(record.energy), 6),
                "accent": round(float(record.accent), 6),
                "rule_hint": hint,
            }
        )

    for highlight in highlights:
        _attach(highlight, "single_impact")
    for beat in beats:
        _attach(beat, "rhythmic_group")

    merged.sort(key=lambda item: (float(item["time"]), -float(item["score"])))
    for index, item in enumerate(merged, start=1):
        item["candidate_index"] = index
    return merged


def _apply_ai_music_review(
    track: MusicTrackRecord,
    music_path: Path,
    config: PipelineConfig,
) -> MusicTrackRecord:
    vision_config = load_config_from_env()
    if not vision_config:
        return track

    candidate_payload = _merge_review_candidates(track.highlights, track.beats)
    if not candidate_payload:
        return track

    review_dir = config.paths.build_dir / "stage_03_music_review" / music_path.stem
    image_paths = _build_music_review_images(
        music_path,
        review_dir,
        float(track.duration),
        candidate_payload,
    )
    response = analyze_images(
        image_paths,
        _music_review_prompt(candidate_payload, config),
        vision_config,
    )
    review = _parse_music_review_json(response["text"])

    lookup = {
        int(item["candidate_index"]): MusicHighlightRecord(
            time=round(float(item["time"]), 3),
            score=round(float(item["score"]), 6),
            energy=round(float(item["energy"]), 6),
            accent=round(float(item["accent"]), 6),
        )
        for item in candidate_payload
    }
    reviewed_highlights = [
        lookup[index]
        for index in review["single_impact_indices"]
        if index in lookup
    ]
    reviewed_beats = [
        lookup[index]
        for index in review["rhythmic_group_indices"]
        if index in lookup
    ]

    if not reviewed_highlights:
        reviewed_highlights = list(track.highlights)
    if not reviewed_beats:
        reviewed_beats = list(track.beats)

    return MusicTrackRecord(
        music_path=track.music_path,
        duration=track.duration,
        highlights=reviewed_highlights,
        beats=reviewed_beats,
    )


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


def _select_single_impact_indices(
    impact_scores: np.ndarray,
    support_scores: np.ndarray,
    config: PipelineConfig,
) -> list[int]:
    threshold = max(
        float(np.quantile(impact_scores, min(0.92, config.audio.peak_threshold_quantile + 0.08))),
        float(impact_scores.mean() + impact_scores.std() * 0.95),
    )
    support_threshold = max(
        float(np.quantile(support_scores, min(0.9, config.audio.peak_threshold_quantile + 0.05))),
        float(support_scores.mean() + support_scores.std() * 0.45),
    )
    min_distance_frames = max(
        1,
        int(round(max(config.audio.min_peak_distance_seconds * 0.75, 1.1) * config.audio.sample_rate / config.audio.hop_length)),
    )
    primary_indices = _pick_peaks(
        impact_scores,
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
    return _merge_highlight_indices(
        primary_indices=primary_indices,
        secondary_indices=secondary_indices,
        scores=np.maximum(impact_scores, support_scores),
        min_distance_frames=min_distance_frames,
        limit=config.audio.top_highlights,
    )


def _select_rhythmic_group_indices(
    beat_indices: list[int],
    beat_scores: np.ndarray,
    seconds_per_frame: float,
    config: PipelineConfig,
) -> list[int]:
    minimum_group_size = max(2, config.audio.beat_group_min_size)
    if len(beat_indices) < minimum_group_size:
        return []

    min_gap_seconds = max(
        config.audio.beat_min_distance_seconds * 0.75,
        config.audio.beat_group_min_gap_seconds,
    )
    max_gap_seconds = config.audio.beat_group_max_gap_seconds
    max_gap_delta_seconds = config.audio.beat_group_max_gap_delta_seconds
    min_group_size = minimum_group_size
    groups: list[list[int]] = []
    current_group = [beat_indices[0]]

    for index in beat_indices[1:]:
        previous_index = current_group[-1]
        gap_seconds = (index - previous_index) * seconds_per_frame
        if gap_seconds < min_gap_seconds or gap_seconds > max_gap_seconds:
            if len(current_group) >= min_group_size:
                groups.append(current_group)
            current_group = [index]
            continue

        if len(current_group) >= 2:
            previous_gap_seconds = (current_group[-1] - current_group[-2]) * seconds_per_frame
            if abs(gap_seconds - previous_gap_seconds) > max_gap_delta_seconds:
                if len(current_group) >= min_group_size:
                    groups.append(current_group)
                current_group = [previous_index, index]
                continue

        current_group.append(index)

    if len(current_group) >= min_group_size:
        groups.append(current_group)

    if not groups:
        return []

    ranked_groups = sorted(
        groups,
        key=lambda group: (
            float(np.mean([beat_scores[item] for item in group])),
            len(group),
        ),
        reverse=True,
    )

    selected: list[int] = []
    for group in ranked_groups:
        for index in group:
            if index not in selected:
                selected.append(index)
            if len(selected) >= config.audio.beat_top_candidates:
                return sorted(selected)
    return sorted(selected)


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
    seconds_per_frame = config.audio.hop_length / config.audio.sample_rate
    impact_scores = _moving_average(((normalized_accent * 0.78) + (normalized_energy * 0.22)).astype(np.float32), 5)
    impact_support_scores = _moving_average(((normalized_accent * 0.72) + (normalized_energy * 0.28)).astype(np.float32), 3)
    peak_indices = _select_single_impact_indices(
        impact_scores,
        impact_support_scores,
        config,
    )
    highlights = _records_from_indices(
        peak_indices,
        seconds_per_frame,
        impact_scores,
        energy,
        accent,
    )

    beat_scores = _moving_average(
        (
            (normalized_accent * config.audio.beat_score_accent_weight)
            + (normalized_energy * config.audio.beat_score_energy_weight)
        ).astype(np.float32),
        3,
    )
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
    beat_indices = _select_rhythmic_group_indices(
        beat_indices,
        beat_scores,
        seconds_per_frame,
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
    requested_music = config.match.selected_music_filename
    if requested_music:
        music_files = [path for path in music_files if path.name == requested_music]
        if not music_files:
            available = ", ".join(sorted(path.name for path in config.paths.music_source_dir.glob("*")))
            raise ValueError(
                f"Configured music file '{requested_music}' was not found. Available tracks: {available}"
            )
    reporter.start(f"Analyzing {len(music_files)} music tracks.")

    tracks: list[MusicTrackRecord] = []
    for index, music_path in enumerate(music_files, start=1):
        progress = (index - 1) / max(len(music_files), 1)
        reporter.update(progress, f"Detecting highlights in {music_path.name} ({index}/{len(music_files)}).")
        track = _analyze_track(music_path, config)
        reporter.update(
            min(progress + (0.5 / max(len(music_files), 1)), 0.95),
            f"Reviewing music highlight candidates in {music_path.name} with AI.",
        )
        try:
            track = _apply_ai_music_review(track, music_path, config)
        except (RuntimeError, ValueError):
            pass
        tracks.append(track)

    payload = {
        "stage": "stage_03_detect_music_highlights",
        "tracks": [asdict(track) for track in tracks],
    }
    output_path = config.paths.build_dir / "stage_03_music_highlights.json"
    write_json(output_path, payload)

    reporter.complete(f"Saved highlight metadata for {len(tracks)} tracks.")
    return payload
