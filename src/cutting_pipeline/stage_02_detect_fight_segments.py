from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import PipelineConfig
from .ffmpeg_tools import export_video_frame, iter_gray_frames
from .json_io import read_json, write_json
from .models import FightSegmentRecord
from .progress import StageReporter
from .qwen_vision import (
    QwenVisionConfig,
    QwenVisionContentBlockedError,
    analyze_images,
    load_config_from_env,
)


def _emit_progress(
    callback: Callable[[float, str], None] | None,
    fraction: float,
    message: str,
) -> None:
    if callback is not None:
        callback(max(0.0, min(1.0, fraction)), message)


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size == 0:
        return values
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(values, kernel, mode="same")


def _merge_segments(
    segments: list[tuple[int, int]],
    scores: np.ndarray,
    fps: int,
    merge_gap_seconds: float,
) -> list[tuple[int, int]]:
    del scores
    if not segments:
        return []

    merged: list[tuple[int, int]] = [segments[0]]
    for start_idx, end_idx in segments[1:]:
        previous_start, previous_end = merged[-1]
        gap_seconds = (start_idx - previous_end - 1) / fps
        if gap_seconds <= merge_gap_seconds:
            merged[-1] = (previous_start, end_idx)
        else:
            merged.append((start_idx, end_idx))
    return merged


def _relative(path: Path, project_root: Path) -> str:
    return str(path.relative_to(project_root))


def _segment_from_payload(payload: dict[str, Any]) -> FightSegmentRecord:
    return FightSegmentRecord(
        source_path=str(payload["source_path"]),
        trimmed_path=str(payload["trimmed_path"]),
        video_duration=float(payload["video_duration"]),
        start=float(payload["start"]),
        end=float(payload["end"]),
        peak_time=float(payload["peak_time"]),
        mean_motion=float(payload["mean_motion"]),
        peak_motion=float(payload["peak_motion"]),
        score=float(payload["score"]),
        confidence=float(payload.get("confidence", 0.0)),
        fight_probability=float(payload.get("fight_probability", payload.get("confidence", 0.0))),
        detection_source=str(payload.get("detection_source", "unknown")),
        key_event_times=[round(float(value), 3) for value in payload.get("key_event_times") or []],
    )


def _build_stage_payload(
    detection_mode: str,
    videos_payload: list[dict[str, Any]],
    all_segments: list[FightSegmentRecord],
    all_calm_segments: list[FightSegmentRecord],
    status: str,
) -> dict[str, Any]:
    sorted_segments = sorted(all_segments, key=lambda item: item.score, reverse=True)
    sorted_calm_segments = sorted(all_calm_segments, key=lambda item: item.score, reverse=True)
    return {
        "stage": "stage_02_detect_fight_segments",
        "status": status,
        "detection_mode": detection_mode,
        "videos": videos_payload,
        "top_segments": [asdict(segment) for segment in sorted_segments[:100]],
        "calm_segments": [asdict(segment) for segment in sorted_calm_segments[:150]],
        "segment_count": len(sorted_segments),
        "calm_segment_count": len(sorted_calm_segments),
    }


def _load_stage_checkpoint(
    output_path: Path,
    detection_mode: str,
) -> tuple[list[dict[str, Any]], list[FightSegmentRecord], list[FightSegmentRecord]]:
    if not output_path.exists():
        return [], [], []

    try:
        payload = read_json(output_path)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return [], [], []

    if payload.get("stage") != "stage_02_detect_fight_segments":
        return [], [], []
    if payload.get("status") != "in_progress":
        return [], [], []
    if payload.get("detection_mode") != detection_mode:
        return [], [], []

    videos_payload = list(payload.get("videos") or [])
    all_segments = [
        _segment_from_payload(segment)
        for video in videos_payload
        for segment in video.get("segments") or []
    ]
    all_calm_segments = [
        _segment_from_payload(segment)
        for video in videos_payload
        for segment in video.get("calm_segments") or []
    ]
    return videos_payload, all_segments, all_calm_segments


def _frame_quality_metrics(frame: np.ndarray) -> tuple[float, float, float]:
    frame_float = frame.astype(np.float32) / 255.0
    grad_x = np.abs(np.diff(frame_float, axis=1))
    grad_y = np.abs(np.diff(frame_float, axis=0))
    sharpness = float((grad_x.mean() + grad_y.mean()) * 0.5)
    contrast = float(frame_float.std())
    brightness = float(frame_float.mean())
    return sharpness, contrast, brightness


def _normalize_metric(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lower = float(np.percentile(values, 10))
    upper = float(np.percentile(values, 90))
    spread = upper - lower
    if spread <= 1e-8:
        return np.full_like(values, 0.5, dtype=np.float32)
    normalized = (values - lower) / spread
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _exposure_balance(brightness: np.ndarray) -> np.ndarray:
    if brightness.size == 0:
        return brightness
    return np.clip(1.0 - (np.abs(brightness - 0.5) / 0.5), 0.0, 1.0).astype(np.float32)


def _calm_segment_score(
    duration_ratio: float,
    calmness: float,
    stability: float,
    sharpness: float,
    contrast: float,
    exposure: float,
) -> float:
    return float(
        (duration_ratio * 0.18)
        + (calmness * 0.22)
        + (stability * 0.22)
        + (sharpness * 0.2)
        + (contrast * 0.1)
        + (exposure * 0.08)
    )


def _build_segment_record(
    start_idx: int,
    end_idx: int,
    duration: float,
    fps: int,
    scores: np.ndarray,
    score_value: float,
    confidence: float = 0.0,
    detection_source: str = "motion",
) -> FightSegmentRecord:
    start_time = start_idx / fps
    end_time = (end_idx + 1) / fps
    segment_scores = scores[start_idx : end_idx + 1]
    peak_offset = int(segment_scores.argmax())
    peak_time = (start_idx + peak_offset) / fps
    mean_motion = float(segment_scores.mean())
    peak_motion = float(segment_scores.max())
    return FightSegmentRecord(
        source_path="",
        trimmed_path="",
        video_duration=round(duration, 3),
        start=round(start_time, 3),
        end=round(min(end_time, duration), 3),
        peak_time=round(min(peak_time, duration), 3),
        mean_motion=round(mean_motion, 6),
        peak_motion=round(peak_motion, 6),
        score=round(score_value, 6),
        confidence=round(confidence, 4),
        fight_probability=round(confidence, 4),
        detection_source=detection_source,
        key_event_times=[],
    )


def _safe_timestamp(timestamp: float, duration: float) -> float:
    if duration <= 0.05:
        return 0.0
    return min(max(timestamp, 0.0), max(duration - 0.02, 0.0))


def _window_ranges(duration: float, window_seconds: float, stride_seconds: float) -> list[tuple[float, float]]:
    if duration <= 0.0:
        return []
    window_seconds = max(min(window_seconds, duration), 0.5)
    stride_seconds = max(stride_seconds, 0.25)
    if duration <= window_seconds + 1e-6:
        return [(0.0, round(duration, 3))]

    starts = [0.0]
    cursor = stride_seconds
    while cursor + window_seconds < duration:
        starts.append(cursor)
        cursor += stride_seconds

    final_start = max(duration - window_seconds, 0.0)
    if final_start - starts[-1] > 1e-6:
        starts.append(final_start)

    return [(round(start, 3), round(min(start + window_seconds, duration), 3)) for start in starts]


def _window_frame_times(start: float, end: float, frames_per_window: int) -> list[float]:
    duration = max(end - start, 0.01)
    if frames_per_window <= 1:
        return [round(start + (duration * 0.5), 3)]

    ratios = np.linspace(0.12, 0.88, num=frames_per_window)
    return [round(start + (duration * float(ratio)), 3) for ratio in ratios]


def _coarse_review_prompt(window_duration: float) -> str:
    return (
        "你在做短视频打斗片段的第一轮大跨步粗定位。"
        "我会提供同一时间窗内按时间顺序抽取的多张画面。"
        "请判断这个时间窗里是否持续或明确包含打斗、对抗、攻击、防守、武器碰撞、摔打、追打等动作冲突。"
        "如果存在，请估计打斗真正开始和结束大致位于这个时间窗的什么位置。"
        "请只返回 JSON，不要输出 markdown 代码块，不要补充解释。"
        'JSON 格式必须是: {"contains_fight": true, "confidence": 0.0, '
        '"active_start_ratio": 0.0, "active_end_ratio": 1.0, "summary": "..."}。'
        "active_start_ratio 和 active_end_ratio 是相对于当前时间窗的 0 到 1 位置。"
        f"当前时间窗时长约 {window_duration:.2f} 秒。"
    )


def _parse_coarse_review_json(text: str) -> dict[str, float | bool | str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Coarse review model did not return JSON: {text}")

    parsed = json.loads(cleaned[start : end + 1])
    start_ratio = max(0.0, min(float(parsed.get("active_start_ratio", 0.0)), 1.0))
    end_ratio = max(start_ratio, min(float(parsed.get("active_end_ratio", 1.0)), 1.0))
    return {
        "contains_fight": bool(parsed.get("contains_fight")),
        "confidence": round(float(parsed.get("confidence", 0.0)), 4),
        "active_start_ratio": round(start_ratio, 4),
        "active_end_ratio": round(end_ratio, 4),
        "summary": str(parsed.get("summary", "")).strip(),
    }


def _extract_coarse_frames(
    config: PipelineConfig,
    trimmed_path: Path,
    duration: float,
    video_index: int,
    window_index: int,
    window_start: float,
    window_end: float,
) -> list[Path]:
    frame_dir = (
        config.paths.stage_02_review_frames_dir
        / "coarse"
        / f"video_{video_index:03d}"
        / f"window_{window_index:03d}"
    )
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    for frame_index, timestamp in enumerate(
        _window_frame_times(window_start, window_end, config.fight_ai.coarse_frames_per_window),
        start=1,
    ):
        frame_path = frame_dir / f"frame_{frame_index:02d}.jpg"
        export_video_frame(trimmed_path, _safe_timestamp(timestamp, duration), frame_path)
        frame_paths.append(frame_path)
    return frame_paths


def _merge_ai_windows(
    accepted_windows: list[dict[str, float | int | str]],
    merge_gap_seconds: float,
) -> list[dict[str, float | int | str]]:
    if not accepted_windows:
        return []

    accepted_windows = sorted(accepted_windows, key=lambda item: float(item["start"]))
    merged: list[dict[str, float | int | str]] = []
    for window in accepted_windows:
        if not merged:
            merged.append(dict(window))
            continue

        previous = merged[-1]
        if float(window["start"]) <= float(previous["end"]) + merge_gap_seconds:
            previous["end"] = max(float(previous["end"]), float(window["end"]))
            previous["confidence_sum"] = float(previous["confidence_sum"]) + float(window["confidence"])
            previous["confidence_max"] = max(float(previous["confidence_max"]), float(window["confidence"]))
            previous["window_count"] = int(previous["window_count"]) + 1
            summary = str(window.get("summary", "")).strip()
            if summary:
                previous_summary = str(previous.get("summary", "")).strip()
                previous["summary"] = previous_summary or summary
        else:
            merged.append(dict(window))

    for item in merged:
        window_count = max(int(item["window_count"]), 1)
        confidence_average = float(item["confidence_sum"]) / window_count
        item["confidence"] = round((confidence_average * 0.65) + (float(item["confidence_max"]) * 0.35), 4)
    return merged


def _build_ai_segment_record(
    start: float,
    end: float,
    duration: float,
    confidence: float,
    window_count: int,
) -> FightSegmentRecord:
    start = max(0.0, min(start, duration))
    end = max(start + 0.01, min(end, duration))
    segment_duration = max(end - start, 0.01)
    peak_time = start + (segment_duration * 0.5)
    score = segment_duration * (0.65 + confidence) * (1.0 + (min(window_count, 4) * 0.08))
    return FightSegmentRecord(
        source_path="",
        trimmed_path="",
        video_duration=round(duration, 3),
        start=round(start, 3),
        end=round(end, 3),
        peak_time=round(min(peak_time, duration), 3),
        mean_motion=round(confidence, 6),
        peak_motion=round(confidence, 6),
        score=round(score, 6),
        confidence=round(confidence, 4),
        fight_probability=round(confidence, 4),
        detection_source="ai_coarse",
        key_event_times=[],
    )


def _detect_segments_with_ai(
    trimmed_path: Path,
    duration: float,
    config: PipelineConfig,
    vision_config: QwenVisionConfig,
    video_index: int,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[FightSegmentRecord], dict]:
    windows = _window_ranges(
        duration,
        config.fight_ai.coarse_window_seconds,
        config.fight_ai.coarse_stride_seconds,
    )
    accepted_windows: list[dict[str, float | int | str]] = []
    total_windows = max(len(windows), 1)

    for window_index, (window_start, window_end) in enumerate(windows, start=1):
        _emit_progress(
            progress_callback,
            (window_index - 1) / total_windows,
            f"AI coarse scan window {window_index}/{len(windows)} "
            f"({window_start:.1f}s-{window_end:.1f}s).",
        )
        frame_paths = _extract_coarse_frames(
            config,
            trimmed_path,
            duration,
            video_index,
            window_index,
            window_start,
            window_end,
        )
        response = analyze_images(frame_paths, _coarse_review_prompt(window_end - window_start), vision_config)
        review = _parse_coarse_review_json(response["text"])
        if not review["contains_fight"] or float(review["confidence"]) < config.fight_ai.coarse_min_confidence:
            continue

        active_start = window_start + ((window_end - window_start) * float(review["active_start_ratio"]))
        active_end = window_start + ((window_end - window_start) * float(review["active_end_ratio"]))
        accepted_windows.append(
            {
                "start": round(active_start, 3),
                "end": round(max(active_end, active_start + 0.1), 3),
                "confidence": float(review["confidence"]),
                "confidence_sum": float(review["confidence"]),
                "confidence_max": float(review["confidence"]),
                "window_count": 1,
                "summary": str(review["summary"]),
            }
        )
        _emit_progress(
            progress_callback,
            window_index / total_windows,
            f"AI coarse scan window {window_index}/{len(windows)} finished, "
            f"accepted {len(accepted_windows)} windows.",
        )

    merged_windows = _merge_ai_windows(accepted_windows, config.fight_ai.coarse_merge_gap_seconds)
    segments = [
        _build_ai_segment_record(
            start=float(window["start"]),
            end=float(window["end"]),
            duration=duration,
            confidence=float(window["confidence"]),
            window_count=int(window["window_count"]),
        )
        for window in merged_windows
        if float(window["end"]) - float(window["start"]) >= config.motion.min_segment_seconds
    ]
    stats = {
        "window_count": len(windows),
        "accepted_window_count": len(accepted_windows),
        "coarse_segment_count": len(segments),
        "average_confidence": round(
            float(np.mean([float(window["confidence"]) for window in merged_windows] or [0.0])),
            6,
        ),
    }
    _emit_progress(
        progress_callback,
        1.0,
        f"AI coarse scan complete, merged {len(segments)} fight regions from {len(accepted_windows)} windows.",
    )
    return segments, stats


def _detect_segments_with_motion(
    trimmed_path: Path,
    duration: float,
    config: PipelineConfig,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[FightSegmentRecord], list[FightSegmentRecord], dict]:
    motion_values: list[float] = []
    sharpness_values: list[float] = []
    contrast_values: list[float] = []
    brightness_values: list[float] = []
    previous_frame: np.ndarray | None = None
    estimated_frames = max(int(round(duration * config.motion.analysis_fps)), 1)
    progress_interval = max(estimated_frames // 10, 1)

    for frame_index, frame in enumerate(
        iter_gray_frames(
        trimmed_path,
        fps=config.motion.analysis_fps,
        width=config.motion.analysis_width,
        height=config.motion.analysis_height,
        ),
        start=1,
    ):
        current_frame = frame.astype(np.int16)
        if previous_frame is not None:
            delta = np.abs(current_frame - previous_frame)
            motion_values.append(float(delta.mean() / 255.0))
            sharpness, contrast, brightness = _frame_quality_metrics(frame)
            sharpness_values.append(sharpness)
            contrast_values.append(contrast)
            brightness_values.append(brightness)
        previous_frame = current_frame
        if frame_index == 1 or frame_index % progress_interval == 0:
            _emit_progress(
                progress_callback,
                min(frame_index / estimated_frames, 1.0),
                f"Motion scan frame {frame_index}/{estimated_frames}.",
            )

    if not motion_values:
        _emit_progress(progress_callback, 1.0, "Motion scan complete, no usable frame deltas.")
        return [], [], {"threshold": 0.0, "calm_threshold": 0.0, "frame_diffs": 0}

    raw_scores = np.asarray(motion_values, dtype=np.float32)
    raw_sharpness = np.asarray(sharpness_values, dtype=np.float32)
    raw_contrast = np.asarray(contrast_values, dtype=np.float32)
    raw_brightness = np.asarray(brightness_values, dtype=np.float32)
    smoothed_scores = _moving_average(
        raw_scores,
        max(1, int(round(config.motion.smoothing_seconds * config.motion.analysis_fps))),
    )
    normalized_sharpness = _normalize_metric(raw_sharpness)
    normalized_contrast = _normalize_metric(raw_contrast)
    exposure_scores = _exposure_balance(raw_brightness)
    threshold = max(
        float(np.quantile(smoothed_scores, config.motion.threshold_quantile)),
        float(smoothed_scores.mean() + (smoothed_scores.std() * 0.75)),
        config.motion.threshold_floor,
    )
    calm_threshold = min(
        float(np.quantile(smoothed_scores, config.motion.calm_threshold_quantile)),
        config.motion.calm_threshold_ceiling,
    )

    active_mask = smoothed_scores >= threshold
    segments: list[tuple[int, int]] = []
    active_start: int | None = None
    for index, is_active in enumerate(active_mask):
        if is_active and active_start is None:
            active_start = index
        elif not is_active and active_start is not None:
            segments.append((active_start, index - 1))
            active_start = None
    if active_start is not None:
        segments.append((active_start, len(active_mask) - 1))

    merged_segments = _merge_segments(
        segments,
        smoothed_scores,
        fps=config.motion.analysis_fps,
        merge_gap_seconds=config.motion.merge_gap_seconds,
    )

    fight_segments: list[FightSegmentRecord] = []
    for start_idx, end_idx in merged_segments:
        start_time = start_idx / config.motion.analysis_fps
        end_time = (end_idx + 1) / config.motion.analysis_fps
        segment_duration = end_time - start_time
        if segment_duration < config.motion.min_segment_seconds:
            continue

        segment_scores = smoothed_scores[start_idx : end_idx + 1]
        mean_motion = float(segment_scores.mean())
        peak_motion = float(segment_scores.max())
        score = float(mean_motion * segment_duration * (1.0 + peak_motion))

        fight_segments.append(
            _build_segment_record(
                start_idx,
                end_idx,
                duration,
                config.motion.analysis_fps,
                smoothed_scores,
                score,
                confidence=min(1.0, peak_motion),
                detection_source="motion",
            )
        )

    calm_mask = smoothed_scores <= calm_threshold
    calm_segments_raw: list[tuple[int, int]] = []
    calm_start: int | None = None
    for index, is_calm in enumerate(calm_mask):
        if is_calm and calm_start is None:
            calm_start = index
        elif not is_calm and calm_start is not None:
            calm_segments_raw.append((calm_start, index - 1))
            calm_start = None
    if calm_start is not None:
        calm_segments_raw.append((calm_start, len(calm_mask) - 1))

    merged_calm_segments = _merge_segments(
        calm_segments_raw,
        smoothed_scores,
        fps=config.motion.analysis_fps,
        merge_gap_seconds=config.motion.calm_merge_gap_seconds,
    )

    calm_segments: list[FightSegmentRecord] = []
    for start_idx, end_idx in merged_calm_segments:
        start_time = start_idx / config.motion.analysis_fps
        end_time = (end_idx + 1) / config.motion.analysis_fps
        segment_duration = end_time - start_time
        if segment_duration < config.motion.calm_min_segment_seconds:
            continue

        if segment_duration > config.motion.calm_max_segment_seconds:
            max_frames = int(round(config.motion.calm_max_segment_seconds * config.motion.analysis_fps))
            end_idx = min(end_idx, start_idx + max_frames - 1)
            end_time = (end_idx + 1) / config.motion.analysis_fps
            segment_duration = end_time - start_time

        segment_scores = smoothed_scores[start_idx : end_idx + 1]
        mean_motion = float(segment_scores.mean())
        calmness = max(0.0, calm_threshold - mean_motion) + 1e-6
        stability = max(0.0, 1.0 - (mean_motion / max(calm_threshold, 1e-6)))
        sharpness_score = float(normalized_sharpness[start_idx : end_idx + 1].mean())
        contrast_score = float(normalized_contrast[start_idx : end_idx + 1].mean())
        exposure_score = float(exposure_scores[start_idx : end_idx + 1].mean())
        duration_ratio = min(segment_duration / config.motion.calm_max_segment_seconds, 1.0)
        calmness_score = min(calmness / max(calm_threshold, 1e-6), 1.0)
        score = _calm_segment_score(
            duration_ratio=duration_ratio,
            calmness=calmness_score,
            stability=stability,
            sharpness=sharpness_score,
            contrast=contrast_score,
            exposure=exposure_score,
        )
        calm_segments.append(
            _build_segment_record(
                start_idx,
                end_idx,
                duration,
                config.motion.analysis_fps,
                smoothed_scores,
                score,
                confidence=min(1.0, stability),
                detection_source="motion_calm",
            )
        )

    _emit_progress(
        progress_callback,
        1.0,
        f"Motion scan complete, found {len(fight_segments)} fight candidates and {len(calm_segments)} calm segments.",
    )
    return fight_segments, calm_segments, {
        "threshold": round(threshold, 6),
        "calm_threshold": round(calm_threshold, 6),
        "frame_diffs": len(raw_scores),
        "average_motion": round(float(raw_scores.mean()), 6),
    }


def run(config: PipelineConfig, reporter: StageReporter, trim_manifest: dict) -> dict:
    trimmed_videos = trim_manifest["trimmed_videos"]
    vision_config = load_config_from_env()
    detection_mode = "ai_coarse" if vision_config else "motion_fallback"
    mode_label = "AI coarse scan" if vision_config else "motion fallback"
    output_path = config.paths.build_dir / "stage_02_fight_segments.json"
    reporter.start(f"Analyzing {len(trimmed_videos)} trimmed videos with {mode_label}.")

    videos_payload, all_segments, all_calm_segments = _load_stage_checkpoint(output_path, detection_mode)
    processed_videos = {str(item["trimmed_path"]): item for item in videos_payload}
    if processed_videos:
        reporter.update(
            len(processed_videos) / max(len(trimmed_videos), 1),
            f"Resuming stage_02 from checkpoint with {len(processed_videos)} completed videos.",
        )

    for index, record in enumerate(trimmed_videos, start=1):
        if record["trimmed_path"] in processed_videos:
            reporter.update(
                index / max(len(trimmed_videos), 1),
                f"Skipping {Path(record['trimmed_path']).name} ({index}/{len(trimmed_videos)}), already in checkpoint.",
            )
            continue

        total_videos = max(len(trimmed_videos), 1)
        video_base = (index - 1) / total_videos
        video_span = 1.0 / total_videos
        progress = video_base
        reporter.update(progress, f"Detecting fight segments in {Path(record['trimmed_path']).name} ({index}/{len(trimmed_videos)}).")

        trimmed_path = config.paths.project_root / record["trimmed_path"]
        video_name = Path(record["trimmed_path"]).name

        def _video_progress(local_fraction: float, message: str) -> None:
            reporter.update(
                video_base + (video_span * local_fraction),
                f"{video_name} ({index}/{len(trimmed_videos)}): {message}",
            )

        legacy_segments, calm_segments, motion_stats = _detect_segments_with_motion(
            trimmed_path,
            record["trimmed_duration"],
            config,
            progress_callback=(
                lambda fraction, message: _video_progress(
                    fraction * (0.35 if vision_config else 1.0),
                    message,
                )
            ),
        )

        if vision_config:
            try:
                segments, ai_stats = _detect_segments_with_ai(
                    trimmed_path,
                    record["trimmed_duration"],
                    config,
                    vision_config,
                    index,
                    progress_callback=lambda fraction, message: _video_progress(
                        0.35 + (fraction * 0.65),
                        message,
                    ),
                )
                stats = {
                    "detection_mode": "ai_coarse",
                    **ai_stats,
                    "calm_source": "motion",
                    "legacy_motion": motion_stats,
                }
            except QwenVisionContentBlockedError as exc:
                segments = legacy_segments
                reporter.update(
                    video_base + video_span,
                    f"{video_name} ({index}/{len(trimmed_videos)}) AI blocked by content inspection, "
                    f"using motion fallback.",
                )
                stats = {
                    "detection_mode": "content_block_motion_fallback",
                    "content_blocked": True,
                    "content_block_message": str(exc),
                    "calm_source": "motion",
                    "legacy_motion": motion_stats,
                }
        else:
            segments = legacy_segments
            stats = {
                "detection_mode": "motion_fallback",
                **motion_stats,
            }
        reporter.update(
            video_base + video_span,
            f"{video_name} ({index}/{len(trimmed_videos)}) complete: "
            f"{len(segments)} fight segments, {len(calm_segments)} calm segments.",
        )

        hydrated_segments: list[FightSegmentRecord] = []
        for segment in segments:
            hydrated = FightSegmentRecord(
                source_path=record["source_path"],
                trimmed_path=record["trimmed_path"],
                video_duration=segment.video_duration,
                start=segment.start,
                end=segment.end,
                peak_time=segment.peak_time,
                mean_motion=segment.mean_motion,
                peak_motion=segment.peak_motion,
                score=segment.score,
                confidence=segment.confidence,
                fight_probability=segment.fight_probability,
                detection_source=segment.detection_source,
                key_event_times=list(segment.key_event_times),
            )
            hydrated_segments.append(hydrated)
            all_segments.append(hydrated)

        hydrated_calm_segments: list[FightSegmentRecord] = []
        for segment in calm_segments:
            hydrated = FightSegmentRecord(
                source_path=record["source_path"],
                trimmed_path=record["trimmed_path"],
                video_duration=segment.video_duration,
                start=segment.start,
                end=segment.end,
                peak_time=segment.peak_time,
                mean_motion=segment.mean_motion,
                peak_motion=segment.peak_motion,
                score=segment.score,
                confidence=segment.confidence,
                fight_probability=segment.fight_probability,
                detection_source=segment.detection_source,
                key_event_times=list(segment.key_event_times),
            )
            hydrated_calm_segments.append(hydrated)
            all_calm_segments.append(hydrated)

        videos_payload.append(
            {
                "source_path": record["source_path"],
                "trimmed_path": record["trimmed_path"],
                "trimmed_duration": record["trimmed_duration"],
                "analysis_stats": stats,
                "segments": [asdict(segment) for segment in hydrated_segments],
                "calm_segments": [asdict(segment) for segment in hydrated_calm_segments],
            }
        )
        checkpoint_payload = _build_stage_payload(
            detection_mode=detection_mode,
            videos_payload=videos_payload,
            all_segments=all_segments,
            all_calm_segments=all_calm_segments,
            status="in_progress",
        )
        write_json(output_path, checkpoint_payload)

    payload = _build_stage_payload(
        detection_mode=detection_mode,
        videos_payload=videos_payload,
        all_segments=all_segments,
        all_calm_segments=all_calm_segments,
        status="completed",
    )
    write_json(output_path, payload)

    reporter.complete(f"Saved {payload['segment_count']} fight segments.")
    return payload
