from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .audio_features import frame_metric, merge_peak_indices, moving_average, normalize_robust, pick_peaks
from .config import PipelineConfig
from .ffmpeg_tools import concat_video_clips, decode_audio_mono, export_video_frame, iter_gray_frames, render_video_clip
from .json_io import write_json
from .progress import StageReporter
from .qwen_vision import QwenVisionContentBlockedError, analyze_images, load_config_from_env


def _review_prompt(segment_duration: float, anchor_count: int) -> str:
    return (
        "你在做打斗片段的第二轮精细定位。"
        "这个候选片段已经大概率属于打斗，但边界还不够精确。"
        "所有图片都按顺序输入。"
        f"这 {anchor_count} 张是整段候选片段从头到尾等间隔抽帧。"
        "请只基于这些画面判断："
        "1. 这段是否真的属于明确打斗；"
        "2. 真正的打斗开始和结束大致位于当前候选片段内部什么位置。"
        "请只返回 JSON，不要输出 markdown 代码块，不要补充解释。"
        'JSON 格式必须是: {"contains_fight": true, "confidence": 0.0, '
        '"refined_start_ratio": 0.0, "refined_end_ratio": 1.0, '
        '"summary": "...", "ocr_text": "..."}。'
        "refined_start_ratio 和 refined_end_ratio 是相对当前候选片段起止位置的 0 到 1。"
        f"当前候选片段时长约 {segment_duration:.2f} 秒。"
    )


def _parse_review_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Review model did not return JSON: {text}")

    parsed = json.loads(cleaned[start : end + 1])
    start_ratio = max(0.0, min(float(parsed.get("refined_start_ratio", 0.0)), 1.0))
    end_ratio = max(start_ratio, min(float(parsed.get("refined_end_ratio", 1.0)), 1.0))

    return {
        "contains_fight": bool(parsed.get("contains_fight")),
        "confidence": round(float(parsed.get("confidence", 0.0)), 4),
        "summary": str(parsed.get("summary", "")).strip(),
        "ocr_text": str(parsed.get("ocr_text", "")).strip(),
        "refined_start_ratio": round(start_ratio, 4),
        "refined_end_ratio": round(end_ratio, 4),
    }


def _segment_frame_times(segment: dict) -> list[float]:
    start = float(segment["start"])
    end = float(segment["end"])
    peak = float(segment["peak_time"])
    duration = max(end - start, 0.01)
    early = start + min(duration * 0.2, 0.5)
    late = end - min(duration * 0.2, 0.5)
    return [round(early, 3), round(peak, 3), round(max(start, late), 3)]


def _segment_anchor_times(segment: dict, anchor_count: int) -> list[float]:
    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(end - start, 0.01)
    if anchor_count <= 1:
        return [round(start + (duration * 0.5), 3)]

    begin = start + (duration * 0.08)
    finish = end - (duration * 0.08)
    if finish <= begin:
        finish = end
        begin = start
    return [round(float(value), 3) for value in np.linspace(begin, finish, num=anchor_count)]


def _clamp_timestamp(timestamp: float, lower: float, upper: float) -> float:
    return max(lower, min(timestamp, upper))


def _extract_audio_candidates(config: PipelineConfig, segment: dict) -> list[dict[str, Any]]:
    return _extract_audio_candidates_with_params(
        config,
        segment,
        min_spacing_seconds=config.fight_ai.audio_candidate_min_spacing_seconds,
        peak_quantile=config.fight_ai.audio_candidate_peak_quantile,
        limit=config.fight_ai.fine_max_event_candidates,
    )


def _extract_audio_candidates_with_params(
    config: PipelineConfig,
    segment: dict,
    min_spacing_seconds: float,
    peak_quantile: float,
    limit: int,
) -> list[dict[str, Any]]:
    trimmed_path = config.paths.project_root / segment["trimmed_path"]
    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(end - start, 0.01)

    try:
        samples = decode_audio_mono(
            trimmed_path,
            sample_rate=config.audio.sample_rate,
            start=start,
            duration=duration,
        )
    except RuntimeError:
        return []

    accent_samples = np.diff(samples, prepend=samples[0])
    energy = frame_metric(samples, config.audio.frame_length, config.audio.hop_length)
    accent = frame_metric(accent_samples, config.audio.frame_length, config.audio.hop_length)
    if energy.size == 0 or accent.size == 0:
        return []

    energy_focus = np.maximum(
        energy - moving_average(energy, max(3, int(round(config.audio.sample_rate * 0.6 / config.audio.hop_length)))),
        0.0,
    )
    accent_focus = np.maximum(
        accent - moving_average(accent, max(3, int(round(config.audio.sample_rate * 0.25 / config.audio.hop_length)))),
        0.0,
    )

    normalized_energy = normalize_robust(energy_focus).astype(np.float32)
    normalized_accent = normalize_robust(accent_focus).astype(np.float32)
    combined = moving_average(((normalized_accent * 0.72) + (normalized_energy * 0.28)).astype(np.float32), 3)
    support_scores = moving_average(normalized_accent.astype(np.float32), 3)

    threshold = max(
        float(np.quantile(combined, peak_quantile)),
        float(combined.mean() + (combined.std() * 0.25)),
    )
    support_threshold = max(
        float(np.quantile(support_scores, min(peak_quantile, 0.82))),
        float(support_scores.mean() + (support_scores.std() * 0.2)),
    )
    min_distance_frames = max(
        1,
        int(round(min_spacing_seconds * config.audio.sample_rate / config.audio.hop_length)),
    )
    primary_indices = pick_peaks(
        combined,
        min_distance_frames=min_distance_frames,
        limit=limit,
        threshold=threshold,
    )
    secondary_indices = pick_peaks(
        support_scores,
        min_distance_frames=max(1, min_distance_frames // 2),
        limit=limit * 2,
        threshold=support_threshold,
    )
    peak_indices = merge_peak_indices(
        primary_indices=primary_indices,
        secondary_indices=secondary_indices,
        scores=np.maximum(combined, support_scores),
        min_distance_frames=min_distance_frames,
        limit=limit,
    )
    if not peak_indices and combined.size:
        peak_indices = [int(combined.argmax())]

    seconds_per_frame = config.audio.hop_length / config.audio.sample_rate
    candidates: list[dict[str, Any]] = []
    for candidate_index, peak_index in enumerate(peak_indices, start=1):
        absolute_time = min(start + (peak_index * seconds_per_frame), end)
        candidates.append(
            {
                "candidate_index": candidate_index,
                "time": round(absolute_time, 3),
                "score": round(float(combined[peak_index]), 6),
                "audio_energy": round(float(energy[peak_index]), 6),
                "audio_accent": round(float(accent[peak_index]), 6),
                "candidate_source": "audio",
            }
        )
    return candidates


def _extract_refined_collision_candidates(
    config: PipelineConfig,
    segment: dict,
) -> list[dict[str, Any]]:
    audio_candidates = _extract_audio_candidates_with_params(
        config,
        segment,
        min_spacing_seconds=config.fight_ai.refined_audio_candidate_min_spacing_seconds,
        peak_quantile=config.fight_ai.refined_audio_candidate_peak_quantile,
        limit=config.fight_ai.refined_max_event_candidates,
    )
    visual_candidates = _extract_visual_collision_candidates(config, segment)
    return _merge_collision_candidates(
        audio_candidates,
        visual_candidates,
        merge_window_seconds=config.fight_ai.refined_visual_candidate_min_spacing_seconds,
        limit=config.fight_ai.refined_max_event_candidates,
    )


def _extract_visual_collision_candidates(
    config: PipelineConfig,
    segment: dict,
) -> list[dict[str, Any]]:
    trimmed_path = config.paths.project_root / segment["trimmed_path"]
    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(end - start, 0.01)

    motion_values: list[float] = []
    flash_values: list[float] = []
    previous_frame: np.ndarray | None = None

    for frame in iter_gray_frames(
        trimmed_path,
        fps=config.fight_ai.refined_visual_analysis_fps,
        width=config.motion.analysis_width,
        height=config.motion.analysis_height,
        start=start,
        duration=duration,
    ):
        current_frame = frame.astype(np.int16)
        brightness = float(frame.astype(np.float32).mean() / 255.0)
        if previous_frame is not None:
            delta = np.abs(current_frame - previous_frame)
            motion_values.append(float(delta.mean() / 255.0))
            previous_brightness = float(previous_frame.astype(np.float32).mean() / 255.0)
            flash_values.append(max(0.0, brightness - previous_brightness))
        previous_frame = current_frame

    if not motion_values:
        return []

    motion_scores = np.asarray(motion_values, dtype=np.float32)
    flash_scores = np.asarray(flash_values, dtype=np.float32)
    motion_norm = normalize_robust(motion_scores).astype(np.float32)
    flash_norm = normalize_robust(flash_scores).astype(np.float32) if flash_scores.size else np.zeros_like(motion_norm)
    combined = moving_average(((motion_norm * 0.78) + (flash_norm * 0.22)).astype(np.float32), 3)
    threshold = max(
        float(np.quantile(combined, 0.72)),
        float(combined.mean() + (combined.std() * 0.08)),
    )
    min_distance_frames = max(
        1,
        int(
            round(
                config.fight_ai.refined_visual_candidate_min_spacing_seconds
                * config.fight_ai.refined_visual_analysis_fps
            )
        ),
    )
    candidate_indices = pick_peaks(
        combined,
        min_distance_frames=min_distance_frames,
        limit=config.fight_ai.refined_max_event_candidates,
        threshold=threshold,
    )
    if not candidate_indices and combined.size:
        candidate_indices = [int(combined.argmax())]

    candidates: list[dict[str, Any]] = []
    seconds_per_frame = 1.0 / config.fight_ai.refined_visual_analysis_fps
    for candidate_index, frame_index in enumerate(candidate_indices, start=1):
        absolute_time = min(start + (frame_index * seconds_per_frame), end)
        candidates.append(
            {
                "candidate_index": candidate_index,
                "time": round(absolute_time, 3),
                "score": round(float(combined[frame_index]), 6),
                "visual_motion": round(float(motion_scores[frame_index]), 6),
                "visual_flash": round(float(flash_scores[frame_index]) if frame_index < len(flash_scores) else 0.0, 6),
                "candidate_source": "visual",
            }
        )
    return candidates


def _merge_collision_candidates(
    audio_candidates: list[dict[str, Any]],
    visual_candidates: list[dict[str, Any]],
    merge_window_seconds: float,
    limit: int,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []

    def _best_value(candidate: dict[str, Any], key: str) -> float:
        return float(candidate.get(key, 0.0))

    for candidate in sorted(
        list(audio_candidates) + list(visual_candidates),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    ):
        candidate_time = float(candidate["time"])
        attached = False
        for existing in merged:
            if abs(float(existing["time"]) - candidate_time) <= merge_window_seconds:
                existing["time"] = round((float(existing["time"]) + candidate_time) * 0.5, 3)
                existing["audio_score"] = max(_best_value(existing, "audio_score"), _best_value(candidate, "score") if candidate.get("candidate_source") != "visual" else _best_value(existing, "audio_score"))
                existing["visual_score"] = max(_best_value(existing, "visual_score"), _best_value(candidate, "score") if candidate.get("candidate_source") == "visual" else _best_value(existing, "visual_score"))
                existing["audio_energy"] = max(_best_value(existing, "audio_energy"), _best_value(candidate, "audio_energy"))
                existing["audio_accent"] = max(_best_value(existing, "audio_accent"), _best_value(candidate, "audio_accent"))
                existing["visual_motion"] = max(_best_value(existing, "visual_motion"), _best_value(candidate, "visual_motion"))
                existing["visual_flash"] = max(_best_value(existing, "visual_flash"), _best_value(candidate, "visual_flash"))
                audio_component = _best_value(existing, "audio_score")
                visual_component = _best_value(existing, "visual_score")
                flash_component = _best_value(existing, "visual_flash")
                existing["score"] = round((audio_component * 0.55) + (visual_component * 0.35) + (flash_component * 0.10), 6)
                existing["candidate_source"] = "audio_visual" if audio_component > 0.0 and visual_component > 0.0 else existing["candidate_source"]
                attached = True
                break
        if attached:
            continue

        audio_component = float(candidate.get("score", 0.0)) if candidate.get("candidate_source") != "visual" else 0.0
        visual_component = float(candidate.get("score", 0.0)) if candidate.get("candidate_source") == "visual" else 0.0
        flash_component = float(candidate.get("visual_flash", 0.0))
        merged.append(
            {
                "candidate_index": len(merged) + 1,
                "time": round(candidate_time, 3),
                "score": round((audio_component * 0.55) + (visual_component * 0.35) + (flash_component * 0.10), 6),
                "audio_score": round(audio_component, 6),
                "visual_score": round(visual_component, 6),
                "audio_energy": round(float(candidate.get("audio_energy", 0.0)), 6),
                "audio_accent": round(float(candidate.get("audio_accent", 0.0)), 6),
                "visual_motion": round(float(candidate.get("visual_motion", 0.0)), 6),
                "visual_flash": round(float(candidate.get("visual_flash", 0.0)), 6),
                "candidate_source": str(candidate.get("candidate_source", "audio")),
            }
        )

    merged.sort(key=lambda item: float(item["score"]), reverse=True)
    selected = merged[:limit]
    for index, item in enumerate(sorted(selected, key=lambda candidate: float(candidate["time"])), start=1):
        item["candidate_index"] = index
    return sorted(selected, key=lambda candidate: float(candidate["time"]))


def _extract_review_frames(
    config: PipelineConfig,
    segment: dict,
    segment_index: int,
) -> list[Path]:
    trimmed_path = config.paths.project_root / segment["trimmed_path"]
    segment_dir = config.paths.stage_02_review_frames_dir / "review" / f"segment_{segment_index:03d}"
    segment_dir.mkdir(parents=True, exist_ok=True)

    anchor_paths: list[Path] = []
    for frame_index, timestamp in enumerate(
        _segment_anchor_times(segment, config.fight_ai.fine_anchor_frames),
        start=1,
    ):
        frame_path = segment_dir / f"anchor_{frame_index:02d}.jpg"
        export_video_frame(trimmed_path, timestamp, frame_path)
        anchor_paths.append(frame_path)
    return anchor_paths


def _resolve_collision_event_times(
    collision_events: list[dict[str, Any]],
    config: PipelineConfig,
) -> list[float]:
    ranked_candidates = sorted(collision_events, key=lambda item: float(item["score"]), reverse=True)
    selected_times: list[float] = []
    for candidate in ranked_candidates:
        event_time = round(float(candidate["time"]), 3)
        if event_time in selected_times:
            continue
        selected_times.append(event_time)
        if len(selected_times) >= config.fight_ai.max_key_events_per_segment:
            break
    return sorted(selected_times)


def _refined_bounds(segment: dict, review: dict[str, Any]) -> tuple[float, float]:
    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(end - start, 0.01)
    refined_start = start + (duration * float(review["refined_start_ratio"]))
    refined_end = start + (duration * float(review["refined_end_ratio"]))
    refined_end = min(end, max(refined_end, refined_start + min(0.18, duration)))
    if refined_end - refined_start < 0.12:
        return start, end
    return round(refined_start, 3), round(refined_end, 3)


def _content_blocked_review(
    segment: dict,
    config: PipelineConfig,
    error_message: str,
) -> dict[str, Any]:
    segment_confidence = float(segment.get("confidence", 0.0))
    fallback_confidence = max(config.review.min_confidence, segment_confidence, 0.72)
    return {
        "contains_fight": True,
        "confidence": round(fallback_confidence, 4),
        "summary": "内容审核拦截，按高冲突候选片段保留。",
        "ocr_text": "",
        "refined_start_ratio": 0.0,
        "refined_end_ratio": 1.0,
        "content_blocked": True,
        "fallback_reason": error_message,
    }


def _build_reviewed_segment(
    project_root: Path,
    segment: dict,
    review: dict[str, Any],
    frame_paths: list[Path],
    config: PipelineConfig,
    accepted: bool,
) -> dict[str, Any]:
    payload = dict(segment)
    refined_start, refined_end = _refined_bounds(segment, review)
    fight_probability = min(
        1.0,
        (
            (float(review["confidence"]) * 0.6)
            + (min(float(segment.get("score", 0.0)) / 20.0, 1.0) * 0.25)
            + (0.15 if review["contains_fight"] else 0.0)
        ),
    )
    peak_time = _clamp_timestamp(float(segment["peak_time"]), refined_start, refined_end)
    if peak_time <= refined_start + 1e-6 or peak_time >= refined_end - 1e-6:
        peak_time = (refined_start + refined_end) * 0.5

    payload["start"] = refined_start
    payload["end"] = refined_end
    payload["peak_time"] = round(peak_time, 3)
    payload["confidence"] = round(float(review["confidence"]), 4)
    payload["fight_probability"] = round(fight_probability, 4)
    payload["detection_source"] = "content_block_fallback" if review.get("content_blocked") else "ai_refined"
    payload["key_event_times"] = []
    payload["mean_motion"] = round(max(float(segment.get("mean_motion", 0.0)), float(review["confidence"])), 6)
    payload["peak_motion"] = round(max(float(segment.get("peak_motion", 0.0)), float(review["confidence"])), 6)
    payload["review"] = {
        "accepted": accepted,
        "accepted_by_relaxation": False,
        "contains_fight": review["contains_fight"],
        "confidence": review["confidence"],
        "summary": review["summary"],
        "ocr_text": review["ocr_text"],
        "content_blocked": bool(review.get("content_blocked", False)),
        "fallback_reason": str(review.get("fallback_reason", "")).strip(),
        "refined_start": refined_start,
        "refined_end": refined_end,
        "ai_key_event_times": [],
        "key_event_times": [],
        "frame_paths": [str(path.relative_to(project_root)) for path in frame_paths],
    }

    base_score = max(float(segment["score"]), 0.01)
    original_duration = max(float(segment["end"]) - float(segment["start"]), 0.01)
    refined_duration = max(refined_end - refined_start, 0.01)
    duration_adjustment = 0.82 + (0.18 * min(refined_duration / original_duration, 1.0))
    if accepted:
        payload["score"] = round(base_score * duration_adjustment * (0.9 + (review["confidence"] * 0.6)), 6)
    else:
        payload["score"] = round(base_score * 0.25, 6)
    return payload


def _find_collision_event_score(segment: dict[str, Any], event_time: float) -> float:
    collision_events = (segment.get("review") or {}).get("collision_events") or []
    best_score = 0.0
    best_distance = float("inf")
    for candidate in collision_events:
        candidate_time = float(candidate.get("time", 0.0))
        distance = abs(candidate_time - event_time)
        if distance < best_distance:
            best_distance = distance
            best_score = float(candidate.get("score", 0.0))
    return round(best_score, 6)


def _collision_preview_window(segment: dict[str, Any], event_time: float, config: PipelineConfig) -> tuple[float, float]:
    segment_start = float(segment.get("start", segment.get("segment_start", 0.0)))
    segment_end = float(segment.get("end", segment.get("segment_end", segment_start)))
    lead = config.fight_ai.collision_preview_lead_seconds
    tail = config.fight_ai.collision_preview_tail_seconds
    minimum_duration = config.fight_ai.collision_preview_min_duration_seconds

    clip_start = max(segment_start, float(event_time) - lead)
    clip_end = min(segment_end, float(event_time) + tail)

    if clip_end - clip_start >= minimum_duration:
        return round(clip_start, 3), round(clip_end, 3)

    missing = minimum_duration - (clip_end - clip_start)
    extend_before = min(clip_start - segment_start, missing * 0.5)
    extend_after = min(segment_end - clip_end, missing - extend_before)
    clip_start -= extend_before
    clip_end += extend_after

    if clip_end - clip_start < minimum_duration:
        if clip_start <= segment_start + 1e-6:
            clip_end = min(segment_end, clip_start + minimum_duration)
        else:
            clip_start = max(segment_start, clip_end - minimum_duration)

    return round(clip_start, 3), round(clip_end, 3)


def _export_collision_event_previews(
    config: PipelineConfig,
    reporter: StageReporter,
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    export_dir = config.paths.build_dir / "stage_02_collision_event_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = export_dir / "clips"
    individual_dir.mkdir(parents=True, exist_ok=True)

    events: list[dict[str, Any]] = []
    for segment in segments:
        key_event_times = list(segment.get("key_event_times") or [])
        for event_time in key_event_times:
            events.append(
                {
                    "source_path": segment["source_path"],
                    "trimmed_path": segment["trimmed_path"],
                    "segment_start": round(float(segment["start"]), 3),
                    "segment_end": round(float(segment["end"]), 3),
                    "event_time": round(float(event_time), 3),
                    "score": _find_collision_event_score(segment, float(event_time)),
                    "fight_probability": round(float(segment.get("fight_probability", 0.0)), 4),
                }
            )

    events.sort(
        key=lambda item: (
            float(item["fight_probability"]),
            float(item["score"]),
            -float(item["event_time"]),
        ),
        reverse=True,
    )

    rendered_paths: list[Path] = []
    manifest_events: list[dict[str, Any]] = []
    total_events = len(events)

    for index, event in enumerate(events, start=1):
        progress = index / max(total_events + 1, 1)
        reporter.update(
            progress,
            (
                f"Exporting collision event preview {index}/{total_events} "
                f"from {Path(event['trimmed_path']).name}."
            ),
        )
        trimmed_path = config.paths.project_root / event["trimmed_path"]
        clip_start, clip_end = _collision_preview_window(event, float(event["event_time"]), config)
        output_path = individual_dir / f"event_{index:03d}.mp4"
        render_video_clip(
            trimmed_path,
            output_path,
            start_time=clip_start,
            duration=clip_end - clip_start,
            width=config.render.output_width,
            height=config.render.output_height,
            fps=config.render.output_fps,
            video_preset=config.render.video_preset,
            video_crf=config.render.video_crf,
            audio_bitrate=config.render.audio_bitrate,
        )
        rendered_paths.append(output_path)
        manifest_events.append(
            {
                "order": index,
                "source_path": event["source_path"],
                "trimmed_path": event["trimmed_path"],
                "segment_start": event["segment_start"],
                "segment_end": event["segment_end"],
                "event_time": event["event_time"],
                "event_score": event["score"],
                "fight_probability": event["fight_probability"],
                "clip_start": clip_start,
                "clip_end": clip_end,
                "duration": round(clip_end - clip_start, 3),
                "output_path": str(output_path.relative_to(config.paths.project_root)),
            }
        )

    preview_video_path: str | None = None
    if rendered_paths:
        concat_list_path = export_dir / "collision_event_concat_list.txt"
        preview_output_path = export_dir / "collision_events_preview.mp4"
        reporter.update(1.0, f"Concatenating {len(rendered_paths)} collision event previews.")
        concat_video_clips(rendered_paths, preview_output_path, concat_list_path)
        preview_video_path = str(preview_output_path.relative_to(config.paths.project_root))

    manifest_path = export_dir / "collision_events_manifest.json"
    manifest = {
        "event_count": len(manifest_events),
        "preview_video_path": preview_video_path,
        "events": manifest_events,
    }
    write_json(manifest_path, manifest)

    return {
        "event_count": len(manifest_events),
        "preview_video_path": preview_video_path,
        "manifest_path": str(manifest_path.relative_to(config.paths.project_root)),
        "events": manifest_events,
    }


def _review_rank_key(segment: dict[str, Any]) -> tuple[float, float, float, float]:
    review = segment["review"]
    contains_fight_score = 1.0 if review["contains_fight"] else 0.0
    return (
        contains_fight_score,
        float(review["confidence"]),
        float(len(segment.get("key_event_times") or [])),
        float(segment["score"]),
    )


def _apply_relaxed_acceptance(reviewed_segments: list[dict[str, Any]], config: PipelineConfig) -> list[dict[str, Any]]:
    accepted_segments = [segment for segment in reviewed_segments if segment["review"]["accepted"]]
    minimum_needed = min(config.review.min_accepted_segments, len(reviewed_segments))
    if len(accepted_segments) >= minimum_needed:
        return accepted_segments

    promoted_segments = sorted(
        [
            segment
            for segment in reviewed_segments
            if not segment["review"]["accepted"]
            and segment["review"]["contains_fight"]
            and float(segment["review"]["confidence"]) >= config.review.relaxed_min_confidence
        ],
        key=_review_rank_key,
        reverse=True,
    )

    remaining_needed = minimum_needed - len(accepted_segments)
    if len(promoted_segments) < remaining_needed:
        extra_candidates = sorted(
            [
                segment
                for segment in reviewed_segments
                if not segment["review"]["accepted"] and segment not in promoted_segments
            ],
            key=_review_rank_key,
            reverse=True,
        )
        promoted_segments.extend(extra_candidates[: remaining_needed - len(promoted_segments)])

    for segment in promoted_segments[:remaining_needed]:
        segment["review"]["accepted"] = True
        segment["review"]["accepted_by_relaxation"] = True
        segment["score"] = round(float(segment["score"]) * 2.2, 6)
        accepted_segments.append(segment)

    return accepted_segments


def run(config: PipelineConfig, reporter: StageReporter, fight_segments_payload: dict) -> dict:
    candidates = list(fight_segments_payload.get("top_segments") or [])[: config.review.max_candidate_segments]
    reporter.start(f"Refining {len(candidates)} candidate fight segments with Qwen VL.")

    vision_config = load_config_from_env() if config.review.enabled else None
    if not vision_config:
        payload = {
            "stage": "stage_02_review_fight_segments",
            "review_enabled": False,
            "candidate_count": len(candidates),
            "reviewed_count": 0,
            "accepted_count": len(candidates),
            "top_segments": candidates,
            "calm_segments": list(fight_segments_payload.get("calm_segments") or []),
            "reviewed_segments": [],
        }
        output_path = config.paths.build_dir / "stage_02_reviewed_fight_segments.json"
        write_json(output_path, payload)
        reporter.complete("Vision review skipped because no API key was configured.")
        return payload

    reviewed_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(candidates, start=1):
        progress = (index - 1) / max(len(candidates), 1)
        reporter.update(progress, f"Refining segment {index}/{len(candidates)}.")

        ordered_frames = _extract_review_frames(config, segment, index)
        try:
            response = analyze_images(
                ordered_frames,
                _review_prompt(
                    float(segment["end"]) - float(segment["start"]),
                    anchor_count=len(ordered_frames),
                ),
                vision_config,
            )
            review = _parse_review_json(response["text"])
            accepted = review["contains_fight"] and review["confidence"] >= config.review.min_confidence
        except QwenVisionContentBlockedError as exc:
            reporter.update(progress, f"Segment {index}/{len(candidates)} blocked by content inspection, using fallback.")
            review = _content_blocked_review(segment, config, str(exc))
            accepted = True
        reviewed_segments.append(
            _build_reviewed_segment(
                config.paths.project_root,
                segment,
                review,
                ordered_frames,
                config,
                accepted,
            )
        )

    accepted_segments = _apply_relaxed_acceptance(reviewed_segments, config)
    top_segments = accepted_segments or candidates
    top_segments = sorted(top_segments, key=lambda item: float(item["score"]), reverse=True)
    payload = {
        "stage": "stage_02_review_fight_segments",
        "review_enabled": True,
        "candidate_count": len(candidates),
        "reviewed_count": len(reviewed_segments),
        "accepted_count": len(accepted_segments),
        "relaxed_acceptance_applied": any(
            segment["review"]["accepted_by_relaxation"] for segment in reviewed_segments
        ),
        "top_segments": top_segments,
        "calm_segments": list(fight_segments_payload.get("calm_segments") or []),
        "reviewed_segments": reviewed_segments,
    }
    output_path = config.paths.build_dir / "stage_02_reviewed_fight_segments.json"
    write_json(output_path, payload)

    reporter.complete(f"Accepted {len(accepted_segments)} of {len(candidates)} candidate segments.")
    return payload
