from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .ffmpeg_tools import export_video_frame
from .json_io import write_json
from .progress import StageReporter
from .qwen_vision import QwenVisionContentBlockedError, analyze_images, load_config_from_env
from .stage_02_review_fight_segments import (
    _clamp_timestamp,
    _export_collision_event_previews,
    _extract_audio_candidates,
    _segment_anchor_times,
)


def _collision_review_prompt(segment_duration: float, anchor_count: int, candidate_events: list[dict[str, Any]]) -> str:
    candidate_lines = "\n".join(
        (
            f"{candidate['candidate_index']}. {float(candidate['time']):.3f}s, "
            f"综合分={float(candidate.get('score', 0.0)):.3f}, "
            f"audio={float(candidate.get('audio_score', 0.0)):.3f}, "
            f"visual={float(candidate.get('visual_score', 0.0)):.3f}, "
            f"flash={float(candidate.get('visual_flash', 0.0)):.3f}"
        )
        for candidate in candidate_events
    ) or "无候选碰撞点。"

    return (
        "你在做打斗片段的碰撞点精确识别。"
        "这个片段已经确认是打斗，现在要从音频先找到的候选点里挑出真正适合卡点的瞬间。"
        "所有图片都按顺序输入。"
        f"前 {anchor_count} 张是整段打斗片段的锚点抽帧。"
        "后续每 3 张为一个候选点的 前一瞬间 / 当前瞬间 / 后一瞬间。"
        "请判断这些候选点对应的画面，是否真的属于以下任一情况："
        "1. 人物正在用手脚打斗并且发生了有效击中或接触；"
        "2. 武器之间或武器与人体/物体发生了明确碰撞；"
        "3. 正在运用功法、招式、法术、气劲，并且这一瞬间是明显的释放、命中、对撞或爆开时刻。"
        "只选择真正发生动作接触或功法爆发的瞬间。"
        "不要选择蓄力、起手、挥空、普通运镜、镜头晃动、纯闪光、切镜头。"
        "请只返回 JSON，不要输出 markdown 代码块，不要补充解释。"
        'JSON 格式必须是: {"primary_candidate_index": 0, "selected_candidate_indices": [1, 3], "summary": "..."}。'
        "primary_candidate_index 是最精准、最值得卡点的那个候选编号，没有就填 0。"
        f"当前片段时长约 {segment_duration:.2f} 秒。"
        "\n候选点列表:\n"
        f"{candidate_lines}"
    )


def _parse_collision_review_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Collision review model did not return JSON: {text}")

    parsed = json.loads(cleaned[start : end + 1])
    selected_candidate_indices: list[int] = []
    for value in parsed.get("selected_candidate_indices") or []:
        try:
            candidate_index = int(value)
        except (TypeError, ValueError):
            continue
        if candidate_index > 0 and candidate_index not in selected_candidate_indices:
            selected_candidate_indices.append(candidate_index)

    try:
        primary_candidate_index = int(parsed.get("primary_candidate_index", 0))
    except (TypeError, ValueError):
        primary_candidate_index = 0

    return {
        "primary_candidate_index": max(0, primary_candidate_index),
        "selected_candidate_indices": selected_candidate_indices,
        "summary": str(parsed.get("summary", "")).strip(),
    }


def _extract_collision_review_frames(
    config: PipelineConfig,
    segment: dict,
    segment_index: int,
    candidate_events: list[dict[str, Any]],
) -> list[Path]:
    trimmed_path = config.paths.project_root / segment["trimmed_path"]
    segment_dir = config.paths.stage_02_review_frames_dir / "collision_review" / f"segment_{segment_index:03d}"
    segment_dir.mkdir(parents=True, exist_ok=True)

    start = float(segment["start"])
    end = float(segment["end"])
    frame_paths: list[Path] = []
    for frame_index, timestamp in enumerate(
        _segment_anchor_times(segment, config.fight_ai.fine_anchor_frames),
        start=1,
    ):
        frame_path = segment_dir / f"anchor_{frame_index:02d}.jpg"
        export_video_frame(trimmed_path, timestamp, frame_path)
        frame_paths.append(frame_path)

    event_context = config.fight_ai.fine_event_context_seconds
    for candidate in candidate_events:
        candidate_index = int(candidate["candidate_index"])
        base_time = float(candidate["time"])
        for suffix, offset in (("before", -event_context), ("hit", 0.0), ("after", event_context)):
            timestamp = _clamp_timestamp(base_time + offset, start, end)
            frame_path = segment_dir / f"event_{candidate_index:02d}_{suffix}.jpg"
            export_video_frame(trimmed_path, timestamp, frame_path)
            frame_paths.append(frame_path)
    return frame_paths


def _resolve_ai_collision_times(
    candidate_events: list[dict[str, Any]],
    review: dict[str, Any],
    config: PipelineConfig,
) -> list[float]:
    candidate_lookup = {
        int(candidate["candidate_index"]): round(float(candidate["time"]), 3)
        for candidate in candidate_events
    }
    selected_times: list[float] = []

    primary_index = int(review.get("primary_candidate_index", 0))
    if primary_index > 0 and primary_index in candidate_lookup:
        selected_times.append(candidate_lookup[primary_index])

    for candidate_index in review.get("selected_candidate_indices") or []:
        event_time = candidate_lookup.get(int(candidate_index))
        if event_time is None or event_time in selected_times:
            continue
        selected_times.append(event_time)
        if len(selected_times) >= config.fight_ai.max_key_events_per_segment:
            break

    return sorted(selected_times)


def _enrich_segment_with_collision_events(config: PipelineConfig, segment: dict) -> dict:
    payload = dict(segment)
    # Candidate discovery is intentionally audio-first; the vision model only verifies
    # whether each audio hit aligns with a true fight impact / weapon clash / skill burst.
    collision_events = _extract_audio_candidates(config, payload)
    review = dict(payload.get("review") or {})
    review["collision_events"] = collision_events
    review["collision_ai_summary"] = ""
    review["key_event_times"] = []
    payload["review"] = review
    payload["key_event_times"] = []
    return payload


def _select_diversified_key_event_times(
    segments: list[dict],
    config: PipelineConfig,
) -> list[dict]:
    candidate_lists: list[list[dict]] = []
    for segment in segments:
        ranked_candidates = sorted(
            list((segment.get("review") or {}).get("collision_events") or []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        preferred_times = [round(float(value), 3) for value in (segment.get("key_event_times") or [])]
        if preferred_times:
            preferred_candidates = [
                candidate
                for candidate in ranked_candidates
                if round(float(candidate.get("time", 0.0)), 3) in preferred_times
            ]
            remaining_candidates = [
                candidate
                for candidate in ranked_candidates
                if round(float(candidate.get("time", 0.0)), 3) not in preferred_times
            ]
            ranked_candidates = preferred_candidates + remaining_candidates
        candidate_lists.append(ranked_candidates[: config.fight_ai.max_key_events_per_segment])

    selected_times_by_segment: list[list[float]] = [[] for _ in segments]
    pass_index = 0
    while pass_index < config.fight_ai.max_key_events_per_segment:
        any_selected = False
        for segment_index, candidates in enumerate(candidate_lists):
            if pass_index >= len(candidates):
                continue

            candidate = candidates[pass_index]
            candidate_time = round(float(candidate["time"]), 3)
            candidate_score = float(candidate.get("score", 0.0))
            existing_times = selected_times_by_segment[segment_index]
            if candidate_time in existing_times:
                continue

            if pass_index > 0 and candidates:
                best_score = float(candidates[0].get("score", 0.0))
                minimum_repeat_score = best_score * config.fight_ai.collision_repeat_score_ratio
                if candidate_score < minimum_repeat_score:
                    continue

            existing_times.append(candidate_time)
            any_selected = True

        if not any_selected:
            break
        pass_index += 1

    enriched_segments: list[dict] = []
    for segment, selected_times in zip(segments, selected_times_by_segment):
        payload = dict(segment)
        review = dict(payload.get("review") or {})
        review["key_event_times"] = sorted(selected_times)
        payload["review"] = review
        payload["key_event_times"] = sorted(selected_times)
        enriched_segments.append(payload)
    return enriched_segments


def run(
    config: PipelineConfig,
    reporter: StageReporter,
    reviewed_fight_segments_payload: dict,
) -> dict:
    top_segments = list(reviewed_fight_segments_payload.get("top_segments") or [])
    vision_config = load_config_from_env()
    reporter.start(f"Extracting collision events from {len(top_segments)} refined fight segments.")

    enriched_segments: list[dict] = []
    segment_total = max(len(top_segments), 1)
    for index, segment in enumerate(top_segments, start=1):
        def _segment_progress(phase: float, message: str) -> None:
            overall = ((index - 1) + max(0.0, min(phase, 1.0))) / segment_total
            reporter.update(overall, message)

        _segment_progress(0.0, f"Segment {index}/{len(top_segments)}: extracting rule-based collision candidates.")
        enriched = _enrich_segment_with_collision_events(config, segment)
        candidate_events = list((enriched.get("review") or {}).get("collision_events") or [])
        _segment_progress(
            0.25,
            (
                f"Segment {index}/{len(top_segments)}: found {len(candidate_events)} collision candidates "
                f"in {Path(segment['trimmed_path']).name}."
            ),
        )
        if vision_config and candidate_events:
            _segment_progress(
                0.45,
                f"Segment {index}/{len(top_segments)}: exporting {len(candidate_events)} candidate event frame groups for AI review.",
            )
            frame_paths = _extract_collision_review_frames(config, enriched, index, candidate_events)
            try:
                _segment_progress(
                    0.65,
                    f"Segment {index}/{len(top_segments)}: waiting for AI collision review on {len(frame_paths)} frames.",
                )
                response = analyze_images(
                    frame_paths,
                    _collision_review_prompt(
                        float(enriched["end"]) - float(enriched["start"]),
                        anchor_count=config.fight_ai.fine_anchor_frames,
                        candidate_events=candidate_events,
                    ),
                    vision_config,
                )
                ai_review = _parse_collision_review_json(response["text"])
                ai_key_event_times = _resolve_ai_collision_times(candidate_events, ai_review, config)
                if ai_key_event_times:
                    enriched["key_event_times"] = ai_key_event_times
                    enriched["review"]["key_event_times"] = ai_key_event_times
                enriched["review"]["collision_ai_summary"] = ai_review["summary"]
                _segment_progress(
                    0.9,
                    (
                        f"Segment {index}/{len(top_segments)}: AI kept {len(enriched['key_event_times'])} "
                        f"collision points."
                    ),
                )
            except (QwenVisionContentBlockedError, ValueError, RuntimeError):
                _segment_progress(
                    0.9,
                    f"Segment {index}/{len(top_segments)}: AI collision review unavailable, falling back to rule-based ranking.",
                )
        else:
            _segment_progress(
                0.9,
                f"Segment {index}/{len(top_segments)}: using rule-based collision ranking without AI review.",
            )
        enriched_segments.append(enriched)

    reporter.update(0.98, "Applying cross-segment collision point diversification.")
    enriched_segments = _select_diversified_key_event_times(enriched_segments, config)

    enriched_segments.sort(
        key=lambda item: (
            float(item.get("fight_probability", 0.0)),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )

    collision_event_preview = _export_collision_event_previews(config, reporter, enriched_segments)
    payload = dict(reviewed_fight_segments_payload)
    payload["stage"] = "stage_02_extract_collision_events"
    payload["top_segments"] = enriched_segments
    payload["collision_event_preview"] = collision_event_preview

    output_path = config.paths.build_dir / "stage_02_collision_events.json"
    write_json(output_path, payload)
    reporter.complete(f"Extracted collision events for {len(enriched_segments)} refined fight segments.")
    return payload
