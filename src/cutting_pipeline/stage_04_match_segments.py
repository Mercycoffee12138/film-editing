from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict

from .config import MatchConfig, PipelineConfig
from .json_io import write_json
from .models import MatchPlanRecord, MatchedClipRecord, MusicHighlightRecord
from .progress import StageReporter


def select_highlight_cluster(
    highlights: list[dict],
    config: MatchConfig,
) -> list[dict]:
    if not highlights:
        return []

    best_selection: list[dict] = []
    best_score = float("-inf")

    for start_index in range(len(highlights)):
        window_start = highlights[start_index]["time"]
        candidates = [
            highlight
            for highlight in highlights[start_index:]
            if highlight["time"] - window_start <= config.highlight_cluster_window_seconds
        ]
        if not candidates:
            continue

        ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
        selected = sorted(
            ranked[: config.max_highlights_per_track],
            key=lambda item: item["time"],
        )
        span = max(selected[-1]["time"] - selected[0]["time"], 1.0)
        score = sum(item["score"] for item in selected) + (0.35 * len(selected)) - (0.01 * span)
        if score > best_score:
            best_score = score
            best_selection = selected

    if not best_selection:
        return []

    selected_times = {round(float(item["time"]), 3) for item in best_selection}
    spacing_seconds = max(float(config.highlight_min_spacing_seconds), 0.0)
    cluster_start = min(float(item["time"]) for item in best_selection)
    cluster_end = max(float(item["time"]) for item in best_selection)
    outside_candidates = sorted(
        (
            highlight
            for highlight in highlights
            if round(float(highlight["time"]), 3) not in selected_times
            and (
                float(highlight["time"]) < cluster_start - spacing_seconds
                or float(highlight["time"]) > cluster_end + spacing_seconds
            )
        ),
        key=lambda item: float(item["score"]),
        reverse=True,
    )

    enriched = list(best_selection)
    outside_limit = min(
        max(int(config.highlight_global_fill_count), 0),
        max(config.max_highlights_per_track - len(enriched), 0),
    )
    for highlight in outside_candidates:
        highlight_time = float(highlight["time"])
        if any(abs(float(existing["time"]) - highlight_time) < spacing_seconds for existing in enriched):
            continue
        enriched.append(highlight)
        if len(enriched) >= len(best_selection) + outside_limit:
            break

    if len(enriched) < config.max_highlights_per_track:
        remaining_candidates = sorted(
            (
                highlight
                for highlight in highlights
                if round(float(highlight["time"]), 3) not in {round(float(item["time"]), 3) for item in enriched}
            ),
            key=lambda item: float(item["score"]),
            reverse=True,
        )
        for highlight in remaining_candidates:
            highlight_time = float(highlight["time"])
            if any(abs(float(existing["time"]) - highlight_time) < spacing_seconds for existing in enriched):
                continue
            enriched.append(highlight)
            if len(enriched) >= config.max_highlights_per_track:
                break

    return sorted(enriched, key=lambda item: item["time"])


def _merge_unique_highlights(highlights: list[dict]) -> list[dict]:
    deduped: dict[float, dict] = {}
    for highlight in highlights:
        deduped[float(highlight["time"])] = highlight
    return sorted(deduped.values(), key=lambda item: item["time"])


def enrich_selected_highlights(
    track: dict,
    selected_highlights: list[dict],
    config: MatchConfig,
) -> list[dict]:
    if not selected_highlights or not config.use_full_track_duration:
        return selected_highlights

    track_duration = float(track["duration"])
    all_highlights = list(track["highlights"])
    enriched = list(selected_highlights)

    late_window_start = track_duration * 0.8
    end_window_start = track_duration * 0.92

    late_candidates = [item for item in all_highlights if float(item["time"]) >= late_window_start]
    end_candidates = [item for item in all_highlights if float(item["time"]) >= end_window_start]

    if late_candidates:
        enriched.append(max(late_candidates, key=lambda item: float(item["score"])))
    if end_candidates:
        enriched.append(max(end_candidates, key=lambda item: float(item["score"])))

    enriched = _merge_unique_highlights(enriched)
    return enriched


def build_timeline_durations(
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
    beat_points: list[dict] | None = None,
) -> list[float]:
    return [
        chunk["duration"]
        for chunk in build_timeline_chunks(audio_start, audio_end, selected_highlights, config, beat_points=beat_points)
    ]


def _normalized_highlight_scores(selected_highlights: list[dict]) -> list[float]:
    if not selected_highlights:
        return []

    scores = [float(highlight["score"]) for highlight in selected_highlights]
    minimum = min(scores)
    maximum = max(scores)
    spread = maximum - minimum
    if spread <= 1e-8:
        return [1.0 for _ in scores]
    return [(score - minimum) / spread for score in scores]


def _target_intensity(
    center_time: float,
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
) -> float:
    total_duration = max(audio_end - audio_start, 1e-6)
    progress = (center_time - audio_start) / total_duration

    # Keep the first stretch more restrained, then raise overall energy later in the song.
    base_intensity = 0.2 + (0.35 * progress)

    peak_window_seconds = min(8.0, max(total_duration * 0.12, 3.0))
    peak_influence = 0.0
    normalized_scores = _normalized_highlight_scores(selected_highlights)
    for highlight, score_weight in zip(selected_highlights, normalized_scores):
        distance = abs(center_time - float(highlight["time"]))
        closeness = max(0.0, 1.0 - (distance / peak_window_seconds))
        peak_influence = max(peak_influence, (0.45 + (0.55 * score_weight)) * closeness)

    intensity = base_intensity + (0.55 * peak_influence)
    return max(0.0, min(1.0, intensity))


def _segment_story_payload(segment: dict) -> dict:
    review = segment.get("review") or {}
    return dict(segment.get("story") or review.get("story") or {})


def _segment_story_role_scores(segment: dict) -> dict[str, float]:
    payload = _segment_story_payload(segment)
    raw_scores = payload.get("role_scores") or {}
    roles = ("setup", "tension", "clash", "pursuit", "aftermath")
    if raw_scores:
        return {
            role: float(raw_scores.get(role, 0.0))
            for role in roles
        }

    fallback_scores = {role: 0.0 for role in roles}
    duration = max(float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)), 0.01)
    motion_level = max(
        float(segment.get("fight_probability", 0.0)),
        float(segment.get("peak_motion", 0.0)),
        float(segment.get("mean_motion", 0.0)),
    )
    fallback_scores["clash"] = min(0.45 + (motion_level * 0.4), 1.0)
    fallback_scores["tension"] = 0.28 if duration >= 6.0 else 0.14
    fallback_scores["pursuit"] = 0.16 if duration >= 8.0 else 0.06
    fallback_scores["setup"] = 0.18 if duration >= 9.5 else 0.08
    fallback_scores["aftermath"] = 0.12 if duration >= 7.5 else 0.04
    return fallback_scores


def _segment_story_role(segment: dict) -> str:
    payload = _segment_story_payload(segment)
    primary_role = str(payload.get("primary_role", "")).strip()
    if primary_role:
        return primary_role
    scores = _segment_story_role_scores(segment)
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0]


def _segment_story_progress(segment: dict) -> float:
    payload = _segment_story_payload(segment)
    if "narrative_progress" in payload:
        return max(0.0, min(float(payload["narrative_progress"]), 1.0))

    role_order = {"setup": 0, "tension": 1, "clash": 2, "pursuit": 3, "aftermath": 4}
    scores = _segment_story_role_scores(segment)
    weight_total = max(sum(scores.values()), 1e-6)
    weighted = sum(role_order[role] * score for role, score in scores.items())
    return max(0.0, min(weighted / (4.0 * weight_total), 1.0))


def _segment_story_intensity(segment: dict) -> float:
    payload = _segment_story_payload(segment)
    if "narrative_intensity" in payload:
        return max(0.0, min(float(payload["narrative_intensity"]), 1.0))

    scores = _segment_story_role_scores(segment)
    motion_level = max(
        float(segment.get("fight_probability", 0.0)),
        float(segment.get("peak_motion", 0.0)),
        float(segment.get("mean_motion", 0.0)),
    )
    return max(
        0.0,
        min(
            (
                (scores["clash"] * 0.56)
                + (scores["pursuit"] * 0.2)
                + (scores["tension"] * 0.12)
                + (scores["aftermath"] * 0.04)
                + (motion_level * 0.08)
            ),
            1.0,
        ),
    )


def _weighted_music_density(center_time: float, beat_points: list[dict], beat_scores: list[float]) -> float:
    if not beat_points:
        return 0.0

    density = 0.0
    for beat, normalized_score in zip(beat_points, beat_scores):
        distance = abs(center_time - float(beat["time"]))
        if distance > 4.2:
            continue
        closeness = max(0.0, 1.0 - (distance / 4.2))
        density += closeness * max(0.25, normalized_score)
    return density


def _section_role_targets(
    section: str,
    progress: float,
    target_intensity: float,
    beat_density: float,
    peak_pressure: float,
) -> dict[str, float]:
    roles = {
        "setup": 0.0,
        "tension": 0.0,
        "clash": 0.0,
        "pursuit": 0.0,
        "aftermath": 0.0,
    }
    if section == "intro":
        roles.update({"setup": 1.0, "tension": 0.72, "clash": 0.18})
    elif section == "build":
        roles.update({"tension": 1.0, "setup": 0.46, "clash": 0.48, "pursuit": 0.18})
    elif section == "drive":
        roles.update({"clash": 1.0, "pursuit": 0.74, "tension": 0.34})
    elif section == "climax":
        roles.update({"clash": 1.0, "pursuit": 0.84, "aftermath": 0.32, "tension": 0.18})
    else:
        roles.update({"aftermath": 1.0, "clash": 0.36, "tension": 0.24, "setup": 0.12})

    if target_intensity <= 0.34:
        roles["setup"] += 0.18
        if progress >= 0.82:
            roles["aftermath"] += 0.14
    if target_intensity >= 0.66:
        roles["clash"] += 0.2
        roles["pursuit"] += 0.12
    if beat_density >= 0.58:
        roles["pursuit"] += 0.1
        roles["clash"] += 0.08
    if peak_pressure >= 0.72:
        roles["clash"] += 0.12
        if progress >= 0.72:
            roles["aftermath"] += 0.08
    if progress >= 0.9:
        roles["aftermath"] += 0.2

    max_weight = max(max(roles.values()), 1e-6)
    return {
        role: round(min(weight / max_weight, 1.0), 4)
        for role, weight in roles.items()
    }


def _classify_music_section(
    progress: float,
    target_intensity: float,
    beat_density: float,
    peak_pressure: float,
) -> str:
    if progress >= 0.9 and target_intensity <= 0.58:
        return "resolve"
    if peak_pressure >= 0.78 and progress >= 0.62:
        return "climax"
    if progress <= 0.12 and target_intensity <= 0.44:
        return "intro"
    if target_intensity >= 0.64 or beat_density >= 0.62:
        return "drive" if progress < 0.78 else "climax"
    if progress <= 0.38:
        return "build"
    if progress >= 0.84 and target_intensity <= 0.62:
        return "resolve"
    return "build" if target_intensity <= 0.48 else "drive"


def _build_music_story_targets(
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    beat_points: list[dict],
    timeline_chunks: list[dict],
) -> list[dict]:
    if not timeline_chunks:
        return []

    normalized_highlights = _normalized_highlight_scores(selected_highlights)
    highlight_pairs = list(zip(selected_highlights, normalized_highlights))
    beat_scores_raw = [float(item.get("score", 0.0)) for item in beat_points]
    if beat_scores_raw:
        beat_min = min(beat_scores_raw)
        beat_spread = max(max(beat_scores_raw) - beat_min, 1e-8)
        beat_scores = [(score - beat_min) / beat_spread for score in beat_scores_raw]
    else:
        beat_scores = []

    raw_densities: list[float] = []
    peak_pressures: list[float] = []
    for chunk in timeline_chunks:
        center_time = (float(chunk["start"]) + float(chunk["end"])) * 0.5
        raw_densities.append(_weighted_music_density(center_time, beat_points, beat_scores))
        peak_window = max(min((audio_end - audio_start) * 0.1, 8.0), 3.5)
        peak_pressure = 0.0
        for highlight, score_weight in highlight_pairs:
            distance = abs(center_time - float(highlight["time"]))
            closeness = max(0.0, 1.0 - (distance / peak_window))
            peak_pressure = max(peak_pressure, closeness * (0.45 + (score_weight * 0.55)))
        peak_pressures.append(peak_pressure)

    density_min = min(raw_densities) if raw_densities else 0.0
    density_spread = max((max(raw_densities) - density_min) if raw_densities else 0.0, 1e-8)
    role_order = {"setup": 0, "tension": 1, "clash": 2, "pursuit": 3, "aftermath": 4}

    story_targets: list[dict] = []
    total_duration = max(audio_end - audio_start, 1e-6)
    for chunk, raw_density, peak_pressure in zip(timeline_chunks, raw_densities, peak_pressures):
        center_time = (float(chunk["start"]) + float(chunk["end"])) * 0.5
        progress = (center_time - audio_start) / total_duration
        target_intensity = float(chunk["target_intensity"])
        beat_density = (raw_density - density_min) / density_spread if raw_densities else 0.0
        section = _classify_music_section(progress, target_intensity, beat_density, peak_pressure)
        role_targets = _section_role_targets(section, progress, target_intensity, beat_density, peak_pressure)
        weight_total = max(sum(role_targets.values()), 1e-6)
        story_progress = sum(
            role_order[role] * float(weight)
            for role, weight in role_targets.items()
        ) / (4.0 * weight_total)
        primary_roles = [
            role
            for role, weight in sorted(role_targets.items(), key=lambda item: (item[1], item[0]), reverse=True)
            if weight >= 0.55
        ][:3]
        pace = "tight" if target_intensity >= 0.68 or beat_density >= 0.62 else "measured"
        if section == "resolve":
            pace = "release"
        story_targets.append(
            {
                "start": round(float(chunk["start"]), 3),
                "end": round(float(chunk["end"]), 3),
                "section": section,
                "pace": pace,
                "music_progress": round(progress, 4),
                "target_story_progress": round(story_progress, 4),
                "target_story_intensity": round(max(target_intensity, peak_pressure * 0.8), 4),
                "beat_density": round(beat_density, 4),
                "peak_pressure": round(peak_pressure, 4),
                "desired_roles": role_targets,
                "primary_roles": primary_roles,
            }
        )
    return story_targets


def _merge_music_story_arc(story_targets: list[dict]) -> list[dict]:
    if not story_targets:
        return []

    merged: list[dict] = []
    for target in story_targets:
        if (
            merged
            and merged[-1]["section"] == target["section"]
            and merged[-1]["pace"] == target["pace"]
            and merged[-1]["primary_roles"] == target["primary_roles"]
        ):
            merged[-1]["end"] = target["end"]
            merged[-1]["target_story_intensity"] = round(
                max(float(merged[-1]["target_story_intensity"]), float(target["target_story_intensity"])),
                4,
            )
            merged[-1]["beat_density"] = round(
                max(float(merged[-1]["beat_density"]), float(target["beat_density"])),
                4,
            )
            merged[-1]["peak_pressure"] = round(
                max(float(merged[-1]["peak_pressure"]), float(target["peak_pressure"])),
                4,
            )
            continue
        merged.append(dict(target))
    return merged


def _story_match_components(
    segment: dict,
    story_target: dict | None,
    previous_segment: dict | None,
    config: MatchConfig,
) -> tuple[float, float, float, float]:
    if not story_target:
        return 0.0, 0.0, 0.0, 0.0

    segment_scores = _segment_story_role_scores(segment)
    desired_roles = {
        role: float(weight)
        for role, weight in (story_target.get("desired_roles") or {}).items()
    }
    desired_total = max(sum(desired_roles.values()), 1e-6)
    role_match = sum(
        desired_roles.get(role, 0.0) * segment_scores.get(role, 0.0)
        for role in ("setup", "tension", "clash", "pursuit", "aftermath")
    ) / desired_total

    target_progress = float(story_target.get("target_story_progress", 0.5))
    segment_progress = _segment_story_progress(segment)
    progress_match = max(0.0, 1.0 - abs(segment_progress - target_progress))

    target_story_intensity = float(story_target.get("target_story_intensity", 0.5))
    segment_intensity = _segment_story_intensity(segment)
    intensity_match = max(0.0, 1.0 - abs(segment_intensity - target_story_intensity))
    segment_primary_role = _segment_story_role(segment)
    primary_roles = [str(role) for role in (story_target.get("primary_roles") or [])]
    primary_role_bonus = 0.0
    primary_role_penalty = 0.0
    if segment_primary_role in primary_roles:
        primary_role_bonus = 0.055
    elif primary_roles and story_target.get("section") in {"intro", "resolve"}:
        primary_role_penalty = 0.03

    transition_bonus = 0.0
    backtrack_penalty = 0.0
    if previous_segment is not None:
        previous_progress = _segment_story_progress(previous_segment)
        progress_delta = segment_progress - previous_progress
        target_delta = target_progress - previous_progress
        if progress_delta >= -0.03 and target_delta >= -0.03:
            transition_bonus = config.story_transition_weight * min(progress_delta + 0.06, 0.18) / 0.18
        elif progress_delta < -0.05 and target_progress >= previous_progress:
            backtrack_penalty = config.story_backtrack_penalty * min(abs(progress_delta) / 0.4, 1.0)

    story_fit = (
        (role_match * config.story_role_weight)
        + (progress_match * config.story_progress_weight)
        + (intensity_match * (config.story_progress_weight * 0.6))
        + primary_role_bonus
        - primary_role_penalty
    )
    return story_fit, role_match, transition_bonus, backtrack_penalty


def _segment_reuse_key(segment: dict) -> tuple[str, str, float, float]:
    return (
        str(segment["source_path"]),
        str(segment["trimmed_path"]),
        round(float(segment["start"]), 3),
        round(float(segment["end"]), 3),
    )


def _target_chunk_duration(
    center_time: float,
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
) -> float:
    intensity = _target_intensity(center_time, audio_start, audio_end, selected_highlights)
    peak_window_seconds = min(5.0, max((audio_end - audio_start) * 0.08, 2.0))
    peak_proximity = 0.0
    for highlight in selected_highlights:
        distance = abs(center_time - float(highlight["time"]))
        peak_proximity = max(peak_proximity, max(0.0, 1.0 - (distance / peak_window_seconds)))

    # Favor longer shots through calmer passages and reserve rapid cutting for stronger peaks.
    compression = max(0.0, (intensity - 0.42) / 0.58)
    compression = min(1.0, compression + (peak_proximity * 0.28))
    duration_span = config.max_clip_seconds - config.min_clip_seconds
    target_duration = config.max_clip_seconds - (duration_span * 0.82 * compression)
    if intensity <= 0.3:
        target_duration = max(target_duration, config.max_clip_seconds * 0.96)
    elif intensity <= 0.41:
        target_duration = max(target_duration, config.max_clip_seconds * 0.88)
    elif intensity <= 0.5:
        target_duration = max(target_duration, config.max_clip_seconds * 0.78)
    return max(config.min_clip_seconds, min(config.max_clip_seconds, target_duration))


def _make_chunk(
    start: float,
    end: float,
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    sync_target_time: float | None = None,
    sync_target_position: str | None = None,
    sync_target_times: list[float] | None = None,
    disable_implicit_sync: bool = False,
) -> dict:
    center_time = (start + end) / 2.0
    resolved_sync_target_time = sync_target_time
    resolved_sync_target_position = sync_target_position
    if resolved_sync_target_time is None and not disable_implicit_sync:
        for highlight in selected_highlights:
            highlight_time = float(highlight["time"])
            if abs(end - highlight_time) <= 1e-3 and end < audio_end - 1e-3:
                resolved_sync_target_time = round(end, 3)
                resolved_sync_target_position = "end"
                break
    if resolved_sync_target_time is None and not disable_implicit_sync:
        for highlight in selected_highlights:
            highlight_time = float(highlight["time"])
            if abs(start - highlight_time) <= 1e-3 and start > audio_start + 1e-3:
                resolved_sync_target_time = round(start, 3)
                resolved_sync_target_position = "start"
                break

    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "target_intensity": round(
            _target_intensity(center_time, audio_start, audio_end, selected_highlights),
            4,
        ),
        "sync_target_time": resolved_sync_target_time,
        "sync_target_position": resolved_sync_target_position,
        "sync_target_times": sorted(
            {
                round(float(value), 3)
                for value in ((sync_target_times or []) + ([resolved_sync_target_time] if resolved_sync_target_time is not None else []))
            }
        ),
    }


def _merge_sync_targets(
    selected_highlights: list[dict],
    beat_points: list[dict],
    config: MatchConfig,
) -> list[float]:
    beat_scores = [float(item.get("score", 0.0)) for item in beat_points]
    beat_threshold = 0.0
    if beat_scores:
        ordered_scores = sorted(beat_scores)
        quantile_index = min(
            len(ordered_scores) - 1,
            max(0, int(round((len(ordered_scores) - 1) * config.beat_boundary_threshold_quantile))),
        )
        beat_threshold = ordered_scores[quantile_index]

    raw_targets = sorted(
        round(float(item["time"]), 3)
        for item in [
            *selected_highlights,
            *[
                beat
                for beat in beat_points
                if float(beat.get("score", 0.0)) >= beat_threshold
            ],
        ]
    )
    if not raw_targets:
        return []

    merge_threshold = max(config.beat_cut_min_clip_seconds, 0.12)
    merged: list[list[float]] = [[raw_targets[0]]]
    for target_time in raw_targets[1:]:
        if abs(target_time - merged[-1][-1]) <= merge_threshold:
            merged[-1].append(target_time)
            continue
        merged.append([target_time])

    return [round(group[-1], 3) for group in merged]


def _all_sync_targets(
    selected_highlights: list[dict],
    beat_points: list[dict],
) -> list[float]:
    return sorted(
        {
            round(float(item["time"]), 3)
            for item in [*selected_highlights, *beat_points]
        }
    )


def _build_sync_timeline_chunks(
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    sync_targets: list[float],
    beat_points: list[dict],
    config: MatchConfig,
) -> list[dict]:
    filtered_targets = sorted(
        {
            round(float(target_time), 3)
            for target_time in sync_targets
            if audio_start < float(target_time) < audio_end
        }
    )
    if not filtered_targets:
        return []

    cut_points = [audio_start] + filtered_targets + [audio_end]
    timeline: list[dict] = []
    min_duration = config.beat_cut_min_clip_seconds
    max_duration = config.beat_cut_max_clip_seconds

    chunk_start = cut_points[0]
    for next_cut in cut_points[1:]:
        proposed_end = next_cut
        proposed_duration = proposed_end - chunk_start
        if proposed_duration < min_duration and timeline:
            timeline[-1]["end"] = round(proposed_end, 3)
            timeline[-1]["duration"] = round(float(timeline[-1]["end"]) - float(timeline[-1]["start"]), 3)
            if proposed_end < audio_end - 1e-3:
                timeline[-1]["sync_target_time"] = round(proposed_end, 3)
                timeline[-1]["sync_target_position"] = "end"
                timeline[-1]["sync_target_times"] = sorted(
                    {
                        *list(timeline[-1].get("sync_target_times") or []),
                        round(proposed_end, 3),
                    }
                )
            chunk_start = proposed_end
            continue

        while proposed_duration > max_duration:
            split_end = min(chunk_start + max_duration, proposed_end)
            is_target_boundary = abs(split_end - proposed_end) <= 1e-3 and proposed_end < audio_end - 1e-3
            timeline.append(
                _make_chunk(
                    chunk_start,
                    split_end,
                    audio_start,
                    audio_end,
                    selected_highlights,
                    sync_target_time=round(proposed_end, 3) if is_target_boundary else None,
                    sync_target_position="end" if is_target_boundary else None,
                    sync_target_times=[round(proposed_end, 3)] if is_target_boundary else [],
                    disable_implicit_sync=True,
                )
            )
            chunk_start = split_end
            proposed_duration = proposed_end - chunk_start

        if proposed_duration <= 0.0:
            continue
        if proposed_end < audio_end - 1e-3:
            timeline.append(
                _make_chunk(
                    chunk_start,
                    proposed_end,
                    audio_start,
                    audio_end,
                    selected_highlights,
                    sync_target_time=round(proposed_end, 3),
                    sync_target_position="end",
                    sync_target_times=[round(proposed_end, 3)],
                    disable_implicit_sync=True,
                )
            )
        else:
            sync_start = chunk_start if chunk_start > audio_start + 1e-3 else None
            timeline.append(
                _make_chunk(
                    chunk_start,
                    proposed_end,
                    audio_start,
                    audio_end,
                    selected_highlights,
                    sync_target_time=round(sync_start, 3) if sync_start is not None else None,
                    sync_target_position="start" if sync_start is not None else None,
                    sync_target_times=[round(sync_start, 3)] if sync_start is not None else [],
                    disable_implicit_sync=True,
                )
            )
        chunk_start = proposed_end

    return timeline


def _merge_sequence_chunks(
    timeline: list[dict],
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
) -> list[dict]:
    if not timeline:
        return []

    merged: list[dict] = []
    index = 0
    while index < len(timeline):
        current = dict(timeline[index])
        current_targets = sorted(round(float(value), 3) for value in (current.get("sync_target_times") or []))

        best_chunk = current
        best_target_count = len(current_targets)
        probe_index = index + 1
        while probe_index < len(timeline):
            candidate = timeline[probe_index]
            combined_start = float(current["start"])
            combined_end = float(candidate["end"])
            combined_duration = combined_end - combined_start
            if combined_duration > config.max_clip_seconds + 1e-6:
                break

            combined_targets = sorted(
                {
                    *current_targets,
                    *(round(float(value), 3) for value in (candidate.get("sync_target_times") or [])),
                }
            )
            if len(combined_targets) <= best_target_count:
                break

            current = _make_chunk(
                combined_start,
                combined_end,
                audio_start,
                audio_end,
                selected_highlights,
                sync_target_time=combined_targets[-1] if combined_targets else None,
                sync_target_position="end" if combined_targets else None,
                sync_target_times=combined_targets,
                disable_implicit_sync=True,
            )
            current_targets = combined_targets
            best_chunk = current
            best_target_count = len(combined_targets)

            if best_target_count >= config.preferred_sequence_match_count:
                probe_index += 1
                while probe_index < len(timeline):
                    extra = timeline[probe_index]
                    extra_targets = sorted(
                        {
                            *current_targets,
                            *(round(float(value), 3) for value in (extra.get("sync_target_times") or [])),
                        }
                    )
                    extra_duration = float(extra["end"]) - float(best_chunk["start"])
                    if extra_duration > config.max_clip_seconds + 1e-6 or len(extra_targets) == len(current_targets):
                        break
                    current = _make_chunk(
                        float(best_chunk["start"]),
                        float(extra["end"]),
                        audio_start,
                        audio_end,
                        selected_highlights,
                        sync_target_time=extra_targets[-1] if extra_targets else None,
                        sync_target_position="end" if extra_targets else None,
                        sync_target_times=extra_targets,
                        disable_implicit_sync=True,
                    )
                    current_targets = extra_targets
                    best_chunk = current
                    best_target_count = len(extra_targets)
                    probe_index += 1
                break

            probe_index += 1

        merged.append(best_chunk)
        consumed_until = probe_index if best_chunk is not timeline[index] and float(best_chunk["end"]) != float(timeline[index]["end"]) else index + 1
        while consumed_until < len(timeline) and float(timeline[consumed_until - 1]["end"]) < float(best_chunk["end"]) - 1e-6:
            consumed_until += 1
        index = max(consumed_until, index + 1)

    return merged


def _assign_chunk_sync_target_times(
    timeline: list[dict],
    sync_targets: list[float],
) -> list[dict]:
    enriched: list[dict] = []
    for chunk in timeline:
        start = round(float(chunk["start"]), 3)
        end = round(float(chunk["end"]), 3)
        chunk_targets = [
            round(float(target_time), 3)
            for target_time in sync_targets
            if start - 1e-3 <= float(target_time) <= end + 1e-3
        ]
        payload = dict(chunk)
        payload["sync_target_times"] = chunk_targets
        enriched.append(payload)
    return enriched


def _segment_event_times(segment: dict) -> list[float]:
    event_times = sorted(
        round(float(value), 3)
        for value in (segment.get("key_event_times") or [])
        if isinstance(value, (int, float))
    )
    return event_times or [round(float(segment["peak_time"]), 3)]


def _alignment_plan(
    segment: dict,
    duration: float,
    chunk: dict,
) -> tuple[float, float, float, float | None, float]:
    max_start = max(float(segment["video_duration"]) - duration, 0.0)
    sync_target_time = chunk.get("sync_target_time")
    sync_target_position = chunk.get("sync_target_position")
    if sync_target_position == "start":
        target_position = 0.0
    elif sync_target_position == "end":
        target_position = duration
    else:
        target_position = duration * 0.5

    best_clip_start = 0.0
    best_source_event_time = float(segment["peak_time"])
    best_error = float("inf")

    for source_event_time in _segment_event_times(segment):
        if sync_target_position == "start":
            ideal_start = source_event_time
        elif sync_target_position == "end":
            ideal_start = source_event_time - duration
        else:
            ideal_start = source_event_time - (duration * 0.5)

        clip_start = max(0.0, min(ideal_start, max_start))
        actual_event_position = source_event_time - clip_start
        alignment_error = abs(actual_event_position - target_position)
        if alignment_error < best_error:
            best_error = alignment_error
            best_clip_start = clip_start
            best_source_event_time = source_event_time

    clip_end = best_clip_start + duration
    return (
        round(best_clip_start, 3),
        round(clip_end, 3),
        round(best_source_event_time, 3),
        round(float(sync_target_time), 3) if sync_target_time is not None else None,
        round(best_error, 3),
    )


def _single_alignment_plan(
    segment: dict,
    duration: float,
    chunk: dict,
) -> tuple[float, float, float, float | None, float, int, float | None, list[float], list[float]]:
    clip_start, clip_end, source_event_time, target_highlight_time, alignment_error = _alignment_plan(
        segment,
        duration,
        chunk,
    )
    return (
        clip_start,
        clip_end,
        source_event_time,
        target_highlight_time,
        alignment_error,
        1 if target_highlight_time is not None else 0,
        None,
        [round(float(source_event_time), 3)] if target_highlight_time is not None else [],
        [round(float(target_highlight_time), 3)] if target_highlight_time is not None else [],
    )


def _chunk_target_times(timeline_chunks: list[dict], chunk_index: int) -> list[float]:
    chunk = timeline_chunks[chunk_index]
    return sorted(round(float(value), 3) for value in (chunk.get("sync_target_times") or []))


def _resolve_chunk_alignment_target(
    chunk: dict,
    target_times: list[float],
) -> dict:
    if chunk.get("sync_target_time") is not None or not target_times:
        return chunk

    start = float(chunk["start"])
    end = float(chunk["end"])
    center = (start + end) * 0.5
    resolved_target = min(
        (round(float(value), 3) for value in target_times),
        key=lambda value: (
            min(abs(value - start), abs(end - value)),
            abs(value - center),
            -value,
        ),
    )
    resolved_position = "start" if abs(resolved_target - start) <= abs(end - resolved_target) else "end"

    payload = dict(chunk)
    payload["sync_target_time"] = resolved_target
    payload["sync_target_position"] = resolved_position
    return payload


def _segment_time_gap(left_segment: dict, right_segment: dict) -> float:
    if left_segment.get("source_path") != right_segment.get("source_path"):
        return float("inf")

    left_start = float(left_segment["start"])
    left_end = float(left_segment["end"])
    right_start = float(right_segment["start"])
    right_end = float(right_segment["end"])

    if left_end < right_start:
        return right_start - left_end
    if right_end < left_start:
        return left_start - right_end
    return 0.0


def _clip_continuity_metrics(
    previous_clip: MatchedClipRecord,
    clip_start: float,
    clip_end: float,
) -> tuple[float, float, float, float]:
    previous_start = float(previous_clip.clip_start)
    previous_end = float(previous_clip.clip_end)
    previous_center = (previous_start + previous_end) * 0.5
    current_center = (clip_start + clip_end) * 0.5
    forward_gap = clip_start - previous_end
    center_gap = abs(current_center - previous_center)
    overlap_seconds = max(0.0, min(clip_end, previous_end) - max(clip_start, previous_start))
    return forward_gap, abs(forward_gap), center_gap, overlap_seconds


def _extends_local_continuity_run(
    previous_clip: MatchedClipRecord | None,
    source_path: str,
    clip_start: float,
    clip_end: float,
    requires_event_match: bool,
) -> bool:
    if previous_clip is None or source_path != previous_clip.source_path:
        return False

    forward_gap, handoff_gap, center_gap, overlap_seconds = _clip_continuity_metrics(
        previous_clip,
        clip_start,
        clip_end,
    )
    soft_handoff_window = 16.0 if requires_event_match else 24.0
    soft_center_window = 24.0 if requires_event_match else 40.0
    overlap_limit = max(min(float(previous_clip.duration), clip_end - clip_start) * 0.35, 0.12)
    return (
        forward_gap >= -0.12
        and handoff_gap <= soft_handoff_window
        and center_gap <= soft_center_window
        and overlap_seconds <= overlap_limit
    )


def _trajectory_bonus(
    previous_previous_clip: MatchedClipRecord | None,
    previous_clip: MatchedClipRecord | None,
    source_path: str,
    clip_start: float,
    clip_end: float,
) -> float:
    if (
        previous_previous_clip is None
        or previous_clip is None
        or source_path != previous_clip.source_path
        or previous_previous_clip.source_path != previous_clip.source_path
    ):
        return 0.0

    previous_previous_center = (float(previous_previous_clip.clip_start) + float(previous_previous_clip.clip_end)) * 0.5
    previous_center = (float(previous_clip.clip_start) + float(previous_clip.clip_end)) * 0.5
    current_center = (clip_start + clip_end) * 0.5
    last_delta = previous_center - previous_previous_center
    current_delta = current_center - previous_center

    if abs(last_delta) < 3.0 or abs(current_delta) < 3.0:
        return 0.0

    if last_delta * current_delta < 0.0:
        return -0.18 * min(abs(current_delta) / max(abs(last_delta), 1e-6), 1.5)

    pace_ratio = min(abs(current_delta), abs(last_delta)) / max(abs(current_delta), abs(last_delta), 1e-6)
    return 0.08 + (0.08 * pace_ratio)


def _continuity_bonus(
    previous_previous_clip: MatchedClipRecord | None,
    previous_clip: MatchedClipRecord | None,
    previous_segment: dict | None,
    segment: dict,
    clip_start: float,
    clip_end: float,
    requires_event_match: bool,
    continuity_streak: int,
) -> float:
    if previous_clip is None:
        return 0.0

    source_path = str(segment.get("source_path", ""))
    if source_path != previous_clip.source_path:
        return -0.06 * min(max(continuity_streak, 1), 3)

    previous_start = float(previous_clip.clip_start)
    forward_gap, handoff_gap, center_gap, overlap_seconds = _clip_continuity_metrics(
        previous_clip,
        clip_start,
        clip_end,
    )

    preferred_center_window = 12.0 if requires_event_match else 18.0
    soft_center_window = 32.0 if requires_event_match else 50.0
    hard_center_window = 140.0 if requires_event_match else 220.0
    preferred_handoff_window = 3.5 if requires_event_match else 6.5
    soft_handoff_window = 14.0 if requires_event_match else 22.0
    hard_handoff_window = 65.0 if requires_event_match else 110.0

    center_score = max(0.0, 1.0 - (center_gap / preferred_center_window))
    handoff_score = max(0.0, 1.0 - (handoff_gap / preferred_handoff_window))

    forward_progress_bonus = 0.0
    if -0.08 <= forward_gap <= preferred_handoff_window:
        forward_progress_bonus = 0.12 * (
            1.0 - (max(forward_gap, 0.0) / max(preferred_handoff_window, 1e-6))
        )

    segment_proximity_bonus = 0.0
    if previous_segment is not None:
        segment_gap = _segment_time_gap(previous_segment, segment)
        if math.isfinite(segment_gap):
            segment_proximity_bonus = 0.16 * max(0.0, 1.0 - (segment_gap / 12.0))
            if segment_gap <= 2.5:
                segment_proximity_bonus += 0.06

    jump_penalty = 0.0
    if handoff_gap > soft_handoff_window:
        jump_penalty += 0.24 * min(
            (handoff_gap - soft_handoff_window) / max(hard_handoff_window - soft_handoff_window, 1e-6),
            1.0,
        )
    if center_gap > soft_center_window:
        jump_penalty += 0.32 * min(
            (center_gap - soft_center_window) / max(hard_center_window - soft_center_window, 1e-6),
            1.0,
        )

    overlap_penalty = 0.0
    if overlap_seconds > 0.0:
        overlap_denominator = max(min(float(previous_clip.duration), clip_end - clip_start), 1e-6)
        overlap_penalty = 0.22 * min(overlap_seconds / overlap_denominator, 1.0)

    backward_penalty = 0.0
    if forward_gap < -0.12:
        backward_penalty += 0.18 * min(abs(forward_gap) / 30.0, 1.0)
    if clip_start + 0.12 < previous_start:
        backward_penalty += 0.18 * min((previous_start - clip_start) / 35.0, 1.0)

    streak_bonus = 0.0
    if continuity_streak > 0 and _extends_local_continuity_run(
        previous_clip,
        source_path,
        clip_start,
        clip_end,
        requires_event_match,
    ):
        streak_bonus = 0.05 * min(continuity_streak, 3)

    trajectory_bonus = _trajectory_bonus(
        previous_previous_clip,
        previous_clip,
        source_path,
        clip_start,
        clip_end,
    )
    return (
        0.02
        + (center_score * 0.22)
        + (handoff_score * 0.16)
        + forward_progress_bonus
        + segment_proximity_bonus
        + streak_bonus
        + trajectory_bonus
        - jump_penalty
        - overlap_penalty
        - backward_penalty
    )


def _timeline_order_penalty(
    previous_clip: MatchedClipRecord | None,
    segment: dict,
    clip_start: float,
    config: MatchConfig,
) -> float:
    if previous_clip is None:
        return 0.0

    source_path = str(segment.get("source_path", ""))
    if source_path != previous_clip.source_path:
        return 0.08

    previous_start = float(previous_clip.clip_start)
    tolerance = max(float(config.source_timeline_backtrack_tolerance_seconds), 0.0)
    if clip_start + tolerance >= previous_start:
        return 0.0

    backtrack_seconds = previous_start - clip_start
    return 0.18 + (0.34 * min(backtrack_seconds / 36.0, 1.0))


def _sequence_alignment_plan(
    segment: dict,
    duration: float,
    chunk: dict,
    target_times: list[float],
    config: MatchConfig,
) -> tuple[float, float, float, float | None, float, int, float | None, list[float], list[float]]:
    single_alignment_plan = _single_alignment_plan(
        segment,
        duration,
        chunk,
    )
    event_times = _segment_event_times(segment)
    minimum_match_count = max(2, config.minimum_sequence_match_count)
    if len(target_times) < minimum_match_count or len(event_times) < minimum_match_count:
        return single_alignment_plan

    max_start = max(float(segment["video_duration"]) - duration, 0.0)
    chunk_start = float(chunk["start"])
    sync_target_time = round(float(chunk["sync_target_time"]), 3) if chunk.get("sync_target_time") is not None else None
    target_positions = [round(float(target_time) - chunk_start, 3) for target_time in target_times]

    best_sequence: tuple[int, float, float, float, list[float], list[float]] | None = None
    best_reference_target: float | None = None
    best_reference_event: float | None = None

    max_match_count = min(len(target_positions), len(event_times))
    for match_count in range(max_match_count, minimum_match_count - 1, -1):
        for target_index in range(0, len(target_positions) - match_count + 1):
            target_window = target_positions[target_index : target_index + match_count]
            target_window_times = target_times[target_index : target_index + match_count]
            for event_index in range(0, len(event_times) - match_count + 1):
                event_window = event_times[event_index : event_index + match_count]
                unclamped_start = sum(
                    source_time - target_position
                    for source_time, target_position in zip(event_window, target_window)
                ) / match_count
                candidate_start = max(0.0, min(unclamped_start, max_start))
                position_errors = [
                    abs((source_time - candidate_start) - target_position)
                    for source_time, target_position in zip(event_window, target_window)
                ]
                average_error = sum(position_errors) / match_count

                interval_error: float | None = None
                if match_count >= 2:
                    interval_ratios: list[float] = []
                    for source_left, source_right, target_left, target_right in zip(
                        event_window,
                        event_window[1:],
                        target_window,
                        target_window[1:],
                    ):
                        target_gap = max(target_right - target_left, 1e-3)
                        source_gap = source_right - source_left
                        interval_ratios.append(abs(source_gap - target_gap) / target_gap)
                    interval_error = sum(interval_ratios) / len(interval_ratios) if interval_ratios else 0.0
                if average_error > config.sequence_match_max_average_error_seconds:
                    continue
                if (
                    interval_error is not None
                    and interval_error > config.sequence_match_max_interval_error
                ):
                    continue
                effective_interval_error = interval_error if interval_error is not None else 1.0
                candidate_key = (
                    match_count,
                    -average_error,
                    -effective_interval_error,
                    -abs(unclamped_start - candidate_start),
                )
                if best_sequence is None or candidate_key > (
                    best_sequence[0],
                    -best_sequence[1],
                    -best_sequence[2],
                    -best_sequence[3],
                ):
                    best_sequence = (
                        match_count,
                        average_error,
                        effective_interval_error,
                        candidate_start,
                        event_window,
                        target_window_times,
                    )
                    if sync_target_time in target_window_times:
                        reference_index = target_window_times.index(sync_target_time)
                    else:
                        reference_index = len(target_window_times) - 1
                    best_reference_target = target_window_times[reference_index]
                    best_reference_event = event_window[reference_index]

    if best_sequence is None or best_reference_event is None:
        return single_alignment_plan

    matched_event_count, average_error, interval_error, best_clip_start, matched_event_times, matched_target_times = best_sequence
    best_clip_end = best_clip_start + duration
    return (
        round(best_clip_start, 3),
        round(best_clip_end, 3),
        round(best_reference_event, 3),
        round(float(best_reference_target), 3) if best_reference_target is not None else None,
        round(average_error, 3),
        matched_event_count,
        round(interval_error, 3) if interval_error is not None else None,
        [round(float(value), 3) for value in matched_event_times],
        [round(float(value), 3) for value in matched_target_times],
    )


def _rebalance_timeline_chunks(
    timeline: list[dict],
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
) -> list[dict]:
    rebalanced: list[dict] = []
    for chunk in timeline:
        start = float(chunk["start"])
        end = float(chunk["end"])
        duration = end - start
        if duration <= config.max_clip_seconds + 1e-6:
            rebalanced.append(chunk)
            continue

        split_count = max(2, math.ceil(duration / config.max_clip_seconds))
        split_duration = duration / split_count
        cursor = start
        for split_index in range(split_count):
            next_end = end if split_index == split_count - 1 else cursor + split_duration
            rebalanced.append(
                _make_chunk(
                    cursor,
                    next_end,
                    audio_start,
                    audio_end,
                    selected_highlights,
                )
            )
            cursor = next_end
    return rebalanced


def _accelerate_high_intensity_chunks(
    timeline: list[dict],
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
) -> list[dict]:
    accelerated: list[dict] = []
    total_duration = max(audio_end - audio_start, 1e-6)

    for chunk in timeline:
        duration = float(chunk["duration"])
        intensity = float(chunk["target_intensity"])
        progress = ((float(chunk["start"]) + float(chunk["end"])) * 0.5 - audio_start) / total_duration

        should_split = (
            duration >= max(config.min_clip_seconds * 2.2, 2.6)
            and intensity >= 0.66
            and progress >= 0.56
        )
        if not should_split:
            accelerated.append(chunk)
            continue

        split_count = 2
        if duration >= max(config.min_clip_seconds * 2.8, 3.2) and intensity >= 0.72 and progress >= 0.68:
            split_count = 3

        split_duration = duration / split_count
        if split_duration < config.min_clip_seconds:
            accelerated.append(chunk)
            continue

        cursor = float(chunk["start"])
        for split_index in range(split_count):
            next_end = float(chunk["end"]) if split_index == split_count - 1 else cursor + split_duration
            accelerated.append(
                _make_chunk(
                    cursor,
                    next_end,
                    audio_start,
                    audio_end,
                    selected_highlights,
                )
            )
            cursor = next_end

    return accelerated


def _enforce_minimum_chunk_duration(
    timeline: list[dict],
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
) -> list[dict]:
    if not timeline:
        return []

    minimum_duration = config.beat_cut_min_clip_seconds if config.beat_cut_enabled else config.min_clip_seconds
    merged: list[dict] = []

    for chunk in timeline:
        current = dict(chunk)
        if not merged:
            merged.append(current)
            continue

        if float(current["duration"]) < minimum_duration or float(merged[-1]["duration"]) < minimum_duration:
            previous = merged.pop()
            merged.append(
                _make_chunk(
                    float(previous["start"]),
                    float(current["end"]),
                    audio_start,
                    audio_end,
                    selected_highlights,
                    sync_target_time=current.get("sync_target_time") or previous.get("sync_target_time"),
                    sync_target_position=current.get("sync_target_position") or previous.get("sync_target_position"),
                    disable_implicit_sync=True,
                )
            )
            continue

        merged.append(current)

    if len(merged) >= 2 and float(merged[-1]["duration"]) < minimum_duration:
        last = merged.pop()
        previous = merged.pop()
        merged.append(
            _make_chunk(
                float(previous["start"]),
                float(last["end"]),
                audio_start,
                audio_end,
                selected_highlights,
                sync_target_time=last.get("sync_target_time") or previous.get("sync_target_time"),
                sync_target_position=last.get("sync_target_position") or previous.get("sync_target_position"),
                disable_implicit_sync=True,
            )
        )

    return merged


def _linger_on_calm_chunks(
    timeline: list[dict],
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
) -> list[dict]:
    if not timeline:
        return []

    merged: list[dict] = []
    max_calm_duration = max(config.max_clip_seconds * 1.55, 5.2)

    for chunk in timeline:
        current = dict(chunk)
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        previous_targets = list(previous.get("sync_target_times") or [])
        current_targets = list(current.get("sync_target_times") or [])
        has_sync_targets = bool(previous_targets or current_targets)
        calm_pair = (
            float(previous["target_intensity"]) <= 0.43
            and float(current["target_intensity"]) <= 0.43
        )
        combined_duration = float(current["end"]) - float(previous["start"])

        if not has_sync_targets and calm_pair and combined_duration <= max_calm_duration:
            merged[-1] = _make_chunk(
                float(previous["start"]),
                float(current["end"]),
                audio_start,
                audio_end,
                selected_highlights,
                disable_implicit_sync=True,
            )
            continue

        merged.append(current)

    return merged


def build_timeline_chunks(
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
    beat_points: list[dict] | None = None,
) -> list[dict]:
    if config.beat_cut_enabled and beat_points:
        sync_targets = _merge_sync_targets(selected_highlights, beat_points, config)
        all_sync_targets = _all_sync_targets(selected_highlights, beat_points)
        beat_timeline = _build_sync_timeline_chunks(
            audio_start,
            audio_end,
            selected_highlights,
            sync_targets,
            beat_points,
            config,
        )
        if beat_timeline:
            beat_timeline = _merge_sequence_chunks(
                beat_timeline,
                audio_start,
                audio_end,
                selected_highlights,
                config,
            )
            beat_timeline = _enforce_minimum_chunk_duration(
                beat_timeline,
                audio_start,
                audio_end,
                selected_highlights,
                config,
            )
            beat_timeline = _linger_on_calm_chunks(
                beat_timeline,
                audio_start,
                audio_end,
                selected_highlights,
                config,
            )
            return _assign_chunk_sync_target_times(beat_timeline, all_sync_targets)

    cut_points = [audio_start] + [highlight["time"] for highlight in selected_highlights] + [audio_end]
    raw_gaps = [end - start for start, end in zip(cut_points, cut_points[1:])]

    timeline: list[dict] = []
    cursor = audio_start
    for gap in raw_gaps:
        region_start = cursor
        region_end = cursor + gap
        remaining = gap

        while remaining > 0:
            center_time = region_start + (remaining / 2.0)
            target_duration = _target_chunk_duration(
                center_time,
                audio_start,
                audio_end,
                selected_highlights,
                config,
            )

            if remaining <= config.max_clip_seconds:
                if remaining < config.min_clip_seconds and timeline:
                    timeline[-1]["end"] = round(timeline[-1]["end"] + remaining, 3)
                    timeline[-1]["duration"] = round(timeline[-1]["duration"] + remaining, 3)
                    remaining = 0.0
                    break
                chunk_duration = remaining
            else:
                chunk_duration = min(remaining, target_duration)
                leftover = remaining - chunk_duration
                if 0.0 < leftover < config.min_clip_seconds:
                    chunk_duration = remaining

            chunk_end = region_start + chunk_duration
            timeline.append(_make_chunk(region_start, chunk_end, audio_start, audio_end, selected_highlights))
            remaining = round(region_end - chunk_end, 6)
            region_start = chunk_end

        cursor = region_end

    timeline = _rebalance_timeline_chunks(
        timeline,
        audio_start,
        audio_end,
        selected_highlights,
        config,
    )
    timeline = _accelerate_high_intensity_chunks(
        timeline,
        audio_start,
        audio_end,
        selected_highlights,
        config,
    )
    timeline = _linger_on_calm_chunks(
        timeline,
        audio_start,
        audio_end,
        selected_highlights,
        config,
    )
    return _enforce_minimum_chunk_duration(
        timeline,
        audio_start,
        audio_end,
        selected_highlights,
        config,
    )


def assign_clips(
    fight_segments: list[dict],
    calm_segments: list[dict],
    timeline_chunks: list[dict],
    config: MatchConfig,
    music_story_targets: list[dict] | None = None,
) -> list[MatchedClipRecord]:
    def _fight_probability_value(segment: dict) -> float:
        return float(segment.get("fight_probability", segment.get("confidence", segment.get("score", 0.0))))

    ranked_fight_segments = sorted(
        fight_segments,
        key=lambda item: (_fight_probability_value(item), float(item["score"])),
        reverse=True,
    )
    ranked_calm_segments = sorted(calm_segments, key=lambda item: item["score"], reverse=True)
    if not ranked_fight_segments and not ranked_calm_segments:
        raise ValueError("No fight segments are available for clip assignment.")

    fight_scores = [float(segment["score"]) for segment in ranked_fight_segments] or [0.0]
    calm_scores = [float(segment["score"]) for segment in ranked_calm_segments] or [0.0]
    fight_probabilities = [_fight_probability_value(segment) for segment in ranked_fight_segments] or [0.0]
    fight_min = min(fight_scores)
    fight_spread = max(max(fight_scores) - fight_min, 1e-8)
    fight_probability_min = min(fight_probabilities)
    fight_probability_spread = max(max(fight_probabilities) - fight_probability_min, 1e-8)
    calm_min = min(calm_scores)
    calm_spread = max(max(calm_scores) - calm_min, 1e-8)
    calm_threshold = float(config.calm_target_intensity_threshold)
    total_timeline_duration = sum(float(chunk["duration"]) for chunk in timeline_chunks)
    desired_calm_duration = sum(
        float(chunk["duration"])
        for chunk in timeline_chunks
        if float(chunk["target_intensity"]) <= calm_threshold
    )
    max_calm_duration = min(
        desired_calm_duration,
        total_timeline_duration * max(min(float(config.calm_max_total_share), 1.0), 0.0),
    )

    remaining_fight_segments = list(ranked_fight_segments)
    remaining_calm_segments = list(ranked_calm_segments)
    source_reuse_count: defaultdict[str, int] = defaultdict(int)
    segment_reuse_count: defaultdict[tuple[str, str, float, float], int] = defaultdict(int)
    clips: list[MatchedClipRecord] = []
    assigned_calm_duration = 0.0
    previous_segment: dict | None = None
    continuity_streak = 0
    alignment_soft_limit = config.sequence_match_max_average_error_seconds
    alignment_hard_limit = max(
        config.candidate_alignment_max_error_seconds,
        alignment_soft_limit + 1e-6,
    )

    def _pick_best_candidate(
        candidate_segments: list[tuple[str, int, dict]],
        *,
        duration: float,
        target_intensity: float,
        story_target: dict | None,
        scoring_chunk: dict,
        chunk_target_times: list[float],
        requires_event_match: bool,
        previous_clip: MatchedClipRecord | None,
        previous_previous_clip: MatchedClipRecord | None,
    ) -> dict | None:
        if not candidate_segments:
            return None

        best_candidate: dict | None = None
        fallback_candidate: dict | None = None
        fallback_key: tuple[int, float, float] | None = None

        for pool_name, index, segment in candidate_segments:
            if pool_name == "fight":
                normalized_segment_score = (float(segment["score"]) - fight_min) / fight_spread
                normalized_fight_probability = (
                    _fight_probability_value(segment) - fight_probability_min
                ) / fight_probability_spread
                if target_intensity >= 0.6:
                    probability_blend = 0.55
                    normalized_segment_level = (
                        normalized_segment_score * (1.0 - probability_blend)
                    ) + (
                        normalized_fight_probability * probability_blend
                    )
                    segment_score_weight = 0.12
                    fight_probability_weight = 0.22
                else:
                    normalized_segment_level = normalized_segment_score
                    segment_score_weight = 0.2
                    fight_probability_weight = 0.0
                desired_level = target_intensity
                pool_bias = 0.08 if target_intensity >= 0.52 else 0.0
            else:
                normalized_segment_score = (float(segment["score"]) - calm_min) / calm_spread
                normalized_fight_probability = 0.0
                normalized_segment_level = normalized_segment_score
                segment_score_weight = 0.22
                fight_probability_weight = 0.0
                desired_level = 1.0 - target_intensity
                if target_intensity <= 0.41:
                    pool_bias = 0.18
                elif target_intensity <= 0.58:
                    pool_bias = 0.08
                else:
                    pool_bias = 0.0

            intensity_match = 1.0 - abs(normalized_segment_level - desired_level)
            candidate_plan = _sequence_alignment_plan(
                segment,
                duration,
                scoring_chunk,
                chunk_target_times,
                config,
            )
            (
                clip_start,
                clip_end,
                _,
                _,
                alignment_error,
                matched_event_count,
                sequence_interval_error,
                _,
                _,
            ) = candidate_plan
            alignment_score = 1.0 - min(alignment_error / max(duration, 1e-6), 1.0)
            has_key_events = bool(segment.get("key_event_times"))
            event_bonus = 0.05 if has_key_events else 0.0
            if has_key_events and scoring_chunk.get("sync_target_time") is not None:
                event_bonus += 0.08
            sequence_bonus = 0.0
            if matched_event_count >= config.minimum_sequence_match_count:
                sequence_bonus += 0.14
                sequence_bonus += 0.06 * max(
                    0,
                    matched_event_count - config.minimum_sequence_match_count,
                )
                if sequence_interval_error is not None:
                    sequence_bonus += 0.10 * (1.0 - min(sequence_interval_error, 1.0))
            elif len(chunk_target_times) >= config.minimum_sequence_match_count:
                if alignment_error <= alignment_soft_limit:
                    sequence_bonus -= config.single_point_match_penalty * 0.25
                else:
                    sequence_bonus -= config.single_point_match_penalty
            segment_key = _segment_reuse_key(segment)
            reuse_penalty = 1.0 + (
                source_reuse_count[segment["source_path"]] * config.source_reuse_penalty
            ) + (
                segment_reuse_count[segment_key] * config.segment_reuse_penalty
            )
            continuity_bonus = _continuity_bonus(
                previous_previous_clip,
                previous_clip,
                previous_segment,
                segment,
                clip_start,
                clip_end,
                requires_event_match,
                continuity_streak,
            )
            timeline_order_penalty = _timeline_order_penalty(
                previous_clip,
                segment,
                clip_start,
                config,
            )
            alignment_penalty = 0.0
            if requires_event_match and alignment_error > alignment_soft_limit:
                alignment_penalty = 0.32 * min(
                    (alignment_error - alignment_soft_limit) / (alignment_hard_limit - alignment_soft_limit),
                    1.0,
                )
            story_fit, story_role_match, story_transition_bonus, story_backtrack_penalty = _story_match_components(
                segment,
                story_target,
                previous_segment,
                config,
            )
            weighted_score = (
                (intensity_match * 0.34)
                + (normalized_segment_score * segment_score_weight)
                + (normalized_fight_probability * fight_probability_weight)
                + (alignment_score * 0.22)
                + pool_bias
                + event_bonus
                + sequence_bonus
                + story_fit
            ) / reuse_penalty
            weighted_score += story_transition_bonus
            weighted_score += (continuity_bonus * config.source_timeline_order_weight)
            weighted_score -= timeline_order_penalty + alignment_penalty + story_backtrack_penalty

            candidate_payload = {
                "pool_name": pool_name,
                "index": index,
                "segment": segment,
                "plan": candidate_plan,
                "score": weighted_score,
                "story_fit_score": story_fit + story_transition_bonus - story_backtrack_penalty,
                "story_role_match": story_role_match,
            }
            fallback_candidate_key = (
                1 if (not requires_event_match or alignment_error <= alignment_hard_limit) else 0,
                -alignment_error if requires_event_match else 0.0,
                weighted_score,
            )
            if fallback_key is None or fallback_candidate_key > fallback_key:
                fallback_key = fallback_candidate_key
                fallback_candidate = candidate_payload
            if requires_event_match and alignment_error > alignment_hard_limit:
                continue
            if requires_event_match and matched_event_count <= 0:
                continue
            if best_candidate is None or weighted_score > float(best_candidate["score"]):
                best_candidate = candidate_payload

        return best_candidate or fallback_candidate

    for order, chunk in enumerate(timeline_chunks, start=1):
        chunk_index = order - 1
        duration = float(chunk["duration"])
        target_intensity = float(chunk["target_intensity"])
        story_target = music_story_targets[chunk_index] if music_story_targets and chunk_index < len(music_story_targets) else None
        chunk_target_times = _chunk_target_times(timeline_chunks, chunk_index)
        scoring_chunk = _resolve_chunk_alignment_target(chunk, chunk_target_times)
        requires_event_match = bool(chunk_target_times) or scoring_chunk.get("sync_target_time") is not None
        previous_clip = clips[-1] if clips else None
        previous_previous_clip = clips[-2] if len(clips) >= 2 else None

        if not remaining_fight_segments and ranked_fight_segments:
            remaining_fight_segments = list(ranked_fight_segments)
        if not remaining_calm_segments and ranked_calm_segments:
            remaining_calm_segments = list(ranked_calm_segments)

        fight_candidates = [
            ("fight", index, segment) for index, segment in enumerate(remaining_fight_segments)
        ]
        calm_candidates = [
            ("calm", index, segment) for index, segment in enumerate(remaining_calm_segments)
        ]

        selected_candidate = _pick_best_candidate(
            fight_candidates,
            duration=duration,
            target_intensity=target_intensity,
            story_target=story_target,
            scoring_chunk=scoring_chunk,
            chunk_target_times=chunk_target_times,
            requires_event_match=requires_event_match,
            previous_clip=previous_clip,
            previous_previous_clip=previous_previous_clip,
        )

        remaining_calm_budget = max(0.0, max_calm_duration - assigned_calm_duration)
        allow_calm_override = (
            calm_candidates
            and not requires_event_match
            and target_intensity <= calm_threshold
            and duration <= remaining_calm_budget + 1e-6
        )
        if allow_calm_override:
            calm_candidate = _pick_best_candidate(
                calm_candidates,
                duration=duration,
                target_intensity=target_intensity,
                story_target=story_target,
                scoring_chunk=scoring_chunk,
                chunk_target_times=chunk_target_times,
                requires_event_match=requires_event_match,
                previous_clip=previous_clip,
                previous_previous_clip=previous_previous_clip,
            )
            if calm_candidate is not None:
                if selected_candidate is None:
                    selected_candidate = calm_candidate
                else:
                    replace_margin = float(config.calm_replace_score_margin)
                    if target_intensity <= float(config.calm_force_intensity_threshold):
                        replace_margin = min(replace_margin, 0.0)
                    if float(calm_candidate["score"]) >= float(selected_candidate["score"]) + replace_margin:
                        selected_candidate = calm_candidate

        if selected_candidate is None:
            fallback_candidates = calm_candidates or fight_candidates
            selected_candidate = _pick_best_candidate(
                fallback_candidates,
                duration=duration,
                target_intensity=target_intensity,
                story_target=story_target,
                scoring_chunk=scoring_chunk,
                chunk_target_times=chunk_target_times,
                requires_event_match=requires_event_match,
                previous_clip=previous_clip,
                previous_previous_clip=previous_previous_clip,
            )

        if selected_candidate is None:
                raise ValueError(f"No candidate clips were available for chunk {order}.")

        best_pool = str(selected_candidate["pool_name"])
        best_index = int(selected_candidate["index"])
        best_plan = selected_candidate["plan"]

        if best_pool == "fight":
            selected_segment = remaining_fight_segments.pop(best_index)
        else:
            selected_segment = remaining_calm_segments.pop(best_index)
            assigned_calm_duration += duration
        source_reuse_count[selected_segment["source_path"]] += 1
        segment_reuse_count[_segment_reuse_key(selected_segment)] += 1

        (
            clip_start,
            clip_end,
            source_event_time,
            target_highlight_time,
            alignment_error,
            matched_event_count,
            sequence_interval_error,
            matched_source_event_times,
            matched_target_times,
        ) = best_plan

        clips.append(
            MatchedClipRecord(
                order=order,
                source_path=selected_segment["source_path"],
                trimmed_path=selected_segment["trimmed_path"],
                clip_start=clip_start,
                clip_end=clip_end,
                duration=round(duration, 3),
                segment_score=round(selected_segment["score"], 6),
                source_event_time=source_event_time,
                target_highlight_time=target_highlight_time,
                alignment_error=alignment_error,
                matched_event_count=matched_event_count,
                sequence_interval_error=sequence_interval_error,
                matched_source_event_times=matched_source_event_times,
                matched_target_times=matched_target_times,
                segment_story_role=_segment_story_role(selected_segment),
                music_story_section=str(story_target.get("section")) if story_target and story_target.get("section") else None,
                story_fit_score=round(float(selected_candidate.get("story_fit_score", 0.0)), 4),
            )
        )
        if _extends_local_continuity_run(
            previous_clip,
            selected_segment["source_path"],
            clip_start,
            clip_end,
            requires_event_match,
        ):
            continuity_streak += 1
        else:
            continuity_streak = 0
        previous_segment = selected_segment

    return clips


def build_plan_for_track(track: dict, fight_segments: list[dict], config: MatchConfig) -> MatchPlanRecord | None:
    selected_highlights = select_highlight_cluster(track["highlights"], config)
    if not selected_highlights:
        return None
    selected_highlights = enrich_selected_highlights(track, selected_highlights, config)
    selected_beats = list(track.get("beats") or [])

    if config.use_full_track_duration:
        audio_start = 0.0
        audio_end = track["duration"]
    else:
        audio_start = max(0.0, selected_highlights[0]["time"] - config.intro_padding_seconds)
        audio_end = min(track["duration"], selected_highlights[-1]["time"] + config.outro_padding_seconds)

    timeline_chunks = build_timeline_chunks(
        audio_start,
        audio_end,
        selected_highlights,
        config,
        beat_points=selected_beats,
    )
    timeline_durations = [chunk["duration"] for chunk in timeline_chunks]
    if not timeline_durations:
        return None

    calm_segments = track.get("calm_segments_override") or []
    music_story_targets = _build_music_story_targets(
        audio_start,
        audio_end,
        selected_highlights,
        selected_beats,
        timeline_chunks,
    )
    clips = assign_clips(
        fight_segments,
        calm_segments,
        timeline_chunks,
        config,
        music_story_targets=music_story_targets,
    )
    highlight_records = [MusicHighlightRecord(**highlight) for highlight in selected_highlights]
    beat_records = [MusicHighlightRecord(**beat) for beat in selected_beats]
    music_story_arc = _merge_music_story_arc(music_story_targets)

    average_segment_score = sum(clip.segment_score for clip in clips) / max(len(clips), 1)
    average_highlight_score = sum(highlight.score for highlight in highlight_records) / max(len(highlight_records), 1)
    average_story_fit = sum(float(clip.story_fit_score or 0.0) for clip in clips) / max(len(clips), 1)
    output_duration = round(sum(timeline_durations), 3)
    plan_score = round(
        (average_segment_score * 100.0)
        + average_highlight_score
        + (len(clips) * 0.15)
        + (average_story_fit * 8.0),
        6,
    )

    return MatchPlanRecord(
        music_path=track["music_path"],
        audio_excerpt_start=round(audio_start, 3),
        audio_excerpt_end=round(audio_end, 3),
        output_duration=output_duration,
        selected_highlights=highlight_records,
        timeline_durations=timeline_durations,
        clips=clips,
        plan_score=plan_score,
        selected_beats=beat_records,
        music_story_arc=music_story_arc,
    )


def run(
    config: PipelineConfig,
    reporter: StageReporter,
    fight_segments_payload: dict,
    music_highlights_payload: dict,
) -> dict:
    tracks = music_highlights_payload["tracks"]
    fight_segments = fight_segments_payload["top_segments"]
    calm_segments = fight_segments_payload.get("calm_segments") or []

    requested_music = config.match.selected_music_filename
    if requested_music:
        requested_path = f"source/music/{requested_music}"
        filtered_tracks = [track for track in tracks if track["music_path"] == requested_path]
        if not filtered_tracks:
            available = ", ".join(track["music_path"] for track in tracks)
            raise ValueError(
                f"Configured music file '{requested_music}' was not found. Available tracks: {available}"
            )
        tracks = filtered_tracks

    reporter.start(f"Building match plans for {len(tracks)} tracks.")

    plans: list[MatchPlanRecord] = []
    for index, track in enumerate(tracks, start=1):
        progress = (index - 1) / max(len(tracks), 1)
        reporter.update(progress, f"Matching clips to {track['music_path']} ({index}/{len(tracks)}).")
        track_with_calm_segments = dict(track)
        track_with_calm_segments["calm_segments_override"] = calm_segments
        plan = build_plan_for_track(track_with_calm_segments, fight_segments, config.match)
        if plan is not None:
            plans.append(plan)

    if not plans:
        raise ValueError("No match plans could be created.")

    plans.sort(key=lambda item: item.plan_score, reverse=True)
    selected_plan = plans[0]

    payload = {
        "stage": "stage_04_match_segments",
        "selected_music_path": selected_plan.music_path,
        "plans": [asdict(plan) for plan in plans],
    }
    output_path = config.paths.build_dir / "stage_04_match_plan.json"
    write_json(output_path, payload)

    if requested_music:
        reporter.complete(f"Using configured music track {selected_plan.music_path}.")
    else:
        reporter.complete(f"Selected {selected_plan.music_path} as the best match plan.")
    return payload
