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

    return best_selection


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

    # Keep calmer passages longer, and tighten cuts near musical peaks.
    compression = max(0.0, (intensity - 0.35) / 0.65)
    compression = min(1.0, compression + (peak_proximity * 0.35))
    duration_span = config.max_clip_seconds - config.min_clip_seconds
    target_duration = config.max_clip_seconds - (duration_span * 0.9 * compression)
    return max(config.min_clip_seconds, min(config.max_clip_seconds, target_duration))


def _make_chunk(
    start: float,
    end: float,
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
) -> dict:
    center_time = (start + end) / 2.0
    sync_target_time = None
    sync_target_position = None
    for highlight in selected_highlights:
        highlight_time = float(highlight["time"])
        if abs(end - highlight_time) <= 1e-3 and end < audio_end - 1e-3:
            sync_target_time = round(end, 3)
            sync_target_position = "end"
            break
    if sync_target_time is None:
        for highlight in selected_highlights:
            highlight_time = float(highlight["time"])
            if abs(start - highlight_time) <= 1e-3 and start > audio_start + 1e-3:
                sync_target_time = round(start, 3)
                sync_target_position = "start"
                break

    return {
        "start": round(start, 3),
        "end": round(end, 3),
        "duration": round(end - start, 3),
        "target_intensity": round(
            _target_intensity(center_time, audio_start, audio_end, selected_highlights),
            4,
        ),
        "sync_target_time": sync_target_time,
        "sync_target_position": sync_target_position,
    }


def _build_beat_timeline_chunks(
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    beat_points: list[dict],
    config: MatchConfig,
) -> list[dict]:
    filtered_beats = sorted(
        {
            round(float(beat["time"]), 3): beat
            for beat in beat_points
            if audio_start < float(beat["time"]) < audio_end
        }.values(),
        key=lambda item: float(item["time"]),
    )
    if not filtered_beats:
        return []

    cut_points = [audio_start] + [float(beat["time"]) for beat in filtered_beats] + [audio_end]
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
            chunk_start = proposed_end
            continue

        while proposed_duration > max_duration:
            split_end = min(chunk_start + max_duration, proposed_end)
            timeline.append(_make_chunk(chunk_start, split_end, audio_start, audio_end, selected_highlights))
            chunk_start = split_end
            proposed_duration = proposed_end - chunk_start

        if proposed_duration <= 0.0:
            continue
        timeline.append(_make_chunk(chunk_start, proposed_end, audio_start, audio_end, selected_highlights))
        chunk_start = proposed_end

    return timeline


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


def build_timeline_chunks(
    audio_start: float,
    audio_end: float,
    selected_highlights: list[dict],
    config: MatchConfig,
    beat_points: list[dict] | None = None,
) -> list[dict]:
    if config.beat_cut_enabled and beat_points:
        beat_timeline = _build_beat_timeline_chunks(
            audio_start,
            audio_end,
            selected_highlights,
            beat_points,
            config,
        )
        if beat_timeline:
            return beat_timeline

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
    return _accelerate_high_intensity_chunks(
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

    remaining_fight_segments = list(ranked_fight_segments)
    remaining_calm_segments = list(ranked_calm_segments)
    reuse_count: defaultdict[str, int] = defaultdict(int)
    clips: list[MatchedClipRecord] = []

    for order, chunk in enumerate(timeline_chunks, start=1):
        duration = float(chunk["duration"])
        target_intensity = float(chunk["target_intensity"])
        if target_intensity <= 0.36 and ranked_calm_segments:
            candidate_pool_name = "calm"
        elif (target_intensity >= 0.52 and ranked_fight_segments) or not ranked_calm_segments:
            candidate_pool_name = "fight"
        else:
            candidate_pool_name = "mixed"

        if candidate_pool_name == "fight":
            if not remaining_fight_segments:
                remaining_fight_segments = list(ranked_fight_segments)
            candidate_segments = [("fight", index, segment) for index, segment in enumerate(remaining_fight_segments)]
        elif candidate_pool_name == "calm":
            if not remaining_calm_segments:
                remaining_calm_segments = list(ranked_calm_segments)
            candidate_segments = [("calm", index, segment) for index, segment in enumerate(remaining_calm_segments)]
        else:
            if not remaining_calm_segments:
                remaining_calm_segments = list(ranked_calm_segments)
            if not remaining_fight_segments:
                remaining_fight_segments = list(ranked_fight_segments)
            candidate_segments = [
                ("calm", index, segment) for index, segment in enumerate(remaining_calm_segments)
            ] + [
                ("fight", index, segment) for index, segment in enumerate(remaining_fight_segments)
            ]

        if not candidate_segments:
            if ranked_fight_segments:
                remaining_fight_segments = list(ranked_fight_segments)
                candidate_segments = [("fight", index, segment) for index, segment in enumerate(remaining_fight_segments)]
            else:
                remaining_calm_segments = list(ranked_calm_segments)
                candidate_segments = [("calm", index, segment) for index, segment in enumerate(remaining_calm_segments)]

        best_pool = candidate_segments[0][0]
        best_index = 0
        best_score = float("-inf")
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
                segment_score_weight = 0.18
                fight_probability_weight = 0.0
                desired_level = 1.0 - target_intensity
                pool_bias = 0.08 if target_intensity <= 0.36 else 0.0

            intensity_match = 1.0 - abs(normalized_segment_level - desired_level)
            _, _, _, _, alignment_error = _alignment_plan(segment, duration, chunk)
            alignment_score = 1.0 - min(alignment_error / max(duration, 1e-6), 1.0)
            has_key_events = bool(segment.get("key_event_times"))
            event_bonus = 0.05 if has_key_events else 0.0
            if has_key_events and chunk.get("sync_target_time") is not None:
                event_bonus += 0.08
            reuse_penalty = 1.0 + (reuse_count[segment["source_path"]] * config.source_reuse_penalty)
            weighted_score = (
                (intensity_match * 0.38)
                + (normalized_segment_score * segment_score_weight)
                + (normalized_fight_probability * fight_probability_weight)
                + (alignment_score * 0.28)
                + pool_bias
                + event_bonus
            ) / reuse_penalty
            if weighted_score > best_score:
                best_score = weighted_score
                best_pool = pool_name
                best_index = index

        if best_pool == "fight":
            selected_segment = remaining_fight_segments.pop(best_index)
        else:
            selected_segment = remaining_calm_segments.pop(best_index)
        reuse_count[selected_segment["source_path"]] += 1

        clip_start, clip_end, source_event_time, target_highlight_time, alignment_error = _alignment_plan(
            selected_segment,
            duration,
            chunk,
        )

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
            )
        )

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
    clips = assign_clips(fight_segments, calm_segments, timeline_chunks, config)
    highlight_records = [MusicHighlightRecord(**highlight) for highlight in selected_highlights]
    beat_records = [MusicHighlightRecord(**beat) for beat in selected_beats]

    average_segment_score = sum(clip.segment_score for clip in clips) / max(len(clips), 1)
    average_highlight_score = sum(highlight.score for highlight in highlight_records) / max(len(highlight_records), 1)
    output_duration = round(sum(timeline_durations), 3)
    plan_score = round((average_segment_score * 100.0) + average_highlight_score + (len(clips) * 0.15), 6)

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
