from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from .config import PipelineConfig
from .ffmpeg_tools import concat_video_clips, run_command
from .json_io import read_json, write_json
from .progress import StageReporter


_FFMPEG_ENCODER_SUPPORT_CACHE: dict[str, bool] = {}
_TRIM_SOURCE_LOOKUP_CACHE: dict[str, dict[str, dict[str, str | float]]] = {}
_COLLISION_PREVIEW_LOOKUP_CACHE: dict[str, dict[str, list[dict[str, str | float]]]] = {}


def _supports_ffmpeg_encoder(encoder_name: str) -> bool:
    cached = _FFMPEG_ENCODER_SUPPORT_CACHE.get(encoder_name)
    if cached is not None:
        return cached

    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    supported = result.returncode == 0 and encoder_name in result.stdout
    _FFMPEG_ENCODER_SUPPORT_CACHE[encoder_name] = supported
    return supported


def _video_encode_args(config: PipelineConfig) -> list[str]:
    if _supports_ffmpeg_encoder("h264_videotoolbox"):
        return [
            "-c:v",
            "h264_videotoolbox",
            "-b:v",
            "12M",
            "-maxrate",
            "18M",
        ]

    return [
        "-c:v",
        "libx264",
        "-preset",
        config.render.video_preset,
        "-crf",
        str(config.render.video_crf),
    ]


def _software_video_encode_args(config: PipelineConfig) -> list[str]:
    return [
        "-c:v",
        "libx264",
        "-preset",
        config.render.video_preset,
        "-crf",
        str(config.render.video_crf),
    ]


def _path_is_nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


def _resolve_project_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _relative_to_project(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)

def _load_cache_manifest(path: Path) -> dict | None:
    if not _path_is_nonempty(path):
        return None
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _cache_manifest_matches(path: Path, expected_payload: dict) -> bool:
    return _load_cache_manifest(path) == expected_payload


def _cache_key(payload: dict) -> str:
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:20]


def _trim_source_lookup(config: PipelineConfig) -> dict[str, dict[str, str | float]]:
    manifest_path = config.paths.build_dir / "stage_01_trim_manifest.json"
    cache_key = str(manifest_path)
    cached = _TRIM_SOURCE_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lookup: dict[str, dict[str, str | float]] = {}
    payload = _load_cache_manifest(manifest_path)
    if payload is not None:
        for item in payload.get("trimmed_videos") or []:
            if not isinstance(item, dict):
                continue
            trimmed_path = str(item.get("trimmed_path") or "").strip()
            source_path = str(item.get("source_path") or "").strip()
            if not trimmed_path or not source_path:
                continue
            lookup[trimmed_path] = {
                "source_path": source_path,
                "trim_start": float(item.get("trim_start", 0.0) or 0.0),
            }

    _TRIM_SOURCE_LOOKUP_CACHE[cache_key] = lookup
    return lookup


def _collision_preview_lookup(config: PipelineConfig) -> dict[str, list[dict[str, str | float]]]:
    payload_path = config.paths.build_dir / "stage_02_collision_events.json"
    cache_key = str(payload_path)
    cached = _COLLISION_PREVIEW_LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lookup: dict[str, list[dict[str, str | float]]] = {}
    payload = _load_cache_manifest(payload_path)
    preview = payload.get("collision_event_preview") if payload is not None else None
    events = preview.get("events") if isinstance(preview, dict) else None
    if isinstance(events, list):
        for item in events:
            if not isinstance(item, dict):
                continue
            trimmed_path = str(item.get("trimmed_path") or "").strip()
            output_path = str(item.get("output_path") or "").strip()
            if not trimmed_path or not output_path:
                continue
            lookup.setdefault(trimmed_path, []).append(
                {
                    "event_time": round(float(item.get("event_time", 0.0) or 0.0), 3),
                    "clip_start": float(item.get("clip_start", 0.0) or 0.0),
                    "clip_end": float(item.get("clip_end", 0.0) or 0.0),
                    "output_path": output_path,
                }
            )

    _COLLISION_PREVIEW_LOOKUP_CACHE[cache_key] = lookup
    return lookup


def _resolve_hit_audio_source(
    config: PipelineConfig,
    segment: dict,
) -> tuple[str, float]:
    trimmed_path = str(segment["trimmed_path"])
    trimmed_source_start = float(segment["source_start"])
    trimmed_lookup = _trim_source_lookup(config).get(trimmed_path)
    if trimmed_lookup is not None:
        return (
            str(trimmed_lookup["source_path"]),
            round(trimmed_source_start + float(trimmed_lookup.get("trim_start", 0.0) or 0.0), 3),
        )

    return trimmed_path, trimmed_source_start


def _renderable_hit_audio_source(
    config: PipelineConfig,
    segment: dict,
) -> tuple[str, float] | None:
    trimmed_path = str(segment["trimmed_path"])
    trimmed_source_start = float(segment["source_start"])
    trimmed_input_path = _resolve_project_path(config.paths.project_root, trimmed_path)
    if _path_is_nonempty(trimmed_input_path):
        return trimmed_path, trimmed_source_start

    logical_source_path, logical_source_start = _resolve_hit_audio_source(config, segment)
    logical_input_path = _resolve_project_path(config.paths.project_root, logical_source_path)
    if _path_is_nonempty(logical_input_path):
        return logical_source_path, logical_source_start

    event_time = segment.get("event_time")
    if isinstance(event_time, (int, float)):
        desired_start = float(segment["source_start"])
        desired_end = desired_start + float(segment["duration"])
        for preview in _collision_preview_lookup(config).get(trimmed_path, []):
            preview_event_time = float(preview["event_time"])
            if abs(preview_event_time - float(event_time)) > 0.02:
                continue
            clip_start = float(preview["clip_start"])
            clip_end = float(preview["clip_end"])
            if clip_start - 1e-3 > desired_start or clip_end + 1e-3 < desired_end:
                continue
            preview_output_path = str(preview["output_path"])
            preview_input_path = _resolve_project_path(config.paths.project_root, preview_output_path)
            if not _path_is_nonempty(preview_input_path):
                continue
            return preview_output_path, round(max(desired_start - clip_start, 0.0), 3)

    return None


def _video_cache_payload(config: PipelineConfig, clips: list[dict]) -> dict:
    unique_inputs = sorted({str(clip["trimmed_path"]) for clip in clips})
    return {
        "cache_version": 1,
        "render": {
            "output_width": config.render.output_width,
            "output_height": config.render.output_height,
            "output_fps": config.render.output_fps,
            "video_crf": config.render.video_crf,
            "video_preset": config.render.video_preset,
        },
        "inputs": unique_inputs,
        "clips": [
            {
                "order": int(clip["order"]),
                "trimmed_path": str(clip["trimmed_path"]),
                "clip_start": round(float(clip["clip_start"]), 6),
                "clip_end": round(float(clip["clip_end"]), 6),
                "duration": round(float(clip["duration"]), 6),
            }
            for clip in clips
        ],
    }


def _audio_excerpt_cache_payload(
    config: PipelineConfig,
    music_path: str,
    start: float,
    end: float,
) -> dict:
    return {
        "cache_version": 1,
        "render": {
            "audio_bitrate": config.render.audio_bitrate,
        },
        "music_path": music_path,
        "start": round(start, 6),
        "end": round(end, 6),
    }


def _render_clip(
    config: PipelineConfig,
    clip: dict,
    output_path: Path,
) -> None:
    safe_start = max(float(clip["clip_start"]), 0.0)
    safe_duration = max(float(clip["duration"]), 0.01)
    seek_start = max(safe_start - 1.0, 0.0)
    trim_start = safe_start - seek_start
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{seek_start:.3f}",
        "-i",
        str(config.paths.project_root / clip["trimmed_path"]),
        "-an",
        "-vf",
        (
            f"trim=start={trim_start:.3f}:duration={safe_duration:.3f},"
            "setpts=PTS-STARTPTS,"
            f"scale={config.render.output_width}:{config.render.output_height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={config.render.output_width}:{config.render.output_height}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={config.render.output_fps},"
            "format=yuv420p"
        ),
        "-movflags",
        "+faststart",
    ]
    encode_args = _video_encode_args(config)
    output_arg = [str(output_path)]
    try:
        run_command([*command, *encode_args, *output_arg])
    except RuntimeError:
        uses_hardware = "h264_videotoolbox" in encode_args
        if not uses_hardware:
            raise
        run_command([*command, *_software_video_encode_args(config), *output_arg])


def _render_audio_excerpt(config: PipelineConfig, music_path: str, start: float, end: float, output_path: Path) -> None:
    duration = max(end - start, 0.1)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(config.paths.project_root / music_path),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        config.render.audio_bitrate,
        str(output_path),
    ]
    run_command(command)


def _render_hit_audio_excerpt(
    config: PipelineConfig,
    source_path: str,
    start: float,
    duration: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{max(duration, 0.01):.3f}",
        "-i",
        str(_resolve_project_path(config.paths.project_root, source_path)),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        config.render.audio_bitrate,
        str(output_path),
    ]
    run_command(command)


def _clip_match_count(clip: dict, source_event_times: list[float]) -> int:
    raw_value = clip.get("matched_event_count")
    if isinstance(raw_value, bool):
        matched_event_count = int(raw_value)
    elif isinstance(raw_value, int):
        matched_event_count = raw_value
    elif isinstance(raw_value, float):
        matched_event_count = int(round(raw_value))
    else:
        matched_event_count = 0
    fallback_count = 1 if clip.get("source_event_time") is not None else 0
    return max(matched_event_count, len(source_event_times), fallback_count)


def _source_hit_volume_for_match_count(match_count: int, config: PipelineConfig) -> float:
    if match_count >= 3:
        return float(config.render.source_hit_three_point_volume)
    if match_count == 2:
        return float(config.render.source_hit_two_point_volume)
    if match_count == 1:
        return float(config.render.source_hit_single_point_volume)
    return float(config.render.source_hit_volume)


def _source_hit_event_limit(match_count: int, config: PipelineConfig) -> int:
    if match_count <= 1:
        return 1
    return max(1, min(match_count, int(config.render.source_hit_max_events_per_clip)))


def _clip_hit_audio_segments(clips: list[dict], config: PipelineConfig) -> list[dict]:
    if not config.render.include_source_hit_audio:
        return []

    segments: list[dict] = []
    timeline_cursor = 0.0
    for clip in clips:
        matched_target_times = [
            float(value)
            for value in (clip.get("matched_target_times") or [])
            if isinstance(value, (int, float))
        ]
        source_event_times: list[float] = []
        if matched_target_times:
            source_event_times = [
                float(value)
                for value in (clip.get("matched_source_event_times") or [])
                if isinstance(value, (int, float))
            ]
            if not source_event_times and clip.get("source_event_time") is not None:
                source_event_times = [float(clip["source_event_time"])]

            deduped_times: list[float] = []
            for event_time in source_event_times:
                rounded_time = round(event_time, 3)
                if rounded_time not in deduped_times:
                    deduped_times.append(rounded_time)
            match_count = _clip_match_count(clip, deduped_times)
            source_event_times = deduped_times[: _source_hit_event_limit(match_count, config)]
        else:
            match_count = 0

        if not source_event_times:
            timeline_cursor += float(clip["duration"])
            continue

        clip_hit_volume = _source_hit_volume_for_match_count(match_count, config)
        clip_start = float(clip["clip_start"])
        clip_end = float(clip["clip_end"])
        clip_duration = max(clip_end - clip_start, 0.0)
        minimum_duration = min(config.render.source_hit_min_segment_seconds, clip_duration)

        for event_time in sorted(set(source_event_times)):
            local_start = max(clip_start, event_time - config.render.source_hit_pre_seconds)
            local_end = min(clip_end, event_time + config.render.source_hit_post_seconds)

            duration = max(local_end - local_start, 0.0)
            if 0.0 < duration < minimum_duration:
                center = min(max(event_time, clip_start + (minimum_duration * 0.5)), clip_end - (minimum_duration * 0.5))
                local_start = max(clip_start, center - (minimum_duration * 0.5))
                local_end = min(clip_end, local_start + minimum_duration)
                local_start = max(clip_start, local_end - minimum_duration)
                duration = max(local_end - local_start, 0.0)

            if duration <= 0.01:
                continue

            timeline_offset = timeline_cursor + max(local_start - clip_start, 0.0)
            segments.append(
                {
                    "trimmed_path": clip["trimmed_path"],
                    "source_start": round(local_start, 3),
                    "duration": round(duration, 3),
                    "timeline_offset": round(timeline_offset, 3),
                    "event_time": round(float(event_time), 3),
                    "volume": round(clip_hit_volume, 3),
                }
            )
        timeline_cursor += float(clip["duration"])

    return segments


def _hit_audio_export_payload(config: PipelineConfig, segment: dict) -> dict:
    source_path, source_start = _resolve_hit_audio_source(config, segment)
    return {
        "cache_version": 1,
        "render": {
            "audio_bitrate": config.render.audio_bitrate,
        },
        "source_path": source_path,
        "source_start": round(source_start, 3),
        "duration": round(float(segment["duration"]), 3),
    }


def _prepared_hit_audio_segments(
    config: PipelineConfig,
    hit_segments: list[dict],
) -> list[dict]:
    if not hit_segments:
        return []

    hit_audio_dir = config.paths.stage_05_temp_dir / "hit_audio"
    hit_audio_dir.mkdir(parents=True, exist_ok=True)

    prepared_segments: list[dict] = []
    for segment in hit_segments:
        export_payload = _hit_audio_export_payload(config, segment)
        export_key = _cache_key(export_payload)
        relative_audio_path = _relative_to_project(
            config.paths.project_root,
            hit_audio_dir / f"{export_key}.m4a",
        )
        prepared_segments.append(
            {
                **segment,
                "audio_path": relative_audio_path,
            }
        )
    return prepared_segments


def _ensure_hit_audio_exports(
    config: PipelineConfig,
    hit_segments: list[dict],
) -> list[dict]:
    prepared_segments = _prepared_hit_audio_segments(config, hit_segments)
    if not prepared_segments:
        return []

    available_segments: list[dict] = []
    for segment in prepared_segments:
        output_path = _resolve_project_path(config.paths.project_root, str(segment["audio_path"]))
        if _path_is_nonempty(output_path):
            available_segments.append(segment)
            continue

        renderable_source = _renderable_hit_audio_source(config, segment)
        if renderable_source is None:
            continue

        source_path, source_start = renderable_source
        _render_hit_audio_excerpt(
            config,
            source_path,
            source_start,
            float(segment["duration"]),
            output_path,
        )
        available_segments.append(segment)

    return available_segments


def _hit_audio_mix_cache_payload(
    config: PipelineConfig,
    hit_segments: list[dict],
    total_duration: float,
) -> dict:
    return {
        "cache_version": 1,
        "render": {
            "audio_bitrate": config.render.audio_bitrate,
            "source_hit_fade_seconds": round(float(config.render.source_hit_fade_seconds), 6),
        },
        "total_duration": round(total_duration, 6),
        "segments": [
            {
                "audio_path": str(segment["audio_path"]),
                "duration": round(float(segment["duration"]), 3),
                "timeline_offset": round(float(segment["timeline_offset"]), 3),
                "volume": round(float(segment["volume"]), 3),
            }
            for segment in hit_segments
        ],
    }


def _render_hit_audio_mix(
    config: PipelineConfig,
    hit_segments: list[dict],
    total_duration: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
    ]
    for segment in hit_segments:
        command.extend(
            [
                "-i",
                str(_resolve_project_path(config.paths.project_root, str(segment["audio_path"]))),
            ]
        )

    filter_parts: list[str] = []
    mix_inputs: list[str] = []
    fade_seconds = max(config.render.source_hit_fade_seconds, 0.0)
    for index, segment in enumerate(hit_segments):
        duration = float(segment["duration"])
        fade_duration = min(fade_seconds, duration * 0.5)
        fade_out_start = max(duration - fade_duration, 0.0)
        delay_ms = max(int(round(float(segment["timeline_offset"]) * 1000.0)), 0)
        filter_parts.append(
            (
                f"[{index}:a]"
                "asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d={fade_duration:.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={fade_duration:.3f},"
                f"volume={float(segment['volume']):.3f},"
                f"adelay={delay_ms}|{delay_ms}[hit{index}]"
            )
        )
        mix_inputs.append(f"[hit{index}]")

    filter_parts.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)},volume={len(mix_inputs):.3f}[aout]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[aout]",
            "-t",
            f"{total_duration:.3f}",
            "-c:a",
            "aac",
            "-b:a",
            config.render.audio_bitrate,
            str(output_path),
        ]
    )
    run_command(command)


def _ensure_hit_audio_mix(
    config: PipelineConfig,
    clips: list[dict],
) -> Path | None:
    hit_segments = _clip_hit_audio_segments(clips, config)
    if not hit_segments:
        return None

    prepared_segments = _ensure_hit_audio_exports(config, hit_segments)
    if not prepared_segments:
        return None

    total_duration = sum(float(clip["duration"]) for clip in clips)
    mix_manifest_path = config.paths.stage_05_temp_dir / "stage_05_hit_audio_mix_manifest.json"
    mix_output_path = config.paths.stage_05_temp_dir / "stage_05_hit_audio_mix.m4a"
    mix_payload = _hit_audio_mix_cache_payload(config, prepared_segments, total_duration)
    if _cache_manifest_matches(mix_manifest_path, mix_payload) and _path_is_nonempty(mix_output_path):
        return mix_output_path

    _render_hit_audio_mix(
        config,
        prepared_segments,
        total_duration,
        mix_output_path,
    )
    write_json(mix_manifest_path, mix_payload)
    return mix_output_path


def _normalize_render_clips(clips: list[dict], fps: int) -> list[dict]:
    if fps <= 0:
        return [dict(clip) for clip in clips]

    normalized: list[dict] = []
    cumulative_target = 0.0
    previous_frame_total = 0

    for index, clip in enumerate(clips, start=1):
        updated = dict(clip)
        duration = max(float(clip["duration"]), 1.0 / fps)
        cumulative_target += duration
        target_frame_total = max(index, int(round(cumulative_target * fps)))
        frame_count = max(1, target_frame_total - previous_frame_total)
        snapped_duration = frame_count / fps
        updated["duration"] = round(snapped_duration, 6)
        updated["clip_end"] = round(float(updated["clip_start"]) + snapped_duration, 6)
        normalized.append(updated)
        previous_frame_total = target_frame_total

    return normalized


def _mux_final_audio(
    config: PipelineConfig,
    concat_video_path: Path,
    audio_excerpt_path: Path,
    hit_audio_mix_path: Path | None,
    final_output_path: Path,
) -> None:
    if hit_audio_mix_path is None:
        mux_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(concat_video_path),
            "-i",
            str(audio_excerpt_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            config.render.audio_bitrate,
            "-shortest",
            str(final_output_path),
        ]
        run_command(mux_command)
        return

    filter_parts = [f"[1:a]volume={config.render.music_volume:.3f}[music]"]
    mix_inputs = ["[music]"]

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(concat_video_path),
        "-i",
        str(audio_excerpt_path),
        "-i",
        str(hit_audio_mix_path),
    ]

    mix_inputs.append("[2:a]")

    filter_parts.append(
        f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)},volume={len(mix_inputs):.3f}[aout]"
    )

    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "0:v:0",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            config.render.audio_bitrate,
            "-shortest",
            str(final_output_path),
        ]
    )
    run_command(command)


def _rendered_clip_output_path(config: PipelineConfig, index: int) -> Path:
    return config.paths.stage_05_clip_dir / f"clip_{index:03d}_stage_05.mp4"


def _ensure_video_assets(
    config: PipelineConfig,
    clips: list[dict],
    reporter: StageReporter,
) -> Path:
    concat_list_path = config.paths.stage_05_temp_dir / "stage_05_concat_list.txt"
    concat_video_path = config.paths.stage_05_temp_dir / "stage_05_concat_video.mp4"
    video_manifest_path = config.paths.stage_05_temp_dir / "stage_05_video_manifest.json"
    video_payload = _video_cache_payload(config, clips)
    rendered_clip_paths = [
        _rendered_clip_output_path(config, index)
        for index in range(1, len(clips) + 1)
    ]

    if _cache_manifest_matches(video_manifest_path, video_payload):
        if _path_is_nonempty(concat_video_path):
            reporter.update(0.965, "Reusing cached concatenated video.")
            return concat_video_path

        missing_indices = [
            index
            for index, output_path in enumerate(rendered_clip_paths, start=1)
            if not _path_is_nonempty(output_path)
        ]
        for missing_index in missing_indices:
            clip = clips[missing_index - 1]
            reporter.update(
                0.90,
                f"Rendering missing cached clip {missing_index}/{len(clips)} from {Path(clip['trimmed_path']).name}.",
            )
            _render_clip(config, clip, rendered_clip_paths[missing_index - 1])

        reporter.update(0.975, "Concatenating cached rendered clips.")
        concat_video_clips(rendered_clip_paths, concat_video_path, concat_list_path)
        write_json(video_manifest_path, video_payload)
        return concat_video_path

    for index, clip in enumerate(clips, start=1):
        progress = (index - 1) / max(len(clips) + 2, 1)
        reporter.update(progress, f"Rendering clip {index}/{len(clips)} from {Path(clip['trimmed_path']).name}.")
        _render_clip(config, clip, rendered_clip_paths[index - 1])

    reporter.update(
        (len(clips) + 1) / max(len(clips) + 2, 1),
        "Concatenating rendered clips.",
    )
    concat_video_clips(rendered_clip_paths, concat_video_path, concat_list_path)
    write_json(video_manifest_path, video_payload)
    return concat_video_path


def _ensure_audio_excerpt(
    config: PipelineConfig,
    music_path: str,
    start: float,
    end: float,
    reporter: StageReporter,
    progress: float,
) -> Path:
    output_path = config.paths.stage_05_temp_dir / "stage_05_audio_excerpt.m4a"
    manifest_path = config.paths.stage_05_temp_dir / "stage_05_audio_excerpt_manifest.json"
    cache_payload = _audio_excerpt_cache_payload(config, music_path, start, end)
    if _cache_manifest_matches(manifest_path, cache_payload) and _path_is_nonempty(output_path):
        reporter.update(progress, f"Reusing cached music excerpt from {Path(music_path).name}.")
        return output_path

    reporter.update(progress, f"Rendering music excerpt from {Path(music_path).name}.")
    _render_audio_excerpt(config, music_path, start, end, output_path)
    write_json(manifest_path, cache_payload)
    return output_path


def run(config: PipelineConfig, reporter: StageReporter, match_payload: dict) -> dict:
    config.paths.stage_05_clip_dir.mkdir(parents=True, exist_ok=True)
    config.paths.stage_05_temp_dir.mkdir(parents=True, exist_ok=True)

    selected_music_path = match_payload["selected_music_path"]
    selected_plan = next(
        plan for plan in match_payload["plans"] if plan["music_path"] == selected_music_path
    )

    clips = _normalize_render_clips(selected_plan["clips"], config.render.output_fps)
    reporter.start(f"Rendering {len(clips)} final clips from {selected_music_path}.")
    total_duration = round(selected_plan["audio_excerpt_start"] + sum(float(clip["duration"]) for clip in clips), 6)
    concat_video_path = _ensure_video_assets(config, clips, reporter)
    audio_excerpt_path = _ensure_audio_excerpt(
        config,
        selected_music_path,
        selected_plan["audio_excerpt_start"],
        total_duration,
        reporter,
        (len(clips) + 1.15) / max(len(clips) + 2, 1),
    )

    final_output_path = config.paths.build_dir / "stage_05_final_video.mp4"
    reporter.update(
        (len(clips) + 1.5) / max(len(clips) + 2, 1),
        "Preparing source hit audio cache.",
    )
    hit_audio_mix_path = _ensure_hit_audio_mix(config, clips)
    reporter.update(
        (len(clips) + 1.7) / max(len(clips) + 2, 1),
        "Mixing music with cached source hit audio.",
    )
    _mux_final_audio(
        config,
        concat_video_path,
        audio_excerpt_path,
        hit_audio_mix_path,
        final_output_path,
    )

    reporter.complete(f"Final video saved to {final_output_path.name}.")
    return {
        "stage": "stage_05_render_final_video",
        "selected_music_path": selected_music_path,
        "final_video_path": str(final_output_path.relative_to(config.paths.project_root)),
        "clip_count": len(clips),
    }
