from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .ffmpeg_tools import run_command
from .progress import StageReporter


def _render_clip(
    config: PipelineConfig,
    clip: dict,
    output_path: Path,
) -> None:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{clip['clip_start']:.3f}",
        "-t",
        f"{clip['duration']:.3f}",
        "-i",
        str(config.paths.project_root / clip["trimmed_path"]),
        "-an",
        "-vf",
        (
            f"scale={config.render.output_width}:{config.render.output_height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={config.render.output_width}:{config.render.output_height}:(ow-iw)/2:(oh-ih)/2,"
            "format=yuv420p"
        ),
        "-r",
        str(config.render.output_fps),
        "-c:v",
        "libx264",
        "-preset",
        config.render.video_preset,
        "-crf",
        str(config.render.video_crf),
        str(output_path),
    ]
    run_command(command)


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


def _clip_hit_audio_segments(clips: list[dict], config: PipelineConfig) -> list[dict]:
    if not config.render.include_source_hit_audio:
        return []

    segments: list[dict] = []
    timeline_cursor = 0.0
    for clip in clips:
        source_event_time = clip.get("source_event_time")
        if source_event_time is None:
            timeline_cursor += float(clip["duration"])
            continue

        clip_start = float(clip["clip_start"])
        clip_end = float(clip["clip_end"])
        event_time = float(source_event_time)
        local_start = max(clip_start, event_time - config.render.source_hit_pre_seconds)
        local_end = min(clip_end, event_time + config.render.source_hit_post_seconds)
        duration = max(local_end - local_start, 0.0)
        if duration <= 0.01:
            timeline_cursor += float(clip["duration"])
            continue

        timeline_offset = timeline_cursor + max(local_start - clip_start, 0.0)
        segments.append(
            {
                "trimmed_path": clip["trimmed_path"],
                "source_start": round(local_start, 3),
                "duration": round(duration, 3),
                "timeline_offset": round(timeline_offset, 3),
            }
        )
        timeline_cursor += float(clip["duration"])

    return segments


def _mux_final_audio(
    config: PipelineConfig,
    concat_video_path: Path,
    audio_excerpt_path: Path,
    clips: list[dict],
    final_output_path: Path,
) -> None:
    hit_segments = _clip_hit_audio_segments(clips, config)
    if not hit_segments:
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
    ]

    for segment in hit_segments:
        command.extend(
            [
                "-i",
                str(config.paths.project_root / segment["trimmed_path"]),
            ]
        )

    filter_parts = [f"[1:a]volume={config.render.music_volume:.3f}[music]"]
    mix_inputs = ["[music]"]
    fade_seconds = max(config.render.source_hit_fade_seconds, 0.0)

    for index, segment in enumerate(hit_segments, start=2):
        duration = float(segment["duration"])
        fade_out_start = max(duration - fade_seconds, 0.0)
        delay_ms = max(int(round(float(segment["timeline_offset"]) * 1000.0)), 0)
        filter_parts.append(
            (
                f"[{index}:a]"
                f"atrim=start={float(segment['source_start']):.3f}:duration={duration:.3f},"
                "asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d={min(fade_seconds, duration * 0.5):.3f},"
                f"afade=t=out:st={fade_out_start:.3f}:d={min(fade_seconds, duration * 0.5):.3f},"
                f"volume={config.render.source_hit_volume:.3f},"
                f"adelay={delay_ms}|{delay_ms}[hit{index - 2}]"
            )
        )
        mix_inputs.append(f"[hit{index - 2}]")

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


def run(config: PipelineConfig, reporter: StageReporter, match_payload: dict) -> dict:
    config.paths.stage_05_clip_dir.mkdir(parents=True, exist_ok=True)
    config.paths.stage_05_temp_dir.mkdir(parents=True, exist_ok=True)

    selected_music_path = match_payload["selected_music_path"]
    selected_plan = next(
        plan for plan in match_payload["plans"] if plan["music_path"] == selected_music_path
    )

    clips = selected_plan["clips"]
    reporter.start(f"Rendering {len(clips)} final clips from {selected_music_path}.")

    rendered_clip_paths: list[Path] = []
    for index, clip in enumerate(clips, start=1):
        progress = (index - 1) / max(len(clips) + 2, 1)
        reporter.update(progress, f"Rendering clip {index}/{len(clips)} from {Path(clip['trimmed_path']).name}.")
        output_path = config.paths.stage_05_clip_dir / f"clip_{index:03d}_stage_05.mp4"
        _render_clip(config, clip, output_path)
        rendered_clip_paths.append(output_path)

    audio_excerpt_path = config.paths.stage_05_temp_dir / "stage_05_audio_excerpt.m4a"
    reporter.update(
        len(clips) / max(len(clips) + 2, 1),
        f"Rendering music excerpt from {Path(selected_music_path).name}.",
    )
    _render_audio_excerpt(
        config,
        selected_music_path,
        selected_plan["audio_excerpt_start"],
        selected_plan["audio_excerpt_end"],
        audio_excerpt_path,
    )

    concat_list_path = config.paths.stage_05_temp_dir / "stage_05_concat_list.txt"
    concat_lines = [f"file '{path.resolve()}'" for path in rendered_clip_paths]
    concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    concat_video_path = config.paths.stage_05_temp_dir / "stage_05_concat_video.mp4"
    reporter.update(
        (len(clips) + 1) / max(len(clips) + 2, 1),
        "Concatenating rendered clips.",
    )
    concat_command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
        str(concat_video_path),
    ]
    run_command(concat_command)

    final_output_path = config.paths.build_dir / "stage_05_final_video.mp4"
    reporter.update(
        (len(clips) + 1.5) / max(len(clips) + 2, 1),
        "Mixing music with source hit audio.",
    )
    _mux_final_audio(
        config,
        concat_video_path,
        audio_excerpt_path,
        clips,
        final_output_path,
    )

    reporter.complete(f"Final video saved to {final_output_path.name}.")
    return {
        "stage": "stage_05_render_final_video",
        "selected_music_path": selected_music_path,
        "final_video_path": str(final_output_path.relative_to(config.paths.project_root)),
        "clip_count": len(clips),
    }
