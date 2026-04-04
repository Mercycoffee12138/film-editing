from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .config import PipelineConfig
from .ffmpeg_tools import get_media_duration, run_command
from .json_io import write_json
from .models import TrimmedVideoRecord
from .progress import StageReporter


def _relative(path: Path, project_root: Path) -> str:
    return str(path.relative_to(project_root))


def run(config: PipelineConfig, reporter: StageReporter) -> dict:
    paths = config.paths
    paths.stage_01_trim_dir.mkdir(parents=True, exist_ok=True)

    source_videos = sorted(paths.video_source_dir.glob("*.mp4"))
    trimmed_records: list[TrimmedVideoRecord] = []

    reporter.start(f"Preparing to trim {len(source_videos)} videos.")

    for index, source_path in enumerate(source_videos, start=1):
        progress = (index - 1) / max(len(source_videos), 1)
        reporter.update(progress, f"Trimming {source_path.name} ({index}/{len(source_videos)}).")

        duration = get_media_duration(source_path)
        trim_start = config.trim.head_trim_seconds
        trim_end = config.trim.tail_trim_seconds
        trimmed_duration = duration - trim_start - trim_end

        if trimmed_duration < config.trim.minimum_remaining_seconds:
            raise ValueError(f"Trimmed duration too short for {source_path.name}")

        output_path = paths.stage_01_trim_dir / f"{source_path.stem}_stage_01_trimmed.mp4"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{trim_start:.3f}",
            "-i",
            str(source_path),
            "-t",
            f"{trimmed_duration:.3f}",
            "-map",
            "0",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
        run_command(command)

        trimmed_records.append(
            TrimmedVideoRecord(
                source_path=_relative(source_path, paths.project_root),
                trimmed_path=_relative(output_path, paths.project_root),
                original_duration=round(duration, 3),
                trim_start=trim_start,
                trim_end=trim_end,
                trimmed_duration=round(trimmed_duration, 3),
            )
        )

    manifest = {
        "stage": "stage_01_trim_videos",
        "trimmed_videos": [asdict(record) for record in trimmed_records],
    }
    manifest_path = paths.build_dir / "stage_01_trim_manifest.json"
    write_json(manifest_path, manifest)

    reporter.complete(f"Finished trimming {len(trimmed_records)} videos.")
    return manifest
