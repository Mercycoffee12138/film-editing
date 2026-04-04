from __future__ import annotations

from pathlib import Path

from .config import build_default_config
from .json_io import read_json
from .progress import ProgressReporter
from . import stage_01_trim_videos
from . import stage_02_detect_fight_segments
from . import stage_02_review_fight_segments
from . import stage_02_extract_collision_events
from . import stage_03_detect_music_highlights
from . import stage_04_match_segments
from . import stage_05_render_final_video


STAGE_SEQUENCE = (
    "stage_01_trim_videos",
    "stage_02_detect_fight_segments",
    "stage_02_review_fight_segments",
    "stage_02_extract_collision_events",
    "stage_03_detect_music_highlights",
    "stage_04_match_segments",
    "stage_05_render_final_video",
)


def _artifact_path(project_root: Path, stage_name: str) -> Path:
    artifact_map = {
        "stage_01_trim_videos": project_root / "build" / "stage_01_trim_manifest.json",
        "stage_02_detect_fight_segments": project_root / "build" / "stage_02_fight_segments.json",
        "stage_02_review_fight_segments": project_root / "build" / "stage_02_reviewed_fight_segments.json",
        "stage_02_extract_collision_events": project_root / "build" / "stage_02_collision_events.json",
        "stage_03_detect_music_highlights": project_root / "build" / "stage_03_music_highlights.json",
        "stage_04_match_segments": project_root / "build" / "stage_04_match_plan.json",
    }
    return artifact_map[stage_name]


def _load_required_artifact(project_root: Path, stage_name: str) -> dict:
    path = _artifact_path(project_root, stage_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot start from a later stage because required artifact is missing: {path}"
        )
    return read_json(path)


def run_pipeline(project_root: Path, start_stage: str = "stage_01_trim_videos") -> None:
    config = build_default_config(project_root)
    config.paths.build_dir.mkdir(parents=True, exist_ok=True)

    if start_stage not in STAGE_SEQUENCE:
        supported = ", ".join(STAGE_SEQUENCE)
        raise ValueError(f"Unsupported start stage '{start_stage}'. Supported stages: {supported}")

    reporter = ProgressReporter(config.stage_windows)

    if STAGE_SEQUENCE.index(start_stage) <= STAGE_SEQUENCE.index("stage_01_trim_videos"):
        trim_manifest = stage_01_trim_videos.run(
            config,
            reporter.stage("stage_01_trim_videos"),
        )
    else:
        trim_manifest = _load_required_artifact(project_root, "stage_01_trim_videos")

    if STAGE_SEQUENCE.index(start_stage) <= STAGE_SEQUENCE.index("stage_02_detect_fight_segments"):
        fight_segments = stage_02_detect_fight_segments.run(
            config,
            reporter.stage("stage_02_detect_fight_segments"),
            trim_manifest,
        )
    else:
        fight_segments = _load_required_artifact(project_root, "stage_02_detect_fight_segments")

    if STAGE_SEQUENCE.index(start_stage) <= STAGE_SEQUENCE.index("stage_02_review_fight_segments"):
        reviewed_fight_segments = stage_02_review_fight_segments.run(
            config,
            reporter.stage("stage_02_review_fight_segments"),
            fight_segments,
        )
    else:
        reviewed_fight_segments = _load_required_artifact(project_root, "stage_02_review_fight_segments")

    if STAGE_SEQUENCE.index(start_stage) <= STAGE_SEQUENCE.index("stage_02_extract_collision_events"):
        collision_event_segments = stage_02_extract_collision_events.run(
            config,
            reporter.stage("stage_02_extract_collision_events"),
            reviewed_fight_segments,
        )
    else:
        collision_event_segments = _load_required_artifact(project_root, "stage_02_extract_collision_events")

    if STAGE_SEQUENCE.index(start_stage) <= STAGE_SEQUENCE.index("stage_03_detect_music_highlights"):
        music_highlights = stage_03_detect_music_highlights.run(
            config,
            reporter.stage("stage_03_detect_music_highlights"),
        )
    else:
        music_highlights = _load_required_artifact(project_root, "stage_03_detect_music_highlights")

    if STAGE_SEQUENCE.index(start_stage) <= STAGE_SEQUENCE.index("stage_04_match_segments"):
        match_payload = stage_04_match_segments.run(
            config,
            reporter.stage("stage_04_match_segments"),
            collision_event_segments,
            music_highlights,
        )
    else:
        match_payload = _load_required_artifact(project_root, "stage_04_match_segments")

    stage_05_render_final_video.run(
        config,
        reporter.stage("stage_05_render_final_video"),
        match_payload,
    )
