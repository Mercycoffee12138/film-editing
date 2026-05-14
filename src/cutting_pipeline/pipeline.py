from __future__ import annotations

from pathlib import Path

from .config import build_default_config
from .json_io import read_json, write_json
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


def _artifact_path(build_dir: Path, stage_name: str) -> Path:
    artifact_map = {
        "stage_01_trim_videos": build_dir / "stage_01_trim_manifest.json",
        "stage_02_detect_fight_segments": build_dir / "stage_02_fight_segments.json",
        "stage_02_review_fight_segments": build_dir / "stage_02_reviewed_fight_segments.json",
        "stage_02_extract_collision_events": build_dir / "stage_02_collision_events.json",
        "stage_03_detect_music_highlights": build_dir / "stage_03_music_highlights.json",
        "stage_04_match_segments": build_dir / "stage_04_match_plan.json",
    }
    return artifact_map[stage_name]


def _resolve_effective_start_stage(build_dir: Path, requested_start_stage: str) -> tuple[str, str | None]:
    requested_index = STAGE_SEQUENCE.index(requested_start_stage)
    for stage_name in STAGE_SEQUENCE[:requested_index]:
        if not _artifact_path(build_dir, stage_name).exists():
            return stage_name, stage_name
    return requested_start_stage, None


def _load_required_artifact(build_dir: Path, stage_name: str) -> dict:
    path = _artifact_path(build_dir, stage_name)
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot start from a later stage because required artifact is missing: {path}"
        )
    return read_json(path)


def _skip_collision_event_extraction(config, reporter, reviewed_fight_segments: dict) -> dict:
    reporter.start("Skipping collision extraction for warm editorial style.")
    payload = {
        "stage": "stage_02_extract_collision_events",
        "skipped": True,
        "skip_reason": "warm_style",
        "top_segments": list(reviewed_fight_segments.get("top_segments") or []),
        "calm_segments": list(reviewed_fight_segments.get("calm_segments") or []),
        "fight_segment_exports": list(reviewed_fight_segments.get("fight_segment_exports") or []),
        "reviewed_segments": list(reviewed_fight_segments.get("reviewed_segments") or []),
        "collision_event_preview": {"events": []},
    }
    write_json(config.paths.build_dir / "stage_02_collision_events.json", payload)
    reporter.complete("Collision extraction skipped; warm mode will align clips by pacing instead of impacts.")
    return payload


def run_pipeline(
    project_root: Path,
    start_stage: str = "stage_01_trim_videos",
    selected_video_filenames: tuple[str, ...] | None = None,
    analysis_label: str | None = None,
    editorial_style: str = "fight",
) -> None:
    config = build_default_config(
        project_root,
        selected_video_filenames=selected_video_filenames,
        analysis_label=analysis_label,
        editorial_style=editorial_style,
    )
    config.paths.build_dir.mkdir(parents=True, exist_ok=True)

    if start_stage not in STAGE_SEQUENCE:
        supported = ", ".join(STAGE_SEQUENCE)
        raise ValueError(f"Unsupported start stage '{start_stage}'. Supported stages: {supported}")

    effective_start_stage, fallback_stage = _resolve_effective_start_stage(config.paths.build_dir, start_stage)
    if fallback_stage is not None:
        missing_artifact = _artifact_path(config.paths.build_dir, fallback_stage)
        print(
            (
                f"[pipeline] Requested start stage '{start_stage}', but required artifact is missing: "
                f"{missing_artifact}. Falling back to '{effective_start_stage}'."
            ),
            flush=True,
        )

    reporter = ProgressReporter(config.stage_windows)

    if STAGE_SEQUENCE.index(effective_start_stage) <= STAGE_SEQUENCE.index("stage_01_trim_videos"):
        trim_manifest = stage_01_trim_videos.run(
            config,
            reporter.stage("stage_01_trim_videos"),
        )
    else:
        trim_manifest = _load_required_artifact(config.paths.build_dir, "stage_01_trim_videos")

    if STAGE_SEQUENCE.index(effective_start_stage) <= STAGE_SEQUENCE.index("stage_02_detect_fight_segments"):
        fight_segments = stage_02_detect_fight_segments.run(
            config,
            reporter.stage("stage_02_detect_fight_segments"),
            trim_manifest,
        )
    else:
        fight_segments = _load_required_artifact(config.paths.build_dir, "stage_02_detect_fight_segments")

    if STAGE_SEQUENCE.index(effective_start_stage) <= STAGE_SEQUENCE.index("stage_02_review_fight_segments"):
        reviewed_fight_segments = stage_02_review_fight_segments.run(
            config,
            reporter.stage("stage_02_review_fight_segments"),
            fight_segments,
        )
    else:
        reviewed_fight_segments = _load_required_artifact(config.paths.build_dir, "stage_02_review_fight_segments")

    if STAGE_SEQUENCE.index(effective_start_stage) <= STAGE_SEQUENCE.index("stage_02_extract_collision_events"):
        if config.source.editorial_style == "warm":
            collision_event_segments = _skip_collision_event_extraction(
                config,
                reporter.stage("stage_02_extract_collision_events"),
                reviewed_fight_segments,
            )
        else:
            collision_event_segments = stage_02_extract_collision_events.run(
                config,
                reporter.stage("stage_02_extract_collision_events"),
                reviewed_fight_segments,
            )
    else:
        collision_event_segments = _load_required_artifact(config.paths.build_dir, "stage_02_extract_collision_events")

    if STAGE_SEQUENCE.index(effective_start_stage) <= STAGE_SEQUENCE.index("stage_03_detect_music_highlights"):
        music_highlights = stage_03_detect_music_highlights.run(
            config,
            reporter.stage("stage_03_detect_music_highlights"),
        )
    else:
        music_highlights = _load_required_artifact(config.paths.build_dir, "stage_03_detect_music_highlights")

    if STAGE_SEQUENCE.index(effective_start_stage) <= STAGE_SEQUENCE.index("stage_04_match_segments"):
        match_payload = stage_04_match_segments.run(
            config,
            reporter.stage("stage_04_match_segments"),
            collision_event_segments,
            music_highlights,
        )
    else:
        match_payload = _load_required_artifact(config.paths.build_dir, "stage_04_match_segments")

    stage_05_render_final_video.run(
        config,
        reporter.stage("stage_05_render_final_video"),
        match_payload,
    )
