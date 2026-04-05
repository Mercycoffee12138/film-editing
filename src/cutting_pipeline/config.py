from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrimConfig:
    head_trim_seconds: float = 3.0
    tail_trim_seconds: float = 3.0
    minimum_remaining_seconds: float = 5.0


@dataclass(frozen=True)
class MotionConfig:
    analysis_fps: int = 6
    analysis_width: int = 192
    analysis_height: int = 108
    smoothing_seconds: float = 1.5
    threshold_quantile: float = 0.86
    threshold_floor: float = 0.045
    min_segment_seconds: float = 1.2
    merge_gap_seconds: float = 0.7
    calm_threshold_quantile: float = 0.35
    calm_threshold_ceiling: float = 0.03
    calm_min_segment_seconds: float = 1.8
    calm_max_segment_seconds: float = 5.5
    calm_merge_gap_seconds: float = 1.2


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 22050
    frame_length: int = 2048
    hop_length: int = 1024
    min_peak_distance_seconds: float = 2.2
    top_highlights: int = 20
    peak_threshold_quantile: float = 0.8
    beat_min_distance_seconds: float = 0.22
    beat_top_candidates: int = 360
    beat_threshold_quantile: float = 0.6
    beat_sparse_threshold_quantile: float = 0.84
    beat_sparse_distance_scale: float = 1.15
    beat_score_accent_weight: float = 0.86
    beat_score_energy_weight: float = 0.14
    beat_group_min_gap_seconds: float = 0.14
    beat_group_max_gap_seconds: float = 1.2
    beat_group_max_gap_delta_seconds: float = 0.28
    beat_group_min_size: int = 2
    beat_ai_dense_rhythm_bias: bool = True


@dataclass(frozen=True)
class FightAIConfig:
    coarse_window_seconds: float = 8.0
    coarse_stride_seconds: float = 4.0
    coarse_frames_per_window: int = 4
    coarse_min_confidence: float = 0.42
    coarse_merge_gap_seconds: float = 1.4
    fine_anchor_frames: int = 5
    fine_event_context_seconds: float = 0.18
    fine_max_event_candidates: int = 10
    audio_candidate_min_spacing_seconds: float = 0.12
    audio_candidate_peak_quantile: float = 0.86
    max_key_events_per_segment: int = 10
    refined_audio_candidate_min_spacing_seconds: float = 0.12
    refined_audio_candidate_peak_quantile: float = 0.74
    refined_max_event_candidates: int = 8
    refined_visual_analysis_fps: int = 12
    refined_visual_candidate_min_spacing_seconds: float = 0.12
    collision_repeat_score_ratio: float = 0.6
    collision_preview_lead_seconds: float = 0.35
    collision_preview_tail_seconds: float = 0.35
    collision_preview_min_duration_seconds: float = 0.5


@dataclass(frozen=True)
class MatchConfig:
    selected_music_filename: str | None = "002.mp3"
    use_full_track_duration: bool = True
    highlight_cluster_window_seconds: float = 42.0
    max_highlights_per_track: int = 16
    intro_padding_seconds: float = 4.0
    outro_padding_seconds: float = 6.0
    min_clip_seconds: float = 0.5
    max_clip_seconds: float = 3.0
    beat_cut_enabled: bool = True
    beat_cut_min_clip_seconds: float = 0.14
    beat_cut_max_clip_seconds: float = 1.05
    source_reuse_penalty: float = 0.25


@dataclass(frozen=True)
class ReviewConfig:
    enabled: bool = True
    max_candidate_segments: int = 300
    min_confidence: float = 0.55
    min_accepted_segments: int = 24
    relaxed_min_confidence: float = 0.35


@dataclass(frozen=True)
class RenderConfig:
    output_width: int = 1920
    output_height: int = 1080
    output_fps: int = 24
    video_crf: int = 18
    video_preset: str = "veryfast"
    audio_bitrate: str = "192k"
    include_source_hit_audio: bool = True
    music_volume: float = 0.95
    source_hit_volume: float = 9
    source_hit_pre_seconds: float = 0.04
    source_hit_post_seconds: float = 0.1
    source_hit_fade_seconds: float = 0.02
    source_hit_min_segment_seconds: float = 0.12
    source_hit_full_clip_threshold_seconds: float = 0.0


@dataclass(frozen=True)
class StageWindow:
    stage_name: str
    label: str
    start_percent: float
    end_percent: float


@dataclass(frozen=True)
class PipelinePaths:
    project_root: Path
    source_root: Path
    video_source_dir: Path
    music_source_dir: Path
    build_dir: Path
    stage_01_trim_dir: Path
    stage_02_review_frames_dir: Path
    stage_05_clip_dir: Path
    stage_05_temp_dir: Path


@dataclass(frozen=True)
class PipelineConfig:
    paths: PipelinePaths
    trim: TrimConfig = field(default_factory=TrimConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    fight_ai: FightAIConfig = field(default_factory=FightAIConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    match: MatchConfig = field(default_factory=MatchConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    stage_windows: tuple[StageWindow, ...] = (
        StageWindow("stage_01_trim_videos", "Trim Videos", 0.0, 18.0),
        StageWindow("stage_02_detect_fight_segments", "Detect Fight Segments", 18.0, 38.0),
        StageWindow("stage_02_review_fight_segments", "Review Fight Segments", 38.0, 50.0),
        StageWindow("stage_02_extract_collision_events", "Extract Collision Events", 50.0, 60.0),
        StageWindow("stage_03_detect_music_highlights", "Detect Music Highlights", 60.0, 75.0),
        StageWindow("stage_04_match_segments", "Match Segments", 75.0, 87.0),
        StageWindow("stage_05_render_final_video", "Render Final Video", 87.0, 100.0),
    )


def build_default_config(project_root: Path) -> PipelineConfig:
    source_root = project_root / "source"
    build_dir = project_root / "build"

    paths = PipelinePaths(
        project_root=project_root,
        source_root=source_root,
        video_source_dir=source_root / "video",
        music_source_dir=source_root / "music",
        build_dir=build_dir,
        stage_01_trim_dir=build_dir / "stage_01_trimmed_videos",
        stage_02_review_frames_dir=build_dir / "stage_02_review_frames",
        stage_05_clip_dir=build_dir / "stage_05_render_clips",
        stage_05_temp_dir=build_dir / "stage_05_temp",
    )
    return PipelineConfig(paths=paths)
