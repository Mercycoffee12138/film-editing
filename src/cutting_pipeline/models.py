from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TrimmedVideoRecord:
    source_path: str
    trimmed_path: str
    original_duration: float
    trim_start: float
    trim_end: float
    trimmed_duration: float


@dataclass(frozen=True)
class FightSegmentRecord:
    source_path: str
    trimmed_path: str
    video_duration: float
    start: float
    end: float
    peak_time: float
    mean_motion: float
    peak_motion: float
    score: float
    confidence: float = 0.0
    fight_probability: float = 0.0
    detection_source: str = "unknown"
    key_event_times: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class MusicHighlightRecord:
    time: float
    score: float
    energy: float
    accent: float


@dataclass(frozen=True)
class MusicTrackRecord:
    music_path: str
    duration: float
    highlights: list[MusicHighlightRecord]
    beats: list[MusicHighlightRecord] = field(default_factory=list)


@dataclass(frozen=True)
class MatchedClipRecord:
    order: int
    source_path: str
    trimmed_path: str
    clip_start: float
    clip_end: float
    duration: float
    segment_score: float
    source_event_time: float | None = None
    target_highlight_time: float | None = None
    alignment_error: float | None = None


@dataclass(frozen=True)
class MatchPlanRecord:
    music_path: str
    audio_excerpt_start: float
    audio_excerpt_end: float
    output_duration: float
    selected_highlights: list[MusicHighlightRecord]
    timeline_durations: list[float]
    clips: list[MatchedClipRecord]
    plan_score: float
    selected_beats: list[MusicHighlightRecord] = field(default_factory=list)


def to_dict(instance: Any) -> Any:
    return asdict(instance)
