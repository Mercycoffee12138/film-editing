from __future__ import annotations

from dataclasses import dataclass

from .config import StageWindow


@dataclass(frozen=True)
class StageReporter:
    window: StageWindow

    def update(self, fraction: float, message: str) -> None:
        clamped = max(0.0, min(1.0, fraction))
        percent = self.window.start_percent + (
            (self.window.end_percent - self.window.start_percent) * clamped
        )
        print(f"[{percent:6.2f}%] {self.window.stage_name}: {message}", flush=True)

    def start(self, message: str) -> None:
        self.update(0.0, message)

    def complete(self, message: str) -> None:
        self.update(1.0, message)


class ProgressReporter:
    def __init__(self, stage_windows: tuple[StageWindow, ...]) -> None:
        self._stage_lookup = {window.stage_name: window for window in stage_windows}

    def stage(self, stage_name: str) -> StageReporter:
        return StageReporter(self._stage_lookup[stage_name])
