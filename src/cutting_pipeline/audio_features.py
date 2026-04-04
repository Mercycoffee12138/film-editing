from __future__ import annotations

import numpy as np


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or values.size == 0:
        return values
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    return np.convolve(values, kernel, mode="same")


def frame_metric(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if samples.size < frame_length:
        padded = np.pad(samples, (0, frame_length - samples.size))
        return np.asarray([float(np.sqrt(np.mean(np.square(padded)) + 1e-8))], dtype=np.float32)

    frame_values: list[float] = []
    for start in range(0, samples.size - frame_length + 1, hop_length):
        frame = samples[start : start + frame_length]
        frame_values.append(float(np.sqrt(np.mean(np.square(frame)) + 1e-8)))
    return np.asarray(frame_values, dtype=np.float32)


def normalize_robust(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    center = float(np.median(values))
    spread = float(np.percentile(values, 75) - np.percentile(values, 25))
    if spread < 1e-8:
        spread = float(values.std()) + 1e-8
    return (values - center) / spread


def pick_peaks(
    scores: np.ndarray,
    min_distance_frames: int,
    limit: int,
    threshold: float,
) -> list[int]:
    if scores.size < 3:
        return []

    candidates = [
        index
        for index in range(1, len(scores) - 1)
        if scores[index] >= scores[index - 1]
        and scores[index] > scores[index + 1]
        and scores[index] >= threshold
    ]

    kept: list[int] = []
    for index in sorted(candidates, key=lambda item: scores[item], reverse=True):
        if all(abs(index - existing) >= min_distance_frames for existing in kept):
            kept.append(index)
        if len(kept) >= limit:
            break

    return sorted(kept)


def merge_peak_indices(
    primary_indices: list[int],
    secondary_indices: list[int],
    scores: np.ndarray,
    min_distance_frames: int,
    limit: int | None = None,
) -> list[int]:
    kept: list[int] = []

    def _try_add(index: int) -> None:
        nonlocal kept
        if all(abs(index - existing) >= min_distance_frames for existing in kept):
            kept.append(index)

    for index in sorted(set(primary_indices), key=lambda item: scores[item], reverse=True):
        _try_add(index)
        if limit is not None and len(kept) >= limit:
            return sorted(kept)

    for index in sorted(set(secondary_indices), key=lambda item: scores[item], reverse=True):
        _try_add(index)
        if limit is not None and len(kept) >= limit:
            break

    return sorted(kept)
