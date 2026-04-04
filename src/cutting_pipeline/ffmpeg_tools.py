from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Generator

import numpy as np


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{message}")


def run_ffprobe_json(command: list[str]) -> dict:
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def get_media_duration(path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return float(result.stdout.strip())


def iter_gray_frames(
    path: Path,
    fps: int,
    width: int,
    height: int,
    start: float = 0.0,
    duration: float | None = None,
) -> Generator[np.ndarray, None, None]:
    frame_size = width * height
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start > 0.0:
        command.extend(["-ss", f"{max(start, 0.0):.3f}"])
    command.extend(["-i", str(path)])
    if duration is not None:
        command.extend(["-t", f"{max(duration, 0.0):.3f}"])
    command.extend(
        [
            "-vf",
            f"fps={fps},scale={width}:{height}:flags=lanczos,format=gray",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ]
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_size * 8,
    )
    assert process.stdout is not None
    try:
        while True:
            chunk = process.stdout.read(frame_size)
            if len(chunk) < frame_size:
                break
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width)
    finally:
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())


def decode_audio_mono(
    path: Path,
    sample_rate: int,
    start: float = 0.0,
    duration: float | None = None,
) -> np.ndarray:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start > 0.0:
        command.extend(["-ss", f"{max(start, 0.0):.3f}"])
    command.extend(["-i", str(path)])
    if duration is not None:
        command.extend(["-t", f"{max(duration, 0.0):.3f}"])
    command.extend(
        [
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "-",
        ]
    )
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode != 0:
        raise RuntimeError(process.stderr.decode("utf-8", errors="replace").strip())
    samples = np.frombuffer(process.stdout, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return np.zeros(1, dtype=np.float32)
    return samples / 32768.0


def export_video_frame(path: Path, timestamp: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(timestamp, 0.0):.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        "-y",
        str(output_path),
    ]
    run_command(command)


def export_video_clip(
    input_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_start = max(start_time, 0.0)
    safe_duration = max(duration, 0.01)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{safe_start:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{safe_duration:.3f}",
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    run_command(command)


def render_video_clip(
    input_path: Path,
    output_path: Path,
    start_time: float,
    duration: float,
    width: int,
    height: int,
    fps: int,
    video_preset: str,
    video_crf: int,
    audio_bitrate: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_start = max(start_time, 0.0)
    safe_duration = max(duration, 0.01)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{safe_start:.3f}",
        "-t",
        f"{safe_duration:.3f}",
        "-i",
        str(input_path),
        "-vf",
        (
            f"scale={width}:{height}:"
            "force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            "format=yuv420p"
        ),
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-preset",
        video_preset,
        "-crf",
        str(video_crf),
        "-c:a",
        "aac",
        "-b:a",
        audio_bitrate,
        str(output_path),
    ]
    run_command(command)


def concat_video_clips(clip_paths: list[Path], output_path: Path, concat_list_path: Path) -> None:
    if not clip_paths:
        raise ValueError("At least one clip path is required for concatenation.")

    concat_list_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_lines = [f"file '{path.resolve()}'" for path in clip_paths]
    concat_list_path.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    command = [
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
        str(output_path),
    ]
    run_command(command)
