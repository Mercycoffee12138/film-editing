#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from cutting_pipeline.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged video editing pipeline.")
    parser.add_argument(
        "--start-stage",
        default="stage_01_trim_videos",
        help="Start from a specific stage, for example stage_03_detect_music_highlights.",
    )
    parser.add_argument(
        "--videos",
        default="",
        help="Comma-separated source video filenames, for example film1.mp4 or film1.mp4,film2.mp4.",
    )
    parser.add_argument(
        "--analysis-label",
        default="",
        help="Optional output folder label under build/. Defaults to a label derived from selected videos.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_videos = tuple(item.strip() for item in args.videos.split(",") if item.strip()) or None
    run_pipeline(
        PROJECT_ROOT,
        start_stage=args.start_stage,
        selected_video_filenames=selected_videos,
        analysis_label=args.analysis_label.strip() or None,
    )
