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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(PROJECT_ROOT, start_stage=args.start_stage)
