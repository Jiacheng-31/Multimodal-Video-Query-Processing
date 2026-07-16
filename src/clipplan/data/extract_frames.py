from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, interval_seconds: float) -> None:
    if interval_seconds <= 0:
        raise ValueError("The sampling interval must be positive.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "%06d.jpg"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video_path),
        "-vf", f"fps=1/{interval_seconds}",
        "-q:v", "2", str(output_pattern),
    ]
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract uniformly sampled video frames with ffmpeg.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0, help="Frame interval in seconds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_frames(args.video, args.output_dir, args.interval)


if __name__ == "__main__":
    main()
