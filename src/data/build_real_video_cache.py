"""Build a cache of real-video spatiotemporal crops for the real-video nuisance.

Reads the CC-licensed clips under data/real_video/, extracts grayscale
8-frame 16x16 crops with visible motion, and stores them as a compressed
npz archive. The benchmark generator replays these crops forward or backward
as the directional nuisance process, so the "spurious arrow" is literally the
arrow of time of real video.

Usage:
  python -m src.data.build_real_video_cache \
      --src data/real_video --out data/real_video/cache_g16_L8.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frames(path: str, max_frames: int = 2000) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 1, 1), np.uint8)


def extract_crops(
    frames: np.ndarray,
    *,
    grid: int,
    length: int,
    t_stride: int,
    short_side: int,
    per_clip: int,
    min_motion: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(frames) < length * t_stride + 1:
        return np.zeros((0, length, grid, grid), np.uint8)
    h, w = frames.shape[1:]
    scale = short_side / min(h, w)
    frames = np.stack([
        cv2.resize(f, (max(grid, int(w * scale)), max(grid, int(h * scale))),
                   interpolation=cv2.INTER_AREA)
        for f in frames
    ])
    H, W = frames.shape[1:]
    crops = []
    attempts = 0
    while len(crops) < per_clip and attempts < per_clip * 20:
        attempts += 1
        t0 = int(rng.integers(0, len(frames) - length * t_stride))
        r0 = int(rng.integers(0, H - grid + 1))
        c0 = int(rng.integers(0, W - grid + 1))
        clip = frames[t0 : t0 + length * t_stride : t_stride, r0 : r0 + grid, c0 : c0 + grid]
        clip = clip.astype(np.float32)
        # Require visible temporal change so the crop actually carries motion.
        motion = np.abs(np.diff(clip, axis=0)).mean()
        if motion < min_motion:
            continue
        lo, hi = clip.min(), clip.max()
        if hi - lo < 8:
            continue
        clip = (clip - lo) / (hi - lo)
        crops.append((clip * 255).astype(np.uint8))
    return np.stack(crops) if crops else np.zeros((0, length, grid, grid), np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/real_video")
    parser.add_argument("--out", default="data/real_video/cache_g16_L8.npz")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--t-stride", type=int, default=3)
    parser.add_argument("--short-side", type=int, default=48)
    parser.add_argument("--per-clip", type=int, default=3000)
    parser.add_argument("--min-motion", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    src = Path(args.src)
    all_crops, meta = [], []
    for path in sorted(src.glob("clip*.webm")) + sorted(src.glob("clip*.mp4")):
        frames = read_frames(str(path))
        crops = extract_crops(
            frames, grid=args.grid, length=args.length, t_stride=args.t_stride,
            short_side=args.short_side, per_clip=args.per_clip,
            min_motion=args.min_motion, rng=rng,
        )
        meta.append({"clip": path.name, "frames": int(len(frames)), "crops": int(len(crops))})
        print(f"{path.name}: frames={len(frames)} crops={len(crops)}")
        if len(crops):
            all_crops.append(crops)
    crops = np.concatenate(all_crops) if all_crops else np.zeros((0,), np.uint8)
    rng.shuffle(crops)
    np.savez_compressed(args.out, crops=crops)
    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"params": vars(args), "clips": meta}, indent=2), encoding="utf-8")
    print("cache:", args.out, crops.shape, f"{Path(args.out).stat().st_size/1e6:.1f}MB")


if __name__ == "__main__":
    main()
