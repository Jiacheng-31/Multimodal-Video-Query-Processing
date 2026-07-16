from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .ann import build_hnsw, l2_normalize
from .dataset import read_jsonl, write_json
from .schema import GRANULARITIES, INDEX_VERSION, FeatureMeta


def write_metadata(path: Path, rows: list[FeatureMeta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def merge_granularity(
    shard_dirs: list[Path],
    output_dir: Path,
    name: str,
    *,
    ann_backend: str,
    hnsw_ef_construction: int,
    hnsw_m: int,
) -> dict[str, int]:
    arrays: list[np.ndarray] = []
    metadata: list[FeatureMeta] = []
    for shard_dir in shard_dirs:
        gran_dir = shard_dir / name
        feature_path = gran_dir / "features.npy"
        meta_path = gran_dir / "metadata.jsonl"
        if not feature_path.exists() or not meta_path.exists():
            continue
        arr = np.load(feature_path)
        rows = [FeatureMeta.from_json(row) for row in read_jsonl(meta_path)]
        if arr.shape[0] != len(rows):
            raise RuntimeError(f"{gran_dir}: features={arr.shape[0]} metadata={len(rows)}")
        arrays.append(arr.astype("float32", copy=False))
        for row in rows:
            metadata.append(
                FeatureMeta(
                    feature_id=len(metadata),
                    video_name=row.video_name,
                    granularity=row.granularity,
                    timestamp=row.timestamp,
                    frame_file=row.frame_file,
                    end_timestamp=row.end_timestamp,
                    crop_box=row.crop_box,
                    preview_text=row.preview_text,
                )
            )

    gran_out = output_dir / name
    gran_out.mkdir(parents=True, exist_ok=True)
    if arrays:
        features = l2_normalize(np.concatenate(arrays, axis=0).astype("float32"))
    else:
        features = np.zeros((0, 1), dtype="float32")
    np.save(gran_out / "features.npy", features)
    write_metadata(gran_out / "metadata.jsonl", metadata)
    if ann_backend == "hnsw" and features.shape[0] > 0:
        build_hnsw(features, gran_out / "hnsw.bin", ef_construction=hnsw_ef_construction, m=hnsw_m)
    return {"feature_count": int(features.shape[0]), "dim": int(features.shape[1])}


def merge_indices(args: argparse.Namespace) -> None:
    shard_dirs = [Path(path) for path in args.shard_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = [load_json(path / "manifest.json") for path in shard_dirs if (path / "manifest.json").exists()]
    if not manifests:
        raise RuntimeError("No shard manifests found.")

    video_meta: dict[str, Any] = {}
    for shard_dir in shard_dirs:
        meta_path = shard_dir / "video_meta.json"
        if meta_path.exists():
            video_meta.update(load_json(meta_path))

    stats = {
        name: merge_granularity(
            shard_dirs,
            output_dir,
            name,
            ann_backend=args.ann_backend,
            hnsw_ef_construction=args.hnsw_ef_construction,
            hnsw_m=args.hnsw_m,
        )
        for name in GRANULARITIES
    }
    write_json(output_dir / "video_meta.json", video_meta)
    first = manifests[0]
    write_json(
        output_dir / "manifest.json",
        {
            "index_version": INDEX_VERSION,
            "dataset_root": first.get("dataset_root"),
            "clip_model": first.get("clip_model"),
            "ann_backend": args.ann_backend,
            "granularities": list(GRANULARITIES),
            "action_window": first.get("action_window"),
            "action_stride": first.get("action_stride"),
            "merged_from": [str(path) for path in shard_dirs],
            "stats": stats,
        },
    )
    print(f"wrote merged multi-granularity index: {output_dir}", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded ClipPlan multi-granularity indices.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--ann-backend", choices=["hnsw", "exact"], default="hnsw")
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-m", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    merge_indices(parse_args())


if __name__ == "__main__":
    main()
