from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .ann import build_hnsw, l2_normalize
from .dataset import clean_text, frame_path, infer_duration, load_captions, load_duration_map, load_video_names
from .dataset import write_json
from .encoders import CLIPBackbone, crop_image, mean_pool_window, open_image
from .proposals import EntityProposalStore
from .schema import GRANULARITIES, INDEX_VERSION, FeatureMeta


def write_metadata(path: Path, rows: list[FeatureMeta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row.to_json(), ensure_ascii=False) + "\n")


def select_video_shard(video_names: list[str], num_shards: int, shard_index: int) -> list[str]:
    num_shards = max(1, int(num_shards))
    shard_index = int(shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    if num_shards == 1:
        return video_names
    return [video for idx, video in enumerate(video_names) if idx % num_shards == shard_index]


def save_granularity(
    output_dir: Path,
    name: str,
    features: list[np.ndarray],
    metadata: list[FeatureMeta],
    *,
    ann_backend: str,
    hnsw_ef_construction: int,
    hnsw_m: int,
) -> dict[str, Any]:
    gran_dir = output_dir / name
    gran_dir.mkdir(parents=True, exist_ok=True)
    if features:
        arr = l2_normalize(np.stack(features).astype("float32"))
    else:
        arr = np.zeros((0, 1), dtype="float32")
    np.save(gran_dir / "features.npy", arr)
    write_metadata(gran_dir / "metadata.jsonl", metadata)
    if ann_backend == "hnsw" and arr.shape[0] > 0:
        build_hnsw(arr, gran_dir / "hnsw.bin", ef_construction=hnsw_ef_construction, m=hnsw_m)
    return {"feature_count": int(arr.shape[0]), "dim": int(arr.shape[1])}


def action_windows(frame_count: int, window: int, stride: int) -> list[list[int]]:
    if frame_count <= 0:
        return []
    window = max(1, int(window))
    stride = max(1, int(stride))
    windows: list[list[int]] = []
    for start in range(0, frame_count, stride):
        idx = list(range(start, min(frame_count, start + window)))
        if idx:
            windows.append(idx)
    return windows


def build_index(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.ann_backend == "hnsw":
        try:
            import hnswlib  # noqa: F401
        except Exception as exc:
            raise RuntimeError("ClipPlan indexing requires hnswlib for --ann-backend hnsw") from exc

    if args.entity_proposals_path == "" and not args.allow_debug_frame_entities:
        raise RuntimeError(
            "Entity features require --entity-proposals-path with SAM2/offline proposal boxes. "
            "Use --allow-debug-frame-entities only for smoke tests."
        )

    encoder = CLIPBackbone(args.clip_model, device=args.device, dtype=args.dtype)
    proposals = EntityProposalStore(
        Path(args.entity_proposals_path) if args.entity_proposals_path else None,
        min_area_ratio=args.entity_min_area_ratio,
        max_area_ratio=args.entity_max_area_ratio,
        max_regions_per_frame=args.max_entity_regions,
        allow_debug_frame_entities=args.allow_debug_frame_entities,
    )
    duration_map = load_duration_map(dataset_root)
    all_video_names = load_video_names(dataset_root, max_videos=args.max_videos)
    video_names = select_video_shard(all_video_names, args.num_shards, args.shard_index)

    context_features: list[np.ndarray] = []
    context_meta: list[FeatureMeta] = []
    entity_features: list[np.ndarray] = []
    entity_meta: list[FeatureMeta] = []
    action_features: list[np.ndarray] = []
    action_meta: list[FeatureMeta] = []
    video_meta: dict[str, dict[str, Any]] = {}

    started = time.perf_counter()
    for video_idx, video_name in enumerate(video_names, start=1):
        captions = load_captions(dataset_root, video_name)
        if not captions:
            continue
        frame_images = []
        valid_caps = []
        for cap in captions:
            image = open_image(frame_path(dataset_root, video_name, cap))
            if image is None:
                continue
            valid_caps.append(cap)
            frame_images.append(image)
        if not valid_caps:
            continue

        frame_features = encoder.encode_images(frame_images, batch_size=args.image_batch_size)
        duration = infer_duration(video_name, valid_caps, duration_map)
        preview = next((clean_text(cap.get("caption")) for cap in valid_caps if clean_text(cap.get("caption"))), "")
        video_meta[video_name] = {"duration": duration, "frame_count": len(valid_caps), "preview_text": preview}

        for cap, feat in zip(valid_caps, frame_features):
            feature_id = len(context_features)
            caption = clean_text(cap.get("caption"))
            context_features.append(feat)
            context_meta.append(
                FeatureMeta(
                    feature_id=feature_id,
                    video_name=video_name,
                    granularity="context",
                    timestamp=float(cap.get("timestamp", 0.0)),
                    frame_file=str(cap.get("frame_file")),
                    preview_text=caption,
                )
            )

        if args.allow_debug_frame_entities and not args.entity_proposals_path:
            for cap, feat in zip(valid_caps, frame_features):
                feature_id = len(entity_features)
                entity_features.append(feat)
                entity_meta.append(
                    FeatureMeta(
                        feature_id=feature_id,
                        video_name=video_name,
                        granularity="entity",
                        timestamp=float(cap.get("timestamp", 0.0)),
                        frame_file=str(cap.get("frame_file")),
                        crop_box=None,
                        preview_text=clean_text(cap.get("caption")),
                    )
                )
        else:
            crop_images = []
            crop_rows: list[FeatureMeta] = []
            for cap, image in zip(valid_caps, frame_images):
                frame_file = str(cap.get("frame_file"))
                boxes = proposals.boxes(video_name, frame_file, image_width=image.width, image_height=image.height)
                for box in boxes:
                    crop = crop_image(image, box)
                    if crop is None:
                        continue
                    crop_images.append(crop)
                    crop_rows.append(
                        FeatureMeta(
                            feature_id=-1,
                            video_name=video_name,
                            granularity="entity",
                            timestamp=float(cap.get("timestamp", 0.0)),
                            frame_file=frame_file,
                            crop_box=box,
                            preview_text=clean_text(cap.get("caption")),
                        )
                    )
            if crop_images:
                crop_features = encoder.encode_images(crop_images, batch_size=args.image_batch_size)
                for row, feat in zip(crop_rows, crop_features):
                    feature_id = len(entity_features)
                    entity_features.append(feat)
                    entity_meta.append(
                        FeatureMeta(
                            feature_id=feature_id,
                            video_name=row.video_name,
                            granularity=row.granularity,
                            timestamp=row.timestamp,
                            frame_file=row.frame_file,
                            crop_box=row.crop_box,
                            preview_text=row.preview_text,
                        )
                    )

        for window in action_windows(len(valid_caps), args.action_window, args.action_stride):
            feature_id = len(action_features)
            feat = mean_pool_window(frame_features, window)
            start_cap = valid_caps[window[0]]
            end_cap = valid_caps[window[-1]]
            preview_text = " ".join(clean_text(valid_caps[idx].get("caption")) for idx in window)[:240]
            action_features.append(feat)
            action_meta.append(
                FeatureMeta(
                    feature_id=feature_id,
                    video_name=video_name,
                    granularity="action",
                    timestamp=float(start_cap.get("timestamp", 0.0)),
                    end_timestamp=float(end_cap.get("timestamp", start_cap.get("timestamp", 0.0))),
                    frame_file=str(start_cap.get("frame_file")),
                    preview_text=preview_text,
                )
            )

        if args.progress_every > 0 and video_idx % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(f"indexed {video_idx}/{len(video_names)} videos in {elapsed:.1f}s", flush=True)

    stats = {}
    stats["context"] = save_granularity(
        output_dir,
        "context",
        context_features,
        context_meta,
        ann_backend=args.ann_backend,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_m=args.hnsw_m,
    )
    stats["entity"] = save_granularity(
        output_dir,
        "entity",
        entity_features,
        entity_meta,
        ann_backend=args.ann_backend,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_m=args.hnsw_m,
    )
    stats["action"] = save_granularity(
        output_dir,
        "action",
        action_features,
        action_meta,
        ann_backend=args.ann_backend,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_m=args.hnsw_m,
    )
    write_json(output_dir / "video_meta.json", video_meta)
    write_json(
        output_dir / "manifest.json",
        {
            "index_version": INDEX_VERSION,
            "dataset_root": str(dataset_root.resolve()),
            "clip_model": args.clip_model,
            "ann_backend": args.ann_backend,
            "granularities": list(GRANULARITIES),
            "action_window": args.action_window,
            "action_stride": args.action_stride,
            "entity_proposals_path": args.entity_proposals_path,
            "allow_debug_frame_entities": args.allow_debug_frame_entities,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "source_video_count": len(all_video_names),
            "shard_video_count": len(video_names),
            "stats": stats,
        },
    )
    print(f"wrote multi-granularity index: {output_dir}", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ClipPlan multi-granularity CLIP/HNSW indices.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--image-batch-size", type=int, default=64)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--entity-proposals-path", default="")
    parser.add_argument("--entity-min-area-ratio", type=float, default=0.005)
    parser.add_argument("--entity-max-area-ratio", type=float, default=0.80)
    parser.add_argument("--max-entity-regions", type=int, default=8)
    parser.add_argument("--allow-debug-frame-entities", action="store_true")
    parser.add_argument("--action-window", type=int, default=4)
    parser.add_argument("--action-stride", type=int, default=2)
    parser.add_argument("--ann-backend", choices=["hnsw", "exact"], default="hnsw")
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--progress-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    build_index(parse_args())


if __name__ == "__main__":
    main()
