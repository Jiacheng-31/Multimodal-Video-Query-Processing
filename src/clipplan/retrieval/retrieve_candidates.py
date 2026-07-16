from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .ann import load_granularity_index, load_manifest
from .dataset import attach_candidates, load_json, recall_stats, write_json
from .encoders import CLIPBackbone
from .schema import GRANULARITIES, FeatureHit, FusedCandidate, VideoHit


def aggregate_feature_hits(hits: list[FeatureHit], video_depth: int) -> list[VideoHit]:
    best: dict[str, FeatureHit] = {}
    for hit in hits:
        current = best.get(hit.video_name)
        if current is None or hit.score > current.score:
            best[hit.video_name] = hit
    ranked = sorted(best.values(), key=lambda hit: hit.score, reverse=True)[:video_depth]
    return [
        VideoHit(video_name=hit.video_name, score=hit.score, rank=rank, best_feature=hit)
        for rank, hit in enumerate(ranked, start=1)
    ]


def rrf_fuse(
    rankings: dict[str, list[VideoHit]],
    video_meta: dict[str, dict[str, Any]],
    *,
    top_h: int,
    kappa: float,
) -> list[FusedCandidate]:
    fused: dict[str, dict[str, Any]] = {}
    for granularity, video_hits in rankings.items():
        for video_hit in video_hits:
            item = fused.setdefault(
                video_hit.video_name,
                {
                    "retrieval_score": 0.0,
                    "granularity": {},
                },
            )
            item["retrieval_score"] += 1.0 / (float(kappa) + video_hit.rank)
            item["granularity"][granularity] = {
                "rank": video_hit.rank,
                "score": video_hit.score,
                "feature_id": video_hit.best_feature.feature_id,
                "timestamp": video_hit.best_feature.meta.timestamp,
                "end_timestamp": video_hit.best_feature.meta.end_timestamp,
                "frame_file": video_hit.best_feature.meta.frame_file,
                "crop_box": list(video_hit.best_feature.meta.crop_box)
                if video_hit.best_feature.meta.crop_box is not None
                else None,
                "preview_text": video_hit.best_feature.meta.preview_text,
            }
    ordered = sorted(fused.items(), key=lambda item: item[1]["retrieval_score"], reverse=True)[:top_h]
    candidates: list[FusedCandidate] = []
    for rank, (video_name, item) in enumerate(ordered, start=1):
        meta = video_meta.get(video_name, {})
        preview = str(meta.get("preview_text") or "")
        if not preview:
            for granularity in GRANULARITIES:
                preview = str(item["granularity"].get(granularity, {}).get("preview_text") or "")
                if preview:
                    break
        candidates.append(
            FusedCandidate(
                video_name=video_name,
                retrieval_rank=rank,
                retrieval_score=float(item["retrieval_score"]),
                duration=float(meta.get("duration", 0.0)),
                frame_count=int(meta.get("frame_count", 0)),
                preview_text=preview,
                granularity=item["granularity"],
            )
        )
    return candidates


class MultiGranularityRetriever:
    def __init__(self, index_dir: Path, *, backend: str, ef_search: int, clip_model: str, device: str, dtype: str) -> None:
        self.index_dir = index_dir
        self.manifest = load_manifest(index_dir)
        self.backend = backend
        self.indices = {
            name: load_granularity_index(index_dir, name, backend=backend, ef_search=ef_search)
            for name in GRANULARITIES
        }
        self.encoder = CLIPBackbone(clip_model, device=device, dtype=dtype)
        self.video_meta = load_json(index_dir / "video_meta.json")

    def retrieve(self, query: str, *, top_p: int, video_depth: int, top_h: int, rrf_kappa: float) -> list[FusedCandidate]:
        query_emb = self.encoder.encode_text([query], batch_size=1)[0]
        rankings: dict[str, list[VideoHit]] = {}
        for granularity, index in self.indices.items():
            feature_hits = index.search(query_emb, top_p)
            rankings[granularity] = aggregate_feature_hits(feature_hits, video_depth)
        return rrf_fuse(rankings, self.video_meta, top_h=top_h, kappa=rrf_kappa)


def build_annotations(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    annotation_path = Path(args.annotation_path)
    rows = load_json(annotation_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    manifest = load_manifest(index_dir)
    clip_model = args.clip_model or str(manifest.get("clip_model") or "openai/clip-vit-large-patch14")
    backend = args.ann_backend or str(manifest.get("ann_backend") or "hnsw")
    retriever = MultiGranularityRetriever(
        index_dir,
        backend=backend,
        ef_search=args.hnsw_ef_search,
        clip_model=clip_model,
        device=args.device,
        dtype=args.dtype,
    )

    output = []
    config = {
        "index_version": manifest.get("index_version"),
        "index_dir": str(index_dir),
        "top_h": args.top_h,
        "top_p": args.top_p,
        "video_depth": args.video_depth,
        "rrf_kappa": args.rrf_kappa,
        "granularities": list(GRANULARITIES),
        "ann_backend": backend,
    }
    for idx, row in enumerate(rows, start=1):
        candidates = retriever.retrieve(
            str(row.get("query") or ""),
            top_p=args.top_p,
            video_depth=args.video_depth,
            top_h=args.top_h,
            rrf_kappa=args.rrf_kappa,
        )
        output.append(attach_candidates(row, candidates, method="chapter4_clip_hnsw_rrf", config=config))
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"retrieved {idx}/{len(rows)} queries", flush=True)

    stats = recall_stats(output, args.top_h)
    stats.update({"rows": len(output), **config})
    if args.dry_run:
        print(json.dumps({"stats": stats, "sample": output[0] if output else {}}, ensure_ascii=False, indent=2)[:12000])
        return

    output_path = Path(args.output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    write_json(output_path, output)
    write_json(metrics_path, stats)
    print(f"wrote candidates: {output_path}", flush=True)
    print(f"wrote metrics: {metrics_path}", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve candidates with ClipPlan multi-granularity RRF.")
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--clip-model", default="", help="Defaults to the model recorded in index manifest.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--ann-backend", choices=["", "hnsw", "exact"], default="")
    parser.add_argument("--hnsw-ef-search", type=int, default=128)
    parser.add_argument("--top-p", type=int, default=2000)
    parser.add_argument("--video-depth", type=int, default=300)
    parser.add_argument("--top-h", type=int, default=60)
    parser.add_argument("--rrf-kappa", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    build_annotations(parse_args())


if __name__ == "__main__":
    main()
