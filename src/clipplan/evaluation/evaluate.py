from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from clipplan.router.common import GroundTruthClip
from clipplan.router.metrics import PredictedClip, ndcg_at_k


def _load(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}.")
    return payload


def _ground_truth(row: dict[str, Any]) -> list[GroundTruthClip]:
    values = row.get("ground_truth") or row.get("relevant_clips") or row.get("clips") or []
    result: list[GroundTruthClip] = []
    for clip in values:
        timestamp = clip.get("timestamp") or clip.get("moment")
        start = clip.get("start", timestamp[0] if isinstance(timestamp, list) and len(timestamp) >= 2 else 0.0)
        end = clip.get("end", timestamp[1] if isinstance(timestamp, list) and len(timestamp) >= 2 else 0.0)
        result.append(
            GroundTruthClip(
                video_name=str(clip.get("video_name") or clip.get("video_id") or ""),
                start=float(start),
                end=float(end),
                relevance=float(clip.get("relevance", clip.get("score", 1.0))),
            )
        )
    return result


def _predictions(row: dict[str, Any]) -> list[PredictedClip]:
    result: list[PredictedClip] = []
    for clip in row.get("clips", []):
        result.append(
            PredictedClip(
                video_name=str(clip.get("video_name") or clip.get("video_id") or ""),
                start=float(clip.get("start", clip.get("start_time", 0.0))),
                end=float(clip.get("end", clip.get("end_time", 0.0))),
                score=float(clip.get("score", clip.get("relevance", 0.0))),
            )
        )
    return result


def evaluate(prediction_path: Path, annotation_path: Path, k: int, iou_threshold: float) -> dict[str, Any]:
    annotations = {str(row.get("query_id", row.get("qid", row.get("id")))): row for row in _load(annotation_path)}
    predictions = _load(prediction_path)
    per_query: list[dict[str, Any]] = []
    for row in predictions:
        query_id = str(row.get("query_id", row.get("qid", row.get("id"))))
        if query_id not in annotations:
            continue
        score, gains = ndcg_at_k(_predictions(row), _ground_truth(annotations[query_id]), k=k, iou_threshold=iou_threshold)
        per_query.append({"query_id": query_id, "ndcg": score, "matched_gains": gains})
    mean_ndcg = sum(row["ndcg"] for row in per_query) / len(per_query) if per_query else 0.0
    return {"queries": len(per_query), f"ndcg@{k}": mean_ndcg, "iou_threshold": iou_threshold, "per_query": per_query}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ranked video clip predictions with temporal NDCG.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args.predictions, args.annotations, args.k, args.iou_threshold)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
