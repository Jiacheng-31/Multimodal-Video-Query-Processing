from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .dataset import (
    attach_candidates,
    build_video_meta,
    gt_entries,
    load_json,
    load_video_names,
    random_negative_videos,
    recall_stats,
    write_json,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def query_key(row: dict[str, Any]) -> str:
    if row.get("query_id") is not None:
        return str(row["query_id"])
    return str(row.get("query") or "")


def ranked_positive_videos(row: dict[str, Any]) -> list[str]:
    best: dict[str, tuple[float, float, int]] = {}
    for index, entry in enumerate(gt_entries(row)):
        name = entry.get("video_name") or entry.get("video_id") or row.get("video_name")
        if not name:
            continue
        relevance = _safe_float(entry.get("relevance"), 1.0)
        if relevance <= 0:
            continue
        similarity = _safe_float(entry.get("similarity"), 0.0)
        video_name = str(name)
        priority = (relevance, similarity, -index)
        if video_name not in best or priority > best[video_name]:
            best[video_name] = priority
    return [
        video_name
        for video_name, _ in sorted(
            best.items(),
            key=lambda item: (-item[1][0], -item[1][1], -item[1][2]),
        )
    ]


def row_candidate_entries(row: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in row.get("candidate_videos", []) or []:
        name = candidate.get("video_name") or candidate.get("video_id")
        if not name:
            continue
        video_name = str(name)
        if video_name in seen:
            continue
        seen.add(video_name)
        entries.append(dict(candidate, video_name=video_name))
    return entries


def load_recall_entries(path: str) -> dict[str, list[dict[str, Any]]]:
    if not path:
        return {}
    rows = load_json(Path(path))
    return {query_key(row): row_candidate_entries(row) for row in rows}


def make_candidate(
    video_name: str,
    rank: int,
    score: float,
    video_meta: dict[str, dict[str, Any]],
    *,
    oracle_positive: bool,
    source: str,
    recall_rank: int | None = None,
) -> dict[str, Any]:
    meta = video_meta.get(video_name, {})
    payload = {
        "video_name": video_name,
        "retrieval_rank": rank,
        "retrieval_score": float(score),
        "duration": float(meta.get("duration", 0.0)),
        "frame_count": int(meta.get("frame_count", 0)),
        "preview_text": str(meta.get("preview_text") or ""),
        "oracle_positive": oracle_positive,
        "candidate_source": source,
    }
    if recall_rank is not None:
        payload["original_recall_rank"] = int(recall_rank)
    return payload


def recall_score(entry: dict[str, Any], default: float) -> float:
    return _safe_float(
        entry.get("retrieval_score", entry.get("fused_score", entry.get("score", entry.get("similarity")))),
        default,
    )


def inject_positive(
    selected: list[dict[str, Any]],
    positive_video: str,
    positive_set: set[str],
    video_meta: dict[str, dict[str, Any]],
    *,
    top_h: int,
) -> bool:
    if positive_video in {item["video_name"] for item in selected}:
        return False
    injected = {
        "video_name": positive_video,
        "score": 1.0,
        "source": "ground_truth_injected",
        "recall_rank": None,
    }
    if len(selected) < top_h:
        selected.append(injected)
        return True
    for idx in range(len(selected) - 1, -1, -1):
        if selected[idx]["video_name"] not in positive_set:
            selected[idx] = injected
            return True
    return False


def build_hybrid_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_root = Path(args.dataset_root)
    rows = load_json(Path(args.annotation_path))
    if args.limit > 0:
        rows = rows[: args.limit]
    all_videos = load_video_names(dataset_root)
    video_meta = build_video_meta(dataset_root)
    recall_by_query = load_recall_entries(args.recall_candidates_path)
    rng = random.Random(args.seed)
    output: list[dict[str, Any]] = []

    total_positive_videos = 0
    included_positive_videos = 0
    injected_positive_videos = 0
    recall_positive_videos = 0
    overflow_positive_videos = 0
    missing_positive_videos = 0
    recall_candidate_videos = 0
    random_fallback_videos = 0
    queries_without_recall = 0

    for row in rows:
        ranked_positives = ranked_positive_videos(row)
        positives = [video for video in ranked_positives if video in video_meta]
        missing_positives = [video for video in ranked_positives if video not in video_meta]
        positive_set = set(positives)
        selected: list[dict[str, Any]] = []
        selected_names: set[str] = set()

        recall_entries = recall_by_query.get(query_key(row), [])
        queries_without_recall += int(not recall_entries)
        for recall_rank, entry in enumerate(recall_entries, start=1):
            video = str(entry.get("video_name") or "")
            if not video or video in selected_names or video not in video_meta:
                continue
            selected.append(
                {
                    "video_name": video,
                    "score": recall_score(entry, max(0.0, 1.0 - recall_rank * 1e-4)),
                    "source": "external_recall",
                    "recall_rank": recall_rank,
                }
            )
            selected_names.add(video)
            if len(selected) >= args.top_h:
                break

        recall_candidate_videos += len(selected)
        recall_positive_videos += sum(1 for item in selected if item["video_name"] in positive_set)

        injected_this_row = 0
        for positive in positives:
            if positive in {item["video_name"] for item in selected}:
                continue
            if inject_positive(selected, positive, positive_set, video_meta, top_h=args.top_h):
                injected_this_row += 1
            else:
                break

        selected_names = {item["video_name"] for item in selected}
        if args.use_input_candidates and len(selected) < args.top_h:
            for entry in row_candidate_entries(row):
                video = entry["video_name"]
                if video in selected_names or video not in video_meta:
                    continue
                selected.append(
                    {
                        "video_name": video,
                        "score": recall_score(entry, max(0.0, 0.5 - len(selected) * 1e-4)),
                        "source": "input_recall",
                        "recall_rank": None,
                    }
                )
                selected_names.add(video)
                if len(selected) >= args.top_h:
                    break

        if len(selected) < args.top_h:
            negatives = random_negative_videos(all_videos, selected_names, args.top_h - len(selected), rng)
            random_fallback_videos += len(negatives)
            selected.extend(
                {
                    "video_name": video,
                    "score": max(0.0, 0.1 - idx * 1e-4),
                    "source": "random_negative",
                    "recall_rank": None,
                }
                for idx, video in enumerate(negatives, start=1)
            )

        selected = selected[: args.top_h]
        selected_names = {item["video_name"] for item in selected}
        included = [video for video in positives if video in selected_names]
        overflow = [video for video in positives if video not in selected_names]
        injected_positive_videos += injected_this_row
        total_positive_videos += len(ranked_positives)
        included_positive_videos += len(included)
        overflow_positive_videos += len(overflow)
        missing_positive_videos += len(missing_positives)

        candidates = [
            make_candidate(
                item["video_name"],
                rank=rank,
                score=float(item["score"]),
                video_meta=video_meta,
                oracle_positive=item["video_name"] in positive_set,
                source=str(item["source"]),
                recall_rank=item.get("recall_rank"),
            )
            for rank, item in enumerate(selected, start=1)
        ]
        output.append(
            attach_candidates(
                row,
                candidates,
                method="hybrid_recall_with_gt_injection",
                config={
                    "top_h": args.top_h,
                    "ndcg_k": args.ndcg_k,
                    "seed": args.seed,
                    "recall_candidates_path": args.recall_candidates_path,
                    "gt_injection_strategy": "replace_tail_non_gt_then_append_if_space",
                    "positive_priority": "relevance_similarity",
                    "positive_video_count": len(ranked_positives),
                    "included_positive_count": len(included),
                    "injected_positive_count": injected_this_row,
                    "overflow_positive_count": len(overflow),
                    "missing_positive_count": len(missing_positives),
                    "use_input_candidates": args.use_input_candidates,
                    "purpose": "router_and_baseline_fair_candidate_pool",
                },
            )
        )

    stats = recall_stats(output, args.top_h)
    stats.update(
        {
            "rows": len(output),
            "top_h": args.top_h,
            "ndcg_k": args.ndcg_k,
            "dataset_root": str(dataset_root),
            "recall_candidates_path": args.recall_candidates_path,
            "total_positive_videos": total_positive_videos,
            "included_positive_videos": included_positive_videos,
            "injected_positive_videos": injected_positive_videos,
            "recall_positive_videos": recall_positive_videos,
            "overflow_positive_videos": overflow_positive_videos,
            "missing_positive_videos": missing_positive_videos,
            "recall_candidate_videos": recall_candidate_videos,
            "random_fallback_videos": random_fallback_videos,
            "queries_without_recall": queries_without_recall,
        }
    )
    return output, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build hybrid recall candidate pools with forced GT inclusion.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--annotation-path", required=True)
    parser.add_argument("--recall-candidates-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--top-h", type=int, default=60)
    parser.add_argument("--ndcg-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--use-input-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use candidate_videos already present in the annotation as fallback after recall and GT injection.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, stats = build_hybrid_rows(args)
    if args.dry_run:
        print(json.dumps({"stats": stats, "sample": rows[0] if rows else {}}, ensure_ascii=False, indent=2)[:12000])
        return
    output_path = Path(args.output_path)
    write_json(output_path, rows)
    write_json(output_path.with_suffix(".metrics.json"), stats)
    print(f"wrote hybrid candidates: {output_path}", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
