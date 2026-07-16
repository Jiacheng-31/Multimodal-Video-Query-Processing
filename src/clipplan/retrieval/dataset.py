from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from .schema import FusedCandidate


QVH_NAME_RE = re.compile(r"_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clean_text(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def load_captions(dataset_root: Path, video_name: str) -> list[dict[str, Any]]:
    caption_path = dataset_root / "caption" / f"{video_name}.jsonl"
    captions: list[dict[str, Any]] = []
    with caption_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(json.loads(line))
    return captions


def frame_path(dataset_root: Path, video_name: str, caption: dict[str, Any]) -> Path:
    return dataset_root / "frames" / video_name / str(caption["frame_file"])


def load_video_names(dataset_root: Path, max_videos: int = 0) -> list[str]:
    video_list_path = dataset_root / "annotations" / "video_list.json"
    if video_list_path.exists():
        names = [str(item) for item in load_json(video_list_path)]
    else:
        names = [path.stem for path in sorted((dataset_root / "caption").glob("*.jsonl"))]
    if max_videos > 0:
        return names[:max_videos]
    return names


def load_duration_map(dataset_root: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    corpus_path = dataset_root / "annotations" / "video_corpus.json"
    if corpus_path.exists():
        data = load_json(corpus_path)
        if isinstance(data, dict):
            for key, value in data.items():
                try:
                    result[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
    return result


def infer_duration(video_name: str, captions: list[dict[str, Any]], duration_map: dict[str, float]) -> float:
    if video_name in duration_map:
        return float(duration_map[video_name])
    match = QVH_NAME_RE.search(video_name)
    if match:
        return max(0.0, float(match.group(2)) - float(match.group(1)))
    timestamps = [float(cap.get("timestamp", 0.0)) for cap in captions]
    if not timestamps:
        return 0.0
    step = timestamps[-1] - timestamps[-2] if len(timestamps) >= 2 else 2.0
    return timestamps[-1] + max(0.0, step)


def build_video_meta(dataset_root: Path, max_videos: int = 0) -> dict[str, dict[str, Any]]:
    duration_map = load_duration_map(dataset_root)
    meta: dict[str, dict[str, Any]] = {}
    for video_name in load_video_names(dataset_root, max_videos=max_videos):
        caption_path = dataset_root / "caption" / f"{video_name}.jsonl"
        if not caption_path.exists():
            continue
        captions = load_captions(dataset_root, video_name)
        if not captions:
            continue
        preview = next((clean_text(cap.get("caption")) for cap in captions if clean_text(cap.get("caption"))), "")
        meta[video_name] = {
            "duration": infer_duration(video_name, captions, duration_map),
            "frame_count": len(captions),
            "preview_text": preview,
        }
    return meta


def gt_entries(item: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(item.get("ground_truth"), list):
        return list(item["ground_truth"])
    if isinstance(item.get("relevant_moment"), list):
        return list(item["relevant_moment"])
    if item.get("video_name"):
        return [item]
    return []


def gt_video_names(item: dict[str, Any], *, positive_only: bool = True) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for entry in gt_entries(item):
        name = entry.get("video_name") or entry.get("video_id")
        if not name:
            continue
        try:
            relevance = float(entry.get("relevance", 1.0))
        except (TypeError, ValueError):
            relevance = 1.0
        if positive_only and relevance <= 0:
            continue
        text_name = str(name)
        if text_name not in seen:
            seen.add(text_name)
            names.append(text_name)
    return names


def attach_candidates(
    item: dict[str, Any],
    candidates: list[FusedCandidate | dict[str, Any]],
    *,
    method: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    row = dict(item)
    serialized = [candidate.to_json() if isinstance(candidate, FusedCandidate) else candidate for candidate in candidates]
    row["candidate_videos"] = serialized
    row["retrieval"] = {"method": method, **config}
    return row


def recall_stats(rows: list[dict[str, Any]], top_h: int) -> dict[str, Any]:
    covered = 0
    total_gt = 0
    hit_queries = 0
    query_count = 0
    for row in rows:
        gt = set(gt_video_names(row))
        if not gt:
            continue
        top = {str(candidate["video_name"]) for candidate in row.get("candidate_videos", [])[:top_h]}
        hit_count = len(gt & top)
        covered += hit_count
        total_gt += len(gt)
        hit_queries += int(hit_count > 0)
        query_count += 1
    return {
        "recall_at_h": covered / max(1, total_gt),
        "query_hit_at_h": hit_queries / max(1, query_count),
        "covered_gt_videos": covered,
        "total_gt_videos": total_gt,
        "hit_queries": hit_queries,
        "query_count": query_count,
    }


def random_negative_videos(
    all_videos: list[str],
    exclude: set[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    pool = [video for video in all_videos if video not in exclude]
    if count <= 0 or not pool:
        return []
    if len(pool) <= count:
        rng.shuffle(pool)
        return pool
    return rng.sample(pool, count)
