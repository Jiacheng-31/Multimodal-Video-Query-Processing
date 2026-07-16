from __future__ import annotations

import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .common import CandidateVideo, GroundTruthClip, QueryEpisode


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_captions(path: Path) -> list[dict[str, Any]]:
    captions: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(json.loads(line))
    return captions


def _coerce_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clean_text(value: Any) -> str:
    return str(value or "").replace("\n", " ").strip()


def _dataset_name(dataset_root: Path) -> str:
    name = dataset_root.name.lower()
    if "tvr" in name:
        return "tvr"
    if "qvh" in name:
        return "qvh"
    return name or "dataset"


def _ground_truth_entries(item: dict[str, Any]) -> list[dict[str, Any]]:
    if item.get("ground_truth"):
        return list(item["ground_truth"])
    if item.get("relevant_moment"):
        return list(item["relevant_moment"])
    if "video_name" in item and "timestamp" in item:
        return [item]
    if "video_name" in item:
        moments = item.get("target_relevant_segments") or []
        return [
            {
                "video_name": item["video_name"],
                "timestamp": moment.get("timestamp"),
                "duration": item.get("duration", moment.get("duration", 150)),
                "relevance": moment.get("relevance", 1),
            }
            for moment in moments
        ]
    return []


def _candidate_entries(item: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("candidate_videos", "candidates", "retrieved_candidates"):
        value = item.get(key)
        if isinstance(value, list) and value:
            return list(value)
    return _ground_truth_entries(item)


def _has_explicit_candidates(item: dict[str, Any]) -> bool:
    for key in ("candidate_videos", "candidates", "retrieved_candidates"):
        value = item.get(key)
        if isinstance(value, list) and value:
            return True
    return False


def _collect_candidate_metadata(item: dict[str, Any]) -> OrderedDict[str, dict[str, Any]]:
    candidates: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for moment in _candidate_entries(item):
        video_name = str(
            moment.get("video_name")
            or moment.get("video_id")
            or moment.get("id")
            or item.get("video_name")
            or ""
        )
        if not video_name:
            continue
        duration = _coerce_float(moment.get("duration"), _coerce_float(item.get("duration"), 150.0))
        score = _coerce_float(
            moment.get("retrieval_score"),
            _coerce_float(moment.get("fused_score"), _coerce_float(moment.get("score"), _coerce_float(moment.get("similarity"), None))),
        )
        caption = _clean_text(moment.get("preview_text") or moment.get("caption"))
        if video_name not in candidates:
            candidates[video_name] = {
                "video_name": video_name,
                "duration": duration or 150.0,
                "retrieval_score": score,
                "preview_text": caption,
            }
            continue
        current = candidates[video_name]
        if current.get("retrieval_score") is None and score is not None:
            current["retrieval_score"] = score
        if not current.get("preview_text") and caption:
            current["preview_text"] = caption
    return candidates


def _collect_ground_truth(item: dict[str, Any]) -> list[GroundTruthClip]:
    clips: list[GroundTruthClip] = []
    seen: set[tuple[str, float, float, float]] = set()
    for moment in _ground_truth_entries(item):
        video_name = str(moment.get("video_name") or moment.get("video_id") or item.get("video_name") or "")
        ts = moment.get("timestamp")
        if not video_name or not isinstance(ts, (list, tuple)) or len(ts) < 2:
            continue
        start = _coerce_float(ts[0])
        end = _coerce_float(ts[1])
        relevance = _coerce_float(moment.get("relevance"), 1.0)
        if start is None or end is None or end <= start or relevance is None or relevance <= 0:
            continue
        key = (video_name, round(start, 4), round(end, 4), round(relevance, 4))
        if key in seen:
            continue
        seen.add(key)
        clips.append(GroundTruthClip(video_name=video_name, start=start, end=end, relevance=relevance))
    clips.sort(key=lambda clip: clip.relevance, reverse=True)
    return clips


def _load_candidate(
    metadata: dict[str, Any],
    dataset_root: Path,
    rank: int,
    total: int,
) -> CandidateVideo | None:
    video_name = str(metadata["video_name"])
    caption_path = dataset_root / "caption" / f"{video_name}.jsonl"
    frame_dir = dataset_root / "frames" / video_name
    if not caption_path.exists() or not frame_dir.is_dir():
        return None
    captions = load_captions(caption_path)
    if not captions:
        return None
    timestamps = [float(cap["timestamp"]) for cap in captions]
    return CandidateVideo(
        video_name=video_name,
        captions=captions,
        frame_dir=frame_dir,
        duration=float(metadata.get("duration") or 150.0),
        timestamps=timestamps,
        retrieval_rank=rank,
        retrieval_total=total,
        retrieval_score=_coerce_float(metadata.get("retrieval_score"), None),
        preview_text=_clean_text(metadata.get("preview_text")),
    )


def _episode_budget(
    candidates: list[CandidateVideo],
    *,
    budget_mode: str,
    budget_ratio: float,
    visual_token_cost: int,
    budget_per_frame_text_tokens: int,
    budget_visual_frame_ratio: float,
) -> int:
    frames = sum(len(candidate.captions) for candidate in candidates)
    if frames <= 0:
        return 0
    if budget_mode == "full_visual_ratio":
        full_visual_cost = frames * max(1, int(visual_token_cost))
        return max(1, math.ceil(full_visual_cost * float(budget_ratio)))
    text_budget = frames * max(0, int(budget_per_frame_text_tokens))
    visual_frames = math.ceil(frames * max(0.0, float(budget_visual_frame_ratio)))
    visual_budget = visual_frames * max(1, int(visual_token_cost))
    return max(1, int(text_budget + visual_budget))


def build_episode(
    item: dict[str, Any],
    dataset_root: Path,
    *,
    budget_ratio: float,
    visual_token_cost: int,
    budget_mode: str = "text_plus_visual",
    budget_per_frame_text_tokens: int = 8,
    budget_visual_frame_ratio: float = 0.05,
    max_candidates: int = 0,
) -> QueryEpisode | None:
    metadata = _collect_candidate_metadata(item)
    if max_candidates > 0:
        metadata = OrderedDict(list(metadata.items())[:max_candidates])
    total = len(metadata)
    if total <= 0:
        return None

    candidates: list[CandidateVideo] = []
    for rank, info in enumerate(metadata.values(), start=1):
        candidate = _load_candidate(info, dataset_root, rank=rank, total=total)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None

    ground_truth = _collect_ground_truth(item)
    if not _has_explicit_candidates(item):
        loaded_names = {candidate.video_name for candidate in candidates}
        ground_truth = [clip for clip in ground_truth if clip.video_name in loaded_names]
    if not ground_truth:
        return None
    for idx, candidate in enumerate(candidates, start=1):
        candidate.retrieval_rank = idx
        candidate.retrieval_total = len(candidates)

    return QueryEpisode(
        query_id=int(item["query_id"]),
        query=str(item["query"]),
        candidates=candidates,
        ground_truth=ground_truth,
        route_budget=_episode_budget(
            candidates,
            budget_mode=budget_mode,
            budget_ratio=budget_ratio,
            visual_token_cost=visual_token_cost,
            budget_per_frame_text_tokens=budget_per_frame_text_tokens,
            budget_visual_frame_ratio=budget_visual_frame_ratio,
        ),
        dataset_root=dataset_root,
        dataset_name=_dataset_name(dataset_root),
    )


def load_episodes(
    annotation_path: Path,
    dataset_root: Path,
    *,
    limit: int,
    budget_ratio: float,
    visual_token_cost: int,
    budget_mode: str = "text_plus_visual",
    budget_per_frame_text_tokens: int = 8,
    budget_visual_frame_ratio: float = 0.05,
    max_candidates: int = 0,
) -> list[QueryEpisode]:
    data = load_json(annotation_path)
    episodes: list[QueryEpisode] = []
    for item in data:
        episode = build_episode(
            item,
            dataset_root,
            budget_ratio=budget_ratio,
            visual_token_cost=visual_token_cost,
            budget_mode=budget_mode,
            budget_per_frame_text_tokens=budget_per_frame_text_tokens,
            budget_visual_frame_ratio=budget_visual_frame_ratio,
            max_candidates=max_candidates,
        )
        if episode is None:
            continue
        episodes.append(episode)
        if limit > 0 and len(episodes) >= limit:
            break
    return episodes
