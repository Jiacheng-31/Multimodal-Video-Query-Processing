from __future__ import annotations

import math
from dataclasses import dataclass

from .common import GroundTruthClip


@dataclass(frozen=True)
class PredictedClip:
    video_name: str
    start: float
    end: float
    score: float


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter_start = max(a_start, b_start)
    inter_end = min(a_end, b_end)
    if inter_start >= inter_end:
        return 0.0
    inter = inter_end - inter_start
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0


def dcg(relevances: list[float], k: int) -> float:
    total = 0.0
    for idx, relevance in enumerate(relevances[:k], start=1):
        total += (2.0 ** float(relevance) - 1.0) / math.log2(idx + 1)
    return total


def ndcg_at_k(
    predictions: list[PredictedClip],
    ground_truth: list[GroundTruthClip],
    *,
    k: int,
    iou_threshold: float,
) -> tuple[float, list[float]]:
    if k <= 0 or not ground_truth:
        return 0.0, []
    sorted_gt = sorted(ground_truth, key=lambda clip: clip.relevance, reverse=True)
    ideal = dcg([clip.relevance for clip in sorted_gt], k)
    if ideal <= 0:
        return 0.0, []

    unmatched = set(range(len(ground_truth)))
    ranked = sorted(predictions, key=lambda clip: clip.score, reverse=True)
    assigned: list[float] = []
    for pred in ranked[:k]:
        best_idx = None
        best_iou = 0.0
        for idx in list(unmatched):
            gt = ground_truth[idx]
            if gt.video_name != pred.video_name:
                continue
            overlap = temporal_iou(pred.start, pred.end, gt.start, gt.end)
            if overlap > best_iou:
                best_idx = idx
                best_iou = overlap
        if best_idx is not None and best_iou >= iou_threshold:
            assigned.append(float(ground_truth[best_idx].relevance))
            unmatched.remove(best_idx)
        else:
            assigned.append(0.0)
    return dcg(assigned, k) / ideal, assigned
