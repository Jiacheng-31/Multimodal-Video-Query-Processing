from __future__ import annotations

from dataclasses import dataclass
from typing import Any


GRANULARITIES = ("context", "entity", "action")
INDEX_VERSION = "chapter4_clip_hnsw_v1"


@dataclass(frozen=True)
class FeatureMeta:
    feature_id: int
    video_name: str
    granularity: str
    timestamp: float | None = None
    frame_file: str | None = None
    end_timestamp: float | None = None
    crop_box: tuple[float, float, float, float] | None = None
    preview_text: str = ""

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "feature_id": self.feature_id,
            "video_name": self.video_name,
            "granularity": self.granularity,
            "preview_text": self.preview_text,
        }
        if self.timestamp is not None:
            payload["timestamp"] = self.timestamp
        if self.end_timestamp is not None:
            payload["end_timestamp"] = self.end_timestamp
        if self.frame_file is not None:
            payload["frame_file"] = self.frame_file
        if self.crop_box is not None:
            payload["crop_box"] = list(self.crop_box)
        return payload

    @staticmethod
    def from_json(payload: dict[str, Any]) -> "FeatureMeta":
        crop = payload.get("crop_box")
        return FeatureMeta(
            feature_id=int(payload["feature_id"]),
            video_name=str(payload["video_name"]),
            granularity=str(payload["granularity"]),
            timestamp=float(payload["timestamp"]) if payload.get("timestamp") is not None else None,
            frame_file=str(payload["frame_file"]) if payload.get("frame_file") is not None else None,
            end_timestamp=float(payload["end_timestamp"]) if payload.get("end_timestamp") is not None else None,
            crop_box=tuple(float(x) for x in crop) if isinstance(crop, list) and len(crop) == 4 else None,
            preview_text=str(payload.get("preview_text") or ""),
        )


@dataclass(frozen=True)
class FeatureHit:
    feature_id: int
    video_name: str
    score: float
    rank: int
    meta: FeatureMeta


@dataclass(frozen=True)
class VideoHit:
    video_name: str
    score: float
    rank: int
    best_feature: FeatureHit


@dataclass(frozen=True)
class FusedCandidate:
    video_name: str
    retrieval_rank: int
    retrieval_score: float
    duration: float
    frame_count: int
    preview_text: str
    granularity: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "video_name": self.video_name,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": self.retrieval_score,
            "duration": self.duration,
            "frame_count": self.frame_count,
            "preview_text": self.preview_text,
            "granularity": self.granularity,
        }
