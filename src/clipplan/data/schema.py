from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClipLabel:
    video_name: str
    start: float
    end: float
    relevance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_name": self.video_name,
            "start": self.start,
            "end": self.end,
            "relevance": self.relevance,
        }


@dataclass
class QueryRecord:
    query_id: str
    query: str
    clips: list[ClipLabel] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = dict(self.metadata)
        row.update(
            {
                "query_id": self.query_id,
                "query": self.query,
                "ground_truth": [clip.to_dict() for clip in self.clips],
            }
        )
        return row
