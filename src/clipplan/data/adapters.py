from __future__ import annotations

from typing import Any, Iterable

from .schema import ClipLabel, QueryRecord


def _first(row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return default


def _clip_from_mapping(value: dict[str, Any], parent: dict[str, Any]) -> ClipLabel | None:
    video_name = str(_first(value, ("video_name", "video_id", "vid"), _first(parent, ("video_name", "video_id", "vid"), "")))
    timestamp = _first(value, ("timestamp", "timestamps", "moment"))
    start = _first(value, ("start", "start_time"))
    end = _first(value, ("end", "end_time"))
    if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
        start, end = timestamp[0], timestamp[1]
    if not video_name or start is None or end is None:
        return None
    relevance = _first(value, ("relevance", "score", "saliency"), 1.0)
    return ClipLabel(video_name, float(start), float(end), float(relevance))


def normalize_record(row: dict[str, Any], index: int) -> QueryRecord:
    query_id = str(_first(row, ("query_id", "qid", "id"), index))
    query = str(_first(row, ("query", "sentence", "description", "text"), "")).strip()
    if not query:
        raise ValueError(f"Record {query_id} does not contain a query string.")

    raw_clips = _first(row, ("ground_truth", "relevant_clips", "moments", "clips"), [])
    if isinstance(raw_clips, dict):
        raw_clips = [raw_clips]
    clips = [clip for item in raw_clips if isinstance(item, dict) if (clip := _clip_from_mapping(item, row))]

    if not clips:
        direct = _clip_from_mapping(row, row)
        if direct is not None:
            clips.append(direct)

    consumed = {
        "query_id", "qid", "id", "query", "sentence", "description", "text",
        "ground_truth", "relevant_clips", "moments", "clips",
    }
    metadata = {key: value for key, value in row.items() if key not in consumed}
    return QueryRecord(query_id=query_id, query=query, clips=clips, metadata=metadata)


def normalize_annotations(rows: list[dict[str, Any]]) -> list[QueryRecord]:
    return [normalize_record(row, index) for index, row in enumerate(rows)]
