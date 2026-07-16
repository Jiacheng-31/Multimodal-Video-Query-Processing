"""Qianfan multimodal scorer utilities for ClipPlan."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

import requests
from PIL import Image

from .qianfan import DEFAULT_BASE_URL, DEFAULT_MODEL, QianfanConfig


DEFAULT_DATASET_ROOT = Path("data")
DEFAULT_CACHE_PATH = Path("outputs/cache/qianfan.sqlite")
SCORER_PARSER_VERSION = "json_regex_v2"
CLIP_SCORER_PARSER_VERSION = "json_regex_v3_clips"

QIANFAN_API_URL = DEFAULT_BASE_URL
QIANFAN_API_MODEL = DEFAULT_MODEL


def _config_value(name: str, default: Any = None) -> Any:
    """Return a built-in default retained by the original scorer call sites."""
    return default


def _qianfan_model() -> str:
    return os.environ.get("QIANFAN_API_MODEL", QIANFAN_API_MODEL)


def _qianfan_headers() -> dict[str, str]:
    return QianfanConfig.from_env().headers()


def _cache_path() -> Path:
    return Path(os.environ.get("QIANFAN_CACHE_PATH", str(DEFAULT_CACHE_PATH)))


def _dataset_root() -> Path:
    return Path(os.environ.get("CLIPPLAN_DATASET_ROOT", str(DEFAULT_DATASET_ROOT)))


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"Unsupported ground_truth type: {type(value).__name__}")


def _canonical_timestamp(value: float, valid_timestamps: list[float], tolerance: float = 0.51) -> float | None:
    if not valid_timestamps:
        return None
    best = min(valid_timestamps, key=lambda ts: abs(ts - value))
    return best if abs(best - value) <= tolerance else None


def _calc_iou(pred_start: float, pred_end: float, gt_start: float, gt_end: float) -> float:
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    if inter_start >= inter_end:
        return 0.0
    inter = inter_end - inter_start
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    return inter / union if union > 0 else 0.0


def _parse_json_response(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{[\s\S]*?\}", text or "")
    if not match:
        return None
    raw = re.sub(r"//[^\n]*", "", match.group())
    raw = re.sub(r",\s*([}\]])", r"\1", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value)
        if match:
            return float(match.group())
    return None


def _normalize_segment_fields(parsed: dict[str, Any]) -> dict[str, float] | None:
    start_keys = ("start", "start_time", "start_sec", "start_seconds")
    end_keys = ("end", "end_time", "end_sec", "end_seconds")

    start = None
    for key in start_keys:
        if key in parsed:
            start = _coerce_number(parsed.get(key))
            if start is not None:
                break
    end = None
    for key in end_keys:
        if key in parsed:
            end = _coerce_number(parsed.get(key))
            if end is not None:
                break
    if start is None or end is None:
        return None
    return {"start": start, "end": end}


def _find_number_after_key(text: str, keys: tuple[str, ...]) -> float | None:
    key_pattern = "|".join(re.escape(key) for key in keys)
    number_pattern = r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    pattern = rf"(?i)(?<![A-Za-z0-9_])[\"']?(?:{key_pattern})[\"']?\s*[:=]\s*[\"']?{number_pattern}"
    match = re.search(pattern, text or "")
    return float(match.group(1)) if match else None


def _parse_regex_response(text: str) -> dict[str, float] | None:
    start = _find_number_after_key(text, ("start", "start_time", "start_sec", "start_seconds"))
    end = _find_number_after_key(text, ("end", "end_time", "end_sec", "end_seconds"))
    if start is None or end is None:
        return None
    return {"start": start, "end": end}


def _parse_scorer_response(text: str) -> tuple[dict[str, float] | None, str]:
    parsed = _parse_json_response(text)
    if parsed is not None:
        normalized = _normalize_segment_fields(parsed)
        if normalized is not None:
            return normalized, "json"

    parsed = _parse_regex_response(text)
    if parsed is not None:
        return parsed, "regex"
    return None, "failed"


def _json_payloads(text: str) -> list[Any]:
    decoder = json.JSONDecoder()
    payloads: list[Any] = []
    raw_text = text or ""
    for idx, char in enumerate(raw_text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(raw_text[idx:])
        except json.JSONDecodeError:
            continue
        payloads.append(value)
    return payloads


def _normalize_clip_fields(parsed: dict[str, Any]) -> dict[str, float] | None:
    segment = _normalize_segment_fields(parsed)
    if segment is None:
        return None
    score = None
    for key in ("score", "relevance_score", "relevance", "confidence", "probability"):
        if key in parsed:
            score = _coerce_number(parsed.get(key))
            if score is not None:
                break
    if score is None:
        score = 1.0
    return {"start": segment["start"], "end": segment["end"], "score": float(score)}


def _clip_dicts_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("clips", "predictions", "results", "segments", "moments"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return [payload]


def _parse_clip_list_response(text: str) -> tuple[list[dict[str, float]], str]:
    for payload in _json_payloads(text):
        clips: list[dict[str, float]] = []
        for item in _clip_dicts_from_payload(payload):
            normalized = _normalize_clip_fields(item)
            if normalized is not None:
                clips.append(normalized)
        if clips or payload == []:
            return clips, "json"

    clips = []
    for match in re.finditer(r"\{[^{}]*\}", text or ""):
        parsed = _parse_regex_response(match.group())
        if parsed is None:
            continue
        score = _find_number_after_key(match.group(), ("score", "relevance_score", "relevance", "confidence"))
        clips.append({"start": parsed["start"], "end": parsed["end"], "score": 1.0 if score is None else score})
    if clips:
        return clips, "regex_objects"
    return [], "failed"


def _load_captions(video_name: str) -> list[dict[str, Any]]:
    caption_path = _dataset_root() / "caption" / f"{video_name}.jsonl"
    captions: list[dict[str, Any]] = []
    with caption_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(json.loads(line))
    return captions


def _frame_dir(video_name: str, extra_info: Any) -> Path:
    if isinstance(extra_info, dict) and extra_info.get("frame_dir"):
        path = Path(str(extra_info["frame_dir"]))
        if path.exists():
            return path
    return _dataset_root() / "frames" / video_name


def _encode_frame(path: Path) -> str:
    max_width = int(os.environ.get("CLIPPLAN_JUDGE_IMAGE_MAX_WIDTH", _config_value("IMAGE_MAX_WIDTH", 320)))
    max_height = int(os.environ.get("CLIPPLAN_JUDGE_IMAGE_MAX_HEIGHT", _config_value("IMAGE_MAX_HEIGHT", 180)))
    quality = int(os.environ.get("CLIPPLAN_JUDGE_IMAGE_QUALITY", _config_value("IMAGE_QUALITY", 85)))
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_width, max_height), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _build_scorer_messages(
    query: str,
    video_name: str,
    duration: float,
    captions: list[dict[str, Any]],
    frame_dir: Path,
    upgraded_timestamps: list[float],
) -> list[dict[str, Any]]:
    upgraded = {float(ts) for ts in upgraded_timestamps}

    header = (
        "You are a video moment localization expert.\n\n"
        "## Query\n"
        f'"{query}"\n\n'
        f"## Video: {video_name} ({len(captions)} frames, {duration:.0f}s)\n"
        "## Frame Representations:\n"
    )

    content_parts: list[dict[str, Any]] = []
    text_buf = header
    for cap in captions:
        ts = float(cap["timestamp"])
        if ts in upgraded:
            text_buf += f"[{ts:.1f}s] VISUAL\n"
            content_parts.append({"type": "text", "text": text_buf})
            text_buf = ""
            image_path = frame_dir / str(cap["frame_file"])
            if image_path.exists():
                b64 = _encode_frame(image_path)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                    }
                )
        else:
            caption = str(cap.get("caption") or "[FAILED]").replace("\n", " ").strip()
            text_buf += f'[{ts:.1f}s] TEXT: "{caption}"\n'

    text_buf += (
        "\nLocate the time segment in this video that best matches the query.\n"
        f"The start and end must be within [0, {duration:.1f}].\n\n"
        "Return exactly one JSON object. No markdown, comments, or explanation.\n"
        'The only allowed format is: {"start": <seconds>, "end": <seconds>}'
    )
    content_parts.append({"type": "text", "text": text_buf})

    return [{"role": "user", "content": content_parts}]


def _build_routed_scorer_messages(
    query: str,
    video_name: str,
    duration: float,
    captions: list[dict[str, Any]],
    frame_dir: Path,
    text_timestamps: list[float],
    visual_timestamps: list[float],
) -> list[dict[str, Any]]:
    text_set = {float(ts) for ts in text_timestamps}
    visual_set = {float(ts) for ts in visual_timestamps}

    header = (
        "You are a video moment localization expert.\n\n"
        "## Query\n"
        f'"{query}"\n\n'
        f"## Video: {video_name} ({len(captions)} sampled frames, {duration:.0f}s)\n"
        "## Routed Frame Representations:\n"
    )

    content_parts: list[dict[str, Any]] = []
    text_buf = header
    selected_count = 0
    for cap in captions:
        ts = float(cap["timestamp"])
        if ts in visual_set:
            selected_count += 1
            text_buf += f"[{ts:.1f}s] VISUAL\n"
            content_parts.append({"type": "text", "text": text_buf})
            text_buf = ""
            image_path = frame_dir / str(cap["frame_file"])
            if image_path.exists():
                b64 = _encode_frame(image_path)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                    }
                )
        elif ts in text_set:
            selected_count += 1
            caption = str(cap.get("caption") or "[FAILED]").replace("\n", " ").strip()
            text_buf += f'[{ts:.1f}s] TEXT: "{caption}"\n'

    if selected_count == 0:
        text_buf += "No frames were selected by the router.\n"

    text_buf += (
        "\nLocate the time segment in this video that best matches the query using only the routed representations above.\n"
        f"The start and end must be within [0, {duration:.1f}].\n\n"
        "Return exactly one JSON object. No markdown, comments, or explanation.\n"
        'The only allowed format is: {"start": <seconds>, "end": <seconds>}'
    )
    content_parts.append({"type": "text", "text": text_buf})

    return [{"role": "user", "content": content_parts}]


def _build_routed_candidate_messages(
    query: str,
    video_name: str,
    duration: float,
    captions: list[dict[str, Any]],
    frame_dir: Path,
    text_timestamps: list[float],
    visual_timestamps: list[float],
    max_clips: int,
) -> list[dict[str, Any]]:
    text_set = {float(ts) for ts in text_timestamps}
    visual_set = {float(ts) for ts in visual_timestamps}

    header = (
        "You are a video moment retrieval expert.\n\n"
        "## Query\n"
        f'"{query}"\n\n'
        f"## Video: {video_name} ({len(captions)} sampled frames, {duration:.0f}s)\n"
        "## Routed Frame Representations:\n"
    )

    content_parts: list[dict[str, Any]] = []
    text_buf = header
    selected_count = 0
    for cap in captions:
        ts = float(cap["timestamp"])
        if ts in visual_set:
            selected_count += 1
            text_buf += f"[{ts:.1f}s] VISUAL\n"
            content_parts.append({"type": "text", "text": text_buf})
            text_buf = ""
            image_path = frame_dir / str(cap["frame_file"])
            if image_path.exists():
                b64 = _encode_frame(image_path)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                    }
                )
        elif ts in text_set:
            selected_count += 1
            caption = str(cap.get("caption") or "[FAILED]").replace("\n", " ").strip()
            text_buf += f'[{ts:.1f}s] TEXT: "{caption}"\n'

    if selected_count == 0:
        text_buf += "No frames were selected by the router.\n"

    text_buf += (
        "\nUsing only the routed representations above, return the clips in this video that match the query.\n"
        f"The start and end of every clip must be within [0, {duration:.1f}].\n"
        f"Return at most {int(max_clips)} distinct clips.\n"
        "Each clip must include a relevance score in [0, 1], where higher means more likely/relevant.\n"
        "If no routed evidence supports a matching clip, return an empty list.\n\n"
        "Return exactly one JSON array. No markdown, comments, or explanation.\n"
        'The only allowed format is: [{"start": <seconds>, "end": <seconds>, "score": <0_to_1>}]'
    )
    content_parts.append({"type": "text", "text": text_buf})

    return [{"role": "user", "content": content_parts}]


def _db_connect() -> sqlite3.Connection:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS scorer_cache ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)"
    )
    conn.execute("CREATE TABLE IF NOT EXISTS rate_events (ts REAL NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS rate_events_v2 (provider TEXT NOT NULL, ts REAL NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS token_rate_events (provider TEXT NOT NULL, ts REAL NOT NULL, tokens INTEGER NOT NULL)")
    return conn


def _cache_key(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get_cache(key: str) -> dict[str, Any] | None:
    try:
        with _db_connect() as conn:
            row = conn.execute("SELECT value FROM scorer_cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return None


def _set_cache(key: str, value: dict[str, Any]) -> None:
    try:
        with _db_connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scorer_cache(key, value, created_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=False), time.time()),
            )
    except Exception:
        return


def _provider_rpm(_provider: str) -> int:
    return int(os.environ.get("CLIPPLAN_QIANFAN_RPM_LIMIT", os.environ.get("QIANFAN_RPM_LIMIT", "900")))


def _provider_tpm(_provider: str) -> int:
    return int(os.environ.get("CLIPPLAN_QIANFAN_TPM_LIMIT", os.environ.get("QIANFAN_TPM_LIMIT", "0")))


def _estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    text_chars = 0
    image_chars = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            text_chars += len(content)
            continue
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    text_chars += len(str(item.get("text") or ""))
                elif item.get("type") == "image_url":
                    image_url = item.get("image_url") or {}
                    if isinstance(image_url, dict):
                        image_chars += len(str(image_url.get("url") or ""))
                    else:
                        image_chars += len(str(image_url))
    chars_per_token = float(os.environ.get("CLIPPLAN_TEXT_CHARS_PER_TOKEN", "4.0"))
    image_chars_per_token = float(os.environ.get("CLIPPLAN_IMAGE_CHARS_PER_TOKEN", os.environ.get("CLIPPLAN_TEXT_CHARS_PER_TOKEN", "4.0")))
    text_tokens = int((float(text_chars) / max(1.0, chars_per_token)) + 0.999)
    image_tokens = int((float(image_chars) / max(1.0, image_chars_per_token)) + 0.999)
    return max(1, text_tokens + image_tokens)


def _take_rate_limit_slot(provider: str, rpm: int, wait: bool) -> bool:
    if rpm <= 0:
        return True

    while True:
        now = time.time()
        with _db_connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM rate_events_v2 WHERE provider = ? AND ts < ?", (provider, now - 60.0))
            count = conn.execute("SELECT COUNT(*) FROM rate_events_v2 WHERE provider = ?", (provider,)).fetchone()[0]
            if count < rpm:
                conn.execute("INSERT INTO rate_events_v2(provider, ts) VALUES (?, ?)", (provider, now))
                conn.commit()
                return True
            oldest = conn.execute("SELECT MIN(ts) FROM rate_events_v2 WHERE provider = ?", (provider,)).fetchone()[0]
            conn.commit()

        if not wait:
            return False

        sleep_s = max(0.2, float(oldest) + 60.0 - now) if oldest else 1.0
        time.sleep(min(sleep_s, 5.0))


def _wait_for_rate_limit(provider: str) -> None:
    _take_rate_limit_slot(provider, _provider_rpm(provider), wait=True)


def _try_take_rate_limit_slot(provider: str) -> bool:
    return _take_rate_limit_slot(provider, _provider_rpm(provider), wait=False)


def _wait_for_token_rate_limit(provider: str, tokens: int) -> None:
    tpm = _provider_tpm(provider)
    if tpm <= 0:
        return

    tokens = max(1, int(tokens))
    if tokens > tpm:
        return

    while True:
        now = time.time()
        with _db_connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM token_rate_events WHERE provider = ? AND ts < ?", (provider, now - 60.0))
            total = conn.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_rate_events WHERE provider = ?",
                (provider,),
            ).fetchone()[0]
            if int(total) + tokens <= tpm:
                conn.execute(
                    "INSERT INTO token_rate_events(provider, ts, tokens) VALUES (?, ?, ?)",
                    (provider, now, tokens),
                )
                conn.commit()
                return
            oldest = conn.execute(
                "SELECT MIN(ts) FROM token_rate_events WHERE provider = ?",
                (provider,),
            ).fetchone()[0]
            conn.commit()

        sleep_s = max(0.2, float(oldest) + 60.0 - now) if oldest else 1.0
        time.sleep(min(sleep_s, 5.0))


def _request_timeout(read_timeout: float) -> tuple[float, float]:
    connect_timeout = float(os.environ.get("CLIPPLAN_QIANFAN_CONNECT_TIMEOUT", "10"))
    return (connect_timeout, float(read_timeout))


def _messages_to_text_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            if content.strip():
                parts.append(content.strip())
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        parts.append(text)
    return "\n\n".join(parts)


def get_res_qianfan(
    prompt: str | list[dict[str, Any]],
    max_retries: int = 3,
    timeout: int = 300,
    model: str = QIANFAN_API_MODEL,
) -> Optional[str]:
    """Call Qianfan API with retry. Accepts text or OpenAI-style multimodal messages."""
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    temperature = float(os.environ.get("CLIPPLAN_SCORER_TEMPERATURE", _config_value("SCORER_TEMPERATURE", 0.0)))
    top_p = float(os.environ.get("CLIPPLAN_SCORER_TOP_P", _config_value("SCORER_TOP_P", 1.0)))
    last_error = "unknown"
    for attempt in range(max_retries):
        try:
            payload = json.dumps(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "disable_search": False,
                    "enable_citation": False,
                    "safety": {"input_level": "none"},
                },
                ensure_ascii=False,
            )
            _wait_for_rate_limit("qianfan")
            _wait_for_token_rate_limit("qianfan", _estimate_message_tokens(messages))
            response = requests.request(
                "POST",
                os.environ.get("QIANFAN_API_BASE_URL", QIANFAN_API_URL),
                headers=_qianfan_headers(),
                data=payload.encode("utf-8"),
                timeout=_request_timeout(timeout),
            )
            if response.status_code == 200:
                return str(response.json()["choices"][0]["message"]["content"])
            last_error = f"http_{response.status_code}:{response.text[:240]}"
            if response.status_code == 429:
                sleep_s = int(os.environ.get("CLIPPLAN_QIANFAN_RATE_LIMIT_SLEEP", "65"))
            else:
                sleep_s = 2 * (attempt + 1)
            if attempt < max_retries - 1:
                print(f"qianfan retry {attempt + 1}/{max_retries}: {last_error}", flush=True)
                time.sleep(sleep_s)
        except requests.exceptions.Timeout as exc:
            last_error = repr(exc)
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))
        except Exception as exc:
            last_error = repr(exc)
            if attempt < max_retries - 1:
                time.sleep(2)
    os.environ["CLIPPLAN_QIANFAN_LAST_ERROR"] = last_error
    return None


def _call_qianfan_scorer_api(messages: list[dict[str, Any]]) -> str:
    if not messages:
        raise RuntimeError("missing qianfan messages")

    max_retries = int(os.environ.get("CLIPPLAN_QIANFAN_MAX_RETRIES", "3"))
    timeout = int(os.environ.get("CLIPPLAN_QIANFAN_TIMEOUT", "300"))
    content = get_res_qianfan(messages, max_retries=max_retries, timeout=timeout, model=_qianfan_model())
    if content:
        return content
    raise RuntimeError(f"qianfan_failed:{os.environ.get('CLIPPLAN_QIANFAN_LAST_ERROR', 'unknown')}")


def _call_scorer_api(messages: list[dict[str, Any]]) -> tuple[str, str]:
    return _call_qianfan_scorer_api(messages), "qianfan"


def _score_state(
    query: str,
    video_name: str,
    duration: float,
    gt_start: float,
    gt_end: float,
    captions: list[dict[str, Any]],
    frame_dir: Path,
    upgraded_timestamps: list[float],
) -> dict[str, Any]:
    normalized_upgraded = sorted(float(ts) for ts in upgraded_timestamps)
    cache_payload = {
        "query": query,
        "video_name": video_name,
        "duration": duration,
        "gt": [gt_start, gt_end],
        "upgraded": normalized_upgraded,
        "provider": "qianfan",
        "model": _qianfan_model(),
        "parser_version": SCORER_PARSER_VERSION,
    }
    key = _cache_key(cache_payload)
    cached = _get_cache(key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    messages = _build_scorer_messages(query, video_name, duration, captions, frame_dir, normalized_upgraded)
    raw, provider = _call_scorer_api(messages)
    parsed, parse_method = _parse_scorer_response(raw)
    if parsed is None:
        value = {
            "pred": None,
            "iou": 0.0,
            "raw": raw[:500],
            "provider": provider,
            "parse_error": True,
            "parse_method": parse_method,
            "cache_hit": False,
        }
        _set_cache(key, value)
        return value

    start = max(0.0, min(float(parsed["start"]), duration))
    end = max(start, min(float(parsed["end"]), duration))
    pred = [start, end] if end > start else None
    iou = _calc_iou(start, end, gt_start, gt_end) if pred else 0.0
    value = {
        "pred": pred,
        "iou": float(iou),
        "raw": raw[:500],
        "provider": provider,
        "parse_error": pred is None,
        "parse_method": parse_method,
        "cache_hit": False,
    }
    _set_cache(key, value)
    return value


def _score_routed_state(
    query: str,
    video_name: str,
    duration: float,
    gt_start: float,
    gt_end: float,
    captions: list[dict[str, Any]],
    frame_dir: Path,
    text_timestamps: list[float],
    visual_timestamps: list[float],
) -> dict[str, Any]:
    normalized_visual = sorted(float(ts) for ts in visual_timestamps)
    visual_set = set(normalized_visual)
    normalized_text = sorted(float(ts) for ts in text_timestamps if float(ts) not in visual_set)
    if not normalized_text and not normalized_visual:
        return {
            "pred": None,
            "iou": 0.0,
            "raw": "",
            "provider": "none",
            "parse_error": False,
            "parse_method": "empty_route",
            "cache_hit": False,
        }

    cache_payload = {
        "query": query,
        "video_name": video_name,
        "duration": duration,
        "gt": [gt_start, gt_end],
        "text": normalized_text,
        "visual": normalized_visual,
        "provider": "qianfan",
        "model": _qianfan_model(),
        "parser_version": SCORER_PARSER_VERSION,
        "route_schema": "drop_text_visual_stop_v1",
    }
    key = _cache_key(cache_payload)
    cached = _get_cache(key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    messages = _build_routed_scorer_messages(
        query,
        video_name,
        duration,
        captions,
        frame_dir,
        normalized_text,
        normalized_visual,
    )
    raw, provider = _call_scorer_api(messages)
    parsed, parse_method = _parse_scorer_response(raw)
    if parsed is None:
        value = {
            "pred": None,
            "iou": 0.0,
            "raw": raw[:500],
            "provider": provider,
            "parse_error": True,
            "parse_method": parse_method,
            "cache_hit": False,
        }
        _set_cache(key, value)
        return value

    start = max(0.0, min(float(parsed["start"]), duration))
    end = max(start, min(float(parsed["end"]), duration))
    pred = [start, end] if end > start else None
    iou = _calc_iou(start, end, gt_start, gt_end) if pred else 0.0
    value = {
        "pred": pred,
        "iou": float(iou),
        "raw": raw[:500],
        "provider": provider,
        "parse_error": pred is None,
        "parse_method": parse_method,
        "cache_hit": False,
    }
    _set_cache(key, value)
    return value


def _score_routed_candidate(
    query: str,
    video_name: str,
    duration: float,
    captions: list[dict[str, Any]],
    frame_dir: Path,
    text_timestamps: list[float],
    visual_timestamps: list[float],
    max_clips: int = 5,
) -> dict[str, Any]:
    normalized_visual = sorted(float(ts) for ts in visual_timestamps)
    visual_set = set(normalized_visual)
    normalized_text = sorted(float(ts) for ts in text_timestamps if float(ts) not in visual_set)
    max_clips = max(1, int(max_clips))
    if not normalized_text and not normalized_visual:
        return {
            "clips": [],
            "raw": "",
            "provider": "none",
            "parse_error": False,
            "parse_method": "empty_route",
            "cache_hit": False,
        }

    cache_payload = {
        "query": query,
        "video_name": video_name,
        "duration": duration,
        "text": normalized_text,
        "visual": normalized_visual,
        "max_clips": max_clips,
        "provider": "qianfan",
        "model": _qianfan_model(),
        "parser_version": CLIP_SCORER_PARSER_VERSION,
        "route_schema": "drop_text_visual_stop_ranked_clips_v1",
    }
    key = _cache_key(cache_payload)
    cached = _get_cache(key)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    messages = _build_routed_candidate_messages(
        query,
        video_name,
        duration,
        captions,
        frame_dir,
        normalized_text,
        normalized_visual,
        max_clips,
    )
    raw, provider = _call_scorer_api(messages)
    parsed_clips, parse_method = _parse_clip_list_response(raw)
    clips: list[dict[str, float]] = []
    for clip in parsed_clips:
        start = max(0.0, min(float(clip["start"]), duration))
        end = max(start, min(float(clip["end"]), duration))
        score = max(0.0, min(float(clip.get("score", 1.0)), 1.0))
        if end > start:
            clips.append({"start": start, "end": end, "score": score})
    clips.sort(key=lambda item: item["score"], reverse=True)
    value = {
        "clips": clips[:max_clips],
        "raw": raw,
        "provider": provider,
        "parse_error": parse_method == "failed",
        "parse_method": parse_method,
        "cache_hit": False,
    }
    _set_cache(key, value)
    return value


score_state = _score_state
score_routed_state = _score_routed_state
score_routed_candidate = _score_routed_candidate
build_routed_candidate_messages = _build_routed_candidate_messages
