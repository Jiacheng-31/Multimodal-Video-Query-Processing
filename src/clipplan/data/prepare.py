from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters import normalize_annotations


def _load(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    for key in ("annotations", "queries", "data"):
        rows = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(rows, list):
            return rows
    raise ValueError("The annotation file must contain a list of query records.")


def prepare(input_path: Path, output_path: Path) -> None:
    rows = _load(input_path)
    normalized = normalize_annotations(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump([row.to_dict() for row in normalized], handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(normalized)} normalized queries to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize ranked video retrieval annotations for ClipPlan.")
    parser.add_argument("--input", type=Path, required=True, help="Source JSON or JSONL annotations.")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(args.input, args.output)


if __name__ == "__main__":
    main()
