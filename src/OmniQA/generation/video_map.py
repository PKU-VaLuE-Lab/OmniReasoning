"""Resolve source video identities to accessible media files."""
from __future__ import annotations

import json
from pathlib import Path


def load_video_map(path: str) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Load a video_map.jsonl -> ((source, video_id) -> path, video_id -> path).

    Only rows whose ``status`` is ``"mapped"`` (and that carry a ``video_path``)
    are kept. A missing or empty file yields two empty maps.
    """
    by_pair: dict[tuple[str, str], str] = {}
    by_id: dict[str, str] = {}
    if not path or not Path(path).exists():
        return by_pair, by_id
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "mapped" or not row.get("video_path"):
                continue
            by_pair[(row.get("source", ""), row.get("video_id", ""))] = row["video_path"]
            by_id.setdefault(row.get("video_id", ""), row["video_path"])
    return by_pair, by_id
