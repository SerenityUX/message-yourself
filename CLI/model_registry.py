"""Append-only registry of trained Tinker checkpoints (``saved_models.json`` at package root)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def registry_path(root: Path) -> Path:
    return root / "saved_models.json"


def _load_raw(root: Path) -> dict[str, Any]:
    p = registry_path(root)
    if not p.is_file():
        return {"models": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"models": []}
    if not isinstance(data, dict):
        return {"models": []}
    models = data.get("models")
    if not isinstance(models, list):
        data["models"] = []
    return data


def save_models(root: Path, data: dict[str, Any]) -> None:
    registry_path(root).write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_models(root: Path) -> list[dict[str, Any]]:
    data = _load_raw(root)
    return [m for m in data["models"] if isinstance(m, dict)]


def register_from_metadata_file(
    root: Path,
    metadata_path: Path,
    *,
    learning_rate: float | None = None,
    lora_rank: int | None = None,
    source: str = "",
    label: str | None = None,
) -> dict[str, Any] | None:
    """Read ``sft_tinker_metadata.json`` and append to registry under ``root``."""
    meta_path = metadata_path.expanduser().resolve()
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    uri = meta.get("tinker_checkpoint_path")
    if not isinstance(uri, str) or not uri.startswith("tinker://"):
        return None

    data = _load_raw(root)
    models: list[dict[str, Any]] = data["models"]
    key = str(meta_path)
    for m in models:
        if isinstance(m, dict) and m.get("metadata_path") == key:
            return m

    lr = learning_rate if learning_rate is not None else meta.get("learning_rate")
    rk = lora_rank if lora_rank is not None else meta.get("lora_rank")
    entry: dict[str, Any] = {
        "id": uuid.uuid4().hex[:10],
        "label": label or (meta_path.parent.name or meta_path.stem),
        "metadata_path": key,
        "tinker_uri": uri,
        "learning_rate": lr,
        "lora_rank": rk,
        "checkpoint_name": meta.get("checkpoint_name"),
        "added_utc": datetime.now(timezone.utc).isoformat(),
        "source": source or "unknown",
    }
    models.append(entry)
    save_models(root, data)
    return entry


def ensure_legacy_metadata(root: Path) -> None:
    legacy = root / "sft_tinker_metadata.json"
    if not legacy.is_file() or list_models(root):
        return
    register_from_metadata_file(root, legacy, source="legacy")


def pick_model_interactive(root: Path) -> Path | None:
    ensure_legacy_metadata(root)
    models = list_models(root)
    if not models:
        return None
    for i, m in enumerate(models, 1):
        lbl = m.get("label", "?")
        uri = (m.get("tinker_uri") or "")[:52]
        print(f"  {i}  {lbl}  {uri}…", flush=True)
    raw = input("> ").strip()
    if not raw.isdigit():
        return None
    idx = int(raw) - 1
    if idx < 0 or idx >= len(models):
        return None
    p = models[idx].get("metadata_path")
    return Path(p) if isinstance(p, str) else None
