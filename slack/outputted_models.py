"""Persist trained sampler checkpoints for ``chat.py`` and tooling."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUTTED_MODELS_JSON = _SCRIPT_DIR / "outputted_models.json"
DEFAULT_CPT_METADATA = _SCRIPT_DIR / "cpt_tinker_metadata.json"


def _read_store() -> dict[str, Any]:
    if not OUTPUTTED_MODELS_JSON.is_file():
        return {"models": []}
    try:
        data = json.loads(OUTPUTTED_MODELS_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"models": []}
    if not isinstance(data, dict):
        return {"models": []}
    data.setdefault("models", [])
    if not isinstance(data["models"], list):
        data["models"] = []
    return data


def _write_store(store: dict[str, Any]) -> None:
    OUTPUTTED_MODELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUTTED_MODELS_JSON.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def upsert_training_run(entry: dict[str, Any]) -> None:
    """Insert or replace a run keyed by ``tinker_checkpoint_path`` (stable chat/menu identity)."""

    uri = entry.get("tinker_checkpoint_path")
    if not uri or not str(uri).startswith("tinker://"):
        return

    store = _read_store()
    models: list[dict[str, Any]] = store["models"]
    now = datetime.now(timezone.utc).isoformat()
    row = {**entry, "last_updated_utc": now}

    for i, existing in enumerate(models):
        if existing.get("tinker_checkpoint_path") == uri:
            first = existing.get("recorded_at_utc")
            row["recorded_at_utc"] = first or now
            models[i] = row
            _write_store(store)
            return

    row["recorded_at_utc"] = now
    models.append(row)
    _write_store(store)


def append_training_run(entry: dict[str, Any]) -> None:
    """Backward-compatible name: same as :func:`upsert_training_run`."""

    upsert_training_run(entry)


def sync_from_cpt_metadata(
    metadata_path: Path | None = None,
    *,
    api_dump_path: str | None = None,
) -> bool:
    """Load ``cpt_tinker_metadata.json`` and upsert into ``outputted_models.json``. Returns True if written."""

    path = (metadata_path or DEFAULT_CPT_METADATA).expanduser().resolve()
    if not path.is_file():
        return False
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(meta, dict):
        return False

    ckpt = meta.get("tinker_checkpoint_path")
    if not ckpt or not str(ckpt).startswith("tinker://"):
        return False

    base = meta.get("base_model") or ""
    short = base.split("/")[-1] if base else "model"
    name = meta.get("checkpoint_name") or "slack-cpt-lora"
    dump = api_dump_path or str((_SCRIPT_DIR / "tinker_api_full_response.json").resolve())

    upsert_training_run(
        {
            "name": name,
            "display_name": f"{name} ({short})",
            "base_model": base,
            "tinker_checkpoint_path": str(ckpt),
            "openai_compatible_base_url": "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1",
            "metadata_path": str(path),
            "api_dump_path": dump,
            "corpus": meta.get("corpus", ""),
            "steps": meta.get("steps"),
            "training_datums": meta.get("training_datums"),
        }
    )
    return True


def list_models() -> list[dict[str, Any]]:
    return list(_read_store().get("models") or [])


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Manage slack/outputted_models.json")
    p.add_argument(
        "command",
        nargs="?",
        default="sync",
        choices=("sync",),
        help="sync — import from cpt_tinker_metadata.json (default)",
    )
    p.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Path to cpt_tinker_metadata.json (default: next to this file)",
    )
    args = p.parse_args()
    if args.command == "sync":
        ok = sync_from_cpt_metadata(args.metadata)
        if ok:
            print(f"Updated {OUTPUTTED_MODELS_JSON}", flush=True)
        else:
            raise SystemExit(
                "Nothing to sync: missing or invalid cpt_tinker_metadata.json "
                "(need tinker_checkpoint_path)."
            )


if __name__ == "__main__":
    main()
