#!/usr/bin/env python3
"""Build root ``eval_bundle.json`` from ``imessageWithContextSelfExperimentation/experiment_results/``.

Keeps only dialogue turns + rubrics (no Tinker paths / manifests). Run from repo root::

  python3 scripts/build_eval_bundle.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "imessageWithContextSelfExperimentation" / "experiment_results"
OUT = REPO / "eval_bundle.json"

SCENARIOS = (
    "hike_invite",
    "advice_confide",
    "cafe_after_event",
    "podcast_scheduling",
)

RUBRIC_KEYS = (
    "realistic",
    "kind",
    "casual",
    "concise",
    "repetition_issue",
    "natural",
)


def folder_to_lr_rank(name: str) -> tuple[str, int] | None:
    m = re.match(r"^lr(.+)_r(\d+)$", name)
    if not m:
        return None
    lr = m.group(1).replace("p", ".")
    rank = int(m.group(2))
    return lr, rank


def rubric_mean(r: dict) -> float | None:
    if not r:
        return None
    vals: list[float] = []
    for k in RUBRIC_KEYS:
        if k in r:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def normalize_turns(raw_turns: list) -> list[list]:
    out: list[list] = []
    for t in raw_turns:
        if not isinstance(t, dict):
            continue
        role, text = t.get("role"), t.get("text")
        if role in ("friend", "thomas") and isinstance(text, str):
            out.append([role, text])
    return out


def load_combo(dirname: str) -> dict | None:
    d = RESULTS / dirname
    if not d.is_dir():
        return None
    scenarios: dict[str, dict] = {}
    for sid in SCENARIOS:
        p = d / f"{sid}.json"
        if not p.is_file():
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dlg = j.get("dialogue") or {}
        raw_turns = dlg.get("turns") or []
        turns = normalize_turns(raw_turns)
        rubric = j.get("rubric")
        title = dlg.get("scenario_title") or sid
        scenarios[sid] = {
            "title": title,
            "turns": turns,
            "rubric": rubric,
        }
    if len(scenarios) < len(SCENARIOS):
        return None
    return scenarios


def combo_mean_score(scenarios: dict[str, dict]) -> float | None:
    ms: list[float] = []
    for sid in SCENARIOS:
        sc = scenarios.get(sid)
        if not sc:
            return None
        m = rubric_mean(sc.get("rubric") or {})
        if m is None:
            return None
        ms.append(m)
    return sum(ms) / len(ms) if ms else None


def main() -> None:
    if not RESULTS.is_dir():
        raise SystemExit(f"Missing results dir: {RESULTS}")

    runs: dict[str, dict[str, dict]] = {}
    sweep_rows: list[tuple[str, int, float]] = []

    for child in sorted(RESULTS.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        name = child.name
        parsed = folder_to_lr_rank(name)
        if not parsed:
            continue
        lr, rank = parsed
        scenarios = load_combo(name)
        if not scenarios:
            continue
        folder_key = f"imessageWithContextSelfExperimentation/experiment_results/{name}"
        runs[folder_key] = scenarios
        mean = combo_mean_score(scenarios)
        if mean is not None:
            sweep_rows.append((lr, rank, mean))

    sweep_rows.sort(key=lambda x: -x[2])
    sweep = [[a, b, c] for a, b, c in sweep_rows]

    bundle = {
        "version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": "imessageWithContextSelfExperimentation/experiment_results",
        "sweep": sweep,
        "runs": runs,
    }
    OUT.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} ({len(runs)} runs, {len(sweep)} sweep rows)", flush=True)


if __name__ == "__main__":
    main()
