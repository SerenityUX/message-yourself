#!/usr/bin/env python3
"""
Claude-controlled LR×LoRA sweep: proposes (lr, rank) bands, train+eval once per combo, logs reasoning.

Run via ``main.py`` (Train → SFT → Agent) or:

  python3 -m experiment.agent_loop --bands 0 --max-per-band 9 --steps 200

``--bands 0`` (default) = **run forever** until Ctrl+C. Positive ``--bands`` caps controller rounds.
Within each band, the controller picks **1..N** new (lr, rank) pairs (N ≤ ``--max-per-band``, default 9); each
that needs training runs **in parallel** (thread pool).

State: ``experiment_results/agent_state.json`` (completed pairs).
Log: ``experiment_results/claude_agent_log.md``.
Chat history for the controller: ``experiment_results/agent_controller_history.json`` (OpenRouter-style
messages, persisted across sessions; truncated to the last few dozen turns).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from experiment.config import (
    EXPERIMENT_RESULTS,
    PACKAGE_ROOT,
    default_openrouter_agent_model,
    tinker_max_lora_rank,
)
from experiment.openrouter_client import ensure_openrouter_key, get_openrouter_client
from experiment.run_experiment import _load_tinker_uri, combo_dir, run_eval_for_uri, train_one_combo
from experiment.scenarios import SCENARIOS

AGENT_STATE_NAME = "agent_state.json"
AGENT_LOG_NAME = "claude_agent_log.md"
AGENT_HISTORY_NAME = "agent_controller_history.json"
# OpenRouter message cap (user+assistant turns); older context dropped from file, not from model tail in one message.
HISTORY_MAX_MESSAGES = 48
# Hard cap on how many (lr, rank) jobs run in parallel per band; the controller chooses 1..N.
DEFAULT_MAX_COMBOS_PER_BAND = 9
SCENARIO_IDS = tuple(s.id for s in SCENARIOS)


def _combo_fully_evaluated(lr: float, rank: int) -> bool:
    d = combo_dir(lr, rank)
    if not d.is_dir() or not (d / "manifest.json").is_file():
        return False
    return all((d / f"{sid}.json").is_file() for sid in SCENARIO_IDS)


def _rank_is_power_of_two(rank: int) -> bool:
    r = int(rank)
    return r >= 1 and (r & (r - 1)) == 0


def _rank_ok_for_tinker(rank: int) -> bool:
    """Power of 2 and within Tinker model cap (e.g. gpt-oss-120b → max 32)."""
    if not _rank_is_power_of_two(rank):
        return False
    return int(rank) <= tinker_max_lora_rank()


def _agent_model() -> str:
    return default_openrouter_agent_model()


def _state_path() -> Path:
    return EXPERIMENT_RESULTS / AGENT_STATE_NAME


def _log_path() -> Path:
    return EXPERIMENT_RESULTS / AGENT_LOG_NAME


def _history_path() -> Path:
    return EXPERIMENT_RESULTS / AGENT_HISTORY_NAME


def load_controller_history() -> list[dict[str, str]]:
    """Prior controller conversation (user/assistant), persisted across runs."""
    p = _history_path()
    if not p.is_file():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        raw = data.get("messages")
        if not isinstance(raw, list):
            return []
        out: list[dict[str, str]] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            role, content = m.get("role"), m.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                out.append({"role": str(role), "content": content})
        return out
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return []


def save_controller_history(messages: list[dict[str, str]]) -> None:
    EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    trimmed = messages[-HISTORY_MAX_MESSAGES:]
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "messages": trimmed,
    }
    _history_path().write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _norm_lr(lr: float) -> float:
    return round(float(lr), 10)


def _pair_key(lr: float, rank: int) -> tuple[float, int]:
    return (_norm_lr(lr), int(rank))


def load_completed_pairs() -> set[tuple[float, int]]:
    """Pairs that already have a full train+eval under ``experiment_results/``."""
    done: set[tuple[float, int]] = set()
    p = _state_path()
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for row in data.get("completed", []):
                if isinstance(row, (list, tuple)) and len(row) == 2:
                    done.add(_pair_key(float(row[0]), int(row[1])))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    # Reconcile with disk (folders with all scenario JSONs)
    for d in EXPERIMENT_RESULTS.iterdir():
        if not d.is_dir() or d.name.startswith("."):
            continue
        m = re.match(r"^lr(.+)_r(\d+)$", d.name)
        if not m:
            continue
        lr_s, r_s = m.group(1).replace("p", "."), m.group(2)
        try:
            lr_v, r_v = float(lr_s), int(r_s)
        except ValueError:
            continue
        if all((d / f"{sid}.json").is_file() for sid in SCENARIO_IDS) and (d / "manifest.json").is_file():
            done.add(_pair_key(lr_v, r_v))
    return done


def _save_state(completed: set[tuple[float, int]]) -> None:
    EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "completed": [[str(lr), rank] for lr, rank in sorted(completed)],
    }
    _state_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_agent_log(section: str) -> None:
    EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    path = _log_path()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    block = f"\n\n---\n\n## {stamp}\n\n{section.strip()}\n"
    if path.is_file():
        path.write_text(path.read_text(encoding="utf-8") + block, encoding="utf-8")
    else:
        path.write_text(
            "# Claude controller log (LR×LoRA agent sweep)\n\n"
            "Decisions and reasoning appended per band.\n" + block,
            encoding="utf-8",
        )


def _dedupe_training_seed(lr: float, rank: int) -> int:
    """Stable per (lr, rank); differs across combos so curves are not lockstep."""
    import hashlib

    h = hashlib.sha256(f"{lr:.12g}:{rank}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31 - 2) + 1


def _rubric_mean(r: dict[str, Any]) -> float | None:
    keys = ("realistic", "kind", "casual", "concise", "repetition_issue", "natural")
    if not isinstance(r, dict) or r.get("error"):
        return None
    vals = []
    for k in keys:
        v = r.get(k)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def summarize_experiment_results() -> str:
    """Markdown summary for the controller."""
    lines: list[str] = ["### Completed runs (from experiment_results/)", ""]
    rows: list[tuple[float, int, float | None, str]] = []
    for d in sorted(EXPERIMENT_RESULTS.iterdir()):
        if not d.is_dir():
            continue
        m = re.match(r"^lr(.+)_r(\d+)$", d.name)
        if not m:
            continue
        try:
            lr_v, r_v = float(m.group(1).replace("p", ".")), int(m.group(2))
        except ValueError:
            continue
        means: list[float] = []
        notes: list[str] = []
        ok = True
        for sid in SCENARIO_IDS:
            jf = d / f"{sid}.json"
            if not jf.is_file():
                ok = False
                break
            data = json.loads(jf.read_text(encoding="utf-8"))
            rub = data.get("rubric") or {}
            mscore = _rubric_mean(rub)
            if mscore is not None:
                means.append(mscore)
            n = rub.get("notes")
            if isinstance(n, str) and n.strip():
                notes.append(f"{sid}: {n.strip()}")
        if not ok:
            continue
        overall = sum(means) / len(means) if means else None
        note_s = " | ".join(notes[:2])
        if len(notes) > 2:
            note_s += " …"
        rows.append((lr_v, r_v, overall, note_s))
    rows.sort(key=lambda x: (x[2] is not None, x[2] or -1), reverse=True)
    if not rows:
        lines.append("_No scored runs yet._")
    else:
        lines.append("| lr | rank | mean_rubric_6dims | sample notes |")
        lines.append("|---:|---:|---:|---|")
        for lr_v, r_v, overall, note_s in rows:
            om = f"{overall:.3f}" if overall is not None else "n/a"
            lines.append(f"| {lr_v:g} | {r_v} | {om} | {note_s[:120]} |")
    lines.append("")
    return "\n".join(lines)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _band_label(band_index: int, max_bands: int | None) -> str:
    if max_bands is None:
        return f"{band_index + 1} (continuous — no fixed end; Ctrl+C to stop)"
    return f"{band_index + 1} / {max_bands}"


def _phase_hint(band_index: int, max_bands: int | None) -> str:
    if max_bands is None:
        cycle = 10
        if (band_index % cycle) < (cycle // 2):
            return (
                "**Explore (this half-cycle):** wide spread of LR and rank (powers of 2); avoid only mid values. "
                "The run continues indefinitely — keep finding *new* (lr, rank) not in the completed list."
            )
        return (
            "**Narrow (this half-cycle):** refine around the best rubric rows; small LR steps / adjacent ranks. "
            "Still propose only pairs not already completed."
        )
    if band_index < max(1, max_bands // 2):
        return (
            "**Explore (early bands):** try a wide spread — e.g. mix low/high LR and low/high rank (powers of 2 only). "
            "Avoid clustering everything in the middle."
        )
    return (
        "**Narrow (later bands):** focus around the best-scoring (lr, rank) from the table; small moves, "
        "refine. Still avoid duplicates."
    )


def _build_outcomes_block(
    *,
    band_display: str,
    ran: list[dict[str, Any]],
    skips: list[str],
    errors: list[str],
) -> str:
    lines = [
        f"## Execution results (after your plan for {band_display})",
        "",
    ]
    if skips:
        lines.append("**Skipped (already on disk):**")
        for s in skips:
            lines.append(f"- {s}")
        lines.append("")
    if errors:
        lines.append("**Training or eval failed (pair was NOT marked complete — you may retry or adjust):**")
        for e in errors:
            lines.append(f"- {e}")
        lines.append("")
    if ran:
        lines.append("**Trained + evaluated:**")
        for info in ran:
            lines.append(
                f"- lr={info['lr']:g} rank={info['rank']}: mean_rubric={info.get('mean_rubric')} "
                f"dir={info.get('out_dir')}"
            )
    elif not skips and not errors:
        lines.append("_No activity this band._")
    elif not ran:
        lines.append("_No successful training runs this band (see skips/failures above)._")
    lines.append("")
    lines.append(
        "Use this when choosing the next (lr, rank) pairs. After failures, pick safer alternatives "
        "(e.g. different rank, lower LR, or backoff) rather than repeating the exact same combo."
    )
    return "\n".join(lines)


def call_controller(
    *,
    band_index: int,
    max_bands: int | None,  # None = infinite run (UI label only)
    max_combos_per_band: int,
    completed: set[tuple[float, int]],
    summary_md: str,
    history: list[dict[str, str]],
    prior_outcomes: str | None,
    retry_hint: str | None = None,
    include_disk_preamble: bool = False,
) -> tuple[dict[str, Any], str, str]:
    """Returns ``(parsed, raw_assistant_text, full_user_message)`` for logging into chat history."""
    ensure_openrouter_key()
    client = get_openrouter_client()
    model = _agent_model()

    tried_sorted = sorted(completed)
    if len(tried_sorted) > 120:
        head = "\n".join(f"- lr={lr:g}, rank={rank}" for lr, rank in tried_sorted[:40])
        tail = "\n".join(f"- lr={lr:g}, rank={rank}" for lr, rank in tried_sorted[-40:])
        tried_lines = f"{head}\n- … ({len(tried_sorted) - 80} more) …\n{tail}"
    else:
        tried_lines = "\n".join(f"- lr={lr:g}, rank={rank}" for lr, rank in tried_sorted) or "- (none yet)"

    phase_hint = _phase_hint(band_index, max_bands)

    preamble = ""
    if include_disk_preamble and "_No scored runs yet_" not in summary_md:
        preamble = (
            "**Cold start — no prior controller chat on disk.** Below is every **scored (lr, rank) run** "
            "currently under `experiment_results/` from past experiments (do not repeat those pairs).\n\n"
            f"{summary_md}\n\n---\n\n"
        )

    rubric_section = (
        "**(Rubric table is in the preamble above.)**\n"
        if include_disk_preamble and preamble
        else f"**Current rubric table (all scored runs on disk right now):**\n{summary_md}"
    )

    rank_cap = tinker_max_lora_rank()
    user_core = f"""You control an automated hyperparameter search for LoRA SFT on Tinker.

**Goal:** Thomas’s texts should feel like a real person texting: **personality**, **not repetitive**, **not bland** / not generic-assistant. Rubric scores (1–5) reflect that.

**Band:** {_band_label(band_index, max_bands)}
**How many combos this band:** You choose how many distinct **(lr, rank)** pairs to run **in parallel** this band.
Pick **any count from 1 up to {max_combos_per_band}** (hard cap). Use **more** runs when you want broad exploration
or to bracket uncertainty; use **fewer** when you want cheap refinement or after failures. All chosen pairs train
at the same time (up to the cap).

**Already completed (do NOT repeat):**
{tried_lines}

{phase_hint}

{rubric_section}

{retry_hint or ""}

Reply with **only** valid JSON (no markdown fences), shape:
{{
  "phase": "explore" | "narrow",
  "reasoning": "<why this many runs and these picks; cite prior results / chat history>",
  "next_pairs": [ {{"lr": <float>, "rank": <int>}}, ... ]
}}

Rules:
- ``rank`` must be a **power of 2** and **≤ {rank_cap}** for the current Tinker base model (openai/gpt-oss-120b rejects rank 64; cap via env ``TINKER_MAX_LORA_RANK`` if you use a different model).
- ``lr`` must be positive; typical range ~5e-6 to 5e-4 for Adam on this stack.
- Every pair in ``next_pairs`` must be **absent** from the completed list; **no duplicates** inside ``next_pairs``.
- ``next_pairs`` length must be **≥ 1** and **≤ {max_combos_per_band}** (if you list more, only the first {max_combos_per_band} valid new pairs will be used).
"""

    full_user = (
        (prior_outcomes.strip() + "\n\n---\n\n") if prior_outcomes and prior_outcomes.strip() else ""
    ) + (preamble if preamble else "") + user_core

    system = (
        "You are a careful ML engineer optimizing LoRA learning rate and rank for conversational "
        "SMS finetuning. You have an ongoing message history: remember your past JSON plans, reasoning, "
        "and the execution outcomes the user reported after each band. "
        "If a run failed (Tinker/OpenRouter/subprocess error), read the error text and propose adjusted "
        "hyperparameters — do not assume the sweep stopped. Always output valid JSON only "
        "for each new plan (no markdown fences)."
    )

    api_messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    api_messages.extend(history)
    api_messages.append({"role": "user", "content": full_user})

    r = client.chat.completions.create(
        model=model,
        messages=api_messages,
        temperature=0.35,
        max_tokens=4096,
    )
    text = (r.choices[0].message.content or "").strip()
    parsed = _extract_json_object(text)
    if not isinstance(parsed, dict):
        return {"error": "parse", "raw": text}, text, full_user
    return parsed, text, full_user


def _normalize_proposed_pairs(raw: Any, *, max_pairs: int) -> list[tuple[float, int]]:
    """Parse model ``next_pairs`` in order (allow extra entries before filter/dedupe)."""
    out: list[tuple[float, int]] = []
    if not isinstance(raw, list):
        return out
    cap = max(1, max_pairs) * 3
    for item in raw:
        if len(out) >= cap:
            break
        if not isinstance(item, dict):
            continue
        lr = item.get("lr")
        rank = item.get("rank")
        if lr is None or rank is None:
            continue
        try:
            out.append((_norm_lr(float(lr)), int(rank)))
        except (TypeError, ValueError):
            continue
    return out


def _train_eval_combo_task(
    lr: float,
    rank: int,
    *,
    steps: int,
    input_json: Path,
    completed: set[tuple[float, int]],
    state_lock: threading.Lock,
    log_lock: threading.Lock,
    band_no: int,
) -> dict[str, Any]:
    """Run one combo in a worker thread; returns ``{ok, lr, rank, info?, error?}``."""
    print(f"\n[agent] Band {band_no}: train+eval lr={lr:g} rank={rank} (parallel worker)", flush=True)
    try:
        info = run_one_combo_train_eval(lr, rank, steps=steps, input_json=input_json)
        with state_lock:
            completed.add(_pair_key(lr, rank))
            _save_state(completed)
        with log_lock:
            _append_agent_log(
                f"**Finished** lr={lr:g} rank={rank} | seed={info.get('seed')} | "
                f"mean_rubric={info.get('mean_rubric')} | dir=`{info.get('out_dir')}`"
            )
        return {"ok": True, "lr": lr, "rank": rank, "info": info, "error": None}
    except (Exception, SystemExit) as exc:
        err_s = str(exc).strip().replace("\n", " ")
        if len(err_s) > 900:
            err_s = err_s[:900] + "…"
        with log_lock:
            _append_agent_log(f"**TRAIN/EVAL FAILED** lr={lr:g} rank={rank}: {exc!r}")
        print(
            f"\n[agent] FAILED lr={lr:g} rank={rank}: {err_s}\n"
            "[agent] (parallel worker) — other jobs may still be running.\n",
            flush=True,
        )
        return {"ok": False, "lr": lr, "rank": rank, "info": None, "error": err_s}


def run_one_combo_train_eval(
    lr: float,
    rank: int,
    *,
    steps: int,
    input_json: Path,
) -> dict[str, Any]:
    """Train + full scenario eval; writes under ``experiment_results/lr…_r…/``."""
    seed = _dedupe_training_seed(lr, rank)
    meta = train_one_combo(lr, rank, steps=steps, input_json=input_json, seed=seed)
    uri = _load_tinker_uri(meta)
    out_dir = combo_dir(lr, rank)
    run_eval_for_uri(
        tinker_uri=uri,
        out_dir=out_dir,
        lr=lr,
        lora_rank=rank,
        metadata_source=meta,
    )
    # Post-hoc rubric mean for log
    means = []
    for sid in SCENARIO_IDS:
        jf = out_dir / f"{sid}.json"
        if jf.is_file():
            data = json.loads(jf.read_text(encoding="utf-8"))
            m = _rubric_mean(data.get("rubric") or {})
            if m is not None:
                means.append(m)
    return {
        "lr": lr,
        "rank": rank,
        "seed": seed,
        "mean_rubric": sum(means) / len(means) if means else None,
        "out_dir": str(out_dir),
    }


def launch_agent_sweep(
    *,
    input_path: Path,
    max_bands: int | None,
    max_combos_per_band: int,
    steps: int,
) -> None:
    load_dotenv(PACKAGE_ROOT / ".env")
    EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    inp = input_path.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"SFT JSON not found: {inp}")

    infinite = max_bands is None or max_bands <= 0
    cap = None if infinite else max(1, int(max_bands))

    # Deterministic eval / friend side for cross-run comparability
    os.environ.setdefault("MESSAGES_EXPERIMENT_FRIEND_TEMPERATURE", "0")
    os.environ.setdefault("MESSAGES_EXPERIMENT_THOMAS_TEMPERATURE", "0")

    mcb = max(1, min(int(max_combos_per_band), 32))
    completed = load_completed_pairs()
    _append_agent_log(
        f"**Agent sweep started**\n\n"
        f"- mode={'infinite (Ctrl+C to stop)' if infinite else f'capped at {cap} band(s)'}\n"
        f"- max_combos_per_band={mcb} (controller picks 1..{mcb} per band; all run in parallel)\n"
        f"- steps={steps}\n"
        f"- input={inp}\n"
        f"- controller_model={_agent_model()!r}\n"
        f"- already_completed={len(completed)} pair(s)\n"
    )
    if infinite:
        print(
            "\n[agent] Infinite mode: new bands until Ctrl+C (positive --bands caps rounds).\n",
            flush=True,
        )

    chat_history = load_controller_history()
    prior_outcomes: str | None = None
    print(
        f"[agent] Controller chat history: {_history_path()} ({len(chat_history)} message(s) loaded)\n",
        flush=True,
    )

    band = 0
    try:
        while True:
            if cap is not None and band >= cap:
                break

            summary = summarize_experiment_results()
            retry_hint: str | None = None
            pairs: list[tuple[float, int]] = []
            raw_controller: dict[str, Any] | None = None
            last_full_user = ""
            last_raw_text = ""

            disk_preamble = (
                band == 0
                and not chat_history
                and "_No scored runs yet_" not in summary
                and len(completed) > 0
            )

            for attempt in range(5 if infinite else 3):
                parsed, raw_text, full_user = call_controller(
                    band_index=band,
                    max_bands=cap,
                    max_combos_per_band=mcb,
                    completed=completed,
                    summary_md=summary,
                    history=chat_history,
                    prior_outcomes=prior_outcomes,
                    retry_hint=retry_hint,
                    include_disk_preamble=disk_preamble,
                )
                raw_controller = parsed
                last_full_user = full_user
                last_raw_text = raw_text
                if parsed.get("error"):
                    retry_hint = f"Previous output was not valid JSON. Raw snippet: {str(parsed.get('raw', ''))[:500]}"
                    continue

                cand = _normalize_proposed_pairs(parsed.get("next_pairs"), max_pairs=mcb)
                filtered: list[tuple[float, int]] = []
                seen_in_batch: set[tuple[float, int]] = set()
                for lr, rank in cand:
                    if len(filtered) >= mcb:
                        break
                    k = _pair_key(lr, rank)
                    if k in completed or k in seen_in_batch:
                        continue
                    if not _rank_ok_for_tinker(rank):
                        continue
                    seen_in_batch.add(k)
                    filtered.append((lr, rank))
                if 1 <= len(filtered) <= mcb:
                    pairs = filtered
                    break
                retry_hint = (
                    f"You proposed {len(cand)} parseable pair(s); after removing duplicates vs completed "
                    f"and invalid ranks (power of 2 and ≤ {tinker_max_lora_rank()} for Tinker), only {len(filtered)} valid **new** pair(s) remain. "
                    f"You must output **between 1 and {mcb}** distinct new pairs in `next_pairs`. "
                    f"Completed (do not repeat): {sorted(completed)!r}"
                )

            reasoning = ""
            if isinstance(raw_controller, dict):
                reasoning = str(raw_controller.get("reasoning", "")).strip()
            phase = str(raw_controller.get("phase", "")).strip() if isinstance(raw_controller, dict) else ""

            _append_agent_log(
                f"### Band {_band_label(band, cap)}\n\n"
                f"**phase:** {phase or '?'}\n\n"
                f"**reasoning:**\n{reasoning or '(none)'}\n\n"
                f"**raw next_pairs (pre-filter):** `{json.dumps(raw_controller.get('next_pairs') if isinstance(raw_controller, dict) else None)}`\n"
            )

            if not pairs:
                msg = "_No valid new pairs this band._"
                if infinite:
                    _append_agent_log(msg + " Retrying after a short wait (infinite mode).")
                    print("[agent] No valid pairs; waiting 5s and continuing…", flush=True)
                    time.sleep(5)
                    band += 1
                    continue
                _append_agent_log(msg + " Stopping.")
                break

            if not raw_controller.get("error") and last_full_user and last_raw_text:
                chat_history.append({"role": "user", "content": last_full_user})
                chat_history.append({"role": "assistant", "content": last_raw_text})
                chat_history = chat_history[-HISTORY_MAX_MESSAGES:]
                save_controller_history(chat_history)

            ran_infos: list[dict[str, Any]] = []
            skip_msgs: list[str] = []
            error_msgs: list[str] = []
            ran_any = False
            state_lock = threading.Lock()
            log_lock = threading.Lock()
            to_parallel: list[tuple[float, int]] = []

            for lr, rank in pairs:
                if _combo_fully_evaluated(lr, rank):
                    k = _pair_key(lr, rank)
                    with state_lock:
                        completed.add(k)
                        _save_state(completed)
                    skip_msgs.append(f"lr={lr:g} rank={rank} (folder already complete)")
                    with log_lock:
                        _append_agent_log(
                            f"**Skipped** (results already present): lr={lr:g} rank={rank} — not re-training."
                        )
                    print(f"\n[agent] Skip existing results lr={lr:g} rank={rank}", flush=True)
                else:
                    to_parallel.append((lr, rank))

            if to_parallel:
                ran_any = True
                bn = band + 1
                print(
                    f"\n[agent] Band {bn}: running {len(to_parallel)} train+eval job(s) **in parallel**…\n",
                    flush=True,
                )
                with ThreadPoolExecutor(max_workers=len(to_parallel)) as pool:
                    futures = {
                        pool.submit(
                            _train_eval_combo_task,
                            lr,
                            rank,
                            steps=steps,
                            input_json=inp,
                            completed=completed,
                            state_lock=state_lock,
                            log_lock=log_lock,
                            band_no=bn,
                        ): (lr, rank)
                        for lr, rank in to_parallel
                    }
                    batch_results: list[dict[str, Any]] = []
                    for fut in as_completed(futures):
                        batch_results.append(fut.result())
                    for r in sorted(batch_results, key=lambda x: (x["lr"], x["rank"])):
                        if r.get("ok") and r.get("info"):
                            ran_infos.append(r["info"])
                        elif not r.get("ok"):
                            err = r.get("error") or "unknown error"
                            error_msgs.append(f"lr={r['lr']:g} rank={r['rank']}: {err}")
                print(
                    f"\n[agent] Band {bn} parallel batch finished "
                    f"({len(ran_infos)} ok, {len(error_msgs)} failed).\n",
                    flush=True,
                )

            prior_outcomes = _build_outcomes_block(
                band_display=_band_label(band, cap),
                ran=ran_infos,
                skips=skip_msgs,
                errors=error_msgs,
            )

            if not ran_any:
                msg = "_No runnable pairs this band (all skipped as already complete)._"
                if infinite:
                    _append_agent_log(
                        msg + " Waiting and continuing — propose **different** (lr, rank) not in the list."
                    )
                    print("[agent] All proposals already on disk; waiting 5s and continuing…", flush=True)
                    time.sleep(5)
                    band += 1
                    continue
                _append_agent_log(msg + " Stopping.")
                break

            band += 1

    except KeyboardInterrupt:
        save_controller_history(chat_history)
        _append_agent_log("**Stopped by user (KeyboardInterrupt).**\n")
        print("\n[agent] Interrupted. Controller chat history saved; resume by running again.\n", flush=True)
        raise

    if cap is not None:
        _append_agent_log("**Agent sweep finished (reached band cap).**\n")


def main_cli() -> None:
    load_dotenv(PACKAGE_ROOT / ".env")
    p = argparse.ArgumentParser(description="Claude-controlled LR×LoRA agent sweep")
    p.add_argument(
        "--bands",
        type=int,
        default=0,
        help="0 = run forever until Ctrl+C; N>0 = stop after N controller bands",
    )
    p.add_argument(
        "--max-per-band",
        type=int,
        default=DEFAULT_MAX_COMBOS_PER_BAND,
        metavar="N",
        help="Hard cap: controller may propose 1..N (lr,rank) pairs per band (default 9)",
    )
    p.add_argument(
        "--per-band",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument("--steps", type=int, default=200, help="Tinker training steps per combo")
    p.add_argument("--input", type=Path, default=PACKAGE_ROOT / "sft_output.json")
    args = p.parse_args()
    mxb = args.per_band if args.per_band is not None else args.max_per_band
    launch_agent_sweep(
        input_path=args.input,
        max_bands=args.bands,
        max_combos_per_band=max(1, mxb),
        steps=max(1, args.steps),
    )


if __name__ == "__main__":
    main_cli()
