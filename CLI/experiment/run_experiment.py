#!/usr/bin/env python3
"""
LR × LoRA experimentation: optional Tinker train sweep + fixed opener + OpenRouter friend ↔ Thomas + rubric JSON.

Run from ``imessageWithContextExperimentation/``:

  python3 -m experiment.run_experiment eval --metadata sft_tinker_metadata.json
  python3 -m experiment.run_experiment train-sweep --lrs 1e-4,1.5e-4 --ranks 8,16
  python3 -m experiment.run_experiment train-sweep --parallel 3
  python3 -m experiment.run_experiment full --lrs 1e-4 --ranks 16

Env: TINKER_API_KEY, OPEN_ROUTER_API_KEY (OpenRouter → Claude by default for rubric rating).
Optional OPEN_ROUTER_RATER_MODEL (slug, e.g. anthropic/claude-sonnet-4.5).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from experiment.config import (
    DEFAULT_LEARNING_RATES,
    DEFAULT_LORA_RANKS,
    EXPERIMENT_RESULTS,
    MAX_ROLEPLAY_MESSAGES,
    PACKAGE_ROOT,
    default_openrouter_friend_model,
    default_openrouter_rater_model,
    tinker_max_lora_rank,
)
from experiment.gemini_roleplay import friend_continue, parse_friend_raw, start_friend_chat
from experiment.openrouter_client import ensure_openrouter_key, openrouter_base_url
from experiment.rate_thread import rate_transcript
from experiment.scenarios import SCENARIOS
from experiment.thomas_client import build_thomas_messages, complete_thomas, open_tinker_client


def _load_tinker_uri(metadata_path: Path) -> str:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    uri = meta.get("tinker_checkpoint_path")
    if not isinstance(uri, str) or not uri.startswith("tinker://"):
        raise SystemExit(f"No tinker_checkpoint_path in {metadata_path}")
    return uri


def format_thread_for_rating(turns: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for t in turns:
        role, text = t["role"], t["text"]
        if role == "friend":
            lines.append(f"[Friend] {text}")
        else:
            lines.append(f"[Thomas] {text}")
    return "\n".join(lines)


def run_scenario_dialogue(
    *,
    client,
    tinker_uri: str,
    scenario_id: str,
) -> dict:
    scenario = next(s for s in SCENARIOS if s.id == scenario_id)
    turns: list[dict[str, str]] = []
    pairs: list[tuple[str, str]] = []
    message_count = 0
    stopped_reason = "ok"

    incoming = scenario.first_friend_message.strip()
    end_mode = "none"
    chat = None

    while message_count < MAX_ROLEPLAY_MESSAGES:
        if not incoming:
            stopped_reason = "empty_incoming"
            break

        msgs = build_thomas_messages(pairs)
        msgs.append({"role": "user", "content": incoming})
        thomas = complete_thomas(client=client, tinker_model_uri=tinker_uri, messages=msgs)
        pairs.append((incoming, thomas))
        turns.append({"role": "friend", "text": incoming})
        turns.append({"role": "thomas", "text": thomas})
        message_count += 2

        if end_mode == "stop_after_thomas":
            stopped_reason = "friend_signaled_end_after_thomas"
            break
        if message_count >= MAX_ROLEPLAY_MESSAGES:
            stopped_reason = "max_messages"
            break

        if chat is None:
            chat = start_friend_chat(
                scenario.friend_instruction,
                seed_assistant_text=scenario.first_friend_message.strip(),
            )
        raw = friend_continue(chat, thomas)
        incoming, end_mode = parse_friend_raw(raw)
        if end_mode == "stop_now" and not incoming:
            stopped_reason = "friend_endconv"
            break

    if stopped_reason == "ok" and not scenario.first_friend_message.strip():
        stopped_reason = "no_opener"

    return {
        "scenario_id": scenario.id,
        "scenario_title": scenario.title,
        "turns": turns,
        "transcript_for_rating": format_thread_for_rating(turns),
        "stopped_reason": stopped_reason,
    }


def run_eval_for_uri(
    *,
    tinker_uri: str,
    out_dir: Path,
    lr: float | None,
    lora_rank: int | None,
    metadata_source: Path | None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_openrouter_key()
    client = open_tinker_client()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "learning_rate": lr,
        "lora_rank": lora_rank,
        "tinker_checkpoint_path": tinker_uri,
        "metadata_source": str(metadata_source) if metadata_source else None,
        "openrouter_base_url": openrouter_base_url(),
        "friend_turns_source": "first_message_fixed_then_openrouter",
        "openrouter_friend_model": default_openrouter_friend_model(),
        "openrouter_rater_model": default_openrouter_rater_model(),
        "max_messages": MAX_ROLEPLAY_MESSAGES,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for sc in SCENARIOS:
        print(f"  Scenario: {sc.id} …", flush=True)
        dialogue = run_scenario_dialogue(client=client, tinker_uri=tinker_uri, scenario_id=sc.id)
        rating = rate_transcript(dialogue["transcript_for_rating"])
        payload = {"dialogue": dialogue, "rubric": rating}
        (out_dir / f"{sc.id}.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"  Wrote results under {out_dir}", flush=True)


def combo_dir(lr: float, rank: int) -> Path:
    safe_lr = str(lr).replace(".", "p")
    return EXPERIMENT_RESULTS / f"lr{safe_lr}_r{rank}"


def train_one_combo(
    lr: float,
    rank: int,
    *,
    steps: int,
    input_json: Path,
    seed: int | None = None,
) -> Path:
    """Run Tinker SFT; return path to metadata JSON for this combo (written under experiment_results)."""
    out_dir = combo_dir(lr, rank)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = f"imessage-sft-lora-lr{str(lr).replace('.', 'p')}-r{rank}"
    script = PACKAGE_ROOT / "sft_tinker.py"
    cmd = [
        sys.executable,
        str(script),
        "--lr",
        str(lr),
        "--lora-rank",
        str(rank),
        "--steps",
        str(steps),
        "--input",
        str(input_json),
        "--checkpoint-name",
        ck,
    ]
    meta_dest = out_dir / "sft_tinker_metadata.json"
    dump_dest = out_dir / "sft_tinker_api_dump.json"
    cmd.extend(
        [
            "--metadata-out",
            str(meta_dest),
            "--api-dump-out",
            str(dump_dest),
        ]
    )
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    print(f"[train] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(PACKAGE_ROOT), env={**os.environ}, check=True)

    if not meta_dest.is_file():
        raise SystemExit(f"Missing {meta_dest} after training")
    try:
        from model_registry import register_from_metadata_file

        register_from_metadata_file(
            PACKAGE_ROOT,
            meta_dest,
            learning_rate=float(lr),
            lora_rank=int(rank),
            source="train",
        )
    except Exception:
        pass
    return meta_dest


def launch_train_sweep_interactive(
    *,
    input_path: Path,
    lrs: str,
    ranks: str,
    steps: int,
    parallel: int = 1,
) -> None:
    """Load ``.env``, ensure output root exists, run train+eval for each lr×rank (for ``main.py``)."""
    load_dotenv(PACKAGE_ROOT / ".env")
    EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(lrs=lrs, ranks=ranks, steps=steps, input=input_path, parallel=parallel)
    cmd_train_sweep(ns)


def _validate_tinker_lora_ranks(ranks: list[int]) -> None:
    """Tinker rejects non–power-of-two ranks and ranks above the model cap (e.g. gpt-oss-120b → max 32)."""
    cap = tinker_max_lora_rank()
    for r in ranks:
        if r < 1 or (r & (r - 1)) != 0:
            raise SystemExit(
                f"Tinker LoRA rank must be a power of 2 (e.g. 8, 16, 32); got {r!r}. "
                "Update ranks and re-run."
            )
        if r > cap:
            raise SystemExit(
                f"Tinker LoRA rank {r} exceeds max {cap} for this base model "
                f"(set TINKER_MAX_LORA_RANK if your model allows higher). Update ranks and re-run."
            )


def _train_eval_one_combo(lr: float, rank: int, *, steps: int, inp: Path) -> None:
    print(f"\n=== Training lr={lr} rank={rank} ===", flush=True)
    meta = train_one_combo(lr, rank, steps=steps, input_json=inp)
    uri = _load_tinker_uri(meta)
    run_eval_for_uri(
        tinker_uri=uri,
        out_dir=combo_dir(lr, rank),
        lr=lr,
        lora_rank=rank,
        metadata_source=meta,
    )


def cmd_train_sweep(args: argparse.Namespace) -> None:
    lrs = [float(x.strip()) for x in args.lrs.split(",") if x.strip()]
    ranks = [int(x.strip()) for x in args.ranks.split(",") if x.strip()]
    inp = Path(args.input).resolve()
    _validate_tinker_lora_ranks(ranks)
    parallel = max(1, int(getattr(args, "parallel", 1) or 1))
    combos = [(lr, r) for lr in lrs for r in ranks]

    if parallel <= 1:
        for lr, rank in combos:
            _train_eval_one_combo(lr, rank, steps=args.steps, inp=inp)
        return

    print(
        f"[train-sweep] Running up to {parallel} lr×rank combos in parallel "
        f"({len(combos)} total). Tinker/OpenRouter quotas may limit real throughput.\n",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(_train_eval_one_combo, lr, rank, steps=args.steps, inp=inp): (lr, rank)
            for lr, rank in combos
        }
        for fut in as_completed(futures):
            lr, rank = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                raise SystemExit(f"Combo lr={lr} rank={rank} failed: {exc}") from exc


def cmd_eval(args: argparse.Namespace) -> None:
    load_dotenv(PACKAGE_ROOT / ".env")
    meta_path = Path(args.metadata).resolve()
    uri = _load_tinker_uri(meta_path)
    if args.out:
        out = Path(args.out).resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out = EXPERIMENT_RESULTS / f"eval_{meta_path.stem}_{stamp}"
    lr = float(args.lr) if args.lr is not None else None
    rank = int(args.rank) if args.rank is not None else None
    run_eval_for_uri(
        tinker_uri=uri,
        out_dir=out,
        lr=lr,
        lora_rank=rank,
        metadata_source=meta_path,
    )


def cmd_full(args: argparse.Namespace) -> None:
    """train_sweep then each eval is inside train_one in train_sweep already for full - actually cmd_train_sweep already runs eval after each train. cmd_full could duplicate - user asked full = train + eval which is same as train_sweep. I'll make `full` alias to train_sweep."""
    cmd_train_sweep(args)


def main() -> None:
    load_dotenv(PACKAGE_ROOT / ".env")
    p = argparse.ArgumentParser(
        description="LR/LoRA Tinker + fixed opener + OpenRouter friend + Claude rubric"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train-sweep", help="Train each lr×rank (Tinker), then eval each")
    pt.add_argument(
        "--lrs",
        default=",".join(str(x) for x in DEFAULT_LEARNING_RATES),
        help="Comma-separated learning rates",
    )
    pt.add_argument(
        "--ranks",
        default=",".join(str(x) for x in DEFAULT_LORA_RANKS),
        help="Comma-separated LoRA ranks",
    )
    pt.add_argument("--steps", type=int, default=200)
    pt.add_argument("--input", type=Path, default=PACKAGE_ROOT / "sft_output.json")
    pt.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Run up to N lr×rank combos concurrently (default 1 = serial). Uses thread pool; each combo writes its own metadata under experiment_results/.",
    )
    pt.set_defaults(func=cmd_train_sweep)

    pe = sub.add_parser("eval", help="Eval only from existing sft_tinker_metadata.json")
    pe.add_argument("--metadata", type=Path, required=True)
    pe.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: same folder as metadata)",
    )
    pe.add_argument("--lr", type=float, default=None, help="Label for manifest only")
    pe.add_argument("--rank", type=int, default=None, help="Label for manifest only")
    pe.set_defaults(func=cmd_eval)

    pf = sub.add_parser("full", help="Same as train-sweep (train + eval per combo)")
    pf.add_argument("--lrs", default=",".join(str(x) for x in DEFAULT_LEARNING_RATES))
    pf.add_argument("--ranks", default=",".join(str(x) for x in DEFAULT_LORA_RANKS))
    pf.add_argument("--steps", type=int, default=200)
    pf.add_argument("--input", type=Path, default=PACKAGE_ROOT / "sft_output.json")
    pf.add_argument(
        "--parallel",
        type=int,
        default=1,
        metavar="N",
        help="Same as train-sweep --parallel",
    )
    pf.set_defaults(func=cmd_full)

    args = p.parse_args()
    EXPERIMENT_RESULTS.mkdir(parents=True, exist_ok=True)
    args.func(args)


if __name__ == "__main__":
    main()
