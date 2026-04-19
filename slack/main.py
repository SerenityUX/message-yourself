#!/usr/bin/env python3
"""
Slack pipeline: optional export → ``prepare_slack_cpt`` → optional Tinker LoRA CPT + inference smoke test.

Interactive (default). **Use the project venv** (Python 3.11+) before training; otherwise
``python3`` may be macOS/Xcode 3.9 and Tinker will refuse to run::

    cd slack && source .venv/bin/activate
    python main.py

You will be asked:

1. Export Slack messages? (Y/N) — writes ``my_slack_messages.json``
2. Regenerate ``cpt_out.txt``? (Y/N) — runs ``prepare_slack_cpt``
3. Train on Tinker using ``cpt_out.txt``? (Y/N) — choose **CPT**, **RL** (cold), or **CPT then RL**
   (recommended: CPT, then RL loaded on top of that LoRA), then **Qwen3-8B** vs **Qwen3-4B-Instruct**.
   Metadata / registry: CPT → ``cpt_tinker_metadata.json``; RL-only → ``rl_tinker_metadata.json``;
   CPT+RL → final ``rl_tinker_metadata.json`` (CPT checkpoint kept as intermediate).

4. After training (or if you skip training but have saved models), optional **chat** via ``chat.py``.

Non-interactive (export + prepare only, same as before)::

    cd slack && python3 main.py --no-prompt
    cd slack && python3 main.py --no-prompt --start-from 2025-05-01

With training and no prompts::

    cd slack && python3 main.py --no-prompt --train
    cd slack && python3 main.py --no-prompt --train --rl
    cd slack && python3 main.py --no-prompt --train --cpt-then-rl
    cd slack && python3 main.py --no-prompt --train --cpt-then-rl --base-model Qwen/Qwen3-4B-Instruct-2507

Override export window: ``SLACK_EXPORT_START=… SLACK_EXPORT_END=…`` (see ``exportSlackMessages.py``).

Export auth: set ``SLACK_XOXC_TOKEN`` and ``SLACK_D_COOKIE`` in ``slack/.env``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path


SLACK_DIR = Path(__file__).resolve().parent
# Tinker IDs for LoRA CPT (see https://tinker-docs.thinkingmachines.ai/tinker/models/ ).
BASE_MODEL_8B = "Qwen/Qwen3-8B"
BASE_MODEL_4B_INSTRUCT = "Qwen/Qwen3-4B-Instruct-2507"
# Default when user does not choose (interactive default selection + --no-prompt --train).
DEFAULT_BASE_MODEL = BASE_MODEL_8B

# Corpus RL from **base** LoRA (cold): shorter run, sampling tuned for exploration.
_RL_COLD_BATCHES = 20
_RL_COLD_LR = 3e-5
_RL_COLD_MAX_TOKENS = 96
_RL_COLD_TEMPERATURE = 0.7
_RL_COLD_CHECKPOINT = "slack-rl-corpus"

# RL **after CPT** (warm-start): more batches, cooler/shorter rollouts — easier oracle match, stabler updates.
_RL_WARM_BATCHES = 40
_RL_WARM_LR = 2.5e-5
_RL_WARM_MAX_TOKENS = 96
_RL_WARM_TEMPERATURE = 0.65
_RL_WARM_CHECKPOINT = "slack-cpt-rl-corpus"

# (label for menus, tinker id) — older checkpoints trained on 4B remain valid; chat uses each URI as-is.
_BASE_MODEL_MENU: tuple[tuple[str, str], ...] = (
    ("Qwen3-8B (denser, higher train/sample cost)", BASE_MODEL_8B),
    ("Qwen3-4B-Instruct-2507 (lighter, cheaper)", BASE_MODEL_4B_INSTRUCT),
)


def _pick_base_model_interactive() -> str:
    """Let the user pick 8B vs 4B instruct; Enter defaults to 8B."""
    print("\n[main] Choose Tinker base model for this training run:\n", flush=True)
    for i, (label, tid) in enumerate(_BASE_MODEL_MENU):
        print(f"  [{i}] {label}", flush=True)
        print(f"      {tid}", flush=True)
    line = input(
        f"\nEnter index 0–{len(_BASE_MODEL_MENU) - 1} (blank = {DEFAULT_BASE_MODEL}): "
    ).strip()
    if not line:
        return DEFAULT_BASE_MODEL
    try:
        idx = int(line)
    except ValueError:
        print(f"[main] Invalid index; using {DEFAULT_BASE_MODEL}.", flush=True)
        return DEFAULT_BASE_MODEL
    if idx < 0 or idx >= len(_BASE_MODEL_MENU):
        print(f"[main] Out of range; using {DEFAULT_BASE_MODEL}.", flush=True)
        return DEFAULT_BASE_MODEL
    return _BASE_MODEL_MENU[idx][1]


def _pick_training_mode_interactive() -> str:
    """Return ``\"cpt\"``, ``\"rl\"``, or ``\"cpt_then_rl\"``."""
    print("\n[main] Training mode:\n", flush=True)
    print("  [0] CPT — continued pre-training (next-token loss on cpt_out.txt)", flush=True)
    print("  [1] RL  — corpus RL from base model (cold start)", flush=True)
    print(
        "  [2] CPT then RL — CPT first, then RL on that LoRA (warm-start; recommended if you want RL)",
        flush=True,
    )
    line = input("\nEnter 0, 1, or 2 (blank = 0 CPT): ").strip().lower()
    if not line:
        return "cpt"
    if line in ("0", "cpt"):
        return "cpt"
    if line in ("1", "rl"):
        return "rl"
    if line in ("2", "cpt_rl", "cpt-then-rl", "both"):
        return "cpt_then_rl"
    print("[main] Invalid choice; using CPT.", flush=True)
    return "cpt"


def _yn(prompt: str, *, default: bool | None = None) -> bool:
    if default is True:
        hint = " [Y/n]: "
    elif default is False:
        hint = " [y/N]: "
    else:
        hint = " [y/n]: "
    line = input(f"{prompt}{hint}").strip().lower()
    if not line and default is not None:
        return default
    return line in ("y", "yes")


def main() -> None:
    os.chdir(SLACK_DIR)

    parent = argparse.ArgumentParser(
        description="Slack export → CPT text → optional Tinker LoRA training.",
    )
    parent.add_argument(
        "--no-prompt",
        action="store_true",
        help="Run export + prepare without questions (pass export flags after this). Use --train to also train.",
    )
    parent.add_argument(
        "--train",
        action="store_true",
        help="With --no-prompt, run Tinker training (no questions).",
    )
    parent.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        metavar="TINKER_ID",
        help=(
            f"Tinker base for training (default: {DEFAULT_BASE_MODEL}). "
            f"Common: {BASE_MODEL_8B} | {BASE_MODEL_4B_INSTRUCT}."
        ),
    )
    parent.add_argument(
        "--rl",
        action="store_true",
        help="With --no-prompt --train, run corpus RL from base (cold). Ignored if --cpt-then-rl.",
    )
    parent.add_argument(
        "--cpt-then-rl",
        action="store_true",
        help="With --no-prompt --train, run CPT then corpus RL on the CPT sampler (warm-start).",
    )
    main_args, rest = parent.parse_known_args()

    from exportSlackMessages import export_slack_messages, parse_export_args
    from prepare_slack_cpt import main as prepare_slack_cpt

    exp = parse_export_args(rest if rest else None)
    last_train_result: dict | None = None

    if main_args.no_prompt:
        print("\n[main] Non-interactive: export → prepare_slack_cpt\n", flush=True)
        export_slack_messages(
            start_from=exp.start_from,
            start_page=exp.start_page,
            load_existing=not exp.fresh,
            fresh=exp.fresh,
        )
        prepare_slack_cpt()
        if main_args.train:
            if main_args.cpt_then_rl:
                last_train_result = _run_cpt_then_rl_training(base_model=main_args.base_model)
            elif main_args.rl:
                last_train_result = _run_rl_training(base_model=main_args.base_model)
            else:
                last_train_result = _run_training(base_model=main_args.base_model)
        return

    print("\n[main] Slack pipeline (interactive)\n", flush=True)

    if _yn("Export Slack messages to my_slack_messages.json?", default=False):
        print("\n[main] exportSlackMessages\n", flush=True)
        export_slack_messages(
            start_from=exp.start_from,
            start_page=exp.start_page,
            load_existing=not exp.fresh,
            fresh=exp.fresh,
        )
    else:
        print("[main] Skipped export.\n", flush=True)

    if _yn("Regenerate cpt_out.txt from my_slack_messages.json (prepare_slack_cpt)?", default=True):
        print("\n[main] prepare_slack_cpt → cpt_out.txt\n", flush=True)
        prepare_slack_cpt()
    else:
        print("[main] Skipped prepare_slack_cpt.\n", flush=True)

    if _yn("Train LoRA on Tinker using cpt_out.txt?", default=False):
        mode = _pick_training_mode_interactive()
        base_model = _pick_base_model_interactive()
        if mode == "rl":
            last_train_result = _run_rl_training(base_model=base_model)
        elif mode == "cpt_then_rl":
            last_train_result = _run_cpt_then_rl_training(base_model=base_model)
        else:
            last_train_result = _run_training(base_model=base_model)
    else:
        print("[main] Skipped training.\n", flush=True)

    _maybe_interactive_chat(last_train_result)


def _maybe_interactive_chat(last_train_result: dict | None) -> None:
    """Offer OpenAI-compatible chat via ``chat.py`` (requires 3.11+ venv)."""

    if sys.version_info < (3, 11):
        return
    try:
        from outputted_models import list_models
    except ImportError:
        return

    from chat import run_interactive_chat

    if last_train_result and last_train_result.get("tinker_checkpoint_path"):
        if _yn(
            "Chat with the model you just trained (OpenAI-compatible Tinker API)?",
            default=True,
        ):
            run_interactive_chat(model_path=str(last_train_result["tinker_checkpoint_path"]))
            return

    if list_models() and _yn(
        "Chat with a saved model from outputted_models.json (pick in menu)?",
        default=False,
    ):
        run_interactive_chat(model_path=None)


def _require_python_311_for_tinker() -> None:
    """Tinker SDK requires 3.11+; macOS ``python3`` is often 3.9 unless the venv is activated."""

    if sys.version_info >= (3, 11):
        return
    venv_python = SLACK_DIR / ".venv" / "bin" / "python"
    msg = [
        "[main] Tinker training needs Python 3.11+, but this process is:",
        f"       {sys.executable}",
        f"       → Python {sys.version.split()[0]}",
        "",
    ]
    if venv_python.is_file():
        msg.extend(
            [
                "Activate the slack venv (then run main again), or call the venv Python directly:",
                f"  cd {SLACK_DIR}",
                "  source .venv/bin/activate",
                "  python main.py",
                "",
                "One-shot:",
                f"  {venv_python} {SLACK_DIR / 'main.py'}",
            ]
        )
    else:
        msg.append(
            "No slack/.venv found. Create one with Python 3.11+ (e.g. python3.12 -m venv .venv) "
            "and pip install -r requirements.txt — see requirements.txt header."
        )
    raise SystemExit("\n".join(msg))


def _run_training(*, base_model: str) -> dict:
    _require_python_311_for_tinker()

    from dotenv import load_dotenv

    load_dotenv(SLACK_DIR / ".env")

    from continued_pretrain import (
        DEFAULT_API_DUMP_FILE,
        DEFAULT_CHECKPOINT_NAME,
        DEFAULT_CPT_FILE,
        run_cpt_async,
    )

    inp = DEFAULT_CPT_FILE
    if not inp.is_file():
        raise SystemExit(f"Missing {inp.name}; run prepare_slack_cpt (or export + prepare) first.")

    print(
        f"\n[main] Tinker LoRA CPT + smoke inference (Sampling API) | base_model={base_model!r}\n",
        flush=True,
    )
    result = asyncio.run(
        run_cpt_async(
            input_path=inp,
            base_model=base_model,
            lora_rank=16,
            steps=200,
            batch_size=1,
            lr=2e-4,
            max_length=2048,
            chunk_chars=7000,
            stride_chars=3000,
            checkpoint_name=DEFAULT_CHECKPOINT_NAME,
            seed=42,
            download_adapter_dir=None,
            skip_indefinite_ttl=False,
            run_smoke_inference=True,
            api_dump_path=DEFAULT_API_DUMP_FILE,
        )
    )
    _print_training_summary(result, mode="cpt")
    return result


def _run_rl_training(*, base_model: str) -> dict:
    """Corpus RL from base weights (cold)."""

    _require_python_311_for_tinker()

    from dotenv import load_dotenv

    load_dotenv(SLACK_DIR / ".env")

    import logging

    from continued_pretrain import DEFAULT_CPT_FILE
    from rl_corpus import run_rl

    inp = DEFAULT_CPT_FILE
    if not inp.is_file():
        raise SystemExit(f"Missing {inp.name}; run prepare_slack_cpt (or export + prepare) first.")

    logging.basicConfig(level=logging.INFO, format="[rl] %(message)s")

    print(
        f"\n[main] Tinker corpus RL (cold start, importance_sampling) | base_model={base_model!r}\n",
        flush=True,
    )
    result = run_rl(
        input_path=inp,
        base_model=base_model,
        lora_rank=16,
        batches=_RL_COLD_BATCHES,
        batch_size=4,
        group_size=8,
        learning_rate=_RL_COLD_LR,
        max_tokens=_RL_COLD_MAX_TOKENS,
        temperature=_RL_COLD_TEMPERATURE,
        min_prefix_tokens=48,
        min_suffix_tokens=32,
        checkpoint_name=_RL_COLD_CHECKPOINT,
        seed=42,
        chunk_chars=7000,
        stride_chars=3000,
        resume_state_path=None,
        init_from_tinker_path=None,
    )
    _print_training_summary(result, mode="rl")
    return result


def _run_cpt_then_rl_training(*, base_model: str) -> dict:
    """CPT (``continued_pretrain``), then RL with LoRA loaded from CPT **training** checkpoint (``…/weights/…``)."""

    _require_python_311_for_tinker()

    from dotenv import load_dotenv

    load_dotenv(SLACK_DIR / ".env")

    import logging

    from continued_pretrain import (
        DEFAULT_API_DUMP_FILE,
        DEFAULT_CHECKPOINT_NAME,
        DEFAULT_CPT_FILE,
        run_cpt_async,
    )
    from rl_corpus import run_rl

    inp = DEFAULT_CPT_FILE
    if not inp.is_file():
        raise SystemExit(f"Missing {inp.name}; run prepare_slack_cpt (or export + prepare) first.")

    print(
        f"\n[main] Phase 1/2: CPT | base_model={base_model!r}\n",
        flush=True,
    )
    cpt_result = asyncio.run(
        run_cpt_async(
            input_path=inp,
            base_model=base_model,
            lora_rank=16,
            steps=200,
            batch_size=1,
            lr=2e-4,
            max_length=2048,
            chunk_chars=7000,
            stride_chars=3000,
            checkpoint_name=DEFAULT_CHECKPOINT_NAME,
            seed=42,
            download_adapter_dir=None,
            skip_indefinite_ttl=False,
            run_smoke_inference=True,
            api_dump_path=DEFAULT_API_DUMP_FILE,
        )
    )
    cpt_train_path = cpt_result.get("tinker_training_weights_path")
    if not cpt_train_path or not str(cpt_train_path).startswith("tinker://"):
        raise SystemExit(
            "[main] CPT did not return tinker_training_weights_path (save_state / …/weights/…). "
            "Sampler-only URIs cannot load into TrainingClient. Re-run CPT with an updated continued_pretrain.py."
        )

    logging.basicConfig(level=logging.INFO, format="[rl] %(message)s")

    print(
        f"\n[main] Phase 2/2: corpus RL (warm-start from CPT training weights) | init={cpt_train_path!r}\n",
        flush=True,
    )
    rl_result = run_rl(
        input_path=inp,
        base_model=base_model,
        lora_rank=16,
        batches=_RL_WARM_BATCHES,
        batch_size=4,
        group_size=8,
        learning_rate=_RL_WARM_LR,
        max_tokens=_RL_WARM_MAX_TOKENS,
        temperature=_RL_WARM_TEMPERATURE,
        min_prefix_tokens=48,
        min_suffix_tokens=32,
        checkpoint_name=_RL_WARM_CHECKPOINT,
        seed=42,
        chunk_chars=7000,
        stride_chars=3000,
        resume_state_path=None,
        init_from_tinker_path=str(cpt_train_path),
    )

    combined = {
        **rl_result,
        "cpt_metadata_path": cpt_result["metadata_path"],
        "cpt_tinker_checkpoint_path": cpt_result["tinker_checkpoint_path"],
        "cpt_tinker_training_weights_path": cpt_result.get("tinker_training_weights_path"),
        "training_flow": "cpt_then_rl",
    }
    _print_training_summary(combined, mode="cpt_then_rl")
    return combined


def _print_training_summary(result: dict, *, mode: str) -> None:
    _mode_label = {
        "cpt": "CPT",
        "rl": "RL (cold)",
        "cpt_then_rl": "CPT then RL (warm)",
    }.get(mode, mode)
    print("\n[main] Training finished.", flush=True)
    print(f"  Mode:                                 {_mode_label}", flush=True)
    if mode == "cpt_then_rl":
        print(f"  CPT metadata:                         {result.get('cpt_metadata_path', '?')}", flush=True)
        print(
            f"  CPT sampler (intermediate):           {result.get('cpt_tinker_checkpoint_path', '?')}",
            flush=True,
        )
        print(f"  Final RL metadata:                    {result['metadata_path']}", flush=True)
    else:
        print(f"  Metadata (checkpoint + hyperparams): {result['metadata_path']}", flush=True)
    ap = result.get("api_dump_path")
    if ap:
        print(f"  Full Tinker API capture (JSON):       {ap}", flush=True)
    print(f"  tinker:// checkpoint URI (use for chat): {result['tinker_checkpoint_path']}", flush=True)
    print(f"  Registry:                             {SLACK_DIR / 'outputted_models.json'}", flush=True)
    print(
        "\n  Chat (OpenAI-compatible):  python chat.py\n"
        "  Or SamplingClient:           python run_slack_lora_inference.py --prompt \"…\"\n",
        flush=True,
    )


if __name__ == "__main__":
    main()
