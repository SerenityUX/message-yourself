#!/usr/bin/env python3
"""
Interactive entrypoint: choose train vs chat, then CPT vs SFT when training.
SFT can run locally (PEFT) or on Tinker GPUs (``TINKER_API_KEY`` in this directory’s ``.env``).

  python3 main.py

Train: export iMessage → ``cpt_out.txt`` + ``sft_output.json``, then one LoRA stage (or an **LR×LoRA experiment** sweep).
Chat: **Tinker only** — sampler from ``sft_tinker_metadata.json`` or ``TINKER_CHAT_MODEL_URI`` (no local PEFT chat).

**SFT option 3 (experiment)** is not the same as option 2: option 2 runs **one** Tinker train; option 3 runs **several** (lr×rank) and **OpenRouter rubric eval** (default Claude) into ``experiment_results/``.

Requires: ``pip install -r requirements.txt`` for train + chat. Tinker SFT + experiment need Python 3.11+ and ``OPEN_ROUTER_API_KEY`` (OpenRouter; Claude by default for rating) for eval.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _pick(prompt: str, *, hint: str, choices: dict[str, str]) -> str:
    """Prompt until input matches a key in ``choices`` (case-insensitive)."""
    while True:
        try:
            raw = input(f"{prompt}\n  ({hint})\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] Cancelled.", flush=True)
            sys.exit(0)
        if not raw:
            continue
        key = raw.lower()
        if key in choices:
            return choices[key]
        print(f"  Not recognized: {raw!r}. {hint}\n", flush=True)


def main() -> None:
    print(
        "\n[main] Message yourself\n"
        "────────────────────────\n"
        "  1 — Train   (export Messages, then LoRA: CPT or SFT)\n"
        "  2 — Chat    (inference only)\n",
        flush=True,
    )
    mode = _pick(
        "Train or chat?",
        hint="Type 1 or 2, or the word train / chat",
        choices={
            "1": "train",
            "2": "chat",
            "train": "train",
            "chat": "chat",
        },
    )

    if mode == "train":
        print(
            "\n[main] Which LoRA?\n"
            "──────────────────\n"
            "  1 — CPT  (text corpus → models/messages-lora-cpt)\n"
            "  2 — SFT  (reply pairs → local adapter or Tinker LoRA)\n",
            flush=True,
        )
        stage = _pick(
            "CPT or SFT?",
            hint="Type 1 or 2, or the word cpt / sft",
            choices={"1": "cpt", "2": "sft", "cpt": "cpt", "sft": "sft"},
        )

        from export_imessage import run_export

        print("\n[main] === Export → cpt_out.txt + sft_output.json ===\n", flush=True)
        run_export()

        sft_backend: str | None = None
        if stage == "cpt":
            from continued_pretrain import run_cpt

            print("\n[main] === CPT → models/messages-lora-cpt ===\n", flush=True)
            run_cpt()
        else:
            print(
                "\n[main] SFT backend\n"
                "─────────────────\n"
                "  1 — Local  (PEFT on this machine → models/messages-lora-sft)\n"
                "  2 — Tinker (one run: remote LoRA on openai/gpt-oss-120b)\n"
                "  3 — Experiment (several lr×LoRA Tinker trains + scenario eval + rubric → experiment_results/)\n",
                flush=True,
            )
            sft_backend = _pick(
                "Local, single Tinker, or experiment sweep?",
                hint="Type 1, 2, or 3 — or local / tinker / experiment",
                choices={
                    "1": "local",
                    "2": "tinker",
                    "3": "experiment",
                    "local": "local",
                    "tinker": "tinker",
                    "experiment": "experiment",
                },
            )
            if sft_backend == "local":
                from sft import run_sft

                print("\n[main] === SFT (local) → models/messages-lora-sft ===\n", flush=True)
                run_sft()
            elif sft_backend == "tinker":
                if sys.version_info < (3, 11):
                    raise SystemExit(
                        "\n[main] Tinker SFT needs Python 3.11+ (your interpreter is "
                        f"{sys.version.split()[0]}: {sys.executable}).\n"
                        "  cd imessageWithContextExperimentation && rm -rf .venv && python3.12 -m venv .venv && source .venv/bin/activate\n"
                        "  pip install -r requirements.txt\n"
                        "Then run main.py again and choose Train → SFT → Tinker.\n"
                    )
                try:
                    from sft_tinker import run_sft_tinker
                except ImportError as exc:
                    raise SystemExit(
                        "Tinker SFT could not import dependencies. With Python 3.11+:\n"
                        "  pip install -r requirements.txt\n"
                        f"  ({exc})"
                    ) from exc
                print(
                    "\n[main] === SFT (Tinker) → checkpoint in sft_tinker_metadata.json ===\n",
                    flush=True,
                )
                run_sft_tinker()
            else:
                if sys.version_info < (3, 11):
                    raise SystemExit(
                        "\n[main] Experiment needs Python 3.11+ (Tinker + deps). "
                        f"Current: {sys.version.split()[0]} ({sys.executable}).\n"
                    )
                from experiment.config import DEFAULT_LEARNING_RATES, DEFAULT_LORA_RANKS
                from experiment.run_experiment import launch_train_sweep_interactive

                print(
                    "\n[main] === LR×LoRA experiment (Tinker train + fixed opener + friend LLM + rubric per combo) ===\n"
                    "[main] Needs TINKER_API_KEY and OPEN_ROUTER_API_KEY (OpenRouter) in .env.\n",
                    flush=True,
                )
                default_lrs = ",".join(str(x) for x in DEFAULT_LEARNING_RATES)
                default_ranks = ",".join(str(x) for x in DEFAULT_LORA_RANKS)
                try:
                    lr_line = input(
                        f"[main] Learning rates (comma-separated) [default: {default_lrs}]: "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    lr_line = ""
                lrs = lr_line if lr_line else default_lrs
                try:
                    rk_line = input(
                        f"[main] LoRA ranks (comma-separated) [default: {default_ranks}]: "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    rk_line = ""
                ranks = rk_line if rk_line else default_ranks
                try:
                    st_line = input("[main] Training steps per combo [200]: ").strip()
                except (EOFError, KeyboardInterrupt):
                    st_line = ""
                steps = int(st_line) if st_line else 200
                try:
                    par_line = input(
                        "[main] Parallel combos (1 = one at a time; higher uses threads + separate output files) [1]: "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    par_line = ""
                parallel = int(par_line) if par_line else 1
                if parallel < 1:
                    parallel = 1
                sft_json = Path(__file__).resolve().parent / "sft_output.json"
                launch_train_sweep_interactive(
                    input_path=sft_json,
                    lrs=lrs,
                    ranks=ranks,
                    steps=steps,
                    parallel=parallel,
                )
                print(
                    "\n[main] Experiment finished. See experiment_results/<lr>_<r>/ "
                    "for manifest.json, scenario JSON, and rubric scores.\n"
                    "[main] Chat uses one checkpoint at a time: copy a combo’s "
                    "sft_tinker_metadata.json or set TINKER_CHAT_MODEL_URI.\n",
                    flush=True,
                )

        skip_follow_chat = stage == "sft" and sft_backend == "experiment"
        if not skip_follow_chat:
            try:
                follow = input("\n[main] Open chat after training? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                follow = "n"
            if follow in ("", "y", "yes"):
                print("\n[main] === Chat ===\n", flush=True)
                from chat import run_chat

                run_chat()
            else:
                print("\n[main] Done (no chat). Run `python3 main.py` → option 2 when ready.\n", flush=True)
        return

    print("\n[main] === Chat ===\n", flush=True)
    from chat import run_chat

    run_chat()
    print("\n[main] Session finished.\n", flush=True)


if __name__ == "__main__":
    main()
