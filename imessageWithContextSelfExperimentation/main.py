#!/usr/bin/env python3
"""
Interactive entrypoint: choose train vs chat, then CPT vs SFT when training.
SFT can run locally (PEFT) or on Tinker GPUs (``TINKER_API_KEY`` in ``.env``).

  cd imessageWithContextSelfExperimentation && python3 main.py

Train: export iMessage → ``cpt_out.txt`` + ``sft_output.json``, then LoRA (**CPT** or **SFT on Tinker**).

**SFT (this tree):** Claude **agent sweep** (default path), a **single Tinker** run, or a **fixed LR×rank grid**
(no controller). See ``README.md`` for env vars and Git rules.

Chat: **Tinker only** — sampler from ``sft_tinker_metadata.json`` or ``TINKER_CHAT_MODEL_URI``.

Requires: ``pip install -r requirements.txt``. Tinker + eval need Python **3.11+**, ``TINKER_API_KEY``,
and ``OPEN_ROUTER_API_KEY`` in this directory’s ``.env`` (see ``.env.example``).
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
        "\n[main] Message yourself (self-experimentation)\n"
        "──────────────────────────────────────────────\n"
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
            "  2 — SFT  (reply pairs → local adapter, Tinker, or agent sweep)\n",
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
                "\n[main] SFT on Tinker (this package)\n"
                "────────────────────────────────────\n"
                "  1 — **Claude agent sweep** — controller picks LR×rank each band; parallel jobs\n"
                "  2 — **Single Tinker run** — one checkpoint (``sft_tinker.py``)\n"
                "  3 — **Classic grid** — you enter lr / rank lists; train+eval per combo\n"
                "\n"
                "  (Local PEFT is not offered here; use sibling ``imessageWithContextExperimentation`` if needed.)\n",
                flush=True,
            )
            sft_backend = _pick(
                "How do you want to train?",
                hint="Type 1–3, or agent / tinker / grid",
                choices={
                    "1": "agent",
                    "2": "tinker",
                    "3": "grid",
                    "agent": "agent",
                    "tinker": "tinker",
                    "grid": "grid",
                },
            )
            if sft_backend == "tinker":
                if sys.version_info < (3, 11):
                    raise SystemExit(
                        "\n[main] Tinker SFT needs Python 3.11+ (your interpreter is "
                        f"{sys.version.split()[0]}: {sys.executable}).\n"
                        "  cd imessageWithContextSelfExperimentation && python3.12 -m venv .venv && source .venv/bin/activate\n"
                        "  pip install -r requirements.txt\n"
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
            elif sft_backend == "grid":
                if sys.version_info < (3, 11):
                    raise SystemExit(
                        "\n[main] Experiment needs Python 3.11+ (Tinker + deps). "
                        f"Current: {sys.version.split()[0]} ({sys.executable}).\n"
                    )
                from experiment.config import DEFAULT_LEARNING_RATES, DEFAULT_LORA_RANKS
                from experiment.run_experiment import launch_train_sweep_interactive

                print(
                    "\n[main] === Classic LR×LoRA grid (Tinker + eval per combo) ===\n"
                    "[main] Needs TINKER_API_KEY and OPEN_ROUTER_API_KEY in .env.\n",
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
                        "[main] Parallel combos (1 = serial) [1]: "
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
                    "\n[main] Grid sweep finished. See experiment_results/ …\n",
                    flush=True,
                )
            else:
                if sys.version_info < (3, 11):
                    raise SystemExit(
                        "\n[main] Agent sweep needs Python 3.11+ (Tinker + deps). "
                        f"Current: {sys.version.split()[0]} ({sys.executable}).\n"
                    )
                from experiment.agent_loop import launch_agent_sweep

                print(
                    "\n[main] === Claude agent sweep (LR×Rank) ===\n"
                    "[main] Needs TINKER_API_KEY and OPEN_ROUTER_API_KEY. "
                    "Optional OPEN_ROUTER_AGENT_MODEL.\n"
                    "[main] Logs: experiment_results/claude_agent_log.md\n"
                    "[main] State: experiment_results/agent_state.json\n",
                    flush=True,
                )
                try:
                    b_line = input(
                        "[main] Max bands (0 = run forever until Ctrl+C) [0]: "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    b_line = ""
                max_bands = int(b_line) if b_line else 0
                try:
                    pb_line = input(
                        "[main] Max (lr,rank) combos per band — hard cap; model picks 1..N [9]: "
                    ).strip()
                except (EOFError, KeyboardInterrupt):
                    pb_line = ""
                max_combos = int(pb_line) if pb_line else 9
                try:
                    st_line = input("[main] Tinker training steps per combo [200]: ").strip()
                except (EOFError, KeyboardInterrupt):
                    st_line = ""
                steps = int(st_line) if st_line else 200
                sft_json = Path(__file__).resolve().parent / "sft_output.json"
                launch_agent_sweep(
                    input_path=sft_json,
                    max_bands=max_bands,
                    max_combos_per_band=max(1, max_combos),
                    steps=max(1, steps),
                )
                print(
                    "\n[main] Agent sweep finished. See experiment_results/ and claude_agent_log.md.\n",
                    flush=True,
                )

        skip_follow_chat = stage == "sft" and sft_backend in ("agent", "grid")
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
