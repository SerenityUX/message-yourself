#!/usr/bin/env python3
"""
Interactive entrypoint: choose train vs chat, then CPT vs SFT when training.

  python3 main.py

Train: export iMessage → ``cpt_out.txt`` + ``sft_output.json``, then one LoRA stage.
Chat: inference only (adapter under ``models/``; prefers ``messages-lora-sft``).

Requires: ``pip install -r requirements.txt`` for train + chat.
"""

from __future__ import annotations

import sys


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
            "  2 — SFT  (reply pairs → models/messages-lora-sft)\n",
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

        if stage == "cpt":
            from continued_pretrain import run_cpt

            print("\n[main] === CPT → models/messages-lora-cpt ===\n", flush=True)
            run_cpt()
        else:
            from sft import run_sft

            print("\n[main] === SFT → models/messages-lora-sft ===\n", flush=True)
            run_sft()

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
