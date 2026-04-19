#!/usr/bin/env python3
"""
Interactive entrypoint: choose train vs chat, then CPT vs SFT when training.
SFT can run locally (PEFT) or on Tinker GPUs (``TINKER_API_KEY`` in ``imessage/.env``).

  python3 main.py

Train: export iMessage → ``cpt_out.txt`` + ``sft_output.json``, then one LoRA stage.
Chat: inference only (adapter under ``models/``; prefers ``messages-lora-sft``).

Requires: ``pip install -r requirements.txt`` for train + chat. Tinker SFT needs Python 3.11+.
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

        if stage == "cpt":
            from continued_pretrain import run_cpt

            print("\n[main] === CPT → models/messages-lora-cpt ===\n", flush=True)
            run_cpt()
        else:
            print(
                "\n[main] SFT backend\n"
                "─────────────────\n"
                "  1 — Local  (PEFT on this machine → models/messages-lora-sft)\n"
                "  2 — Tinker (remote LoRA on Qwen3-4B-Instruct-2507; needs TINKER_API_KEY + Python 3.11+)\n",
                flush=True,
            )
            sft_backend = _pick(
                "Local or Tinker?",
                hint="Type 1 or 2, or local / tinker",
                choices={
                    "1": "local",
                    "2": "tinker",
                    "local": "local",
                    "tinker": "tinker",
                },
            )
            if sft_backend == "local":
                from sft import run_sft

                print("\n[main] === SFT (local) → models/messages-lora-sft ===\n", flush=True)
                run_sft()
            else:
                if sys.version_info < (3, 11):
                    raise SystemExit(
                        "\n[main] Tinker SFT needs Python 3.11+ (your interpreter is "
                        f"{sys.version.split()[0]}: {sys.executable}).\n"
                        "  cd imessage && rm -rf .venv && python3.12 -m venv .venv && source .venv/bin/activate\n"
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
