#!/usr/bin/env python3
"""Run from this folder::

  cd CLI && python3 cli.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _setup() -> None:
    os.chdir(ROOT)
    r = str(ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def _preflight() -> tuple[list[str], list[str]]:
    """(export_blockers, train_blockers)"""
    export_b: list[str] = []
    train_b: list[str] = []

    if sys.platform != "darwin":
        export_b.append("iMessage export needs macOS.")
    else:
        db = Path.home() / "Library/Messages/chat.db"
        if not db.is_file():
            export_b.append("No ~/Library/Messages/chat.db (Full Disk Access for this app).")
        else:
            try:
                import sqlite3

                sqlite3.connect(f"file:{db}?mode=ro", uri=True).close()
            except Exception:
                export_b.append("Cannot open Messages DB (Full Disk Access?).")

    if sys.version_info < (3, 11):
        train_b.append(f"Python 3.11+ required for Tinker (have {sys.version.split()[0]}).")

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        train_b.append("Run: pip install -r requirements.txt")

    if not (os.environ.get("TINKER_API_KEY") or "").strip():
        train_b.append("TINKER_API_KEY unset (.env)")
    or_key = (os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not or_key:
        train_b.append("OPEN_ROUTER_API_KEY unset (.env) — needed for eval / sweeps")

    try:
        import openai  # noqa: F401
    except ImportError:
        train_b.append("pip install openai")

    if sys.version_info >= (3, 11):
        try:
            import tinker  # noqa: F401
        except ImportError:
            train_b.append("pip install -r requirements.txt (tinker)")

    return export_b, train_b


def _print_preflight(export_b: list[str], train_b: list[str]) -> None:
    if export_b:
        print("  Export:", flush=True)
        for x in export_b:
            print(f"    ! {x}", flush=True)
    if train_b:
        print("  Train / eval / sweeps:", flush=True)
        for x in train_b:
            print(f"    ! {x}", flush=True)
    if export_b or train_b:
        print("", flush=True)


def _require_sft_json() -> Path:
    p = ROOT / "sft_output.json"
    if not p.is_file():
        raise SystemExit("Need sft_output.json (option 1 or copy from elsewhere).")
    return p


def _require_py311() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("Need Python 3.11+.")


def _run_export() -> None:
    from export_imessage import run_export

    run_export()


def _run_quick_defaults() -> None:
    _require_py311()
    from experiment.run_experiment import launch_train_sweep_interactive

    launch_train_sweep_interactive(
        input_path=_require_sft_json(),
        lrs="3e-5",
        ranks="8",
        steps=200,
        parallel=1,
    )


def _run_quick_interactive() -> None:
    _require_py311()
    from experiment.run_experiment import launch_train_sweep_interactive

    raw = input("LR [3e-5]: ").strip() or "3e-5"
    rk = input("LoRA rank [8]: ").strip() or "8"
    st = input("Steps [200]: ").strip() or "200"
    launch_train_sweep_interactive(
        input_path=_require_sft_json(),
        lrs=raw,
        ranks=rk,
        steps=int(st),
        parallel=1,
    )


def _run_shotgun() -> None:
    _require_py311()
    from experiment.run_experiment import launch_train_sweep_interactive

    default_lrs = "1.5e-5,2e-5,2.5e-5,3e-5,3.5e-5,4e-5"
    raw = input(f"LRs [{default_lrs}]: ").strip() or default_lrs
    rk = input("Ranks [8]: ").strip() or "8"
    st = input("Steps [200]: ").strip() or "200"
    par = input("Parallel [1]: ").strip() or "1"
    launch_train_sweep_interactive(
        input_path=_require_sft_json(),
        lrs=raw,
        ranks=rk,
        steps=int(st),
        parallel=max(1, int(par)),
    )


def _run_agent() -> None:
    _require_py311()
    from experiment.agent_loop import launch_agent_sweep

    b = input("Max bands (0 = until Ctrl+C) [0]: ").strip() or "0"
    n = input("Max combos / band [9]: ").strip() or "9"
    st = input("Steps [200]: ").strip() or "200"
    launch_agent_sweep(
        input_path=_require_sft_json(),
        max_bands=int(b),
        max_combos_per_band=max(1, int(n)),
        steps=max(1, int(st)),
    )


def _run_eval() -> None:
    _require_py311()
    from model_registry import ensure_legacy_metadata, list_models

    ensure_legacy_metadata(ROOT)
    models = list_models(ROOT)
    print("  0  custom path → sft_tinker_metadata.json", flush=True)
    for i, m in enumerate(models, 1):
        print(f"  {i}  {m.get('label', '?')}", flush=True)
    default_c = "1" if models else "0"
    choice = input(f"Pick [{default_c}]: ").strip() or default_c
    meta: Path | None = None
    if choice == "0":
        line = input("Path: ").strip()
        meta = Path(line).expanduser().resolve()
    elif choice.isdigit() and int(choice) >= 1:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            meta = Path(str(models[idx]["metadata_path"]))
    if meta is None or not meta.is_file():
        raise SystemExit("Bad metadata path.")
    subprocess.run(
        [sys.executable, "-m", "experiment.run_experiment", "eval", "--metadata", str(meta)],
        cwd=str(ROOT),
        check=True,
    )


def _run_chat() -> None:
    from model_registry import ensure_legacy_metadata, list_models

    ensure_legacy_metadata(ROOT)
    models = list_models(ROOT)
    if not models:
        raise SystemExit("No saved models.")
    for i, m in enumerate(models, 1):
        print(f"  {i}  {m.get('label', '?')}", flush=True)
    line = input("# [1]: ").strip() or "1"
    if not line.isdigit():
        raise SystemExit("?")
    idx = int(line) - 1
    if idx < 0 or idx >= len(models):
        raise SystemExit("?")
    uri = models[idx].get("tinker_uri")
    if not isinstance(uri, str) or not uri.startswith("tinker://"):
        raise SystemExit("Bad registry entry.")
    os.environ["TINKER_CHAT_MODEL_URI"] = uri
    from chat import run_chat

    run_chat()


def main() -> None:
    _setup()
    ex, tr = _preflight()
    print("\nmessage-yourself CLI\n", flush=True)
    _print_preflight(ex, tr)

    from model_registry import ensure_legacy_metadata, list_models

    ensure_legacy_metadata(ROOT)
    has_models = bool(list_models(ROOT))

    opts = [
        "  1  Export iMessage",
        "  2  Export + SFT+eval (3e-5, r8, 200 steps)",
        "  3  SFT+eval (prompts LR/rank/steps)",
        "  4  Shotgun LR×rank grid",
        "  5  Agent sweep",
        "  6  Eval",
    ]
    if has_models:
        opts.append("  7  Chat")
    opts.append("  0  Exit")
    print("\n".join(opts) + "\n", flush=True)

    no_mac = sys.platform != "darwin"
    try:
        raw = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.", flush=True)
        return

    if raw in ("0", "q", "quit", "exit", ""):
        return
    if raw == "1":
        if no_mac:
            raise SystemExit("macOS only.")
        _run_export()
        return
    if raw == "2":
        if no_mac:
            raise SystemExit("macOS only.")
        _run_export()
        _run_quick_defaults()
        return
    if raw == "3":
        _run_quick_interactive()
        return
    if raw == "4":
        _run_shotgun()
        return
    if raw == "5":
        _run_agent()
        return
    if raw == "6":
        _run_eval()
        return
    if raw == "7":
        if not has_models:
            raise SystemExit("No models.")
        _run_chat()
        return
    print("?", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.", flush=True)
