#!/usr/bin/env python3
"""
Chat with a Tinker sampler checkpoint using the **OpenAI-compatible** HTTP API.

See https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/

Requires ``TINKER_API_KEY`` (``slack/.env``). Models are listed from ``outputted_models.json``
(written when you train via ``continued_pretrain`` / ``main.py``).

Responses **stream** by default. Defaults are tuned for calmer, shorter replies (CPT on Slack can
otherwise look like a chat log). Override with ``--max-tokens``, ``--temperature``, etc.

Examples::

    cd slack && source .venv/bin/activate
    python chat.py
    python chat.py --latest
    python chat.py --index 0
    python chat.py --no-stream
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR / ".env")

# Beta OpenAI-compatible inference (Thinking Machines)
TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# CPT on Slack can bias toward chat-log rambling and phrase loops; steer hard + sampling knobs.
DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question directly in a few sentences unless they "
    "ask for more. Stay on topic. Do not imitate Slack threads, filler enthusiasm, or chat-log style.\n"
    "Critical: do not repeat the same sentence, clause, or catchphrase. Never loop the same idea "
    "multiple times. If you notice repetition, stop immediately."
)


def _require_py311() -> None:
    if sys.version_info < (3, 11):
        raise SystemExit("chat.py requires Python 3.11+ (same venv as Tinker).")


def _pick_model_interactive(models: list[dict]) -> str | None:
    print("\nSaved models (from outputted_models.json):\n", flush=True)
    for i, m in enumerate(models):
        name = m.get("display_name") or m.get("name") or f"model_{i}"
        path = m.get("tinker_checkpoint_path") or ""
        print(f"  [{i}] {name}", flush=True)
        if path:
            print(f"      {path[:100]}{'…' if len(path) > 100 else ''}", flush=True)
    line = input("\nEnter index (or blank to cancel): ").strip()
    if not line:
        return None
    try:
        idx = int(line)
    except ValueError:
        print("Invalid index.", flush=True)
        return None
    if idx < 0 or idx >= len(models):
        print("Out of range.", flush=True)
        return None
    p = models[idx].get("tinker_checkpoint_path")
    if not p or not str(p).startswith("tinker://"):
        print("That record has no valid tinker_checkpoint_path.", flush=True)
        return None
    return str(p)


def run_interactive_chat(
    *,
    model_path: str | None = None,
    max_tokens: int = 180,
    temperature: float = 0.45,
    top_p: float = 0.88,
    frequency_penalty: float = 0.6,
    presence_penalty: float = 0.15,
    stream: bool = True,
    system_prompt: str | None = DEFAULT_SYSTEM,
) -> None:
    _require_py311()
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY in slack/.env (see https://tinker-console.thinkingmachines.ai/)")

    from outputted_models import list_models

    if model_path is None:
        models = list_models()
        if not models:
            raise SystemExit(
                f"No models in outputted_models.json — train once (main.py or continued_pretrain.py) first."
            )
        model_path = _pick_model_interactive(models)
        if model_path is None:
            print("Cancelled.", flush=True)
            return
    elif not str(model_path).startswith("tinker://"):
        raise SystemExit("model_path must be a tinker:// sampler URI")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install the OpenAI SDK: pip install openai") from exc

    client = OpenAI(
        base_url=TINKER_OAI_BASE_URL,
        api_key=os.environ["TINKER_API_KEY"],
    )

    print(
        f"\nChat (OpenAI-compatible Tinker). Model:\n  {model_path}\n"
        "Commands: quit | exit | :q — end session.\n",
        flush=True,
    )

    history: list[dict[str, str]] = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    def _create_kwargs(streaming: bool, *, with_penalties: bool = True) -> dict:
        kw: dict = {
            "model": model_path,
            "messages": history,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": streaming,
        }
        if with_penalties and frequency_penalty is not None:
            kw["frequency_penalty"] = frequency_penalty
        if with_penalties and presence_penalty is not None:
            kw["presence_penalty"] = presence_penalty
        return kw

    def _create_stream():
        try:
            return client.chat.completions.create(**_create_kwargs(True, with_penalties=True))
        except Exception:
            try:
                return client.chat.completions.create(**_create_kwargs(True, with_penalties=False))
            except Exception:
                return client.chat.completions.create(
                    model=model_path,
                    messages=history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )

    def _create_blocking():
        try:
            return client.chat.completions.create(**_create_kwargs(False, with_penalties=True))
        except Exception:
            try:
                return client.chat.completions.create(**_create_kwargs(False, with_penalties=False))
            except Exception:
                return client.chat.completions.create(
                    model=model_path,
                    messages=history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.", flush=True)
            break
        if not user:
            continue
        if user.lower() in ("quit", "exit", ":q"):
            break

        history.append({"role": "user", "content": user})
        text = ""
        try:
            if stream:
                print("Assistant: ", end="", flush=True)
                stream_resp = _create_stream()
                parts: list[str] = []
                for chunk in stream_resp:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue
                    piece = getattr(delta, "content", None) or ""
                    if piece:
                        print(piece, end="", flush=True)
                        parts.append(piece)
                print("\n", flush=True)
                text = "".join(parts).strip()
            else:
                resp = _create_blocking()
                text = (resp.choices[0].message.content or "").strip()
                print(f"Assistant: {text}\n", flush=True)
        except Exception as exc:
            print(f"\n[error] {exc}", flush=True)
            history.pop()
            continue

        history.append({"role": "assistant", "content": text})


def main() -> None:
    os.chdir(_SCRIPT_DIR)
    p = argparse.ArgumentParser(description="Chat with a Tinker checkpoint (OpenAI-compatible API)")
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="Full tinker://… sampler path (skips menu)",
    )
    p.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently recorded model in outputted_models.json",
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Select model by index in outputted_models.json (0 = oldest)",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=180,
        help="Cap completion length (lower = less rambling; default 180)",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.45,
        help="Sampling temperature (default 0.45; lower = steadier)",
    )
    p.add_argument("--top-p", type=float, default=0.88)
    p.add_argument(
        "--frequency-penalty",
        type=float,
        default=0.6,
        help="Reduce repetition (default 0.6; CPT models may need 0.7+ if still loopy)",
    )
    p.add_argument("--presence-penalty", type=float, default=0.15)
    p.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait for full response instead of streaming tokens",
    )
    p.add_argument(
        "--no-system",
        action="store_true",
        help="Disable the default anti-chat-log system prompt",
    )
    args = p.parse_args()

    model_path: str | None = args.model
    if args.latest or args.index is not None:
        from outputted_models import list_models

        models = list_models()
        if not models:
            raise SystemExit("outputted_models.json has no models.")
        if args.latest:
            m = models[-1]
        else:
            if args.index is None or args.index < 0 or args.index >= len(models):
                raise SystemExit(f"--index must be 0..{len(models) - 1}")
            m = models[args.index]
        model_path = m.get("tinker_checkpoint_path")
        if not model_path:
            raise SystemExit("Selected record has no tinker_checkpoint_path.")

    run_interactive_chat(
        model_path=model_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        stream=not args.no_stream,
        system_prompt=None if args.no_system else DEFAULT_SYSTEM,
    )


if __name__ == "__main__":
    main()
