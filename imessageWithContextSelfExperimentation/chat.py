#!/usr/bin/env python3
"""
Interactive “text yourself” chat — **Tinker (remote) inference only**.

Uses ``tinker_checkpoint_path`` from ``sft_tinker_metadata.json`` (written by ``sft_tinker.py``)
or ``TINKER_CHAT_MODEL_URI`` in the environment, via the OpenAI-compatible Tinker HTTP API.
Same message format as training (Thomas / iMessage, alternating user/assistant history).
GPT-OSS (Harmony) may leak ``<|channel|>final<|message|>`` fragments into streamed ``content``;
we strip those so history does not accumulate spurious ``final`` prefixes.

Requires ``TINKER_API_KEY`` in ``.env``. Local PEFT under ``./models/`` is not used in this
package (use the sibling ``imessage/`` project if you need on-device chat).

https://tinker-docs.thinkingmachines.ai/tinker/compatible-apis/openai/

Run:  python3 chat.py
"""

from __future__ import annotations

import difflib
import json
import os
import re
import sys
from pathlib import Path

from export_imessage import TEXTING_SYSTEM_PROMPT

_SCRIPT_DIR = Path(__file__).resolve().parent
SFT_TINKER_METADATA = _SCRIPT_DIR / "sft_tinker_metadata.json"
TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"

# Tinker API: short SMS ceiling + strong repetition penalties (long max_tokens → stutter loops).
CHAT_MAX_TOKENS_TINKER = 72
CHAT_TEMPERATURE_TINKER = 0.44
CHAT_TOP_P_TINKER = 0.74
CHAT_FREQUENCY_PENALTY_TINKER = 1.05
CHAT_PRESENCE_PENALTY_TINKER = 0.42

# If the new reply is almost the same as the prior assistant bubble, retry once with a nudge.
CHAT_REPEAT_SIMILARITY_THRESHOLD = 0.9

# Max non-system messages kept (each user or assistant line counts as one).
MAX_HISTORY_NON_SYSTEM = 10


def _trim_chat_history(messages: list[dict[str, str]], *, max_non_system: int) -> None:
    """
    Keep ``messages[0]`` (system) + the tail of the rest, at most ``max_non_system`` items.

    A naive ``messages[-N:]`` can start with an **assistant** turn (slice cuts inside a
    pair), which breaks chat order and looks like “forgot the last messages.” We drop
    leading assistant fragments so the first non-system message is always **user**.
    """
    if len(messages) <= 1:
        return
    system = messages[0]
    tail = messages[1:]
    if len(tail) <= max_non_system:
        while tail and tail[0]["role"] == "assistant":
            tail.pop(0)
        messages[:] = [system] + tail
        return
    tail = tail[-max_non_system:]
    while tail and tail[0]["role"] == "assistant":
        tail.pop(0)
    messages[:] = [system] + tail


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_tinker_sampler_uri() -> str | None:
    """``tinker://…/sampler_weights/…`` from env or ``sft_tinker_metadata.json`` (after Tinker SFT)."""
    env = os.environ.get("TINKER_CHAT_MODEL_URI", "").strip()
    if env.startswith("tinker://"):
        return env
    path = SFT_TINKER_METADATA
    if not path.is_file():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    uri = meta.get("tinker_checkpoint_path")
    if isinstance(uri, str) and uri.startswith("tinker://"):
        return uri
    return None


def _strip_gpt_oss_harmony_leakage(text: str) -> str:
    """Drop GPT-OSS Harmony wire tokens that leak into ``content`` from the OAI stream.

    Assistant turns are framed as ``<|channel|>final<|message|>…``; partial decoding can
    expose the literal ``final`` prefix. If that text is stored in history, the model
    stacks more ``final`` each turn.
    """
    s = text
    for lit in (
        "<|channel|>final<|message|>",
        "<|channel|>analysis<|message|>",
        "<|channel|>commentary<|message|>",
        "<|start|>assistant",
        "<|message|>",
        "<|return|>",
        "<|end|>",
        "<|call|>",
    ):
        s = s.replace(lit, "")
    # Bare ``final`` from split markers: peel repeated leaks, keep real words (finally, finals).
    while len(s) >= 5 and s.startswith("final"):
        rest = s[5:]
        if not rest:
            return ""
        if rest.startswith("final"):
            s = rest
            continue
        c0 = rest[0]
        if c0.isascii() and c0.isalpha() and c0.islower():
            break
        s = rest
    return s


def _scrub_assistant_harmony_leaks(messages: list[dict[str, str]]) -> None:
    for m in messages:
        if m.get("role") == "assistant" and isinstance(m.get("content"), str):
            m["content"] = _strip_gpt_oss_harmony_leakage(m["content"])


def _norm_reply(s: str) -> str:
    return " ".join(s.split()).lower()


def _prior_assistant_text(messages: list[dict[str, str]]) -> str | None:
    """Last assistant message before the current final user turn (``messages`` ends with user)."""
    for m in reversed(messages[:-1]):
        if m.get("role") == "assistant":
            c = m.get("content")
            return c if isinstance(c, str) else None
    return None


def _is_stuck_repeat(new_reply: str, prior_assistant: str | None) -> bool:
    if not prior_assistant or not new_reply:
        return False
    a, b = _norm_reply(new_reply), _norm_reply(prior_assistant)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) < 24 or len(b) < 24:
        return a == b
    return difflib.SequenceMatcher(None, a, b).ratio() >= CHAT_REPEAT_SIMILARITY_THRESHOLD


def _has_intra_reply_stutter(text: str) -> bool:
    """True if the same sentence (or near-duplicate) appears twice in a row in one reply."""
    t = text.strip()
    if not t:
        return False
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    for i in range(1, len(parts)):
        a, b = _norm_reply(parts[i]), _norm_reply(parts[i - 1])
        if a == b:
            return True
        if len(a) > 18 and len(b) > 18 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.93:
            return True
    return False


def _truncate_intra_reply_loops(text: str) -> tuple[str, bool]:
    """Drop duplicate / near-duplicate consecutive sentences; return (text, did_truncate)."""
    t = text.strip()
    if not t:
        return t, False
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return t, False
    out: list[str] = [parts[0]]
    for i in range(1, len(parts)):
        a, b = _norm_reply(parts[i]), _norm_reply(out[-1])
        if a == b:
            return (" ".join(out), True)
        if len(a) > 18 and len(b) > 18 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.93:
            return (" ".join(out), True)
        out.append(parts[i])
    joined = " ".join(out)
    return joined, joined != t


def chat_loop_tinker(
    *,
    model_path: str,
    max_tokens: int = CHAT_MAX_TOKENS_TINKER,
    temperature: float = CHAT_TEMPERATURE_TINKER,
    top_p: float = CHAT_TOP_P_TINKER,
    frequency_penalty: float = CHAT_FREQUENCY_PENALTY_TINKER,
    presence_penalty: float = CHAT_PRESENCE_PENALTY_TINKER,
) -> None:
    """Conversation loop: inference via Tinker OpenAI-compatible API."""
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "Set TINKER_API_KEY in .env (see https://tinker-console.thinkingmachines.ai/)"
        )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install: pip install openai") from exc

    client = OpenAI(
        base_url=TINKER_OAI_BASE_URL,
        api_key=os.environ["TINKER_API_KEY"],
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": TEXTING_SYSTEM_PROMPT},
    ]

    _log(
        "\n[chat] ─────────────────────────────────────────\n"
        "[chat] Backend: **Tinker** (OpenAI-compatible API)\n"
        f"[chat] Model: {model_path}\n"
        "[chat] You are Thomas — same framing as training (iMessage: gray = them, blue = you).\n"
        "[chat] Paste or type the incoming iMessage to reply to; history builds like Messages.\n"
        "[chat] /quit /exit or Ctrl+D to leave.\n"
        "[chat] ─────────────────────────────────────────\n"
    )

    def _stream_chat(
        msgs: list[dict[str, str]],
        *,
        temperature_override: float | None = None,
        frequency_penalty_override: float | None = None,
        presence_penalty_override: float | None = None,
    ) -> object:
        t = temperature if temperature_override is None else temperature_override
        fp = frequency_penalty if frequency_penalty_override is None else frequency_penalty_override
        pp = presence_penalty if presence_penalty_override is None else presence_penalty_override
        full: dict = {
            "model": model_path,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": t,
            "top_p": top_p,
            "frequency_penalty": fp,
            "presence_penalty": pp,
            "stream": True,
        }
        try:
            return client.chat.completions.create(**full)
        except Exception:
            try:
                no_fp = {k: v for k, v in full.items() if k not in ("frequency_penalty", "presence_penalty")}
                return client.chat.completions.create(**no_fp)
            except Exception:
                return client.chat.completions.create(
                    model=model_path,
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=t,
                    stream=True,
                )

    def _collect_reply(stream) -> str:
        parts: list[str] = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue
            piece = getattr(delta, "content", None) or ""
            if piece:
                parts.append(piece)
        return "".join(parts)

    while True:
        try:
            user = input("Incoming iMessage: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.", flush=True)
            break
        if not user:
            continue
        if user.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Bye.", flush=True)
            break

        messages.append({"role": "user", "content": user})
        prev_len = len(messages)
        _trim_chat_history(messages, max_non_system=MAX_HISTORY_NON_SYSTEM)
        if len(messages) < prev_len:
            _log(
                f"[chat] (trimmed older turns; keeping last {MAX_HISTORY_NON_SYSTEM} "
                "user/assistant lines + system)"
            )

        _scrub_assistant_harmony_leaks(messages)

        prior_assistant = _prior_assistant_text(messages)

        _log(f"[chat] Generating (max_tokens={max_tokens})…")
        print("Thomas: ", end="", flush=True)
        raw_reply = ""
        try:
            raw_reply = _collect_reply(_stream_chat(messages))
            reply = _strip_gpt_oss_harmony_leakage(raw_reply).strip()

            needs_quality_retry = _is_stuck_repeat(reply, prior_assistant) or _has_intra_reply_stutter(
                reply
            )
            if needs_quality_retry:
                _log(
                    "[chat] (reply repeated itself or echoed your last bubble — "
                    "retrying once with stronger anti-repeat settings…)"
                )
                nudge = {
                    "role": "user",
                    "content": (
                        "That reply repeats the same sentence or sounds like your last message. "
                        "Send one or two new short sentences only. No copy-paste loops, no stacked "
                        "“I think …” lines. If you already said your point, stop."
                    ),
                }
                raw_reply = _collect_reply(
                    _stream_chat(
                        [*messages, nudge],
                        temperature_override=min(temperature + 0.2, 0.9),
                        frequency_penalty_override=min(frequency_penalty + 0.35, 1.95),
                        presence_penalty_override=min(presence_penalty + 0.3, 1.2),
                    )
                )
                reply = _strip_gpt_oss_harmony_leakage(raw_reply).strip()

            reply, truncated = _truncate_intra_reply_loops(reply)
            if truncated:
                _log("[chat] (trimmed repeated sentences from model output)")
        except Exception as exc:
            print(f"\n[generation error: {exc}]", flush=True)
            messages.pop()
            continue
        print(reply, flush=True)
        if reply:
            messages.append({"role": "assistant", "content": reply})


def run_chat() -> None:
    from dotenv import load_dotenv

    load_dotenv(_SCRIPT_DIR / ".env")

    tinker_uri = load_tinker_sampler_uri()
    if not tinker_uri:
        print(
            "No Tinker sampler URI.\n"
            f"  • Run Train → SFT → Tinker (writes {SFT_TINKER_METADATA.name}), or\n"
            "  • Set TINKER_CHAT_MODEL_URI=tinker://… in .env\n"
            "This package uses Tinker for chat only (no local ./models LoRA).",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.environ.get("TINKER_API_KEY"):
        print(
            "Set TINKER_API_KEY in .env (https://tinker-console.thinkingmachines.ai/)",
            file=sys.stderr,
        )
        sys.exit(1)

    _log(f"[chat] Tinker model: {tinker_uri}")
    chat_loop_tinker(model_path=tinker_uri)


def main() -> None:
    run_chat()


if __name__ == "__main__":
    main()
