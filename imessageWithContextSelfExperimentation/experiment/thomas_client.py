"""Single-turn Tinker (OpenAI-compatible) completion for Thomas in evals."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root: imessageWithContextExperimentation/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from export_imessage import TEXTING_SYSTEM_PROMPT

from experiment.config import (
    THOMAS_FREQUENCY_PENALTY,
    THOMAS_MAX_TOKENS,
    THOMAS_PRESENCE_PENALTY,
    THOMAS_TEMPERATURE,
    THOMAS_TOP_P,
    TINKER_OAI_BASE_URL,
)

import chat as _chat_mod


def _thomas_sampling_temperature() -> float:
    raw = os.environ.get("MESSAGES_EXPERIMENT_THOMAS_TEMPERATURE")
    if raw is None or not str(raw).strip():
        return THOMAS_TEMPERATURE
    return float(raw)


def complete_thomas(
    *,
    client: object,
    tinker_model_uri: str,
    messages: list[dict[str, str]],
) -> str:
    """Non-streaming chat completion; returns cleaned assistant text."""
    temp = _thomas_sampling_temperature()
    full: dict = {
        "model": tinker_model_uri,
        "messages": messages,
        "max_tokens": THOMAS_MAX_TOKENS,
        "temperature": temp,
        "top_p": THOMAS_TOP_P,
        "frequency_penalty": THOMAS_FREQUENCY_PENALTY,
        "presence_penalty": THOMAS_PRESENCE_PENALTY,
        "stream": False,
    }
    try:
        resp = client.chat.completions.create(**full)
    except Exception:
        try:
            no_fp = {k: v for k, v in full.items() if k not in ("frequency_penalty", "presence_penalty")}
            resp = client.chat.completions.create(**no_fp)
        except Exception:
            resp = client.chat.completions.create(
                model=tinker_model_uri,
                messages=messages,
                max_tokens=THOMAS_MAX_TOKENS,
                temperature=temp,
                stream=False,
            )
    text = (resp.choices[0].message.content or "").strip()
    return _chat_mod._strip_gpt_oss_harmony_leakage(text).strip()


def open_tinker_client():
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY (e.g. in .env via dotenv in run_experiment).")
    from openai import OpenAI

    return OpenAI(base_url=TINKER_OAI_BASE_URL, api_key=os.environ["TINKER_API_KEY"])


def build_thomas_messages(transcript: list[tuple[str, str]]) -> list[dict[str, str]]:
    """transcript: [(\"incoming\" (friend), \"thomas\"), ...] in order."""
    out: list[dict[str, str]] = [{"role": "system", "content": TEXTING_SYSTEM_PROMPT}]
    for incoming, thomas in transcript:
        out.append({"role": "user", "content": incoming})
        out.append({"role": "assistant", "content": thomas})
    return out
