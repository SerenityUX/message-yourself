"""OpenRouter (OpenAI-compatible API) — Claude, Gemini, etc. without vendor-specific SDKs."""

from __future__ import annotations

import os

from openai import OpenAI


def get_openrouter_api_key() -> str:
    k = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    if not k or not k.strip():
        raise SystemExit(
            "Set OPEN_ROUTER_API_KEY in .env for experiment roleplay + rubric (OpenRouter)."
        )
    return k.strip()


def openrouter_base_url() -> str:
    return (
        os.environ.get("OPENROUTER_BASE_URL")
        or os.environ.get("OPEN_ROUTER_BASE_URL")
        or "https://openrouter.ai/api/v1"
    ).strip()


def get_openrouter_client() -> OpenAI:
    """Client for chat completions. Optional HTTP-Referer for OpenRouter rankings."""
    headers: dict[str, str] = {}
    ref = (
        os.environ.get("OPENROUTER_HTTP_REFERER")
        or os.environ.get("OPEN_ROUTER_HTTP_REFERER")
        or os.environ.get("HTTP_REFERER")
    )
    if ref:
        headers["HTTP-Referer"] = ref
    title = os.environ.get("OPENROUTER_APP_NAME") or "message-yourself-experiment"
    headers["X-Title"] = title
    return OpenAI(
        base_url=openrouter_base_url(),
        api_key=get_openrouter_api_key(),
        default_headers=headers,
    )


def ensure_openrouter_key() -> None:
    """Validate env before a long train/eval run."""
    _ = get_openrouter_api_key()
