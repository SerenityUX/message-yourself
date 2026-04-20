"""Paths and defaults for experimentation."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_RESULTS = PACKAGE_ROOT / "experiment_results"

# OpenRouter model slugs (Claude via OpenRouter by default). Override OPEN_ROUTER_MODEL / OPEN_ROUTER_RATER_MODEL.
_DEFAULT_OPENROUTER_CLAUDE = "anthropic/claude-sonnet-4.5"


def default_openrouter_friend_model() -> str:
    return os.environ.get("OPEN_ROUTER_MODEL", _DEFAULT_OPENROUTER_CLAUDE).strip()


def default_openrouter_rater_model() -> str:
    return (
        os.environ.get("OPEN_ROUTER_RATER_MODEL")
        or os.environ.get("OPEN_ROUTER_MODEL")
        or _DEFAULT_OPENROUTER_CLAUDE
    ).strip()


def default_openrouter_agent_model() -> str:
    """Controller for LR×rank agent sweep (``experiment.agent_loop``)."""
    return (
        os.environ.get("OPEN_ROUTER_AGENT_MODEL")
        or os.environ.get("OPEN_ROUTER_CONTROLLER_MODEL")
        or default_openrouter_rater_model()
    ).strip()

# Default grids when running train sweep (edit or pass CLI).
DEFAULT_LEARNING_RATES = (1e-4, 1.5e-4, 2e-4)
# Tinker (e.g. gpt-oss-120b) requires LoRA rank to be a **power of 2** (8, 16, 32, … not 24).
DEFAULT_LORA_RANKS = (8, 16, 32)


def tinker_max_lora_rank() -> int:
    """Upper bound Tinker accepts for LoRA rank on the configured base model (gpt-oss-120b → 32)."""
    raw = os.environ.get("TINKER_MAX_LORA_RANK", "32").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 32

# Roleplay: max total messages (friend + Thomas). 5 rounds × 2 = 10.
MAX_ROLEPLAY_MESSAGES = 10

# Thomas generation (match chat.py-style brevity for eval).
THOMAS_MAX_TOKENS = 72
THOMAS_TEMPERATURE = 0.44
THOMAS_TOP_P = 0.74
THOMAS_FREQUENCY_PENALTY = 1.05
THOMAS_PRESENCE_PENALTY = 0.42

TINKER_OAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"
