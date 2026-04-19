"""
Normalize Slack mrkdwn from search/export JSON and write CPT text (one message per line).

Filters are **stricter than iMessage** (min words/chars, alphanumeric mass, spam/patterns) to
reduce reactions, acknowledgements, and link-only noise from public channels.

Loads ``slack/.env`` (``python-dotenv``). Optional paths:

- ``SLACK_CPT_INPUT`` — JSON export (default: ``my_slack_messages.json`` next to this script)
- ``SLACK_CPT_OUTPUT`` — CPT text file (default: ``cpt_out.txt``)

Run from repo root::

    python slack/prepare_slack_cpt.py

or from ``slack/``::

    python prepare_slack_cpt.py
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent

# Stricter than iMessage for public Slack (short reactions, link-only, etc.).
MIN_MESSAGE_WORDS = 3
MIN_MESSAGE_CHARS = 12
MIN_ALNUM_CHARS = 15  # letters+digits; drops emoji / mention-only lines

_TAPBACK_PREFIXES = (
    "liked ",
    "loved ",
    "disliked ",
    "emphasized ",
    "questioned ",
    "laughed at ",
)

_EFFECT_RE = re.compile(
    r"^\s*(?:Edited to|Edited from|This message responded to with)\b",
    re.IGNORECASE,
)

# Keyboard mash / accidental repeats
_REPEATED_CHAR_SPAM = re.compile(r"(.)\1{7,}", re.DOTALL)

# Standalone filler / acknowledgement only (after normalization; emojis are ``:shortcode:`` in Slack)
_LOW_INFO_ONLY = re.compile(
    r"^(?:\+1|same|this|yep|yeah|yes|no|nah|ok|okay|k|kk|ty|tysm|thanks|thank you|"
    r"lol|haha|heh|nice|cool|mhm|hm+|👍|✅|❤️)+[\s!.]*$",
    re.IGNORECASE,
)

# <@USERID|name> or <@USERID>
_RE_USER = re.compile(r"<@([A-Z0-9]+)(?:\|([^>]+))?>")
# <#CHANNELID|name>
_RE_CHANNEL = re.compile(r"<#([A-Z0-9]+)(?:\|([^>]+))?>")
_RE_MAILTO = re.compile(r"<mailto:([^>|]+)(?:\|([^>]+))?>")
_RE_LINK = re.compile(r"<(https?://[^|>]+)(?:\|([^>]+))?>")
# Bare URLs in message bodies (Slack does not always wrap these in <>).
_BARE_HTTP_URL_RE = re.compile(r"https?://[^\s<>]+", re.IGNORECASE)

# Slack emoji in mrkdwn: ``:smile:``, ``:+1:``, ``:skin-tone-2:``, custom ``:blob-wave:``, etc.
_SLACK_EMOJI_SHORTCODE_RE = re.compile(
    r":(?:\+1|[a-zA-Z0-9_][a-zA-Z0-9_+\-]{0,62}):",
    re.IGNORECASE,
)


def _is_tapback_or_reaction_summary(text: str) -> bool:
    t = text.strip().lower()
    return any(t.startswith(p) for p in _TAPBACK_PREFIXES)


def _is_noise_body(text: str) -> bool:
    if _EFFECT_RE.match(text):
        return True
    stripped = text.strip()
    if not re.search(r"[A-Za-z0-9]", stripped) and len(stripped) <= 20:
        return True
    if _REPEATED_CHAR_SPAM.search(stripped):
        return True
    if _LOW_INFO_ONLY.match(stripped):
        return True
    return False


def _alnum_count(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


def _collapse_blank_lines(text: str) -> str:
    """Drop empty / whitespace-only lines inside multi-line messages (one CPT record per message)."""

    rows = [r.strip() for r in text.splitlines()]
    rows = [r for r in rows if r]
    return "\n".join(rows)


def _is_single_token_spam(text: str) -> bool:
    words = [w for w in text.split() if w.strip()]
    if len(words) <= 3:
        return False
    return len({w.casefold() for w in words}) == 1


def _is_substantive_message(text: str) -> bool:
    if _is_tapback_or_reaction_summary(text):
        return False
    words = [w for w in text.split() if w.strip()]
    if len(words) < MIN_MESSAGE_WORDS:
        return False
    if len(text.strip()) < MIN_MESSAGE_CHARS:
        return False
    if len(words) == MIN_MESSAGE_WORDS and all(len(w) <= 3 for w in words):
        return False
    if _alnum_count(text) < MIN_ALNUM_CHARS:
        return False
    if _is_single_token_spam(text):
        return False
    return True


def normalize_slack_text(text: str) -> str:
    """Turn Slack mrkdwn into plain, readable text for CPT.

    HTTP(S) links are removed: Slack ``<url|label>`` entities and bare ``https://`` / ``http://``
    URLs in the body. Emoji are removed as Slack shortcodes (``:name:``, including ``:+1:``).
    User/channel mentions and mailto display text are unchanged.
    """

    if not text:
        return ""

    def sub_user(m: re.Match[str]) -> str:
        label = m.group(2)
        if label:
            return "@" + label.strip()
        return "@user"

    def sub_channel(m: re.Match[str]) -> str:
        label = m.group(2)
        if label:
            return "#" + label.strip()
        return "#channel"

    def sub_mailto(m: re.Match[str]) -> str:
        if m.group(2):
            return m.group(2).strip()
        return m.group(1).strip()

    def sub_link(_m: re.Match[str]) -> str:
        return ""

    def sub_bare_url(_m: re.Match[str]) -> str:
        return ""

    out = _RE_USER.sub(sub_user, text)
    out = _RE_CHANNEL.sub(sub_channel, out)
    out = _RE_MAILTO.sub(sub_mailto, out)
    out = _RE_LINK.sub(sub_link, out)
    out = html.unescape(out)
    out = _SLACK_EMOJI_SHORTCODE_RE.sub(" ", out)
    out = _BARE_HTTP_URL_RE.sub(sub_bare_url, out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


_SLACK_STRING_KEYS = frozenset({"text", "fallback"})


def normalize_slack_text_fields(obj: Any) -> None:
    """Recursively normalize Slack mrkdwn in common string fields (body, unfurls, etc.)."""

    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _SLACK_STRING_KEYS and isinstance(v, str):
                obj[k] = normalize_slack_text(v)
            else:
                normalize_slack_text_fields(v)
    elif isinstance(obj, list):
        for item in obj:
            normalize_slack_text_fields(item)


def messages_to_cpt_lines(messages: list[dict], *, substantive_only: bool = True) -> list[str]:
    """Extract normalized CPT lines, oldest first (by Slack ``ts``)."""

    def ts_key(m: dict) -> float:
        raw = m.get("ts") or "0"
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    sorted_msgs = sorted(messages, key=ts_key)
    lines: list[str] = []
    for m in sorted_msgs:
        raw = m.get("text")
        if not isinstance(raw, str) or not raw.strip():
            continue
        text = normalize_slack_text(raw)
        text = _collapse_blank_lines(text)
        if not text.strip():
            continue
        if substantive_only and (
            _is_noise_body(text) or not _is_substantive_message(text)
        ):
            continue
        lines.append(text)
    return lines


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    p = Path(str(raw).strip())
    return p if p.is_absolute() else (_SCRIPT_DIR / p)


def write_cpt(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for line in lines:
            if not line.strip():
                continue
            out = _collapse_blank_lines(line)
            if not out.strip():
                continue
            f.write(out + "\n")


def main() -> None:
    load_dotenv(_SCRIPT_DIR / ".env")
    json_path = _env_path("SLACK_CPT_INPUT", _SCRIPT_DIR / "my_slack_messages.json")
    out_path = _env_path("SLACK_CPT_OUTPUT", _SCRIPT_DIR / "cpt_out.txt")

    if not json_path.is_file():
        print(f"Missing {json_path}", file=sys.stderr, flush=True)
        sys.exit(1)

    with json_path.open(encoding="utf-8") as f:
        messages = json.load(f)
    if not isinstance(messages, list):
        print("Expected a JSON array of messages.", file=sys.stderr, flush=True)
        sys.exit(1)

    for m in messages:
        normalize_slack_text_fields(m)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)

    lines = messages_to_cpt_lines(messages)
    write_cpt(lines, out_path)
    print(
        f"Updated {json_path} with normalized text; "
        f"wrote {len(lines)} CPT line(s) to {out_path} "
        f"(substantive filter on, {len(messages)} record(s)).",
        flush=True,
    )


if __name__ == "__main__":
    main()
