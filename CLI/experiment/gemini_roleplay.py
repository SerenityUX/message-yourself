"""Friend roleplay via OpenRouter (default: Claude). Signals end with ENDCONV."""

from __future__ import annotations

import os
import re

from experiment.config import default_openrouter_friend_model
from experiment.openrouter_client import get_openrouter_client

_END = re.compile(r"\bENDCONV\b", re.IGNORECASE)


def friend_model_id() -> str:
    return default_openrouter_friend_model()


def friend_reply_temperature() -> float:
    """Override with MESSAGES_EXPERIMENT_FRIEND_TEMPERATURE (e.g. 0 for deterministic eval)."""
    raw = os.environ.get("MESSAGES_EXPERIMENT_FRIEND_TEMPERATURE")
    if raw is None or not str(raw).strip():
        return 0.85
    return float(raw)


def parse_friend_raw(raw: str) -> tuple[str, str]:
    """
    Returns (incoming_text_for_thomas, end_mode).

    end_mode:
    - "none" — continue after Thomas replies
    - "stop_now" — friend ended with only ENDCONV; do not call Thomas
    - "stop_after_thomas" — strip ENDCONV from tail; Thomas replies once more, then stop
    """
    text = (raw or "").strip()
    if not text:
        return "", "stop_now"
    if _END.fullmatch(text.strip()):
        return "", "stop_now"
    if _END.search(text):
        cleaned = _END.sub("", text).strip()
        if not cleaned:
            return "", "stop_now"
        return cleaned, "stop_after_thomas"
    return text, "none"


class FriendChatSession:
    """Multi-turn friend (assistant) via OpenRouter chat completions."""

    def __init__(self, system_instruction: str, *, seed_assistant_text: str | None = None) -> None:
        self._client = get_openrouter_client()
        self._model = friend_model_id()
        rules = (
            "\n\nRules:\n"
            "- Write only the friend’s iMessage text (no role labels, no quotes around the whole message).\n"
            "- Keep each turn short like real SMS.\n"
            "- To end the conversation, send a message that is exactly: ENDCONV\n"
            "  (you may do this after a normal exchange, or alone to stop).\n"
            "- If you still want Thomas to reply one more time, put your normal text first, then a newline, "
            "then ENDCONV on its own line.\n"
        )
        self._messages: list[dict[str, str]] = [
            {"role": "system", "content": system_instruction + rules},
        ]
        if seed_assistant_text is not None and seed_assistant_text.strip():
            self._messages.append(
                {"role": "assistant", "content": seed_assistant_text.strip()},
            )

    def _complete(self, user_content: str, *, temperature: float | None = None) -> str:
        temp = friend_reply_temperature() if temperature is None else temperature
        self._messages.append({"role": "user", "content": user_content})
        r = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages,
            temperature=temp,
            max_tokens=512,
        )
        text = (r.choices[0].message.content or "").strip()
        self._messages.append({"role": "assistant", "content": text})
        return text

    def opener(self) -> str:
        return self._complete(
            "You text first. Send one short opening message as the friend (one or two sentences)."
        )

    def after_thomas(self, thomas_message: str) -> str:
        return self._complete(
            "Thomas just replied:\n\n"
            f"{thomas_message}\n\n"
            "Reply as the friend. To end the conversation, send exactly: ENDCONV"
        )


def start_friend_chat(
    system_instruction: str,
    *,
    seed_assistant_text: str | None = None,
) -> FriendChatSession:
    """If ``seed_assistant_text`` is set, the thread starts as if the friend already sent that opener."""
    return FriendChatSession(system_instruction, seed_assistant_text=seed_assistant_text)


def friend_opener(chat: FriendChatSession) -> str:
    return chat.opener()


def friend_continue(chat: FriendChatSession, thomas_message: str) -> str:
    return chat.after_thomas(thomas_message)
