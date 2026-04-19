"""Eval scenarios: fixed first friend line; OpenRouter continues the friend side."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    """Exact first incoming (gray bubble) line every run."""

    first_friend_message: str
    """Instructions for the friend model for turns after the opener."""

    friend_instruction: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="hike_invite",
        title="Hike invite",
        first_friend_message="yo u free sat morning for a hike?",
        friend_instruction=(
            "You are a close friend texting Thomas on iMessage. You already sent your opening line "
            "(shown in the thread). Continue naturally—trail ideas, timing, weather, no pressure. "
            "Do not roleplay as Thomas."
        ),
    ),
    Scenario(
        id="advice_confide",
        title="Confiding / advice",
        first_friend_message="hey can i vent for a sec",
        friend_instruction=(
            "You are a friend texting Thomas. You already opened; you are stressed about something "
            "realistic (work, sleep, etc.) and want his take and support. Stay casual SMS. "
            "Do not roleplay as Thomas."
        ),
    ),
    Scenario(
        id="cafe_after_event",
        title="Café after an event",
        first_friend_message="met someone at that event yesterday want to grab coffee",
        friend_instruction=(
            "You are texting Thomas about someone you met at a recent event and trying to sort "
            "coffee / timing. You already sent your opener; keep the thread natural. "
            "Do not roleplay as Thomas."
        ),
    ),
    Scenario(
        id="podcast_scheduling",
        title="Podcast scheduling",
        first_friend_message="when works for us to record next ep",
        friend_instruction=(
            "You are a friend coordinating a podcast with Thomas. You already sent your opener; "
            "keep proposing or narrowing times in a casual way. Do not roleplay as Thomas."
        ),
    ),
)
