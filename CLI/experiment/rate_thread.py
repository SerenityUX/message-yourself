"""Rubric scoring via OpenRouter (default: Claude). Optimized for voice + anti-bland + anti-repeat."""

from __future__ import annotations

import json
import re

from experiment.config import default_openrouter_rater_model
from experiment.openrouter_client import get_openrouter_client

_RUBRIC_PROMPT = """You are an evaluator for iMessage-style threads between a friend (gray bubble) and Thomas (blue bubble).

**What we want from Thomas:** a **real person texting** — specific word choice, light personality, natural rhythm. **Penalize** generic assistant tone, therapy-speak templates, and repeating the same line or idea across turns (e.g. “I’m here for you” over and over). **Reward** concise, human, slightly imperfect SMS that still fits the relationship.

The format lines are:
- [Friend] = incoming iMessage
- [Thomas] = Thomas’s reply

Score each dimension from 1 (worst) to 5 (best):

- realistic: reads like real people texting (not scripted customer support)
- kind: warm / supportive where appropriate (without being saccharine)
- casual: natural texting voice, contractions, not formal essays
- concise: right-sized for SMS, not rambling
- repetition_issue: 1 = severe repeated phrases or copy-paste vibes across Thomas turns; 5 = varied wording, no obvious loops
- natural: holistic — personality, flow, would you believe a human sent this?

Respond with ONLY valid JSON, no markdown, in this exact shape:
{"realistic": <int>, "kind": <int>, "casual": <int>, "concise": <int>, "repetition_issue": <int>, "natural": <int>, "notes": "<one short sentence>"}
"""


def rate_transcript(formatted_thread: str) -> dict:
    """Call OpenRouter chat completion to produce scores + notes."""
    client = get_openrouter_client()
    model = default_openrouter_rater_model()
    user_content = _RUBRIC_PROMPT + "\n\n--- THREAD ---\n\n" + formatted_thread
    r = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise evaluator. Reply with only valid JSON, no markdown fences.",
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.15,
        max_tokens=1024,
    )
    text = (r.choices[0].message.content or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {"error": "no_json", "raw": text}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"error": "json_parse", "raw": text}
