"""Rubric scoring via OpenRouter (default: Claude through OpenRouter)."""

from __future__ import annotations

import json
import re

from experiment.config import default_openrouter_rater_model
from experiment.openrouter_client import get_openrouter_client

_RUBRIC_PROMPT = """You are an evaluator. Read the SMS thread below.

The format lines are:
- [Friend] = incoming iMessage (gray bubble)
- [Thomas] = Thomas’s reply (blue bubble)

Score each dimension from 1 (worst) to 5 (best), except repetition: use "repetition_issue" where 1 means highly repetitive/loopy and 5 means not repetitive.

Dimensions:
- realistic: reads like real people texting
- kind: warm / supportive where appropriate
- casual: natural texting voice, not formal
- concise: not rambling for the medium
- repetition_issue: 1 = severe repeated phrases; 5 = no repetition problem
- natural: overall human/natural flow

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
        temperature=0.2,
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
