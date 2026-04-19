#!/usr/bin/env python3
"""
Load a Slack CPT checkpoint saved by ``continued_pretrain.py`` and run sampling.

Reads ``cpt_tinker_metadata.json`` (``tinker_checkpoint_path``, ``base_model``).
Requires ``TINKER_API_KEY`` (``slack/.env`` is loaded automatically).

Example::

    cd slack && python3 run_slack_lora_inference.py --prompt "Hello from Slack CPT"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        "run_slack_lora_inference.py requires Python 3.11+ (Tinker SDK).\n"
        "  cd slack && rm -rf .venv && python3.12 -m venv .venv && source .venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
        f"  Current Python: {sys.version.split()[0]}"
    )

from dotenv import load_dotenv

import tinker
from tinker import types

_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR / ".env")
META_FILE = _SCRIPT_DIR / "cpt_tinker_metadata.json"


def _load_meta() -> dict:
    if not META_FILE.is_file():
        raise SystemExit(
            f"Missing {META_FILE.name}; run continued_pretrain.py first to train and save metadata."
        )
    return json.loads(META_FILE.read_text(encoding="utf-8"))


async def sample_async(*, prompt: str, max_tokens: int, temperature: float) -> str:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY (see https://tinker-docs.thinkingmachines.ai/)")

    meta = _load_meta()
    model_path = meta.get("tinker_checkpoint_path")
    if not model_path or not str(model_path).startswith("tinker://"):
        raise SystemExit(
            f"{META_FILE.name} has no valid tinker_checkpoint_path; re-run continued_pretrain.py."
        )

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(model_path=model_path)
    tokenizer = sampling_client.get_tokenizer()

    prompt_ids = tokenizer.encode(prompt)
    model_input = types.ModelInput.from_ints(prompt_ids)
    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )
    result = await sampling_client.sample_async(
        prompt=model_input,
        sampling_params=params,
        num_samples=1,
    )
    seq = result.sequences[0]
    return tokenizer.decode(seq.tokens)


def main() -> None:
    p = argparse.ArgumentParser(description="Sample from Slack Tinker LoRA checkpoint")
    p.add_argument("--prompt", default="Hi — quick check that inference works.", help="User prompt text")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    text = asyncio.run(
        sample_async(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    )
    print(text, end="" if text.endswith("\n") else "\n", flush=True)


if __name__ == "__main__":
    main()
