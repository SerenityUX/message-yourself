#!/usr/bin/env python3
"""
Corpus-only RL on ``cpt_out.txt`` (training-set oracle reward).

For each step we sample random text chunks, split into **prefix | suffix** from the same chunk,
draw multiple completions from the current LoRA policy, and assign **reward** = string similarity
between each completion and the **true suffix** (``difflib.SequenceMatcher`` ratio). Advantages are
**group-centered** (same idea as ``tinker_cookbook/recipes/rl_loop.py``). The policy update uses
Tinker's ``importance_sampling`` loss (GRPO-style signal on sampled tokens).

This is **not** CPT: you optimize expected reward under rollouts, not plain next-token CE on the file.

Requires Python **3.11+**, ``TINKER_API_KEY``, and the same venv as ``continued_pretrain.py``::

    cd slack && source .venv/bin/activate
    python rl_corpus.py --help
    python rl_corpus.py --batches 30 --batch-size 4 --group-size 8

**Warm-start RL after CPT:** pass ``init_from_tinker_path`` with the CPT **training** URI
(``tinker://…/weights/{name}`` from ``save_state``). **Do not** use ``…/sampler_weights/…`` (inference
only); the API returns 400 for ``load_state``. ``continued_pretrain`` / ``main.py`` now save both.

Optional: ``--resume-state-path`` for full optimizer resume (different use case).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import torch
import tinker
from tinker import types
from tinker.types.tensor_data import TensorData

from continued_pretrain import DEFAULT_CPT_FILE, _json_safe, load_text_chunks
from outputted_models import OUTPUTTED_MODELS_JSON, sync_from_cpt_metadata

logger = logging.getLogger(__name__)

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RL_METADATA = _SCRIPT_DIR / "rl_tinker_metadata.json"
DEFAULT_API_DUMP = _SCRIPT_DIR / "rl_api_full_response.json"


def _reward(generated: str, reference_suffix: str, *, max_ref_chars: int = 2000) -> float:
    ref = reference_suffix.strip()[:max_ref_chars]
    if not ref:
        return 0.0
    gen = generated.strip()
    if not gen:
        return 0.0
    return float(SequenceMatcher(None, gen, ref).ratio())


def _pick_prefix_suffix(
    tokenizer,
    chunk: str,
    *,
    rng: random.Random,
    min_prefix_tokens: int,
    min_suffix_tokens: int,
) -> tuple[list[int], str] | None:
    """Return (prefix_token_ids, reference_suffix_text)."""
    ids = tokenizer.encode(chunk)
    lo = min_prefix_tokens
    hi = len(ids) - min_suffix_tokens
    if hi <= lo + 4:
        return None
    split = rng.randint(lo, hi - 1)
    prefix_ids = ids[:split]
    suffix_ids = ids[split:]
    suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=False)
    if len(suffix_text.strip()) < 8:
        return None
    return prefix_ids, suffix_text


def run_rl(
    *,
    input_path: Path,
    base_model: str,
    lora_rank: int,
    batches: int,
    batch_size: int,
    group_size: int,
    learning_rate: float,
    max_tokens: int,
    temperature: float,
    min_prefix_tokens: int,
    min_suffix_tokens: int,
    checkpoint_name: str,
    seed: int,
    chunk_chars: int,
    stride_chars: int,
    resume_state_path: str | None,
    init_from_tinker_path: str | None = None,
) -> dict:
    import os

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Set TINKER_API_KEY (slack/.env or environment).")

    rng = random.Random(seed)
    chunks = load_text_chunks(input_path, chunk_chars=chunk_chars, stride_chars=stride_chars)
    if not chunks:
        raise SystemExit("No chunks from corpus; check cpt_out.txt and chunk/stride settings.")

    service_client = tinker.ServiceClient()

    if resume_state_path and init_from_tinker_path:
        raise SystemExit("Use only one of resume_state_path or init_from_tinker_path.")
    if resume_state_path:
        training_client = service_client.create_training_client_from_state_with_optimizer(
            resume_state_path
        )
        logger.info("Resumed training from state %s", resume_state_path)
    elif init_from_tinker_path:
        training_client = service_client.create_training_client_from_state(init_from_tinker_path)
        logger.info(
            "Loaded LoRA for RL from CPT checkpoint %s (optimizer fresh; warm-start)",
            init_from_tinker_path,
        )
    else:
        training_client = service_client.create_lora_training_client(
            base_model=base_model,
            rank=lora_rank,
        )

    tokenizer = training_client.get_tokenizer()
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=[],
    )
    adam_params = types.AdamParams(
        learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    api_capture: dict = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "rl_corpus",
        "base_model": base_model,
        "corpus": str(input_path.resolve()),
        "init_from_tinker_path": init_from_tinker_path,
    }

    for batch_idx in range(batches):
        t0 = __import__("time").time()
        datums_D: list[types.Datum] = []
        rewards_batch: list[float] = []
        futures_P = []
        prompts_P: list[types.ModelInput] = []
        refs_P: list[str] = []

        # Fresh policy weights for this batch (on-policy RL; matches rl_loop.py pattern).
        sampling_client = training_client.save_weights_and_get_sampling_client()

        for _ in range(batch_size):
            chunk = rng.choice(chunks)
            ps = _pick_prefix_suffix(
                tokenizer,
                chunk,
                rng=rng,
                min_prefix_tokens=min_prefix_tokens,
                min_suffix_tokens=min_suffix_tokens,
            )
            if ps is None:
                continue
            prefix_ids, ref_suffix = ps
            prompt = types.ModelInput.from_ints(prefix_ids)
            fut = sampling_client.sample(
                prompt=prompt,
                num_samples=group_size,
                sampling_params=sampling_params,
            )
            futures_P.append(fut)
            prompts_P.append(prompt)
            refs_P.append(ref_suffix)

        if not futures_P:
            logger.warning("Batch %s: no valid prefixes; skip", batch_idx)
            continue

        for fut, prompt, ref_suffix in zip(futures_P, prompts_P, refs_P):
            sample_result = fut.result()
            rewards_G: list[float] = []
            sampled_tokens_G_T: list[list[int]] = []
            logprobs_G_T: list[list[float]] = []

            for sequence in sample_result.sequences:
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs
                if sampled_logprobs is None:
                    continue
                sampled_tokens_G_T.append(sampled_tokens)
                logprobs_G_T.append(sampled_logprobs)
                text = tokenizer.decode(sampled_tokens, skip_special_tokens=False)
                rewards_G.append(_reward(text, ref_suffix))

            if len(rewards_G) != group_size:
                continue

            mean_r = sum(rewards_G) / len(rewards_G)
            advantages_G = [r - mean_r for r in rewards_G]
            rewards_batch.append(mean_r)

            if all(a == 0.0 for a in advantages_G):
                continue

            for sampled_tokens, logprobs, advantage in zip(
                sampled_tokens_G_T, logprobs_G_T, advantages_G
            ):
                ob_len = prompt.length - 1
                model_input = prompt.append(types.EncodedTextChunk(tokens=sampled_tokens[:-1]))
                target_tokens = [0] * ob_len + sampled_tokens
                padded_logprobs = [0.0] * ob_len + logprobs
                padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
                if not (
                    model_input.length
                    == len(target_tokens)
                    == len(padded_logprobs)
                    == len(padded_advantages)
                ):
                    continue
                datums_D.append(
                    types.Datum(
                        model_input=model_input,
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                        },
                    )
                )

        if not datums_D:
            logger.warning("Batch %s: no datums after sampling; skip", batch_idx)
            continue

        fwd_bwd_future = training_client.forward_backward(datums_D, loss_fn="importance_sampling")
        optim_future = training_client.optim_step(adam_params)
        _ = fwd_bwd_future.result()
        _ = optim_future.result()

        mean_reward = sum(rewards_batch) / max(len(rewards_batch), 1)
        logger.info(
            "batch %s/%s  mean_reward=%.4f  datums=%s  sec=%.1f",
            batch_idx + 1,
            batches,
            mean_reward,
            len(datums_D),
            __import__("time").time() - t0,
        )

    # Save sampler weights (sync API returns a Future)
    logger.info("Saving sampler weights as %r…", checkpoint_name)
    save_future = training_client.save_weights_for_sampler(checkpoint_name)
    save_result = save_future.result()
    tinker_path = getattr(save_result, "path", None) or getattr(save_result, "checkpoint_path", None)
    if not tinker_path:
        raise SystemExit("save_weights_for_sampler returned no path")
    tinker_path = str(tinker_path)
    api_capture["save"] = _json_safe(save_result)
    api_capture["tinker_checkpoint_path"] = tinker_path

    try:
        rest_client = service_client.create_rest_client()
        ttl_f = rest_client.set_checkpoint_ttl_from_tinker_path(tinker_path, ttl_seconds=None)
        ttl_f.result()
        logger.info("Cleared checkpoint TTL (no expiry).")
    except Exception as exc:
        logger.warning("Could not clear TTL: %s", exc)
        api_capture["ttl_error"] = str(exc)

    meta = {
        "backend": "tinker",
        "training": "rl_corpus",
        "init_from_tinker_path": init_from_tinker_path,
        "base_model": base_model,
        "lora_rank": lora_rank,
        "batches": batches,
        "batch_size": batch_size,
        "group_size": group_size,
        "learning_rate": learning_rate,
        "max_tokens_sample": max_tokens,
        "temperature": temperature,
        "checkpoint_name": checkpoint_name,
        "corpus": str(input_path.resolve()),
        "tinker_checkpoint_path": tinker_path,
        "chunk_chars": chunk_chars,
        "stride_chars": stride_chars,
    }
    DEFAULT_RL_METADATA.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Wrote %s", DEFAULT_RL_METADATA)

    DEFAULT_API_DUMP.write_text(json.dumps(api_capture, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        if sync_from_cpt_metadata(DEFAULT_RL_METADATA, api_dump_path=str(DEFAULT_API_DUMP.resolve())):
            logger.info("Updated %s", OUTPUTTED_MODELS_JSON.name)
    except OSError as exc:
        logger.warning("Registry update failed: %s", exc)

    return {
        "metadata_path": str(DEFAULT_RL_METADATA.resolve()),
        "api_dump_path": str(DEFAULT_API_DUMP.resolve()),
        "tinker_checkpoint_path": tinker_path,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[rl] %(message)s")
    p = argparse.ArgumentParser(description="RL (importance_sampling) on Slack cpt_out.txt with oracle suffix reward")
    p.add_argument("--input", type=Path, default=DEFAULT_CPT_FILE, help="Plain text corpus (default: cpt_out.txt)")
    p.add_argument("--base-model", default="Qwen/Qwen3-8B")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--batches", type=int, default=20, help="Number of optimizer steps")
    p.add_argument("--batch-size", type=int, default=4, help="Prefixes per batch (each gets group_size samples)")
    p.add_argument("--group-size", type=int, default=8, help="Samples per prefix (for advantage baseline)")
    p.add_argument("--lr", type=float, default=3e-5, help="Adam LR for RL step")
    p.add_argument("--max-tokens", type=int, default=128, help="Max new tokens per rollout")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--min-prefix-tokens", type=int, default=48)
    p.add_argument("--min-suffix-tokens", type=int, default=32)
    p.add_argument("--checkpoint-name", default="slack-rl-corpus", help="Sampler name on Tinker")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-chars", type=int, default=7000)
    p.add_argument("--stride-chars", type=int, default=3000)
    p.add_argument(
        "--resume-state-path",
        default=None,
        help="Optional tinker state path to resume optimizer state",
    )
    p.add_argument(
        "--init-from-tinker",
        default=None,
        metavar="TINKER_URI",
        help=(
            "Training checkpoint tinker://…/weights/{name} from CPT save_state (NOT …/sampler_weights/…)."
        ),
    )
    args = p.parse_args()
    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    if sys.version_info < (3, 11):
        raise SystemExit("rl_corpus.py requires Python 3.11+")

    run_rl(
        input_path=inp,
        base_model=args.base_model,
        lora_rank=args.lora_rank,
        batches=args.batches,
        batch_size=args.batch_size,
        group_size=args.group_size,
        learning_rate=args.lr,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        min_prefix_tokens=args.min_prefix_tokens,
        min_suffix_tokens=args.min_suffix_tokens,
        checkpoint_name=args.checkpoint_name,
        seed=args.seed,
        chunk_chars=args.chunk_chars,
        stride_chars=args.stride_chars,
        resume_state_path=args.resume_state_path,
        init_from_tinker_path=args.init_from_tinker,
    )


if __name__ == "__main__":
    main()
