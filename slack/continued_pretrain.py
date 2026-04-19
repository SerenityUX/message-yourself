#!/usr/bin/env python3
"""
Continued pre-training (CPT) on **Slack** ``cpt_out.txt`` using the
`Tinker <https://tinker-docs.thinkingmachines.ai/>`_ API (remote LoRA, cross-entropy).

Unlike ``imessage/continued_pretrain.py`` (local PyTorch + PEFT), this script runs the
training loop on your machine but **forward/backward + optim** execute on Tinker GPUs.

Prerequisites::

    Python **3.11+** (Tinker SDK). Then: ``pip install -r requirements.txt``

Set ``TINKER_API_KEY`` in ``slack/.env`` (loaded automatically) or in the environment.
See https://tinker-console.thinkingmachines.ai/

Default corpus: ``slack/cpt_out.txt`` (from ``prepare_slack_cpt`` / ``main.py``).

**Training cost (rough):** Tinker bills **Train** by **tokens processed** through
``forward_backward`` (see `Models & Pricing <https://tinker-docs.thinkingmachines.ai/tinker/models/>`__).
For ``Qwen/Qwen3-8B``, public list prices have been on the order of **~\\$0.40 / 1M train
tokens** (check `Tinker Console <https://tinker-console.thinkingmachines.ai/>`__ for current rates).

**Corpus size vs bill:** The **file size of** ``cpt_out.txt`` does **not** set cost by itself. Cost scales
with **training steps × batch × tokens per forward** (each datum is capped by ``--max-length``, default
2048). Example defaults (``steps=200``, ``batch_size=1``, ``max_length=2048``): at most **~200 × 2048 ≈
419k train-token positions** per run if every batch uses a full-length sequence — **order ~\\$0.17** at
~\\$0.40/M (often less if chunks tokenize shorter). A **~1 MiB** ``cpt_out.txt`` mostly affects how many
**datum** chunks you build, not the token count per step.

**Checkpoints:** After training, the script saves a ``tinker://…`` sampler path, **removes server TTL**
so the checkpoint is **kept indefinitely** (see `Weights <https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/weights/>`__),
writes ``cpt_tinker_metadata.json``, and optionally **downloads** LoRA files under ``slack/`` for local
merge/serve. Use ``run_slack_lora_inference.py`` for API sampling from the saved path.

**LoRA vs full weights:** Tinker’s ``create_lora_training_client`` **only optimizes LoRA adapter
weights**; the base model stays frozen (see `Tinker overview <https://tinker-docs.thinkingmachines.ai/tinker/>`__).
You still pay Train pricing on **tokens** run through the model. MoE models are billed using **active**
experts per token (different pricing table), not a switch to “train fewer params” on dense Qwen.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

if sys.version_info < (3, 11):
    raise SystemExit(
        "continued_pretrain.py requires Python 3.11+ (the Tinker SDK does not support 3.9/3.10).\n"
        "  Recreate the venv with a newer interpreter, e.g.:\n"
        "    cd slack && rm -rf .venv && python3.12 -m venv .venv && source .venv/bin/activate\n"
        "    pip install -r requirements.txt\n"
        f"  Current Python: {sys.version.split()[0]}"
    )

from dotenv import load_dotenv

import tinker
import torch
from tinker import types

from tinker_cookbook.supervised.common import datum_from_model_input_weights

from outputted_models import OUTPUTTED_MODELS_JSON, sync_from_cpt_metadata

_SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(_SCRIPT_DIR / ".env")
DEFAULT_CPT_FILE = _SCRIPT_DIR / "cpt_out.txt"
DEFAULT_CHECKPOINT_NAME = "slack-cpt-lora"
DEFAULT_ADAPTER_DIR = _SCRIPT_DIR / "tinker_lora_adapter"
DEFAULT_API_DUMP_FILE = _SCRIPT_DIR / "tinker_api_full_response.json"
DEFAULT_SMOKE_PROMPT = "Hi — quick sanity check after training. Reply in one short sentence."


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_text_chunks(path: Path, chunk_chars: int, stride_chars: int) -> list[str]:
    """Same chunking idea as ``imessage/continued_pretrain.py`` — join non-empty lines, slice."""

    _log(f"[cpt] Reading {path.name}…")
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No non-empty lines in {path}")
    joined = "\n".join(lines)
    if len(joined) <= 20:
        raise ValueError("Not enough text after stripping empty lines")

    _log(f"[cpt] {len(lines):,} lines, {len(joined):,} chars → chunks…")
    chunks: list[str] = []
    i = 0
    while i < len(joined):
        piece = joined[i : i + chunk_chars]
        if len(piece.strip()) > 20:
            chunks.append(piece)
        i += stride_chars
        if stride_chars < 1:
            break
    if not chunks:
        raise ValueError("Chunking produced no segments; adjust chunk/stride")
    _log(f"[cpt] {len(chunks)} chunk(s) (chunk_chars={chunk_chars}, stride={stride_chars})")
    return chunks


def text_to_datum(
    tokenizer,
    text: str,
    max_length: int,
):
    """Plain-text CPT: next-token CE on all tokens (weights = 1)."""

    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return None
    model_input = types.ModelInput.from_ints(ids)
    weights = torch.ones(len(ids), dtype=torch.float32)
    try:
        # API: (model_input, weights, max_length) — no ``reduction`` (see tinker_cookbook.supervised.common).
        return datum_from_model_input_weights(
            model_input,
            weights,
            max_length=max_length,
        )
    except Exception:
        return None


def _resolve_tinker_checkpoint_path(
    training_client: object,
    sampling_client: object,
    save_result: object | None,
) -> str | None:
    if save_result is not None:
        p = getattr(save_result, "path", None)
        if p:
            return str(p)
    for obj in (sampling_client, training_client):
        for attr in ("model_path", "path", "checkpoint_path"):
            p = getattr(obj, attr, None)
            if p and str(p).startswith("tinker://"):
                return str(p)
    return None


def _loss_from_forward_backward_output(fw_result: Any) -> float | None:
    """``ForwardBackwardOutput`` uses ``metrics`` / ``loss_fn_outputs``, not a top-level ``loss``."""

    loss = getattr(fw_result, "loss", None)
    if loss is not None:
        try:
            return float(loss)
        except (TypeError, ValueError):
            pass

    metrics = getattr(fw_result, "metrics", None)
    if isinstance(metrics, dict) and metrics:
        for key in (
            "loss",
            "mean_loss",
            "nll",
            "mean_nll",
            "cross_entropy",
            "lm_loss",
            "train/loss",
            "train_loss",
        ):
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    continue
        for mk, mv in metrics.items():
            lk = str(mk).lower()
            if "loss" in lk or "nll" in lk:
                try:
                    return float(mv)
                except (TypeError, ValueError):
                    continue

    outputs = getattr(fw_result, "loss_fn_outputs", None)
    if outputs:
        for out in outputs:
            if not isinstance(out, dict):
                continue
            for name in ("loss", "nll", "mean_nll"):
                if name not in out:
                    continue
                td = out[name]
                data = getattr(td, "data", None)
                if isinstance(data, list) and data:
                    try:
                        return float(data[0])
                    except (TypeError, ValueError, IndexError):
                        continue

    return None


def _json_safe(o: Any, depth: int = 0) -> Any:
    """Best-effort JSON-serialization of Tinker SDK objects, tensors, and nested structures."""

    if depth > 10:
        return "<max depth>"
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    if isinstance(o, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(x, depth + 1) for x in o]
    if hasattr(o, "tolist"):
        try:
            return o.tolist()
        except Exception:
            pass
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "__dict__") and type(o).__module__ not in ("builtins",):
        try:
            return _json_safe(vars(o), depth + 1)
        except Exception:
            pass
    out: dict[str, Any] = {}
    for name in dir(o):
        if name.startswith("_"):
            continue
        try:
            val = getattr(o, name)
            if callable(val):
                continue
            out[name] = _json_safe(val, depth + 1)
        except Exception as exc:
            out[name] = f"<error reading: {exc}>"
    return out if out else repr(o)


async def run_cpt_async(
    *,
    input_path: Path,
    base_model: str,
    lora_rank: int,
    steps: int,
    batch_size: int,
    lr: float,
    max_length: int,
    chunk_chars: int,
    stride_chars: int,
    checkpoint_name: str,
    seed: int,
    download_adapter_dir: Path | None,
    skip_indefinite_ttl: bool,
    run_smoke_inference: bool = True,
    smoke_prompt: str = DEFAULT_SMOKE_PROMPT,
    api_dump_path: Path | None = None,
) -> dict[str, Any]:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "Set TINKER_API_KEY (see https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)"
        )

    dump_path = api_dump_path if api_dump_path is not None else DEFAULT_API_DUMP_FILE
    api_capture: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "checkpoint_name": checkpoint_name,
        "phases": {},
    }

    random.seed(seed)
    chunks = load_text_chunks(input_path, chunk_chars=chunk_chars, stride_chars=stride_chars)

    _log(f"[cpt] Connecting to Tinker | base_model={base_model!r} LoRA rank={lora_rank}")
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=base_model,
        rank=lora_rank,
    )
    try:
        if hasattr(training_client, "get_info_async"):
            info = await training_client.get_info_async()
        else:
            info = training_client.get_info()
        api_capture["phases"]["create_lora_training_client"] = {
            "training_client_info": _json_safe(info),
        }
    except Exception as exc:
        api_capture["phases"]["create_lora_training_client"] = {"note": str(exc)}
    tokenizer = training_client.get_tokenizer()

    datums: list = []
    for i, text in enumerate(chunks):
        d = text_to_datum(tokenizer, text, max_length=max_length)
        if d is not None:
            datums.append(d)
        if (i + 1) % 500 == 0:
            _log(f"[cpt] Encoded {i + 1}/{len(chunks)} chunks → {len(datums)} valid datums")

    if not datums:
        raise SystemExit("[cpt] No valid training datums (text too short after tokenization?)")

    _log(f"[cpt] Training on {len(datums)} datum(s); {steps} step(s), batch_size={batch_size}, lr={lr}")

    log_interval = max(1, steps // 20)
    last_fw_snapshot: dict[str, Any] | None = None
    for step in range(1, steps + 1):
        k = min(batch_size, len(datums))
        batch = random.sample(datums, k=k)

        fw_future = await training_client.forward_backward_async(batch, "cross_entropy")
        opt_future = await training_client.optim_step_async(tinker.AdamParams(learning_rate=lr))
        fw_result = await fw_future.result_async()
        await opt_future.result_async()

        loss = _loss_from_forward_backward_output(fw_result)
        if step == 1 or step == steps or step % log_interval == 0:
            loss_s = f"{loss:.4f}" if loss is not None else "?"
            if loss is None and step == 1:
                m = getattr(fw_result, "metrics", None)
                _log(
                    f"[cpt] (debug) could not parse loss; metrics keys: "
                    f"{list(m.keys()) if isinstance(m, dict) else m!r}"
                )
            _log(f"[cpt]   step {step:>5}/{steps}  loss {loss_s}")
        if step == steps:
            last_fw_snapshot = {
                "step": step,
                "loss": float(loss) if loss is not None else None,
                "forward_backward_result": _json_safe(fw_result),
            }
    if last_fw_snapshot:
        api_capture["phases"]["last_training_step"] = last_fw_snapshot

    # Training checkpoint under ``tinker://…/weights/{name}`` — required for ``create_training_client_from_state``
    # (e.g. CPT→RL warm-start). Sampler-only paths ``…/sampler_weights/…`` are invalid for ``load_state`` (400).
    _log(f"[cpt] Saving training checkpoint for resume / RL warm-start ({checkpoint_name!r}, weights/)…")
    training_weights_path: str | None = None
    try:
        st_future = training_client.save_state(checkpoint_name)
        st_result = await st_future.result_async()
        training_weights_path = getattr(st_result, "path", None) or getattr(st_result, "checkpoint_path", None)
        if training_weights_path:
            training_weights_path = str(training_weights_path)
        api_capture["phases"]["save_state"] = {
            "path": training_weights_path,
            "raw": _json_safe(st_result) if st_result is not None else None,
        }
        _log(f"[cpt] Training weights URI (for RL load_state): {training_weights_path!r}")
    except Exception as exc:
        api_capture["phases"]["save_state"] = {"error": str(exc)}
        _log(f"[cpt] Warning: save_state failed ({exc}); CPT→RL warm-start will not work until fixed.")

    _log(f"[cpt] Saving sampler weights as {checkpoint_name!r}…")
    save_result = None
    # ``save_weights_for_sampler_async`` is a coroutine that *returns* an APIFuture without
    # awaiting it, so ``await save_weights_for_sampler_async()`` yields the Future object,
    # not SaveWeightsForSamplerResponse (no ``.path``). Use the sync API + ``await`` the future.
    if hasattr(training_client, "save_weights_for_sampler"):
        save_future = training_client.save_weights_for_sampler(checkpoint_name)
        save_result = await save_future
        tinker_path = getattr(save_result, "path", None) or getattr(save_result, "checkpoint_path", None)
        if not tinker_path:
            raise SystemExit("[cpt] save_weights_for_sampler returned no path")
        tinker_path = str(tinker_path)
        sampling_client = await service_client.create_sampling_client_async(model_path=tinker_path)
    else:
        sampling_client = training_client.save_weights_and_get_sampling_client(name=checkpoint_name)
        tinker_path = _resolve_tinker_checkpoint_path(training_client, sampling_client, None)

    api_capture["phases"]["save_weights_for_sampler"] = {
        "raw_save_result": _json_safe(save_result) if save_result is not None else None,
        "tinker_checkpoint_path": tinker_path,
    }
    _log(f"[cpt] Checkpoint URI: {tinker_path!r}")

    if not skip_indefinite_ttl:
        try:
            rest_client = service_client.create_rest_client()
            ttl_cleared: list[str] = []
            for path in (training_weights_path, tinker_path):
                if not path:
                    continue
                await rest_client.set_checkpoint_ttl_from_tinker_path_async(path, ttl_seconds=None)
                ttl_cleared.append(path)
            api_capture["phases"]["set_checkpoint_ttl"] = {
                "ttl_seconds": None,
                "paths": ttl_cleared,
            }
            _log("[cpt] Removed checkpoint TTL on saved weights (no expiration).")
        except Exception as exc:
            api_capture["phases"]["set_checkpoint_ttl"] = {"error": str(exc)}
            _log(f"[cpt] Warning: could not clear checkpoint TTL ({exc}); path may use default TTL.")

    local_adapter: str | None = None
    if download_adapter_dir is not None:
        out_dir = download_adapter_dir.expanduser().resolve()
        try:
            from tinker_cookbook import weights as tinker_weights

            local_adapter = str(
                tinker_weights.download(tinker_path=tinker_path, output_dir=str(out_dir))
            )
            _log(f"[cpt] Downloaded LoRA adapter to {local_adapter}")
            api_capture["phases"]["download_adapter"] = {"local_adapter_dir": local_adapter}
        except Exception as exc:
            _log(f"[cpt] Warning: adapter download failed ({exc}); inference via tinker:// path still works.")
            api_capture["phases"]["download_adapter"] = {"error": str(exc)}

    _ = sampling_client  # ensure client constructed; inference uses metadata + fresh client

    if run_smoke_inference and tinker_path:
        _log("[cpt] Smoke test: Tinker Sampling API (sample_async)…")
        try:
            infer_client = await service_client.create_sampling_client_async(model_path=tinker_path)
            infer_tok = infer_client.get_tokenizer()
            prompt_ids = infer_tok.encode(smoke_prompt)
            model_input = types.ModelInput.from_ints(prompt_ids)
            params = types.SamplingParams(max_tokens=128, temperature=0.7)
            sample_resp = await infer_client.sample_async(
                prompt=model_input,
                sampling_params=params,
                num_samples=1,
            )
            decoded = infer_tok.decode(sample_resp.sequences[0].tokens)
            api_capture["phases"]["smoke_inference"] = {
                "prompt": smoke_prompt,
                "decoded_text": decoded,
                "sampling_params": _json_safe(params),
                "raw_sample_response": _json_safe(sample_resp),
            }
            _log(f"[cpt] Smoke reply (first line): {decoded[:200]!r}…")
        except Exception as exc:
            api_capture["phases"]["smoke_inference"] = {"error": str(exc)}
            _log(f"[cpt] Smoke inference failed ({exc}); metadata still saved.")

    meta = {
        "backend": "tinker",
        "base_model": base_model,
        "lora_rank": lora_rank,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_length": max_length,
        "chunk_chars": chunk_chars,
        "stride_chars": stride_chars,
        "training_datums": len(datums),
        "checkpoint_name": checkpoint_name,
        "corpus": str(input_path.resolve()),
        "tinker_checkpoint_path": tinker_path,
        "tinker_training_weights_path": training_weights_path,
        "local_adapter_dir": local_adapter,
    }
    meta_path = _SCRIPT_DIR / "cpt_tinker_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"[cpt] Wrote {meta_path} (use run_slack_lora_inference.py to sample)")

    api_capture["metadata_file"] = str(meta_path.resolve())
    api_capture["summary"] = {
        "tinker_checkpoint_path": tinker_path,
        "training_datums": len(datums),
        "steps": steps,
        "corpus": str(input_path.resolve()),
    }
    dump_path.write_text(json.dumps(api_capture, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"[cpt] Wrote full API capture → {dump_path}")

    if tinker_path:
        try:
            if sync_from_cpt_metadata(meta_path, api_dump_path=str(dump_path.resolve())):
                _log(f"[cpt] Saved registry → {OUTPUTTED_MODELS_JSON.name}")
        except OSError as exc:
            _log(f"[cpt] Warning: could not update outputted_models.json ({exc})")

    return {
        "metadata_path": str(meta_path.resolve()),
        "api_dump_path": str(dump_path.resolve()),
        "tinker_checkpoint_path": tinker_path,
        "tinker_training_weights_path": training_weights_path,
        "capture": api_capture,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Slack CPT via Tinker LoRA on slack/cpt_out.txt")
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CPT_FILE,
        help="Plain-text corpus (default: slack/cpt_out.txt next to this file)",
    )
    p.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="Tinker base model id (e.g. Qwen/Qwen3-8B, Qwen/Qwen3-4B-Instruct-2507; see docs)",
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=2048, help="Token cap per chunk (truncate)")
    p.add_argument("--chunk-chars", type=int, default=7000)
    p.add_argument("--stride-chars", type=int, default=3000)
    p.add_argument(
        "--checkpoint-name",
        default=DEFAULT_CHECKPOINT_NAME,
        help="Sampler checkpoint name on Tinker",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--download-adapter-dir",
        type=Path,
        nargs="?",
        const=DEFAULT_ADAPTER_DIR,
        default=None,
        metavar="DIR",
        help="Download LoRA files to this directory (default if flag alone: slack/tinker_lora_adapter). "
        "Omit flag to skip download.",
    )
    p.add_argument(
        "--skip-indefinite-ttl",
        action="store_true",
        help="Do not call set_checkpoint_ttl to remove expiry (use Tinker default TTL).",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    dl = args.download_adapter_dir
    if dl is not None:
        dl = dl.expanduser().resolve()

    asyncio.run(
        run_cpt_async(
            input_path=inp,
            base_model=args.base_model,
            lora_rank=args.lora_rank,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            chunk_chars=args.chunk_chars,
            stride_chars=args.stride_chars,
            checkpoint_name=args.checkpoint_name,
            seed=args.seed,
            download_adapter_dir=dl,
            skip_indefinite_ttl=args.skip_indefinite_ttl,
            run_smoke_inference=True,
            api_dump_path=DEFAULT_API_DUMP_FILE,
        )
    )


if __name__ == "__main__":
    main()
