#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) on ``sft_output.json`` via Tinker LoRA (remote GPUs).

Uses the same chat rows as ``sft.py`` (``load_sft_messages``) and
``tinker_cookbook.supervised.data.conversation_to_datum`` with a Qwen3-Instruct
renderer, training the last assistant turn only (matches 1-reply ``sft_output.json`` rows).

Requires Python **3.11+**, ``TINKER_API_KEY`` in ``imessage/.env`` (or the environment),
and ``pip install -r requirements.txt`` (includes ``tinker`` + ``tinker-cookbook``).

Default base model: ``Qwen/Qwen3-4B-Instruct-2507`` (Qwen3 4B Instruct on Tinker; the
plain ``Qwen/Qwen3-4B`` hub id is not available for training on Tinker).

Example::

  python3 sft_tinker.py
  python3 sft_tinker.py --steps 300 --base-model Qwen/Qwen3-4B-Instruct-2507
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sft import load_sft_messages

_SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_SFT_JSON = _SCRIPT_DIR / "sft_output.json"
DEFAULT_CHECKPOINT_NAME = "imessage-sft-lora"
DEFAULT_ADAPTER_DIR = _SCRIPT_DIR / "tinker_lora_sft_adapter"
DEFAULT_API_DUMP = _SCRIPT_DIR / "sft_tinker_api_dump.json"
DEFAULT_METADATA = _SCRIPT_DIR / "sft_tinker_metadata.json"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_RENDERER = "qwen3_instruct"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _require_py311_or_exit() -> None:
    """Tinker and its wheel require Python 3.11+; keep imports lazy so ``main.py`` loads on 3.9."""
    if sys.version_info < (3, 11):
        raise SystemExit(
            "Tinker SFT requires Python 3.11+ (the Tinker SDK does not support older Python).\n"
            "  cd imessage && rm -rf .venv && python3.12 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            f"  Current interpreter: {sys.version.split()[0]} ({sys.executable})"
        )


def _json_safe(o: Any, depth: int = 0) -> Any:
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


def _loss_from_forward_backward_output(fw_result: Any) -> float | None:
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


async def run_sft_tinker_async(
    *,
    input_path: Path,
    base_model: str,
    renderer_name: str,
    lora_rank: int,
    steps: int,
    batch_size: int,
    lr: float,
    max_length: int,
    checkpoint_name: str,
    seed: int,
    download_adapter_dir: Path | None,
    skip_indefinite_ttl: bool,
    max_examples: int | None,
    run_smoke_inference: bool = True,
    api_dump_path: Path | None = None,
) -> dict[str, Any]:
    _require_py311_or_exit()

    from dotenv import load_dotenv

    load_dotenv(_SCRIPT_DIR / ".env")

    import tinker
    from tinker import types
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "Set TINKER_API_KEY in imessage/.env (see https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)"
        )

    inp = input_path.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"SFT JSON not found: {inp}")

    dump_path = api_dump_path if api_dump_path is not None else DEFAULT_API_DUMP
    api_capture: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "renderer": renderer_name,
        "checkpoint_name": checkpoint_name,
        "phases": {},
    }

    random.seed(seed)

    _log(f"[sft-tinker] Connecting to Tinker (validates base_model) | {base_model!r} LoRA rank={lora_rank}")
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
        api_capture["phases"]["create_lora_training_client"] = {"training_client_info": _json_safe(info)}
    except Exception as exc:
        api_capture["phases"]["create_lora_training_client"] = {"note": str(exc)}

    tokenizer = training_client.get_tokenizer()
    renderer = get_renderer(renderer_name, tokenizer, model_name=base_model)

    _log(f"[sft-tinker] Loading SFT JSON → datums (max_length={max_length})…")
    rows = load_sft_messages(inp)
    if max_examples is not None and max_examples > 0 and len(rows) > max_examples:
        rows = random.sample(rows, k=max_examples)
        _log(f"[sft-tinker] Subsampled to {len(rows):,} example(s) (--max-examples)")

    datums: list = []
    for i, msgs in enumerate(rows):
        try:
            d = conversation_to_datum(
                msgs,
                renderer,
                max_length=max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            datums.append(d)
        except Exception:
            continue
        if (i + 1) % 2000 == 0:
            _log(f"[sft-tinker] Encoded {i + 1}/{len(rows)} rows → {len(datums)} valid datums")

    if not datums:
        raise SystemExit("[sft-tinker] No valid training datums (check messages / renderer)")

    _log(f"[sft-tinker] {len(datums):,} datum(s); {steps} step(s), batch_size={batch_size}, lr={lr}")

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
            _log(f"[sft-tinker]   step {step:>5}/{steps}  loss {loss_s}")
        if step == steps:
            last_fw_snapshot = {
                "step": step,
                "loss": float(loss) if loss is not None else None,
                "forward_backward_result": _json_safe(fw_result),
            }
    if last_fw_snapshot:
        api_capture["phases"]["last_training_step"] = last_fw_snapshot

    _log(f"[sft-tinker] Saving training checkpoint ({checkpoint_name!r})…")
    training_weights_path: str | None = None
    try:
        st_future = training_client.save_state(checkpoint_name)
        st_result = await st_future.result_async()
        training_weights_path = getattr(st_result, "path", None) or getattr(
            st_result, "checkpoint_path", None
        )
        if training_weights_path:
            training_weights_path = str(training_weights_path)
        api_capture["phases"]["save_state"] = {
            "path": training_weights_path,
            "raw": _json_safe(st_result) if st_result is not None else None,
        }
        _log(f"[sft-tinker] Training weights URI: {training_weights_path!r}")
    except Exception as exc:
        api_capture["phases"]["save_state"] = {"error": str(exc)}
        _log(f"[sft-tinker] Warning: save_state failed ({exc}).")

    _log(f"[sft-tinker] Saving sampler weights as {checkpoint_name!r}…")
    save_result = None
    if hasattr(training_client, "save_weights_for_sampler"):
        save_future = training_client.save_weights_for_sampler(checkpoint_name)
        save_result = await save_future
        tinker_path = getattr(save_result, "path", None) or getattr(save_result, "checkpoint_path", None)
        if not tinker_path:
            raise SystemExit("[sft-tinker] save_weights_for_sampler returned no path")
        tinker_path = str(tinker_path)
        sampling_client = await service_client.create_sampling_client_async(model_path=tinker_path)
    else:
        sampling_client = training_client.save_weights_and_get_sampling_client(name=checkpoint_name)
        tinker_path = _resolve_tinker_checkpoint_path(training_client, sampling_client, None)

    api_capture["phases"]["save_weights_for_sampler"] = {
        "raw_save_result": _json_safe(save_result) if save_result is not None else None,
        "tinker_checkpoint_path": tinker_path,
    }
    _log(f"[sft-tinker] Checkpoint URI: {tinker_path!r}")

    if not skip_indefinite_ttl:
        try:
            rest_client = service_client.create_rest_client()
            ttl_cleared: list[str] = []
            for path in (training_weights_path, tinker_path):
                if not path:
                    continue
                await rest_client.set_checkpoint_ttl_from_tinker_path_async(path, ttl_seconds=None)
                ttl_cleared.append(path)
            api_capture["phases"]["set_checkpoint_ttl"] = {"ttl_seconds": None, "paths": ttl_cleared}
            _log("[sft-tinker] Removed checkpoint TTL on saved weights (no expiration).")
        except Exception as exc:
            api_capture["phases"]["set_checkpoint_ttl"] = {"error": str(exc)}
            _log(f"[sft-tinker] Warning: could not clear checkpoint TTL ({exc}).")

    local_adapter: str | None = None
    if download_adapter_dir is not None:
        out_dir = download_adapter_dir.expanduser().resolve()
        try:
            from tinker_cookbook import weights as tinker_weights

            local_adapter = str(tinker_weights.download(tinker_path=tinker_path, output_dir=str(out_dir)))
            _log(f"[sft-tinker] Downloaded LoRA adapter to {local_adapter}")
            api_capture["phases"]["download_adapter"] = {"local_adapter_dir": local_adapter}
        except Exception as exc:
            _log(f"[sft-tinker] Warning: adapter download failed ({exc}).")
            api_capture["phases"]["download_adapter"] = {"error": str(exc)}

    _ = sampling_client

    if run_smoke_inference and tinker_path:
        _log("[sft-tinker] Smoke test: sample_async…")
        try:
            infer_client = await service_client.create_sampling_client_async(model_path=tinker_path)
            infer_tok = infer_client.get_tokenizer()
            smoke_prompt = "Hi — quick sanity check after SFT. Reply in one short sentence."
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
                "raw_sample_response": _json_safe(sample_resp),
            }
            _log(f"[sft-tinker] Smoke reply (first 200 chars): {decoded[:200]!r}…")
        except Exception as exc:
            api_capture["phases"]["smoke_inference"] = {"error": str(exc)}
            _log(f"[sft-tinker] Smoke inference failed ({exc}); metadata still saved.")

    meta = {
        "backend": "tinker",
        "kind": "sft",
        "base_model": base_model,
        "renderer": renderer_name,
        "lora_rank": lora_rank,
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": lr,
        "max_length": max_length,
        "training_datums": len(datums),
        "checkpoint_name": checkpoint_name,
        "dataset": str(inp.resolve()),
        "tinker_checkpoint_path": tinker_path,
        "tinker_training_weights_path": training_weights_path,
        "local_adapter_dir": local_adapter,
    }
    meta_path = DEFAULT_METADATA
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"[sft-tinker] Wrote {meta_path}")

    api_capture["metadata_file"] = str(meta_path.resolve())
    api_capture["summary"] = {
        "tinker_checkpoint_path": tinker_path,
        "training_datums": len(datums),
        "steps": steps,
        "dataset": str(inp.resolve()),
    }
    dump_path.write_text(json.dumps(api_capture, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"[sft-tinker] Wrote API capture → {dump_path}")

    return {
        "metadata_path": str(meta_path.resolve()),
        "api_dump_path": str(dump_path.resolve()),
        "tinker_checkpoint_path": tinker_path,
        "tinker_training_weights_path": training_weights_path,
        "capture": api_capture,
    }


def run_sft_tinker(
    *,
    input_path: Path | None = None,
    base_model: str | None = None,
    renderer_name: str = DEFAULT_RENDERER,
    lora_rank: int = 16,
    steps: int = 200,
    batch_size: int = 1,
    lr: float = 2e-4,
    max_length: int = 1024,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    seed: int = 42,
    download_adapter_dir: Path | None = None,
    skip_indefinite_ttl: bool = False,
    max_examples: int | None = None,
) -> dict[str, Any]:
    """Run Tinker SFT; returns metadata paths. Requires asyncio event loop."""
    inp = (input_path or DEFAULT_SFT_JSON).expanduser().resolve()
    dl = download_adapter_dir
    if dl is not None:
        dl = dl.expanduser().resolve()
    return asyncio.run(
        run_sft_tinker_async(
            input_path=inp,
            base_model=base_model or DEFAULT_BASE_MODEL,
            renderer_name=renderer_name,
            lora_rank=lora_rank,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            max_length=max_length,
            checkpoint_name=checkpoint_name,
            seed=seed,
            download_adapter_dir=dl,
            skip_indefinite_ttl=skip_indefinite_ttl,
            max_examples=max_examples,
            run_smoke_inference=True,
            api_dump_path=DEFAULT_API_DUMP,
        )
    )


def main() -> None:
    p = argparse.ArgumentParser(description="iMessage SFT via Tinker LoRA on sft_output.json")
    p.add_argument("--input", type=Path, default=DEFAULT_SFT_JSON, help="JSON from export_imessage")
    p.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Tinker base model id (default: Qwen/Qwen3-4B-Instruct-2507)",
    )
    p.add_argument(
        "--renderer",
        default=DEFAULT_RENDERER,
        help="tinker_cookbook renderer (default: qwen3_instruct for Instruct-2507)",
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-length", type=int, default=1024, help="Token cap per example")
    p.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--download-adapter-dir",
        type=Path,
        nargs="?",
        const=DEFAULT_ADAPTER_DIR,
        default=None,
        metavar="DIR",
        help="Download LoRA to this dir (default if flag alone: imessage/tinker_lora_sft_adapter)",
    )
    p.add_argument(
        "--skip-indefinite-ttl",
        action="store_true",
        help="Do not remove checkpoint TTL on Tinker",
    )
    p.add_argument(
        "--max-examples",
        type=int,
        default=None,
        metavar="N",
        help="Train on a random subset of N rows (debug / cheaper runs)",
    )
    args = p.parse_args()

    dl = args.download_adapter_dir
    if dl is not None:
        dl = dl.expanduser().resolve()

    asyncio.run(
        run_sft_tinker_async(
            input_path=args.input,
            base_model=args.base_model,
            renderer_name=args.renderer,
            lora_rank=args.lora_rank,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            checkpoint_name=args.checkpoint_name,
            seed=args.seed,
            download_adapter_dir=dl,
            skip_indefinite_ttl=args.skip_indefinite_ttl,
            max_examples=args.max_examples,
            run_smoke_inference=True,
            api_dump_path=DEFAULT_API_DUMP,
        )
    )


if __name__ == "__main__":
    main()
