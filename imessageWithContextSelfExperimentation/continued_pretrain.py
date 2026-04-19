#!/usr/bin/env python3
"""
Continued pre-training (CPT) — LoRA on Qwen3-4B from plain-text ``cpt_out.txt``.

CPT = next-token prediction on chunked corpus (same idea as SerenityUX/fine-tuned-design-qwen ``cpt_train.py``).

Default input: ``cpt_out.txt`` (from ``export_imessage.run_export()``).
Adapter output: ``models/messages-lora-cpt``.

For conversational tuning on (previous message → your reply), use ``sft.py`` on ``sft_output.json``.

Default **learning rate** here is higher than in ``sft.py``: CPT does full-sequence next-token
CE on text chunks, while SFT masks to assistant tokens only—different loss scale and typical LR range.
LoRA **rank/alpha** follow the same idea as SFT (see ``sft.py`` docstring).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

IGNORE = -100

# CPT adapter directory (see ``sft.py`` for ``messages-lora-sft``).
DEFAULT_ADAPTER_DIR = Path("models") / "messages-lora-cpt"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _check_training_deps() -> None:
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc).split()[-1]
        raise SystemExit(
            f"Missing training dependency ({missing!r}). Install packages from the project root:\n"
            "  python3 -m venv .venv && source .venv/bin/activate\n"
            "  pip install -r requirements.txt\n"
            "Then re-run this command."
        ) from exc


def resolve_base_model(explicit: str | None) -> str:
    if explicit:
        return explicit
    local = Path("models") / "Qwen3-4B"
    if local.is_dir():
        return str(local.resolve())
    return "Qwen/Qwen3-4B"


def load_text_chunks(path: Path, chunk_chars: int, stride_chars: int) -> list[str]:
    """
    Turn the export file into overlapping/non-overlapping text blocks for CPT.
    Blank lines are dropped; remaining lines are joined with newlines, then sliced.
    """
    _log(f"[cpt] Reading export file ({path.name})…")
    raw = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No non-empty lines in {path}")
    joined = "\n".join(lines)
    if len(joined) <= 20:
        raise ValueError(f"Not enough text in {path} after stripping empty lines")

    _log(f"[cpt] Joined {len(lines):,} non-empty lines ({len(joined):,} chars); building chunks…")
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
        raise ValueError("Chunking produced no usable segments; lower chunking threshold or add text")
    _log(f"[cpt] Built {len(chunks)} chunk(s) (chunk_chars={chunk_chars}, stride={stride_chars}).")
    return chunks


class ChunkTextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer, max_length: int):
        self.rows = [t for t in texts if len(t.strip()) > 20]
        if not self.rows:
            raise ValueError("No training rows after filtering (need chunks longer than 20 characters)")
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        text = self.rows[i]
        enc = self.tok(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attn = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attn == 0] = IGNORE
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def continue_pretrain_messages(
    *,
    texts: list[str],
    output_dir: Path,
    base_model: str,
    steps: int = 500,
    batch_size: int = 1,
    lr: float = 2.5e-4,  # typically > SFT LR; not interchangeable with ``sft.py`` defaults
    max_length: Optional[int] = None,
    grad_clip: float = 1.0,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
) -> Path:
    _check_training_deps()
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    if max_length is None:
        # Shorter on MPS to reduce OOM risk; still higher than the old 128 default.
        max_length = 256 if device.type == "mps" else 512

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    _log(
        f"[cpt] LoRA CPT | device={device} dtype={dtype} seq_len={max_length} "
        f"dataset_rows={len(texts)}"
    )
    _log(f"[cpt] Loading tokenizer from {base_model!r}…")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _log("[cpt] Loading base model weights (first run may download)…")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    _log("[cpt] Base model on device; attaching LoRA (attention + MLP)…")

    # Train all standard LoRA targets on MPS too for higher capacity (watch memory on 16GB).
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    _log("[cpt] LoRA ready; enabling gradient checkpointing…")

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    ds = ChunkTextDataset(texts, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    _log(
        f"[cpt] Starting optimization: {steps} step(s), batch_size={batch_size}, "
        f"lr={lr}, grad_clip={grad_clip}"
    )
    log_interval = max(1, steps // 25)
    it = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss
        loss_scalar = float(loss.item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if step == 1 or step == steps or step % log_interval == 0:
            pct = 100.0 * step / steps
            _log(f"[cpt]   step {step:>4}/{steps} ({pct:5.1f} %)  loss {loss_scalar:.4f}")

        del out, loss, batch
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    meta = {
        "base_model": base_model,
        "steps": steps,
        "training_rows": len(ds),
        "max_length": max_length,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
    }
    (output_dir / "cpt_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _log(f"[cpt] Saving adapter + tokenizer to {output_dir}…")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _log("[cpt] CPT finished.")
    return output_dir


def train_from_export_file(
    *,
    input_path: Path,
    output_dir: Path,
    base: str | None = None,
    steps: int = 500,
    batch_size: int = 1,
    lr: float = 2.5e-4,
    max_length: int | None = None,
    chunk_chars: int = 7000,
    stride_chars: int = 3000,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> Path:
    """CPT only: build chunks from an existing export file and save adapter."""
    inp = input_path.expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")

    _log(f"[cpt] Output directory: {output_dir.resolve()}")
    base_resolved = resolve_base_model(base)
    _log(f"[cpt] Base model resolved to: {base_resolved!r}")
    chunks = load_text_chunks(inp, chunk_chars=chunk_chars, stride_chars=stride_chars)

    return continue_pretrain_messages(
        texts=chunks,
        output_dir=output_dir,
        base_model=base_resolved,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )


def run_cpt(
    *,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    base: str | None = None,
    steps: int = 500,
    batch_size: int = 1,
    lr: float = 2.5e-4,
    max_length: int | None = None,
    chunk_chars: int = 7000,
    stride_chars: int = 3000,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> Path:
    """CPT LoRA from ``cpt_out.txt`` (or ``input_path``) into ``models/messages-lora-cpt`` by default."""
    from export_imessage import CPT_OUTPUT_FILE

    _log("[cpt] run_cpt() starting…")
    inp = (input_path or CPT_OUTPUT_FILE).expanduser().resolve()
    out = (output_dir or DEFAULT_ADAPTER_DIR).expanduser().resolve()
    return train_from_export_file(
        input_path=inp,
        output_dir=out,
        base=base,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        chunk_chars=chunk_chars,
        stride_chars=stride_chars,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )


def run_export_and_train(
    *,
    skip_export: bool = False,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    base: str | None = None,
    steps: int = 500,
    batch_size: int = 1,
    lr: float = 2.5e-4,
    max_length: int | None = None,
    chunk_chars: int = 7000,
    stride_chars: int = 3000,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> Path:
    """
    Export iMessages to ``cpt_out.txt`` / ``sft_output.json``, then CPT LoRA on Qwen3-4B.
    Defaults are tuned for a stronger CPT signal (see module docstring if you need to ease off).
    """
    from export_imessage import CPT_OUTPUT_FILE, run_export

    if not skip_export:
        run_export()

    return run_cpt(
        input_path=input_path or CPT_OUTPUT_FILE,
        output_dir=output_dir,
        base=base,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        chunk_chars=chunk_chars,
        stride_chars=stride_chars,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )


# Backward-compatible name
run_train_adapter = run_cpt


def main() -> None:
    _log("[cpt] continued_pretrain.py (CLI)")
    p = argparse.ArgumentParser(description="Continued pre-training (CPT) LoRA on Qwen3-4B from cpt_out.txt.")
    p.add_argument(
        "--skip-export",
        action="store_true",
        help="Do not re-run export; train from existing --input file",
    )
    p.add_argument(
        "--input",
        type=Path,
        default=Path("cpt_out.txt"),
        help="Plain-text CPT corpus (default: ./cpt_out.txt)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Directory for saved PEFT adapter",
    )
    p.add_argument("--base", default=None, help="Base model id or local path (default: models/Qwen3-4B or Hub)")
    p.add_argument("--steps", type=int, default=500, help="CPT optimizer steps (default: higher intensity)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Tokens per example; default 256 on MPS, 512 on CUDA",
    )
    p.add_argument(
        "--chunk-chars",
        type=int,
        default=7000,
        help="Max characters per CPT row",
    )
    p.add_argument(
        "--stride-chars",
        type=int,
        default=3000,
        help="Stride between chunks (smaller = more overlap and more training rows)",
    )
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    args = p.parse_args()

    if not args.skip_export:
        from export_imessage import run_export

        run_export()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    run_cpt(
        input_path=inp,
        output_dir=args.output,
        base=args.base,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        chunk_chars=args.chunk_chars,
        stride_chars=args.stride_chars,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
