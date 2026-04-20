#!/usr/bin/env python3
"""
Supervised fine-tuning (SFT) — LoRA on Qwen3-4B from ``sft_output.json``.

The file is a JSON array of objects with ``messages`` =
``[system, user, assistant]`` (texting-style), as produced by ``export_imessage``.
Exports may be a single ``[system, user, assistant]`` triple or a longer chain
``[system, user, assistant, user, …, assistant]`` matching ``chat.py`` (alternating incoming /
Thomas). Legacy 2-turn ``[user, assistant]`` rows get the same system prompt prepended at load time.

SFT masks loss on user tokens so the model learns to continue like *you* after that context.
This often improves conversational fit vs CPT alone, which only sees raw message text.

Adapter output: ``models/messages-lora-sft``.

Example:
  python3 sft.py
  python3 sft.py --input sft_output.json --output models/messages-lora-sft --steps 300

Tuning notes (learning rate & LoRA)
  **Learning rate** — Step size for AdamW on LoRA weights only. Too **high** → loss spikes,
  garbled or generic output, forgetting the base model’s behavior. Too **low** → little
  change unless you add steps. For local SFT on a 4B-class model, ``1e-5``–``3e-5`` is a
  common sweep range; this file defaults to ``1.5e-5`` (slightly gentler than ``2e-5`` to
  reduce memorized “stock” phrases). Try ``1e-5`` if unstable, ``2e-5``–``3e-5`` if underfitting.

  **LoRA rank (``r``)** — Width of the low-rank update matrices. Higher **r** → more
  trainable capacity (can fit quirks / risk overfitting on small data). **8–32** is typical
  for personal adapters; **16** is a solid default.

  **LoRA alpha** — Scales how strongly the adapter is blended in (PEFT uses scaling
  related to ``alpha / r``). Raising **alpha** with fixed **r** pushes the model further
  from the base checkpoint per step; often **alpha ≈ 2× r** (e.g. 32 with ``r=16``).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from continued_pretrain import resolve_base_model
from export_imessage import TEXTING_SYSTEM_PROMPT

IGNORE = -100
DEFAULT_ADAPTER_DIR = Path("models") / "messages-lora-sft"
DEFAULT_SFT_JSON = Path("sft_output.json")

# Local SFT defaults (see module docstring “Tuning notes”).
DEFAULT_LR = 1.5e-5
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32


def _log(msg: str) -> None:
    print(msg, flush=True)


def _check_training_deps() -> None:
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc).split()[-1]
        raise SystemExit(
            f"Missing training dependency ({missing!r}). Install from the project root:\n"
            "  pip install -r requirements.txt\n"
            "Then re-run."
        ) from exc


def _roles_alternate_after_system(clean: list[dict[str, str]]) -> bool:
    """``user`` / ``assistant`` / ``user`` / … ending with ``assistant`` (Thomas’s reply)."""
    if len(clean) < 3 or clean[0]["role"] != "system":
        return False
    rest = clean[1:]
    if rest[-1]["role"] != "assistant":
        return False
    for i, m in enumerate(rest):
        want = "user" if i % 2 == 0 else "assistant"
        if m["role"] != want:
            return False
    return True


def _normalize_sft_messages(msgs: Any) -> list[dict[str, str]] | None:
    """Return messages for training: system + alternating user/assistant, ending with assistant."""
    if not isinstance(msgs, list) or len(msgs) < 2:
        return None
    clean: list[dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if not role or content is None:
            continue
        clean.append({"role": str(role), "content": str(content)})
    if len(clean) == 2 and clean[0]["role"] == "user" and clean[1]["role"] == "assistant":
        return [
            {"role": "system", "content": TEXTING_SYSTEM_PROMPT},
            clean[0],
            clean[1],
        ]
    if len(clean) >= 3 and _roles_alternate_after_system(clean):
        return clean
    return None


def load_sft_messages(path: Path) -> list[list[dict[str, str]]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if isinstance(data, dict):
        for key in ("examples", "data", "items"):
            inner = data.get(key)
            if isinstance(inner, list):
                data = inner
                break
        else:
            raise ValueError(
                f"{path}: expected a JSON array of {{'messages': [...]}}, "
                f"or an object with an 'examples' / 'data' / 'items' array"
            )

    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level JSON must be an array (or a dict containing one)")

    rows: list[list[dict[str, str]]] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        msgs = obj.get("messages")
        norm = _normalize_sft_messages(msgs)
        if norm is not None:
            rows.append(norm)
    if not rows:
        raise ValueError(f"No valid SFT examples in {path}")
    _log(f"[sft] Loaded {len(rows):,} conversation example(s) from {path.name}")
    return rows


def _apply_chat_template_ids(tokenizer, messages: list[dict[str, str]], **kwargs: Any) -> Any:
    """Qwen3-style thinking off when the tokenizer supports it."""
    kwargs = dict(kwargs)
    kwargs.setdefault("enable_thinking", False)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


class SFTChatDataset(Dataset):
    """One JSON example (system + alternating user/assistant, ending with assistant) → train last assistant only."""

    def __init__(self, rows: list[list[dict[str, str]]], tokenizer, max_length: int):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        def _ids(x: Any) -> list[int]:
            if isinstance(x, torch.Tensor):
                return x.squeeze().tolist()
            return list(x)

        messages = self.rows[i]
        prompt_messages = messages[:-1]
        prompt_ids: list[int] = _ids(
            _apply_chat_template_ids(
                self.tok,
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=self.max_length,
            )
        )
        full_ids: list[int] = _ids(
            _apply_chat_template_ids(
                self.tok,
                messages,
                tokenize=True,
                add_generation_prompt=False,
                truncation=True,
                max_length=self.max_length,
            )
        )
        plen = min(len(prompt_ids), len(full_ids))

        pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        ids = full_ids[: self.max_length]
        pad_len = self.max_length - len(ids)
        input_ids = ids + [pad_id] * pad_len

        labels_list = list(input_ids)
        for j in range(min(plen, len(labels_list))):
            labels_list[j] = IGNORE
        for j in range(len(full_ids), len(labels_list)):
            labels_list[j] = IGNORE

        input_ids_t = torch.tensor(input_ids, dtype=torch.long)
        attn = torch.tensor([1] * len(ids) + [0] * pad_len, dtype=torch.long)
        labels_t = torch.tensor(labels_list, dtype=torch.long)
        labels_t[attn == 0] = IGNORE
        return {"input_ids": input_ids_t, "attention_mask": attn, "labels": labels_t}


def supervised_finetune_lora(
    *,
    rows: list[list[dict[str, str]]],
    output_dir: Path,
    base_model: str,
    steps: int = 300,
    batch_size: int = 1,
    lr: float = DEFAULT_LR,
    max_length: Optional[int] = None,
    grad_clip: float = 1.0,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
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
        max_length = 256 if device.type == "mps" else 512

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    _log(
        f"[sft] LoRA SFT | device={device} dtype={dtype} seq_len={max_length} "
        f"examples={len(rows)}"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)

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

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    ds = SFTChatDataset(rows, tokenizer, max_length)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    log_interval = max(1, steps // 25)
    it = iter(loader)
    _log(f"[sft] Starting SFT: {steps} step(s), lr={lr}, batch_size={batch_size}")
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
            _log(f"[sft]   step {step:>4}/{steps} ({pct:5.1f} %)  loss {loss_scalar:.4f}")

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
        "kind": "sft",
    }
    (output_dir / "sft_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _log(f"[sft] Saving adapter to {output_dir}…")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    _log("[sft] SFT finished.")
    return output_dir


def run_sft(
    *,
    input_path: Path | None = None,
    output_dir: Path | None = None,
    base: str | None = None,
    steps: int = 300,
    batch_size: int = 1,
    lr: float = DEFAULT_LR,
    max_length: int | None = None,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
) -> Path:
    """SFT from ``sft_output.json`` into ``models/messages-lora-sft`` by default."""
    from export_imessage import SFT_OUTPUT_FILE

    _log("[sft] run_sft() starting…")
    inp = (input_path or SFT_OUTPUT_FILE).expanduser().resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"SFT JSON not found: {inp} (run export first)")
    out = (output_dir or DEFAULT_ADAPTER_DIR).expanduser().resolve()
    rows = load_sft_messages(inp)
    base_resolved = resolve_base_model(base)
    _log(f"[sft] Base model: {base_resolved!r}")
    _log(f"[sft] Output dir: {out}")
    return supervised_finetune_lora(
        rows=rows,
        output_dir=out,
        base_model=base_resolved,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        max_length=max_length,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )


def main() -> None:
    _log("[sft] sft.py (CLI)")
    p = argparse.ArgumentParser(description="LoRA SFT on Qwen3-4B from sft_output.json")
    p.add_argument("--input", type=Path, default=DEFAULT_SFT_JSON, help="JSON array from export_imessage")
    p.add_argument("--output", type=Path, default=DEFAULT_ADAPTER_DIR, help="PEFT output directory")
    p.add_argument("--base", default=None, help="Base model id or local path")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
        help="AdamW LR on LoRA only (default 1.5e-5; try 1e-5 if unstable, 2e-5–3e-5 if weak)",
    )
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument(
        "--lora-r",
        type=int,
        default=DEFAULT_LORA_R,
        help="LoRA rank (adapter width; try 8–32)",
    )
    p.add_argument(
        "--lora-alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
        help="LoRA alpha (scaling vs base; often ~2× rank)",
    )
    args = p.parse_args()

    run_sft(
        input_path=args.input,
        output_dir=args.output,
        base=args.base,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_length=args.max_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
