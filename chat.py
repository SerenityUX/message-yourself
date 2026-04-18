#!/usr/bin/env python3
"""
Interactive “text yourself” chat with the newest LoRA under ./models/.

Uses the same system prompt as ``export_imessage`` / ``sft.py``, ``enable_thinking=False``
in the chat template when supported, and short SMS-friendly generations on MPS.

Run:  python3 chat.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Thread
from typing import Iterator

import torch

from continued_pretrain import resolve_base_model
from export_imessage import TEXTING_SYSTEM_PROMPT

MODELS_DIR = Path("models")
BASE_DIR_NAME = "Qwen3-4B"
# ~16GB unified memory; SMS bursts can be a bit longer than assistant blurbs
DEFAULT_MAX_NEW = 256
# Max non-system messages kept (each user or assistant line counts as one).
# ~5 full turns (user+assistant) + current user before reply ≈ 11 before trim.
MAX_HISTORY_NON_SYSTEM = 10


def _trim_chat_history(messages: list[dict[str, str]], *, max_non_system: int) -> None:
    """
    Keep ``messages[0]`` (system) + the tail of the rest, at most ``max_non_system`` items.

    A naive ``messages[-N:]`` can start with an **assistant** turn (slice cuts inside a
    pair), which breaks chat order and looks like “forgot the last messages.” We drop
    leading assistant fragments so the first non-system message is always **user**.
    """
    if len(messages) <= 1:
        return
    system = messages[0]
    tail = messages[1:]
    if len(tail) <= max_non_system:
        while tail and tail[0]["role"] == "assistant":
            tail.pop(0)
        messages[:] = [system] + tail
        return
    tail = tail[-max_non_system:]
    while tail and tail[0]["role"] == "assistant":
        tail.pop(0)
    messages[:] = [system] + tail


def _log(msg: str) -> None:
    print(msg, flush=True)


def find_latest_adapter(models_dir: Path) -> Path | None:
    """Newest directory under models/ that looks like a PEFT adapter (not the raw base weights)."""
    if not models_dir.is_dir():
        return None
    candidates: list[tuple[float, Path]] = []
    for p in models_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name == BASE_DIR_NAME:
            continue
        if (p / "adapter_config.json").is_file():
            candidates.append((p.stat().st_mtime, p))
    if not candidates:
        return None
    paths = [p for _, p in candidates]
    # Prefer SFT adapter for chat when present (usually the more “you-shaped” model).
    sft_named = [p for p in paths if p.name == "messages-lora-sft"]
    pool = sft_named if sft_named else paths
    return max(pool, key=lambda p: p.stat().st_mtime)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _base_for_adapter(adapter_path: Path) -> str:
    cfg_path = adapter_path / "adapter_config.json"
    if cfg_path.is_file():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        b = data.get("base_model_name_or_path")
        if isinstance(b, str) and b.strip():
            if Path(b).is_dir():
                return str(Path(b).resolve())
            return b.strip()
    return resolve_base_model(None)


def _load_model(adapter_path: Path, base_model: str):
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc).split()[-1]
        raise SystemExit(
            f"Missing dependency ({missing!r}). From the project root run:\n"
            "  pip install -r requirements.txt\n"
            "Then try chat again."
        ) from exc

    device = _device()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float16

    _log(f"[chat] Device: {device}  dtype: {dtype}")
    _log(f"[chat] Base model: {base_model}")
    _log(f"[chat] Adapter:    {adapter_path}")

    _log("[chat] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _log("[chat] Loading base weights (may download on first use)…")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    _log("[chat] Merging LoRA adapter…")
    model = PeftModel.from_pretrained(model, str(adapter_path), torch_dtype=dtype)
    model = model.to(device)
    model.eval()
    _log("[chat] Model ready for inference.")
    return model, tokenizer, device


def _stream_reply(
    model,
    tokenizer,
    device,
    messages: list[dict[str, str]],
    *,
    max_new: int,
    temperature: float,
) -> Iterator[str]:
    from transformers import TextIteratorStreamer

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new,
        "do_sample": True,
        "temperature": temperature,
        "top_p": 0.85,
        "top_k": 20,
        "pad_token_id": tokenizer.pad_token_id,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    for piece in streamer:
        if piece:
            yield piece
    thread.join()


def chat_loop(
    *,
    adapter_path: Path,
    base_model: str | None = None,
    max_new: int = DEFAULT_MAX_NEW,
    temperature: float = 0.7,
) -> None:
    base = _base_for_adapter(adapter_path) if base_model is None else resolve_base_model(base_model)
    model, tokenizer, device = _load_model(adapter_path, base)

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": TEXTING_SYSTEM_PROMPT,
        }
    ]

    _log(
        "\n[chat] ─────────────────────────────────────────\n"
        "[chat] Texting-style chat (thinking off in the template). Type an incoming line to\n"
        "[chat] reply to—like the last text you got—or /quit /exit, or Ctrl+D to leave.\n"
        "[chat] ─────────────────────────────────────────\n"
    )
    while True:
        try:
            user = input("To reply to: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.", flush=True)
            break
        if not user:
            continue
        if user.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Bye.", flush=True)
            break

        messages.append({"role": "user", "content": user})
        prev_len = len(messages)
        _trim_chat_history(messages, max_non_system=MAX_HISTORY_NON_SYSTEM)
        if len(messages) < prev_len:
            _log(
                f"[chat] (trimmed older turns; keeping last {MAX_HISTORY_NON_SYSTEM} "
                "user/assistant lines + system)"
            )

        _log(f"[chat] Generating (max_new_tokens={max_new})…")
        print("Me: ", end="", flush=True)
        parts: list[str] = []
        try:
            with torch.no_grad():
                for chunk in _stream_reply(
                    model,
                    tokenizer,
                    device,
                    messages,
                    max_new=max_new,
                    temperature=temperature,
                ):
                    print(chunk, end="", flush=True)
                    parts.append(chunk)
        except Exception as exc:
            print(f"\n[generation error: {exc}]", flush=True)
            messages.pop()
            continue
        print(flush=True)
        reply = "".join(parts).strip()
        if reply:
            messages.append({"role": "assistant", "content": reply})

        if device.type == "mps":
            torch.mps.empty_cache()


def run_chat() -> None:
    try:
        import peft  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc).split()[-1]
        raise SystemExit(
            f"Missing dependency ({missing!r}). From the project root run:\n"
            "  pip install -r requirements.txt\n"
            "Then try chat again."
        ) from exc

    root = MODELS_DIR.resolve()
    _log(f"[chat] Looking for newest PEFT adapter under {root}…")
    adapter = find_latest_adapter(root)
    if adapter is None:
        print(
            f"No LoRA adapter found under {root}.\n"
            "Run `python3 main.py` first (export + train), or save an adapter in models/<name>/ "
            "with adapter_config.json.",
            file=sys.stderr,
        )
        sys.exit(1)

    _log(f"[chat] Using adapter: {adapter}")
    chat_loop(adapter_path=adapter)


def main() -> None:
    run_chat()


if __name__ == "__main__":
    main()
