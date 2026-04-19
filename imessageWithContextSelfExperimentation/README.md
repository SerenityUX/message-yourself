# iMessage with context — **self** experimentation

Tinker-only SFT + **Claude-controlled LR×LoRA sweeps**, fixed scenario eval (OpenRouter friend + rubric), and Tinker chat. Same message export as the sibling `imessageWithContextExperimentation` tree, but `main.py` is trimmed to **agent / single Tinker / grid** (no local PEFT here).

## Quick start

```bash
cd imessageWithContextSelfExperimentation
python3.12 -m venv .venv && source .venv/bin/activate   # or use repo `slack/.venv` if you prefer
pip install -r requirements.txt
cp .env.example .env    # then fill keys (never commit `.env`)
python3 main.py
```

Use **Train → SFT → (1) agent**, **(2) one Tinker run**, or **(3) classic grid**. Chat always uses **Tinker** (`sft_tinker_metadata.json` or `TINKER_CHAT_MODEL_URI`).

## Environment (`.env`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `TINKER_API_KEY` | Yes | SFT + Thomas completions on Tinker |
| `OPEN_ROUTER_API_KEY` | Yes | Friend roleplay + rubric + agent controller (via OpenRouter) |
| `OPEN_ROUTER_MODEL` | No | Friend model slug (default Claude Sonnet) |
| `OPEN_ROUTER_RATER_MODEL` | No | Rubric model |
| `OPEN_ROUTER_AGENT_MODEL` | No | Controller for agent sweep |
| `TINKER_BASE_MODEL` | No | Base slug for `eval-base` (default `openai/gpt-oss-120b`) |
| `TINKER_CHAT_MODEL_URI` | No | Override chat sampler `tinker://…` |
| `HF_TOKEN` | No | Hugging Face Hub (faster tokenizer/model fetch) |

Copy **`.env.example`** to **`.env`** in this directory. `python-dotenv` loads `.env` from here when you run scripts from this folder.

## Git: what is ignored vs tracked

This folder includes **`.gitignore`** (same rules can be copied if you split this tree into its own repo).

**Tracked (safe to commit):**

- `experiment_results/**` — per–(lr,rank) folders with scenario JSON, rubrics, `manifest.json`, agent log/history, `base_results/`, etc.
- `.env.example`, source, `README.md`.

**Not committed (keep locally only):**

- `sft_output.json` — training pairs can contain pasted API keys in message bodies; GitHub push protection rejects it. Regenerate via Train → export next to `main.py`.

**Ignored:**

- `.env` and any secret files.
- `**/sft_tinker_api_dump.json` — large API captures.
- `models/`, `*.safetensors`, `*.bin`, … local weights.
- `.venv/`, `__pycache__/`.
- `cpt_out.txt`, `messages_export.txt` (large regenerated exports).

The **repository root** `.gitignore` still applies globally (OS junk, venvs, etc.).

## CLI besides `main.py`

```bash
python3 -m experiment.run_experiment eval-base     # base gpt-oss eval → experiment_results/base_results/
python3 -m experiment.run_experiment eval --metadata path/to/sft_tinker_metadata.json
python3 -m experiment.run_experiment train-sweep --lrs … --ranks …
```

See `experiment/run_experiment.py` `--help` for flags.

## Related

- **`imessageWithContextExperimentation/`** — same ideas + **local PEFT** + **fixed grid experiment** menu (no Claude agent).
- Repo root `.gitignore` — shared defaults.
