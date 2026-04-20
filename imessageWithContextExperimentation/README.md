# iMessage with context — experimentation (classic)

Export iMessage → CPT / SFT with **local PEFT** or **Tinker**, plus a **fixed LR×LoRA train+sweep** (no Claude agent). Eval writes under `experiment_results/<lr>_<rank>/`.

## Quick start

```bash
cd imessageWithContextExperimentation
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python3 main.py
```

**Train → SFT** offers **local**, **single Tinker**, or **experiment** (grid of lr×rank + scenario rubric eval).

## Environment (`.env`)

| Variable | Required | Purpose |
|----------|----------|---------|
| `TINKER_API_KEY` | For Tinker paths | SFT + chat on Tinker |
| `OPEN_ROUTER_API_KEY` | For experiment eval | Friend + rubric (OpenRouter) |
| `OPEN_ROUTER_MODEL` / `OPEN_ROUTER_RATER_MODEL` | No | Slugs (defaults in `experiment/config.py`) |
| `HF_TOKEN` | No | HF Hub downloads |

Use **`.env.example`** as the template. Never commit **`.env`**.

## Git: what is ignored vs tracked

See **`.gitignore`** in this directory (replicable if you vendor only this tree).

**Tracked:** source, `.env.example`. **`experiment_results/**` is not committed** (manifests include `tinker://` URIs and local paths). Only `experiment_results/.gitkeep` (and `-2`/`-3` placeholders) stays in git.

**Ignored:** `.env`, `**/sft_tinker_metadata.json`, `**/sft_tinker_api_dump.json`, `sft_output.json`, `models/`, weight blobs, `.venv/`, large exports (`cpt_out.txt`, `messages_export.txt`).

Root **`.gitignore`** still applies for repo-wide rules.

## `main.py` menu (this package only)

1. **Train** — export → CPT or SFT (**local** / **Tinker** / **experiment grid**).  
2. **Chat** — Tinker inference from metadata or `TINKER_CHAT_MODEL_URI`.

For **Claude agent sweeps**, use sibling **`imessageWithContextSelfExperimentation`**.
