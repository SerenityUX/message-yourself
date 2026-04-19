# Slack → CPT / RL (Tinker)

Interactive pipeline: optional **Slack export** → `prepare_slack_cpt` → optional **Tinker CPT / RL** → optional **chat**.

## Environment (`.env`)

Copy **`.env.example`** → **`.env`** in this directory.

| Variable | When |
|----------|------|
| `TINKER_API_KEY` | Training + chat on Tinker |
| `SLACK_XOXC_TOKEN`, `SLACK_D_COOKIE` | Browser export (`exportSlackMessages.py`) |
| `HF_TOKEN` | Optional; HF Hub rate limits |

Never commit **`.env`**.

## Git (`.gitignore` here)

**Ignored locally:** `.env`, `.venv/`, `my_slack_messages*.json`, `cpt_out.txt`, Tinker metadata dumps listed in `.gitignore`, weight dirs.

**Tracked:** source, `requirements.txt`, `.env.example`, README.

The repository **root** `.gitignore` still applies.

## Run

```bash
cd slack && source .venv/bin/activate   # Python 3.11+
python main.py
# or: python main.py --no-prompt
```

See docstring in `main.py` for flags (`--train`, `--rl`, base model, etc.).
