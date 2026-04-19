# iMessage with context (core)

Tinker chat + SFT (`sft_tinker.py`, `chat.py`) without the full **experiment** harness. Use sibling folders for sweeps.

## Git & environment

- **`.env`** in this directory (not committed). Copy **`.env.example`** here; for OpenRouter + agent-related keys see **`../imessageWithContextSelfExperimentation/.env.example`**.
- **`.gitignore`** here + **root** `.gitignore`: ignores weights, `.venv/`, `sft_output.json`, dumps, secrets.

## Experiments

- **`../imessageWithContextExperimentation/`** — local PEFT + Tinker + **fixed LR×grid** eval.
- **`../imessageWithContextSelfExperimentation/`** — **Claude agent** sweep + Tinker + same eval layout.
