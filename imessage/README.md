# iMessage (no “with context” wrapper)

Training / export stack for raw iMessage → corpora and local LoRA. See package scripts and parent repo docs.

## Git & environment

- **`.env`** — never commit. Copy **`.env.example`** in this folder (or **`../slack/.env.example`** for Slack tokens).
- **`.gitignore`** in this folder lists local-only patterns; the **repository root** `.gitignore` still applies (including ignored `sft_output.json`, `cpt_out.txt`, `models/`, `.venv/`).

## Where to go next

- **Context + Tinker experiments:** `../imessageWithContextExperimentation/` or `../imessageWithContextSelfExperimentation/` (read each README).
