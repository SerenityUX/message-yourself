# message-yourself

Monorepo for iMessage-style corpora, LoRA training (local + **Tinker**), and eval harnesses.

## Packages

| Directory | Purpose |
|-----------|---------|
| **`imessageWithContextSelfExperimentation/`** | Tinker SFT + **Claude agent** LR×rank sweep + rubric eval. Start with its **README** + **`.env.example`**. |
| **`imessageWithContextExperimentation/`** | Local PEFT + Tinker + **fixed grid** experiment (no agent). |
| **`imessageWithContext/`** | Core Tinker chat / SFT scripts. |
| **`imessage/`** | Raw iMessage export + training entrypoints. |
| **`slack/`** | Slack export → CPT / RL on Tinker. |

Each package has its own **README** (setup + **Git / `.env`**) and, where useful, a **`.gitignore`** fragment you can copy if you split a subtree into a new repo.

## Git (repository root)

The root **`.gitignore`** applies everywhere: OS cruft, Python caches, all `.venv`s, global secret globs, weight file extensions, `**/sft_tinker_api_dump.json`, etc.

**`sft_output.json`** is ignored everywhere: message bodies can contain pasted secrets, and GitHub blocks pushes that include them.

Never commit **`.env`**. Always commit **`.env.example`** templates.

## Static site (GitHub Pages)

Landing page at repo root: **`index.html`**, **`favicon.svg`**. The page loads eval run JSON from **`CLI/experiment_results/`** by default (`MESSAGE_YOURSELF_DATA_BASE` in `index.html`).

**Local preview:** from repo root, `python3 -m http.server 8000`, then open http://localhost:8000/

To publish, use a GitHub Actions workflow (e.g. `.github/workflows/pages.yml`) and set **Settings → Pages → GitHub Actions** as the source.
