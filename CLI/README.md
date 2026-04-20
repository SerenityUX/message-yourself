# CLI

```bash
cd CLI
/opt/homebrew/bin/python3.12 -m venv .venv   # once; macOS system python3 is often 3.9
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill keys
python3 cli.py
```

Trains register in `saved_models.json` (gitignored). Chat appears only when at least one model is saved. `.env`, `.venv/`, `sft_output.json`, `cpt_out.txt`, `experiment_results/*` (except `.gitkeep`), and Tinker dumps are gitignored under `CLI/.gitignore`.
