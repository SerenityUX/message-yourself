# site/

Public static site for **Message Yourself**. Deployed by `.github/workflows/pages.yml` to GitHub Pages.

## Files
- `index.html` — the page
- `favicon.svg` — sun icon

## Data
The page fetches run results from `../CLI/experiment_results/` (configured via the `MESSAGE_YOURSELF_DATA_BASE` override in `index.html`). The deploy workflow copies that folder alongside the site, so at deploy time the relative path resolves.

## Deploy
1. Push to `main` — the workflow at `.github/workflows/pages.yml` runs.
2. Repo → **Settings → Pages** → Source = **GitHub Actions** (one-time).
3. Site appears at `https://<user>.github.io/<repo>/`.

## Local preview
```bash
# from the repo root:
python3 -m http.server 8000
# then open http://localhost:8000/site/
```
This works because `site/` sits next to `CLI/` in the repo — the relative `../CLI/experiment_results/` path resolves correctly.
