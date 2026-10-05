# AI Personality Development Coach: with CI/CD

![CI](https://github.com/YOUR_USERNAME/ai-personality-coach/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/YOUR_USERNAME/ai-personality-coach/actions/workflows/cd.yml/badge.svg)

Flask + scikit-learn app that scores 7 personality traits from a 15-question scenario quiz and
generates improvement plans. This repo adds a full DevOps pipeline around it.

## Pipeline

```
push / PR → GitHub Actions CI
              ├─ ruff lint
              ├─ pytest (+ coverage gate ≥ 80%)
              ├─ model retrain + accuracy gate (≥ 85% per trait)
              ├─ pip-audit (dependency CVEs)
              ├─ docker build → Trivy image scan → container smoke test
merge to main → CD
              ├─ build & push image to GHCR (:latest and :<commit-sha>)
              ├─ trigger Render deploy
              └─ post-deploy /health check
```

## Run locally

```bash
# plain Python (needs Python 3.11+)
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt
pytest
FLASK_DEBUG=1 python app.py                          # http://127.0.0.1:5000

# or Docker
docker compose up --build                            # http://localhost:8000
```

Demo login: `admin` / `1234` (override with `DEMO_USERNAME`, `DEMO_PASSWORD`, `SECRET_KEY` env vars).

## Retrain the models

```bash
python train.py                # writes models.pkl + metrics.json
```
`models.pkl` is tied to **scikit-learn 1.8.0** (pinned in requirements.txt).
If you upgrade scikit-learn, retrain and commit a new `models.pkl`, or the model test will fail.

## Required GitHub secrets

| Secret | Value |
|---|---|
| `RENDER_DEPLOY_HOOK` | Deploy Hook URL from Render service settings |
| `PROD_URL` | e.g. `https://your-app.onrender.com` (no trailing slash) |
