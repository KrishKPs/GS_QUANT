# Factor Risk Terminal

Portfolio factor attribution and risk decomposition, measured on realised data.
Analytics engine is **gs-quant** (`gs_quant.timeseries`, offline — no Marquee
credentials); HAC inference, Euler risk decomposition, Ledoit–Wolf, Cornish–Fisher
and the optimisers are built on top. Nothing here forecasts anything.

- `core/` — the maths, pure and unit-tested
- `api.py` — FastAPI JSON API over `core/`
- `frontend/` — React + Vite + TypeScript UI, eight views
- `app.py` — the original Streamlit UI, superseded by the React frontend

## Run

```bash
# 1. engine + API  (http://127.0.0.1:8000)
.venv/bin/uvicorn api:app --port 8000 --reload

# 2. frontend      (http://localhost:5173 — proxies /api to the port above)
cd frontend && npm run dev
```

Browse sample data with no backend running: `http://localhost:5173/?mock=1`.

Portfolio and date range live in the URL, so any analysis is shareable:
`/?p=AAPL:0.25,MSFT:0.20,JPM:0.20,XOM:0.20,JNJ:0.15&start=2018-01-01&end=2025-06-30`

## Test

```bash
.venv/bin/python -m pytest -q          # 43 known-answer + invariant tests
cd frontend && npx tsc -b && npm run build
```
