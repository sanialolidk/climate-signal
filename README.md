# climate-signal

Sania Thankan — Penn State, Computational Data Science

Classifies country-years with elevated GHG temperature forcing using the [Our World in Data CO₂ dataset](https://ourworldindata.org/co2-emissions). Time split: train ≤ 2010, test > 2010.

## React frontend (recommended)

**Terminal 1 — API**
```bash
cd climate-signal
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — React UI**
```bash
cd climate-signal/frontend
npm install
npm run dev
```

Open https://climate-signal-fmmu6szzappwzmblrzafjek.streamlit.app/ — Vite proxies `/api` to the backend.

### API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/meta` | Countries, year range |
| `GET /api/assess?country=&year=` | Classification + indicators |
| `GET /api/emitters` | Top per-capita emitters |
| `GET /api/timeseries?countries=` | CO₂ time series |
| `GET /api/metrics` | Model validation stats |

## Streamlit app (legacy)

```bash
pip install -r requirements-train.txt
streamlit run app.py
```

## Train model

```bash
python scripts/build_panel.py   # refresh data/panel.csv from OWID
python main.py
```

## Results (holdout 2011+)

| Model | Acc | Macro F1 | ROC-AUC |
|-------|-----|----------|---------|
| Logistic regression | 0.90 | 0.90 | 0.979 |
| HistGradientBoosting | 0.94 | 0.94 | 0.988 |

## Stack

Python, FastAPI, React, Vite, Recharts, scikit-learn
