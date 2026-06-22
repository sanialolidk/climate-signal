# climate-signal

Sania Thankan — Penn State, Computational Data Science

Classifies country-years with elevated GHG temperature forcing using the [Our World in Data CO₂ dataset](https://ourworldindata.org/co2-emissions). Time split: train ≤ 2010, test > 2010.

**Live app:** deploy from [share.streamlit.io](https://share.streamlit.io) → `sanialolidk/climate-signal` → `app.py`

## Run locally

```bash
git clone https://github.com/sanialolidk/climate-signal.git
cd climate-signal
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-train.txt
streamlit run app.py
```

Retrain (optional):

```bash
python scripts/build_panel.py   # refreshes data/panel.csv from OWID
python main.py                  # writes models/ and plots/
```

## Results (holdout 2011+)

| Model | Acc | Macro F1 | ROC-AUC |
|-------|-----|----------|---------|
| Logistic regression | 0.90 | 0.90 | 0.979 |
| HistGradientBoosting | 0.94 | 0.94 | 0.988 |

## Deploy notes

- `data/panel.csv` is committed (~560 KB) so Streamlit Cloud never downloads OWID at runtime
- Use `environment.yml` only (Python 3.11.9)
- Main file: `app.py`

## Stack

Python, pandas, scikit-learn, matplotlib, streamlit