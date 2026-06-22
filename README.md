# climate-signal

Sania Thankan — Penn State, Computational Data Science

Small ML project I built on the [Our World in Data CO₂ dataset](https://github.com/owid/co2-data). It flags country-years that sit in the top quartile of GHG-attributed temperature change, using emissions features (per-capita CO₂, cumulative totals, methane, land-use change, etc.).

I used a time split (train ≤ 2010, test > 2010) instead of shuffling rows because the data is a panel.

**Live app:** https://climate-signal.streamlit.app *(deploy after first push)*

## Run locally

```bash
git clone https://github.com/sanialolidk/climate-signal.git
cd climate-signal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-train.txt
python scripts/download_data.py
python main.py
streamlit run app.py
```

## Results (holdout 2011+)

| Model | Acc | Macro F1 | ROC-AUC |
|-------|-----|----------|---------|
| Logistic regression | 0.90 | 0.90 | 0.979 |
| HistGradientBoosting | 0.94 | 0.94 | 0.988 |

## Notes

- 161 countries, ~5k train rows after filtering missing fields
- Permutation importance: population scale and cumulative CO₂ matter most
- This is not a weather model — just emissions panel classification

## Stack

Python, pandas, scikit-learn, matplotlib, streamlit