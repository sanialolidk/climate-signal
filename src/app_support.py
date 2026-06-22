import json

import joblib
import pandas as pd
import streamlit as st

from .data import FEATURES, TARGET, load_panel
from .paths import p


@st.cache_resource
def load_bundle():
    return joblib.load(p("models", "bundle.pkl"))


@st.cache_data
def load_metrics():
    path = p("models", "metrics.json")
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data
def load_panel_cached():
    return load_panel()


def predict_country_year(row_dict, bundle):
    cols = bundle["features"]
    row = pd.DataFrame([[row_dict.get(c, 0.0) for c in cols]], columns=cols)
    scaled = bundle["scaler"].transform(row)

    lr_p = float(bundle["lr"].predict_proba(scaled)[0, 1])
    gb_p = float(bundle["gb"].predict_proba(scaled)[0, 1])
    gb_l = int(gb_p >= 0.5)

    return {
        "label": gb_l,
        "class_name": "Elevated forcing" if gb_l else "Within normal range",
        "probability": gb_p,
        "lr_probability": lr_p,
    }


def country_year_lookup(panel, iso_code, year):
    hit = panel[(panel["iso_code"] == iso_code) & (panel["year"] == year)]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def top_emitters_recent(panel, year=None, n=12):
    sub = panel.copy()
    if year:
        sub = sub[sub["year"] == year]
    else:
        sub = sub[sub["year"] == sub["year"].max()]
    return (
        sub.nlargest(n, "co2_per_capita")[
            ["country", "iso_code", "year", "co2_per_capita", "temperature_change_from_ghg", TARGET]
        ]
        .rename(columns={TARGET: "elevated_forcing"})
    )