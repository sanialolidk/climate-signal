"""FastAPI backend for the Climate Signal React frontend."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from src.app_support import (
    country_year_lookup,
    load_bundle,
    load_metrics,
    load_panel_cached,
    predict_country_year,
    top_emitters_recent,
)
from src.data import FEATURES, TARGET

FEATURE_LABELS = {
    "co2_per_capita": "CO₂ per capita (tonnes)",
    "co2_growth_prct": "Annual CO₂ growth (%)",
    "cumulative_co2": "Cumulative CO₂ (Mt)",
    "share_global_co2": "Share of global CO₂ (%)",
    "energy_per_capita": "Energy per capita",
    "land_use_change_co2": "Land-use change CO₂",
    "methane_per_capita": "Methane per capita",
    "nitrous_oxide_per_capita": "N₂O per capita",
    "co2_per_gdp": "CO₂ per GDP",
    "population_millions": "Population (millions)",
    "co2_5yr_mean_growth": "Five-year mean CO₂ growth (%)",
    "years_since_1990": "Years since 1990",
}

SEVERITY_LABELS = {
    "standard": "Standard observed impact",
    "moderate": "Elevated observed impact",
    "high": "High observed impact",
}


@lru_cache(maxsize=1)
def get_panel() -> pd.DataFrame:
    return load_panel_cached()


@lru_cache(maxsize=1)
def get_bundle():
    return load_bundle()


@lru_cache(maxsize=1)
def get_metrics_data():
    return load_metrics()


def format_confidence(probability: float) -> str:
    if probability >= 0.9995:
        return ">99.9%"
    return f"{probability * 100:.1f}%"


def as_bool_flag(value) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(int(value))


def severity_level(observed_elevated: bool, temp, panel: pd.DataFrame) -> str:
    if not observed_elevated:
        return "standard"
    if temp is None or pd.isna(temp):
        return "moderate"
    temps = panel["temperature_change_from_ghg"].dropna()
    if temp >= temps.quantile(0.9):
        return "high"
    return "moderate"


def format_value(val, feat: str) -> str:
    if pd.isna(val):
        return "Not available"
    if feat in ("co2_per_capita", "methane_per_capita", "nitrous_oxide_per_capita"):
        return f"{float(val):.3f}"
    if feat == "cumulative_co2":
        return f"{float(val):,.1f}"
    if feat == "population_millions":
        return f"{float(val):.2f}"
    return f"{float(val):.3f}"


def row_to_indicators(row: dict) -> list[dict[str, Any]]:
    items = [
        {"feature": FEATURE_LABELS.get(f, f), "value": format_value(row.get(f, 0), f)}
        for f in FEATURES
    ]
    items.append({
        "feature": "Observed GHG forcing classification",
        "value": "Elevated" if as_bool_flag(row.get(TARGET, 0)) else "Standard",
    })
    return items


app = FastAPI(title="Climate Signal API", version="1.0.0")

origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "climate-signal"}


@app.get("/api/meta")
def meta():
    panel = get_panel()
    countries = sorted(panel["country"].unique().tolist())
    years = sorted(int(y) for y in panel["year"].unique())
    return {
        "countries": countries,
        "years": {"min": years[0], "max": years[-1]},
        "jurisdiction_count": int(panel["iso_code"].nunique()),
        "default_country": "United States" if "United States" in countries else countries[0],
        "default_year": years[-8] if len(years) > 8 else years[-1],
    }


@app.get("/api/assess")
def assess(country: str = Query(...), year: int = Query(...)):
    panel = get_panel()
    hits = panel.loc[panel["country"] == country, "iso_code"]
    if hits.empty:
        raise HTTPException(404, f"Unknown jurisdiction: {country}")

    iso = hits.iloc[0]
    row = country_year_lookup(panel, iso, year)
    if row is None:
        raise HTTPException(404, f"No data for {country} in {year}")

    out = predict_country_year(row, get_bundle())
    temp = row.get("temperature_change_from_ghg")
    observed = as_bool_flag(row.get(TARGET, 0))
    severity = severity_level(observed, temp, panel)

    return {
        "country": country,
        "year": year,
        "iso_code": iso,
        "classification": out["class_name"],
        "label": out["label"],
        "confidence": format_confidence(out["probability"]),
        "confidence_raw": out["probability"],
        "baseline_confidence": format_confidence(out["lr_probability"]),
        "temperature_change": None if pd.isna(temp) else round(float(temp), 4),
        "severity": severity,
        "severity_label": SEVERITY_LABELS[severity],
        "indicators": row_to_indicators(row),
    }


@app.get("/api/emitters")
def emitters(year: Optional[int] = None, limit: int = Query(15, ge=1, le=50)):
    panel = get_panel()
    latest = int(panel["year"].max()) if year is None else year
    table = top_emitters_recent(panel, year=latest, n=limit)
    records = []
    for _, r in table.iterrows():
        records.append({
            "jurisdiction": r["country"],
            "iso_code": r["iso_code"],
            "year": int(r["year"]),
            "co2_per_capita": round(float(r["co2_per_capita"]), 3),
            "temperature_change": round(float(r["temperature_change_from_ghg"]), 4),
            "classification": "Elevated" if as_bool_flag(r["elevated_forcing"]) else "Standard",
        })
    return {"year": latest, "rows": records}


@app.get("/api/timeseries")
def timeseries(countries: str = Query(..., description="Comma-separated country names")):
    panel = get_panel()
    names = [c.strip() for c in countries.split(",") if c.strip()]
    if not names:
        raise HTTPException(400, "Provide at least one country")

    series = []
    for name in names:
        grp = panel[panel["country"] == name].sort_values("year")
        if grp.empty:
            continue
        series.append({
            "country": name,
            "points": [
                {"year": int(row["year"]), "co2_per_capita": round(float(row["co2_per_capita"]), 3)}
                for _, row in grp.iterrows()
            ],
        })

    if not series:
        raise HTTPException(404, "No matching jurisdictions in panel")

    return {"series": series}


@app.get("/api/metrics")
def metrics():
    data = get_metrics_data()
    if not data:
        raise HTTPException(404, "Metrics not available")
    return data