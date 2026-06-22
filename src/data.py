import numpy as np
import pandas as pd

from .paths import p

FEATURES = [
    "co2_per_capita",
    "co2_growth_prct",
    "cumulative_co2",
    "share_global_co2",
    "energy_per_capita",
    "land_use_change_co2",
    "methane_per_capita",
    "nitrous_oxide_per_capita",
    "co2_per_gdp",
    "population_millions",
    "co2_5yr_mean_growth",
    "years_since_1990",
]

TARGET = "elevated_forcing"


def load_panel(csv_path=None, min_year=1990):
    # OWID export — filter to ISO countries with enough history
    path = p("data", "owid-co2-data.csv") if csv_path is None else csv_path
    raw = pd.read_csv(path)
    raw = raw[raw["year"] >= min_year].copy()
    raw = raw[raw["iso_code"].notna() & (raw["iso_code"].str.len() == 3)]
    raw = raw[~raw["iso_code"].isin(["OWID_WRL"])]

    raw["population_millions"] = raw["population"] / 1e6
    raw = raw.sort_values(["iso_code", "year"])

    raw["co2_5yr_mean_growth"] = (
        raw.groupby("iso_code")["co2_growth_prct"]
        .transform(lambda s: s.rolling(5, min_periods=2).mean())
    )
    raw["years_since_1990"] = raw["year"] - 1990

    temp = raw["temperature_change_from_ghg"]
    valid = temp.notna()
    cutoff = temp[valid].quantile(0.75)
    raw[TARGET] = np.where(valid, (temp >= cutoff).astype(int), np.nan)

    keep = FEATURES + [TARGET, "country", "year", "iso_code", "temperature_change_from_ghg"]
    panel = raw[keep].dropna(subset=FEATURES + [TARGET])
    return panel.reset_index(drop=True)


def split_by_year(panel, train_end=2010):
    train = panel[panel["year"] <= train_end].copy()
    test = panel[panel["year"] > train_end].copy()
    return train, test