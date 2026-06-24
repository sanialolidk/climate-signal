import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import FEATURES, TARGET, build_panel_from_owid, load_panel, split_by_year
from src.paths import p

PANEL_PATH = p("data", "panel.csv")


def test_load_panel_returns_dataframe():
    panel = load_panel()
    assert isinstance(panel, pd.DataFrame)
    assert len(panel) > 0


def test_panel_has_required_columns():
    panel = load_panel()
    for col in FEATURES + [TARGET, "country", "year", "iso_code"]:
        assert col in panel.columns, f"Missing column: {col}"


def test_panel_target_is_binary():
    panel = load_panel()
    assert set(panel[TARGET].dropna().unique()).issubset({0, 1, 0.0, 1.0})


def test_split_by_year():
    panel = load_panel()
    train, test = split_by_year(panel, train_end=2010)
    assert train["year"].max() <= 2010
    assert test["year"].min() > 2010
    assert len(train) + len(test) == len(panel)


def test_load_panel_raises_on_missing_file(tmp_path, monkeypatch):
    import src.data as data_mod
    monkeypatch.setattr(data_mod, "PANEL_PATH", tmp_path / "missing.csv")
    with pytest.raises(FileNotFoundError):
        data_mod.load_panel()


def test_build_panel_from_owid_columns():
    raw_path = p("data", "owid-co2-data.csv")
    if not raw_path.exists():
        pytest.skip("owid-co2-data.csv not present locally")
    panel = build_panel_from_owid(raw_path, min_year=2000)
    assert TARGET in panel.columns
    assert "iso_code" in panel.columns
    assert panel["iso_code"].str.len().eq(3).all()
    assert panel["year"].min() >= 2000
