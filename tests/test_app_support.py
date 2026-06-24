import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.app_support import country_year_lookup, predict_country_year, top_emitters_recent
from src.data import FEATURES, TARGET, load_panel


@pytest.fixture(scope="module")
def panel():
    return load_panel()


def test_country_year_lookup_hit(panel):
    iso = panel["iso_code"].iloc[0]
    year = int(panel["year"].iloc[0])
    row = country_year_lookup(panel, iso, year)
    assert row is not None
    assert row["iso_code"] == iso


def test_country_year_lookup_miss(panel):
    assert country_year_lookup(panel, "ZZZ", 1800) is None


def test_top_emitters_returns_n_rows(panel):
    result = top_emitters_recent(panel, n=5)
    assert len(result) <= 5


def test_top_emitters_year_none_uses_max(panel):
    result_none = top_emitters_recent(panel, year=None, n=5)
    max_year = int(panel["year"].max())
    result_max = top_emitters_recent(panel, year=max_year, n=5)
    pd.testing.assert_frame_equal(result_none.reset_index(drop=True), result_max.reset_index(drop=True))


def test_top_emitters_empty_year(panel):
    result = top_emitters_recent(panel, year=1800, n=5)
    assert len(result) == 0


def test_predict_country_year_shape(panel):
    row = country_year_lookup(panel, panel["iso_code"].iloc[0], int(panel["year"].iloc[0]))
    from src.app_support import load_bundle
    try:
        bundle = load_bundle()
    except FileNotFoundError:
        pytest.skip("bundle.pkl not present — run python main.py first")
    out = predict_country_year(row, bundle)
    assert "label" in out
    assert "probability" in out
    assert "lr_probability" in out
    assert out["label"] in (0, 1)
    assert 0.0 <= out["probability"] <= 1.0
