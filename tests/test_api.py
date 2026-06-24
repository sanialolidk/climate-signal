import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import FEATURES, TARGET, load_panel


MOCK_PANEL = load_panel().head(200).copy()

MOCK_BUNDLE = {
    "scaler": MagicMock(**{"transform.return_value": np.zeros((1, len(FEATURES)))}),
    "lr": MagicMock(**{"predict_proba.return_value": np.array([[0.3, 0.7]])}),
    "gb": MagicMock(**{"predict_proba.return_value": np.array([[0.2, 0.8]])}),
    "features": FEATURES,
}


@pytest.fixture(scope="module")
def client():
    with (
        patch("api.main.get_panel", return_value=MOCK_PANEL),
        patch("api.main.get_bundle", return_value=MOCK_BUNDLE),
        patch("api.main.get_metrics_data", return_value={"gradient_boosting": {}, "baseline": {}}),
    ):
        from api.main import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_meta_returns_countries_and_years(client):
    res = client.get("/api/meta")
    assert res.status_code == 200
    data = res.json()
    assert "countries" in data
    assert "years" in data
    assert len(data["countries"]) > 0


def test_assess_valid(client):
    country = MOCK_PANEL["country"].iloc[0]
    year = int(MOCK_PANEL["year"].iloc[0])
    res = client.get("/api/assess", params={"country": country, "year": year})
    assert res.status_code == 200
    body = res.json()
    assert "classification" in body
    assert "confidence" in body
    assert "indicators" in body


def test_assess_unknown_country(client):
    res = client.get("/api/assess", params={"country": "Nonexistentland", "year": 2000})
    assert res.status_code == 404


def test_assess_unknown_year(client):
    country = MOCK_PANEL["country"].iloc[0]
    res = client.get("/api/assess", params={"country": country, "year": 2095})
    assert res.status_code == 404


def test_assess_year_out_of_bounds(client):
    res = client.get("/api/assess", params={"country": "USA", "year": 99999})
    assert res.status_code == 422


def test_assess_empty_country(client):
    res = client.get("/api/assess", params={"country": "", "year": 2000})
    assert res.status_code == 422


def test_emitters(client):
    res = client.get("/api/emitters", params={"limit": 5})
    assert res.status_code == 200
    assert "rows" in res.json()


def test_emitters_limit_too_large(client):
    res = client.get("/api/emitters", params={"limit": 999})
    assert res.status_code == 422


def test_timeseries_valid(client):
    country = MOCK_PANEL["country"].iloc[0]
    res = client.get("/api/timeseries", params={"countries": country})
    assert res.status_code == 200
    body = res.json()
    assert "series" in body
    assert len(body["series"]) == 1
    assert "points" in body["series"][0]


def test_timeseries_no_match(client):
    res = client.get("/api/timeseries", params={"countries": "Atlantis"})
    assert res.status_code == 404


def test_timeseries_empty_param(client):
    res = client.get("/api/timeseries", params={"countries": "  ,  "})
    assert res.status_code == 400


def test_metrics(client):
    res = client.get("/api/metrics")
    assert res.status_code == 200
