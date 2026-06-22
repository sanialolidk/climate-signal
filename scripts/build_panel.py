"""Download OWID CSV and write data/panel.csv. Run locally before retraining."""
import urllib.request
from pathlib import Path

from src.data import PANEL_PATH, build_panel_from_owid

OWID_URL = "https://nyc3.digitaloceanspaces.com/owid-public/data/co2/owid-co2-data.csv"
RAW = Path(__file__).resolve().parent.parent / "data" / "owid-co2-data.csv"


def main():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        print(f"Downloading {OWID_URL}")
        urllib.request.urlretrieve(OWID_URL, RAW)
    panel = build_panel_from_owid(RAW)
    panel.to_csv(PANEL_PATH, index=False)
    print(f"Wrote {len(panel)} rows to {PANEL_PATH}")


if __name__ == "__main__":
    main()