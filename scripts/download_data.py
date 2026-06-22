"""Fetch OWID CO2 CSV into data/. Run once before training."""
import urllib.request
from pathlib import Path

URL = "https://nyc3.digitaloceanspaces.com/owid-public/data/co2/owid-co2-data.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "owid-co2-data.csv"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, OUT)
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()