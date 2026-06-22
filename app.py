import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.app_support import (
    country_year_lookup,
    load_bundle,
    load_metrics,
    load_panel_cached,
    predict_country_year,
    top_emitters_recent,
)
from src.data import FEATURES, TARGET

st.set_page_config(page_title="Climate Signal", layout="wide")

CSS = """
<style>
    .block-container { padding-top: 1.5rem; max-width: 1050px; }
    h1 { font-size: 1.6rem; font-weight: 600; letter-spacing: -0.02em; }
    .subtitle { color: #555; font-size: 0.92rem; line-height: 1.5; margin-bottom: 1.5rem; }
    .card {
        background: #fff;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .card-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #888;
        font-weight: 600;
    }
    .card-value { font-size: 1.3rem; font-weight: 600; margin-top: 0.15rem; }
    .result {
        border-left: 4px solid #3d5a4c;
        padding: 1rem 1.2rem;
        background: #fff;
        border: 1px solid #ddd;
        border-left: 4px solid #3d5a4c;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .result.elevated { border-left-color: #a0522d; }
    .footnote { font-size: 0.82rem; color: #666; line-height: 1.5; margin-top: 1.5rem; }
    #MainMenu, footer, header { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

FEATURE_LABELS = {
    "co2_per_capita": "CO₂ per capita (t)",
    "co2_growth_prct": "CO₂ growth (%)",
    "cumulative_co2": "Cumulative CO₂ (Mt)",
    "share_global_co2": "Share of global CO₂ (%)",
    "energy_per_capita": "Energy per capita",
    "land_use_change_co2": "Land-use CO₂",
    "methane_per_capita": "Methane per capita",
    "nitrous_oxide_per_capita": "N₂O per capita",
    "co2_per_gdp": "CO₂ per GDP",
    "population_millions": "Population (M)",
    "co2_5yr_mean_growth": "5-yr mean CO₂ growth (%)",
    "years_since_1990": "Years since 1990",
}

CHART_COLORS = ["#3d5a4c", "#8b6914", "#6b5b95", "#4a6670", "#9b4d3a"]


with st.sidebar:
    st.title("Climate Signal")
    st.caption("OWID emissions panel · country-year classifier")
    page = st.radio("Go to", ["Lookup", "Trends", "Evaluation"], label_visibility="collapsed")
    st.divider()
    st.markdown("[OWID CO₂ data](https://ourworldindata.org/co2-emissions)")
    st.markdown("[GitHub repo](https://github.com/sanialolidk/climate-signal)")
    st.caption("Sania Thankan · Penn State CDS")

panel = load_panel_cached()
bundle = load_bundle()
metrics = load_metrics()

if page == "Lookup":
    st.markdown("# Country lookup")
    st.markdown(
        '<p class="subtitle">Select a country and year. Trained on rows through 2010, '
        "evaluated on 2011+.</p>",
        unsafe_allow_html=True,
    )

    countries = sorted(panel["country"].unique())
    years = sorted(panel["year"].unique())

    c1, c2 = st.columns(2)
    with c1:
        country = st.selectbox(
            "Country",
            countries,
            index=countries.index("United States") if "United States" in countries else 0,
        )
    with c2:
        year = st.selectbox("Year", years, index=max(0, len(years) - 8))

    iso = panel.loc[panel["country"] == country, "iso_code"].iloc[0]
    row = country_year_lookup(panel, iso, year)

    if row is None:
        st.warning("No data for that country-year in the OWID panel.")
    else:
        out = predict_country_year(row, bundle)
        cls = "elevated" if out["label"] == 1 else ""
        st.markdown(
            f'<div class="result {cls}">'
            f'<div class="card-label">Prediction · {out["probability"]:.0%}</div>'
            f'<div class="card-value">{out["class_name"]}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("GB score", f"{out['probability']:.0%}")
        m2.metric("LR baseline", f"{out['lr_probability']:.0%}")
        temp = row.get("temperature_change_from_ghg")
        m3.metric("Observed GHG ΔT", f"{temp:.4f} °C" if pd.notna(temp) else "n/a")

        feat_rows = []
        for f in FEATURES:
            val = row.get(f, 0)
            feat_rows.append({"Feature": FEATURE_LABELS.get(f, f), "Value": val})
        feat_rows.append({"Feature": "Ground truth (elevated)", "Value": bool(row[TARGET])})
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

elif page == "Trends":
    latest = int(panel["year"].max())
    st.markdown("# Emissions trends")
    st.markdown(f'<p class="subtitle">Per-capita CO₂ from the OWID panel (latest: {latest}).</p>', unsafe_allow_html=True)

    emitters = top_emitters_recent(panel, year=latest)
    emitters = emitters.rename(columns={
        "co2_per_capita": "CO₂/capita",
        "temperature_change_from_ghg": "GHG ΔT",
        "elevated_forcing": "Elevated",
    })
    st.dataframe(emitters, use_container_width=True, hide_index=True)

    default = [c for c in ["United States", "China", "India", "Germany", "Brazil"] if c in set(panel["country"])]
    picks = st.multiselect("Countries", sorted(panel["country"].unique()), default=default)

    if picks:
        fig, ax = plt.subplots(figsize=(10, 4))
        for i, name in enumerate(picks):
            grp = panel[panel["country"] == name].sort_values("year")
            ax.plot(grp["year"], grp["co2_per_capita"], label=name, linewidth=2, color=CHART_COLORS[i % len(CHART_COLORS)])
        ax.set_xlabel("Year")
        ax.set_ylabel("Tonnes CO₂ per capita")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig, use_container_width=True)

else:
    st.markdown("# Evaluation")
    st.markdown('<p class="subtitle">HistGradientBoosting vs balanced logistic regression.</p>', unsafe_allow_html=True)

    if not metrics:
        st.info("Run `python main.py` to generate metrics.")
    else:
        gb = metrics["gradient_boosting"]
        base = metrics["baseline"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{gb['accuracy']:.1%}")
        c2.metric("Macro F1", f"{gb['macro_f1']:.3f}")
        c3.metric("ROC-AUC", f"{gb['roc_auc']:.3f}")
        c4.metric("Test rows", f"{metrics['test_rows']:,}")

        st.caption(f"Split: {metrics['split']} · {metrics['countries']} countries")

        comp = pd.DataFrame({
            "Metric": ["Accuracy", "Macro F1", "ROC-AUC", "Precision (elevated)", "Recall (elevated)"],
            "Baseline": [
                f"{base['accuracy']:.1%}", f"{base['macro_f1']:.3f}", f"{base['roc_auc']:.3f}",
                f"{base['class_1']['precision']:.3f}", f"{base['class_1']['recall']:.3f}",
            ],
            "Gradient boosting": [
                f"{gb['accuracy']:.1%}", f"{gb['macro_f1']:.3f}", f"{gb['roc_auc']:.3f}",
                f"{gb['class_1']['precision']:.3f}", f"{gb['class_1']['recall']:.3f}",
            ],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)

        imp = pd.DataFrame(
            list(metrics["feature_importance"].items()),
            columns=["feature", "importance"],
        ).sort_values("importance", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(imp["feature"], imp["importance"], color="#3d5a4c")
        ax.set_xlabel("Permutation importance")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, use_container_width=True)

        if st.checkbox("Confusion matrices"):
            i1, i2 = st.columns(2)
            i1.image("plots/cm_baseline.png", caption="Baseline")
            i2.image("plots/cm_main.png", caption="Gradient boosting")

        st.markdown(
            '<p class="footnote">Panel rows are correlated across time. OWID coverage is thin '
            "before ~1995 for some regions. Screening model only — not causal inference.</p>",
            unsafe_allow_html=True,
        )