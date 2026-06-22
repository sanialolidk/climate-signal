import matplotlib

matplotlib.use("Agg")

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

st.set_page_config(
    page_title="Climate Signal | GHG Emissions Classification",
    layout="wide",
)

CSS = """
<style>
    .block-container { padding-top: 0; padding-bottom: 2rem; max-width: 1200px; }
    .site-header {
        background: #1a4480;
        margin: -1rem -1rem 0 -1rem;
        padding: 1.1rem 2rem 0.9rem 2rem;
        border-bottom: 4px solid #0b2d5c;
    }
    .site-header h1 { color: #fff; font-size: 1.45rem; font-weight: 700; margin: 0; }
    .site-header .tagline { color: #c8d9ed; font-size: 0.84rem; margin: 0.3rem 0 0 0; }
    .site-subnav {
        background: #f1f3f5;
        margin: 0 -1rem 1.25rem -1rem;
        padding: 0.55rem 2rem;
        border-bottom: 1px solid #d0d5dc;
        font-size: 0.78rem;
        color: #4a5568;
    }
    .page-title {
        font-size: 1.25rem; font-weight: 700; color: #1b1b1b;
        margin: 0 0 0.35rem 0; padding-bottom: 0.5rem;
        border-bottom: 2px solid #1a4480; display: inline-block;
    }
    .page-desc {
        font-size: 0.9rem; color: #4a5568; line-height: 1.65;
        margin: 0.75rem 0 1.25rem 0; max-width: 54rem;
    }
    .section-heading {
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.08em; color: #1a4480; margin-bottom: 0.65rem;
    }
    .result-kicker {
        font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.07em; color: #6b7280; margin: 0 0 0.35rem 0; text-align: center;
    }
    .result-primary {
        font-size: 1.35rem; font-weight: 700; line-height: 1.25;
        margin: 0; text-align: center;
    }
    .severity-legend { font-size: 0.76rem; color: #6b7280; margin-bottom: 0.65rem; }
    .severity-legend .dot {
        display: inline-block; width: 9px; height: 9px; border-radius: 50%;
        margin: 0 0.2rem 0 0.65rem; vertical-align: middle;
    }
    .indicator-table { width: 100%; border-collapse: collapse; font-size: 0.84rem; }
    .indicator-table th {
        text-align: left; font-size: 0.7rem; font-weight: 700;
        text-transform: uppercase; color: #6b7280;
        padding: 0.45rem 0; border-bottom: 2px solid #d0d5dc;
    }
    .indicator-table td {
        padding: 0.4rem 0; border-bottom: 1px solid #e8eaed; color: #1b1b1b;
    }
    .indicator-table td.val { text-align: right; font-weight: 600; }
    .metric-block {
        background: #fff; border: 1px solid #d0d5dc; border-top: 3px solid #1a4480;
        padding: 0.9rem; text-align: center;
    }
    .metric-block .value { font-size: 1.45rem; font-weight: 700; color: #1a4480; }
    .metric-block .label {
        font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.06em; color: #6b7280; margin-top: 0.15rem;
    }
    .disclaimer {
        background: #f1f3f5; border-left: 4px solid #1a4480;
        padding: 0.9rem 1.1rem; font-size: 0.82rem; color: #4a5568;
        line-height: 1.6; margin-top: 2rem;
    }
    .site-footer {
        margin-top: 2rem; padding: 1.25rem 0; border-top: 2px solid #d0d5dc;
        font-size: 0.78rem; color: #6b7280; line-height: 1.6;
    }
    #MainMenu, footer, header { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

FEATURE_LABELS = {
    "co2_per_capita": "Carbon Dioxide Emissions per Capita (tonnes)",
    "co2_growth_prct": "Annual CO₂ Emissions Growth Rate (%)",
    "cumulative_co2": "Cumulative CO₂ Emissions (Mt)",
    "share_global_co2": "Share of Global CO₂ Emissions (%)",
    "energy_per_capita": "Energy Consumption per Capita",
    "land_use_change_co2": "Land Use Change CO₂ Emissions",
    "methane_per_capita": "Methane Emissions per Capita",
    "nitrous_oxide_per_capita": "Nitrous Oxide Emissions per Capita",
    "co2_per_gdp": "CO₂ Intensity per Unit GDP",
    "population_millions": "Population (millions)",
    "co2_5yr_mean_growth": "Five-Year Mean CO₂ Growth Rate (%)",
    "years_since_1990": "Years Elapsed Since 1990",
}
CHART_COLORS = ["#1a4480", "#2e6ba8", "#4a8bc4", "#6b5b95", "#3d6b5a", "#5c4a3a"]
SEVERITY_COLORS = {"standard": "#2e7d32", "moderate": "#b35c00", "high": "#b50909"}
SEVERITY_LABELS = {
    "standard": "Standard observed impact",
    "moderate": "Elevated observed impact",
    "high": "High observed impact",
}
PAGES = ["Country Assessment", "Emissions Data", "Model Validation"]


@st.cache_resource(show_spinner=False)
def get_bundle():
    return load_bundle()


@st.cache_data(show_spinner=False)
def get_panel():
    return load_panel_cached()


@st.cache_data(show_spinner=False)
def get_metrics():
    return load_metrics()


def format_confidence(probability):
    if probability >= 0.9995:
        return ">99.9%"
    return f"{probability * 100:.1f}%"


def as_bool_flag(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return bool(int(value))


def severity_level(observed_elevated, temp, panel):
    if not observed_elevated:
        return "standard"
    if temp is None or pd.isna(temp):
        return "moderate"
    temps = panel["temperature_change_from_ghg"].dropna()
    if temp >= temps.quantile(0.9):
        return "high"
    return "moderate"


def format_value(val, feat):
    if pd.isna(val):
        return "Not available"
    if feat in ("co2_per_capita", "methane_per_capita", "nitrous_oxide_per_capita"):
        return f"{val:.3f}"
    if feat == "cumulative_co2":
        return f"{val:,.1f}"
    if feat == "population_millions":
        return f"{val:.2f}"
    return f"{val:.3f}"


def site_header():
    st.markdown(
        """
        <div class="site-header">
            <h1>Climate Signal</h1>
            <p class="tagline">GHG Emissions Classification System &mdash; Country-Year Analysis Platform</p>
        </div>
        <div class="site-subnav">
            Data Source: Our World in Data CO₂ Emissions Dataset &nbsp;|&nbsp;
            Coverage: 161 Jurisdictions &nbsp;|&nbsp;
            Methodology: Supervised Machine Learning Classification
        </div>
        """,
        unsafe_allow_html=True,
    )


def site_footer():
    st.markdown(
        """
        <div class="site-footer">
            <strong>Climate Signal</strong> &mdash; Research and screening tool for elevated GHG forcing profiles.<br>
            Data: <strong>Our World in Data</strong> &nbsp;|&nbsp;
            <a href="https://github.com/sanialolidk/climate-signal" style="color:#1a4480;">Documentation</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(country, year, out, row, panel):
    temp = row.get("temperature_change_from_ghg")
    observed = as_bool_flag(row.get(TARGET, 0))
    severity = severity_level(observed, temp, panel)
    accent = SEVERITY_COLORS[severity]
    temp_s = f"{temp:.4f} °C" if pd.notna(temp) else "Not available"
    conf = format_confidence(out["probability"])
    lr_conf = format_confidence(out["lr_probability"])

    st.caption(f"{country} — {year}")
    st.markdown(
        f'<p class="severity-legend">Severity (observed impact): '
        f'<span class="dot" style="background:#2e7d32"></span>Standard '
        f'<span class="dot" style="background:#b35c00"></span>Elevated '
        f'<span class="dot" style="background:#b50909"></span>High '
        f"&mdash; {SEVERITY_LABELS[severity]}</p>",
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        with st.container(border=True):
            st.markdown('<p class="result-kicker">Classification</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-primary" style="color:{accent}">{out["class_name"]}</p>', unsafe_allow_html=True)
            st.caption("Model-assigned emissions profile type")
    with m2:
        with st.container(border=True):
            st.markdown('<p class="result-kicker">GHG Temperature Impact</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-primary" style="color:{accent}">{temp_s}</p>', unsafe_allow_html=True)
            st.caption("Observed change attributed to greenhouse gases")
    with m3:
        with st.container(border=True):
            st.markdown('<p class="result-kicker">Model Confidence</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-primary" style="color:#1a4480">{conf}</p>', unsafe_allow_html=True)
            st.caption("Profile match probability — not a severity score")

    st.info(
        f"**How to read this:** Classification and temperature are observed outcomes. "
        f"Model confidence ({conf}) measures training-set similarity — not danger. "
        f"Baseline (logistic regression): {lr_conf}."
    )


def render_indicators(row):
    rows_html = ""
    for feat in FEATURES:
        label = FEATURE_LABELS.get(feat, feat)
        val = format_value(row.get(feat, 0), feat)
        rows_html += f"<tr><td>{label}</td><td class='val'>{val}</td></tr>"
    observed = "Elevated" if as_bool_flag(row.get(TARGET, 0)) else "Standard"
    rows_html += f"<tr><td>Observed GHG Forcing Classification</td><td class='val'>{observed}</td></tr>"
    st.markdown(
        f'<table class="indicator-table"><tr><th>Indicator</th><th style="text-align:right">Value</th></tr>{rows_html}</table>',
        unsafe_allow_html=True,
    )


def page_assessment(panel):
    st.markdown('<h2 class="page-title">Country Assessment</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-desc">Select a jurisdiction and reporting year to obtain a machine learning '
        "classification of its greenhouse gas emissions profile.</p>",
        unsafe_allow_html=True,
    )

    bundle = get_bundle()
    left, mid, right = st.columns([1, 1.15, 1.25], gap="medium")
    countries = sorted(panel["country"].unique())
    years = sorted(panel["year"].unique())

    with left:
        st.markdown('<p class="section-heading">Query Parameters</p>', unsafe_allow_html=True)
        with st.container(border=True):
            country = st.selectbox(
                "Jurisdiction",
                countries,
                index=countries.index("United States") if "United States" in countries else 0,
                key="jurisdiction",
            )
            year = st.slider(
                "Reporting Year",
                int(min(years)),
                int(max(years)),
                int(max(years)) - 7,
                key="reporting_year",
            )
        st.caption("Training period: years ≤ 2010 · Evaluation period: years > 2010")

    iso = panel.loc[panel["country"] == country, "iso_code"].iloc[0]
    row = country_year_lookup(panel, iso, year)

    with mid:
        st.markdown('<p class="section-heading">Classification Result</p>', unsafe_allow_html=True)
        if row is None:
            st.warning("No record found for this jurisdiction and reporting year.")
        else:
            out = predict_country_year(row, bundle)
            render_result_card(country, year, out, row, panel)

    with right:
        st.markdown('<p class="section-heading">Emissions Indicators</p>', unsafe_allow_html=True)
        if row is None:
            st.caption("Select a valid jurisdiction-year to view indicators.")
        else:
            with st.container(border=True):
                render_indicators(row)


def page_emissions(panel):
    latest = int(panel["year"].max())
    st.markdown('<h2 class="page-title">Emissions Data</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="page-desc">Comparative emissions statistics and temporal trends (through {latest}).</p>',
        unsafe_allow_html=True,
    )

    rank_col, chart_col = st.columns([1, 1.6], gap="large")
    with rank_col:
        st.markdown('<p class="section-heading">Highest Per-Capita Emitters</p>', unsafe_allow_html=True)
        emitters = top_emitters_recent(panel, year=latest, n=15)
        emitters = emitters.rename(columns={
            "country": "Jurisdiction",
            "iso_code": "ISO Code",
            "year": "Year",
            "co2_per_capita": "CO₂ per Capita (t)",
            "temperature_change_from_ghg": "GHG Temperature Change (°C)",
            "elevated_forcing": "Classification",
        })
        emitters["Classification"] = emitters["Classification"].map(
            {1: "Elevated", 0: "Standard", 1.0: "Elevated", 0.0: "Standard", True: "Elevated", False: "Standard"}
        )
        st.dataframe(emitters, use_container_width=True, hide_index=True, height=440)

    with chart_col:
        st.markdown('<p class="section-heading">Per-Capita CO₂ Emissions — Time Series</p>', unsafe_allow_html=True)
        default = [c for c in ["United States", "China", "India", "Germany", "Brazil"] if c in set(panel["country"])]
        picks = st.multiselect(
            "Jurisdictions for Comparison",
            sorted(panel["country"].unique()),
            default=default,
            key="chart_countries",
        )
        if picks:
            fig, ax = plt.subplots(figsize=(9, 4.2))
            for i, name in enumerate(picks):
                grp = panel[panel["country"] == name].sort_values("year")
                ax.plot(grp["year"], grp["co2_per_capita"], label=name, linewidth=2, color=CHART_COLORS[i % len(CHART_COLORS)])
            ax.set_xlabel("Year")
            ax.set_ylabel("Tonnes CO₂ per Capita")
            ax.legend(fontsize=8, ncol=2, loc="upper left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig, clear_figure=True)


def page_validation():
    st.markdown('<h2 class="page-title">Model Validation</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-desc">HistGradientBoosting vs balanced logistic regression on a temporal holdout split.</p>',
        unsafe_allow_html=True,
    )

    metrics = get_metrics()
    if not metrics:
        st.warning("Validation artifacts not found. Run `python main.py` locally.")
        return

    gb = metrics["gradient_boosting"]
    base = metrics["baseline"]

    tiles = st.columns(4)
    for col, (label, val) in zip(
        tiles,
        [
            ("Classification Accuracy", f"{gb['accuracy']:.1%}"),
            ("Macro F1 Score", f"{gb['macro_f1']:.3f}"),
            ("ROC-AUC", f"{gb['roc_auc']:.3f}"),
            ("Holdout Observations", f"{metrics['test_rows']:,}"),
        ],
    ):
        col.markdown(
            f'<div class="metric-block"><div class="value">{val}</div><div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.caption(
        f"{metrics['split']} · {metrics['train_rows']:,} train · "
        f"{metrics['test_rows']:,} test · {metrics['countries']} jurisdictions"
    )

    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.markdown('<p class="section-heading">Comparative Model Performance</p>', unsafe_allow_html=True)
        rows = []
        for name, key in [("Accuracy", "accuracy"), ("Macro F1", "macro_f1"), ("ROC-AUC", "roc_auc")]:
            fmt = ".1%" if key == "accuracy" else ".3f"
            rows.append({
                "Performance Metric": name,
                "Logistic Regression": f"{base[key]:{fmt}}",
                "Gradient Boosting": f"{gb[key]:{fmt}}",
            })
        rows.append({
            "Performance Metric": "Precision (Elevated Class)",
            "Logistic Regression": f"{base['class_1']['precision']:.3f}",
            "Gradient Boosting": f"{gb['class_1']['precision']:.3f}",
        })
        rows.append({
            "Performance Metric": "Recall (Elevated Class)",
            "Logistic Regression": f"{base['class_1']['recall']:.3f}",
            "Gradient Boosting": f"{gb['class_1']['recall']:.3f}",
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.toggle("Display Confusion Matrices", value=False, key="show_cm"):
            i1, i2 = st.columns(2)
            i1.image("plots/cm_baseline.png", caption="Logistic Regression")
            i2.image("plots/cm_main.png", caption="HistGradientBoosting")

    with right:
        st.markdown('<p class="section-heading">Permutation Feature Importance</p>', unsafe_allow_html=True)
        imp = pd.DataFrame(
            list(metrics["feature_importance"].items()),
            columns=["feature", "importance"],
        ).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(imp["feature"], imp["importance"], color="#1a4480", height=0.65)
        ax.set_xlabel("Permutation Importance Score")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig, clear_figure=True)

    st.markdown(
        '<div class="disclaimer"><strong>Disclaimer:</strong> Screening classifier only — not causal inference.</div>',
        unsafe_allow_html=True,
    )


def main():
    site_header()

    try:
        with st.spinner("Loading reference dataset…"):
            panel = get_panel()
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

    page = st.radio("Section", PAGES, horizontal=True, label_visibility="collapsed", key="main_page")

    if page == "Country Assessment":
        page_assessment(panel)
    elif page == "Emissions Data":
        page_emissions(panel)
    else:
        page_validation()

    st.markdown("---")
    st.markdown(
        "[Our World in Data — CO₂](https://ourworldindata.org/co2-emissions) · "
        "[GitHub — climate-signal](https://github.com/sanialolidk/climate-signal)"
    )
    site_footer()


main()