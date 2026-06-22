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
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Source Sans 3', 'Segoe UI', sans-serif; }
    .block-container { padding-top: 0; padding-bottom: 2rem; max-width: 1200px; }

    [data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }

    .site-header {
        background: #1a4480;
        margin: -1rem -1rem 0 -1rem;
        padding: 1.1rem 2rem 0.9rem 2rem;
        border-bottom: 4px solid #0b2d5c;
    }
    .site-header h1 {
        color: #ffffff;
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.01em;
    }
    .site-header .tagline {
        color: #c8d9ed;
        font-size: 0.84rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }
    .site-subnav {
        background: #f1f3f5;
        margin: 0 -1rem 1.75rem -1rem;
        padding: 0.55rem 2rem;
        border-bottom: 1px solid #d0d5dc;
        font-size: 0.78rem;
        color: #4a5568;
    }

    .page-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1b1b1b;
        margin: 0 0 0.35rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1a4480;
        display: inline-block;
    }
    .page-desc {
        font-size: 0.9rem;
        color: #4a5568;
        line-height: 1.65;
        margin: 0.75rem 0 1.5rem 0;
        max-width: 54rem;
    }

    .panel-box {
        background: #ffffff;
        border: 1px solid #d0d5dc;
        border-top: 3px solid #1a4480;
        padding: 1.2rem 1.3rem;
    }
    .panel-box h4 {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #1a4480;
        margin: 0 0 1rem 0;
    }

    .result-box {
        background: #ffffff;
        border: 1px solid #d0d5dc;
        padding: 1.5rem;
        text-align: center;
        min-height: 230px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .result-box.elevated { border-top: 4px solid #b50909; }
    .result-box.standard { border-top: 4px solid #1a4480; }
    .result-jurisdiction {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7280;
        margin-bottom: 0.4rem;
    }
    .result-score {
        font-size: 2.75rem;
        font-weight: 700;
        color: #1a4480;
        line-height: 1;
    }
    .result-box.elevated .result-score { color: #b50909; }
    .result-classification {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1b1b1b;
        margin-top: 0.65rem;
    }
    .result-method {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.35rem;
    }

    .meta-row {
        margin-top: 0.85rem;
        font-size: 0.82rem;
        color: #4a5568;
    }
    .meta-row span {
        display: inline-block;
        margin-right: 1.25rem;
        padding-right: 1.25rem;
        border-right: 1px solid #d0d5dc;
    }
    .meta-row span:last-child { border-right: none; }
    .meta-row strong { color: #1b1b1b; }

    .section-heading {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #1a4480;
        margin-bottom: 0.65rem;
    }

    .indicator-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
    }
    .indicator-table th {
        text-align: left;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
        padding: 0.45rem 0;
        border-bottom: 2px solid #d0d5dc;
    }
    .indicator-table td {
        padding: 0.4rem 0;
        border-bottom: 1px solid #e8eaed;
        color: #1b1b1b;
    }
    .indicator-table td.val { text-align: right; font-weight: 600; }
    .indicator-table tr:last-child td { border-bottom: none; }

    .metric-block {
        background: #ffffff;
        border: 1px solid #d0d5dc;
        border-top: 3px solid #1a4480;
        padding: 0.9rem;
        text-align: center;
    }
    .metric-block .value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #1a4480;
    }
    .metric-block .label {
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
        margin-top: 0.15rem;
    }

    .disclaimer {
        background: #f1f3f5;
        border-left: 4px solid #1a4480;
        padding: 0.9rem 1.1rem;
        font-size: 0.82rem;
        color: #4a5568;
        line-height: 1.6;
        margin-top: 2rem;
    }
    .site-footer {
        margin-top: 2.5rem;
        padding: 1.25rem 0;
        border-top: 2px solid #d0d5dc;
        font-size: 0.78rem;
        color: #6b7280;
        line-height: 1.6;
    }
    .site-footer strong { color: #1b1b1b; }

    div[data-testid="stSegmentedControl"] {
        background: #e8eaed;
        border: 1px solid #d0d5dc;
        border-radius: 4px;
        padding: 3px;
    }
    div[data-testid="stSegmentedControl"] button {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.04em !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: #1a4480 !important;
        color: #ffffff !important;
    }

    .stSelectbox label, .stSlider label {
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #4a5568 !important;
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


def render_indicators(row):
    rows_html = ""
    for feat in FEATURES:
        label = FEATURE_LABELS.get(feat, feat)
        val = format_value(row.get(feat, 0), feat)
        rows_html += f"<tr><td>{label}</td><td class='val'>{val}</td></tr>"
    observed = "Elevated" if row.get(TARGET) else "Standard"
    rows_html += f"<tr><td>Observed GHG Forcing Classification</td><td class='val'>{observed}</td></tr>"
    st.markdown(
        f'<table class="indicator-table"><tr><th>Indicator</th><th style="text-align:right">Value</th></tr>{rows_html}</table>',
        unsafe_allow_html=True,
    )


def site_footer():
    st.markdown(
        """
        <div class="site-footer">
            <strong>Climate Signal</strong> &mdash; An analytical tool for identifying country-year
            emissions profiles associated with elevated greenhouse gas temperature forcing.<br>
            Data provided by <strong>Our World in Data</strong>. Model outputs are intended for
            research and screening purposes only and do not constitute policy recommendations.<br>
            &copy; 2026 &nbsp;|&nbsp;
            <a href="https://github.com/sanialolidk/climate-signal" style="color:#1a4480;">Documentation &amp; Source Code</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


site_header()

page = st.segmented_control(
    "Navigation",
    ["Country Assessment", "Emissions Data", "Model Validation"],
    default="Country Assessment",
    label_visibility="collapsed",
)

panel = load_panel_cached()
bundle = load_bundle()
metrics = load_metrics()

if page == "Country Assessment":
    st.markdown('<h2 class="page-title">Country Assessment</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-desc">Select a jurisdiction and reporting year to obtain a machine learning '
        "classification of its greenhouse gas emissions profile. The model identifies country-year "
        "observations exhibiting characteristics consistent with elevated GHG-attributed temperature "
        "forcing, defined as the upper quartile of the reference panel.</p>",
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1, 1.15, 1.25], gap="medium")
    countries = sorted(panel["country"].unique())
    years = sorted(panel["year"].unique())

    with left:
        st.markdown('<div class="panel-box"><h4>Query Parameters</h4>', unsafe_allow_html=True)
        country = st.selectbox("Jurisdiction", countries, index=countries.index("United States") if "United States" in countries else 0)
        year = st.slider("Reporting Year", int(min(years)), int(max(years)), int(max(years)) - 7)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.78rem;color:#6b7280;margin-top:0.75rem;line-height:1.5;">'
            "<strong>Training period:</strong> years &le; 2010<br>"
            "<strong>Evaluation period:</strong> years &gt; 2010</p>",
            unsafe_allow_html=True,
        )

    iso = panel.loc[panel["country"] == country, "iso_code"].iloc[0]
    row = country_year_lookup(panel, iso, year)

    with mid:
        st.markdown('<p class="section-heading">Classification Result</p>', unsafe_allow_html=True)
        if row is None:
            st.markdown(
                '<div class="result-box">'
                '<div class="result-jurisdiction">Data Unavailable</div>'
                '<div class="result-classification">No Record Found</div>'
                '<div class="result-method">The selected jurisdiction and reporting year '
                "are not present in the reference dataset.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            out = predict_country_year(row, bundle)
            elevated = out["label"] == 1
            box_cls = "elevated" if elevated else "standard"
            st.markdown(
                f'<div class="result-box {box_cls}">'
                f'<div class="result-jurisdiction">{country} &mdash; {year}</div>'
                f'<div class="result-score">{out["probability"]:.1%}</div>'
                f'<div class="result-classification">{out["class_name"]}</div>'
                f'<div class="result-method">HistGradientBoosting Classifier &mdash; Estimated Probability</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            temp = row.get("temperature_change_from_ghg")
            temp_s = f"{temp:.4f} &deg;C" if pd.notna(temp) else "Not available"
            st.markdown(
                f'<div class="meta-row">'
                f"<span>Baseline Model (Logistic Regression): <strong>{out['lr_probability']:.1%}</strong></span>"
                f"<span>Observed GHG Temperature Change: <strong>{temp_s}</strong></span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<p class="section-heading">Emissions Indicators</p>', unsafe_allow_html=True)
        if row is None:
            st.markdown(
                '<div class="panel-box"><p style="margin:0;font-size:0.85rem;color:#6b7280;">'
                "Indicator values will be displayed upon selection of a valid jurisdiction-year record."
                "</p></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="panel-box">', unsafe_allow_html=True)
            render_indicators(row)
            st.markdown("</div>", unsafe_allow_html=True)

elif page == "Emissions Data":
    latest = int(panel["year"].max())
    st.markdown('<h2 class="page-title">Emissions Data</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="page-desc">Comparative emissions statistics and temporal trends derived from the '
        f"Our World in Data country-year panel. Reference period extends through {latest}.</p>",
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
            {1: "Elevated", 0: "Standard", True: "Elevated", False: "Standard"}
        )
        st.dataframe(emitters, use_container_width=True, hide_index=True, height=440)

    with chart_col:
        st.markdown('<p class="section-heading">Per-Capita CO₂ Emissions &mdash; Time Series</p>', unsafe_allow_html=True)
        default = [c for c in ["United States", "China", "India", "Germany", "Brazil"] if c in set(panel["country"])]
        picks = st.pills("Jurisdictions for Comparison", sorted(panel["country"].unique()), default=default, selection_mode="multi")

        if picks:
            fig, ax = plt.subplots(figsize=(9, 4.2))
            fig.patch.set_facecolor("#f8f9fa")
            for i, name in enumerate(picks):
                grp = panel[panel["country"] == name].sort_values("year")
                ax.plot(grp["year"], grp["co2_per_capita"], label=name, linewidth=2, color=CHART_COLORS[i % len(CHART_COLORS)])
            ax.set_facecolor("#f8f9fa")
            ax.set_xlabel("Year", fontsize=10, color="#4a5568")
            ax.set_ylabel("Tonnes CO₂ per Capita", fontsize=10, color="#4a5568")
            ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.95, edgecolor="#d0d5dc")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#d0d5dc")
            ax.spines["bottom"].set_color("#d0d5dc")
            ax.tick_params(colors="#6b7280", labelsize=8)
            ax.grid(axis="y", color="#d0d5dc", alpha=0.7)
            st.pyplot(fig, use_container_width=True)
        else:
            st.caption("Please select one or more jurisdictions to display the time series.")

else:
    st.markdown('<h2 class="page-title">Model Validation</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-desc">Performance metrics for the primary classification model '
        "(HistGradientBoosting) evaluated against a balanced logistic regression baseline. "
        "Models are assessed on a temporal holdout split to prevent data leakage across reporting years.</p>",
        unsafe_allow_html=True,
    )

    if not metrics:
        st.warning("Model validation artifacts are not available. Execute the training pipeline to generate metrics.")
    else:
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

        st.markdown(
            f'<p style="font-size:0.82rem;color:#4a5568;margin:1rem 0 1.5rem 0;">'
            f"<strong>Data partition:</strong> {metrics['split']} &nbsp;|&nbsp; "
            f"<strong>Training observations:</strong> {metrics['train_rows']:,} &nbsp;|&nbsp; "
            f"<strong>Jurisdictions:</strong> {metrics['countries']}</p>",
            unsafe_allow_html=True,
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

            if st.toggle("Display Confusion Matrices", value=False):
                i1, i2 = st.columns(2)
                i1.image("plots/cm_baseline.png", caption="Logistic Regression (Baseline)")
                i2.image("plots/cm_main.png", caption="HistGradientBoosting (Primary)")

        with right:
            st.markdown('<p class="section-heading">Permutation Feature Importance</p>', unsafe_allow_html=True)
            imp = pd.DataFrame(
                list(metrics["feature_importance"].items()),
                columns=["feature", "importance"],
            ).sort_values("importance", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            fig.patch.set_facecolor("#f8f9fa")
            ax.barh(imp["feature"], imp["importance"], color="#1a4480", height=0.65)
            ax.set_facecolor("#f8f9fa")
            ax.set_xlabel("Permutation Importance Score", fontsize=10, color="#4a5568")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#d0d5dc")
            ax.spines["bottom"].set_color("#d0d5dc")
            ax.tick_params(colors="#6b7280", labelsize=8)
            st.pyplot(fig, use_container_width=True)

        st.markdown(
            '<div class="disclaimer"><strong>Limitations and Disclaimer:</strong> '
            "Country-year observations within the panel exhibit temporal correlation that may affect "
            "generalization. Data coverage prior to 1995 is limited for certain jurisdictions. "
            "Model outputs represent statistical classifications based on historical emissions patterns "
            "and should not be interpreted as causal attribution or used as the sole basis for policy decisions."
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")
link1, link2, link3 = st.columns(3)
link1.link_button("Our World in Data — CO₂ Emissions", "https://ourworldindata.org/co2-emissions", use_container_width=True)
link2.link_button("Technical Documentation", "https://github.com/sanialolidk/climate-signal", use_container_width=True)
link3.link_button("Methodology & Source Code", "https://github.com/sanialolidk/climate-signal#readme", use_container_width=True)

site_footer()