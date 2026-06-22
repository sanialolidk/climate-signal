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
    page_title="Climate Signal",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    [data-testid="stSidebar"] {
        background-color: #eeebe5;
        border-right: 1px solid #d4cfc4;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1.75rem; }

    .brand-title {
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: #1c1c1c;
        margin: 0 0 0.35rem 0;
        line-height: 1.2;
    }
    .brand-tag {
        font-size: 0.82rem;
        color: #5c5c5c;
        line-height: 1.5;
        margin: 0;
    }
    .sidebar-links {
        font-size: 0.8rem;
        color: #5c5c5c;
        line-height: 1.7;
    }
    .sidebar-links a { color: #3d5a4c; text-decoration: none; }
    .sidebar-links a:hover { text-decoration: underline; }
    .sidebar-credit {
        font-size: 0.72rem;
        color: #8a8a8a;
        margin-top: 1.5rem;
    }

    .page-head {
        margin-bottom: 1.75rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid #d4cfc4;
    }
    .page-head h1 {
        font-size: 1.65rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem 0;
        color: #1c1c1c;
    }
    .page-head p {
        margin: 0;
        font-size: 0.92rem;
        color: #5c5c5c;
        line-height: 1.55;
        max-width: 640px;
    }

    .section-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8a8a8a;
        margin-bottom: 0.5rem;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #d4cfc4;
        border-radius: 6px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
    }
    .panel-tight { padding: 1rem 1.2rem; }

    .verdict {
        border-left: 4px solid #3d5a4c;
        padding: 1.1rem 1.25rem;
        background: #ffffff;
        border-radius: 0 6px 6px 0;
        border-top: 1px solid #d4cfc4;
        border-right: 1px solid #d4cfc4;
        border-bottom: 1px solid #d4cfc4;
        margin-bottom: 1.25rem;
    }
    .verdict.elevated { border-left-color: #9b4d3a; }
    .verdict-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8a8a8a;
        margin-bottom: 0.25rem;
    }
    .verdict-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1c1c1c;
        margin: 0 0 0.5rem 0;
    }
    .verdict-note {
        font-size: 0.85rem;
        color: #5c5c5c;
        line-height: 1.45;
        margin: 0;
    }

    .prob-bar-wrap {
        margin-top: 0.75rem;
        height: 6px;
        background: #eeebe5;
        border-radius: 3px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        background: #3d5a4c;
        border-radius: 3px;
    }
    .prob-bar-fill.elevated { background: #9b4d3a; }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.25rem;
    }
    .stat-cell {
        background: #ffffff;
        border: 1px solid #d4cfc4;
        border-radius: 6px;
        padding: 0.9rem 1rem;
    }
    .stat-cell .label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8a8a8a;
        margin-bottom: 0.2rem;
    }
    .stat-cell .value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.25rem;
        font-weight: 500;
        color: #1c1c1c;
    }

    .feature-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
    }
    .feature-table th {
        text-align: left;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #8a8a8a;
        padding: 0.5rem 0.75rem 0.5rem 0;
        border-bottom: 1px solid #d4cfc4;
    }
    .feature-table td {
        padding: 0.45rem 0.75rem 0.45rem 0;
        border-bottom: 1px solid #eeebe5;
        color: #1c1c1c;
    }
    .feature-table td.mono {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        text-align: right;
    }
    .feature-table tr:last-child td { border-bottom: none; }

    .compare-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.84rem;
    }
    .compare-table th {
        text-align: left;
        font-weight: 600;
        color: #5c5c5c;
        padding: 0.55rem 0.75rem;
        border-bottom: 2px solid #d4cfc4;
        font-size: 0.78rem;
    }
    .compare-table td {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #eeebe5;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
    }
    .compare-table tr:last-child td { border-bottom: none; }

    .footnote {
        font-size: 0.82rem;
        color: #5c5c5c;
        line-height: 1.55;
        padding: 1rem 0 0 0;
        border-top: 1px solid #d4cfc4;
        margin-top: 1.5rem;
    }

    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d4cfc4;
        border-radius: 6px;
        padding: 0.75rem 1rem;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.68rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #8a8a8a !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.2rem !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)

CHART_COLORS = {
    "United States": "#3d5a4c",
    "China": "#8b6914",
    "India": "#6b5b95",
    "Germany": "#4a6670",
    "Brazil": "#9b4d3a",
    "Russia": "#5c7a8a",
    "Japan": "#7a6b5c",
    "Canada": "#4d6b5a",
}

FEATURE_LABELS = {
    "co2_per_capita": "CO₂ per capita (t)",
    "co2_growth_prct": "CO₂ growth (%)",
    "cumulative_co2": "Cumulative CO₂ (Mt)",
    "share_global_co2": "Share of global CO₂ (%)",
    "energy_per_capita": "Energy per capita",
    "land_use_change_co2": "Land-use change CO₂",
    "methane_per_capita": "Methane per capita",
    "nitrous_oxide_per_capita": "N₂O per capita",
    "co2_per_gdp": "CO₂ per GDP",
    "population_millions": "Population (M)",
    "co2_5yr_mean_growth": "5-yr mean CO₂ growth (%)",
    "years_since_1990": "Years since 1990",
}


def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-head"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def render_verdict(out, row):
    elevated = out["label"] == 1
    cls = "elevated" if elevated else ""
    bar_cls = "elevated" if elevated else ""
    note = (
        "Emissions profile matches high-impact country-years in the training set."
        if elevated
        else "Profile falls within the normal range for this OWID panel."
    )
    temp = row.get("temperature_change_from_ghg")
    temp_str = f"{temp:.4f} °C" if temp is not None and pd.notna(temp) else "n/a"

    st.markdown(
        f"""
        <div class="verdict {cls}">
            <div class="verdict-label">Prediction · {out['probability']:.0%} confidence</div>
            <div class="verdict-title">{out['class_name']}</div>
            <p class="verdict-note">{note}</p>
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill {bar_cls}" style="width: {out['probability'] * 100:.1f}%;"></div>
            </div>
        </div>
        <div class="stat-grid">
            <div class="stat-cell">
                <div class="label">Model score</div>
                <div class="value">{out['probability']:.0%}</div>
            </div>
            <div class="stat-cell">
                <div class="label">Baseline (LR)</div>
                <div class="value">{out['lr_probability']:.0%}</div>
            </div>
            <div class="stat-cell">
                <div class="label">Observed GHG ΔT</div>
                <div class="value">{temp_str}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_table(row):
    rows_html = ""
    for feat in FEATURES:
        val = row.get(feat, 0)
        if pd.isna(val):
            display = "—"
        elif feat in ("co2_per_capita", "methane_per_capita", "nitrous_oxide_per_capita"):
            display = f"{val:.4f}"
        elif feat == "cumulative_co2":
            display = f"{val:,.1f}"
        elif feat == "population_millions":
            display = f"{val:.2f}"
        else:
            display = f"{val:.3f}"
        label = FEATURE_LABELS.get(feat, feat)
        rows_html += f"<tr><td>{label}</td><td class='mono'>{display}</td></tr>"

    truth = "Yes" if row.get(TARGET) else "No"
    rows_html += f"<tr><td>Ground truth (elevated)</td><td class='mono'>{truth}</td></tr>"

    st.markdown('<div class="section-label">Model inputs</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="panel panel-tight"><table class="feature-table">'
        f"<tr><th>Feature</th><th style='text-align:right'>Value</th></tr>{rows_html}</table></div>",
        unsafe_allow_html=True,
    )


def style_chart(ax):
    ax.set_facecolor("#f7f5f2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d4cfc4")
    ax.spines["bottom"].set_color("#d4cfc4")
    ax.tick_params(colors="#5c5c5c", labelsize=9)
    ax.grid(axis="y", linestyle="-", alpha=0.4, color="#d4cfc4")
    ax.set_xlabel("Year", color="#5c5c5c", fontsize=10)
    ax.set_ylabel("Tonnes CO₂ per capita", color="#5c5c5c", fontsize=10)
    leg = ax.legend(loc="upper left", frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("#ffffff")
    leg.get_frame().set_edgecolor("#d4cfc4")


# --- sidebar ---
with st.sidebar:
    st.markdown('<p class="brand-title">Climate Signal</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-tag">Country-year classifier for elevated GHG temperature forcing, '
        "trained on OWID emissions panel data.</p>",
        unsafe_allow_html=True,
    )
    st.divider()
    page = st.radio(
        "Navigate",
        ["Country lookup", "Emissions trends", "Evaluation"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        '<div class="sidebar-links">'
        '<strong>Data</strong><br>'
        '<a href="https://ourworldindata.org/co2-emissions" target="_blank">'
        "Our World in Data — CO₂</a><br><br>"
        '<strong>Code</strong><br>'
        '<a href="https://github.com/sanialolidk/climate-signal" target="_blank">'
        "github.com/sanialolidk/climate-signal</a>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sidebar-credit">Sania Thankan · Penn State CDS · 2026</p>',
        unsafe_allow_html=True,
    )

metrics = load_metrics()
panel = load_panel_cached()
bundle = load_bundle()

if page == "Country lookup":
    page_header(
        "Country lookup",
        "Pick a country and year. The model trains on rows through 2010 and is evaluated on 2011 onward.",
    )

    years = sorted(panel["year"].unique())
    countries = sorted(panel["country"].unique())

    ctrl, result = st.columns([1, 1.4], gap="large")
    with ctrl:
        st.markdown('<div class="section-label">Selection</div>', unsafe_allow_html=True)
        with st.container(border=True):
            country = st.selectbox(
                "Country",
                countries,
                index=countries.index("United States") if "United States" in countries else 0,
            )
            year = st.selectbox("Year", years, index=max(0, len(years) - 8))

    iso = panel.loc[panel["country"] == country, "iso_code"].iloc[0]
    row = country_year_lookup(panel, iso, year)

    with result:
        if row is None:
            st.markdown(
                '<div class="panel"><p style="margin:0;color:#5c5c5c;font-size:0.9rem;">'
                "No row for that country-year — OWID has coverage gaps in some regions."
                "</p></div>",
                unsafe_allow_html=True,
            )
        else:
            out = predict_country_year(row, bundle)
            render_verdict(out, row)
            render_feature_table(row)

elif page == "Emissions trends":
    latest = int(panel["year"].max())
    page_header(
        "Emissions trends",
        f"Per-capita CO₂ rankings and time series from the OWID panel (latest year: {latest}).",
    )

    st.markdown('<div class="section-label">Top emitters</div>', unsafe_allow_html=True)
    emitters = top_emitters_recent(panel, year=latest)
    emitters = emitters.rename(
        columns={
            "co2_per_capita": "CO₂/capita (t)",
            "temperature_change_from_ghg": "GHG ΔT (°C)",
            "elevated_forcing": "Elevated",
        }
    )
    emitters["Elevated"] = emitters["Elevated"].map({True: "Yes", False: "No"})
    st.dataframe(
        emitters,
        use_container_width=True,
        hide_index=True,
        column_config={
            "CO₂/capita (t)": st.column_config.NumberColumn(format="%.2f"),
            "GHG ΔT (°C)": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.markdown('<div class="section-label" style="margin-top:1.5rem">Time series</div>', unsafe_allow_html=True)
    default = [c for c in ["United States", "China", "India", "Germany", "Brazil"] if c in set(panel["country"])]
    picks = st.multiselect("Countries to overlay", sorted(panel["country"].unique()), default=default, label_visibility="collapsed")

    if picks:
        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#f7f5f2")
        for name in picks:
            grp = panel[panel["country"] == name].sort_values("year")
            color = CHART_COLORS.get(name, "#5c5c5c")
            ax.plot(grp["year"], grp["co2_per_capita"], label=name, linewidth=2, color=color)
        style_chart(ax)
        st.pyplot(fig, use_container_width=True)

else:
    page_header(
        "Model evaluation",
        "Histogram-based gradient boosting vs. balanced logistic regression on a temporal split.",
    )

    if not metrics:
        st.info("Run `python main.py` locally to generate metrics and plots.")
    else:
        gb = metrics["gradient_boosting"]
        base = metrics["baseline"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{gb['accuracy']:.1%}")
        c2.metric("Macro F1", f"{gb['macro_f1']:.3f}")
        c3.metric("ROC-AUC", f"{gb['roc_auc']:.3f}")
        c4.metric("Test rows", f"{metrics['test_rows']:,}")

        st.markdown(
            f'<p style="font-size:0.85rem;color:#5c5c5c;margin:1rem 0 1.25rem 0;">'
            f"<strong>Split:</strong> {metrics['split']} · "
            f"{metrics['train_rows']:,} train / {metrics['test_rows']:,} test · "
            f"{metrics['countries']} countries<br>"
            "Target = top quartile of <code>temperature_change_from_ghg</code> across the panel."
            "</p>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label">Model comparison</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="panel panel-tight">
            <table class="compare-table">
                <tr>
                    <th>Metric</th>
                    <th>Baseline (LR)</th>
                    <th>Gradient boosting</th>
                </tr>
                <tr>
                    <td style="font-family:inherit;color:#5c5c5c">Accuracy</td>
                    <td>{base['accuracy']:.1%}</td>
                    <td>{gb['accuracy']:.1%}</td>
                </tr>
                <tr>
                    <td style="font-family:inherit;color:#5c5c5c">Macro F1</td>
                    <td>{base['macro_f1']:.3f}</td>
                    <td>{gb['macro_f1']:.3f}</td>
                </tr>
                <tr>
                    <td style="font-family:inherit;color:#5c5c5c">ROC-AUC</td>
                    <td>{base['roc_auc']:.3f}</td>
                    <td>{gb['roc_auc']:.3f}</td>
                </tr>
                <tr>
                    <td style="font-family:inherit;color:#5c5c5c">Precision (elevated)</td>
                    <td>{base['class_1']['precision']:.3f}</td>
                    <td>{gb['class_1']['precision']:.3f}</td>
                </tr>
                <tr>
                    <td style="font-family:inherit;color:#5c5c5c">Recall (elevated)</td>
                    <td>{base['class_1']['recall']:.3f}</td>
                    <td>{gb['class_1']['recall']:.3f}</td>
                </tr>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="section-label" style="margin-top:1.5rem">Feature importance</div>', unsafe_allow_html=True)
        imp = pd.DataFrame(
            list(metrics["feature_importance"].items()),
            columns=["feature", "importance"],
        ).sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor("#f7f5f2")
        vals = imp.iloc[:, -1]
        ax.barh(imp["feature"], vals, color="#3d5a4c", height=0.65)
        ax.set_facecolor("#f7f5f2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#d4cfc4")
        ax.spines["bottom"].set_color("#d4cfc4")
        ax.tick_params(colors="#5c5c5c", labelsize=9)
        ax.set_xlabel("Permutation importance", color="#5c5c5c", fontsize=10)
        st.pyplot(fig, use_container_width=True)

        if st.checkbox("Show confusion matrices", value=False):
            img1, img2 = st.columns(2)
            with img1:
                st.image("plots/cm_baseline.png", caption="Baseline")
            with img2:
                st.image("plots/cm_main.png", caption="Gradient boosting")

        st.markdown(
            '<p class="footnote">'
            "Limitations: country-year rows are correlated across time, OWID coverage is thin "
            "for some regions before ~1995, and this is a screening model — not causal inference."
            "</p>",
            unsafe_allow_html=True,
        )