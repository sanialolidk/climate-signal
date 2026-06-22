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
    page_icon="◎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding-top: 0; padding-bottom: 2.5rem; max-width: 1180px; }

    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }

    .topbar {
        background: #0f1923;
        margin: -1rem -1rem 2rem -1rem;
        padding: 0.85rem 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 3px solid #d4a054;
    }
    .topbar-brand { color: #f0ebe3; font-size: 1.05rem; font-weight: 600; letter-spacing: 0.04em; }
    .topbar-brand span { color: #d4a054; }
    .topbar-meta { color: #7a8a96; font-size: 0.75rem; }

    .page-lead {
        font-size: 0.9rem;
        color: #4a5568;
        line-height: 1.55;
        margin: -0.5rem 0 1.5rem 0;
        max-width: 52rem;
    }

    .filter-box {
        background: #f4f1ec;
        border: 1px solid #d8d2c8;
        border-radius: 8px;
        padding: 1.25rem 1.1rem;
    }
    .filter-box h3 {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #6b7280;
        margin: 0 0 1rem 0;
        font-weight: 600;
    }

    .score-panel {
        background: #0f1923;
        color: #f0ebe3;
        border-radius: 10px;
        padding: 1.75rem 1.5rem;
        text-align: center;
        min-height: 220px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .score-panel.elevated { border: 2px solid #d4a054; }
    .score-panel.normal { border: 2px solid #3d6b5a; }
    .score-big {
        font-family: 'JetBrains Mono', monospace;
        font-size: 3rem;
        font-weight: 500;
        line-height: 1;
        color: #d4a054;
    }
    .score-panel.normal .score-big { color: #6db89a; }
    .score-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: #7a8a96;
        margin-bottom: 0.5rem;
    }
    .score-verdict {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.75rem;
        color: #f0ebe3;
    }
    .score-sub {
        font-size: 0.8rem;
        color: #7a8a96;
        margin-top: 0.35rem;
    }

    .stat-pill {
        display: inline-block;
        background: #fff;
        border: 1px solid #d8d2c8;
        border-radius: 20px;
        padding: 0.35rem 0.85rem;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .stat-pill strong {
        font-family: 'JetBrains Mono', monospace;
        color: #0f1923;
    }

    .section-tag {
        display: inline-block;
        background: #0f1923;
        color: #d4a054;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.25rem 0.6rem;
        border-radius: 3px;
        margin-bottom: 0.75rem;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.35rem 1rem;
        font-size: 0.82rem;
    }
    .feature-grid .name { color: #6b7280; }
    .feature-grid .val {
        font-family: 'JetBrains Mono', monospace;
        text-align: right;
        color: #0f1923;
    }

    .metric-tile {
        background: #fff;
        border: 1px solid #d8d2c8;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-tile .n {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 500;
        color: #0f1923;
    }
    .metric-tile .l {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #6b7280;
        margin-top: 0.2rem;
    }

    .footnote {
        font-size: 0.8rem;
        color: #6b7280;
        line-height: 1.55;
        border-top: 1px solid #d8d2c8;
        padding-top: 1rem;
        margin-top: 2rem;
    }

    div[data-testid="stSegmentedControl"] {
        background: #1a2835;
        border-radius: 8px;
        padding: 4px;
    }
    div[data-testid="stSegmentedControl"] button {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: #d4a054 !important;
        color: #0f1923 !important;
    }

    .stSelectbox label, .stMultiSelect label, .stSlider label {
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
        color: #6b7280 !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

FEATURE_LABELS = {
    "co2_per_capita": "CO₂ / capita",
    "co2_growth_prct": "CO₂ growth",
    "cumulative_co2": "Cumulative CO₂",
    "share_global_co2": "Global share",
    "energy_per_capita": "Energy / capita",
    "land_use_change_co2": "Land-use CO₂",
    "methane_per_capita": "Methane / capita",
    "nitrous_oxide_per_capita": "N₂O / capita",
    "co2_per_gdp": "CO₂ / GDP",
    "population_millions": "Population",
    "co2_5yr_mean_growth": "5-yr CO₂ trend",
    "years_since_1990": "Yrs since 1990",
}

CHART_COLORS = ["#d4a054", "#3d6b5a", "#5b7c99", "#9b6b5c", "#6b5b8a", "#4a8a7a"]


def topbar():
    st.markdown(
        """
        <div class="topbar">
            <div class="topbar-brand">climate <span>signal</span></div>
            <div class="topbar-meta">OWID panel · 161 countries · Penn State CDS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_feature(val, feat):
    if pd.isna(val):
        return "—"
    if feat in ("co2_per_capita", "methane_per_capita", "nitrous_oxide_per_capita"):
        return f"{val:.3f}"
    if feat == "cumulative_co2":
        return f"{val:,.0f}"
    if feat == "population_millions":
        return f"{val:.1f}M"
    return f"{val:.2f}"


def render_feature_grid(row):
    cells = ""
    for feat in FEATURES:
        val = row.get(feat, 0)
        cells += (
            f'<div class="name">{FEATURE_LABELS.get(feat, feat)}</div>'
            f'<div class="val">{format_feature(val, feat)}</div>'
        )
    truth = "yes" if row.get(TARGET) else "no"
    cells += '<div class="name">Ground truth</div><div class="val">' + truth + "</div>"
    st.markdown(f'<div class="feature-grid">{cells}</div>', unsafe_allow_html=True)


topbar()

page = st.segmented_control(
    "Section",
    ["Explore", "Panel data", "Model metrics"],
    default="Explore",
    label_visibility="collapsed",
)

panel = load_panel_cached()
bundle = load_bundle()
metrics = load_metrics()

if page == "Explore":
    st.markdown(
        '<p class="page-lead">Pick any country-year from the OWID emissions panel. '
        "The classifier flags profiles that match elevated GHG temperature forcing "
        "(top quartile in the training set).</p>",
        unsafe_allow_html=True,
    )

    left, mid, right = st.columns([1, 1.1, 1.2], gap="medium")

    countries = sorted(panel["country"].unique())
    years = sorted(panel["year"].unique())

    with left:
        st.markdown('<div class="filter-box"><h3>Parameters</h3>', unsafe_allow_html=True)
        country = st.selectbox(
            "Country",
            countries,
            index=countries.index("United States") if "United States" in countries else 0,
            label_visibility="visible",
        )
        year = st.slider("Year", int(min(years)), int(max(years)), int(max(years)) - 7)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.78rem;color:#6b7280;margin-top:0.75rem;">'
            "Train ≤ 2010 · test 2011+</p>",
            unsafe_allow_html=True,
        )

    iso = panel.loc[panel["country"] == country, "iso_code"].iloc[0]
    row = country_year_lookup(panel, iso, year)

    with mid:
        if row is None:
            st.markdown(
                '<div class="score-panel"><div class="score-label">No data</div>'
                '<div class="score-verdict">Coverage gap</div>'
                '<div class="score-sub">OWID has no row for this country-year.</div></div>',
                unsafe_allow_html=True,
            )
        else:
            out = predict_country_year(row, bundle)
            elevated = out["label"] == 1
            panel_cls = "elevated" if elevated else "normal"
            st.markdown(
                f'<div class="score-panel {panel_cls}">'
                f'<div class="score-label">{country} · {year}</div>'
                f'<div class="score-big">{out["probability"]:.0%}</div>'
                f'<div class="score-verdict">{out["class_name"]}</div>'
                f'<div class="score-sub">Gradient boosting probability</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            temp = row.get("temperature_change_from_ghg")
            temp_s = f"{temp:.4f} °C" if pd.notna(temp) else "n/a"
            st.markdown(
                f'<span class="stat-pill">LR baseline <strong>{out["lr_probability"]:.0%}</strong></span>'
                f'<span class="stat-pill">GHG ΔT <strong>{temp_s}</strong></span>',
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<span class="section-tag">Input features</span>', unsafe_allow_html=True)
        if row is None:
            st.caption("Select a valid country-year to see features.")
        else:
            with st.container(border=True):
                render_feature_grid(row)

elif page == "Panel data":
    latest = int(panel["year"].max())
    st.markdown(
        f'<p class="page-lead">Rankings and trajectories from the committed OWID panel '
        f"(through {latest}).</p>",
        unsafe_allow_html=True,
    )

    rank_col, chart_col = st.columns([1, 1.6], gap="large")

    with rank_col:
        st.markdown('<span class="section-tag">Top per-capita emitters</span>', unsafe_allow_html=True)
        emitters = top_emitters_recent(panel, year=latest, n=15)
        emitters = emitters.rename(columns={
            "co2_per_capita": "t/cap",
            "temperature_change_from_ghg": "ΔT",
            "elevated_forcing": "flag",
        })
        emitters["flag"] = emitters["flag"].map({1: "●", 0: "○", True: "●", False: "○"})
        st.dataframe(emitters, use_container_width=True, hide_index=True, height=420)

    with chart_col:
        st.markdown('<span class="section-tag">CO₂ per capita over time</span>', unsafe_allow_html=True)
        default = [c for c in ["United States", "China", "India", "Germany", "Brazil"] if c in set(panel["country"])]
        picks = st.pills("Compare", sorted(panel["country"].unique()), default=default, selection_mode="multi")

        if picks:
            fig, ax = plt.subplots(figsize=(9, 4.2))
            fig.patch.set_facecolor("#f4f1ec")
            for i, name in enumerate(picks):
                grp = panel[panel["country"] == name].sort_values("year")
                ax.plot(
                    grp["year"], grp["co2_per_capita"],
                    label=name, linewidth=2.2,
                    color=CHART_COLORS[i % len(CHART_COLORS)],
                )
            ax.set_facecolor("#f4f1ec")
            ax.set_xlabel("Year", fontsize=10, color="#4a5568")
            ax.set_ylabel("t CO₂ / capita", fontsize=10, color="#4a5568")
            ax.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#d8d2c8")
            ax.spines["bottom"].set_color("#d8d2c8")
            ax.tick_params(colors="#6b7280", labelsize=8)
            ax.grid(axis="y", color="#d8d2c8", alpha=0.6)
            st.pyplot(fig, use_container_width=True)
        else:
            st.caption("Select at least one country.")

else:
    st.markdown(
        '<p class="page-lead">Holdout evaluation on country-years after 2010. '
        "Histogram gradient boosting vs. balanced logistic regression.</p>",
        unsafe_allow_html=True,
    )

    if not metrics:
        st.info("Run `python main.py` locally to generate metrics.")
    else:
        gb = metrics["gradient_boosting"]
        base = metrics["baseline"]

        tiles = st.columns(4)
        for col, (label, val) in zip(
            tiles,
            [
                ("Accuracy", f"{gb['accuracy']:.1%}"),
                ("Macro F1", f"{gb['macro_f1']:.3f}"),
                ("ROC-AUC", f"{gb['roc_auc']:.3f}"),
                ("Test rows", f"{metrics['test_rows']:,}"),
            ],
        ):
            col.markdown(
                f'<div class="metric-tile"><div class="n">{val}</div><div class="l">{label}</div></div>',
                unsafe_allow_html=True,
            )

        st.caption(f"{metrics['split']} · {metrics['countries']} countries · {metrics['train_rows']:,} train rows")

        left, right = st.columns([1, 1.3], gap="large")

        with left:
            st.markdown('<span class="section-tag">Model comparison</span>', unsafe_allow_html=True)
            rows = []
            for name, key in [
                ("Accuracy", "accuracy"), ("Macro F1", "macro_f1"), ("ROC-AUC", "roc_auc"),
            ]:
                b = base[key]
                g = gb[key]
                fmt = ".1%" if key == "accuracy" else ".3f"
                rows.append({
                    "": name,
                    "Logistic": f"{b:{fmt}}",
                    "GBM": f"{g:{fmt}}",
                })
            rows.append({"": "Prec. (elevated)", "Logistic": f"{base['class_1']['precision']:.3f}", "GBM": f"{gb['class_1']['precision']:.3f}"})
            rows.append({"": "Recall (elevated)", "Logistic": f"{base['class_1']['recall']:.3f}", "GBM": f"{gb['class_1']['recall']:.3f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            show_cm = st.toggle("Show confusion matrices", value=False)
            if show_cm:
                i1, i2 = st.columns(2)
                i1.image("plots/cm_baseline.png", caption="Logistic")
                i2.image("plots/cm_main.png", caption="GBM")

        with right:
            st.markdown('<span class="section-tag">Feature importance</span>', unsafe_allow_html=True)
            imp = pd.DataFrame(
                list(metrics["feature_importance"].items()),
                columns=["feature", "importance"],
            ).sort_values("importance", ascending=True)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            fig.patch.set_facecolor("#f4f1ec")
            ax.barh(imp["feature"], imp["importance"], color="#d4a054", height=0.6)
            ax.set_facecolor("#f4f1ec")
            ax.set_xlabel("Permutation importance", fontsize=10, color="#4a5568")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#d8d2c8")
            ax.spines["bottom"].set_color("#d8d2c8")
            ax.tick_params(colors="#6b7280", labelsize=8)
            st.pyplot(fig, use_container_width=True)

        st.markdown(
            '<p class="footnote">'
            "Panel rows correlate across time. OWID coverage is sparse before ~1995 in some regions. "
            "This is a screening classifier, not a causal model. "
            '<a href="https://github.com/sanialolidk/climate-signal" style="color:#0f1923;">Source on GitHub</a>'
            "</p>",
            unsafe_allow_html=True,
        )

with st.container():
    st.markdown("---")
    link1, link2, link3 = st.columns(3)
    link1.link_button("OWID dataset", "https://ourworldindata.org/co2-emissions", use_container_width=True)
    link2.link_button("GitHub repo", "https://github.com/sanialolidk/climate-signal", use_container_width=True)
    link3.link_button("About", "https://github.com/sanialolidk/climate-signal#readme", use_container_width=True)