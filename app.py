import matplotlib.pyplot as plt
import streamlit as st

from src.app_support import (
    country_year_lookup,
    load_bundle,
    load_metrics,
    load_panel_cached,
    predict_country_year,
    top_emitters_recent,
)
from src.data import TARGET

st.set_page_config(page_title="Climate Signal", layout="wide")

# sidebar — kept simple on purpose
with st.sidebar:
    st.title("Climate Signal")
    st.markdown(
        "Classifies country-year emissions profiles against elevated GHG "
        "temperature forcing (OWID panel data)."
    )
    st.divider()
    page = st.radio("Pages", ["Country lookup", "Emissions trends", "Evaluation"])
    st.divider()
    st.markdown("**Data:** [OWID CO₂](https://ourworldindata.org/co2-emissions)")
    st.markdown("**Repo:** [GitHub](https://github.com/sanialolidk/climate-signal)")
    st.caption("Sania Thankan · Penn State CDS · 2026")

metrics = load_metrics()
panel = load_panel_cached()
bundle = load_bundle()

if page == "Country lookup":
    st.header("Country lookup")
    st.write(
        "Select a country and year. The model was trained on years through 2010 "
        "and evaluated on 2011 onward."
    )

    years = sorted(panel["year"].unique())
    countries = sorted(panel["country"].unique())

    left, right = st.columns(2)
    with left:
        country = st.selectbox(
            "Country",
            countries,
            index=countries.index("United States") if "United States" in countries else 0,
        )
    with right:
        year = st.selectbox("Year", years, index=max(0, len(years) - 8))

    iso = panel.loc[panel["country"] == country, "iso_code"].iloc[0]
    row = country_year_lookup(panel, iso, year)

    if row is None:
        st.warning("Missing data for that combination — OWID has gaps for some country-years.")
    else:
        out = predict_country_year(row, bundle)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Model score", f"{out['probability']:.0%}")
        with m2:
            st.metric("Label", out["class_name"])
        with m3:
            temp = row.get("temperature_change_from_ghg")
            st.metric("Observed GHG temp. change", f"{temp:.4f} °C" if temp else "n/a")

        if out["label"] == 1:
            st.warning("Elevated forcing band — emissions profile matches high-impact country-years in the training set.")
        else:
            st.success("Within normal range for this panel.")

        st.subheader("Inputs the model saw")
        st.write(
            {
                "co2_per_capita (t)": round(row.get("co2_per_capita", 0), 2),
                "co2_growth_pct": round(row.get("co2_growth_prct", 0), 2),
                "cumulative_co2 (Mt)": round(row.get("cumulative_co2", 0), 1),
                "share_global_co2 (%)": round(row.get("share_global_co2", 0), 3),
                "methane_per_capita": round(row.get("methane_per_capita", 0), 4),
                "ground_truth_elevated": bool(row[TARGET]),
            }
        )

elif page == "Emissions trends":
    st.header("Emissions trends")
    latest = int(panel["year"].max())
    st.subheader(f"Top per-capita emitters ({latest})")
    st.dataframe(top_emitters_recent(panel, year=latest), use_container_width=True, hide_index=True)

    st.subheader("CO₂ per capita over time")
    default = [c for c in ["United States", "China", "India", "Germany", "Brazil"] if c in set(panel["country"])]
    picks = st.multiselect("Overlay countries", sorted(panel["country"].unique()), default=default)
    if picks:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        fig.patch.set_facecolor("white")
        for name in picks:
            grp = panel[panel["country"] == name].sort_values("year")
            ax.plot(grp["year"], grp["co2_per_capita"], label=name, linewidth=2)
        ax.set_xlabel("Year")
        ax.set_ylabel("Tonnes CO₂ per capita")
        ax.legend(loc="upper left", frameon=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        st.pyplot(fig)

else:
    st.header("Model evaluation")
    if not metrics:
        st.info("Run `python main.py` first to generate metrics.")
    else:
        gb = metrics["gradient_boosting"]
        base = metrics["baseline"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{gb['accuracy']:.1%}")
        c2.metric("Macro F1", f"{gb['macro_f1']:.3f}")
        c3.metric("ROC-AUC", f"{gb['roc_auc']:.3f}")
        c4.metric("Test rows", f"{metrics['test_rows']:,}")

        st.write(f"**Train/test split:** {metrics['split']}")
        st.write(
            "Target = top quartile of `temperature_change_from_ghg` across the panel. "
            "Main model is histogram-based gradient boosting; baseline is balanced logistic regression."
        )

        comp = st.columns(2)
        with comp[0]:
            st.markdown("**Baseline**")
            st.write(f"Precision (elevated): {base['class_1']['precision']:.3f}")
            st.write(f"Recall (elevated): {base['class_1']['recall']:.3f}")
        with comp[1]:
            st.markdown("**Gradient boosting**")
            st.write(f"Precision (elevated): {gb['class_1']['precision']:.3f}")
            st.write(f"Recall (elevated): {gb['class_1']['recall']:.3f}")

        st.subheader("Feature importance (permutation)")
        st.bar_chart(metrics["feature_importance"])

        if st.checkbox("Show confusion matrices"):
            img1, img2 = st.columns(2)
            with img1:
                st.image("plots/cm_baseline.png", caption="Baseline")
            with img2:
                st.image("plots/cm_main.png", caption="Gradient boosting")

        st.markdown("---")
        st.write(
            "Limitations I ran into: country-year rows are correlated across time, "
            "OWID coverage is thin for some regions before ~1995, and this is a screening "
            "model — not something I'd treat as causal inference."
        )