import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, TARGET, load_panel, split_by_year
from .paths import p


def _class_stats(rep, label):
    key = str(int(label))
    if key not in rep:
        key = str(float(label))
    return {
        "precision": round(rep[key]["precision"], 4),
        "recall": round(rep[key]["recall"], 4),
        "f1": round(rep[key]["f1-score"], 4),
        "support": int(rep[key]["support"]),
    }


def _summarize(y_true, y_pred, y_prob):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "class_0": _class_stats(rep, 0),
        "class_1": _class_stats(rep, 1),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _save_cm(matrix, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="YlOrRd")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Elevated"])
    ax.set_yticklabels(["Normal", "Elevated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(train_end=2010):
    panel = load_panel()
    train_df, test_df = split_by_year(panel, train_end=train_end)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    print(f"Rows: train={len(train_df)} test={len(test_df)} countries={panel['iso_code'].nunique()}")
    print(f"Target rate (test): {y_test.mean():.1%}")

    print("\n--- Baseline: Logistic Regression ---")
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(Xtr, y_train)
    lr_prob = lr.predict_proba(Xte)[:, 1]
    lr_pred = lr.predict(Xte)
    lr_m = _summarize(y_test, lr_pred, lr_prob)
    print(classification_report(y_test, lr_pred, zero_division=0))

    print("\n--- Main: HistGradientBoosting ---")
    gb = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=6,
        learning_rate=0.06,
        class_weight="balanced",
        random_state=42,
    )
    gb.fit(Xtr, y_train)
    gb_prob = gb.predict_proba(Xte)[:, 1]
    gb_pred = gb.predict(Xte)
    gb_m = _summarize(y_test, gb_pred, gb_prob)
    print(classification_report(y_test, gb_pred, zero_division=0))

    tscv = TimeSeriesSplit(n_splits=4)
    cv = cross_val_score(gb, Xtr, y_train, cv=tscv, scoring="roc_auc")

    perm = permutation_importance(gb, Xte, y_test.astype(int), n_repeats=8, random_state=42, scoring="roc_auc")
    importances = pd.Series(perm.importances_mean, index=FEATURES).sort_values(ascending=False)

    bundle = {
        "scaler": scaler,
        "lr": lr,
        "gb": gb,
        "features": FEATURES,
        "train_end": train_end,
        "temp_cutoff_note": "elevated = top quartile temperature_change_from_ghg",
    }
    joblib.dump(bundle, p("models", "bundle.pkl"))

    metrics = {
        "source": "Our World in Data CO2 dataset (data/panel.csv)",
        "task": "Flag country-years with elevated GHG temperature forcing",
        "split": f"train <= {train_end}, test > {train_end}",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "countries": int(panel["iso_code"].nunique()),
        "baseline": lr_m,
        "gradient_boosting": gb_m,
        "cv_roc_auc_mean": round(float(cv.mean()), 4),
        "cv_roc_auc_std": round(float(cv.std()), 4),
        "feature_importance": importances.head(10).round(4).to_dict(),
    }
    with p("models", "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    _save_cm(np.array(lr_m["confusion_matrix"]), p("plots", "cm_baseline.png"), "Baseline (test)")
    _save_cm(np.array(gb_m["confusion_matrix"]), p("plots", "cm_main.png"), "Gradient boosting (test)")

    fig, ax = plt.subplots(figsize=(7, 4))
    importances.head(10).sort_values().plot(kind="barh", ax=ax, color="#2d6a4f")
    ax.set_title("Top feature importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(p("plots", "feature_importance.png"), dpi=150)
    plt.close(fig)

    print("\nSaved models/bundle.pkl, models/metrics.json, plots/")


if __name__ == "__main__":
    run()