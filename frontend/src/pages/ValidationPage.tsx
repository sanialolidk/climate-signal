import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { MetricsPayload } from "../types";

export function ValidationPage() {
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .metrics()
      .then(setMetrics)
      .catch((err: Error) => setError(err.message));
  }, []);

  const comparison = useMemo(() => {
    if (!metrics) return [];
    const gb = metrics.gradient_boosting;
    const base = metrics.baseline;
    return [
      { metric: "Accuracy", logistic: base.accuracy, gbm: gb.accuracy, fmt: "pct" },
      { metric: "Macro F1", logistic: base.macro_f1, gbm: gb.macro_f1, fmt: "num" },
      { metric: "ROC-AUC", logistic: base.roc_auc, gbm: gb.roc_auc, fmt: "num" },
      {
        metric: "Precision (elevated)",
        logistic: base.class_1.precision,
        gbm: gb.class_1.precision,
        fmt: "num",
      },
      {
        metric: "Recall (elevated)",
        logistic: base.class_1.recall,
        gbm: gb.class_1.recall,
        fmt: "num",
      },
    ];
  }, [metrics]);

  const importance = useMemo(() => {
    if (!metrics) return [];
    return Object.entries(metrics.feature_importance)
      .map(([feature, importance]) => ({ feature, importance }))
      .sort((a, b) => a.importance - b.importance);
  }, [metrics]);

  if (error) return <div className="error">{error}</div>;
  if (!metrics) return <p className="loading">Loading validation metrics…</p>;

  const gb = metrics.gradient_boosting;

  return (
    <section>
      <h2 className="page-title">Model Validation</h2>
      <p className="page-desc">
        HistGradientBoosting vs balanced logistic regression on a temporal holdout split.
      </p>

      <div className="grid-3" style={{ marginBottom: "1rem" }}>
        {[
          ["Classification Accuracy", `${(gb.accuracy * 100).toFixed(1)}%`],
          ["Macro F1 Score", gb.macro_f1.toFixed(3)],
          ["ROC-AUC", gb.roc_auc.toFixed(3)],
        ].map(([label, val]) => (
          <div key={label} className="stat-tile">
            <div className="n">{val}</div>
            <div className="l">{label}</div>
          </div>
        ))}
      </div>

      <p style={{ fontSize: "0.82rem", color: "var(--gray-700)", marginBottom: "1.25rem" }}>
        {metrics.split} · {metrics.train_rows.toLocaleString()} train ·{" "}
        {metrics.test_rows.toLocaleString()} test · {metrics.countries} jurisdictions
      </p>

      <div className="grid-2">
        <div>
          <p className="section-label">Comparative Model Performance</p>
          <div className="panel">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Logistic Regression</th>
                  <th>Gradient Boosting</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((row) => (
                  <tr key={row.metric}>
                    <td>{row.metric}</td>
                    <td className="num">
                      {row.fmt === "pct"
                        ? `${(row.logistic * 100).toFixed(1)}%`
                        : row.logistic.toFixed(3)}
                    </td>
                    <td className="num">
                      {row.fmt === "pct"
                        ? `${(row.gbm * 100).toFixed(1)}%`
                        : row.gbm.toFixed(3)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <p className="section-label">Permutation Feature Importance</p>
          <div style={{ width: "100%", height: 360 }}>
            <ResponsiveContainer>
              <BarChart data={importance} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d0d5dc" />
                <XAxis type="number" tick={{ fontSize: 11 }} />
                <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={95} />
                <Tooltip />
                <Bar dataKey="importance" fill="#1a4480" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="info-box" style={{ marginTop: "1.5rem" }}>
        <strong>Disclaimer:</strong> Screening classifier only — not causal inference. Panel rows
        correlate across time.
      </div>
    </section>
  );
}