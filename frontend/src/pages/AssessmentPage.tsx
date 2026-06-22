import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Assessment, Meta } from "../types";

interface Props {
  meta: Meta;
}

export function AssessmentPage({ meta }: Props) {
  const [country, setCountry] = useState(meta.default_country);
  const [year, setYear] = useState(meta.default_year);
  const [result, setResult] = useState<Assessment | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .assess(country, year)
      .then((data) => {
        if (!cancelled) setResult(data);
      })
      .catch((err: Error) => {
        if (!cancelled) {
          setResult(null);
          setError(err.message);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [country, year]);

  return (
    <section>
      <h2 className="page-title">Country Assessment</h2>
      <p className="page-desc">
        Select a jurisdiction and reporting year to obtain a machine learning classification of its
        greenhouse gas emissions profile.
      </p>

      <div className="grid-assess">
        <div>
          <p className="section-label">Query Parameters</p>
          <div className="panel">
            <div className="field">
              <label htmlFor="jurisdiction">Jurisdiction</label>
              <select
                id="jurisdiction"
                value={country}
                onChange={(e) => setCountry(e.target.value)}
              >
                {meta.countries.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="year">Reporting Year — {year}</label>
              <input
                id="year"
                type="range"
                min={meta.years.min}
                max={meta.years.max}
                value={year}
                onChange={(e) => setYear(Number(e.target.value))}
              />
            </div>
          </div>
          <p style={{ fontSize: "0.78rem", color: "var(--gray-500)", marginTop: "0.75rem" }}>
            Training period: years ≤ 2010 · Evaluation period: years &gt; 2010
          </p>
        </div>

        <div>
          <p className="section-label">Classification Result</p>
          {loading && <p className="loading">Loading assessment…</p>}
          {error && <div className="error">{error}</div>}
          {result && !loading && (
            <>
              <p style={{ fontSize: "0.8rem", color: "var(--gray-500)", margin: "0 0 0.5rem" }}>
                {result.country} — {result.year}
              </p>
              <p className="severity-legend">
                Severity (observed impact):
                <span className="dot" style={{ background: "var(--green)" }} />
                Standard
                <span className="dot" style={{ background: "var(--amber)" }} />
                Elevated
                <span className="dot" style={{ background: "var(--red)" }} />
                High — {result.severity_label}
              </p>
              <div className="grid-3">
                <div className={`metric-card primary ${result.severity}`}>
                  <div className="metric-kicker">Classification</div>
                  <div className={`metric-value ${result.severity}`}>{result.classification}</div>
                  <div className="metric-hint">Model-assigned profile type</div>
                </div>
                <div className={`metric-card primary ${result.severity}`}>
                  <div className="metric-kicker">GHG Temperature Impact</div>
                  <div className={`metric-value ${result.severity}`}>
                    {result.temperature_change != null
                      ? `${result.temperature_change.toFixed(4)} °C`
                      : "Not available"}
                  </div>
                  <div className="metric-hint">Observed GHG-attributed change</div>
                </div>
                <div className="metric-card">
                  <div className="metric-kicker">Model Confidence</div>
                  <div className="metric-value confidence">{result.confidence}</div>
                  <div className="metric-hint">Match probability — not a severity score</div>
                </div>
              </div>
              <div className="info-box">
                <strong>How to read this:</strong> Classification and temperature are observed
                outcomes. Model confidence ({result.confidence}) measures training-set similarity —
                not danger. Baseline (logistic regression): {result.baseline_confidence}.
              </div>
            </>
          )}
        </div>

        <div>
          <p className="section-label">Emissions Indicators</p>
          <div className="panel">
            {result ? (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Indicator</th>
                    <th style={{ textAlign: "right" }}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {result.indicators.map((row) => (
                    <tr key={row.feature}>
                      <td>{row.feature}</td>
                      <td className="num">{row.value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ margin: 0, color: "var(--gray-500)", fontSize: "0.85rem" }}>
                Indicator values appear after a valid jurisdiction-year is selected.
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}