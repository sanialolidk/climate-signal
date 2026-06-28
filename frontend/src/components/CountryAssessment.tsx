/**
 * Country Assessment — three-panel layout with severity gauge + indicator table.
 *
 * Fonts (also in index.html):
 *   Source Sans 3, Source Serif 4, Source Code Pro
 */
import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import {
  severityColor,
  severityGaugePos,
  T,
  type Severity,
} from "../tokens";
import type { Assessment, Meta } from "../types";

interface Props {
  meta: Meta;
}

const sectionLabel: CSSProperties = {
  fontSize: "0.72rem",
  fontWeight: 700,
  textTransform: "uppercase",
  letterSpacing: "0.08em",
  color: T.navy,
  margin: "0 0 0.65rem",
};

const panel: CSSProperties = {
  background: T.white,
  border: `1px solid ${T.border}`,
  borderRadius: 6,
  padding: "1rem 1.1rem",
};

function SeverityGauge({ severity, label }: { severity: Severity; label: string }) {
  return (
    <div style={{ margin: "0.75rem 0 1rem" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          borderRadius: 4,
          overflow: "hidden",
          height: 10,
          fontSize: 0,
        }}
      >
        <div style={{ background: T.successDark }} />
        <div style={{ background: T.warningDark }} />
        <div style={{ background: T.errorDark }} />
      </div>
      <div style={{ position: "relative", height: 14, marginTop: 2 }}>
        <div
          style={{
            position: "absolute",
            left: severityGaugePos[severity],
            transform: "translateX(-50%)",
            width: 0,
            height: 0,
            borderLeft: "6px solid transparent",
            borderRight: "6px solid transparent",
            borderBottom: `8px solid ${severityColor[severity]}`,
          }}
        />
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          fontSize: "0.72rem",
          color: T.gray500,
          marginTop: 4,
          textAlign: "center",
        }}
      >
        <span>Standard</span>
        <span>Elevated</span>
        <span>High</span>
      </div>
      <p style={{ fontSize: "0.76rem", color: T.gray500, margin: "0.5rem 0 0", textAlign: "center" }}>
        {label}
      </p>
    </div>
  );
}

function StatCard({
  kicker,
  value,
  hint,
  accent,
}: {
  kicker: string;
  value: string;
  hint: string;
  accent?: string;
}) {
  return (
    <div
      style={{
        flex: "1 1 0",
        minWidth: 0,
        background: T.white,
        border: `1px solid ${T.border}`,
        borderTop: `3px solid ${accent ?? T.navy}`,
        borderRadius: 6,
        padding: "0.85rem 0.65rem",
        textAlign: "center",
      }}
    >
      <div
        style={{
          fontSize: "0.68rem",
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.07em",
          color: T.gray500,
          marginBottom: "0.35rem",
          whiteSpace: "nowrap",
        }}
      >
        {kicker}
      </div>
      <div
        style={{
          fontSize: "1.2rem",
          fontWeight: 700,
          lineHeight: 1.2,
          color: accent ?? T.navy,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
          fontFamily: T.fontMono,
        }}
      >
        {value}
      </div>
      <div style={{ fontSize: "0.7rem", color: T.gray500, marginTop: "0.35rem", lineHeight: 1.35 }}>
        {hint}
      </div>
    </div>
  );
}

export function CountryAssessment({ meta }: Props) {
  const [country, setCountry] = useState(meta.default_country);
  const [year, setYear] = useState(meta.default_year);
  const [result, setResult] = useState<Assessment | null>(null);
  const [trend, setTrend] = useState<{ year: number; co2: number }[]>([]);
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

  useEffect(() => {
    let cancelled = false;
    api
      .timeseries([country])
      .then((data) => {
        if (cancelled) return;
        const pts = data.series[0]?.points ?? [];
        setTrend(pts.map((p) => ({ year: p.year, co2: p.co2_per_capita })));
      })
      .catch(() => {
        if (!cancelled) setTrend([]);
      });
    return () => {
      cancelled = true;
    };
  }, [country]);

  const accent = result ? severityColor[result.severity] : T.navy;
  const tempDisplay = useMemo(() => {
    if (!result || result.temperature_change == null) return "N/A";
    return `${result.temperature_change.toFixed(4)} °C`;
  }, [result]);

  return (
    <section>
      <h2
        style={{
          fontFamily: T.fontSerif,
          fontSize: "1.25rem",
          fontWeight: 700,
          margin: "0 0 0.35rem",
          paddingBottom: "0.5rem",
          borderBottom: `2px solid ${T.navy}`,
          display: "inline-block",
        }}
      >
        Country Assessment
      </h2>
      <p
        style={{
          color: T.gray700,
          fontSize: "0.9rem",
          lineHeight: 1.65,
          maxWidth: "52rem",
          margin: "0.75rem 0 1.5rem",
        }}
      >
        Select a jurisdiction and reporting year to obtain a machine learning classification of its
        greenhouse gas emissions profile.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(220px, 0.9fr) minmax(280px, 1.15fr) minmax(260px, 1.2fr)",
          gap: "1rem",
        }}
        className="grid-assess"
      >
        {/* Panel 1 — Query Parameters */}
        <div>
          <p style={sectionLabel}>Query Parameters</p>
          <div style={panel}>
            <label
              htmlFor="jurisdiction"
              style={{
                display: "block",
                fontSize: "0.75rem",
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: "0.05em",
                color: T.gray700,
                marginBottom: "0.35rem",
              }}
            >
              Jurisdiction
            </label>
            <select
              id="jurisdiction"
              value={country}
              onChange={(e) => setCountry(e.target.value)}
              style={{ width: "100%", fontFamily: T.fontSans }}
            >
              {meta.countries.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>

            <div style={{ marginTop: "0.85rem" }}>
              <label
                htmlFor="year"
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.05em",
                  color: T.gray700,
                  marginBottom: "0.35rem",
                }}
              >
                <span>Reporting Year</span>
                <span
                  style={{
                    fontFamily: T.fontMono,
                    background: T.navy,
                    color: T.white,
                    padding: "0.15rem 0.55rem",
                    borderRadius: 4,
                    fontSize: "0.8rem",
                    letterSpacing: 0,
                  }}
                >
                  {year}
                </span>
              </label>
              <input
                id="year"
                type="range"
                min={meta.years.min}
                max={meta.years.max}
                value={year}
                onChange={(e) => setYear(Number(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
          </div>
          <p style={{ fontSize: "0.78rem", color: T.gray500, marginTop: "0.75rem" }}>
            Training period: years ≤ 2010 · Evaluation period: years &gt; 2010
          </p>
        </div>

        {/* Panel 2 — Classification Result */}
        <div>
          <p style={sectionLabel}>Classification Result</p>
          {loading && <p style={{ color: T.gray500, fontSize: "0.9rem" }}>Loading assessment…</p>}
          {error && (
            <div
              style={{
                background: "#fde8e8",
                border: "1px solid #f5c2c2",
                color: "#9b1c1c",
                padding: "0.75rem 1rem",
                borderRadius: 6,
                fontSize: "0.85rem",
              }}
            >
              {error}
            </div>
          )}
          {result && !loading && (
            <div style={{ ...panel, borderTop: `3px solid ${accent}` }}>
              <p style={{ fontSize: "0.8rem", color: T.gray500, margin: "0 0 0.25rem" }}>
                {result.country} — {result.year}
              </p>

              <SeverityGauge severity={result.severity} label={result.severity_label} />

              <div style={{ display: "flex", gap: "0.6rem", marginBottom: "0.85rem" }}>
                <StatCard
                  kicker="Classification"
                  value={result.classification}
                  hint="Model-assigned profile"
                  accent={accent}
                />
                <StatCard
                  kicker="GHG ΔT"
                  value={tempDisplay}
                  hint="Observed GHG-attributed change"
                  accent={accent}
                />
                <StatCard
                  kicker="Confidence"
                  value={result.confidence}
                  hint="Training-set similarity"
                  accent={T.navy}
                />
              </div>

              {/* Recharts slot — emissions trend AreaChart for selected jurisdiction */}
              {trend.length > 0 && (
                <div style={{ marginBottom: "0.85rem" }}>
                  <p
                    style={{
                      fontSize: "0.68rem",
                      fontWeight: 700,
                      textTransform: "uppercase",
                      letterSpacing: "0.07em",
                      color: T.gray500,
                      margin: "0 0 0.4rem",
                    }}
                  >
                    CO₂ per Capita Trend
                  </p>
                  <div style={{ width: "100%", height: 120 }}>
                    <ResponsiveContainer>
                      <AreaChart data={trend} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                        <defs>
                          <linearGradient id="co2Grad" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor={T.navy} stopOpacity={0.35} />
                            <stop offset="100%" stopColor={T.navy} stopOpacity={0.02} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke={T.border} vertical={false} />
                        <XAxis
                          dataKey="year"
                          tick={{ fontSize: 10, fill: T.gray500 }}
                          axisLine={{ stroke: T.border }}
                          tickLine={false}
                        />
                        <YAxis
                          tick={{ fontSize: 10, fill: T.gray500 }}
                          axisLine={false}
                          tickLine={false}
                          width={36}
                        />
                        <Tooltip
                          contentStyle={{
                            fontSize: "0.78rem",
                            border: `1px solid ${T.border}`,
                            borderRadius: 4,
                          }}
                          formatter={(v: number) => [`${v.toFixed(2)} t`, "CO₂/cap"]}
                        />
                        <Area
                          type="monotone"
                          dataKey="co2"
                          stroke={T.navy}
                          strokeWidth={2}
                          fill="url(#co2Grad)"
                          dot={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
              {/*
                // To plug in a custom emissions-trend chart, replace the block above with:
                <ResponsiveContainer width="100%" height={120}>
                  <AreaChart data={yourSeries} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={T.border} vertical={false} />
                    <XAxis dataKey="year" tick={{ fontSize: 10, fill: T.gray500 }} />
                    <YAxis tick={{ fontSize: 10, fill: T.gray500 }} width={36} />
                    <Tooltip />
                    <Area type="monotone" dataKey="co2" stroke={T.navy} fill={T.navy} fillOpacity={0.12} />
                  </AreaChart>
                </ResponsiveContainer>
              */}

              <div
                style={{
                  background: T.infoBg,
                  borderRadius: 6,
                  padding: "0.85rem 1rem",
                  fontSize: "0.84rem",
                  color: T.gray700,
                  lineHeight: 1.55,
                }}
              >
                <strong>How to read this:</strong> Classification and temperature are observed
                outcomes. Model confidence ({result.confidence}) measures training-set similarity —
                not danger. Baseline (logistic regression): {result.baseline_confidence}.
              </div>
            </div>
          )}
        </div>

        {/* Panel 3 — Emissions Indicators */}
        <div>
          <p style={sectionLabel}>Emissions Indicators</p>
          <div style={{ ...panel, padding: 0, overflow: "hidden" }}>
            {result ? (
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.84rem" }}>
                <thead>
                  <tr>
                    <th
                      style={{
                        textAlign: "left",
                        fontSize: "0.7rem",
                        fontWeight: 700,
                        textTransform: "uppercase",
                        letterSpacing: "0.06em",
                        color: T.gray500,
                        padding: "0.55rem 1rem",
                        borderBottom: `2px solid ${T.border}`,
                        background: T.gray100,
                      }}
                    >
                      Indicator
                    </th>
                    <th
                      style={{
                        textAlign: "right",
                        fontSize: "0.7rem",
                        fontWeight: 700,
                        textTransform: "uppercase",
                        letterSpacing: "0.06em",
                        color: T.gray500,
                        padding: "0.55rem 1rem",
                        borderBottom: `2px solid ${T.border}`,
                        background: T.gray100,
                      }}
                    >
                      Value
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {result.indicators.map((row, i) => (
                    <tr
                      key={row.feature}
                      style={{ background: i % 2 === 0 ? T.white : T.gray100 }}
                    >
                      <td
                        style={{
                          padding: "0.45rem 1rem",
                          borderBottom: `1px solid ${T.gray200}`,
                          color: T.gray900,
                        }}
                      >
                        {row.feature}
                      </td>
                      <td
                        style={{
                          padding: "0.45rem 1rem",
                          borderBottom: `1px solid ${T.gray200}`,
                          textAlign: "right",
                          fontFamily: T.fontMono,
                          fontWeight: 600,
                          fontSize: "0.82rem",
                          color: T.gray900,
                          whiteSpace: "nowrap",
                        }}
                      >
                        {row.value}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p style={{ margin: 0, padding: "1rem", color: T.gray500, fontSize: "0.85rem" }}>
                Indicator values appear after a valid jurisdiction-year is selected.
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}