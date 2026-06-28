/**
 * Standalone reference component — integrated copy lives at
 * frontend/src/components/CountryAssessment.tsx
 *
 * Add to index.html <head>:
 * <link rel="preconnect" href="https://fonts.googleapis.com" />
 * <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
 * <link href="https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@400;600&family=Source+Sans+3:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap" rel="stylesheet" />
 */
import { useState } from "react";

export const T = {
  navy: "#1a4480",
  navyDark: "#162e51",
  navyLight: "#c8d9ed",
  bg: "#f8f9fa",
  white: "#ffffff",
  gray900: "#1b1b1b",
  gray700: "#4a5568",
  gray500: "#6b7280",
  gray200: "#e8eaed",
  gray100: "#f1f3f5",
  border: "#d0d5dc",
  successDark: "#2e7d32",
  warningDark: "#b35c00",
  errorDark: "#b50909",
  infoBg: "#eef2f7",
  fontSans: '"Source Sans 3", "Segoe UI", system-ui, sans-serif',
  fontSerif: '"Source Serif 4", Georgia, serif',
  fontMono: '"Source Code Pro", Consolas, monospace',
};

const INDICATORS = [
  { feature: "CO₂ per capita (tonnes)", value: "15.241" },
  { feature: "Annual CO₂ growth (%)", value: "-2.104" },
  { feature: "Cumulative CO₂ (Mt)", value: "5,416.2" },
  { feature: "Share of global CO₂ (%)", value: "13.492" },
  { feature: "Observed GHG forcing classification", value: "Elevated" },
];

const SEVERITY = "moderate"; // standard | moderate | high
const GAUGE_POS = { standard: "16.67%", moderate: "50%", high: "83.33%" };
const SEVERITY_COLOR = { standard: T.successDark, moderate: T.warningDark, high: T.errorDark };

export default function CountryAssessment() {
  const [year, setYear] = useState(2016);

  return (
    <section style={{ fontFamily: T.fontSans, color: T.gray900 }}>
      <h2 style={{ fontFamily: T.fontSerif, borderBottom: `2px solid ${T.navy}`, display: "inline-block" }}>
        Country Assessment
      </h2>

      <div style={{ display: "grid", gridTemplateColumns: "0.9fr 1.15fr 1.2fr", gap: "1rem" }}>
        <div>
          <p style={{ color: T.navy, fontSize: "0.72rem", fontWeight: 700, textTransform: "uppercase" }}>
            Query Parameters
          </p>
          <div style={{ background: T.white, border: `1px solid ${T.border}`, padding: "1rem", borderRadius: 6 }}>
            <label>Jurisdiction</label>
            <select style={{ width: "100%" }} defaultValue="United States">
              <option>United States</option>
              <option>China</option>
            </select>
            <label style={{ display: "flex", justifyContent: "space-between", marginTop: "0.85rem" }}>
              Reporting Year
              <span style={{ fontFamily: T.fontMono, background: T.navy, color: T.white, padding: "0.15rem 0.5rem", borderRadius: 4 }}>
                {year}
              </span>
            </label>
            <input type="range" min={1990} max={2022} value={year} onChange={(e) => setYear(Number(e.target.value))} style={{ width: "100%" }} />
          </div>
        </div>

        <div>
          <p style={{ color: T.navy, fontSize: "0.72rem", fontWeight: 700, textTransform: "uppercase" }}>
            Classification Result
          </p>
          <div style={{ background: T.white, border: `1px solid ${T.border}`, borderTop: `3px solid ${SEVERITY_COLOR[SEVERITY]}`, padding: "1rem", borderRadius: 6 }}>
            <p style={{ color: T.gray500, fontSize: "0.8rem" }}>United States — {year}</p>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", height: 10, borderRadius: 4, overflow: "hidden" }}>
              <div style={{ background: T.successDark }} />
              <div style={{ background: T.warningDark }} />
              <div style={{ background: T.errorDark }} />
            </div>
            <div style={{ position: "relative", height: 14 }}>
              <div style={{ position: "absolute", left: GAUGE_POS[SEVERITY], transform: "translateX(-50%)", borderLeft: "6px solid transparent", borderRight: "6px solid transparent", borderBottom: `8px solid ${SEVERITY_COLOR[SEVERITY]}` }} />
            </div>
            <div style={{ display: "flex", gap: "0.6rem" }}>
              {[
                ["Classification", "Elevated"],
                ["GHG ΔT", "0.0241 °C"],
                ["Confidence", "98.7%"],
              ].map(([k, v]) => (
                <div key={k} style={{ flex: 1, minWidth: 0, textAlign: "center", border: `1px solid ${T.border}`, borderTop: `3px solid ${T.navy}`, padding: "0.75rem 0.5rem", borderRadius: 6 }}>
                  <div style={{ fontSize: "0.68rem", color: T.gray500, textTransform: "uppercase" }}>{k}</div>
                  <div style={{ fontFamily: T.fontMono, fontWeight: 700, whiteSpace: "nowrap" }}>{v}</div>
                </div>
              ))}
            </div>
            {/*
              <ResponsiveContainer width="100%" height={120}>
                <AreaChart data={emissionsTrend}>
                  <Area type="monotone" dataKey="co2" stroke={T.navy} fill={T.navy} fillOpacity={0.12} />
                </AreaChart>
              </ResponsiveContainer>
            */}
          </div>
        </div>

        <div>
          <p style={{ color: T.navy, fontSize: "0.72rem", fontWeight: 700, textTransform: "uppercase" }}>
            Emissions Indicators
          </p>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.84rem" }}>
            <tbody>
              {INDICATORS.map((row, i) => (
                <tr key={row.feature} style={{ background: i % 2 === 0 ? T.white : T.gray100 }}>
                  <td style={{ padding: "0.45rem 1rem" }}>{row.feature}</td>
                  <td style={{ padding: "0.45rem 1rem", textAlign: "right", fontFamily: T.fontMono, fontWeight: 600 }}>{row.value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}