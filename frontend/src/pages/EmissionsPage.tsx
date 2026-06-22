import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api/client";
import type { EmitterRow, Meta, TimeSeries } from "../types";

const COLORS = ["#1a4480", "#2e6ba8", "#4a8bc4", "#6b5b95", "#3d6b5a", "#5c4a3a"];
const DEFAULT_PICKS = ["United States", "China", "India", "Germany", "Brazil"];

interface Props {
  meta: Meta;
}

export function EmissionsPage({ meta }: Props) {
  const available = useMemo(
    () => DEFAULT_PICKS.filter((c) => meta.countries.includes(c)),
    [meta.countries],
  );
  const [picks, setPicks] = useState<string[]>(available);
  const [emitters, setEmitters] = useState<EmitterRow[]>([]);
  const [latestYear, setLatestYear] = useState(meta.years.max);
  const [series, setSeries] = useState<TimeSeries[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .emitters()
      .then((data) => {
        setEmitters(data.rows);
        setLatestYear(data.year);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (picks.length === 0) {
      setSeries([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    api
      .timeseries(picks)
      .then((data) => setSeries(data.series))
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [picks]);

  const chartData = useMemo(() => {
    if (series.length === 0) return [];
    const years = new Set<number>();
    series.forEach((s) => s.points.forEach((p) => years.add(p.year)));
    return [...years]
      .sort((a, b) => a - b)
      .map((year) => {
        const row: Record<string, number> = { year };
        series.forEach((s) => {
          const pt = s.points.find((p) => p.year === year);
          if (pt) row[s.country] = pt.co2_per_capita;
        });
        return row;
      });
  }, [series]);

  const togglePick = (name: string) => {
    setPicks((prev) =>
      prev.includes(name) ? prev.filter((c) => c !== name) : [...prev, name],
    );
  };

  return (
    <section>
      <h2 className="page-title">Emissions Data</h2>
      <p className="page-desc">
        Comparative emissions statistics and temporal trends from the OWID panel (through {latestYear}).
      </p>
      {error && <div className="error">{error}</div>}

      <div className="grid-2">
        <div>
          <p className="section-label">Highest Per-Capita Emitters</p>
          <div className="panel" style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Jurisdiction</th>
                  <th>CO₂/cap</th>
                  <th>ΔT (°C)</th>
                  <th>Class</th>
                </tr>
              </thead>
              <tbody>
                {emitters.map((r) => (
                  <tr key={`${r.iso_code}-${r.year}`}>
                    <td>{r.jurisdiction}</td>
                    <td className="num">{r.co2_per_capita.toFixed(2)}</td>
                    <td className="num">{r.temperature_change.toFixed(4)}</td>
                    <td>{r.classification}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div>
          <p className="section-label">Per-Capita CO₂ — Time Series</p>
          <div className="chip-row">
            {meta.countries.slice(0, 24).map((name) => (
              <button
                key={name}
                type="button"
                className={`chip ${picks.includes(name) ? "active" : ""}`}
                onClick={() => togglePick(name)}
              >
                {name}
              </button>
            ))}
          </div>
          {loading && <p className="loading">Loading chart…</p>}
          {!loading && picks.length === 0 && (
            <p style={{ color: "var(--gray-500)", fontSize: "0.85rem" }}>
              Select jurisdictions to display the time series.
            </p>
          )}
          {!loading && chartData.length > 0 && (
            <div style={{ width: "100%", height: 320 }}>
              <ResponsiveContainer>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#d0d5dc" />
                  <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} unit=" t" />
                  <Tooltip />
                  <Legend />
                  {series.map((s, i) => (
                    <Line
                      key={s.country}
                      type="monotone"
                      dataKey={s.country}
                      stroke={COLORS[i % COLORS.length]}
                      strokeWidth={2}
                      dot={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}