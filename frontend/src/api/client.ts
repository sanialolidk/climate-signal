import type { Assessment, EmitterRow, Meta, MetricsPayload, TimeSeries } from "../types";

const BASE = import.meta.env.VITE_API_URL ?? "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(detail || res.statusText);
  }
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => get<Meta>("/api/meta"),
  assess: (country: string, year: number) =>
    get<Assessment>(`/api/assess?country=${encodeURIComponent(country)}&year=${year}`),
  emitters: (year?: number) =>
    get<{ year: number; rows: EmitterRow[] }>(`/api/emitters${year ? `?year=${year}` : ""}`),
  timeseries: (countries: string[]) =>
    get<{ series: TimeSeries[] }>(
      `/api/timeseries?countries=${encodeURIComponent(countries.join(","))}`,
    ),
  metrics: () => get<MetricsPayload>("/api/metrics"),
};