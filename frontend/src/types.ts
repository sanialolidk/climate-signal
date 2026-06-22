export type Page = "assessment" | "emissions" | "validation";

export type Severity = "standard" | "moderate" | "high";

export interface Meta {
  countries: string[];
  years: { min: number; max: number };
  jurisdiction_count: number;
  default_country: string;
  default_year: number;
}

export interface Indicator {
  feature: string;
  value: string;
}

export interface Assessment {
  country: string;
  year: number;
  iso_code: string;
  classification: string;
  label: number;
  confidence: string;
  confidence_raw: number;
  baseline_confidence: string;
  temperature_change: number | null;
  severity: Severity;
  severity_label: string;
  indicators: Indicator[];
}

export interface EmitterRow {
  jurisdiction: string;
  iso_code: string;
  year: number;
  co2_per_capita: number;
  temperature_change: number;
  classification: string;
}

export interface TimePoint {
  year: number;
  co2_per_capita: number;
}

export interface TimeSeries {
  country: string;
  points: TimePoint[];
}

export interface ClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelMetrics {
  accuracy: number;
  macro_f1: number;
  roc_auc: number;
  class_1: ClassMetrics;
}

export interface MetricsPayload {
  split: string;
  train_rows: number;
  test_rows: number;
  countries: number;
  baseline: ModelMetrics;
  gradient_boosting: ModelMetrics;
  feature_importance: Record<string, number>;
}