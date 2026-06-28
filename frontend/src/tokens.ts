/** Shared design tokens — USWDS-inspired palette for all three tabs. */
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
  // USWDS state colors
  success: "#00a91c",
  successDark: "#2e7d32",
  warning: "#e5a000",
  warningDark: "#b35c00",
  error: "#d54309",
  errorDark: "#b50909",
  infoBg: "#eef2f7",
  fontSans: '"Source Sans 3", "Segoe UI", system-ui, sans-serif',
  fontSerif: '"Source Serif 4", Georgia, "Times New Roman", serif',
  fontMono: '"Source Code Pro", "Consolas", monospace',
  chart: ["#1a4480", "#2e6ba8", "#4a8bc4", "#6b5b95", "#3d6b5a", "#5c4a3a"],
} as const;

export type Severity = "standard" | "moderate" | "high";

export const severityColor: Record<Severity, string> = {
  standard: T.successDark,
  moderate: T.warningDark,
  high: T.errorDark,
};

export const severityGaugePos: Record<Severity, string> = {
  standard: "16.67%",
  moderate: "50%",
  high: "83.33%",
};

/** Inject tokens as CSS custom properties on :root */
export function applyTokenVars(): void {
  const root = document.documentElement;
  root.style.setProperty("--blue-800", T.navy);
  root.style.setProperty("--blue-900", T.navyDark);
  root.style.setProperty("--blue-100", T.navyLight);
  root.style.setProperty("--bg", T.bg);
  root.style.setProperty("--white", T.white);
  root.style.setProperty("--gray-900", T.gray900);
  root.style.setProperty("--gray-700", T.gray700);
  root.style.setProperty("--gray-500", T.gray500);
  root.style.setProperty("--gray-200", T.gray200);
  root.style.setProperty("--gray-100", T.gray100);
  root.style.setProperty("--green", T.successDark);
  root.style.setProperty("--amber", T.warningDark);
  root.style.setProperty("--red", T.errorDark);
  root.style.fontFamily = T.fontSans;
}