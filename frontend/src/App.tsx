import { useEffect, useState } from "react";
import { api } from "./api/client";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { AssessmentPage } from "./pages/AssessmentPage";
import { EmissionsPage } from "./pages/EmissionsPage";
import { ValidationPage } from "./pages/ValidationPage";
import type { Meta, Page } from "./types";

const TABS: { id: Page; label: string }[] = [
  { id: "assessment", label: "Country Assessment" },
  { id: "emissions", label: "Emissions Data" },
  { id: "validation", label: "Model Validation" },
];

export default function App() {
  const [page, setPage] = useState<Page>("assessment");
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .meta()
      .then(setMeta)
      .catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div className="app-shell">
      <Header />

      <nav className="nav-tabs" aria-label="Main navigation">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`nav-tab ${page === tab.id ? "active" : ""}`}
            onClick={() => setPage(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {error && (
        <div className="error" style={{ marginBottom: "1rem" }}>
          Cannot reach API — start the backend with:{" "}
          <code>uvicorn api.main:app --reload --port 8000</code>
          <br />
          {error}
        </div>
      )}

      {!meta && !error && <p className="loading">Connecting to API…</p>}

      {meta && page === "assessment" && <AssessmentPage meta={meta} />}
      {meta && page === "emissions" && <EmissionsPage meta={meta} />}
      {page === "validation" && <ValidationPage />}

      <Footer />
    </div>
  );
}