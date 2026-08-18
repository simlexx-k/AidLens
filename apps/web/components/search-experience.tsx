"use client";

import { FormEvent, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { EvidenceSearchResponse, searchEvidence } from "../lib/api";

export function SearchExperience() {
  const params = useSearchParams();
  const initialQuery = params.get("q") ?? "";
  const [query, setQuery] = useState(initialQuery);
  const [data, setData] = useState<EvidenceSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runSearch(value: string) {
    const trimmed = value.trim();
    if (trimmed.length < 2) return;
    setLoading(true);
    setError(null);
    try {
      setData(await searchEvidence(trimmed));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (initialQuery) void runSearch(initialQuery);
    // The query string is only used to initialize this client search surface.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void runSearch(query);
  }

  return (
    <section className="search-shell">
      <div className="search-heading">
        <span className="eyebrow">Evidence search</span>
        <h1>Search evaluation findings, not just titles.</h1>
        <p>The first baseline uses PostgreSQL full-text ranking. Semantic retrieval is the next model layer.</p>
      </div>

      <form className="search-bar" onSubmit={onSubmit}>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Ask an evidence question…"
        />
        <button disabled={loading}>{loading ? "Searching…" : "Search"}</button>
      </form>

      {error && (
        <div className="status error">
          {error}. Make sure the API is running and the corpus has been ingested.
        </div>
      )}
      {data && (
        <div className="results">
          <div className="results-meta">
            <strong>{data.hits.length} evidence passages</strong>
            <span>{data.mode}</span>
          </div>
          {data.hits.length === 0 && <div className="empty-state">No matching passages yet.</div>}
          {data.hits.map((hit) => (
            <article className="result-card" key={hit.chunk_id}>
              <div className="result-kicker">
                <span>{hit.publication_year ?? "Year unknown"}</span>
                {hit.section && <span>{hit.section.replaceAll("_", " ")}</span>}
                <span>score {hit.score.toFixed(3)}</span>
              </div>
              <h2>{hit.title}</h2>
              <p>{hit.text}</p>
              <a href={hit.source_url} target="_blank" rel="noreferrer">Open source evaluation ↗</a>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
