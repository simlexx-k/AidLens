"use client";

import { FormEvent, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import { EvidenceSearchResponse, searchEvidence } from "../lib/api";

const sections = [
  ["", "All sections"],
  ["abstract", "Abstract"],
  ["executive_summary", "Executive summary"],
  ["introduction", "Introduction / background"],
  ["methodology", "Methodology"],
  ["evaluation_questions", "Evaluation questions"],
  ["findings", "Findings / results"],
  ["conclusions", "Conclusions"],
  ["recommendations", "Recommendations"],
  ["limitations", "Limitations"],
  ["sustainability", "Sustainability"],
  ["lessons_learned", "Lessons learned"],
];

export function SearchExperience() {
  const params = useSearchParams();
  const initialQuery = params.get("q") ?? "";
  const [query, setQuery] = useState(initialQuery);
  const [mode, setMode] = useState<"auto" | "lexical" | "semantic" | "hybrid">("auto");
  const [rerank, setRerank] = useState<"auto" | "disabled" | "aidranker">("auto");
  const [section, setSection] = useState("");
  const [yearFrom, setYearFrom] = useState("");
  const [yearTo, setYearTo] = useState("");
  const [maxPerEvaluation, setMaxPerEvaluation] = useState("3");
  const [data, setData] = useState<EvidenceSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runSearch(value: string) {
    const trimmed = value.trim();
    if (trimmed.length < 2) return;
    setLoading(true);
    setError(null);
    try {
      setData(
        await searchEvidence({
          query: trimmed,
          mode,
          rerank,
          section,
          publicationYearFrom: yearFrom ? Number(yearFrom) : undefined,
          publicationYearTo: yearTo ? Number(yearTo) : undefined,
          maxPerEvaluation: maxPerEvaluation ? Number(maxPerEvaluation) : undefined,
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (initialQuery) void runSearch(initialQuery);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void runSearch(query);
  }

  return (
    <section className="search-shell">
      <div className="search-heading">
        <span className="eyebrow">Evidence retrieval</span>
        <h1>Compare evidence across evaluations, not just passages.</h1>
        <p>
          AidLens uses semantic evidence retrieval and, when available, AidRanker to
          prioritize stronger supporting and direct-answer passages before applying the
          evidence-spread limit across reports.
        </p>
      </div>
      <form onSubmit={onSubmit}>
        <div className="search-bar">
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask an evidence question…"
          />
          <button disabled={loading}>{loading ? "Searching…" : "Search"}</button>
        </div>
        <div className="search-controls">
          <label>
            Retrieval
            <select
              value={mode}
              onChange={(event) => setMode(event.target.value as typeof mode)}
            >
              <option value="auto">Auto</option>
              <option value="semantic">Semantic</option>
              <option value="hybrid">Hybrid</option>
              <option value="lexical">Lexical</option>
            </select>
          </label>
          <label>
            Ranking
            <select
              value={rerank}
              onChange={(event) => setRerank(event.target.value as typeof rerank)}
            >
              <option value="auto">Best available</option>
              <option value="disabled">First-stage only</option>
              <option value="aidranker">Require AidRanker</option>
            </select>
          </label>
          <label>
            Section
            <select value={section} onChange={(event) => setSection(event.target.value)}>
              {sections.map(([value, label]) => (
                <option key={value || "all"} value={value}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            Evidence spread
            <select
              value={maxPerEvaluation}
              onChange={(event) => setMaxPerEvaluation(event.target.value)}
            >
              <option value="3">Balanced · max 3/report</option>
              <option value="1">Broad · max 1/report</option>
              <option value="2">Broader · max 2/report</option>
              <option value="5">Focused · max 5/report</option>
              <option value="">Uncapped · diagnostic</option>
            </select>
          </label>
          <label>
            From year
            <input
              value={yearFrom}
              onChange={(event) => setYearFrom(event.target.value)}
              inputMode="numeric"
            />
          </label>
          <label>
            To year
            <input
              value={yearTo}
              onChange={(event) => setYearTo(event.target.value)}
              inputMode="numeric"
            />
          </label>
        </div>
      </form>
      {error && <div className="status error">{error}</div>}
      {data && (
        <div className="results">
          <div className="results-meta">
            <strong>{data.hits.length} evidence passages</strong>
            <span>
              {data.mode}
              {data.reranker_applied && data.reranker_alpha !== null
                ? ` · AidRanker α ${data.reranker_alpha.toFixed(2)}`
                : ""}
              {data.reranker_fallback_reason ? " · semantic fallback" : ""}
              {data.max_per_evaluation ? ` · max ${data.max_per_evaluation}/report` : " · uncapped"}
            </span>
          </div>
          {data.hits.length === 0 && (
            <div className="empty-state">No matching passages yet.</div>
          )}
          {data.hits.map((hit) => (
            <article className="result-card" key={hit.chunk_id}>
              <div className="result-kicker">
                <span>{hit.publication_year ?? "Year unknown"}</span>
                {hit.section && <span>{hit.section.replaceAll("_", " ")}</span>}
                <span>{hit.retrieval_sources.join(" + ")}</span>
                <span>score {hit.score.toFixed(4)}</span>
              </div>
              <h2>{hit.title}</h2>
              <p>{hit.text}</p>
              {(hit.lexical_score !== null ||
                hit.semantic_score !== null ||
                hit.reranker_score !== null) && (
                <div className="score-row">
                  {hit.lexical_score !== null && (
                    <span>lexical {hit.lexical_score.toFixed(3)}</span>
                  )}
                  {hit.semantic_score !== null && (
                    <span>semantic {hit.semantic_score.toFixed(3)}</span>
                  )}
                  {hit.reranker_score !== null && (
                    <span>AidRanker {hit.reranker_score.toFixed(3)}</span>
                  )}
                </div>
              )}
              <a href={hit.source_url} target="_blank" rel="noreferrer">
                Open source evaluation ↗
              </a>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
