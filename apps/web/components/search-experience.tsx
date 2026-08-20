"use client";

import { FormEvent, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  CrossEvaluationSynthesis,
  EvidenceEvaluationGroup,
  EvidenceSearchResponse,
  searchEvidence,
} from "../lib/api";

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
        <span className="eyebrow">Development evidence intelligence</span>
        <h1>Compare evidence across evaluations without losing the source.</h1>
        <p>
          AidLens ranks passages, groups them back into evaluations, and maps recurring
          evidence and context signals across the result set. It does not turn metadata
          overlap into causal claims or transferability verdicts.
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
            <select value={mode} onChange={(event) => setMode(event.target.value as typeof mode)}>
              <option value="auto">Auto</option>
              <option value="semantic">Semantic</option>
              <option value="hybrid">Hybrid</option>
              <option value="lexical">Lexical</option>
            </select>
          </label>
          <label>
            Ranking
            <select value={rerank} onChange={(event) => setRerank(event.target.value as typeof rerank)}>
              <option value="auto">Best available</option>
              <option value="disabled">First-stage only</option>
              <option value="aidranker">Require AidRanker</option>
            </select>
          </label>
          <label>
            Section
            <select value={section} onChange={(event) => setSection(event.target.value)}>
              {sections.map(([value, label]) => (
                <option key={value || "all"} value={value}>{label}</option>
              ))}
            </select>
          </label>
          <label>
            Evidence spread
            <select value={maxPerEvaluation} onChange={(event) => setMaxPerEvaluation(event.target.value)}>
              <option value="3">Balanced · max 3/report</option>
              <option value="1">Broad · max 1/report</option>
              <option value="2">Broader · max 2/report</option>
              <option value="5">Focused · max 5/report</option>
              <option value="">Uncapped · diagnostic</option>
            </select>
          </label>
          <label>
            From year
            <input value={yearFrom} onChange={(event) => setYearFrom(event.target.value)} inputMode="numeric" />
          </label>
          <label>
            To year
            <input value={yearTo} onChange={(event) => setYearTo(event.target.value)} inputMode="numeric" />
          </label>
        </div>
      </form>
      {error && <div className="status error">{error}</div>}
      {data && (
        <div className="results">
          <div className="results-meta">
            <strong>{data.groups.length} evaluations · {data.hits.length} passages</strong>
            <span>
              {data.mode}
              {data.reranker_applied && data.reranker_alpha !== null
                ? ` · AidRanker α ${data.reranker_alpha.toFixed(2)}`
                : ""}
              {data.reranker_fallback_reason ? " · semantic fallback" : ""}
              {(data.request_latency_ms ?? data.total_search_latency_ms) !== null
                ? ` · ${Math.round(data.request_latency_ms ?? data.total_search_latency_ms ?? 0)} ms`
                : ""}
            </span>
          </div>

          <div className="pipeline-strip">
            <span>{data.ranking_pipeline.join(" → ")}</span>
            {data.reranker_model_fingerprint && (
              <code title={data.reranker_model_fingerprint}>
                model {data.reranker_model_fingerprint.slice(7, 19)}
              </code>
            )}
          </div>

          {data.synthesis && data.hits.length > 0 && (
            <SynthesisPanel synthesis={data.synthesis} groups={data.groups} />
          )}

          {data.hits.length === 0 && <div className="empty-state">No matching evidence yet.</div>}
          {data.groups.map((group) => <EvidenceGroup key={group.evaluation_id} group={group} />)}
        </div>
      )}
    </section>
  );
}

function SynthesisPanel({
  synthesis,
  groups,
}: {
  synthesis: CrossEvaluationSynthesis;
  groups: EvidenceEvaluationGroup[];
}) {
  const keywords = synthesis.recurring_facets.filter((facet) => facet.kind === "keyword").slice(0, 8);
  const contexts = synthesis.recurring_facets.filter((facet) => facet.kind !== "keyword").slice(0, 8);
  const titleById = new Map(groups.map((group) => [group.evaluation_id, group.title]));

  return (
    <section className="synthesis-panel">
      <div className="synthesis-heading">
        <div>
          <span className="eyebrow">Cross-evaluation evidence map</span>
          <h2>What recurs across these results?</h2>
        </div>
        <span className="synthesis-scope">Result-set synthesis</span>
      </div>

      <div className="synthesis-metrics">
        <div><strong>{synthesis.evaluation_count}</strong><span>evaluations compared</span></div>
        <div><strong>{synthesis.outcome_evaluation_count}</strong><span>with outcome evidence</span></div>
        <div><strong>{synthesis.recurring_facets.length}</strong><span>recurring context/theme signals</span></div>
        <div><strong>{synthesis.transferability_pairs.length}</strong><span>context-overlap pairs</span></div>
      </div>

      <div className="synthesis-grid">
        <div className="synthesis-section">
          <span className="synthesis-label">Evidence coverage</span>
          <div className="coverage-list">
            {synthesis.role_coverage.map((coverage) => (
              <span key={coverage.role} className={`coverage-chip role-${coverage.role}`}>
                {coverage.role} · {coverage.evaluation_count} eval · {coverage.passage_count} passage
                {coverage.passage_count === 1 ? "" : "s"}
              </span>
            ))}
          </div>
        </div>

        <div className="synthesis-section">
          <span className="synthesis-label">Recurring themes</span>
          {keywords.length ? (
            <div className="facet-list">
              {keywords.map((facet) => (
                <span key={`${facet.kind}-${facet.value}`}>
                  {facet.value} <em>{facet.evaluation_count} eval</em>
                </span>
              ))}
            </div>
          ) : <p>No keyword appears across multiple returned evaluations.</p>}
        </div>

        <div className="synthesis-section">
          <span className="synthesis-label">Recurring context</span>
          {contexts.length ? (
            <div className="facet-list">
              {contexts.map((facet) => (
                <span key={`${facet.kind}-${facet.value}`}>
                  {facet.value} <em>{facet.kind} · {facet.evaluation_count} eval</em>
                </span>
              ))}
            </div>
          ) : <p>No structured location or institution recurs across these results.</p>}
        </div>

        <div className="synthesis-section transferability-section">
          <span className="synthesis-label">Context overlap · not a transfer verdict</span>
          {synthesis.transferability_pairs.length ? (
            <div className="overlap-list">
              {synthesis.transferability_pairs.slice(0, 5).map((pair) => {
                const shared = [
                  ...pair.shared_locations,
                  ...pair.shared_institutions,
                  ...pair.shared_keywords,
                ].slice(0, 5);
                return (
                  <div key={`${pair.left_evaluation_id}-${pair.right_evaluation_id}`}>
                    <strong>
                      {shortTitle(titleById.get(pair.left_evaluation_id) ?? pair.left_evaluation_id)}
                      {" ↔ "}
                      {shortTitle(titleById.get(pair.right_evaluation_id) ?? pair.right_evaluation_id)}
                    </strong>
                    <span>{Math.round(pair.context_overlap_score * 100)}% metadata overlap</span>
                    <small>{shared.join(" · ")}</small>
                  </div>
                );
              })}
            </div>
          ) : <p>No structured context overlap among the returned evaluations.</p>}
        </div>
      </div>

      <p className="synthesis-caveat">
        {synthesis.caveats[0]} {synthesis.caveats[1]} AidLens does not infer contradictory
        effect stance here; open the ranked passages below before drawing effect conclusions.
      </p>
    </section>
  );
}

function EvidenceGroup({ group }: { group: EvidenceEvaluationGroup }) {
  const context = [...group.locations, ...group.institutions, ...group.keywords].slice(0, 6);

  return (
    <article className="evidence-group">
      <div className="group-heading">
        <div>
          <div className="result-kicker">
            <span>{group.publication_year ?? "Year unknown"}</span>
            <span>{group.evaluation_id}</span>
            <span>{group.hits.length} passage{group.hits.length === 1 ? "" : "s"}</span>
          </div>
          <h2>{group.title}</h2>
        </div>
        <a href={group.source_url} target="_blank" rel="noreferrer">Open evaluation ↗</a>
      </div>

      <div className="evidence-frame">
        <div>
          <span>Intervention</span>
          <strong>{group.intervention}</strong>
        </div>
        <div>
          <span>Context</span>
          <strong>{context.length ? context.join(" · ") : "No structured context metadata"}</strong>
        </div>
        <div>
          <span>Outcome evidence</span>
          <strong>
            {group.outcome_evidence_count > 0
              ? `${group.outcome_evidence_count} outcome/sustainability passage${group.outcome_evidence_count === 1 ? "" : "s"}`
              : "No outcome-tagged passage in these results"}
          </strong>
        </div>
        <div>
          <span>Evidence</span>
          <strong>{group.evidence_roles.join(" · ")}</strong>
        </div>
      </div>

      <div className="group-passages">
        {group.hits.map((hit) => (
          <div className="evidence-passage" key={hit.chunk_id}>
            <div className="passage-meta">
              <span className={`role-badge role-${hit.evidence_role}`}>{hit.evidence_role}</span>
              {hit.section && <span>{hit.section.replaceAll("_", " ")}</span>}
              <span>score {hit.score.toFixed(3)}</span>
            </div>
            <p>{hit.text}</p>
            <div className="score-row">
              {hit.semantic_score !== null && <span>semantic {hit.semantic_score.toFixed(3)}</span>}
              {hit.reranker_score !== null && <span>AidRanker {hit.reranker_score.toFixed(3)}</span>}
              {hit.fusion_score !== null && <span>fusion {hit.fusion_score.toFixed(3)}</span>}
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function shortTitle(value: string): string {
  return value.length > 42 ? `${value.slice(0, 39)}…` : value;
}
