"use client";

import { useEffect, useState } from "react";
import { CorpusStats, getCorpusStats } from "../lib/api";

function number(value: number) {
  return new Intl.NumberFormat("en").format(value);
}

export function CorpusOverview() {
  const [stats, setStats] = useState<CorpusStats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCorpusStats()
      .then(setStats)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Unable to load corpus analytics"),
      );
  }, []);

  if (error) return <div className="status error">{error}</div>;
  if (!stats) return <div className="status">Loading corpus intelligence…</div>;

  const maxSection = Math.max(...stats.section_counts.map((item) => item.count), 1);

  return (
    <div className="corpus-stack">
      <section className="metric-grid">
        <article>
          <span>Evaluations</span>
          <strong>{number(stats.evaluation_count)}</strong>
        </article>
        <article>
          <span>Evidence chunks</span>
          <strong>{number(stats.chunk_count)}</strong>
        </article>
        <article>
          <span>Embedded</span>
          <strong>{stats.embedding_coverage_percent.toFixed(1)}%</strong>
        </article>
        <article>
          <span>Publication range</span>
          <strong>
            {stats.publication_year_min ?? "?"}–{stats.publication_year_max ?? "?"}
          </strong>
        </article>
      </section>

      <section className="analytics-grid">
        <article className="analytics-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Coverage</span>
              <h2>Report sections</h2>
            </div>
            <span>{number(stats.chunk_count)} chunks</span>
          </div>
          <div className="bar-list">
            {stats.section_counts.map((item) => (
              <div className="bar-row" key={item.label}>
                <div>
                  <span>{item.label.replaceAll("_", " ")}</span>
                  <strong>{number(item.count)}</strong>
                </div>
                <div className="bar-track">
                  <span style={{ width: `${(item.count / maxSection) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </article>

        <article className="analytics-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Data quality</span>
              <h2>Quality flags</h2>
            </div>
          </div>
          <div className="quality-list">
            {stats.quality_flags.map((flag) => (
              <div key={flag.code}>
                <strong>{number(flag.count)}</strong>
                <div>
                  <span>{flag.code.replaceAll("_", " ")}</span>
                  <p>{flag.description}</p>
                </div>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section className="analytics-grid">
        <article className="analytics-panel">
          <span className="eyebrow">Metadata</span>
          <h2>Top keywords</h2>
          <div className="chip-cloud">
            {stats.top_keywords.map((item) => (
              <span key={item.label}>
                {item.label} <strong>{item.count}</strong>
              </span>
            ))}
          </div>
        </article>

        <article className="analytics-panel">
          <span className="eyebrow">Organizations</span>
          <h2>Top institutions</h2>
          <div className="chip-cloud">
            {stats.top_institutions.map((item) => (
              <span key={item.label}>
                {item.label} <strong>{item.count}</strong>
              </span>
            ))}
          </div>
        </article>
      </section>

      <section className="analytics-grid">
        <article className="analytics-panel">
          <span className="eyebrow">Pipeline provenance</span>
          <h2>Chunker versions</h2>
          <div className="chip-cloud">
            {stats.chunker_versions.map((item) => (
              <span key={item.label}>
                {item.label} <strong>{item.count}</strong>
              </span>
            ))}
          </div>
        </article>
      </section>

      <div className="status">
        Embedding model: {stats.embedding_model ?? "not enabled"} ·{" "}
        {number(stats.embedded_chunk_count)} of {number(stats.chunk_count)} chunks embedded.
      </div>
    </div>
  );
}
