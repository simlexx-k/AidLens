import Link from "next/link";

export default function Home() {
  return (
    <main>
      <section className="hero">
        <div className="eyebrow">Development evidence, structured for decisions</div>
        <h1>Find what worked, where, and why.</h1>
        <p className="hero-copy">
          AidLens turns evaluation reports into searchable evidence on interventions,
          outcomes, implementation context, success factors, and failure patterns.
        </p>
        <form className="hero-search" action="/search">
          <input
            name="q"
            aria-label="Search development evidence"
            placeholder="e.g. interventions that improved smallholder farmer income"
          />
          <button type="submit">Explore evidence</button>
        </form>
        <div className="hero-meta">
          <span>4,500+ archived evaluations</span>
          <span>Evidence-level retrieval</span>
          <span>ML-ready knowledge layer</span>
        </div>
      </section>

      <section className="feature-grid">
        <article>
          <div className="feature-number">01</div>
          <h2>Evidence, not document dumps</h2>
          <p>Retrieve the passages that support findings instead of scrolling through long PDFs.</p>
        </article>
        <article>
          <div className="feature-number">02</div>
          <h2>Compare interventions</h2>
          <p>Build toward intervention → context → outcome comparisons across countries and sectors.</p>
        </article>
        <article>
          <div className="feature-number">03</div>
          <h2>Model-ready by design</h2>
          <p>Structured chunks and provenance become training data for extraction, ranking, and graph ML.</p>
        </article>
      </section>

      <section className="cta-panel">
        <div>
          <span className="eyebrow">V0.1</span>
          <h2>Start with the evidence corpus.</h2>
          <p>The current build establishes ingestion, provenance, lexical retrieval, and vector-ready storage.</p>
        </div>
        <Link className="button-secondary" href="/search">Open search</Link>
      </section>
    </main>
  );
}
