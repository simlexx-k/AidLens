export type EvidenceHit = {
  chunk_id: string;
  evaluation_id: string;
  title: string;
  publication_year: number | null;
  section: string | null;
  text: string;
  score: number;
  lexical_score: number | null;
  semantic_score: number | null;
  retrieval_sources: string[];
  source_url: string;
};

export type EvidenceSearchResponse = {
  query: string;
  mode: string;
  embedding_model: string | null;
  max_per_evaluation: number | null;
  hits: EvidenceHit[];
};

export type SearchOptions = {
  query: string;
  mode?: "auto" | "lexical" | "semantic" | "hybrid";
  publicationYearFrom?: number;
  publicationYearTo?: number;
  section?: string;
  topK?: number;
  maxPerEvaluation?: number;
};

export type LabelCount = { label: string; count: number };
export type QualityFlag = { code: string; count: number; description: string };
export type CorpusStats = {
  evaluation_count: number;
  chunk_count: number;
  embedded_chunk_count: number;
  embedding_coverage_percent: number;
  embedding_model: string | null;
  publication_year_min: number | null;
  publication_year_max: number | null;
  section_counts: LabelCount[];
  chunker_versions: LabelCount[];
  top_keywords: LabelCount[];
  top_institutions: LabelCount[];
  quality_flags: QualityFlag[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

export async function searchEvidence(options: SearchOptions): Promise<EvidenceSearchResponse> {
  const response = await fetch(`${API_URL}/search/evidence`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query: options.query,
      top_k: options.topK ?? 12,
      mode: options.mode ?? "auto",
      publication_year_from: options.publicationYearFrom,
      publication_year_to: options.publicationYearTo,
      section: options.section || undefined,
      max_per_evaluation: options.maxPerEvaluation,
    }),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail ? `: ${body.detail}` : "";
    throw new Error(`Search failed (${response.status})${detail}`);
  }
  return response.json();
}

export async function getCorpusStats(): Promise<CorpusStats> {
  const response = await fetch(`${API_URL}/analytics/corpus`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Corpus analytics failed (${response.status})`);
  return response.json();
}
