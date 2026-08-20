export type EvidenceRole =
  | "outcome"
  | "recommendation"
  | "method"
  | "context"
  | "implementation"
  | "sustainability"
  | "supporting";

export type EvidenceHit = {
  chunk_id: string;
  evaluation_id: string;
  title: string;
  project_title: string | null;
  publication_year: number | null;
  section: string | null;
  evidence_role: EvidenceRole;
  text: string;
  score: number;
  lexical_score: number | null;
  semantic_score: number | null;
  reranker_score: number | null;
  fusion_score: number | null;
  retrieval_sources: string[];
  locations: string[];
  institutions: string[];
  keywords: string[];
  source_url: string;
};

export type EvidenceEvaluationGroup = {
  evaluation_id: string;
  title: string;
  project_title: string | null;
  intervention: string;
  publication_year: number | null;
  locations: string[];
  institutions: string[];
  keywords: string[];
  evidence_roles: EvidenceRole[];
  outcome_evidence_count: number;
  top_score: number;
  source_url: string;
  hits: EvidenceHit[];
};

export type EvidenceSearchResponse = {
  query: string;
  mode: string;
  embedding_model: string | null;
  max_per_evaluation: number | null;
  reranker_applied: boolean;
  reranker: string | null;
  reranker_model: string | null;
  reranker_model_fingerprint: string | null;
  reranker_alpha: number | null;
  reranker_fallback_reason: string | null;
  ranking_pipeline: string[];
  first_stage_latency_ms: number | null;
  reranker_latency_ms: number | null;
  total_search_latency_ms: number | null;
  groups: EvidenceEvaluationGroup[];
  hits: EvidenceHit[];
};

export type SearchOptions = {
  query: string;
  mode?: "auto" | "lexical" | "semantic" | "hybrid";
  rerank?: "auto" | "disabled" | "aidranker";
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
      top_k: options.topK ?? 10,
      mode: options.mode ?? "auto",
      rerank: options.rerank ?? "auto",
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
