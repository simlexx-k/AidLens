export type EvidenceHit = {
  chunk_id: string;
  evaluation_id: string;
  title: string;
  publication_year: number | null;
  section: string | null;
  text: string;
  score: number;
  source_url: string;
};

export type EvidenceSearchResponse = {
  query: string;
  mode: string;
  hits: EvidenceHit[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

export async function searchEvidence(query: string): Promise<EvidenceSearchResponse> {
  const response = await fetch(`${API_URL}/search/evidence`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: 12 }),
  });
  if (!response.ok) {
    throw new Error(`Search failed (${response.status})`);
  }
  return response.json();
}
