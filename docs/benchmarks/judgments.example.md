# Judgment example

Use this only as a labeling reference. Do not benchmark against placeholder IDs.

```json
{
  "query_id": "q001",
  "query": "What interventions improved household resilience to food insecurity?",
  "judgments": [
    {
      "evaluation_id": "REAL_AIDDATA_ID",
      "section": "findings",
      "relevance": 3
    },
    {
      "evaluation_id": "ANOTHER_REAL_ID",
      "section": "conclusions",
      "relevance": 2
    }
  ]
}
```

A section may be omitted when the entire evaluation is relevant and any passage
from it can satisfy the judgment. Prefer section-level labels when the evidence
question targets findings, recommendations, methodology, or another explicit
report component.
