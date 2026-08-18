# Retrieval metric interpretation

AidLens V0.4 reports three complementary retrieval metrics at a configurable
cutoff `K`.

## Recall@K

The fraction of human-judged relevant evidence targets retrieved within the
first `K` results. Judgments are keyed by stable evaluation ID plus an optional
section. Multiple chunks matching the same judgment count once, preventing long
reports from inflating recall through duplicate passages.

## Reciprocal rank / MRR

For one query, reciprocal rank is `1 / rank` of the first relevant result. The
benchmark summary averages this value across queries to produce MRR. This
measures how quickly a user sees the first useful piece of evidence.

## nDCG@K

Normalized Discounted Cumulative Gain uses graded relevance (`1`, `2`, `3`) and
discounts useful results that appear lower in the ranking. It therefore rewards
both relevance quality and ordering.

Use Recall@K as the coverage signal, MRR as the first-useful-result signal, and
nDCG@K as the ranking-quality signal. A retrieval change should not be promoted
based on one metric alone.
