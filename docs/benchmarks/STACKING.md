# V0.4 branch relationship

V0.4 is intentionally stacked on `agent/v0.3-document-structure` while V0.3
finishes its live semantic/hybrid smoke test. This keeps parser/metadata changes
reviewable separately from retrieval-evaluation tooling.

Once V0.3 is merged, retarget the V0.4 pull request to `main`. No code changes
should be required if `main` contains the V0.3 head.
