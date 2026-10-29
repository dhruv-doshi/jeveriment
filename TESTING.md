# Verification handoff

Verified on 2026-09-23:

- `python -m pytest -q`: **31 passed**.
- `ruff check src tests`: passed.
- `ruff format --check src tests`: 40 files already formatted.
- `uv lock --check --offline`: passed; 86 packages resolved.
- Live Vercel Noul fixtures and Choice checks completed; see `manifests/capabilities.json`.
- Score validation detected inconsistent returned score/probabilities and remains disabled.

Coverage includes contracts, hashes, judgment status, official metric agreement,
blocked retrieval, deterministic pools, resume, malformed API responses, retries,
budget persistence, model drift, synthetic end-to-end exports/fallback, tiny-model
CPU inference, fusion leakage/search budgets, calibration, statistics, blinded
audit requirements, plotting, remote score validation, and RAG evidence/resume.

The 30-query SciFact train pilot has since completed through both the fixed-pool
Vercel workflow and the direct TypeSafe full-corpus scan. The direct scan scored
155,490 pairs and completed its BM25/dense/TypeSafe comparison; metrics and CPU
samples are recorded in `runs/scifact_train_pilot_v1/` and summarized in
`docs/PROJECT.md`. These train-split pilot results are exploratory. Tests listed
above use authored data and tiny random architectures; pretrained-model quality,
MPS performance, multi-domain resource use, multilingual quality, and human
calibration/generation judgments remain experimental validation work.

The guide's other optional expansion paths have code support, but no additional
remote scorer or generator configuration has been selected or executed. No SOTA,
calibration, or broad study-completion claim is made.
