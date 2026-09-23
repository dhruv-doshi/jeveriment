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

SciFact preparation completed locally; dense retrieval stopped during pretrained
weight download, and no benchmark ranking/evaluation has run. Tests use authored
data and tiny random architectures; pretrained-model quality,
MPS performance, full-study resource usage, multilingual quality, and human
calibration/generation judgments remain experimental validation work.

The guide's optional expansion paths have code support, but no remote scorer or
generator configuration has been selected or executed. No SOTA, accuracy,
calibration, or study-completion claim is made.
