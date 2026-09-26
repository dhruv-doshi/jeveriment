# Development verification

Branch: `feature/production-rag`. This report covers local development only; no full experiment or live TypeSafe/OpenRouter load was run.

## Verified

- `uv lock --check --offline`: passed, 106 locked packages.
- `uv run --no-sync --group dev --extra service --extra load --extra observability pytest -q`: 21 passed. One warning comes from the installed FastAPI/Starlette `TestClient` import.
- `uv run --no-sync --group dev --extra service --extra load --extra observability ruff check .`: passed.
- `jev-eval rag matrix ... --datasets qasper --retrievers bm25 --rerankers retrieval_order minilm --limit 5 --workers 4`: dry-run forecast returned two matched cells and zero provider calls.
- The matrix `--execute` path ran a two-query BM25/retrieval-order fixture with two query workers and reused saved artifacts without provider calls.
- `jev-eval rag load-fake ... --requests 20 --workers 4 --latency-ms 2 --every-429 5`: 20/20 succeeded after 24 HTTP attempts. The p99 is marked low-sample because this fixture has fewer than 100 requests.
- Tests use small official-shape records for all five adapters. They cover SQLite BM25, exact dense search and revision mismatch, RRF, shared candidate pools, local reranker contract, parallel JEV mock responses, OpenRouter generation and metadata mock responses, optional judge, global ledger, replay, paired comparison, stage export, service auth, and OpenTelemetry recording.

## Pending before benchmark or production claims

- The five real corpora are absent under `data/sources`; `rag audit-sources` reports every source missing. Official source revisions, terms, corpus compatibility, full source-to-evidence joins, and frozen query manifests need verification.
- Local embedding and reranker model weights were not downloaded or run. Exact device throughput, GPU memory, and model-specific output behavior need a small smoke test after revisions are pinned.
- No real TypeSafe or OpenRouter request was sent. Authentication, current response schema, account rate limits, pricing, resolved provider/model, and provider inference timing need a small metered preflight. The API latency, cost, and rate behavior in tests are injected fixtures.
- The full 75-cell matrix and real-provider stress test remain unrun by the owner's instruction. The $50 cap is configured, but a final run forecast and account limits file must be reviewed once source sizes and provider prices are known.

Use [SOURCES.md](SOURCES.md) for expected source files and [RUNBOOK.md](RUNBOOK.md) for commands. The implementation plan remains the experiment protocol; this report states which gates have actual development evidence.
