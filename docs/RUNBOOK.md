# Development and experiment runbook

Work in the `feature/production-rag` worktree. These commands prepare local artifacts or inspect the plan. A complete five-dataset run and live load test wait for the owner's separate instruction.

```sh
uv sync --group dev --extra local --extra service --extra load --extra observability
uv run jev-eval rag validate configs/rag/experiment.yaml
uv run jev-eval rag audit-sources configs/rag/experiment.yaml --source-root data/sources
uv run pytest -q
uv run ruff check .
```

For a development source audit on the Mac, fetch only the smaller files. This leaves the CLERC passage collection and both full Wikipedia corpora for the VM:

```sh
python3 scripts/fetch_small_sources.py
uv run jev-eval rag prepare configs/rag/experiment.yaml mirage --source-root data/sources
uv run jev-eval rag prepare configs/rag/experiment.yaml qasper --source-root data/sources
```

Preparation refuses to overwrite an existing prepared source; use a new run ID if its inputs change. [SOURCE_AUDIT.md](SOURCE_AUDIT.md) records the current completed joins and VM gate.

After placing a source in the layout in [SOURCES.md](SOURCES.md), prepare it with the same `rag prepare` command and its dataset ID. This builds the same chunk store for all retrieval/reranking arms, writes separate query and label files, and checks evidence joins. It can be slow for CLERC and KILT. It does not call an API.

The local models are pinned to Hugging Face commits in `configs/rag/model_revisions.json`:

```json
{
  "Qwen/Qwen3-Embedding-0.6B": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
  "cross-encoder/ms-marco-MiniLM-L6-v2": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
  "BAAI/bge-reranker-v2-m3": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
  "Qwen/Qwen3-Reranker-0.6B": "e61197ed45024b0ed8a2d74b80b4d909f1255473"
}
```

```sh
uv run jev-eval rag index-dense configs/rag/experiment.yaml qasper --revisions-json configs/rag/model_revisions.json
uv run jev-eval rag run configs/rag/experiment.yaml qasper bm25 retrieval_order --limit 5 --workers 4
uv run jev-eval rag matrix configs/rag/experiment.yaml --datasets qasper --retrievers bm25 --rerankers retrieval_order minilm --limit 5 --workers 4
```

The pinned model smoke and synthetic live-provider preflights are separate from the experiment. Each live preflight makes at most one paid model call and stores raw response/usage without an authorization header. The worktree's ignored `.env` supplies `OPENROUTER_API_KEY`, `TYPESAFE_API_KEY`, and `JEV_MODEL`.

```sh
uv run --extra local python scripts/smoke_local_models.py --device cpu
uv run python scripts/preflight_providers.py typesafe --env-file .env
uv run python scripts/preflight_providers.py openrouter --env-file .env
```

`rag run` and `rag matrix` print a dry-run call forecast by default, including up to three free metadata lookups per OpenRouter generation. `--execute` runs the selected slice or matrix. The matrix command reuses one runtime per dataset and runs queries in parallel within each cell; the same cached retrieval pools are reused across rerankers. Add `--generate` for final answers and `--judge` for the optional fixed, metered answer judge. Live TypeSafe or OpenRouter calls require `.env` credentials, a pinned OpenRouter provider, and an ignored local `data/limits.json` with account-specific `rpm`, `tpm`, `inflight`, `token_budget`, `request_budget`, `input_usd_per_million`, and `output_usd_per_million` for each provider used. The command checks a conservative upper-bound forecast against the shared $50 ledger before dispatch. The budget ledger and SQLite rate gates are shared by parallel workers and processes using the same run root. The default `--workers 4` runs queries concurrently; JEV candidate scoring also uses bounded parallel workers behind the account gate.

The `results/*.json` files hold per-query intermediate and final metrics; `artifacts/` holds stage inputs/outputs; `raw/` holds provider requests/responses and attempts without authorization headers. Repeating an unchanged stage reuses its recorded output. Changing a reranker formula can start from the saved JEV candidate probabilities. A new JEV question, model, or larger pool needs new calls. The run root and source files are ignored by Git.

```sh
uv run jev-eval rag report configs/rag/experiment.yaml runs/production_rag_v1/qasper/results/<result>.json --output runs/production_rag_v1/report.json
uv run jev-eval rag compare configs/rag/experiment.yaml --baseline <baseline.json> --challenger <challenger.json> --metric ranking.ndcg@10 --output runs/production_rag_v1/paired.json
uv run jev-eval rag load-fake configs/rag/experiment.yaml --requests 100 --workers 8 --latency-ms 20 --every-429 10
uv run jev-eval rag export-traces configs/rag/experiment.yaml --output runs/production_rag_v1/parquet
```

The local fake load command injects delay and throttling without external calls. The FastAPI service is available through `rag serve` with `SERVICE_API_KEY`; it binds to localhost by default and disables paid adapters unless started with `--allow-paid`. `rag/locustfile.py` defines a Locust workload for a later authorized service test. Provider-side inference timing is left unavailable when the provider does not expose it. OpenRouter generation metadata is stored with its raw timing fields and billed cost; its timing units are not inferred from client latency. In the 2026-09-26 preflight, OpenRouter's `/key` response showed a $10 key-level spending limit and roughly $5.08 remaining; its `rate_limit` field explicitly says it is deprecated. The configured $50 experiment ceiling cannot override that key limit. TypeSafe exposed no account-specific RPM/TPM headers in its synthetic response. Before any bulk run, obtain current key balances and provider limits and fill ignored `data/limits.json` conservatively.
