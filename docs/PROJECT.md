# Project walkthrough

## Purpose and current status

This repository implements the retrieval study in the [original guide](Jev_RAG_Retrieval_Final_Guide.md). Every system reranks or selects from the same frozen candidate pool so comparisons isolate the scoring method. The default SciFact pilot is a feasibility run over 30 sampled training queries and the full corpus, not a confirmatory result. Synthetic unit/integration tests and Jev capability fixtures have run. A real Phase 2 retrieval run began locally but stopped during pretrained weight download; dense retrieval, ranking, and the final experiment have not completed. No relevance or superiority claim follows from the current repository.

## Five phases

1. `01_setup.sh` uses the pinned `uv.lock` to create `.venv`, validates `.env` and the selected configuration. It does not issue model requests.
2. `02_prepare.sh` obtains a BEIR release (or accepts `--source PATH`), ingests documents/qrels, samples queries deterministically, records corpus/model revisions, and freezes a shared text view. It checks that the same evidence fits each local model.
3. `03_retrieve.sh` indexes frozen views with BM25S, computes dense embeddings, and creates deterministic lexical+dense candidate pools. Dense embedding progress is checkpointed in a memory-mapped array.
4. `04_rerank.sh` scores those pools with Qwen and BGE local rerankers and Jev Noul through Vercel. Run all three systems or give one or more system names as arguments. Each score file is checkpointed, with query-level resume and request caching. `device: auto` tries CUDA, then MPS, then CPU, with CPU fallback on accelerator errors.
5. `05_evaluate.sh` computes fixed-pool rankings, selection sets, per-query and aggregate metrics, TREC exports, and plots. A partial system is marked separately from its operational RRF fallback.

The scripts run from any working directory. `CONFIG=path/to/config.yaml` selects another experiment; the config's `experiment_id` names its run directory. A frozen run rejects changes to its input config, so use a new experiment ID when varying parameters. `04_rerank.sh` can be split across sessions, and repeated phases retain checkpoints where supported. Preparation and retrieval may still repeat some completed work; retain `runs/`, `data/`, and `.cache/` between sessions.

## Data flow and code map

- `config.py`, `contracts.py`, `io.py`: validated inputs, invariant checks, deterministic hashes, and atomic JSON writes.
- `data.py`, `pipeline.py`: BEIR ingest, preparation, shared text, retrieval, reranking, and evaluation orchestration.
- `models.py`, `retrieve.py`, `selection.py`, `fusion.py`, `feature_model.py`: local adapters, pool construction, top-k choice, fusion, and optional development-only learned baseline.
- `jev.py`, `rubrics.py`, `ledger.py`, `preflight.py`: typed Jev requests, task prompts, strict response checks, capability gates, retry/cache, and persistent cost reservations. Score is disabled after a synthetic contract inconsistency; Noul is the experiment path.
- `metrics.py`, `stats.py`, `calibration.py`, `audit.py`, `reporting.py`: relevance measures, paired inference, calibration, blinded audit support, and plots.
- `expansion.py`, `generation.py`: optional externally produced fixed pools/scores and local evidence-constrained RAG generation. These are not part of the five pilot scripts.

The core configurations cover SciFact, NFCorpus, FiQA, and ArguAna. Dataset-specific rubrics matter: SciFact seeks evidence for or against a claim; ArguAna seeks counterarguments and excludes self-matches. The pilot's default fusion is fixed RRF. Weight search and logistic baselines must be fit on development data, not the test split.

## What is preserved

All paths below are under `runs/<experiment_id>/` unless noted. `logs/` contains a new UTC-stamped stdout/stderr transcript for every scripted phase invocation; a nonzero pipeline exit is retained as such. `resources_prepare.json`, `resources_retrieve.json`, `resources_rerank_{qwen,bge,jev}.json`, and `resources_evaluate.json` record elapsed time, peak RSS, swap, platform, and success/failure. They are summaries, not GPU-profiler traces.

Preparation keeps `dataset_manifest.json`, `model_manifest.json`, `input_config.json`, `resolved_config.{json,yaml}`, `corpus.sqlite`, `queries.json`, `qrels.json`, and `text_views.json`. Retrieval keeps the BM25 index and scores, `embeddings.npy`, `embedding_checkpoint.json`, row IDs, `pools.json`, `source_runs.json`, and `candidates.parquet`. Reranking keeps `scores_<system>.json`, Jev raw requests/responses and model catalog, plus the shared `runs/budget.sqlite` ledger. Evaluation keeps `per_query_metrics.{json,parquet}`, `scores.parquet`, `summary.json`, selections, TREC files, and plots; `reports/<experiment_id>.md` is a compact human-readable summary. JSON checkpoints are atomically replaced. Raw scores and per-query rows are preserved so later analyses need not rely on rounded report values.

Generated files are git-ignored and can be large. Back up `runs/`, `reports/`, `data/`, and the model cache if you need reproducibility across machines. Do not publish raw requests, responses, logs, or ledgers without reviewing them for query/document content and account metadata. The API key is not serialized by the client.

## Evaluation and interpretation

The evaluator computes graded nDCG@10, thresholded recall/precision/MRR, judgment coverage, candidate recall, and oracle diagnostics from saved qrels and rankings. It exports rank-derived TREC scores for stable tie order while preserving original model scores separately. Incomplete-query results and an operational RRF fallback are explicitly separate. Local resource use, provider request usage, estimated reservations, and reported charges are separate measurements; a local cost ledger does not replace a Vercel-side spending limit.

Optional commands include `preflight`, `export-audit`, `paired-test`, `calibrate`, `import-fixed-pool`, `import-scores`, and `generate-rag` via `.venv/bin/python -m jev_eval`. Human assessment, immutable revision/protocol gates, leakage review, and independent calibration labels are still required for a confirmatory claim. The Vercel catalog currently exposes a mutable Jev alias, so the confirmatory revision gate is unresolved. Consult the original guide before expanding beyond the pilot.
