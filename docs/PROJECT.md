# Project walkthrough

## Purpose and current status

This repository implements the retrieval study in the [original guide](Jev_RAG_Retrieval_Final_Guide.md). Every fixed-pool system reranks or selects from the same frozen candidate pool so comparisons isolate the scoring method. The 30-query SciFact feasibility pilot completed through Vercel AI Gateway; it is not a confirmatory result. This worktree adds a separate direct TypeSafe full-corpus scan. Direct scan results remain pending until a TypeSafe key is configured and its preflight and scan complete. No superiority claim follows from the pilot alone.

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
- `scan.py`, `decisions.py`: optional Jev full-corpus retrieval comparison and threshold/set-sufficiency decision benchmarks. Their live calls are separately budgeted and never part of the default five scripts.

The core configurations cover SciFact, NFCorpus, FiQA, and ArguAna. Dataset-specific rubrics matter: SciFact seeks evidence for or against a claim; ArguAna seeks counterarguments and excludes self-matches. The pilot's default fusion is fixed RRF. Weight search and logistic baselines must be fit on development data, not the test split.

## Jev as a first-stage retriever

The main five-phase workflow first uses BM25 and dense cosine similarity to find candidates, then asks Jev to rerank their English text. The additional `jev-scan` arm tests a different question: can Jev find relevant documents directly from the whole corpus? It sends each frozen query/document text pair to Jev's independent Noul rubric, without embeddings or retrieval scores. It ranks all documents by Jev probability and compares Jev, BM25, and dense cosine top results on the **same selected queries**, with corpus-denominator nDCG@10, recall@10, MRR@10, and candidate recall at the configured retrieval depth. This is exhaustive model scoring, not a scalable Jev index.

The existing SciFact preparation and retrieval artifacts can be reused for the direct TypeSafe scan. Run `.venv/bin/python -m jev_eval preflight` with a direct TypeSafe key first. The scan writes its own scores and uses `runs/budget_typesafe.sqlite`; it does not consume the earlier Vercel checkpoint. For the pilot, `30 × 5,183 = 155,490` pair evaluations. The script requires an exact pair-count acknowledgment and a request/token budget covering direct attempts, retries, and the sentinel. Check the current TypeSafe price and account limits before a large run.

```bash
./scripts/06_jev_scan.sh 1 5183 REQUEST_BUDGET TOKEN_BUDGET
# Later, to extend the same checkpoint to all 30 pilot queries:
./scripts/06_jev_scan.sh 30 155490 REQUEST_BUDGET TOKEN_BUDGET
```

Replace the budget placeholders with explicit positive integers. The direct scan defaults to 12 workers and at most 10 request starts per second. Add `--workers N --requests-per-second RATE` after the four required arguments to change these ceilings. A 429 or 529 response pauses all workers for the provider's `Retry-After` period when provided and halves the send rate; sustained successful requests gradually restore it up to the chosen ceiling. TypeSafe currently publishes 1,200 requests per minute and 250,000 tokens per second, but says the limits can change. Each worker owns its HTTP and SQLite connections. One process writes the checkpoint, and an exclusive direct-scan lock rejects a second copy.

The scan checkpoints scores every 100 completed documents and at each completed query to `jev_scan_typesafe_scores.json`; successful direct requests are also cached in the direct ledger, so a rerun can recover work since the last score checkpoint. It prints document progress at each checkpoint. `retrieval_typesafe_summary.json`, `retrieval_typesafe_per_query_metrics.json`, and `retrieval_typesafe_*.trec` contain the matched comparison. Raw direct responses and sentinels are stored under `typesafe_scan/`; the Vercel scan files remain separate. Do not compare these full-corpus retrieval results directly with fixed-pool reranking scores as though the tasks had the same candidate access or cost.

## Acceptance and evidence sufficiency

`07_decisions.sh` is another optional benchmark. It requires complete fixed-pool Jev scores from `04_rerank.sh jev`. At a threshold frozen before test evaluation, it compares Jev's yes/no document acceptance against accepting a fixed top-k from the same pool. Pair-level precision, recall, and acceptance rate use **explicitly judged pairs only**; unjudged pairs are never converted into negatives. The two methods need not accept the same number of documents, so report acceptance rate alongside quality and do not call this a matched-coverage comparison.

For each query, the script also sends the selected top-k English documents together to Jev and asks whether the set is sufficient for the task. It compares that decision with two basic rules: accept if the highest individual Jev probability exceeds the same threshold, or always accept. The available BEIR-derived proxy records whether the selected set contains at least one **known** relevant document. That proxy is not an independently verified answer-sufficiency label: a set can contain a relevant document yet lack complete support, and an unjudged document may supply support. Interpret set-level proxy metrics as diagnostics only. A real sufficiency/abstention claim requires an independently assessed set-level or answer-level audit.

```bash
./scripts/07_decisions.sh FROZEN_THRESHOLD REQUEST_BUDGET TOKEN_BUDGET
```

Choose the threshold using development queries, then keep it unchanged on held-out queries; do not tune it on the test split. Request/token budget arguments are explicit positive integers and share the Vercel ledger and `.env` monetary cap. This command makes one set-level Jev evaluation per query plus a daily sentinel, with retries if necessary. `decision_scores.json` stores resumable set probabilities; `decision_benchmark.json` stores all pair/set rows, confusion counts, precision, recall, acceptance rates, threshold, and label-scope warnings. Raw responses, resource telemetry, and timestamped logs remain under the same run directory. The four existing BEIR configurations already cover distinct domains, so no additional dataset is needed for the initial comparison; stronger claims still require held-out runs and human labels.

## What is preserved

All paths below are under `runs/<experiment_id>/` unless noted. `logs/` contains a new UTC-stamped stdout/stderr transcript for every scripted phase invocation; a nonzero pipeline exit is retained as such. `resources_prepare.json`, `resources_retrieve.json`, `resources_rerank_{qwen,bge,jev}.json`, and `resources_evaluate.json` record elapsed time, peak RSS, swap, platform, and success/failure. They are summaries, not GPU-profiler traces.

Preparation keeps `dataset_manifest.json`, `model_manifest.json`, `input_config.json`, `resolved_config.{json,yaml}`, `corpus.sqlite`, `queries.json`, `qrels.json`, and `text_views.json`. Retrieval keeps the BM25 index and scores, `embeddings.npy`, `embedding_checkpoint.json`, row IDs, `pools.json`, `source_runs.json`, and `candidates.parquet`. Reranking keeps `scores_<system>.json`, Jev raw requests/responses and model catalog, plus the shared `runs/budget.sqlite` ledger. Evaluation keeps `per_query_metrics.{json,parquet}`, `scores.parquet`, `summary.json`, selections, TREC files, and plots; `reports/<experiment_id>.md` is a compact human-readable summary. JSON checkpoints are atomically replaced. Raw scores and per-query rows are preserved so later analyses need not rely on rounded report values. The optional Jev scan adds its own scores, retrieval metrics, TREC exports, raw responses, and log.

Generated files are git-ignored and can be large. Back up `runs/`, `reports/`, `data/`, and the model cache if you need reproducibility across machines. Do not publish raw requests, responses, logs, or ledgers without reviewing them for query/document content and account metadata. The API key is not serialized by the client.

## Evaluation and interpretation

The evaluator computes graded nDCG@10, thresholded recall/precision/MRR, judgment coverage, candidate recall, and oracle diagnostics from saved qrels and rankings. It exports rank-derived TREC scores for stable tie order while preserving original model scores separately. Incomplete-query results and an operational RRF fallback are explicitly separate. Local resource use, provider request usage, estimated reservations, and reported charges are separate measurements; a local cost ledger does not replace a Vercel-side spending limit.

Optional commands include `preflight`, `export-audit`, `paired-test`, `calibrate`, `import-fixed-pool`, `import-scores`, and `generate-rag` via `.venv/bin/python -m jev_eval`. Human assessment, immutable revision/protocol gates, leakage review, and independent calibration labels are still required for a confirmatory claim. The Vercel catalog currently exposes a mutable Jev alias, so the confirmatory revision gate is unresolved. Consult the original guide before expanding beyond the pilot.
