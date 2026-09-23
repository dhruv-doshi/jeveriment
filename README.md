# Jev retrieval evaluation

Python 3.11 implementation of the three required retrieval tracks in
`Jev_RAG_Retrieval_Final_Guide.md`, using Jev through Vercel AI Gateway.

**Execution is paused before experiments, at the user's request.** No real
dataset experiment, model-weight download, pilot ranking, or held-out study has
been run. Tests use authored fixtures and tiny randomly initialized local models.
The only live model requests were synthetic Jev capability checks.

## Setup and checks

Dependencies are pinned in `uv.lock`. Secrets belong in the ignored `.env`.

```bash
UV_CACHE_DIR=.cache/uv uv sync --extra local
.venv/bin/python -m jev_eval check-env
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
```

`AI_GATEWAY_API_KEY` authenticates the TypeSafe-compatible Vercel API.
`JEV_MAX_COST_USD` caps the initial preflight/pilot allowance. A shared SQLite
ledger includes all attempts across runs; ambiguous errors keep their estimated
reservations. `reported_cost_usd` and `uncertain_reserved_usd` are separate. Local
reservation accounting cannot replace a provider-side budget or guarantee the
provider's billing behavior after a timeout. API keys never enter saved requests.

## Implemented workflow

| Guide phase | Implemented code | Execution status |
|---|---|---|
| 0 | Locked environment; strict IDs, hashes, qrels and pool contracts; config gates; Gateway preflight; budget ledger | Local tests and synthetic live preflight completed; Score disabled |
| 1 | Corpus-denominator graded nDCG, thresholded recall/precision/AP/MRR, judgment coverage, oracle/retention diagnostics, TREC export | Checked against hand calculations and `ir_measures` |
| 2 | Streaming BEIR-to-SQLite ingestion, frozen revisions/shared text, BM25S, Qwen/GTE dense adapters, resumable vector memmaps, deterministic pools | Synthetic integration tested; actual corpus/model execution pending |
| 3 | Independent/contextual/structured Noul, semantic request cache, bounded retries, daily sentinels, Qwen/BGE rerankers, incomplete-query fallback | HTTP fixtures, live Noul fixtures and tiny-model CPU forwards tested; pilot pending |
| 4 | Fixed-k selection from identical pools; candidate recall and oracle limits | Tested; measured depth curves pending experiments |
| 5 | RRF, min–max weighted fusion, 0.1 simplex search, matched tuning budgets, judged-only logistic feature baseline | Tested; development fitting/frozen weights pending |
| 6 | Blinded two-assessor audit exports/import, logistic recalibration, Brier/log loss/ECE, risk/coverage, query bootstrap, paired macro tests, Holm | Tested; human judgments and study inference pending |
| Optional 7–8 | Remote fixed-pool import, external scorer import, local pinned RAG generation, MMR, alternative evidence-set coverage and independent claim/citation audit helpers | Code supplied; model/dataset/language selection and experimental execution pending |

The four core dataset configuration templates are under `configs/`. They remain
non-confirmatory until revisions and the experimental protocol are frozen.
SciFact uses evidence-for-or-against semantics; ArguAna uses counterarguments and
self-match exclusion. Qwen embeddings use last-token pooling and query
instructions; GTE uses CLS pooling. Qwen reranking uses its yes/no template; BGE
retains raw logits. No method silently truncates text after the shared view is
frozen. The local builder rejects corpora over 100,000 documents until a larger
indexing adapter is profiled.

BM25 defaults are k1=1.2 and b=0.75. Development sweeps can use the guide's grid
through `bm25_k1`/`bm25_b` in separate run configs. `make_pool` supports lexical,
dense, hybrid and the distinct B100∪D100 union. The default CLI builds H_C.
`tune_matched` searches equally sized deterministic subsets of the 0.1 simplex
grids when feature counts differ; it records the full grid size and actual trial
count. `tune` can run the full grid for separately labeled exploratory analysis.

## Live capability findings

See `manifests/capabilities.json` and the ignored `runs/preflight_v1/raw/` records.
The verified account passed Noul positive/negative, multiple-relevant,
all-irrelevant, repeated-input, ID-renaming, permutation, duplicate, and
1/5/20/50-question fixtures. Choice with `none` and 255 options also passed.

Score returned a value inconsistent with its rounded probabilities (one response
reported 1.63 while its probabilities imply 1.64). Strict validation rejects it;
the experiment adapter enables only Noul. Do not loosen the check or retry until
a preferred score appears. A future provider-contract change requires a new
validation protocol.

The catalog exposes a mutable Jev alias, not a verified immutable revision.
Account-specific rate limits and the maximum accepted state size are not measured.
The catalog listed nonzero input pricing while successful synthetic responses
reported zero billed cost; accounting retains both pricing and response metadata.
These findings are capability observations, not relevance-accuracy estimates.

## Experiment commands — deliberately not executed

Run these only when the user starts the experiment:

```bash
.venv/bin/python -m jev_eval validate-config configs/pilot.yaml
.venv/bin/python -m jev_eval prepare configs/pilot.yaml
.venv/bin/python -m jev_eval retrieve configs/pilot.yaml
.venv/bin/python -m jev_eval rerank configs/pilot.yaml --system qwen --resume
.venv/bin/python -m jev_eval rerank configs/pilot.yaml --system bge --resume
.venv/bin/python -m jev_eval rerank configs/pilot.yaml --system jev --resume
.venv/bin/python -m jev_eval evaluate configs/pilot.yaml
.venv/bin/python -m jev_eval plot runs/scifact_train_pilot_v1
```

Models load one at a time in separate commands. Use `device: cpu` if MPS execution
fails; do not silently alter the input length. The pilot samples 30 training
queries by a frozen seed/ID hash and retains the complete SciFact corpus. Expect
1,500 Noul pair evaluations plus preflight, sentinel, and error-only retries.
No calibration or significance claim should be made from this feasibility pilot.

Optional extensions use `import-fixed-pool CONFIG BUNDLE.json`,
`import-scores RUN_DIRECTORY NAME ARTIFACT.json`, and
`generate-rag RUN_DIRECTORY GENERATOR_CONFIG.json --systems rrf qwen jev`.
These commands are not executed automatically. Remote imports validate pool/text
hashes and retain remote first-stage provenance. Generation uses one frozen local
model, deterministic decoding, and the same evidence-token cap across conditions;
human claim/citation audits remain separate from generation.

`prepare --source PATH` accepts an existing BEIR directory instead of downloading
a release. Model and dataset manifests, exact text views, embedding checkpoints,
candidate Parquet, raw responses, score files, TREC runs, per-query metrics,
resource records, and Markdown reports remain inspectable and resumable.
TREC export uses rank-derived scores to preserve the declared ascending-ID tie
policy in external evaluators; original numerical scores stay in score artifacts.

## Statistics and human audits

`paired-test INPUT.json OUTPUT.json` accepts
`dataset -> query_id -> [baseline_metric, challenger_metric]` and preserves equal
dataset weights. `holm` in `stats.py` adjusts the declared primary test family.
Incomplete-query analysis and operational fallback are separately labeled.

`export-audit CONFIG --queries N --pairs 20` creates a blinded task file and a
separate private ID/sampling manifest. Use disjoint development and test queries.
Both assessors must label every pair; disagreements require adjudication and
insufficient information stays separate. `audit.import_judgments` joins scores
only after adjudication. `calibrate DEV.json TEST.json OUTPUT.json` evaluates raw
and recalibrated probabilities, requiring explicit independent human labels.

Before any confirmatory study, archive dataset terms and training-exposure
disclosures, freeze development choices and model versions, and complete the
human audit. If immutable Jev pinning remains unavailable, a separately documented
time-bounded protocol is needed; this implementation does not silently bypass the
immutable-revision gate. Optional multilingual, remote/SOTA and end-to-end RAG
studies still require their dataset/language/generator choices and audits.

References: [Vercel TypeSafe API](https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe),
[Qwen reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B),
[Qwen embeddings](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
[GTE](https://huggingface.co/Alibaba-NLP/gte-modernbert-base),
[BGE](https://huggingface.co/BAAI/bge-reranker-v2-m3).
