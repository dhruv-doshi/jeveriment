# Jev retrieval evaluation

A reproducible retrieval benchmark for Jev, compared with lexical, dense, and local reranking baselines. The existing SciFact pilot used Vercel AI Gateway. This worktree adds a separate full-corpus scan through the [direct TypeSafe API](https://docs.typesafe.ai/api), with its own credentials, budget ledger, checkpoint, and output files.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository and run:

```bash
./scripts/01_setup.sh
```

On a fresh clone, setup creates `.env` and stops. Enter your `TYPESAFE_API_KEY` and confirm `JEV_MAX_COST_USD`, then rerun it. The script creates the local `.venv` from `uv.lock`, checks the environment, and validates the pilot config. The direct scan also requires `.venv/bin/python -m jev_eval preflight` with that key before its first run.

When you are ready to run the pilot, execute each phase in order:

```bash
./scripts/02_prepare.sh
./scripts/03_retrieve.sh
./scripts/04_rerank.sh
./scripts/05_evaluate.sh
```

`04_rerank.sh` can also run one system at a time: `./scripts/04_rerank.sh qwen`, then `bge`, then `jev`. Re-running a phase resumes validated checkpoints. Set `CONFIG=configs/core_nfcorpus.yaml` (or another config) to use a different experiment ID and output directory. Check its confirmatory gates before treating results as a study.

Before a direct TypeSafe Jev rerank, run `.venv/bin/python -m jev_eval preflight` to create `manifests/capabilities_typesafe.json`. The default 1,800-request and 2-million-token caps fit the 30-query pilot, but not a full 300-query SciFact test run with 100 candidates per query. Set `JEV_RERANK_REQUEST_BUDGET` and `JEV_RERANK_TOKEN_BUDGET` when running `04_rerank.sh jev`; these execution caps can be increased without changing the frozen experiment config or completed local scores. The caps include earlier direct TypeSafe attempts recorded in `runs/budget_typesafe.sqlite`; the separate `JEV_MAX_COST_USD` limit in `.env` still applies.

The prepared SciFact data and retrieval files are already available in the linked `runs/` directory for the direct scan, so you can go from direct preflight to `06_jev_scan.sh`. Use a new experiment ID if you also want direct fixed-pool reranking; the existing `scores_jev.json` belongs to the Vercel pilot.

To test Jev as the **retriever itself**, use the separate, opt-in [full-corpus scan](docs/PROJECT.md#jev-as-a-first-stage-retriever) after Phase 3. It scores raw query/document text, not embeddings, and compares Jev's top results with BM25 and dense cosine retrieval. It is not included in the default five scripts because the 30-query SciFact pilot requires 155,490 Jev document evaluations.

The direct scan uses 12 request workers by default, capped at 10 request starts per second, and slows all workers on TypeSafe 429 or 529 responses. Add `--workers N --requests-per-second RATE` after its four required arguments to change the limits. Progress is logged every 100 completed documents; rerunning resumes direct scores and cached responses. Existing Vercel scan scores remain separate.

An additional [decision benchmark](docs/PROJECT.md#acceptance-and-evidence-sufficiency) tests Jev's document-acceptance threshold against fixed top-k selection, and Jev's set-level evidence judgment against simple acceptance rules. Run it explicitly after Jev fixed-pool reranking; it is not a substitute for a human answer-sufficiency audit.

Outputs, raw API responses, checkpoints, metrics, plots, resource records, and timestamped command logs are saved under `runs/<experiment_id>/`; a readable report is written to `reports/<experiment_id>.md`. These generated directories and `.env` are git-ignored. Back them up separately if you need to retain experiment data.

For architecture, outputs, limitations, and analysis guidance, read [the project walkthrough](docs/PROJECT.md). The [original study guide](docs/Jev_RAG_Retrieval_Final_Guide.md) and [verification notes](TESTING.md) provide further detail.
