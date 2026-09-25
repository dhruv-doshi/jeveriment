# Jev retrieval evaluation

A reproducible, fixed-pool retrieval benchmark for [Jev on Vercel AI Gateway](https://vercel.com/ai-gateway/models/jev), compared with lexical, dense, and local reranking baselines. The pilot uses SciFact; other BEIR configurations are included. The benchmark is **not complete**: code and synthetic checks exist, but pretrained-model retrieval and the final study remain to be run.

## Start here

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository and run:

```bash
./scripts/01_setup.sh
```

On a fresh clone, setup creates `.env` and stops. Enter your `AI_GATEWAY_API_KEY` and `JEV_MAX_COST_USD`, then rerun it. The script creates the local `.venv` from `uv.lock`, checks the environment, and validates the pilot config. No separate `requirements.txt` is needed.

When you are ready to run the pilot, execute each phase in order:

```bash
./scripts/02_prepare.sh
./scripts/03_retrieve.sh
./scripts/04_rerank.sh
./scripts/05_evaluate.sh
```

`04_rerank.sh` can also run one system at a time: `./scripts/04_rerank.sh qwen`, then `bge`, then `jev`. Re-running a phase resumes validated checkpoints. Set `CONFIG=configs/core_nfcorpus.yaml` (or another config) to use a different experiment ID and output directory. Check its confirmatory gates before treating results as a study.

To test Jev as the **retriever itself**, use the separate, opt-in [full-corpus scan](docs/PROJECT.md#jev-as-a-first-stage-retriever) after Phase 3. It scores raw query/document text, not embeddings, and compares Jev's top results with BM25 and dense cosine retrieval. It is not included in the default five scripts because the 30-query SciFact pilot requires 155,490 Jev document evaluations.

The scan uses four request workers by default, capped at two request starts per second, and slows all workers when the Gateway returns 429. Add `--workers N --requests-per-second RATE` after its four required arguments to change the limits. Progress is logged every 100 completed documents; rerunning resumes scores and cached responses. Run only one scan for an experiment ID at a time.

An additional [decision benchmark](docs/PROJECT.md#acceptance-and-evidence-sufficiency) tests Jev's document-acceptance threshold against fixed top-k selection, and Jev's set-level evidence judgment against simple acceptance rules. Run it explicitly after Jev fixed-pool reranking; it is not a substitute for a human answer-sufficiency audit.

Outputs, raw API responses, checkpoints, metrics, plots, resource records, and timestamped command logs are saved under `runs/<experiment_id>/`; a readable report is written to `reports/<experiment_id>.md`. These generated directories and `.env` are git-ignored. Back them up separately if you need to retain experiment data.

For architecture, outputs, limitations, and analysis guidance, read [the project walkthrough](docs/PROJECT.md). The [original study guide](docs/Jev_RAG_Retrieval_Final_Guide.md) and [verification notes](TESTING.md) provide further detail.
