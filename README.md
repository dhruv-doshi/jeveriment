# JEV RAG evaluation

This branch is the clean implementation base for the five-dataset RAG reranker study. The old BEIR/SciFact pilot remains in Git on `feature/parallel-jev`; its scripts and run files are intentionally absent here.

The [implementation plan](docs/jev_production_rag_plan.md) defines the plugin contracts, datasets, baselines, metrics, cost and latency accounting, and the development/full-run boundary. The [experiment specification](configs/rag/experiment.yaml) records the selected five datasets, three retrievers, five reranker arms, OpenRouter generator, and $50 cap. The [migration note](docs/MIGRATION.md) lists what was retained and removed from the pilot.

The branch implements official-format source adapters, parallel BM25/dense/hybrid retrieval, five reranker arms, context packing, fixed OpenRouter generation, native and optional judged evaluation, stage replay, shared spend/rate gates, reports, a FastAPI service, and a local fault-injection load harness. See the [runbook](docs/RUNBOOK.md) and [source layout](docs/SOURCES.md). Large corpora, local model weights, and paid provider calls have not been exercised; the complete experiment awaits the owner's instruction.

The [development report](docs/DEVELOPMENT_REPORT.md) records passing checks and the remaining real-source and provider preflights.

```sh
uv sync --group dev --extra service --extra load --extra observability
uv run jev-eval rag validate configs/rag/experiment.yaml
uv run jev-eval rag plan configs/rag/experiment.yaml
uv run pytest
uv run ruff check .
```

`rag matrix` previews or runs selected dataset × retriever × reranker cells with bounded query workers and shared stage caches. It defaults to a dry run; use `--execute` only for a deliberately approved slice or later full experiment.

Never commit `.env`, downloaded datasets, model weights, indexes, or run artifacts. A full dataset or live-provider load run requires the owner's later instruction.
