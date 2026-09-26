# JEV RAG evaluation

This branch is the clean implementation base for the five-dataset RAG reranker study. The old BEIR/SciFact pilot remains in Git on `feature/parallel-jev`; its scripts and run files are intentionally absent here.

The [implementation plan](docs/jev_production_rag_plan.md) defines the plugin contracts, datasets, baselines, metrics, cost and latency accounting, and the development/full-run boundary. The [experiment specification](configs/rag/experiment.yaml) records the selected five datasets, three retrievers, five reranker arms, OpenRouter generator, and $50 cap. The [migration note](docs/MIGRATION.md) lists what was retained and removed from the pilot.

This branch currently implements only the clean configuration and replayable artifact foundation. It makes no provider calls. More modules will be built and development-tested before the owner authorizes the full experiment.

```sh
uv sync --group dev
uv run jev-eval rag validate configs/rag/experiment.yaml
uv run jev-eval rag plan configs/rag/experiment.yaml
uv run pytest
uv run ruff check .
```

Never commit `.env`, downloaded datasets, model weights, indexes, or run artifacts. A full dataset or live-provider load run requires the owner's later instruction.
