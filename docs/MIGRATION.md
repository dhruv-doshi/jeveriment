# From the fixed-pool pilot to the RAG experiment

`feature/production-rag` is a clean branch for the five-dataset study. The BEIR/SciFact pilot, its commands, configs, prompts, and task-specific tests remain on `feature/parallel-jev` and in Git history. Existing run data in the other worktree was not moved or rewritten. Old pilot artifacts are not accepted as new RAG artifacts without an explicit converter.

The new branch directly retains `io.py` for canonical hashes and atomic JSON writes, `metrics.py` for ranking metrics, `stats.py` for paired statistics, and `ledger.py` as the starting point for a shared budget. Pilot-specific provider and model code was removed; useful behavior will be ported behind the new plugin contracts with new tests.

The new entry point is `jev-eval rag`. Its current `validate` and `plan` commands only inspect the experiment specification. The config selects MIRAGE, CLERC, HotpotQA, QASPER, and KILT-FEVER; BM25, Qwen dense, and hybrid retrieval; retrieval-order, MiniLM, BGE, Qwen, and standard JEV reranking; and OpenRouter Kimi K2.5 for fixed generation under a $50 ceiling. Dataset source revisions, provider routes, prompts, and prices must be frozen before a real run.

`rag/contracts.py` and `rag/artifacts.py` establish the new stage record and content-addressed JSON format. The first artifact schema is version 1. This foundation supports fixture replay, but does not yet dispatch benchmark or provider work. Later milestones in the [implementation plan](jev_production_rag_plan.md) cover dataset adapters, plugins, generation, evaluation, and load tests. The full experiment awaits the owner's instruction.
