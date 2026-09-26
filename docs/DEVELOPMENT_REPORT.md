# Development verification

Branch: `feature/production-rag`. This report records development checks through 2026-09-26. The full five-dataset matrix and live-provider load test have not run.

## Source files and joins

Downloaded 26 pinned small files (308 MB on disk) into ignored `data/sources`. The download manifest records every file's URL, byte count, and SHA-256. Real `rag prepare` completed for MIRAGE (7,663 questions, 125,847 textbook chunks) and QASPER (5,049 questions, 113,218 paper units, zero missing gold evidence IDs). QASPER's 12,761 annotations map to paragraphs, headings, and figure/table captions. CLERC query/qrel IDs join; HotpotQA and KILT-FEVER labels parse. Their full passage/Wikipedia joins wait for a VM, as requested by the owner. [SOURCE_AUDIT.md](SOURCE_AUDIT.md) has exact revisions, counts, source terms, and the remaining gate.

## Pinned local models

All four models loaded from the exact commits in `configs/rag/model_revisions.json` using cached weights and CPU inference. The smoke used the original checkout's Python environment (`torch 2.14.0`, `transformers 4.57.6`, `sentence-transformers 5.7.0`) because the separate worktree's `faiss-cpu` extra could not finish downloading during this session. `scripts/smoke_local_models.py` saved the machine-readable result under ignored `runs/production_rag_v1/preflight/local_models.json`. On one synthetic question and two passages, every model produced finite output and ranked the relevant passage above the irrelevant passage. Qwen embedding returned 1,024-dimensional normalized vectors. Cold load plus tiny batch took 5.80 s for Qwen embedding, 0.28 s for MiniLM, 2.92 s for BGE, and 2.75 s for Qwen reranker on this Mac. These are smoke timings, not throughput or p95 estimates; FAISS indexing and GPU behavior remain untested.

## Live synthetic provider preflights

No benchmark passage was sent to a provider. Both preflights used synthetic text and the shared $50 development ledger.

| Provider | Observed result | Timing and spend |
|---|---|---|
| TypeSafe | `jev-1.13.0` requested and returned; valid Noul probability 0.98; usage 335 input / 22 output tokens; HTTP 200. | Client API time 388 ms. No `Server-Timing`, inference duration, or rate-limit headers were returned. Estimated ledger charge $0.00001407 from the [public JEV input price](https://typesafe.ai/blog/introducing-system-one-models-and-jev); account-specific price and RPM/TPM remain unknown. |
| OpenRouter | `moonshotai/kimi-k2.5` requested and returned through pinned `SiliconFlow`; metadata identified backend model `moonshotai/kimi-k2.5-0127`. Valid structured JSON answer and citation; usage 101 prompt / 20 completion tokens, zero reported reasoning tokens. | Client API time 2,942 ms. The metadata endpoint first returned 404, then supplied provider `latency=1394` and `generation_time=2216`; these are stored as provider-reported fields, not labeled model inference time. Billed cost $0.00009045 from OpenRouter usage/metadata. |

The first OpenRouter preflight revealed that `max_tokens=128` still incurred 287 billed completion tokens, including 267 reasoning tokens. The client now requests `reasoning.enabled=false`, sets `max_completion_tokens`, reserves that complete billed output bound, verifies actual usage, and retries delayed generation metadata. A second live preflight confirmed 20 completion tokens with no reasoning tokens. The two OpenRouter calls plus the TypeSafe call consumed $0.00079527 in the shared development ledger; metadata GETs cost zero. The first OpenRouter completion and the metadata lookups remain in the raw attempt log for audit.

The current OpenRouter `/key` response showed a **$10 key spending limit with about $5.08 remaining** at check time. Its `rate_limit` field says it is deprecated, so it is not a usable RPM value. The experiment's $50 local cap cannot raise the key limit. TypeSafe gave no account rate header. No rate-limit or concurrency claim follows from these one-call preflights; configure conservative account limits before a paid batch.

## Code verification

- `UV_CACHE_DIR=/private/tmp/jeveriment-uv-cache uv lock --check --offline`: passed; 106 locked packages.
- `.venv/bin/python -m pytest -q`: 23 passed; one installed FastAPI/Starlette test-client deprecation warning.
- `.venv/bin/ruff check .`: passed.
- The earlier fixture matrix dry run returned two matched cells and zero provider calls; a two-query BM25/retrieval-order fixture executed with two workers and replayed saved artifacts. The local fault-injection load fixture completed 20/20 requests after 24 HTTP attempts. These fixtures are not live quality or load results.

## Before the full experiment

On the VM, obtain the three full corpora, convert HotpotQA's official intro archive to sentence-preserving `wiki.jsonl`, verify CLERC/HotpotQA/KILT evidence joins, check source terms for hosted passage use, and measure storage/index build time. Freeze the labeled reporting split and run forecast, record actual provider limits and key balance, then use a separate owner instruction for the 75-cell matrix or real-provider stress test.
