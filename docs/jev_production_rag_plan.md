# JEV in production-style RAG: implementation plan

**Status:** complete implementation plan; code and experiment pending

**Updated:** 2026-09-26

**Branch:** `plan/production-rag`

**Scope:** five existing benchmarks, pluggable retrieval and reranking, fixed primary generation comparisons, quality and operational evaluation

**Execution state:** plan only. Development may begin after the owner asks to proceed; the complete five-dataset evaluation and live-provider load test require a later explicit instruction from the owner. No implementation or experiment has begun in this branch.

## 1. Purpose and claim boundary

Build a reproducible RAG system that answers two questions:

1. Given the same retrieved candidate pool, when does a simple JEV reranker improve evidence selection and final answers compared with established rerankers?
2. At what quality, latency, throughput, failure rate, and cost does that result hold under realistic load?

The initial deliverable is a solid evaluation system with **standard pointwise JEV** as the JEV arm. A new JEV algorithm is not part of the committed scope. Calling JEV and sorting its scores is an integration and evaluation contribution, not an algorithmic novelty claim. The article should make dataset-specific claims and state the tested workload; it should not call JEV universally “production ready.” The recording system must allow later changes to scoring, selection, prompts, or evaluation to start from the nearest saved stage.

This plan chooses defaults so implementation can begin without waiting for all plugin choices. The few choices requiring the owner's judgment are in §15. No benchmark run should be described as a clinical or legal validation.

## 2. Experimental rules

- A dataset adapter owns the official corpus, queries, splits, references, and evidence labels. It never invents qrels from a model's predictions.
- Corpus text is parsed and chunked **once per dataset and corpus snapshot**. Retrieval variants use the same chunk IDs and text. Dataset-native units (case, article, passage, sentence) remain recoverable.
- For each `(dataset, corpus version, query, retriever, retrieval configuration)`, persist a candidate pool before running rerankers. Every reranker in that arm receives the exact same candidates in the same initial order and with the same displayed text.
- The primary reranker comparison fixes the generator, prompt, decoding settings, context budget, citation format, evaluation rules, and query set. Retrieval is a separate factor; results are reported by retriever × reranker, then summarized across datasets.
- Separate quality evaluation from load testing. Caches may be used for reproducible offline replay; production latency measurements must disclose warm/cold cache conditions and cannot silently replay cached provider responses.
- Record failed and timed-out queries in denominators. A fallback result is reported as a fallback, never as a successful response from the intended reranker.
- Keep validation/tuning queries separate from held-out reporting. Freeze prompts, cutoffs, model revisions, and judging rules before the held-out run.
- Report per-dataset results first. A macro mean can summarize compatible metrics, with equal dataset weight; never pool raw questions across unrelated tasks as the sole conclusion.

## 3. System boundary and plugin contracts

```text
dataset adapter → immutable corpus/chunks → retrieval plugin → saved candidate pool
                                                        ↓
                                              reranker plugin
                                                        ↓
                                        context selector / packer
                                                        ↓
                                               generator plugin
                                                        ↓
                                  native + supplemental evaluators

All arrows write versioned input/output records and trace spans.
```

| Plugin | Interface responsibility | First implementations | Fixed in a primary reranker comparison? |
|---|---|---|---|
| Dataset | Load official corpus, queries, splits, labels, allowed corpus scope, licensing metadata | Five adapters in §4 | Yes |
| Parser/chunker | Convert raw documents to stable chunks; preserve provenance and token counts | Dataset-aware paragraph/sentence boundaries, bounded chunk size | Yes |
| Retriever | Return ordered candidates with scores from a corpus/index | BM25, dense, hybrid RRF | Fixed within each reranker arm |
| Reranker | Score/order the saved pool, preserving candidate IDs and failures | No rerank, MiniLM, BGE, Qwen, simple JEV | Variable |
| Context selector | Deduplicate and pack ranked chunks into a fixed token budget | Deterministic rank-order packer | Yes initially |
| Generator | Produce answer and citations from question + selected context | OpenRouter primary adapter with one fixed model; local adapter optional | Yes |
| Evaluator | Compute applicable native IR/answer metrics and supplemental RAG metrics | Native scripts, `ir_measures`, Ragas | Yes |
| Trace sink | Persist each stage's inputs, outputs, usage, attempts, timing, errors | Local structured artifacts, OpenTelemetry spans | Always on |

The service boundary exposes a single versioned `answer(query, dataset_or_corpus, retrieval, reranker, generator)` request. Experiment configuration specifies plugin IDs and parameters; code does not branch on a particular benchmark or reranker name outside its adapter. A plugin registry validates capabilities before execution: e.g. a metric requiring evidence annotations is unavailable on a benchmark without them.

### Canonical records

Use versioned Pydantic schemas and content-addressed artifact references. Each record has `schema_version`, `run_id`, `trace_id`, `dataset_id/revision/split`, UTC timestamps, and stable query/corpus IDs.

| Record | Required content |
|---|---|
| `QueryRecord` | Original question/claim, task type, corpus scope, reference answers, native evidence/qrels, source IDs; answer choices stored separately from retrieval text where the benchmark requires it. |
| `DocumentRecord` / `ChunkRecord` | Native doc ID, chunk ID, title/section, source URI and version, text or immutable blob reference, content hash, character/token offsets, parsing/chunking revision. |
| `RetrievalRequest` / `CandidatePool` | Query text, index/model version, top-N, ordered candidate IDs, raw scores, scores' meaning, full candidate text hashes/references, missing/duplicate flags, elapsed time. |
| `RerankRequest` / `RerankResult` | Exact pool reference, rubric/prompt or model request, returned score per candidate, ranks, score type, truncation, batch layout, missing scores, exceptions and fallback provenance. |
| `ContextSelection` | Selected and dropped IDs, ordering, per-chunk tokens, packed prompt text, context-token budget, deduplication/truncation reason. |
| `GenerationRequest` / `GenerationResult` | Complete rendered messages or prompt, model and resolved revision, decoding parameters, final answer, structured citations, finish reason, usage, stream events where relevant. |
| `EvaluationRecord` | Metric name/version, applicable label set, score, per-query detail, judge model/prompt if used, metric availability/reason if absent. |
| `AttemptRecord` / `SpanRecord` | Per-attempt request/response artifacts, status, retry cause, queue wait, client timings, provider timing if available, token usage, price calculation, cache/fallback status. |

The trace sink stores full public-benchmark input/output payloads for audit, including raw provider JSON and rendered prompts, alongside a compact Parquet event table and run manifest. Keep large text in immutable compressed blobs rather than repeating it in every row. Never store API keys or authorization headers. For future private corpora, permit payload redaction or encryption while retaining hashes and provenance; label such runs as less reproducible. Resume by stable `(run, stage, query, configuration)` IDs without duplicating billing calls. Validate joins, ranks, and artifact hashes in a trace-integrity report.

### Replay from saved stages

Treat the pipeline as a dependency graph of immutable artifacts. Each plugin execution records its configuration hash, exact upstream artifact hashes, implementation revision, output hash, and whether it was computed live or replayed. A changed plugin invalidates only its output and downstream stages. The runner finds the latest compatible saved boundary and starts there; it never silently mixes artifacts from different corpus, model, prompt, or rubric versions.

| Later experiment | Reuse without another run | Work that must run again |
|---|---|---|
| New metric or judge rule | Saved labels, rankings, contexts, answers, citations and traces | Metric computation; a new LLM judge still requires judge API calls. |
| New ordering/formula over **saved JEV scores** | Retrieval pools, exact JEV request/response payloads, per-candidate probabilities and usage | Reorder, context selection, generation, answer evaluation if selected evidence changes. |
| New context-packing block | Saved retrieval and reranker outputs | Context packing, generation, final-answer evaluation. |
| New generator prompt/model | Saved retrieval, reranking, and selected context | Generation and answer evaluation. |
| New reranker with a different model/rubric | Saved candidate pools and chunk text | New reranker scoring and its downstream stages; new TypeSafe requests may incur cost. |
| New retriever | Saved corpus/chunks and benchmark labels | Retrieval, reranking for newly encountered query–chunk pairs, and downstream stages. |
| New chunking/corpus snapshot | Raw source downloads and query/label manifest if still compatible | Indexing, retrieval, reranking, and downstream stages. |

Capture **every candidate's** raw JEV answer and request input, including candidates outside final top-k, so future score transforms can use the complete saved pool. Replaying a different formula over these scores needs no JEV calls. A genuinely new JEV question/rubric, a larger candidate pool, or a different model has no corresponding saved answers and therefore needs fresh calls. A different context selection or generator cannot reuse the old final answer if its input changed. Replay latency/cost is a separate measurement from live service latency/cost; report it as such.

## 4. Five selected datasets

The five are chosen now. They cover medicine, law, multi-hop Wikipedia QA, scientific papers, and fact verification. Large scale comes chiefly from CLERC, HotpotQA's full-Wikipedia setting, and KILT-FEVER; QASPER is deliberately smaller but tests long-document evidence selection. Dataset adapters use public labels only, with no new annotation project.

| Dataset and official source | Primary task and corpus | Native evaluation available | Adapter rules and limitation |
|---|---|---|---|
| [MIRAGE](https://github.com/gzxiong/MIRAGE) | Medical multiple-choice RAG; 7,663 questions across five sub-benchmarks, using a pinned MedRAG corpus/configuration | Answer accuracy, per sub-benchmark and macro | Count MIRAGE as **one** dataset family. Use question-only retrieval: answer options are excluded from retrieval input. Its benchmark does not provide universal retrieval qrels; do not fabricate nDCG/MRR for missing labels. Report answer accuracy and available evidence diagnostics separately. Pin and document the selected MedRAG corpus and its own terms. |
| [CLERC](https://huggingface.co/datasets/jhu-clsp/CLERC) | Legal case retrieval plus retrieval-augmented analysis/generation; published data has separate retrieval and generation files and a large collection | Citation/positive-passage retrieval; generation references where supplied | Use official document/passage collections and their mapping; the published collection is roughly 18 GB compressed. Do not use the preview's 20 sampled negatives as the whole corpus. The [published qrels directory](https://huggingface.co/datasets/jhu-clsp/CLERC/tree/main/qrels) contains test retrieval judgments; create a declared development/held-out partition of those query IDs if tuning is needed. Check license before ingest; the card is incomplete. Do not reduce legal answer correctness to lexical overlap. |
| [HotpotQA](https://hotpotqa.github.io/) | Multi-hop open-domain QA over the fullwiki corpus | Answer EM/F1, supporting-fact EM/F1 | Use the **fullwiki** setting for primary retrieval; the distractor set supplies curated candidate documents and would overstate real retrieval quality. Preserve the benchmark's supporting-fact IDs and official normalization. Use labeled development data for held-out reporting; hidden test is not assumed scoreable locally. |
| [QASPER](https://huggingface.co/datasets/allenai/qasper) | Long scientific-paper QA; 5,049 questions over 1,585 NLP papers | Answer F1 and evidence-selection F1 from annotations | The official task gives the target paper. Restrict retrieval to that paper's chunks for the primary benchmark; cross-paper retrieval would be a separately named variant. Preserve multiple annotators, yes/no, free-form, extractive, and unanswerable cases. Small corpus, long-context stress. |
| [KILT-FEVER](https://huggingface.co/datasets/facebook/kilt_tasks) | Claim verification against the pinned KILT Wikipedia snapshot; FEVER config has 104,966 train and 10,444 validation claims | Label accuracy plus provenance/evidence retrieval | Use KILT's own knowledge source and provenance mapping, not HotpotQA's different Wikipedia snapshot. Track page and passage match separately. Verify availability of gold provenance by split; use validation for scoring where test evidence is hidden. |

For every adapter: snapshot source URL and revision/hash, exact split counts, corpus units, licenses and attribution, reference format, evaluation script version, and storage footprint. Check individual corpus licenses and data-use terms before sending text to a hosted API. Noncommercial research does **not** automatically grant permission to redistribute data or send it to third parties. Store dataset downloads outside Git with a reproducible manifest. Fail adapter validation when a referenced evidence ID cannot be mapped to the indexed corpus; report the unmapped fraction.

### Tuning and held-out split policy

The adapter declares a `development` query set and a disjoint `reporting` query set before any prompt/rubric or retrieval tuning. For official training/validation/test data, prefer training or validation for development and a labeled, locally scoreable held-out partition for reporting. Where only one labeled public benchmark partition exists, hash the native query ID plus seed 1729 into a 20% development / 80% reporting split; preserve the original IDs and say that the score is on this derived split rather than the full official leaderboard set. A question's multiple annotations and its source paper/case stay together. No answer, qrel, or supporting-fact field enters retrieval or reranking requests.

| Dataset | Planned development data | Planned locally scoreable reporting data |
|---|---|---|
| MIRAGE | Deterministic 20% of each of its five published subtasks | Remaining 80% of each subtask; also report subtask counts. |
| CLERC | Generation `train` for answer-format work; deterministic 20% of public test retrieval queries if retrieval tuning is needed | Remaining 80% of public test retrieval queries and untouched generation `test`, subject to ID/schema audit. Direct/indirect and document/passage qrels stay separate. |
| HotpotQA | Official train data, with any development sample declared | Labeled fullwiki development set; never claim locally scored hidden-test results. |
| QASPER | Official train/validation papers | Official test papers if their public annotations and evaluator are available; otherwise a predeclared holdout from labeled validation papers. Split by paper, not question. |
| KILT-FEVER | Official train claims | Labeled validation claims with available provenance; test labels are not assumed public. |

The source audit verifies every proposed split and the generator–retrieval query alignment before indexing. If any task lacks the expected scoreable references, keep its retrieval and answer results separate, mark unavailable metrics, and revise the split manifest **before** a report run. Do not choose replacements based on observed model performance.

## 5. Retrieval design

Start with three explicit arms using the same corpus/chunks and query text:

1. **Lexical:** Lucene BM25 via Pyserini, with index settings and analyzer recorded.
2. **Dense:** pinned Qwen3-Embedding-0.6B model, batched document/query encoding, cosine or dot-product normalization recorded, FAISS index with exact-search validation on a sample. This reuses the project's existing embedding baseline; it is not labeled universal SOTA.
3. **Hybrid:** reciprocal-rank fusion of lexical and dense rankings at a fixed predeclared depth/constant. RRF belongs to retrieval/fusion, not the reranker list.

Initial candidate depth: 100; also score sensitivity at 50 and 200 where feasible. Save top-200 retrieval pools once, then slice for controlled depth comparisons. For a truly large corpus, index sharding and storage requirements are an implementation concern, not a reason to change qrels or substitute a provided answer-containing pool. A stronger retriever/model can be added through the same contract after independent validation, with its own arm and version. Do not describe any retriever as SOTA across all five datasets without evidence.

Pre-rerank diagnostics: candidate recall at N, hit rate, MRR/nDCG where labels allow, unmapped gold evidence, duplicate rate, index size, build time, and query latency. If relevant evidence is absent from the pool, a reranker cannot recover it; distinguish retrieval ceiling from reranking failure.

## 6. Reranker implementations and comparison

| Arm | Purpose | Scoring behavior |
|---|---|---|
| Retrieval order | No reranking control | Keep input order and score; measures added value of reranking. |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Fast conventional cross-encoder | Score query–chunk pairs with pinned model/revision and fixed max tokens. |
| `BAAI/bge-reranker-v2-m3` | Strong multilingual/general cross-encoder baseline already present in project | Same pairs, no model-specific candidate filtering. |
| `Qwen/Qwen3-Reranker-0.6B` | Stronger, more expensive local baseline already present | Fixed prompt/instruction, truncation and precision/device recorded. |
| Simple TypeSafe JEV | Pointwise JEV baseline | One stable binary **Noul yes-probability** rubric asking whether the passage contains evidence useful for answering the query; rank by returned `noul` probability, with stable retrieval-rank tie break. |

JEV's Noul output is a probability for a binary proposition; a middle value expresses uncertainty, not a separate “maybe” class. Store rubric text and resolved model name. The TypeSafe account's available models come from `/v1/models`; an alias may resolve to a different model, so persist both requested and resolved names. Do not silently switch JEV mode, rubric, or candidate text across datasets. A domain-specific rubric is a separate registered variant, not an unnoticed tweak to the common baseline.

Initial common rubric, recorded verbatim with a rubric version: **“Does this passage contain evidence that would help answer the question accurately?”** The `state` includes the query and exactly one candidate passage with explicit delimiters and stable IDs; the output is the Noul yes probability. The common rubric is frozen after development fixtures and before reporting queries. A relevance score is not automatically a calibrated probability of answer correctness, and a later revised rubric starts a new experimental arm.

Common reranker limits: same max candidate count, same displayed chunk text, documented model-specific token limit/truncation, stable ties, missing-score policy, and fixed timeouts. Record mean and distribution of input tokens by model; different tokenization can make exact token equality impossible, so report both characters and model tokens. All rerankers get the same IDs even if one must truncate text. Local warm-up and provider cold start are reported separately.

The primary output is a complete reordered candidate list. The first context selector is deterministic: top-ranked unique evidence chunks packed until the same generator token budget is reached, preserving source citations. Compare rank top-k **and** actual selected context, because token lengths may cause different numbers of chunks to fit.

Optional controls after the first complete run: MMR/evidence diversity, a larger cross-encoder if hardware allows, and a contextual or batched JEV variant. These are separately named experiments. They can consume compatible saved artifacts without changing the standard JEV baseline.

## 7. Generation and final answers

The **generator** is the language model that reads the question and the evidence selected after reranking, then writes the final answer. It is distinct from the retriever (finds candidates) and reranker (orders candidates). “Primary generator” means the **one fixed generator model** used across all reranker arms in the main answer-quality comparison. Keeping it fixed lets an answer difference be attributed to the changed evidence.

- **Primary provider and model:** OpenRouter with `openai/gpt-6-luna` for the main comparison. Its [OpenRouter listing](https://openrouter.ai/openai/gpt-6-luna/) currently shows $0.10/M input tokens and $0.50/M output tokens; recheck price and account access before bulk runs. This is a budget-conscious controlled generator, not a claim that it is the best medical or legal model. Kimi K3 is available, but its [current listed rates](https://openrouter.ai/moonshotai/kimi-k3) are materially higher, so it is not the default under the $50 cap.
- Pin one verified OpenRouter provider endpoint with `provider.only`, `allow_fallbacks: false`, and `require_parameters: true` for the controlled answer-quality run. Do not use `openrouter/auto` or a cross-model fallback list in that run. Persist requested and returned model, actual provider, routing request, model parameters, generation ID, usage, and cost. Provider failover is a separate operational experiment and must not silently change the quality comparison. [Provider routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection)
- Define a generator protocol with this OpenRouter adapter first and a local Hugging Face adapter as an optional later plugin. Use an initial 4,096-token evidence budget and 768-token maximum answer, then freeze task-specific budgets on validation data before held-out comparisons; every reranker arm in a task uses identical budgets. Any model or budget change creates a new comparison run that reuses compatible earlier stage artifacts.
- Start with **one** generator model to preserve budget and a clean reranker comparison. If a second model is useful for robustness, use a matched, prespecified subset and replay saved retrieval/reranking/context artifacts; only generation and final-answer evaluation need to run again. Do not mix generators in the same primary result table without labeling the generator factor.
- One prompt template per task family, frozen across reranker arms. Cite evidence with stable document/chunk IDs. Medical answer options may be given to the **generator** according to MIRAGE rules but never to its retriever.
- Use deterministic decoding where supported (`temperature=0`, explicit seed if supported), max output tokens, and a fixed context budget. Record resolved model version and any nondeterminism.
- Preserve native answer format: MIRAGE option, HotpotQA short answer plus supporting facts, QASPER answer/abstention, FEVER label plus evidence, CLERC answer with citations. Reject malformed output into a counted format-failure category.
- Include a no-context generator control on a matched subset to estimate how much the supplied evidence changes answers. Do not turn its scores into reranker scores.
- Calibrate the abstain path on tasks with explicit unanswerable or insufficient-evidence labels (for example QASPER and FEVER). No invented unanswerability labels for other datasets.

## 8. Evaluation matrix

“All standard metrics” means the standard metrics applicable to each label type, not filling unsupported columns with invented numbers. Every metric row includes denominator, confidence interval, missing-label count, and exact implementation/version.

| Stage | Metrics | Applicability and interpretation |
|---|---|---|
| Candidate retrieval | Recall@50/100/200, hit@N, candidate oracle recall, MRR@N, nDCG@N | Gold evidence/qrels required. For MIRAGE without common qrels, show `not_applicable` and the reason. |
| Reranked list | nDCG@5/10, MRR@10, MAP@10, Recall@5/10, Precision@5/10, hit@10, judged@10, positive rank distribution | On datasets with mapped relevance labels. Report binary or graded gain explicitly; never conflate unjudged with irrelevant without saying so. |
| Packed context | Gold-evidence coverage/recall and F1, unique-source coverage, redundancy, used/available tokens, citation-source validity | Evaluate the chunks actually passed to generator, including evidence-set completeness on multi-hop tasks. |
| Final answer | Native accuracy, EM/token F1, supporting-fact F1, evidence/provenance F1, unanswerable precision/recall, citation precision/recall where references exist | Dataset-specific. CLERC's legal text needs its native task scoring and a bounded blind human review for correctness; lexical overlap alone is weak. |
| Supplemental RAG | Ragas context precision/recall, faithfulness, answer relevancy/correctness when inputs support them | LLM-judged diagnostics; separate from native primary endpoints. Pin judge model/prompt and test agreement on a stratified human-reviewed sample. |
| Reliability | End-to-end success, format failures, abstentions, fallbacks, incorrect citations, unsupported claims, error categories | Always report on all attempted queries, including failures. |

Native evaluators are authoritative when available; `ir_measures` computes standard ranked retrieval metrics. Ragas augments final-answer analysis, with cost and judge variance reported. For legal and medical outputs, add a **small, prespecified, blinded** human audit of factual support and harmful/unsupported claims if the claims extend beyond native benchmark scores; this is an audit, not a new dataset-building effort. Reviewers see paired outputs without reranker names. If no human review is done, narrow the article's claims accordingly.

Statistical analysis: paired per-query deltas within each fixed candidate pool, bootstrap confidence intervals clustered by query (and paper/case when multiple questions share a source), Holm-adjusted comparisons for the declared primary endpoint, and effect sizes alongside p-values. Predeclare one primary endpoint per task family (MIRAGE accuracy, HotpotQA answer + support, QASPER answer/evidence, FEVER label/evidence, CLERC citation/evidence retrieval plus audited final answer) and treat the rest as diagnostics. Publish per-dataset regressions and examples, not only an aggregate win rate. Report quality against fixed token/cost/latency budgets and a Pareto frontier.

## 9. Timing, throughput, and cost instrumentation

### Timing definitions

Use monotonic client clocks and OpenTelemetry spans. Capture `queue_wait`, `retrieve`, `rerank`, `pack`, `generate`, `evaluate`, and end-to-end durations. For each provider attempt, capture dispatch-to-response wall time, time to first byte where the client stack exposes it, body download/read time, retry backoff, and response parsing; keep connection setup measurements only when the HTTP transport can reliably expose them. Record warm versus cold start and cache status. Use p50/p95/p99 with sample count and confidence/uncertainty where useful; never compute p99 from a tiny sample without flagging it.

**API latency is measurable; server-side model inference time is not necessarily exposed.** TypeSafe's documented `/v1/systemone` response contains resolved `model`, `answers`, and token `usage`, but no inference-time field. The adapter checks for a documented response header such as `Server-Timing` during preflight and records a provider timing only if supplied with clear semantics. Otherwise store `provider_inference_ms = null` and `provider_timing_source = not_exposed`. Never subtract client network time from API duration and call the remainder model inference. Local-model inference time is measured around forward passes, with device synchronization for GPU timing. Generator time to first token is collected only for streaming APIs. OpenRouter's [generation metadata endpoint](https://openrouter.ai/docs/api/api-reference/generations/get-generation) exposes fields such as `latency`, `generation_time`, provider name, usage, and total cost; store these with OpenRouter provenance and verify units/semantics before labeling `generation_time` as model inference. They are additional to client-observed end-to-end API latency.

### Operational metrics

- Throughput: completed queries/s, retrieved candidates/s, reranked pairs/s, provider requests/s, input/output tokens/s.
- Tail latency: p50/p95/p99 by stage and end to end, including queue, retries, and failures; cold/warm separate.
- Load: concurrency, queue length, saturation, CPU/RAM/GPU utilization and memory, index I/O, local model batch utilization.
- Resilience: 429/5xx/timeout rates, retry amplification, rate-limit wait, canceled requests, fallback frequency, recovery time, correctness under fallback.
- Cost: provider billed tokens and USD per query/1,000 queries, retrieval/rerank/generation/judge components, local GPU-hours/energy proxy, storage/index cost. Store currency and price-table revision; distinguish actual invoices from estimates and zero-priced promotional requests from underlying token use.
- Stratification: query/context length bins, candidate depth, document length/truncation, task/dataset, concurrency, cache state, success/failure. Plot quality and latency against input tokens and candidate count.

The **$50 USD ceiling** is a hard global cap on incremental experiment spend across TypeSafe JEV, paid generator/judge APIs, and paid cloud compute/storage used for these runs. Free requests still count toward request/token/rate budgets and are logged at their actual $0 charge. The cost ledger reserves an estimate **before** dispatch and reconciles actual usage after response. It enforces request, token, and USD caps, with a hard stop at $50 and a traceable change only after the owner explicitly changes the budget. Because cloud invoices can lag, reserve projected instance hours/storage before starting paid infrastructure and stop when the remaining headroom cannot cover the next planned block. Do not start a bulk run whose forecast exceeds the remaining cap; scale down the matched sample or split the work into budgeted blocks and report coverage.

Forecast calls as `sum(query × retriever arm × unique candidate pairs)` for pointwise JEV, adjusted for reusable identical query–chunk–rubric scores and retry rates. For example, 1,000 queries × 3 retrieval arms × 100 candidates can approach 300,000 pair scores before deduplication. Compute the forecast from the selected splits before any bulk run. Caching may share a score only when query, chunk text/hash, rubric, model revision, truncation, and JEV mode match exactly; contextual scores are not interchangeable with independent scores.

At TypeSafe's [publicly stated $0.042 per million input tokens](https://typesafe.ai/blog/introducing-system-one-models-and-jev), 300,000 independent pairs averaging 512 billed input tokens would be about 153.6 million tokens, or **$6.45 for JEV input alone**. This is an illustration, not a run quote: actual prompt length, unique-pair overlap, retries, account credits, model pricing, OpenRouter generations/judges, and infrastructure change the total. The preflight must obtain an account-specific price/rate snapshot and a measured token distribution on a tiny development sample, then produce a per-stage upper-bound forecast before any paid batch. A $50 cap does not imply that all five complete datasets can be fully scored and generated; report exact coverage if the cap binds.

The planned comparison matrix is five datasets × three retrieval arms × five reranker arms (75 ranking cells), with the **same query IDs per dataset across its cells**. Stage 1 may compute local retrieval metrics on larger scoreable sets without JEV or generator spend. Stage 2 uses a deterministic, stratified matched subset for all five rerankers; target at least 200 eligible questions per dataset where available, then size the final sample using observed paired variance and the remaining budget before freezing it. Stage 3 generates answers for those same matched cells and uses Ragas judges only on a prespecified subset. A reduced sample is reported as reduced coverage, not a complete-dataset result. The full matrix and any live load profile are part of the owner-triggered experiment, not development testing.

## 10. Rate limits and load methodology

The TypeSafe adapter uses one account-wide async limiter shared by all workers: requests/minute and tokens/minute if known, bounded in-flight concurrency, idempotent retry rules, jittered exponential backoff honoring `Retry-After`, circuit breaking for sustained faults, and per-attempt budget accounting. Discover published/account-specific limits before tuning concurrency; an absent published number is recorded as unknown, not guessed. Rate limits may be dynamic, so a bounded calibration ramp measures observed throttling.

Load phases, after offline quality passes: single-user baseline; controlled concurrency ramp; sustained steady state; burst and recovery; provider slowdown/429/timeout fault injection; local model cold start and process restart. Use fixed query-mix and length strata from the five datasets. Preserve a clean benchmark mode with no induced faults. Run stress tests within provider terms and an approved spend ceiling. Define p95/p99 and availability SLOs before making any production-readiness claim. Report maximum sustainable throughput at the stated quality/error target, not just peak launch rate.

## 11. Tooling selected for implementation

| Need | Selected tool and role |
|---|---|
| Experiment/config/data contracts | Python, `uv` lockfile, Pydantic schemas, YAML run manifests; reuse the repository's existing `jev_eval` code where sound. |
| RAG composition | LangChain adapters for familiar retriever/generator integration; canonical records and ranking logic remain framework-neutral so a framework update cannot erase scores/IDs. |
| Dataset loading | Official files and Hugging Face `datasets` where appropriate; adapter pins revisions and validates schema. |
| Index/retrieval | Pyserini/Lucene BM25, FAISS dense index, Qwen3 embedding model, deterministic RRF fusion. Record all index/model versions. |
| Reranking | `sentence-transformers`/Transformers for local cross-encoders; direct `httpx` for TypeSafe JEV so response/attempt details remain observable. |
| Metrics | Official benchmark scripts + `ir_measures`; Ragas for supplemental RAG metrics. |
| Trace/analysis | OpenTelemetry spans; immutable JSON blobs + Parquet tables; DuckDB for queryable reports and plots. Optional LangSmith export if desired, never required for the canonical record. |
| Primary generation | Direct OpenRouter chat completion adapter using `httpx`, with `usage` requested, generation metadata reconciliation, fixed model/provider routing, and local artifact recording. |
| Service/load | FastAPI for an answer endpoint; Locust for repeatable ramp and burst profiles; standard host/GPU telemetry. |

Exact dependency versions and container image digest are pinned when implementation starts. `LangChain`, `Ragas`, and `LangSmith` are not used as interchangeable “evaluation frameworks”: LangChain composes calls, Ragas computes optional answer metrics, and LangSmith is optional hosted trace inspection. The canonical artifact store is local/exportable, so experiments do not depend on a paid tracing service.

Configuration reads `TYPESAFE_API_KEY` and `JEV_MODEL` for TypeSafe and `OPENROUTER_API_KEY` for generation. The run manifest records only whether credentials are present, never their values. Model IDs, routing settings, prompts, token limits, request budgets, dataset revisions, and the $50 cap live in versioned config, with command-line overrides echoed to the manifest. A missing key blocks only its live adapter; offline fixtures, local retrievers, metrics, and replay remain testable.

## 12. Run layout and reproducibility

One run directory per immutable manifest:

```text
runs/<run_id>/
  manifest.yaml                 # config, git commit, dataset/model/index revisions
  datasets.json                 # source checksums, license/terms review, split counts
  queries.parquet               # query IDs, task/split, references/provenance pointers
  pools/<retriever>.parquet     # saved ordered candidates
  stages/<plugin>.parquet       # normalized inputs/outputs and metrics per stage
  blobs/<hash>                  # exact text, prompts, raw API payloads (sanitized)
  traces.parquet                # spans, attempts, retries, status, usage, timing
  metrics.parquet               # per-query and aggregate scores with denominators
  reports/                      # plots, failure analysis, cost/quality frontier
```

Capture operating system, Python/dependency lock, CUDA/MPS/CPU device, GPU type, index settings, random seeds, timezone/UTC timestamps, requested and resolved provider model, price table/date, and evaluator/judge revision. Produce a machine-readable comparison report and a human-readable methods/report appendix. Artifact hashes and schemas allow a reader to reproduce both an aggregate number and an individual answer path. Keep secrets and downloaded corpora out of Git.

## 13. Implementation sequence and exit gates

This is the order for the **subsequent coding phase**. A milestone is done only when its gate passes; a smoke test on five queries is not the paper's result.

| Milestone | Build | Exit gate / evidence |
|---|---|---|
| 0. Source audit | Download manifests, license/terms notes, schema probes, split inventory, storage/cost forecast | Every selected dataset has readable official source, corpus mapping, scoreable validation partition, and documented restrictions; unresolved access is raised before bulk indexing. |
| 1. Core contracts and recorder | Plugin registry, canonical records, blob/Parquet trace sink, manifest, resume/idempotency | A fixture query flows through every stage; full inputs/outputs and attempt records join by ID; secrets are absent; replay yields identical ranking. |
| 2. Dataset adapters | MIRAGE, CLERC, HotpotQA fullwiki, QASPER, KILT-FEVER | Counts and sampled IDs match source; corpus and reference/evidence mapping checked; no train/test leakage; official eval subset passes known examples. |
| 3. Retrieval | BM25, dense, hybrid indices and frozen candidate pools | Candidate IDs/text deterministic; index smoke test and gold mapping audit pass; pool saved once and reused by every reranker. |
| 4. Baseline rerankers | No rerank, MiniLM, BGE, Qwen, simple TypeSafe JEV | Every plugin returns complete score/rank/failure records from the same pool; JEV usage, resolved model, and actual request count reconcile with ledger. |
| 5. RAG answer path | Fixed packer, OpenRouter generator adapter, task output parsers, citation mapping | `openai/gpt-6-luna` and verified provider routing fixed across reranker arms; native answer formats parse and citations resolve. |
| 6. Quality evaluation | Native metrics, IR measures, selected Ragas metrics, paired analysis, reports | Each metric's applicability and denominator explicit; reranking and answer results connect to traces; held-out settings frozen. |
| 7. Service/operations | FastAPI endpoint, shared rate limiter, fault policy, Locust profiles, cost dashboard | Stage/end-to-end p50/p95/p99, throughput, retries, 429/5xx, queue, usage, cost and fallback quality available; bounded ramp succeeds within cap. |
| 8. Replay and extension checks | Recompute new ranking formulas, context policies, prompts, and metrics from saved boundaries | No upstream provider call is repeated when inputs/configuration are unchanged; changed inputs invalidate only the necessary downstream stages; live and replay costs stay separate. |

Recommended execution scale: tiny fixture → one validation slice per dataset → all validation queries for baseline ranking metrics → matched answer-generation subset → held-out full answer run if affordable → operational workload. Sample queries deterministically and stratify by length/task/answerability, then report the sample definition. Keep all five datasets in the final suite even if early development starts with one. Scale production load independently from article-quality testing.

### Repository implementation map

Keep the existing fixed-pool `jev_eval` pilot commands and run artifacts readable. Build the new harness in a separate `src/jev_eval/rag/` package and expose it through new `jev-eval rag ...` subcommands; shared existing functions may be extracted only when their current tests remain green. Proposed ownership:

| Area | Planned files and responsibilities |
|---|---|
| Contracts/config | `rag/contracts.py`, `rag/config.py`, `rag/registry.py`: versioned records, capability checks, frozen run manifest and plugin registry. |
| Artifact/replay core | `rag/artifacts.py`, `rag/runner.py`: content-addressed blobs, Parquet records, atomic writes, stage dependency keys, resume and selective replay. |
| Data | `rag/datasets/{mirage,clerc,hotpotqa,qasper,kilt_fever}.py`: source readers, native IDs/splits, corpus/evidence mapping, task output formats; shared adapter protocol. |
| Retrieval | `rag/retrieval.py`, `rag/indexes.py`: BM25/FAISS adapters, RRF, frozen pools, index manifests and build validation. |
| Reranking | `rag/reranking.py`, `rag/clients/typesafe.py`: no-rerank/local models/JEV, exact request/response capture, limiter and retry ledger. |
| Answer path | `rag/context.py`, `rag/clients/openrouter.py`, `rag/generation.py`: token packing, fixed prompts, answer parsing, citations, provider metadata. |
| Evaluation/reporting | `rag/evaluation.py`, `rag/metrics.py`, `rag/reporting.py`: native metric wrappers, IR/Ragas registry, paired comparisons, quality/cost/latency reports. |
| Operations | `rag/service.py`, `rag/telemetry.py`, `rag/budget.py`, `rag/load/`: endpoint, spans, cost reservations, fake-provider and later real load profiles. |
| User-facing entry | `configs/rag/` frozen configs; `jev-eval rag audit-sources`, `index`, `retrieve`, `rerank`, `generate`, `evaluate`, `replay`, `report`, `serve`, and `load` commands; `tests/rag/` fixtures and contract tests. |

The command-line runner defaults to `--dry-run` for any command that could call a paid provider or create large indexes. A live command needs a frozen manifest plus an explicit run mode and budget forecast; `rag load` against real providers remains disabled until the owner-triggered experiment. Each command prints the stage boundary it will reuse, the exact number of pending calls, estimated worst-case spend, and artifact destination before dispatch.

Future algorithms can be evaluated from the stored stage boundaries when their required inputs already exist. A future method that asks JEV new questions or scores documents outside the saved pool will require those specific new calls. No novelty claim is planned for the standard JEV baseline.

### Development verification and permission boundary

The owner's instruction is to run all required **development tests** and wait for a separate instruction before the full experiment. Development tests may use synthetic documents and a handful of public benchmark examples. They must not launch the 75-cell matrix, bulk TypeSafe scoring, full answer generation, or live-provider stress testing. Local model smoke tests and at most a small, metered API preflight are allowed only after the $50 ledger and credential checks are working; every real request is counted against the global cap.

| Layer | Required development check | Pass condition |
|---|---|---|
| Schemas and artifacts | Valid/invalid records, stable content hashes, serialization round trips, incomplete/crash recovery | Malformed IDs, mismatched hashes, secret fields, and mixed revisions are rejected; replay resolves exact upstream artifacts. |
| Dataset adapters | Tiny official examples for all five adapters; split assignment and corpus/evidence joins | Native IDs, answer references, and qrels match the source; no label reaches scorer input; unmapped evidence is counted or blocks as specified. |
| Retrieval | Small hand-checked corpus for BM25/dense/hybrid; approximate-vs-exact dense sanity check | Deterministic ordered pools, stable ties, no duplicate IDs, expected positives recoverable at depth, exact candidate text preserved. |
| Rerankers | One/few query–passage fixtures for each local model; mocked TypeSafe responses and failures | All five arms preserve the same pool IDs; truncation, missing scores, retries, 429, timeout, and model mismatch are visible in records. |
| Generation and evaluation | Mock OpenRouter outputs for each task format; native metric gold examples | Prompt/context hashes and citations round-trip; format errors count; each metric's denominator and applicability are correct. |
| Budget/rate limits | Simulated concurrent workers, stale price, retry surge, insufficient remaining dollars/tokens | Shared limiter enforces caps; reservation precedes dispatch; no double charge on resume; at-cap dispatch is prevented. |
| Replay and tracing | Recompute ranking formula, context selection, generator, evaluator independently from saved boundaries | Identical config makes zero new external calls; a changed config invalidates only downstream artifacts; live versus replay timing/cost remains distinct. |
| Service/load harness | Locust against a local fake provider with controlled delay/429/5xx and telemetry | p50/p95/p99, queue, retries, throughput, fallback, and cost attribution match injected conditions without real provider load. |

Run the repository's existing test suite and lint on each implementation milestone, plus the targeted contract tests above. Record test commands, pass/fail counts, skipped optional-model checks, and fixture revisions in a development report. A single live TypeSafe and OpenRouter preflight can verify authentication, response schema, exact model/provider, token usage, and price/latency metadata when keys are available; avoid using a real benchmark question for this check if dataset terms have not cleared hosted inference. An absent key or unverified dataset license is a named untested integration, never a claimed pass.

Before the owner-triggered full experiment, publish a **run-ready packet**: clean code commit and lockfile, five adapter source audits, frozen split/query manifests, model/rubric/prompt/index revisions, all development test results, account-specific price/rate limits, forecast with contingency below $50, planned query counts, and the exact commands that would run. Stop there and wait for the owner's instruction. The later full run begins only from that frozen packet; unexpected model drift, cost overrun, or source mismatch stops it and produces an incident record rather than silently changing the protocol.

## 14. Failure modes and controls

- **Missing gold evidence or corpus mismatch:** adapter validation fails or marks the metric unavailable; no synthetic qrels.
- **Duplicate/overlapping chunks:** deduplicate by canonical source span and audit whether context still covers each required supporting fact.
- **Truncation bias:** record exact text seen by each model; stratify by length; run a controlled long-context sensitivity test.
- **Provider model drift:** persist requested/resolved model; halt a controlled run if the resolved revision changes unexpectedly.
- **Rate-limit distortion:** report queue wait and retry time; separate latency on successful first attempts from user-observed end-to-end latency.
- **Benchmark contamination/leakage:** avoid labels in prompts or retrieval input; split by source paper/case where the dataset requires it; explicitly describe any pretrained-model exposure limitation.
- **LLM-judge bias:** blind outputs, pin judge and rubric, quantify judge-human agreement, and keep native metrics primary.
- **Prompt injection in retrieved text:** delimit retrieved documents as data, record malicious text tests, and prevent documents from altering tool or system instructions.
- **High-stakes wording:** medical/legal benchmark performance is research evidence, not proof the system is safe for patient or legal advice.

## 15. Decisions intentionally pending

These do **not** block building the dataset adapters, retrieval/reranker plugins, trace system, or metric registry. They should be locked before the specified later gate.

| Decision for owner | Needed before | Default path until decided |
|---|---|---|
| Intended service workload: concurrency, query mix, p95/p99, availability and acceptable fallback | Milestone 7 production-readiness claim | Use the $50 global cap, small bounded runs, and publish measured curves without a pass/fail “production ready” label. |

The five datasets, initial retrieval arms, initial rerankers, standard JEV method, OpenRouter primary generator, $50 spend cap, trace/replay design, metric applicability, tooling, and build order are selected in this plan. If a source audit reveals that a dataset's corpus/terms cannot support the intended run, replace it only with a documented reason and a benchmark with equivalent task coverage; do not quietly modify the suite after seeing results.

## Sources and protocols

- [MIRAGE official benchmark](https://github.com/gzxiong/MIRAGE); [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG)
- [CLERC dataset](https://huggingface.co/datasets/jhu-clsp/CLERC); [CLERC paper](https://arxiv.org/abs/2406.17186)
- [HotpotQA official data/evaluation](https://hotpotqa.github.io/)
- [QASPER dataset card](https://huggingface.co/datasets/allenai/qasper)
- [KILT task data](https://huggingface.co/datasets/facebook/kilt_tasks); [KILT repository](https://github.com/facebookresearch/KILT)
- [TypeSafe API schema](https://api.typesafe.ai/openapi.json); [TypeSafe API docs](https://docs.typesafe.ai/api)
- [OpenRouter GPT-6 Luna listing](https://openrouter.ai/openai/gpt-6-luna/); [Kimi K3 listing](https://openrouter.ai/moonshotai/kimi-k3); [provider routing](https://openrouter.ai/docs/guides/routing/provider-selection); [generation metadata](https://openrouter.ai/docs/api/api-reference/generations/get-generation)
- [Ragas metric reference](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/); [LangChain documentation](https://docs.langchain.com/)
