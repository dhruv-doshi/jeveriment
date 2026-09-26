# JEV in production-style RAG: implementation plan

**Status:** implementation plan; no code in this branch yet

**Updated:** 2026-09-26

**Branch:** `plan/production-rag`

**Scope:** five existing benchmarks, pluggable retrieval and reranking, fixed primary generation comparisons, quality and operational evaluation

## 1. Purpose and claim boundary

Build a reproducible RAG system that answers two questions:

1. Given the same retrieved candidate pool, when does a simple JEV reranker improve evidence selection and final answers compared with established rerankers?
2. At what quality, latency, throughput, failure rate, and cost does that result hold under realistic load?

The initial deliverable is a solid evaluation system and a **simple** JEV baseline. Merely calling JEV and sorting its scores is not a novel research contribution. A new JEV method is a later, explicit research decision and must be compared against this baseline and the closest established methods. The article should make dataset-specific claims and state the tested workload; it should not call JEV universally “production ready.”

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
| Generator | Produce answer and citations from question + selected context | Hosted and local adapters; one model chosen for primary run | Yes |
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

## 4. Five selected datasets

The five are chosen now. They cover medicine, law, multi-hop Wikipedia QA, scientific papers, and fact verification. Large scale comes chiefly from CLERC, HotpotQA's full-Wikipedia setting, and KILT-FEVER; QASPER is deliberately smaller but tests long-document evidence selection. Dataset adapters use public labels only, with no new annotation project.

| Dataset and official source | Primary task and corpus | Native evaluation available | Adapter rules and limitation |
|---|---|---|---|
| [MIRAGE](https://github.com/gzxiong/MIRAGE) | Medical multiple-choice RAG; 7,663 questions across five sub-benchmarks, using a pinned MedRAG corpus/configuration | Answer accuracy, per sub-benchmark and macro | Count MIRAGE as **one** dataset family. Use question-only retrieval: answer options are excluded from retrieval input. Its benchmark does not provide universal retrieval qrels; do not fabricate nDCG/MRR for missing labels. Report answer accuracy and available evidence diagnostics separately. Pin and document the selected MedRAG corpus and its own terms. |
| [CLERC](https://huggingface.co/datasets/jhu-clsp/CLERC) | Legal case retrieval plus retrieval-augmented analysis/generation; published data has separate retrieval and generation files and a large collection | Citation/positive-passage retrieval; generation references where supplied | Use official collection and query splits, not the preview's 20 sampled negatives as the whole corpus. Map case IDs/citations to chunks. Check each task file's actual fields and license before ingest; its card is still incomplete. Do not reduce legal answer correctness to lexical overlap. |
| [HotpotQA](https://hotpotqa.github.io/) | Multi-hop open-domain QA over the fullwiki corpus | Answer EM/F1, supporting-fact EM/F1 | Use the **fullwiki** setting for primary retrieval; the distractor set supplies curated candidate documents and would overstate real retrieval quality. Preserve the benchmark's supporting-fact IDs and official normalization. Use labeled development data for held-out reporting; hidden test is not assumed scoreable locally. |
| [QASPER](https://huggingface.co/datasets/allenai/qasper) | Long scientific-paper QA; 5,049 questions over 1,585 NLP papers | Answer F1 and evidence-selection F1 from annotations | The official task gives the target paper. Restrict retrieval to that paper's chunks for the primary benchmark; cross-paper retrieval would be a separately named variant. Preserve multiple annotators, yes/no, free-form, extractive, and unanswerable cases. Small corpus, long-context stress. |
| [KILT-FEVER](https://huggingface.co/datasets/facebook/kilt_tasks) | Claim verification against the pinned KILT Wikipedia snapshot; FEVER config has 104,966 train and 10,444 validation claims | Label accuracy plus provenance/evidence retrieval | Use KILT's own knowledge source and provenance mapping, not HotpotQA's different Wikipedia snapshot. Track page and passage match separately. Verify availability of gold provenance by split; use validation for scoring where test evidence is hidden. |

For every adapter: snapshot source URL and revision/hash, exact split counts, corpus units, licenses and attribution, reference format, evaluation script version, and storage footprint. Check individual corpus licenses and data-use terms before sending text to a hosted API. Noncommercial research does **not** automatically grant permission to redistribute data or send it to third parties. Store dataset downloads outside Git with a reproducible manifest. Fail adapter validation when a referenced evidence ID cannot be mapped to the indexed corpus; report the unmapped fraction.

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

Common reranker limits: same max candidate count, same displayed chunk text, documented model-specific token limit/truncation, stable ties, missing-score policy, and fixed timeouts. Record mean and distribution of input tokens by model; different tokenization can make exact token equality impossible, so report both characters and model tokens. All rerankers get the same IDs even if one must truncate text. Local warm-up and provider cold start are reported separately.

The primary output is a complete reordered candidate list. The first context selector is deterministic: top-ranked unique evidence chunks packed until the same generator token budget is reached, preserving source citations. Compare rank top-k **and** actual selected context, because token lengths may cause different numbers of chunks to fit.

Optional controls after the first complete run: MMR/evidence diversity, a larger cross-encoder if hardware allows, and a contextual or batched JEV variant. These are named experiments with separate configs. The novel method is not assumed to be any of them.

## 7. Generation and final answers

- Define a generator protocol with local Hugging Face and hosted API adapters. Pick one pinned model for the primary factorial comparison before quality runs; the owner may later choose it without requiring architecture changes.
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

**API latency is measurable; server-side model inference time is not necessarily exposed.** TypeSafe's documented `/v1/systemone` response contains resolved `model`, `answers`, and token `usage`, but no inference-time field. The adapter checks for a documented response header such as `Server-Timing` during preflight and records a provider timing only if supplied with clear semantics. Otherwise store `provider_inference_ms = null` and `provider_timing_source = not_exposed`. Never subtract client network time from API duration and call the remainder model inference. Local-model inference time is measured around forward passes, with device synchronization for GPU timing. Generator time to first token is collected only for streaming APIs.

### Operational metrics

- Throughput: completed queries/s, retrieved candidates/s, reranked pairs/s, provider requests/s, input/output tokens/s.
- Tail latency: p50/p95/p99 by stage and end to end, including queue, retries, and failures; cold/warm separate.
- Load: concurrency, queue length, saturation, CPU/RAM/GPU utilization and memory, index I/O, local model batch utilization.
- Resilience: 429/5xx/timeout rates, retry amplification, rate-limit wait, canceled requests, fallback frequency, recovery time, correctness under fallback.
- Cost: provider billed tokens and USD per query/1,000 queries, retrieval/rerank/generation/judge components, local GPU-hours/energy proxy, storage/index cost. Store currency and price-table revision; distinguish actual invoices from estimates and zero-priced promotional requests from underlying token use.
- Stratification: query/context length bins, candidate depth, document length/truncation, task/dataset, concurrency, cache state, success/failure. Plot quality and latency against input tokens and candidate count.

The cost ledger reserves an estimate **before** dispatch and reconciles actual usage after response. It enforces request, token, and USD caps, with hard stops and a traceable override. Forecast calls as `sum(query × retriever arm × unique candidate pairs)` for pointwise JEV, adjusted for reusable identical query–chunk–rubric scores and retry rates. For example, 1,000 queries × 3 retrieval arms × 100 candidates can approach 300,000 pair scores before deduplication. Compute the forecast from the selected splits before any bulk run. Caching may share a score only when query, chunk text/hash, rubric, model revision, truncation, and JEV mode match exactly; contextual scores are not interchangeable with independent scores.

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
| Service/load | FastAPI for an answer endpoint; Locust for repeatable ramp and burst profiles; standard host/GPU telemetry. |

Exact dependency versions and container image digest are pinned when implementation starts. `LangChain`, `Ragas`, and `LangSmith` are not used as interchangeable “evaluation frameworks”: LangChain composes calls, Ragas computes optional answer metrics, and LangSmith is optional hosted trace inspection. The canonical artifact store is local/exportable, so experiments do not depend on a paid tracing service.

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
| 5. RAG answer path | Fixed packer, generator adapters, task output parsers, citation mapping | Same generator prompt settings/context budget across reranker arms; native answer formats parse and citations resolve. |
| 6. Quality evaluation | Native metrics, IR measures, selected Ragas metrics, paired analysis, reports | Each metric's applicability and denominator explicit; reranking and answer results connect to traces; held-out settings frozen. |
| 7. Service/operations | FastAPI endpoint, shared rate limiter, fault policy, Locust profiles, cost dashboard | Stage/end-to-end p50/p95/p99, throughput, retries, 429/5xx, queue, usage, cost and fallback quality available; bounded ramp succeeds within cap. |
| 8. Research extension | Novel algorithm only after hypothesis/prior-art review | It beats or clarifies the simple JEV and closest evidence-selection baselines on held-out data, with ablations and cost/latency trade-offs. |

Recommended execution scale: tiny fixture → one validation slice per dataset → all validation queries for baseline ranking metrics → matched answer-generation subset → held-out full answer run if affordable → operational workload. Sample queries deterministically and stratify by length/task/answerability, then report the sample definition. Keep all five datasets in the final suite even if early development starts with one. Scale production load independently from article-quality testing.

### Novelty design gate

Once the baseline traces show where JEV wins and fails, review primary work on pointwise/listwise reranking, budgeted context selection, uncertainty-aware ranking, diversity/MMR, and cascade reranking. Candidate hypotheses to discuss, **not decisions**:

- Use typed JEV probabilities to allocate a limited scoring budget to uncertain candidates, then hand off to a cheaper/faster scorer for the rest.
- Select a nonredundant evidence set under a fixed generator-token budget using JEV's relevance estimate plus measured marginal coverage.
- Route queries between JEV and local rerankers based on predicted value of additional computation.

For any selected method, write pseudocode, objective, computational/request complexity, required supervision, and falsifiable expected gain **before** held-out evaluation. Compare with the simple JEV sort, no rerank, best cross-encoder, fixed top-k, and the closest established selection or cascade method. Ablate each new component and report whether quality gains survive matched dollar, token, and p95 latency budgets. A method is not called novel merely because its combination has a new name.

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
| Primary generator provider/model and maximum context/output budget | Milestone 5's matched final-answer experiment | Build hosted/local adapters and use a replaceable development model for fixtures; no publishable primary answer scores until fixed. |
| Spend ceiling and intended service workload: concurrency, query mix, p95/p99, availability and acceptable fallback | Bulk API calls and milestone 7 production-readiness claim | Use small capped smoke runs and publish measured curves without a pass/fail “production ready” label. |
| Novel JEV-based algorithm/hypothesis and minimum meaningful gain | Milestone 8 and article novelty claim | Keep simple JEV as a baseline; perform a prior-art review before proposing a method. |

The five datasets, initial retrieval arms, initial rerankers, trace design, metric applicability, tooling, and build order are selected in this plan. If a source audit reveals that a dataset's corpus/terms cannot support the intended run, replace it only with a documented reason and a benchmark with equivalent task coverage; do not quietly modify the suite after seeing results.

## Sources and protocols

- [MIRAGE official benchmark](https://github.com/gzxiong/MIRAGE); [MedRAG](https://github.com/Teddy-XiongGZ/MedRAG)
- [CLERC dataset](https://huggingface.co/datasets/jhu-clsp/CLERC); [CLERC paper](https://arxiv.org/abs/2406.17186)
- [HotpotQA official data/evaluation](https://hotpotqa.github.io/)
- [QASPER dataset card](https://huggingface.co/datasets/allenai/qasper)
- [KILT task data](https://huggingface.co/datasets/facebook/kilt_tasks); [KILT repository](https://github.com/facebookresearch/KILT)
- [TypeSafe API schema](https://api.typesafe.ai/openapi.json); [TypeSafe API docs](https://docs.typesafe.ai/api)
- [Ragas metric reference](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/); [LangChain documentation](https://docs.langchain.com/)
