# JEV Reranking in a Pluggable RAG Pipeline: Working Plan

**Status:** Discussion draft  
**Updated:** 2026-09-25

This is a working plan for testing whether JEV adds useful quality to a RAG system and whether its operating characteristics fit a production workload. Edit this document as decisions are made. No implementation decisions beyond the items marked “agreed” are locked.

## Goal

Build one reproducible RAG pipeline in which datasets, retrieval methods, and rerankers can be compared through stable interfaces. Evaluate both the intermediate retrieval/reranking results and the final generated answers. Measure service behavior under realistic load. Separately define and validate a new reranking method; an integration of the existing JEV reranker alone is not the intended research contribution.

“Production ready” must be defined against a chosen task, data policy, quality targets, cost envelope, and service-level objectives. There is no general production-readiness score.

## Agreed direction

- Use existing public datasets and their existing labels, qrels, answer references, or benchmark judgments. Do not start by creating a new dataset.
- Use a suite of roughly 3–5 datasets to test more than one domain or task. Report dataset-level results; do not let a pooled average hide regressions.
- Make datasets, retrieval methods, and rerankers pluggable.
- For every fixed `(dataset, retriever, query)` candidate pool, pass the same candidate IDs and text to every reranker.
- Evaluate intermediate stages and end-to-end RAG output.
- Measure quality and operational performance, including throughput, p95/p99 latency, throttling, failures, input/output tokens, and cost.
- Use familiar, widely adopted RAG tooling where it helps people understand or reproduce the system.
- Treat novelty as an open research question. JEV reranking by itself is already demonstrated elsewhere.

## Pipeline and boundaries

```text
dataset adapter → corpus/chunk index → retriever → frozen candidate pool
                                                ↓
answer ← generator ← context selector ← reranker
  ↓             ↓             ↓            ↓
answer eval   trace/usage   evidence eval  ranking eval
```

### Components that can be plugged in

| Component | Initial alternatives | Comparison rule |
|---|---|---|
| Dataset adapter | One adapter per selected benchmark | Preserve native IDs, splits, judgments, provenance, and answer references. |
| Retriever | BM25, dense, hybrid (BM25 + dense); optionally a stronger published configuration | Save each retriever’s pool and scores once. A reranker comparison within that arm uses the identical pool. |
| Reranker | No reranking/RRF, strong cross-encoder(s), JEV | One common input/output contract; return original document IDs, scores, order, latency, and errors. |
| Context selector | Fixed top-k, fixed token budget; later a proposed JEV method | Keep the generator’s context budget identical across arms. |
| Generator | Pluggable in the harness; one fixed model in the primary reranker comparison | Change generators only in a separate robustness experiment, or clearly analyze the factorial design. |
| Evaluator | Dataset-native IR/RAG evaluators, plus common supplementary measures | Keep benchmark metrics and judge prompts/versioned evaluators fixed within a comparison. |

Retriever choice still matters: a reranker cannot recover a relevant item absent from its candidate pool. Report candidate recall and results for each retriever–reranker pairing. The retriever can be a controlled factor without being the primary research variable.

### Keep fixed in the primary comparison

- Dataset version, corpus snapshot, query split, qrels/reference answers, and query order.
- Document parsing, chunking, metadata, and text formatting within each dataset.
- The exact candidate pool for each `(dataset, retriever, query)` reranker comparison.
- Generator model/version, prompt, decoding parameters, citation format, and maximum context tokens.
- Evaluation rubric, evaluator version, random seeds where supported, and failure/fallback policy.

Chunking and generator changes can be useful later, but varying them in the first reranker comparison would make the source of a quality change unclear. Use separate controlled experiments for those factors.

## Dataset shortlist to investigate

This is a shortlist, not a final commitment. Prefer datasets that already provide the corpus, queries, retrieval judgments, and (for end-to-end tests) answer references or evidence nuggets. Using a benchmark avoids writing new relevance labels, but adapters, format normalization, and license checks will still be required.

| Candidate | Why it may fit | Check before selecting |
|---|---|---|
| MIRAGE | Biomedical RAG benchmark spanning medical QA datasets. | Corpus and supporting-evidence availability, dataset-specific terms, and which tasks can be run as retrieval plus generation. [Paper](https://aclanthology.org/2024.findings-acl.372/) |
| CLERC | Legal case retrieval and retrieval-augmented analysis/generation. | Data packaging, corpus access, license, and evaluation scripts. [Paper](https://arxiv.org/abs/2406.17186) |
| TREC RAG | Separate retrieval and end-to-end RAG tasks with development materials and benchmark evaluation. | Current corpus access and compute requirements; the 2026 track describes a very large corpus served through Pyserini. [Track](https://trec-rag.github.io/) |
| BEIR task(s), e.g. SciFact or NFCorpus | Existing qrels enable controlled reranker comparisons and broaden IR-task coverage. | These are primarily retrieval evaluations; do not count them as final-answer evaluations unless suitable answer references are available. [BEIR paper](https://arxiv.org/abs/2104.08663) |

Select 3–5 after checking direct pipeline fit. Avoid choosing multiple datasets that mostly repeat the same task while leaving other intended claims unevaluated. Report per-dataset metrics and a macro summary with equal dataset weighting where appropriate.

## Evaluation plan

### Retrieval and reranking

- Candidate recall at the retrieved depth, before reranking.
- nDCG@k, MRR@k, Recall@k, Precision@k, and relevance of the final context set.
- Results at the actual context cutoff/token budget, not just an abstract top-10 ranking.
- Paired per-query comparisons, uncertainty intervals, and a predeclared practically meaningful effect size.
- Full score/rank outputs and per-query failures to diagnose wins and losses.

### Final RAG output

- Answer correctness against benchmark references or nuggets.
- Whether answer claims are supported by retrieved evidence and citations.
- Evidence coverage/completeness and sensitivity to irrelevant or contradictory passages.
- Abstention when the answer is not in the corpus.
- Blind human review on a bounded, representative sample if benchmark references cannot answer these questions reliably.

RAGAS offers measures such as context precision/recall and faithfulness, but LLM-judged scores should supplement benchmark-native or human judgments, not replace them. The RAGAS documentation distinguishes component-level measures from end-to-end outcomes and notes that LLM-based metrics can be nondeterministic. [Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) · [Metric overview](https://docs.ragas.io/en/stable/concepts/metrics/overview/)

### Performance and stress testing

Measure retrieval, reranking/JEV, generation, and full-request behavior separately and end to end:

- Throughput in queries/second, documents/second, and provider requests/second.
- p50, p95, and p99 latency; include queue wait, retries, and cold starts where applicable.
- Concurrency, request size, candidate count, input tokens, output tokens, and generated tokens.
- Cost per request/query and cost-quality trade-off at each candidate depth/context budget.
- HTTP errors, timeouts, 429/529 throttling, retry amplification, queue growth, and recovery after a provider slowdown/outage.
- Quality and fallback behavior under load, not just successful request speed.

Run a bounded ramp, sustained-load period, and burst/recovery test that stays within the provider’s allowed limits and an explicit spend cap. A stress test establishes operational capacity for the tested workload; it does not establish answer quality.

## Familiar tooling

“Industry standard” is not a single stack. Use one mainstream orchestration framework and avoid testing the framework itself as a confound. Candidate choices to evaluate:

- LangChain or LlamaIndex for recognizable component interfaces; select one for the initial integration.
- Standard BM25, dense vector, and hybrid retrieval adapters, with fixed configuration/version records.
- RAGAS as supplemental RAG evaluation, with dataset-native metrics retained as primary where available.
- A conventional HTTP service and load-testing/telemetry tools for the operational harness.

Do not add a framework merely for name recognition if its abstractions obscure candidate IDs, scores, prompts, latency, or cost. Record exact package/model/index versions and all settings.

## Novelty and research questions still open

### New method

The method must do more than call JEV, sort by a JEV relevance score, or apply a fixed threshold. A possible hypothesis to investigate is **budgeted evidence-set construction using JEV’s typed probabilities**: select evidence under a fixed generator token budget while accounting for relevance, uncertainty/missing-evidence risk, and redundancy. This is only a candidate idea, not a novelty claim. Compare against simple JEV score sorting, standard cross-encoders, fixed top-k, and established selection methods such as MMR or other relevant evidence-selection baselines.

Before naming the contribution:

1. Review the JEV reranker articles and primary literature on pointwise/listwise reranking, filtering, adaptive reranking, and evidence-set selection.
2. State the proposed algorithm precisely enough to implement and falsify.
3. Identify what new capability or measured trade-off it provides over the closest prior methods.
4. Define ablations that isolate the contribution (for example, typed probability use, uncertainty handling, redundancy handling, and token-budget constraint).
5. Phrase claims as hypotheses until evaluated on held-out benchmark splits.

### Other decisions pending

- Final 3–5 dataset suite, task scope, and licensing review.
- Which framework and service/load-test tools to use.
- Primary retriever configurations and candidate depth(s).
- Primary reranker baselines and which local/API models are feasible.
- Generator and fixed prompt for the first end-to-end comparison.
- Primary quality endpoints, confidence intervals, smallest practically important effect, and acceptance thresholds.
- Realistic traffic profile, target concurrency/throughput, p95/p99 latency targets, cost cap, and fail-open/fail-closed behavior.
- TypeSafe account limits, data-use terms, uptime/SLA, and how API/model version changes will be monitored.

## Next planning sequence

1. Select the dataset suite using the shortlist and fit checks above.
2. Freeze the static pipeline contract and decide which components are initial plugins.
3. Complete the novelty review and choose a falsifiable new algorithm hypothesis.
4. Choose baselines, metrics, held-out splits, and service targets before implementation.
5. Implement the harness and run offline quality comparisons first.
6. Run end-to-end RAG evaluation, then bounded load tests against the chosen operational targets.

## References

- [MIRAGE: Benchmarking Retrieval-Augmented Generation for Medicine](https://aclanthology.org/2024.findings-acl.372/)
- [CLERC: A Dataset for Legal Case Retrieval and Retrieval-Augmented Analysis Generation](https://arxiv.org/abs/2406.17186)
- [TREC RAG Track](https://trec-rag.github.io/)
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
- [LangChain documentation](https://docs.langchain.com/)
- [RAGAS metrics documentation](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
