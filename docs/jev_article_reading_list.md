# Reading guide for the Jev retrieval article

Updated: 2026-10-01

This guide supports an exploratory article about when Jev's retrieval quality is worth its cost and latency. It preserves the original five readings and adds background for interpreting SciFact, ranking metrics, and service timing. The intended contribution is a controlled account of the trade-offs between exhaustive scoring, shortlist reranking, candidate depth, and request batching. Whether Jev offers a useful advantage remains an experimental question.

Start with readings 1–3 before designing another run. Read 4–5 when framing the research question, then use the additional readings to address specific gaps. Reading estimates below are suggested skim budgets, not publication metadata. Linked developer benchmarks report their authors' experiments; their numbers have not been independently reproduced in this repository.

## Original five readings

### 1 TypeSafe model documentation

[Models](https://docs.typesafe.ai/models) · [API reference](https://docs.typesafe.ai/api) · [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)

**Source:** Provider documentation. **Suggested time:** 10–15 minutes.

Read the pricing, context limits, rate limits, and model alias sections, followed by the request structure and Noul question type. The documentation explains how a shared state is evaluated against multiple questions and how input tokens are billed.

For the article, draw the exact request layout for each experiment: one query/document pair per call, a query in shared state with each passage in a separate question, or all passages in shared state. These layouts expose different context to the model and must be described separately. Record the requested and returned model IDs, the price used for calculations, and the date of the run.

**Question to resolve:** Can request packing reduce overhead while preserving the intended scoring task and measured quality? Treat that as a hypothesis to test.

### 2 Denser Jev versus Qwen benchmark

[Repository and report](https://github.com/denser-org/rerank-bench-jev)

**Source:** Developer benchmark with code and saved responses. **Suggested time:** 15–20 minutes.

This is the closest prior work to the proposed article. It compares Jev and hosted Qwen3-Reranker-0.6B on SciFact and NFCorpus using frozen BM25 top-100 candidates, reporting ranking quality, estimated cost per query, and query latency. Both systems use packed requests at four concurrent queries.

Read the method, packing, repeat-run, and limitations sections before the headline conclusions. Packing equivalence was checked at depth 10, while the main comparison uses depth 100. The report's TypeSafe cost calculation uses $0.040 per million input tokens; check it against the rate applicable to your own run. A confidence interval that includes zero does not establish model equivalence, and p95 does not describe the worst case.

**Question to resolve:** How much of an observed speed difference comes from request design, serving provider, and concurrency? Those are experimental factors alongside model choice.

### 3 Aness Belbati on giving Jev a reranker job

[I gave Jev a reranker's job](https://anessbelbati.com/blog/i-gave-jev-a-rerankers-job)

**Source:** Experiment write-up with linked code and raw responses. **Suggested time:** 15–20 minutes.

Read the shared versus isolated passage setups, scoring rubrics, aggregation choices, and sections on sharing text across questions. The study compares Jev with established rerankers and illustrates how request structure and the choice of averaging method affect interpretation.

Its initial ranking table evaluates 1,617 of 2,327 queries: those whose BM25 candidates contain at least one labelled relevant passage. That measures ranking conditional on candidate availability. It should not be compared directly with an aggregate that includes candidate-retrieval failures.

**Question to resolve:** Are reported gains due to better candidate retrieval, better ordering within a pool, or a change in which queries enter the average?

### 4 BEIR

[BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)

**Source:** Research paper, NeurIPS 2021 Datasets and Benchmarks. **Suggested time:** 20–30 minutes to skim.

Read the introduction, dataset coverage, evaluation setup, and results discussion. BEIR compares retrieval approaches across diverse tasks and domains, including their effectiveness and computational costs.

Use it to motivate retaining a lexical baseline and testing beyond SciFact. Explain which datasets represent which tasks, and report dataset-level results before an aggregate. A result from one scientific evidence task does not establish performance across retrieval workloads.

**Question to resolve:** Which additional datasets would challenge the proposed explanation for Jev's performance, rather than repeat the same task?

### 5 Rerank Before You Reason

[Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents](https://cs.uwaterloo.ca/~jimmylin/publications/Sharifymoghaddam_Lin_ACL2026.pdf)

**Source:** Research paper by Sahel Sharifymoghaddam and Jimmy Lin. **Suggested time:** 20–30 minutes to skim.

Read the abstract, experimental setup, effective token cost definition, and reranking-depth analysis. The paper studies reranking and reasoning budgets in deep search agents on BrowseComp-Plus. It provides a framework for considering downstream computation when evaluating retrieval expense.

Apply that perspective if the article expands into answer generation: additional reranking may change generation tokens, subsequent searches, or final answer quality. This is adjacent research, not evidence of a Jev-specific gain, and its iterative search setting differs from a single retrieval pass.

**Question to resolve:** Does an extra dollar spent on reranking reduce total task cost or improve final answers enough to justify it?

## Additional readings

### 6 The original SciFact paper

[Fact or Fiction: Verifying Scientific Claims](https://aclanthology.org/2020.emnlp-main.609/)

**Source:** Wadden et al., EMNLP 2020. **Suggested time:** 15–20 minutes.

Read the task definition and dataset construction. SciFact involves finding abstracts that support or refute scientific claims and identifying evidence rationales. Use this to explain why relevant evidence can contradict the query's claim.

**Application to this study:** Distinguish abstract retrieval evaluated with BEIR relevance judgments from full claim verification, rationale identification, and generated-answer correctness. Before making calibration or filtering claims, inspect what positive, negative, and unjudged pairs actually exist in the run's labels.

### 7 Retrieval evaluation fundamentals

[Evaluation of ranked retrieval results](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)

**Source:** Stanford's online edition of *Introduction to Information Retrieval*. **Suggested time:** 10–15 minutes.

Focus on precision and recall at a cutoff, variation across queries, and normalized discounted cumulative gain. The chapter explains why ranking position and the number of relevant documents affect metric interpretation.

**Application to this study:** Explain nDCG@10 in plain language as rewarding relevant documents near the top, normalized against an ideal ranking. State the gain function and relevance threshold used in the implementation. Report candidate recall separately so readers can distinguish failures to retrieve evidence from failures to rank it well.

### 8 The Tail at Scale

[The Tail at Scale](https://research.google/pubs/the-tail-at-scale/)

**Source:** Jeffrey Dean and Luiz André Barroso, *Communications of the ACM*, 2013. **Suggested time:** 15–20 minutes.

Read for the relationship between component latency, service utilization, and slow complete requests. It explains why occasional delays can dominate a system assembled from many operations.

**Application to this study:** Measure the entire reranking query, including queueing and retries, as well as individual API attempts. Record concurrency and throughput next to latency percentiles. Dividing batch runtime by query count gives an amortized runtime; under concurrent execution it is not a query latency distribution.

### 9 Another Jev RAG comparison

[English Jev RAG benchmark](https://github.com/emretheus/jev-rag-benchmark)

**Source:** Developer benchmark with reports and per-query artifacts. **Suggested time:** 15–20 minutes.

Read its frozen top-20 comparisons on SciFact and English XQuAD, then the latency and cost tables. It reports a different latency picture from Denser's setup and includes exploratory confidence-based decisions and answerability analysis.

**Application to this study:** Investigate the request layout, endpoint, pacing, and measurement boundary before comparing timings across repositories. Separate actual spend under free tiers from estimated cost at list prices. Inspect the relevance-label construction before adopting its calibration conclusions. Its exploratory decision analyses should not be treated as independently confirmed improvements.

## Notes to collect from each experiment

Use the following fields to build a related-work comparison. Leave missing details marked as unreported; do not infer identical conditions from identical dataset names.

| Field | What to record |
| --- | --- |
| Task and data | Dataset version, split, query count, corpus size, and eligibility rules |
| Candidate access | Retriever, pool depth, candidate recall, and whether every system receives identical IDs and text |
| Request structure | Passages per call, location of each passage, question type, rubric, truncation, and context limits |
| Serving conditions | Model/version, provider, local hardware where applicable, concurrency, pacing, and caching |
| Quality | Metric definition, evaluation cutoff, query inclusion, paired uncertainty, and per-dataset results |
| Timing | API attempt versus complete query versus full RAG request; queueing, retries, warmup, and cold starts |
| Cost | Measured tokens, dated rates, free credits, failed attempts, local compute, and included pipeline stages |
| Reproducibility | Code commit, saved responses, manifests, model identity, and whether you reproduced the result |

## How the reading informs the article

The following are proposed experiments, not findings from the reading list or completed work:

1. Compare exhaustive scoring with shortlist reranking as different architectures, accounting for their different candidate access and number of evaluations.
2. Sweep candidate depth while keeping each depth's pool identical across rerankers. Measure whether extra scoring changes quality enough to justify its cost.
3. Compare sequential calls, bounded concurrent calls, and packed requests. Verify quality as well as speed when the model's visible context changes.
4. Use held-out queries and paired comparisons. Preserve retrieval failures and report uncertainty rather than selecting only favourable queries.
5. Add a fixed-generator answer evaluation only if the article claims a benefit to final RAG answers. Ranking improvements alone support a narrower claim.

Existing work already evaluates Jev reranking, batching, and confidence-based decisions. These readings help position a replication or extension; they do not establish novelty for the proposed article.

## Related project documents

- [Project walkthrough and pilot limitations](PROJECT.md)
- [Production RAG working plan](jev_production_rag_plan.md)
- [Original retrieval study guide](Jev_RAG_Retrieval_Final_Guide.md)

Before publication, record repository commit IDs and dated copies of any changing documentation used to support numerical claims. The linked pages are starting points for inspection, not a frozen evidence archive.
