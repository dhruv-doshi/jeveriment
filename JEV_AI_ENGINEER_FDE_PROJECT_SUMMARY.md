# Jev Evidence Retrieval Evaluation — AI Engineer / FDE Project Dossier

**Project type:** Reproducible retrieval and evidence-selection evaluation for a retrieval-augmented generation (RAG) architecture  
**Completed experiment:** September 2026  
**Status:** Exploratory feasibility pilot on BEIR SciFact **train** queries; no production deployment or generated-answer evaluation

This file is a single source of copy-ready project details for an AI Engineer or Forward Deployed Engineer resume, application, or interview. Results from different experiment arms are separated because their candidate access and API provenance differ.

## Short project description

I built a Python evaluation pipeline to determine where TypeSafe AI's Jev evaluator fits in a scientific-evidence retrieval workflow. I integrated BM25 and Qwen dense retrieval, local Qwen and BGE rerankers, Jev API scoring, fixed-pool and full-corpus experiments, and a document/evidence-set decision benchmark. I evaluated ranking quality alongside latency, local resource use, token consumption, and API cost. On 30 sampled SciFact training queries, Jev achieved **0.759 nDCG@10** when reranking a shared 50-document pool and **0.768 nDCG@10** when scoring all **155,490** query–document pairs. The exhaustive direct-API scan recorded approximately **$4.72** in reported charges. This is a feasibility result, not a held-out or production claim.

## Problem and research questions

RAG systems need to find evidence before they can construct a supported answer. The project tested three distinct questions:

1. **Fixed-pool reranking:** Given identical retrieved candidates, how does Jev's task-specific relevance probability rank documents compared with lexical/dense fusion and local Qwen/BGE rerankers?
2. **First-stage retrieval:** If Jev scores every document directly from query and document text, can it find relevant evidence without a BM25 or embedding shortlist, and what does that cost?
3. **Evidence decisions:** Can Jev's probability threshold select individual documents, and can a separate Jev call judge whether the chosen top-10 evidence set is sufficient for the claim-evidence task?

The task rubric asks whether a scientific document contains evidence **supporting or refuting** a specific claim. Jev receives text as evidence and returns a typed Noul probability. The experiment does not use Jev to generate answers.

## Architecture and data flow

```text
BEIR SciFact corpus + train qrels
            |
            v
Validated ingestion -> SQLite corpus, sampled queries, frozen text views,
                       dataset hashes and pinned local model revisions
            |
            +------------------------------+
            |                              |
            v                              v
BM25S lexical index               Qwen3 dense embeddings
English stemming                  memory-mapped, resumable storage
            |                              |
            +----------> BM25 + dense RRF <-+
                            |
                            v
                 Frozen 50-document pool/query
                            |
             +--------------+---------------+
             |              |               |
             v              v               v
        Qwen reranker   BGE reranker    Jev Noul scorer
             +--------------+---------------+
                            |
                            v
                  Top-10 ranking evaluation

Separate arm: frozen query/document text -> Jev full-corpus scoring
            -> matched comparison against BM25 and dense cosine

Decision arm: direct-TypeSafe fixed-pool Jev scores -> document threshold
            + top-10 evidence-set Jev call -> benchmark against simple rules
```

### Data and experiment controls

- **Dataset:** BEIR SciFact, 5,183 scientific documents and 809 eligible training queries. The run sampled 30 queries deterministically with seed **1729**. The selected query IDs and source file hashes are recorded in the dataset manifest.
- **Input fairness:** Each local reranker and Jev saw the same frozen document text and, in the fixed-pool comparison, the same 50 candidate IDs per query. Documents were capped at 384 tokens, with a 512-token model-input cap. The common view was checked against local tokenizers before running.
- **Retrieval:** BM25S used Lucene-style BM25 with `k1=1.2`, `b=0.75`, English stop words, and stemming. Qwen3-Embedding-0.6B produced normalized document/query embeddings; cosine scores were computed in document blocks instead of building a full query-by-corpus matrix. BM25 and dense top-1,000 rankings were fused with reciprocal rank fusion (RRF) to form each 50-document pool.
- **Reranking:** Compared Qwen3-Reranker-0.6B, BGE-reranker-v2-m3, Jev, and the BM25+dense RRF baseline. A separate fixed RRF combination included BM25, dense, and Jev. Final selection size was 10. Local model revisions were saved, and model execution preferred CUDA or Apple MPS where available, with CPU fallback.
- **Jev provenance:** The original fixed-pool comparison used Jev via Vercel AI Gateway. The exhaustive scan used the **direct TypeSafe API** and requested `jev-1.13.0`. A separate direct-TypeSafe fixed-pool run supplied the decision benchmark. Their checkpoints and ledgers were isolated so scores from different providers were not silently mixed.
- **Experiment integrity:** Pydantic contracts validate documents, queries, judgments, candidates, and pools. Hashes bind candidates to text views and frozen input order. Config changes require a new experiment ID. Per-query scores, raw responses, metrics, TREC rankings, plots, and command logs are retained.

### API and reliability architecture

- The Jev client validates returned question IDs, primitive type, finite probabilities in `[0,1]`, model identity, and token-usage fields. A capability preflight uses synthetic positive/negative and multi-document cases before benchmark calls. An inconsistent optional Score primitive was left disabled; the measured Jev path uses Noul.
- API requests use a persistent SQLite WAL ledger. Each attempted request receives a cost, request-count, and token reservation before sending; the ledger reconciles provider-reported usage afterward. Timeouts retain uncertain reservations rather than assuming they were free. Successful responses are cached against request content and provenance.
- The exhaustive scan uses a worker pool (12 workers by default), an adjustable shared request-start limit, shared backoff on HTTP 429/529, and bounded retries. It checkpoints completed scores every 100 documents and at query boundaries. An exclusive lock prevents concurrent scans from writing the same run.
- Resource records include elapsed time, sampled process CPU and resident memory, provider request latency where available, attempts, tokens, and reported API charges. These are machine/run observations, not a service-level benchmark.

## Measured ranking results

All values below are means across the same **30 SciFact training queries**. nDCG@10, Recall@10, and MRR@10 are ranking metrics. Candidate recall measures coverage of **known relevant documents** at the relevant retrieval or pool depth.

### A. Shared 50-document pool: reranking comparison

| System | nDCG@10 | Recall@10 | MRR@10 | Candidate recall |
|---|---:|---:|---:|---:|
| BM25 + dense RRF | 0.703 | 0.850 | 0.672 | 0.933 |
| BGE reranker | 0.736 | 0.833 | 0.736 | 0.933 |
| **Jev reranker (Vercel run)** | **0.759** | **0.850** | **0.741** | **0.933** |
| Qwen reranker | 0.810 | 0.900 | 0.794 | 0.933 |
| BM25 + dense + Jev RRF | 0.769 | 0.867 | 0.741 | 0.933 |

Jev's nDCG@10 was **0.057 absolute points above** the BM25+dense RRF baseline and **0.023 above** BGE, while Qwen was **0.051 above** Jev on this sample. Every system had the same 0.933 candidate recall because it received the same pool: reranking cannot recover a relevant document missing from that pool. No fixed-pool system had a failed query in the completed evaluation.

### B. Full-corpus first-stage retrieval comparison

| System | nDCG@10 | Recall@10 | MRR@10 | Candidate recall at depth 1,000 |
|---|---:|---:|---:|---:|
| BM25 | 0.647 | 0.817 | 0.605 | 0.967 |
| Dense cosine | 0.735 | 0.850 | 0.704 | 1.000 |
| **Jev direct full-corpus scan** | **0.768** | **0.883** | **0.742** | **1.000** |

The scan scored **30 queries × 5,183 documents = 155,490 intended pairs**. Jev's nDCG@10 was **0.033 absolute points above** dense cosine and **0.121 above** BM25 on this sample. This was exhaustive API scoring, not a scalable Jev index. These results must not be compared as though the systems in the 50-document reranking table had access to the entire corpus.

## Acceptance and evidence-set decision results

A separate direct-TypeSafe rerank scored the 1,500 fixed-pool pairs used by the decision experiment. A **0.5** threshold was fixed for the comparison.

| Decision | Accepted | Precision | Recall | Label scope |
|---|---:|---:|---:|---|
| Jev document threshold | 27 of 36 judged pairs | 1.000 | 0.750 | Explicitly judged pool pairs |
| Fixed top-10 document rule | 33 of 36 judged pairs | 1.000 | 0.917 | Explicitly judged pool pairs |
| Jev evidence-set judgment | 18 of 30 sets | 0.889 | 0.615 | Known-relevant-presence proxy |
| Max-document-probability set rule | 25 of 30 sets | 0.920 | 0.885 | Known-relevant-presence proxy |
| Always-accept set rule | 30 of 30 sets | 0.867 | 1.000 | Known-relevant-presence proxy |

**Interpretation:** All 36 explicitly judged document pairs were relevant; there were **zero judged negatives**. The pair-level 1.000 precision values therefore cannot measure false positives. The set label only says whether the selected documents include a **known** relevant document. It is not an independent human judgment that the set supports a complete answer. The simple max-pair rule outperformed the separate Jev set judgment on this proxy; no answer-sufficiency improvement was established.

## Cost, latency, and local resource observations

| Run | Workload | Recorded result |
|---|---|---|
| Direct Jev full-corpus scan | 155,490 intended pair evaluations | 155,496 ledger attempts; 116,177,237 reported tokens; **$4.7227** reported API cost |
| Final resumed scan invocation | Completion of full-corpus checkpoint | **151.8 minutes** elapsed; process peak resident memory **135.7 MiB** |
| Direct Jev fixed-pool rerank | 1,500 pairs plus sentinel | 1,501 attempts; 1,135,233 tokens; **$0.0462** reported cost; **8.82 minutes** elapsed |
| Direct Jev fixed-pool request latency | 1,501 measured attempts | **342 ms median**, **393 ms p95** |
| Direct Jev evidence-set benchmark | 30 set calls | 112,191 tokens; **$0.0047** reported cost; **11.4 seconds** elapsed |
| Local BGE reranker | 1,500 pairs | **7.57 minutes**; process peak resident memory **872.2 MiB** |
| Local Qwen reranker | 1,500 pairs | **105.6 minutes**; process peak resident memory **3,182.8 MiB** |
| Original Vercel Jev reranker | 1,500 pairs | **85.5 minutes**; process peak resident memory **102.6 MiB** |

The direct full-corpus ledger slice includes retries and operational overhead; the intended workload was 155,490 pairs. The **151.8-minute** resource record is the final successful invocation after an earlier partial attempt, not guaranteed total wall time from a clean start. Ledger amounts are locally recorded provider-reported charges, not an invoice. The original Vercel and direct-TypeSafe runtimes are different runs and are not a controlled speed comparison. Local peak resident memory does not include a complete GPU-memory profile.

**Architecture implication:** A 5,183-document full scan requires roughly **104 times** the pair evaluations of scoring a 50-document pool for the same 30 queries. The pilot shows a small ranking-quality gain over dense cosine for exhaustive Jev retrieval, with substantially greater API work. A candidate-first design with selective reranking is the next architecture to validate against a specified quality, latency, and cost target; the pilot does not prove that it is optimal for production.

## AI Engineer and FDE competencies demonstrated

- **Model integration:** Adapted local transformer models and an external typed-decision API into comparable scoring stages while preserving model and provider provenance.
- **Evaluation design:** Held candidate access and text constant for rerankers, separated first-stage retrieval from reranking, reported per-query and aggregate IR metrics, and documented label limitations.
- **Reliable execution:** Designed resumable checkpoints, deterministic hashes, schema checks, rate limiting, retry handling, response caching, run locking, and persistent budget reservations for long-running API work.
- **Operational judgment:** Quantified the relationship between ranking quality, candidate count, latency, model runtime, memory, tokens, and API spend; identified where an all-pairs design would be expensive to scale.
- **Technical communication:** Produced reproducible configs, logs, reports, TREC exports, plots, and source-linked results that another engineer can audit or extend.
- **Implementation stack:** Python 3.11, PyTorch, Hugging Face Transformers, BM25S, PyStemmer, NumPy, SQLite, HTTPX, Pydantic, PyArrow, Matplotlib, pytest, Ruff, and `uv`.

## Resume-ready bullets

Select three or four based on the job description:

1. **Built a reproducible evidence-retrieval benchmark** over 5,183 SciFact documents, integrating BM25, Qwen dense embeddings, Qwen/BGE rerankers, and the TypeSafe Jev API on matched candidate pools.
2. **Engineered a resumable parallel Jev evaluation pipeline** for 155,490 query–document pairs with adaptive rate limiting, retries, response validation, checkpointing, and SQLite-based request/token/cost controls.
3. **Measured Jev ranking quality across two architectures:** 0.759 nDCG@10 on a fixed 50-document reranking pool and 0.768 nDCG@10 on an exhaustive full-corpus scan, compared with lexical, dense, fusion, and local reranker baselines.
4. **Quantified quality and operating cost:** the direct Jev full-corpus scan recorded $4.72 in API charges, while a separate 1,500-pair direct rerank recorded $0.046 and 342 ms median request latency.
5. **Built document and evidence-set decision evaluations** and identified sparse relevance labels and proxy sufficiency labels as limits on precision and answer-sufficiency claims.

## Interview discussion points

- **Why freeze a pool?** It isolates reranker quality from first-stage retrieval: all rerankers see the same IDs and evidence. The 0.933 candidate recall also defines the ceiling imposed by that pool.
- **Why run the full scan separately?** It answers whether Jev can find evidence from the entire corpus. Its 155,490 API pair evaluations have a very different cost and scaling profile from 1,500 fixed-pool pairs.
- **What made the API integration robust?** Typed response checks, provider/model isolation, preflight fixtures, daily sentinel calls, cached responses, bounded retries, shared throttling, persistent cost reservations, and checkpoint resume.
- **What did the decision test teach?** Jev's 0.5 threshold dropped known relevant pairs relative to fixed top-k, and the set-level result used only a weak proxy. A customer-facing claim about abstention or evidence sufficiency would require independently labeled negatives and human-reviewed answer support.
- **What would come next?** Evaluate held-out queries across the configured domains, add independent negative and set-sufficiency labels, assess paired uncertainty, then evaluate generated answers/citations and run workload-specific latency and load tests. These are next steps, **not completed outcomes**.

## Claim boundaries and status

- The completed quality results cover a **30-query sample from the SciFact training split**. They do not demonstrate generalization, statistical superiority, state-of-the-art performance, or production readiness.
- Configurations exist for **NFCorpus, FiQA, and ArguAna**, but this dossier reports **no completed runs** on them.
- Jev's immutable provider revision and possible model-training exposure were not fully verified for confirmatory research claims. The versioned direct request named `jev-1.13.0`.
- The repository supports optional generation and further analysis, but **no end-to-end generated-answer, citation-quality, human sufficiency, or production load-test result** is reported here.
- Current automated verification: **36 pytest tests passed** in this workspace. Tests do not substitute for held-out model-quality or human-label validation.

## Source artifacts

- Architecture and project status: [docs/PROJECT.md](docs/PROJECT.md)
- Pilot configuration: [configs/pilot.yaml](configs/pilot.yaml)
- Dataset and pinned local model records: [dataset manifest](runs/scifact_train_pilot_v1/dataset_manifest.json), [model manifest](runs/scifact_train_pilot_v1/model_manifest.json)
- Fixed-pool results: [summary.json](runs/scifact_train_pilot_v1/summary.json)
- Full-corpus results: [retrieval_typesafe_summary.json](runs/scifact_train_pilot_v1/retrieval_typesafe_summary.json)
- Decision results: [decision_benchmark.json](runs/scifact_train_typesafe_decisions_v1/decision_benchmark.json)
- Direct API and resource observations: [direct rerank resources](runs/scifact_train_typesafe_decisions_v1/resources_rerank_jev.json), [scan resources](runs/scifact_train_pilot_v1/resources_jev_scan_typesafe.json)
- Core implementation: [pipeline.py](src/jev_eval/pipeline.py), [scan.py](src/jev_eval/scan.py), [jev.py](src/jev_eval/jev.py), [ledger.py](src/jev_eval/ledger.py), [metrics.py](src/jev_eval/metrics.py)
