"""Experiment composition. Stage boundaries are persisted before downstream work."""

import os
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv

from jev_eval.io import digest, file_digest, read_json, write_json

from .answer import generate_answer, pack_context
from .artifacts import ArtifactStore
from .clients import BudgetedHTTP
from .config import ExperimentSpec
from .contracts import (
    CandidatePool,
    GenerationResult,
    GoldRecord,
    JudgeResult,
    QueryRecord,
    RerankResult,
)
from .datasets import adapter_for
from .embedding import QwenEncoder
from .evaluation import evaluate_answer, evaluate_ranking
from .judge import JUDGE_RUBRIC_V1, judge_answer
from .reranking import (
    CrossEncoderModel,
    QwenRerankerModel,
    retrieval_order,
    score_with_jev,
    score_with_model,
)
from .retrieval import ChunkIndex, DenseIndex, freeze_pool, rrf
from .runner import execute_stage


def prepare_dataset(spec: ExperimentSpec, dataset_id: str, source_root: str | Path, run_root: str | Path):
    adapter = adapter_for(dataset_id, source_root)
    adapter.require_files()
    final_root = Path(run_root) / spec.experiment_id / dataset_id
    if final_root.exists():
        raise FileExistsError(f"Prepared source already exists at {final_root}; use a new run ID for changed inputs")
    final_root.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f".{dataset_id}.pending-", dir=final_root.parent))
    source_paths = [adapter.root / name for name in adapter.required_files]
    if dataset_id == "mirage":
        source_paths.extend(adapter._corpus_files())
    if dataset_id == "qasper":
        source_paths.extend([adapter.root / "qasper.jsonl"] if (adapter.root / "qasper.jsonl").exists() else adapter._parquet_files())
    source_files = {str(path.relative_to(adapter.root)): file_digest(path) for path in source_paths}
    index = ChunkIndex(root / "chunks.sqlite")
    chunk_count = index.add(adapter.iter_chunks())
    pairs = list(adapter.iter_queries())
    ids = [query.query_id for query, _ in pairs]
    if len(ids) != len(set(ids)) or not pairs or not chunk_count:
        raise ValueError("Source has duplicate/empty query IDs or empty corpus")
    unmapped = sorted({cid for _, item in pairs for cid in item.qrels if index.db.execute("SELECT 1 FROM chunks WHERE chunk_id=?", (cid,)).fetchone() is None})
    if unmapped:
        raise ValueError(f"{len(unmapped)} gold evidence IDs do not map to corpus; examples: {unmapped[:5]}")
    queries = [query.model_dump() for query, _ in pairs]
    gold = [item.model_dump() for _, item in pairs]
    write_json(root / "queries.json", queries)
    write_json(root / "gold.json", gold)
    manifest = {"dataset": dataset_id, "source_files_sha256": source_files,
                "queries": len(queries), "chunks": chunk_count,
                "dataset_spec_hash": digest(next(item.model_dump(mode="json") for item in spec.datasets if item.id == dataset_id)),
                "query_hash": digest(queries), "gold_hash": digest(gold)}
    write_json(root / "source_manifest.json", manifest)
    root.rename(final_root)
    return manifest


class ExperimentRuntime:
    def __init__(self, spec: ExperimentSpec, run_root: str | Path, dataset_id: str,
                 revisions: dict[str, str] | None = None, provider_only: str | None = None,
                 limits: dict | None = None, transport=None):
        load_dotenv()
        self.spec, self.dataset_id = spec, dataset_id
        self.root = Path(run_root) / spec.experiment_id
        self.dataset_root = self.root / dataset_id
        self.source = read_json(self.dataset_root / "source_manifest.json")
        dataset_spec = next(item for item in spec.datasets if item.id == dataset_id)
        if self.source["dataset_spec_hash"] != digest(dataset_spec.model_dump(mode="json")):
            raise ValueError("Prepared corpus was built from a different dataset specification")
        self.index = ChunkIndex(self.dataset_root / "chunks.sqlite")
        self.store = ArtifactStore(self.root / "artifacts")
        self.revisions = revisions or {}
        self.provider_only, self.limits, self.transport = provider_only, limits or {}, transport
        self.models = {}
        self._dense = None
        self._clients = {}
        self._lock = threading.RLock()
        self._jev_executor = None
        self.queries = [QueryRecord.model_validate(item) for item in read_json(self.dataset_root / "queries.json")]
        self.gold = {item.query_id: item for item in (GoldRecord.model_validate(item) for item in read_json(self.dataset_root / "gold.json"))}

    def _encoder(self):
        name = "Qwen/Qwen3-Embedding-0.6B"
        with self._lock:
            if name not in self.models:
                self.models[name] = QwenEncoder(name, self.revisions.get(name))
            return self.models[name]

    def build_dense(self, batch_size=32):
        return DenseIndex.build(self.index, self.dataset_root / "dense.npy", self._encoder(), batch_size)

    def _dense_index(self):
        with self._lock:
            if self._dense is None:
                path = self.dataset_root / "dense.npy"
                if not path.exists():
                    raise FileNotFoundError(f"Dense index missing: {path}; build it first")
                self._dense = DenseIndex(self.index, path, self._encoder())
            return self._dense

    def _client(self, provider):
        with self._lock:
            if provider in self._clients:
                return self._clients[provider]
            key = os.getenv("TYPESAFE_API_KEY" if provider == "typesafe" else "OPENROUTER_API_KEY")
            limits = self.limits.get(provider)
            if not limits:
                raise ValueError(f"Account-specific rate and price limits missing for {provider}")
            client = BudgetedHTTP(
                provider=provider, root=self.root, token=key or "",
                base_url=os.getenv("TYPESAFE_API_BASE_URL", "https://api.typesafe.ai") if provider == "typesafe" else "https://openrouter.ai/api/v1",
                budget_usd=self.spec.budget_usd,
                max_tokens=int(limits["token_budget"]), max_requests=int(limits["request_budget"]),
                input_usd_per_million=Decimal(str(limits["input_usd_per_million"])),
                output_usd_per_million=Decimal(str(limits["output_usd_per_million"])),
                rpm=int(limits["rpm"]), tpm=int(limits["tpm"]), inflight=int(limits["inflight"]),
                transport=self.transport,
            )
            self._clients[provider] = client
            return client

    def _jev_pool(self):
        with self._lock:
            if self._jev_executor is None:
                self._jev_executor = ThreadPoolExecutor(max_workers=int(self.limits["typesafe"]["inflight"]))
            return self._jev_executor

    def close(self):
        with self._lock:
            if self._jev_executor is not None:
                self._jev_executor.shutdown(wait=True)
                self._jev_executor = None
            for client in self._clients.values():
                client.http.close()

    def _retrieval_hits(self, query, arm):
        depth = self.spec.max_saved_depth
        if arm == "hybrid_rrf":
            lexical, lex_record, _ = self._retrieval_hits(query, "bm25")
            dense, dense_record, _ = self._retrieval_hits(query, "qwen_dense")
            hits = rrf([[cid for cid, _ in lexical], [cid for cid, _ in dense]])[:depth]
            return hits, (lex_record.output_hash, dense_record.output_hash), False
        config = {"arm": arm, "depth": depth, "source_hash": digest(self.source),
                  "revision": self.revisions.get("Qwen/Qwen3-Embedding-0.6B") if arm == "qwen_dense" else "sqlite_fts5"}

        def call(_):
            return self.index.search_bm25(query, depth) if arm == "bm25" else self._dense_index().search(query, depth)

        hits, record, cached = execute_stage(self.store, run_id=self.spec.experiment_id,
                                             stage="retrieve", plugin_id=arm, config=config,
                                             upstream_hashes=(self.source["query_hash"],),
                                             input_value={"query_id": query.query_id, "text": query.text, "scope": query.corpus_scope}, call=call,
                                             query_id=query.query_id, mode="live")
        return [(cid, float(score)) for cid, score in hits], record, cached

    def retrieve(self, query: QueryRecord, arm: str):
        if arm not in {item.id for item in self.spec.retrievers}:
            raise ValueError(f"Unknown retrieval arm: {arm}")
        hits, upstream, _ = self._retrieval_hits(query, arm)
        upstream_hashes = upstream if isinstance(upstream, tuple) else (upstream.output_hash,)
        pool, record, cached = execute_stage(
            self.store, run_id=self.spec.experiment_id, stage="candidate_pool", plugin_id=arm,
            config={"depth": self.spec.candidate_depth, "source_hash": digest(self.source)},
            upstream_hashes=upstream_hashes,
            input_value={"query_id": query.query_id, "hits": hits[:self.spec.candidate_depth]},
            call=lambda _: freeze_pool(query, arm, hits, self.index, self.spec.candidate_depth).model_dump(),
            query_id=query.query_id, mode="live",
        )
        return CandidatePool.model_validate(pool), record, cached

    def _local_model(self, arm):
        item = next(spec for spec in self.spec.rerankers if spec.id == arm)
        name = str(item.params["model"])
        with self._lock:
            if name not in self.models:
                revision = self.revisions.get(name)
                self.models[name] = QwenRerankerModel(name, revision) if arm == "qwen" else CrossEncoderModel(name, revision)
            return self.models[name]

    def rerank(self, query: QueryRecord, pool: CandidatePool, pool_record, arm: str):
        if arm not in {item.id for item in self.spec.rerankers}:
            raise ValueError(f"Unknown reranker arm: {arm}")
        item = next(spec for spec in self.spec.rerankers if spec.id == arm)
        jev_model = os.getenv("JEV_MODEL") if arm == "jev" else None
        if arm == "jev" and not jev_model:
            raise ValueError("JEV_MODEL is required for live JEV scoring")
        config = {"arm": arm, "params": item.params, "revision": self.revisions.get(str(item.params.get("model"))),
                  "jev_model": jev_model}

        def call(_):
            if arm == "retrieval_order":
                result = retrieval_order(pool)
            elif arm == "jev":
                result = score_with_jev(query, pool, self.index, self._client("typesafe"),
                                        model=jev_model, executor=self._jev_pool())
            else:
                result = score_with_model(query, pool, self.index, self._local_model(arm), arm)
            return result.model_dump()

        value, record, cached = execute_stage(self.store, run_id=self.spec.experiment_id,
                                              stage="rerank", plugin_id=arm, config=config,
                                              upstream_hashes=(pool_record.output_hash,),
                                              input_value={"query_id": query.query_id, "pool": pool.model_dump()},
                                              call=call, query_id=query.query_id, mode="live")
        result = RerankResult.model_validate(value)
        if set(result.ordered_ids) != {candidate.chunk_id for candidate in pool.candidates}:
            raise ValueError("Reranked IDs differ from candidate pool")
        return result, record, cached

    def answer(self, query, result, rerank_record):
        context_value, context_record, context_cached = execute_stage(
            self.store, run_id=self.spec.experiment_id, stage="context", plugin_id="rank_pack_v1",
            config={"budget_tokens": self.spec.generator.evidence_tokens},
            upstream_hashes=(rerank_record.output_hash,),
            input_value={"query_id": query.query_id, "ordered_ids": result.ordered_ids},
            call=lambda _: pack_context(query, result, self.index, self.spec.generator.evidence_tokens).model_dump(),
            query_id=query.query_id, mode="live")
        from .contracts import ContextSelection

        context = ContextSelection.model_validate(context_value)
        generation_value, generation_record, generation_cached = execute_stage(
            self.store, run_id=self.spec.experiment_id, stage="generation", plugin_id=self.spec.generator.model,
            config={"model": self.spec.generator.model, "provider_only": self.provider_only,
                    "max_output_tokens": self.spec.generator.max_output_tokens,
                    "completion_limit": "max_completion_tokens", "reasoning_enabled": False},
            upstream_hashes=(context_record.output_hash,),
            input_value={"query": query.model_dump(), "context": context.model_dump()},
            call=lambda _: generate_answer(query, context, self._client("openrouter"), self.spec.generator.model,
                                           self.provider_only or "", self.spec.generator.max_output_tokens).model_dump(),
            query_id=query.query_id, mode="live")
        return (context, GenerationResult.model_validate(generation_value),
                (context_record, generation_record), (context_cached, generation_cached))

    def run_one(self, query, retriever, reranker, generate=False, judge=False):
        if judge and not generate:
            raise ValueError("Judging requires a generated answer")
        gold = self.gold.get(query.query_id)
        pool, pool_record, pool_cached = self.retrieve(query, retriever)
        ranking, rerank_record, rerank_cached = self.rerank(query, pool, pool_record, reranker)
        result = {"dataset": self.dataset_id, "query_id": query.query_id,
                  "retriever": retriever, "reranker": reranker,
                  "pool_hash": pool.ordered_hash, "ranking_hash": rerank_record.output_hash,
                  "retrieval": evaluate_ranking(retrieval_order(pool), gold) if gold else {"applicable": False, "reason": "service_query_has_no_labels"},
                  "ranking": evaluate_ranking(ranking, gold) if gold else {"applicable": False, "reason": "service_query_has_no_labels"},
                  "cache": {"pool": pool_cached, "rerank": rerank_cached}}
        if generate:
            context, answer, records, cached = self.answer(query, ranking, rerank_record)
            result["answer"] = evaluate_answer(answer, gold, context) if gold else {"answer_applicable": False, "reason": "service_query_has_no_labels"}
            result["answer_text"] = answer.answer
            result["citations"] = answer.citations
            result["generation_hash"] = records[1].output_hash
            result["cache"].update({"context": cached[0], "generation": cached[1]})
            if judge:
                judge_gold = gold or GoldRecord(dataset_id=query.dataset_id, query_id=query.query_id)
                value, judge_record, judge_cached = execute_stage(
                    self.store, run_id=self.spec.experiment_id, stage="judge",
                    plugin_id=self.spec.generator.model,
                    config={"rubric": JUDGE_RUBRIC_V1, "model": self.spec.generator.model,
                            "provider_only": self.provider_only},
                    upstream_hashes=(records[1].output_hash,),
                    input_value={"query": query.model_dump(), "context": context.model_dump(),
                                 "answer": answer.model_dump(), "gold": judge_gold.model_dump()},
                    call=lambda _: judge_answer(query, context, answer, judge_gold,
                                                self._client("openrouter"),
                                                self.spec.generator.model,
                                                self.provider_only or "").model_dump(),
                    query_id=query.query_id, mode="live")
                judgement = JudgeResult.model_validate(value)
                result["judge"] = {"faithfulness": judgement.faithfulness,
                                   "answer_relevancy": judgement.answer_relevancy,
                                   "answer_correctness": judgement.answer_correctness}
                result["judge_hash"] = judge_record.output_hash
                result["cache"]["judge"] = judge_cached
        return result

    def run_slice(self, *, retriever: str, reranker: str, split="development", limit=5, generate=False, judge=False, workers=1):
        if limit < 1:
            raise ValueError("Development slice limit must be positive")
        if workers < 1:
            raise ValueError("Worker count must be positive")
        selected = [query for query in self.queries if query.split == split][:limit]
        if not selected:
            raise ValueError(f"No queries in {split} split")
        started = time.monotonic()
        results = [None] * len(selected)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            next_index = 0
            pending = {}
            while next_index < min(workers, len(selected)):
                future = executor.submit(self.run_one, selected[next_index], retriever, reranker, generate, judge)
                pending[future] = next_index
                next_index += 1
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    index = pending.pop(future)
                    try:
                        results[index] = future.result()
                    except Exception as exc:
                        if "cap" in str(exc).lower() or "budget" in str(exc).lower():
                            for remaining in pending:
                                remaining.cancel()
                            raise
                        results[index] = {"dataset": self.dataset_id, "query_id": selected[index].query_id,
                                          "retriever": retriever, "reranker": reranker,
                                          "status": "failed", "error_type": type(exc).__name__,
                                          "error": str(exc)}
                    if next_index < len(selected):
                        new_future = executor.submit(self.run_one, selected[next_index], retriever, reranker, generate, judge)
                        pending[new_future] = next_index
                        next_index += 1
        elapsed = time.monotonic() - started
        path = self.dataset_root / "results" / f"{retriever}__{reranker}__{split}__{limit}__{'judge' if judge else 'answers' if generate else 'ranking'}__{digest(results)[:12]}.json"
        write_json(path, results)
        metadata = {"path": str(path), "queries": len(results),
                    "failed_queries": sum(row.get("status") == "failed" for row in results),
                    "results_hash": digest(results), "elapsed_seconds": elapsed,
                    "queries_per_second": len(results) / elapsed if elapsed else None,
                    "workers": workers}
        write_json(path.with_suffix(".meta.json"), metadata)
        return metadata
