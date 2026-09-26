"""One synthetic, metered request per provider using the experiment clients."""

import argparse
import json
import os
from decimal import Decimal
from pathlib import Path

from dotenv import dotenv_values

from jev_eval.io import digest, read_json, write_json
from jev_eval.rag.answer import generate_answer
from jev_eval.rag.clients import BudgetedHTTP
from jev_eval.rag.contracts import ChunkRecord, ContextSelection, QueryRecord
from jev_eval.rag.reranking import score_with_jev
from jev_eval.rag.retrieval import ChunkIndex, freeze_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("provider", choices=("typesafe", "openrouter"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--root", type=Path, default=Path("runs/production_rag_v1"))
    parser.add_argument("--jev-model")
    args = parser.parse_args()
    secrets = dotenv_values(args.env_file)
    if args.provider == "typesafe":
        model = args.jev_model or secrets.get("JEV_MODEL") or os.getenv("JEV_MODEL")
        if not model:
            parser.error("JEV_MODEL is required")
        token = secrets.get("TYPESAFE_API_KEY") or os.getenv("TYPESAFE_API_KEY")
        client = BudgetedHTTP(provider="typesafe", root=args.root, token=token or "",
                              base_url=secrets.get("TYPESAFE_API_BASE_URL") or secrets.get("JEV_API_BASE_URL") or "https://api.typesafe.ai",
                              budget_usd=Decimal("50"), max_tokens=100000, max_requests=100,
                              input_usd_per_million=Decimal("0.042"), output_usd_per_million=Decimal(0),
                              rpm=2, tpm=10000, inflight=1)
        index = ChunkIndex(args.root / "preflight" / "chunks.sqlite")
        text = "The synthetic patient has a fever."
        chunk = ChunkRecord(dataset_id="preflight", document_id="synthetic", chunk_id="synthetic:0",
                            text=text, text_hash=digest(text))
        if index.count() == 0:
            index.add([chunk])
        query = QueryRecord(dataset_id="preflight", query_id="synthetic-jev", text="What symptom does the patient have?",
                            split="preflight", task="scientific_paper_question_answering")
        pool = freeze_pool(query, "synthetic", [(chunk.chunk_id, 1.0)], index, 1)
        result = score_with_jev(query, pool, index, client, model=model, workers=1)
        raw = read_json(result.raw_response_paths[chunk.chunk_id])
        summary = {"provider": "typesafe", "requested_model": model,
                   "score": result.scores[chunk.chunk_id], "raw_path": result.raw_response_paths[chunk.chunk_id],
                   "resolved_model": raw["response"].get("model"),
                   "usage": raw["response"].get("usage"),
                   "api_latency_ms": raw.get("elapsed_ms"),
                   "server_timing": raw.get("server_timing"),
                   "rate_headers": raw.get("rate_headers"),
                   "budget": client._ledger().totals()}
    else:
        token = secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        client = BudgetedHTTP(provider="openrouter", root=args.root, token=token or "",
                              base_url="https://openrouter.ai/api/v1", budget_usd=Decimal("50"),
                              max_tokens=100000, max_requests=100,
                              input_usd_per_million=Decimal("0.45"), output_usd_per_million=Decimal("2.25"),
                              rpm=2, tpm=10000, inflight=1)
        query = QueryRecord(dataset_id="preflight", query_id="synthetic-openrouter",
                            text="What color is the sky?", split="preflight",
                            task="scientific_paper_question_answering")
        context = ContextSelection(query_id=query.query_id, selected_ids=("synthetic:0",),
                                   dropped_ids=(), text="[synthetic:0] The sky is blue.",
                                   estimated_tokens=12, budget_tokens=128)
        result = generate_answer(query, context, client, "moonshotai/kimi-k2.5", "SiliconFlow", max_output_tokens=512)
        summary = {"provider": "openrouter", "requested_model": result.model_requested,
                   "resolved_model": result.model_resolved, "answer": result.answer,
                   "citations": result.citations, "usage": result.usage,
                   "generation_id": result.provider_generation_id,
                   "provider_name": result.provider_name,
                   "provider_backend_model": result.provider_backend_model,
                   "api_latency_ms": result.api_latency_ms,
                   "provider_reported_latency_ms": result.provider_reported_latency_ms,
                   "provider_reported_generation_time_ms": result.provider_reported_generation_time_ms,
                   "billed_cost_usd": result.billed_cost_usd,
                   "metadata_recorded": bool(result.provider_metadata_hash),
                   "budget": client._ledger().totals()}
    path = args.root / "preflight" / f"{args.provider}_summary.json"
    write_json(path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
