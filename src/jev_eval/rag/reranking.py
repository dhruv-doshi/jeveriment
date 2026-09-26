"""Fixed-pool rerankers; every arm sees the same exact candidate text."""

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from jev_eval.io import digest

from .clients import BudgetedHTTP, validate_jev
from .contracts import CandidatePool, QueryRecord, RerankResult

RUBRIC_V1 = "Does this passage contain evidence that would help answer the question accurately?"
QWEN_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
QWEN_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def _ordered(pool, scores, reranker_id, raw=None, paths=None):
    expected = [item.chunk_id for item in pool.candidates]
    if set(scores) != set(expected) or any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("Reranker must return one finite score per frozen candidate")
    original = {cid: i for i, cid in enumerate(expected)}
    ordered = sorted(expected, key=lambda cid: (-scores[cid], original[cid]))
    return RerankResult(query_id=pool.query_id, pool_hash=pool.ordered_hash,
                        reranker_id=reranker_id, ordered_ids=tuple(ordered),
                        scores=scores, raw_response_hashes=raw or {}, raw_response_paths=paths or {})


def retrieval_order(pool: CandidatePool):
    return _ordered(pool, {item.chunk_id: float(len(pool.candidates) - item.rank) for item in pool.candidates}, "retrieval_order")


def score_with_model(query: QueryRecord, pool: CandidatePool, index, model, reranker_id: str, batch_size=8):
    chunks = [index.get(item.chunk_id) for item in pool.candidates]
    if any(chunk.text_hash != item.text_hash for chunk, item in zip(chunks, pool.candidates)):
        raise ValueError("Candidate text changed after pool freeze")
    scores = {}
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        values = model.score(query.text, [chunk.text for chunk in batch])
        if len(values) != len(batch):
            raise ValueError("Local reranker returned wrong score count")
        scores.update({chunk.chunk_id: float(value) for chunk, value in zip(batch, values)})
    return _ordered(pool, scores, reranker_id)


class CrossEncoderModel:
    def __init__(self, name: str, revision: str, device="auto", max_length=512):
        if not revision:
            raise ValueError("Local reranker revision must be pinned")
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(name, revision=revision, device=None if device == "auto" else device, max_length=max_length)

    def score(self, query, texts):
        return self.model.predict([(query, text) for text in texts], show_progress_bar=False)


class QwenRerankerModel:
    def __init__(self, name: str, revision: str, device="auto", max_length=512):
        if not revision:
            raise ValueError("Qwen reranker revision must be pinned")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.float32)
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
        self.model.to(self.device).eval()
        self.max_length = max_length
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")

    def score(self, query, texts):
        prompts = [QWEN_PREFIX + f"<Instruct>: {RUBRIC_V1}\n<Query>: {query}\n<Document>: {text}" + QWEN_SUFFIX for text in texts]
        tokens = self.tokenizer(prompts, padding=True, truncation=True, max_length=self.max_length, add_special_tokens=False, return_tensors="pt")
        with self.torch.inference_mode():
            logits = self.model(**tokens.to(self.device)).logits[:, -1, [self.no_id, self.yes_id]].float()
            return self.torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()


def jev_score_payload(query: QueryRecord, chunk, model: str):
    return {
        "model": model,
        "state": {"query_id": query.query_id, "query": query.text,
                  "document": {"id": chunk.chunk_id, "text": chunk.text}},
        "questions": {"relevance": {"type": "noul", "instructions": RUBRIC_V1}},
    }


def score_with_jev(query: QueryRecord, pool: CandidatePool, index, client: BudgetedHTTP,
                   model: str | None = None, workers=4, executor=None):
    model = model or os.getenv("JEV_MODEL")
    if not model:
        raise ValueError("JEV_MODEL is required for live JEV scoring")
    scores, raw, paths = {}, {}, {}

    def one(item):
        chunk = index.get(item.chunk_id)
        if chunk.text_hash != item.text_hash:
            raise ValueError("Candidate text changed after pool freeze")
        payload = jev_score_payload(query, chunk, model)
        # UTF-8 bytes are a conservative upper bound for input token count.
        input_bound = len(json.dumps(payload, ensure_ascii=False).encode())
        reply = client.post("/v1/systemone", payload, max_input_tokens=input_bound,
                            max_output_tokens=64, validate=validate_jev,
                            provenance={"query_id": query.query_id, "dataset_id": query.dataset_id,
                                        "chunk_id": chunk.chunk_id, "reranker": "jev"})
        if reply.body["model"] != model:
            raise ValueError(f"JEV resolved model {reply.body['model']} differs from requested {model}")
        return item.chunk_id, float(reply.body["answers"]["relevance"]["noul"]), digest(reply.body), reply.raw_path

    own_pool = executor is None
    executor = executor or ThreadPoolExecutor(max_workers=workers)
    futures = []
    try:
        futures = [executor.submit(one, item) for item in pool.candidates]
        for future in as_completed(futures):
            cid, score, response_hash, raw_path = future.result()
            scores[cid] = score
            raw[cid] = response_hash
            paths[cid] = raw_path
    except Exception:
        for future in futures:
            future.cancel()
        raise
    finally:
        if own_pool:
            executor.shutdown(wait=True)
    return _ordered(pool, scores, "jev", raw, paths)
