"""Load every pinned local model and check one tiny embedding/reranking batch."""

import argparse
import gc
import json
import math
import time
from pathlib import Path

from jev_eval.io import write_json
from jev_eval.rag.embedding import QwenEncoder
from jev_eval.rag.reranking import CrossEncoderModel, QwenRerankerModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revisions", type=Path, default=Path("configs/rag/model_revisions.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/production_rag_v1/preflight/local_models.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    revisions = json.loads(args.revisions.read_text())
    query = "What color is the sky?"
    documents = ["The sky is blue.", "Bananas are yellow."]
    results = {}
    for name, revision in revisions.items():
        started = time.monotonic()
        if name == "Qwen/Qwen3-Embedding-0.6B":
            model = QwenEncoder(name, revision, device=args.device)
            vectors = model.encode(documents)
            query_vector = model.encode([query], query=True)
            if vectors.shape[0] != 2 or query_vector.shape != (1, vectors.shape[1]):
                raise ValueError(f"Embedding output shape mismatch: {name}")
            scores = (vectors @ query_vector[0]).tolist()
            dimension = vectors.shape[1]
        else:
            model = (QwenRerankerModel(name, revision, device=args.device) if name.startswith("Qwen/")
                     else CrossEncoderModel(name, revision, device=args.device))
            scores = [float(value) for value in model.score(query, documents)]
            dimension = None
        if len(scores) != 2 or not all(math.isfinite(value) for value in scores):
            raise ValueError(f"Invalid scores from {name}")
        results[name] = {"revision": revision, "device": str(getattr(model, "device", args.device)),
                         "seconds": round(time.monotonic() - started, 3),
                         "scores": scores, "embedding_dimension": dimension,
                         "relevant_above_irrelevant": scores[0] > scores[1]}
        del model
        gc.collect()
    write_json(args.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
