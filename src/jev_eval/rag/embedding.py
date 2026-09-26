"""Qwen3 dense embedding adapter with pinned weights and normalized outputs."""

import numpy as np


class QwenEncoder:
    def __init__(self, name: str, revision: str, device="auto", max_length=512,
                 task="Retrieve passages that contain evidence for answering the question"):
        if not revision:
            raise ValueError("Embedding revision must be pinned")
        self.name, self.revision = name, revision
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, padding_side="left")
        self.model = AutoModel.from_pretrained(name, revision=revision, dtype=torch.float32)
        self.device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if device == "auto" else device
        self.model.to(self.device).eval()
        self.max_length, self.task = max_length, task

    def encode(self, texts: list[str], query=False):
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        formatted = [f"Instruct: {self.task}\nQuery: {text}" if query else text for text in texts]
        tokens = self.tokenizer(formatted, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        with self.torch.inference_mode():
            hidden = self.model(**tokens.to(self.device)).last_hidden_state
            # Left padding leaves the final token at the same position in each row.
            vectors = hidden[:, -1].float()
            vectors = self.torch.nn.functional.normalize(vectors, p=2, dim=1)
            result = vectors.cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite embedding")
        return result
