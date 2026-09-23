"""Pinned local inference; text truncation is owned by the shared-view builder."""

import gc
import warnings

import numpy as np

TASK = "Given a scientific claim, retrieve evidence that supports or refutes the claim"
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def reranker_text(query, document, task=TASK):
    return (
        PREFIX
        + f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {document}"
        + SUFFIX
    )


def device_name(requested="auto"):
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _DeviceFallback:
    def _place_model(self):
        try:
            self.model.to(self.device)
        except (RuntimeError, NotImplementedError) as error:
            if self.device == "cpu":
                raise
            self._move_to_cpu(error)
        self.model.eval()

    def _move_to_cpu(self, error):
        import torch

        accelerator = self.device
        self.device = "cpu"
        gc.collect()
        if accelerator == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif accelerator.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model.to("cpu")
        warnings.warn(
            f"Inference on {accelerator} failed; continuing on CPU: {error}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _run_with_fallback(self, run):
        try:
            return run()
        except (RuntimeError, NotImplementedError) as error:
            if self.device == "cpu":
                raise
            self._move_to_cpu(error)
            return run()


def release(model):
    del model
    gc.collect()
    import torch

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


class DenseModel(_DeviceFallback):
    def __init__(self, name, revision, device="auto", max_length=512, task=TASK):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if not revision:
            raise ValueError("Dense model revision must be resolved before loading")
        if name not in {"Qwen/Qwen3-Embedding-0.6B", "Alibaba-NLP/gte-modernbert-base"}:
            raise ValueError("Dense model needs a validated pooling adapter")
        self.qwen = name.startswith("Qwen/")
        self.device = device_name(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, revision=revision, padding_side="left" if self.qwen else "right"
        )
        self.model = AutoModel.from_pretrained(
            name, revision=revision, torch_dtype=torch.float32
        )
        self._place_model()
        self.max_length = max_length
        self.task = task

    def encode(self, texts, query=False):
        import torch

        formatted = [
            f"Instruct: {self.task}\nQuery:{t}" if query and self.qwen else t
            for t in texts
        ]
        encoded = self.tokenizer(
            formatted, padding=True, truncation=False, return_tensors="pt"
        )
        if encoded["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                "Dense input exceeds frozen length; shared view must be rebuilt"
            )

        def run():
            with torch.inference_mode():
                hidden = self.model(**encoded.to(self.device)).last_hidden_state
                output = hidden[:, -1] if self.qwen else hidden[:, 0]
                return (
                    torch.nn.functional.normalize(output, p=2, dim=1)
                    .float()
                    .cpu()
                    .numpy()
                )

        result = self._run_with_fallback(run)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite dense embedding")
        return result


class LocalReranker(_DeviceFallback):
    def __init__(self, name, revision, device="auto", max_length=512, task=TASK):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        if not revision:
            raise ValueError("Reranker revision must be resolved before loading")
        self.qwen = name.startswith("Qwen/")
        self.device = device_name(device)
        self.max_length = max_length
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(
            name, revision=revision, padding_side="left" if self.qwen else "right"
        )
        cls = AutoModelForCausalLM if self.qwen else AutoModelForSequenceClassification
        self.model = cls.from_pretrained(
            name, revision=revision, torch_dtype=torch.float32
        )
        self._place_model()

    def score(self, query, texts):
        import torch

        if self.qwen:
            encoded = self.tokenizer(
                [reranker_text(query, t, self.task) for t in texts],
                padding=True,
                add_special_tokens=False,
                truncation=False,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(
                [query] * len(texts),
                texts,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
        if encoded["input_ids"].shape[1] > self.max_length:
            raise ValueError("Reranker input exceeds frozen shared-view limit")

        def run():
            with torch.inference_mode():
                logits = self.model(**encoded.to(self.device)).logits
                if self.qwen:
                    no, yes = [
                        self.tokenizer.convert_tokens_to_ids(t) for t in ("no", "yes")
                    ]
                    logits = logits[:, -1, [no, yes]].float()
                    return torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
                return logits.view(-1).float().cpu().tolist()

        result = self._run_with_fallback(run)
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite reranker scores")
        return result
