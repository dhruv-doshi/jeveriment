"""CPU forward passes through tiny randomly initialized architectures, no downloads."""

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3Model,
    RobertaConfig,
    RobertaForSequenceClassification,
)

from jev_eval.models import DenseModel, LocalReranker


def tokenizer(**kwargs):
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "yes": 2,
                "no": 3,
                "tides": 4,
                "gravity": 5,
                "cake": 6,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        padding_side=kwargs.get("padding_side", "right"),
    )


def qwen_config():
    return Qwen3Config(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        pad_token_id=1,
    )


def test_qwen_dense_and_reranker_cpu_forward(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: tokenizer(**k)
    )
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        lambda *a, **k: Qwen3Model(qwen_config()),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForCausalLM.from_pretrained",
        lambda *a, **k: Qwen3ForCausalLM(qwen_config()),
    )
    dense = DenseModel("Qwen/Qwen3-Embedding-0.6B", "fixture", "cpu")
    embeddings = dense.encode(["tides gravity", "cake"])
    assert embeddings.shape == (2, 16)
    assert np.linalg.norm(embeddings, axis=1) == pytest.approx([1, 1])
    reranker = LocalReranker("Qwen/Qwen3-Reranker-0.6B", "fixture", "cpu")
    scores = reranker.score("tides", ["gravity", "cake"])
    assert len(scores) == 2 and all(0 <= p <= 1 for p in scores)
    reranker.max_length = 1
    with pytest.raises(ValueError, match="exceeds"):
        reranker.score("tides", ["gravity"])


def test_bge_raw_logits_cpu_forward(monkeypatch):
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", lambda *a, **k: tokenizer(**k)
    )
    config = RobertaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_labels=1,
        pad_token_id=1,
        max_position_embeddings=512,
    )
    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        lambda *a, **k: RobertaForSequenceClassification(config),
    )
    model = LocalReranker("BAAI/bge-reranker-v2-m3", "fixture", "cpu")
    scores = model.score("tides", ["gravity", "cake"])
    assert len(scores) == 2 and np.isfinite(scores).all()
