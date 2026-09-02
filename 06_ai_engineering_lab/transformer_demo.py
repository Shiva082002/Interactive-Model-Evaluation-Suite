"""Educational transformer block: embeddings, positions, attention, and MLP."""
from __future__ import annotations

import numpy as np


def positional_encoding(sequence_length: int, dimensions: int) -> np.ndarray:
    positions = np.arange(sequence_length)[:, None]
    rates = 1 / np.power(10000, (2 * (np.arange(dimensions)[None, :] // 2)) / dimensions)
    encoding = positions * rates
    encoding[:, 0::2] = np.sin(encoding[:, 0::2])
    encoding[:, 1::2] = np.cos(encoding[:, 1::2])
    return encoding


def self_attention(tokens: np.ndarray) -> np.ndarray:
    """Compute scaled dot-product self-attention for a token matrix."""
    dimensions = tokens.shape[-1]
    scores = tokens @ tokens.T / np.sqrt(dimensions)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ tokens


def transformer_block(tokens: np.ndarray) -> np.ndarray:
    enriched = tokens + positional_encoding(len(tokens), tokens.shape[-1])
    attended = enriched + self_attention(enriched)
    return attended + np.maximum(attended @ np.eye(tokens.shape[-1]), 0)


def demo_summary() -> dict[str, object]:
    tokens = np.arange(12, dtype=float).reshape(3, 4) / 10
    output = transformer_block(tokens)
    return {"input_shape": list(tokens.shape), "output_shape": list(output.shape), "attention_output": output.round(4).tolist()}
