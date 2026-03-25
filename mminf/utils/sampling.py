"""Generic token sampling utilities.

Supports temperature scaling, top-k filtering, top-p (nucleus) filtering,
and multinomial sampling. Model-agnostic — any AR model returns logits,
this module selects the next token.

Usage:
    from mminf.utils.sampling import sample_tokens
    tokens = sample_tokens(logits, temperature=0.7, top_p=0.9)
"""

import torch
import torch.nn.functional as F


def sample_tokens(
    logits: torch.Tensor,
    temperature: float | torch.Tensor = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample tokens from logits with temperature, top-k, and top-p.

    Args:
        logits: [batch_size, vocab_size] raw logits from lm_head.
        temperature: Scalar or per-request tensor [batch_size, 1].
            0 = greedy (argmax). >0 = scaled sampling.
        top_k: Keep only top-k logits. 0 = disabled.
        top_p: Nucleus sampling threshold. 1.0 = disabled.

    Returns:
        tokens: [batch_size] sampled token IDs.
    """
    # Greedy fast path
    if _is_greedy(temperature):
        return torch.argmax(logits, dim=-1)

    # Temperature scaling
    if isinstance(temperature, torch.Tensor):
        logits = logits / temperature
    else:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        logits = _top_k_filter(logits, top_k)

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        logits = _top_p_filter(logits, top_p)

    # Sample
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _is_greedy(temperature: float | torch.Tensor) -> bool:
    if isinstance(temperature, torch.Tensor):
        return (temperature == 0).all().item()
    return temperature == 0


def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all logits below the top-k values."""
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[:, -1].unsqueeze(-1)
    return logits.where(logits >= threshold, torch.full_like(logits, float("-inf")))


def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filtering: keep smallest set of tokens with cumulative prob >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    # Shift right so the first token above threshold is kept
    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_mask] = float("-inf")

    # Scatter back to original ordering
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)
