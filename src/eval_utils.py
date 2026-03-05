"""
eval_utils.py
-------------
Evaluation utilities for the SAE adversarial robustness project.

Provides:
- Per-prompt perplexity and next-token accuracy computation.
- Aggregate baseline statistics over a dataset of prompts.
- GPU memory logging helpers.

All model calls run under ``torch.no_grad()`` to avoid accumulating gradients
during evaluation.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Per-prompt metrics
# ---------------------------------------------------------------------------


def compute_perplexity(
    model: Any,
    tokens: torch.Tensor,
    device: str = "cuda",
) -> float:
    """Compute per-token perplexity of a model on a token sequence.

    Uses cross-entropy loss between model logits and shifted token targets.
    Position ``i`` predicts token ``i+1``, so the last position is excluded
    from the loss.

    Parameters
    ----------
    model : HookedTransformer
        A TransformerLens ``HookedTransformer`` model in eval mode.
    tokens : torch.Tensor
        Input token IDs, shape ``[1, seq_len]`` or ``[seq_len]``.  A leading
        batch dimension is added automatically if absent.
    device : str, optional
        PyTorch device string; default is ``"cuda"``.

    Returns
    -------
    float
        Perplexity (``exp`` of mean cross-entropy loss over all positions).
        Lower values indicate better next-token prediction.

    Notes
    -----
    Perplexity is defined as :math:`\\exp\\!\\left(\\frac{1}{N}\\sum_{i} \\mathcal{L}_i\\right)`,
    where :math:`\\mathcal{L}_i` is the cross-entropy loss at position ``i``.

    Examples
    --------
    >>> ppl = compute_perplexity(model, tokens, device="cpu")
    >>> print(f"Perplexity: {ppl:.2f}")
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    tokens = tokens.to(device)

    with torch.no_grad():
        logits = model(tokens)   # [1, seq_len, vocab_size]

    # Shift: predict token t+1 from position t.
    logits_shifted = logits[0, :-1, :]       # [seq_len - 1, vocab_size]
    targets = tokens[0, 1:]                  # [seq_len - 1]

    loss = F.cross_entropy(logits_shifted, targets)
    perplexity = torch.exp(loss).item()
    return float(perplexity)


def next_token_accuracy(
    model: Any,
    tokens: torch.Tensor,
    device: str = "cuda",
) -> float:
    """Compute next-token prediction accuracy over all positions.

    At each position the model's top-1 prediction is compared with the actual
    next token.  The accuracy is the fraction of positions where they agree.

    Parameters
    ----------
    model : HookedTransformer
        A TransformerLens ``HookedTransformer`` model in eval mode.
    tokens : torch.Tensor
        Input token IDs, shape ``[1, seq_len]`` or ``[seq_len]``.
    device : str, optional
        PyTorch device string; default is ``"cuda"``.

    Returns
    -------
    float
        Fraction of positions (in [0, 1]) where the top-1 predicted token
        equals the actual next token.

    Examples
    --------
    >>> acc = next_token_accuracy(model, tokens, device="cpu")
    >>> print(f"Next-token accuracy: {acc:.2%}")
    """
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)

    tokens = tokens.to(device)

    with torch.no_grad():
        logits = model(tokens)   # [1, seq_len, vocab_size]

    predictions = logits[0, :-1, :].argmax(dim=-1)   # [seq_len - 1]
    targets = tokens[0, 1:]                            # [seq_len - 1]

    correct = (predictions == targets).float().mean().item()
    return float(correct)


# ---------------------------------------------------------------------------
# Aggregate baseline statistics
# ---------------------------------------------------------------------------


def compute_baseline_stats(
    model: Any,
    sae: Any,
    token_batches: list[torch.Tensor],
    target_layer: int,
    device: str = "cuda",
) -> dict:
    """Compute baseline statistics over a set of prompts.

    Iterates over ``token_batches``, computes per-prompt metrics, and
    aggregates them into means and standard deviations.

    Parameters
    ----------
    model : HookedTransformer
        A TransformerLens ``HookedTransformer`` model in eval mode.
    sae : SAE
        Loaded SAE for ``target_layer``.
    token_batches : list of torch.Tensor
        List of token tensors, each shape ``[1, seq_len]``.
    target_layer : int
        Layer index to extract SAE features from.
    device : str, optional
        PyTorch device string; default is ``"cuda"``.

    Returns
    -------
    dict
        Aggregate statistics with the following keys:

        * ``mean_perplexity`` (float): mean perplexity across prompts.
        * ``std_perplexity`` (float): standard deviation of perplexity.
        * ``mean_l0`` (float): mean active features per token.
        * ``std_l0`` (float): standard deviation of L0.
        * ``mean_fvu`` (float): mean fraction of variance unexplained.
        * ``std_fvu`` (float): standard deviation of FVU.
        * ``mean_next_token_acc`` (float): mean next-token accuracy.
        * ``std_next_token_acc`` (float): standard deviation of accuracy.
        * ``n_prompts`` (int): number of prompts processed.

    Examples
    --------
    >>> stats = compute_baseline_stats(model, sae, token_batches, target_layer=6)
    >>> print(f"Mean PPL: {stats['mean_perplexity']:.2f}")
    """
    # Local imports to avoid circular dependencies.
    from src.sae_utils import compute_reconstruction_error

    perplexities: list[float] = []
    l0_values: list[float] = []
    fvu_values: list[float] = []
    accuracies: list[float] = []

    with torch.no_grad():
        for tokens in tqdm(token_batches, desc="Computing baseline stats"):
            tokens = tokens.to(device)

            # Perplexity and next-token accuracy.
            ppl = compute_perplexity(model, tokens, device=device)
            acc = next_token_accuracy(model, tokens, device=device)

            # SAE reconstruction stats (L0 and FVU).
            recon = compute_reconstruction_error(
                model, sae, tokens, target_layer, device=device
            )

            perplexities.append(ppl)
            accuracies.append(acc)
            l0_values.append(recon["mean_l0"])
            fvu_values.append(recon["fvu"])

    n = len(perplexities)

    return {
        "mean_perplexity": float(np.mean(perplexities)),
        "std_perplexity": float(np.std(perplexities)),
        "mean_l0": float(np.mean(l0_values)),
        "std_l0": float(np.std(l0_values)),
        "mean_fvu": float(np.mean(fvu_values)),
        "std_fvu": float(np.std(fvu_values)),
        "mean_next_token_acc": float(np.mean(accuracies)),
        "std_next_token_acc": float(np.std(accuracies)),
        "n_prompts": n,
    }


# ---------------------------------------------------------------------------
# GPU memory utilities
# ---------------------------------------------------------------------------


def log_gpu_memory(label: str = "") -> dict:
    """Log current GPU memory usage if CUDA is available.

    Prints a human-readable summary and returns the raw values as a
    dictionary for downstream logging (e.g., W&B, CSV).

    Parameters
    ----------
    label : str, optional
        Descriptive label for the log entry, e.g. ``"after adversarial pass"``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        * ``label`` (str): the provided label.
        * ``allocated_gb`` (float): currently allocated GPU memory in GiB.
        * ``reserved_gb`` (float): total reserved (cached) GPU memory in GiB.
        * ``cuda_available`` (bool): ``True`` if CUDA is available on this host.

    Examples
    --------
    >>> mem = log_gpu_memory("before forward pass")
    GPU memory [before forward pass] — allocated: 1.23 GB, reserved: 2.00 GB
    """
    _gb = 1024 ** 3

    if not torch.cuda.is_available():
        result = {
            "label": label,
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "cuda_available": False,
        }
        tag = f" [{label}]" if label else ""
        print(f"GPU memory{tag} — CUDA not available on this host.")
        return result

    allocated_gb = torch.cuda.memory_allocated() / _gb
    reserved_gb = torch.cuda.memory_reserved() / _gb

    result = {
        "label": label,
        "allocated_gb": float(allocated_gb),
        "reserved_gb": float(reserved_gb),
        "cuda_available": True,
    }

    tag = f" [{label}]" if label else ""
    print(
        f"GPU memory{tag} — "
        f"allocated: {allocated_gb:.2f} GB, "
        f"reserved: {reserved_gb:.2f} GB"
    )
    return result
