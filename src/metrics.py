"""
metrics.py
----------
Metrics for evaluating adversarial robustness of Sparse Autoencoder (SAE) feature
interpretations. All functions operate on CPU or GPU tensors and return Python scalars
so they are easy to log and compare across experiments.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


def cosine_similarity_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two activation tensors after flattening.

    Both tensors are flattened to 1-D vectors before the similarity is computed,
    so the input shapes do not need to match as long as their total number of
    elements is the same.

    Parameters
    ----------
    a : torch.Tensor
        First activation tensor of arbitrary shape.
    b : torch.Tensor
        Second activation tensor of the same total number of elements as ``a``.

    Returns
    -------
    float
        Cosine similarity in [-1, 1].  Returns 0.0 when either tensor is the
        zero vector (undefined similarity).

    Examples
    --------
    >>> a = torch.tensor([1.0, 0.0, 0.0])
    >>> cosine_similarity_flat(a, a)
    1.0
    """
    a_flat = a.flatten().unsqueeze(0).float()
    b_flat = b.flatten().unsqueeze(0).float()
    # F.cosine_similarity returns a 1-element tensor; guard against zero norms.
    norm_a = a_flat.norm()
    norm_b = b_flat.norm()
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return F.cosine_similarity(a_flat, b_flat, dim=1).item()


def jaccard_active_features(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: float = 0.0,
) -> float:
    """Jaccard index of the active feature sets defined by a threshold.

    A feature is considered *active* when its value is strictly greater than
    ``threshold``.

    Parameters
    ----------
    a : torch.Tensor
        First feature activation vector (1-D) or tensor (will be flattened).
    b : torch.Tensor
        Second feature activation vector, same shape as ``a``.
    threshold : float, optional
        Activation threshold; default is 0.0.

    Returns
    -------
    float
        Jaccard index in [0, 1].  Returns 1.0 when both tensors have no active
        features (both empty sets are equal by convention).

    Examples
    --------
    >>> a = torch.zeros(10)
    >>> jaccard_active_features(a, a)
    1.0
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    active_a: set = set((a_flat > threshold).nonzero(as_tuple=True)[0].tolist())
    active_b: set = set((b_flat > threshold).nonzero(as_tuple=True)[0].tolist())

    union = active_a | active_b
    if len(union) == 0:
        return 1.0

    intersection = active_a & active_b
    return len(intersection) / len(union)


def rank_correlation_topk(
    a: torch.Tensor,
    b: torch.Tensor,
    k: int = 50,
) -> float:
    """Spearman rank correlation of feature magnitudes over the union of top-k features.

    The top-k features are selected from each tensor independently; their union
    forms the index set over which Spearman correlation is computed.  This
    focuses the metric on the most semantically meaningful features rather than
    the dense tail.

    Parameters
    ----------
    a : torch.Tensor
        First feature activation vector (will be flattened).
    b : torch.Tensor
        Second feature activation vector (will be flattened).
    k : int, optional
        Number of top features to consider from each tensor; default is 50.

    Returns
    -------
    float
        Spearman rank correlation in [-1, 1].  Returns 0.0 when the union
        contains fewer than 2 elements (correlation is undefined).

    Examples
    --------
    >>> a = torch.rand(100)
    >>> rank_correlation_topk(a, a, k=10)
    1.0
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    n = a_flat.numel()
    actual_k = min(k, n)

    top_a_indices = set(torch.topk(a_flat, actual_k).indices.tolist())
    top_b_indices = set(torch.topk(b_flat, actual_k).indices.tolist())
    union_indices = sorted(top_a_indices | top_b_indices)

    if len(union_indices) < 2:
        return 0.0

    a_vals = a_flat[union_indices].cpu().numpy()
    b_vals = b_flat[union_indices].cpu().numpy()

    result = stats.spearmanr(a_vals, b_vals)
    correlation = result.statistic if hasattr(result, "statistic") else result.correlation
    # Guard against NaN (can occur with constant arrays).
    return float(correlation) if not np.isnan(correlation) else 0.0


def kl_divergence_logits(
    logits_clean: torch.Tensor,
    logits_perturbed: torch.Tensor,
) -> float:
    """KL divergence KL(P_clean || P_perturbed) between next-token output distributions.

    Computes the KL divergence between the softmax distributions derived from
    the clean and perturbed logits.  When 3-D logits are provided (batch x seq x
    vocab), only the *last* token position is used, which is the standard
    auto-regressive prediction target.

    Parameters
    ----------
    logits_clean : torch.Tensor
        Logits from the unperturbed model.  Shape can be
        ``[batch, seq_len, vocab_size]``, ``[seq_len, vocab_size]``, or
        ``[vocab_size]``.
    logits_perturbed : torch.Tensor
        Logits from the perturbed model, same shape as ``logits_clean``.

    Returns
    -------
    float
        KL divergence KL(P_clean || P_perturbed) >= 0.  Returns 0.0 for
        identical inputs (up to floating-point precision).

    Notes
    -----
    A small epsilon (1e-10) is added before taking the log to ensure numerical
    stability when a probability mass is near zero.

    Examples
    --------
    >>> a = torch.randn(1, 5, 100)
    >>> kl_divergence_logits(a, a)
    0.0
    """
    epsilon = 1e-10

    # Select last-token position for 3-D tensors.
    if logits_clean.dim() == 3:
        lc = logits_clean[:, -1, :].float()
        lp = logits_perturbed[:, -1, :].float()
    else:
        lc = logits_clean.float()
        lp = logits_perturbed.float()

    # Ensure both are at least 2-D (batch x vocab) for batchmean reduction.
    if lc.dim() == 1:
        lc = lc.unsqueeze(0)
        lp = lp.unsqueeze(0)

    p = F.softmax(lc, dim=-1) + epsilon
    q = F.softmax(lp, dim=-1) + epsilon

    # Renormalise after adding epsilon.
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    # F.kl_div expects log-probabilities as the first argument.
    kl = F.kl_div(q.log(), p, reduction="batchmean")
    return max(0.0, kl.item())  # clamp away tiny negatives from floating-point


def feature_flip_count(
    a: torch.Tensor,
    b: torch.Tensor,
    threshold: float = 0.0,
) -> int:
    """Count features that changed active/inactive status between two tensors.

    A *flip* occurs whenever a feature is active in one tensor and inactive in
    the other, where active means strictly greater than ``threshold``.

    Parameters
    ----------
    a : torch.Tensor
        First feature activation tensor (will be flattened).
    b : torch.Tensor
        Second feature activation tensor, same total elements as ``a``.
    threshold : float, optional
        Activation threshold; default is 0.0.

    Returns
    -------
    int
        Number of features whose active/inactive status differs between ``a``
        and ``b``.

    Examples
    --------
    >>> a = torch.ones(5)
    >>> b = torch.zeros(5)
    >>> feature_flip_count(a, b)
    5
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    flipped = (a_flat > threshold) != (b_flat > threshold)
    return int(flipped.sum().item())


def compute_all_metrics(
    clean_sae_acts: torch.Tensor,
    perturbed_sae_acts: torch.Tensor,
    clean_logits: torch.Tensor,
    perturbed_logits: torch.Tensor,
    topk: int = 50,
    threshold: float = 0.0,
) -> dict:
    """Compute all robustness metrics at once and return them as a dictionary.

    This convenience function bundles every metric defined in this module into a
    single call, making it straightforward to log a complete evaluation snapshot.

    Parameters
    ----------
    clean_sae_acts : torch.Tensor
        SAE feature activations for the clean (unperturbed) input.
    perturbed_sae_acts : torch.Tensor
        SAE feature activations for the adversarially perturbed input.
    clean_logits : torch.Tensor
        Model output logits for the clean input.
    perturbed_logits : torch.Tensor
        Model output logits for the perturbed input.
    topk : int, optional
        ``k`` passed to :func:`rank_correlation_topk`; default is 50.
    threshold : float, optional
        Activation threshold used for Jaccard and flip-count; default is 0.0.

    Returns
    -------
    dict
        Dictionary with the following keys and types:

        * ``cosine_similarity`` (float): cosine similarity of SAE activations.
        * ``jaccard_index`` (float): Jaccard index of active feature sets.
        * ``rank_correlation`` (float): Spearman correlation over top-k features.
        * ``kl_divergence`` (float): KL(P_clean || P_perturbed) over next-token logits.
        * ``feature_flip_count`` (int): number of features that flipped status.
        * ``sae_disruption`` (float): 1 - cosine_similarity; higher means more disrupted.

    Examples
    --------
    >>> acts = torch.rand(1, 10, 512)
    >>> logits = torch.randn(1, 10, 50257)
    >>> metrics = compute_all_metrics(acts, acts, logits, logits)
    >>> metrics["sae_disruption"]
    0.0
    """
    cos_sim = cosine_similarity_flat(clean_sae_acts, perturbed_sae_acts)
    jaccard = jaccard_active_features(clean_sae_acts, perturbed_sae_acts, threshold=threshold)
    rank_corr = rank_correlation_topk(clean_sae_acts, perturbed_sae_acts, k=topk)
    kl_div = kl_divergence_logits(clean_logits, perturbed_logits)
    flip_count = feature_flip_count(clean_sae_acts, perturbed_sae_acts, threshold=threshold)

    return {
        "cosine_similarity": cos_sim,
        "jaccard_index": jaccard,
        "rank_correlation": rank_corr,
        "kl_divergence": kl_div,
        "feature_flip_count": flip_count,
        "sae_disruption": 1.0 - cos_sim,
    }
