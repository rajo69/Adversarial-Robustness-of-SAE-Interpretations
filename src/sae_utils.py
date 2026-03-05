"""
sae_utils.py
------------
Utilities for loading Sparse Autoencoders (SAEs) from SAE Lens and extracting
feature activations from TransformerLens (HookedTransformer) models.

All public functions follow a consistent interface:
- ``model`` is always a ``transformer_lens.HookedTransformer`` instance.
- ``sae`` is always an ``sae_lens.SAE`` instance.
- Tensors are moved to ``device`` before computation.
- Functions use ``torch.no_grad()`` internally where gradients are not needed.
"""

from typing import Any, Optional

import torch


# ---------------------------------------------------------------------------
# SAE loading
# ---------------------------------------------------------------------------


def load_sae_for_layer(
    release: str,
    sae_id: str,
    device: str = "cuda",
) -> tuple:
    """Load a pretrained SAE from SAE Lens for a given layer.

    Parameters
    ----------
    release : str
        SAE release name as used by SAE Lens, e.g.
        ``"gemma-scope-2b-pt-res"`` or ``"gpt2-small-res-jb"``.
    sae_id : str
        SAE identifier within the release, e.g.
        ``"layer_12/width_16k/average_l0_82"``.
    device : str, optional
        PyTorch device string to load the SAE onto; default is ``"cuda"``.

    Returns
    -------
    sae : SAE
        The loaded SAE object, already moved to ``device``.
    cfg_dict : dict
        Configuration dictionary returned by SAE Lens describing the SAE
        architecture and training hyperparameters.
    sparsity : torch.Tensor
        Per-feature sparsity statistics (log frequency of activation) as
        returned by SAE Lens.

    Raises
    ------
    ImportError
        If ``sae_lens`` is not installed in the current environment.
    ValueError
        If ``release`` or ``sae_id`` is not found in the SAE Lens registry.

    Examples
    --------
    >>> sae, cfg, sparsity = load_sae_for_layer(
    ...     "gpt2-small-res-jb", "blocks.6.hook_resid_pre", device="cpu"
    ... )
    """
    from sae_lens import SAE

    sae, cfg_dict, sparsity = SAE.from_pretrained(release=release, sae_id=sae_id)
    sae = sae.to(device)
    return sae, cfg_dict, sparsity


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def extract_sae_features(
    model: Any,
    sae: Any,
    tokens: torch.Tensor,
    target_layer: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a forward pass and extract SAE feature activations for a layer.

    Performs a single forward pass through ``model`` using TransformerLens'
    activation caching API, pulls the residual stream at ``target_layer``, and
    encodes it through ``sae`` to obtain sparse feature activations.

    Parameters
    ----------
    model : HookedTransformer
        A TransformerLens ``HookedTransformer`` model in eval mode.
    sae : SAE
        Loaded SAE for ``target_layer``, already on the correct device.
    tokens : torch.Tensor
        Input token IDs, shape ``[batch, seq_len]``.
    target_layer : int
        Layer index to extract the residual stream from.  The hook name used
        internally is ``f"blocks.{target_layer}.hook_resid_post"``.
    device : str, optional
        Device string; default is ``"cuda"``.

    Returns
    -------
    sae_acts : torch.Tensor
        SAE feature activations, shape ``[batch, seq_len, sae_width]``.
    logits : torch.Tensor
        Full model output logits, shape ``[batch, seq_len, vocab_size]``.

    Notes
    -----
    The function runs entirely under ``torch.no_grad()`` and does not modify
    ``model`` or ``sae`` state.

    Examples
    --------
    >>> sae_acts, logits = extract_sae_features(model, sae, tokens, target_layer=6)
    >>> sae_acts.shape
    torch.Size([1, 12, 16384])
    """
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    tokens = tokens.to(device)

    with torch.no_grad():
        logits, cache = model.run_with_cache(
            tokens,
            names_filter=hook_name,
            device=device,
        )
        resid = cache[hook_name]          # [batch, seq_len, d_model]
        sae_acts = sae.encode(resid)      # [batch, seq_len, sae_width]

    return sae_acts, logits


# ---------------------------------------------------------------------------
# Feature inspection helpers
# ---------------------------------------------------------------------------


def get_active_feature_ids(
    sae_acts: torch.Tensor,
    threshold: float = 0.0,
    position: int = -1,
) -> list[int]:
    """Get indices of active features at a given sequence position.

    A feature is considered *active* when its activation is strictly greater
    than ``threshold``.

    Parameters
    ----------
    sae_acts : torch.Tensor
        SAE feature activations.  Accepted shapes:

        * ``[batch, seq_len, sae_width]`` — first batch item is used.
        * ``[seq_len, sae_width]`` — used directly.
    threshold : float, optional
        Activation threshold; default is 0.0.
    position : int, optional
        Sequence position to inspect.  Negative indices follow Python
        conventions (``-1`` = last token); default is ``-1``.

    Returns
    -------
    list of int
        Indices of active features sorted in ascending order.

    Examples
    --------
    >>> acts = torch.zeros(1, 5, 512)
    >>> acts[0, -1, 7] = 1.0
    >>> get_active_feature_ids(acts)
    [7]
    """
    # Handle batched and unbatched inputs uniformly.
    if sae_acts.dim() == 3:
        acts_at_pos = sae_acts[0, position, :]   # [sae_width]
    elif sae_acts.dim() == 2:
        acts_at_pos = sae_acts[position, :]       # [sae_width]
    else:
        raise ValueError(
            f"sae_acts must be 2-D or 3-D, got {sae_acts.dim()}-D tensor."
        )

    active_indices = (acts_at_pos > threshold).nonzero(as_tuple=True)[0]
    return sorted(active_indices.tolist())


def get_top_k_features(
    sae_acts: torch.Tensor,
    k: int = 20,
    position: int = -1,
) -> tuple[list[int], list[float]]:
    """Get the top-k most active features at a sequence position.

    Parameters
    ----------
    sae_acts : torch.Tensor
        SAE feature activations.  Accepted shapes:

        * ``[batch, seq_len, sae_width]`` — first batch item is used.
        * ``[seq_len, sae_width]`` — used directly.
    k : int, optional
        Number of top features to return; default is 20.  If ``k`` exceeds the
        number of features, all features are returned.
    position : int, optional
        Sequence position to inspect; default is ``-1`` (last token).

    Returns
    -------
    indices : list of int
        Feature indices sorted by activation magnitude in descending order.
    magnitudes : list of float
        Corresponding activation magnitudes.

    Examples
    --------
    >>> acts = torch.zeros(1, 5, 512)
    >>> acts[0, -1, 42] = 3.5
    >>> acts[0, -1, 7] = 1.2
    >>> indices, mags = get_top_k_features(acts, k=2)
    >>> indices
    [42, 7]
    """
    if sae_acts.dim() == 3:
        acts_at_pos = sae_acts[0, position, :]
    elif sae_acts.dim() == 2:
        acts_at_pos = sae_acts[position, :]
    else:
        raise ValueError(
            f"sae_acts must be 2-D or 3-D, got {sae_acts.dim()}-D tensor."
        )

    actual_k = min(k, acts_at_pos.numel())
    top_values, top_indices = torch.topk(acts_at_pos, actual_k)

    indices = top_indices.tolist()
    magnitudes = top_values.tolist()
    return indices, magnitudes


# ---------------------------------------------------------------------------
# Reconstruction error
# ---------------------------------------------------------------------------


def compute_reconstruction_error(
    model: Any,
    sae: Any,
    tokens: torch.Tensor,
    target_layer: int,
    device: str = "cuda",
) -> dict:
    """Compute SAE reconstruction error (FVU = Fraction of Variance Unexplained).

    Runs a forward pass to obtain the residual stream at ``target_layer``,
    encodes it with the SAE and decodes it back, then measures how well the
    reconstruction captures the original activations.

    Parameters
    ----------
    model : HookedTransformer
        A TransformerLens ``HookedTransformer`` model in eval mode.
    sae : SAE
        Loaded SAE for ``target_layer``.
    tokens : torch.Tensor
        Input token IDs, shape ``[batch, seq_len]``.
    target_layer : int
        Layer index to extract the residual stream from.
    device : str, optional
        Device string; default is ``"cuda"``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        * ``fvu`` (float): fraction of variance unexplained,
          ``MSE(resid, decoded) / Var(resid)``.  Lower is better.
        * ``mse`` (float): mean squared error between the original residual
          stream and its SAE reconstruction.
        * ``mean_l0`` (float): average number of active (> 0) features per
          token across the entire sequence.

    Notes
    -----
    FVU is defined as :math:`\\text{MSE} / \\text{Var}(\\text{resid})`.
    A FVU of 0 indicates perfect reconstruction; 1 means the SAE explains none
    of the variance.

    Examples
    --------
    >>> stats = compute_reconstruction_error(model, sae, tokens, target_layer=6)
    >>> print(f"FVU: {stats['fvu']:.4f}, mean L0: {stats['mean_l0']:.1f}")
    """
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    tokens = tokens.to(device)

    with torch.no_grad():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=hook_name,
            device=device,
        )
        resid = cache[hook_name].float()          # [batch, seq_len, d_model]

        sae_encoded = sae.encode(resid)            # [batch, seq_len, sae_width]
        sae_decoded = sae.decode(sae_encoded)      # [batch, seq_len, d_model]

        # MSE: average over all elements.
        mse = torch.mean((resid - sae_decoded) ** 2).item()

        # Variance of the original residual stream (scalar).
        var_resid = torch.var(resid).item()
        fvu = mse / var_resid if var_resid > 0.0 else 0.0

        # L0: number of features > 0 per token, averaged over batch and seq.
        mean_l0 = (sae_encoded > 0).float().sum(dim=-1).mean().item()

    return {
        "fvu": float(fvu),
        "mse": float(mse),
        "mean_l0": float(mean_l0),
    }
