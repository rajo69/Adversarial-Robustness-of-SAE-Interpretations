"""
attacks.py
==========

Adversarial attack routines targeting Sparse Autoencoder (SAE) feature
activations extracted from a TransformerLens model.

Three attack strategies are implemented:

1. ``pgd_attack_sae`` -- Projected Gradient Descent (PGD) in embedding space
   that maximises the L2 distance between clean and perturbed SAE feature
   activations while staying inside an L-inf ball of radius ``epsilon``.

2. ``output_preserving_attack`` -- Lagrangian-relaxation variant of PGD that
   jointly maximises SAE feature disruption and penalises deviations in the
   model's output distribution (KL divergence) beyond a tolerance ``delta_kl``.

3. ``random_perturbation_baseline`` -- Monte-Carlo baseline that evaluates
   how much random Gaussian noise (scaled to an L-inf budget) disrupts SAE
   activations, providing a lower bound for comparison with adversarial attacks.

All functions are designed to work with the TransformerLens API and accept
``device="cpu"`` for testing without a GPU.
"""

from typing import Any, Optional

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _jaccard(a: torch.Tensor, b: torch.Tensor, threshold: float = 0.0) -> float:
    """Compute Jaccard index between the *support* sets of two activation vectors.

    Parameters
    ----------
    a : torch.Tensor
        First activation tensor (any shape; will be flattened).
    b : torch.Tensor
        Second activation tensor (any shape; will be flattened).
    threshold : float, optional
        Activations above this value are considered active.  Default ``0.0``.

    Returns
    -------
    float
        Jaccard index in ``[0, 1]``.  Returns ``0.0`` if both supports are
        empty.
    """
    a_bool = (a.detach().float() > threshold).flatten()
    b_bool = (b.detach().float() > threshold).flatten()
    intersection = (a_bool & b_bool).sum().item()
    union = (a_bool | b_bool).sum().item()
    if union == 0:
        return 1.0  # both sets empty → identical by convention (matches metrics.py)
    return float(intersection / union)


def _feature_flip_count(a: torch.Tensor, b: torch.Tensor, threshold: float = 0.0) -> int:
    """Count features that changed active/inactive status between ``a`` and ``b``.

    Parameters
    ----------
    a : torch.Tensor
        Clean activation tensor (any shape; will be flattened).
    b : torch.Tensor
        Perturbed activation tensor (any shape; will be flattened).
    threshold : float, optional
        Threshold above which a feature is considered active.  Default ``0.0``.

    Returns
    -------
    int
        Number of features whose active/inactive status differs.
    """
    a_bool = (a.detach().float() > threshold).flatten()
    b_bool = (b.detach().float() > threshold).flatten()
    return int((a_bool != b_bool).sum().item())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pgd_attack_sae(
    model: Any,
    sae: Any,
    clean_tokens: torch.Tensor,
    target_layer: int,
    epsilon: float,
    steps: int = 20,
    alpha: Optional[float] = None,
    device: str = "cuda",
) -> torch.Tensor:
    """PGD attack in embedding space that maximises disruption of SAE features.

    Runs Projected Gradient Descent (PGD) to find a perturbation ``delta``
    satisfying ``||delta||_inf <= epsilon`` such that the cosine similarity
    between the SAE feature activations computed on clean embeddings and
    perturbed embeddings is minimised (i.e. the features are maximally
    disrupted).

    The forward pass uses the TransformerLens hook ``"hook_embed"`` to inject
    the perturbed embedding directly, bypassing the model's own embedding
    look-up.

    Parameters
    ----------
    model : Any
        A TransformerLens ``HookedTransformer`` instance (or compatible mock).
        Must expose:
        - ``model.embed(tokens)`` -- returns ``[batch, seq, d_model]``
        - ``model.run_with_cache(tokens, names_filter=..., fwd_hooks=...)``
          returning ``(logits, cache_dict)``.
    sae : Any
        A SAE object with an ``encode(residual_stream)`` method that returns
        sparse feature activations of shape ``[batch, seq, sae_width]``.
    clean_tokens : torch.Tensor
        Integer token ids of shape ``[batch, seq_len]``.
    target_layer : int
        Transformer layer index whose residual stream post-activations are
        passed through the SAE.
    epsilon : float
        L-inf radius of the perturbation ball (in embedding space).
    steps : int, optional
        Number of PGD iterations.  Default ``20``.
    alpha : float, optional
        PGD step size.  Defaults to ``epsilon / 4`` if not provided.
    device : str, optional
        Device string (``"cuda"`` or ``"cpu"``).  Default ``"cuda"``.

    Returns
    -------
    torch.Tensor
        Perturbed embeddings tensor of the same shape as the clean embeddings,
        i.e. ``[batch, seq_len, d_model]``.  Returned detached from the
        computational graph.

    Notes
    -----
    - The returned tensor lives in embedding space, *not* token-id space.  To
      continue a forward pass you must inject it via a hook.
    - Loss used internally: ``-F.cosine_similarity(acts_clean_flat,
      acts_perturbed_flat, dim=-1).mean()``.
    - The L-inf projection is applied after every gradient step.
    """
    if alpha is None:
        alpha = epsilon / 4.0

    clean_tokens = clean_tokens.to(device)

    # --- Obtain clean embeddings (no grad) ---
    with torch.no_grad():
        clean_embeds: torch.Tensor = model.embed(clean_tokens)  # type: ignore[attr-defined]
        clean_embeds = clean_embeds.to(device)

    # --- Obtain clean SAE activations (no grad) ---
    resid_name = f"blocks.{target_layer}.hook_resid_post"

    with torch.no_grad():
        def _clean_hook(value: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ANN001
            return clean_embeds

        _, clean_cache = model.run_with_cache(  # type: ignore[attr-defined]
            clean_tokens,
            names_filter=resid_name,
            fwd_hooks=[("hook_embed", _clean_hook)],
        )
        clean_resid: torch.Tensor = clean_cache[resid_name].to(device)
        clean_acts: torch.Tensor = sae.encode(clean_resid).detach()  # type: ignore[attr-defined]

    # Flatten for cosine similarity: [batch * seq * sae_width] -> [1, N]
    clean_acts_flat = clean_acts.reshape(1, -1)

    # --- Initialise delta at zero ---
    delta = torch.zeros_like(clean_embeds, device=device)

    for _step in range(steps):
        # Build perturbed embeddings; attach gradient to the *combined* tensor
        perturbed = (clean_embeds + delta).detach().requires_grad_(True)

        # Define hook that replaces the embedding layer output
        def hook_fn(value: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ANN001
            return perturbed

        _, cache = model.run_with_cache(  # type: ignore[attr-defined]
            clean_tokens,
            names_filter=resid_name,
            fwd_hooks=[("hook_embed", hook_fn)],
        )
        resid: torch.Tensor = cache[resid_name]

        # SAE encoding of perturbed residual
        acts_perturbed = sae.encode(resid)  # type: ignore[attr-defined]
        acts_perturbed_flat = acts_perturbed.reshape(1, -1)

        # Loss: negative cosine similarity (we *maximise* feature distance)
        loss = -F.cosine_similarity(clean_acts_flat, acts_perturbed_flat, dim=-1).mean()

        loss.backward()

        with torch.no_grad():
            grad = perturbed.grad  # type: ignore[union-attr]
            if grad is None:
                # No gradient signal; skip update
                break

            # Gradient sign update on delta
            delta = delta + alpha * grad.sign()

            # L-inf projection onto epsilon-ball
            delta = delta.clamp(-epsilon, epsilon)

        # Manually zero gradient to avoid accumulation
        perturbed.grad = None

    perturbed_embeds = (clean_embeds + delta).detach()
    return perturbed_embeds


def output_preserving_attack(
    model: Any,
    sae: Any,
    clean_tokens: torch.Tensor,
    target_layer: int,
    epsilon: float,
    delta_kl: float = 0.1,
    steps: int = 40,
    alpha: Optional[float] = None,
    lambda_init: float = 1.0,
    lambda_lr: float = 0.1,
    device: str = "cuda",
) -> tuple[torch.Tensor, dict]:
    """Output-preserving adversarial attack via Lagrangian relaxation.

    Maximises the disruption of SAE feature activations subject to two
    constraints:

    1. The KL divergence between clean and perturbed *output* distributions
       (evaluated at the last token position) must not exceed ``delta_kl``.
    2. The perturbation must lie within an L-inf ball of radius ``epsilon``.

    The second constraint is enforced by projection; the first is relaxed via
    an adaptive Lagrange multiplier ``lambda``.

    Objective (minimised)::

        L = -cosine_sim(SAE_clean, SAE_perturbed)
            + lambda * max(0, KL(p_clean || p_perturbed) - delta_kl)

    The multiplier is updated after each step::

        lambda = max(0, lambda + lambda_lr * (KL - delta_kl))

    Parameters
    ----------
    model : Any
        A TransformerLens ``HookedTransformer`` instance (or compatible mock).
        Must support ``model.embed`` and ``model.run_with_cache`` with
        ``fwd_hooks``.
    sae : Any
        SAE object with ``encode(residual_stream)`` method.
    clean_tokens : torch.Tensor
        Integer token ids of shape ``[batch, seq_len]``.
    target_layer : int
        Transformer layer whose residual stream is fed to the SAE.
    epsilon : float
        L-inf radius of the allowed perturbation ball.
    delta_kl : float, optional
        KL divergence tolerance between clean and perturbed output
        distributions.  Default ``0.1``.
    steps : int, optional
        Number of optimisation iterations.  Default ``40``.
    alpha : float, optional
        Step size for the gradient update.  Defaults to ``epsilon / 4``.
    lambda_init : float, optional
        Initial value of the Lagrange multiplier.  Default ``1.0``.
    lambda_lr : float, optional
        Learning rate for the multiplier update.  Default ``0.1``.
    device : str, optional
        Device string.  Default ``"cuda"``.

    Returns
    -------
    perturbed_embeds : torch.Tensor
        Perturbed embeddings of shape ``[batch, seq_len, d_model]``, detached.
    info_dict : dict
        Diagnostic information with keys:

        - ``"final_kl"`` (*float*) -- KL divergence at the final iteration.
        - ``"final_sae_cosine"`` (*float*) -- Cosine similarity of SAE
          activations at the final iteration (lower = more disrupted).
        - ``"lambda_history"`` (*list of float*) -- Lagrange multiplier value
          recorded after each step.
        - ``"kl_history"`` (*list of float*) -- KL divergence recorded after
          each step.

    Notes
    -----
    - KL divergence is computed as ``F.kl_div(log_q, p, reduction='batchmean')``
      where ``p`` is the clean softmax distribution and ``q`` is the perturbed
      softmax distribution (evaluated at the last token position).
    - The Lagrange multiplier is clipped to be non-negative.
    """
    if alpha is None:
        alpha = epsilon / 4.0

    clean_tokens = clean_tokens.to(device)
    resid_name = f"blocks.{target_layer}.hook_resid_post"

    # --- Clean baseline (no grad) ---
    with torch.no_grad():
        clean_embeds: torch.Tensor = model.embed(clean_tokens)  # type: ignore[attr-defined]
        clean_embeds = clean_embeds.to(device)

        def _clean_hook(value: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ANN001
            return clean_embeds

        clean_logits, clean_cache = model.run_with_cache(  # type: ignore[attr-defined]
            clean_tokens,
            names_filter=resid_name,
            fwd_hooks=[("hook_embed", _clean_hook)],
        )
        clean_logits = clean_logits.to(device)
        clean_resid: torch.Tensor = clean_cache[resid_name].to(device)
        clean_acts: torch.Tensor = sae.encode(clean_resid).detach()  # type: ignore[attr-defined]

    clean_acts_flat = clean_acts.reshape(1, -1)
    # Softmax of clean logits at last token position
    p_clean = F.softmax(clean_logits[:, -1, :], dim=-1).detach()

    # --- Lagrange multiplier initialisation ---
    lam = float(lambda_init)

    # --- Tracking ---
    lambda_history: list[float] = []
    kl_history: list[float] = []

    # --- Initialise perturbation ---
    delta = torch.zeros_like(clean_embeds, device=device)

    final_kl = 0.0
    final_sae_cosine = 1.0

    for _step in range(steps):
        perturbed = (clean_embeds + delta).detach().requires_grad_(True)

        def hook_fn(value: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ANN001
            return perturbed

        perturbed_logits, pert_cache = model.run_with_cache(  # type: ignore[attr-defined]
            clean_tokens,
            names_filter=resid_name,
            fwd_hooks=[("hook_embed", hook_fn)],
        )
        pert_resid: torch.Tensor = pert_cache[resid_name]
        acts_perturbed = sae.encode(pert_resid)  # type: ignore[attr-defined]
        acts_perturbed_flat = acts_perturbed.reshape(1, -1)

        # Cosine similarity between SAE activations (higher = less disrupted)
        sae_cosine = F.cosine_similarity(clean_acts_flat, acts_perturbed_flat, dim=-1).mean()

        # KL divergence: KL(p_clean || p_perturbed)
        q_pert = F.softmax(perturbed_logits[:, -1, :], dim=-1)
        kl = F.kl_div(q_pert.log(), p_clean, reduction="batchmean")

        # Lagrangian loss (minimise)
        constraint_violation = kl - delta_kl
        loss = -sae_cosine + lam * torch.clamp(constraint_violation, min=0.0)

        loss.backward()

        with torch.no_grad():
            grad = perturbed.grad  # type: ignore[union-attr]
            if grad is None:
                break

            delta = delta + alpha * grad.sign()
            delta = delta.clamp(-epsilon, epsilon)

            # Record diagnostics before multiplier update
            kl_val = float(kl.item())
            sae_cos_val = float(sae_cosine.item())
            final_kl = kl_val
            final_sae_cosine = sae_cos_val

            # Adaptive multiplier update
            lam = max(0.0, lam + lambda_lr * (kl_val - delta_kl))

            lambda_history.append(lam)
            kl_history.append(kl_val)

        perturbed.grad = None

    perturbed_embeds = (clean_embeds + delta).detach()

    info_dict: dict = {
        "final_kl": final_kl,
        "final_sae_cosine": final_sae_cosine,
        "lambda_history": lambda_history,
        "kl_history": kl_history,
    }

    return perturbed_embeds, info_dict


def random_perturbation_baseline(
    model: Any,
    sae: Any,
    clean_tokens: torch.Tensor,
    target_layer: int,
    epsilon: float,
    n_samples: int = 10,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """Monte-Carlo baseline: evaluate random L-inf-bounded perturbations.

    For each of ``n_samples`` random perturbations, samples Gaussian noise,
    rescales it to have L-inf norm exactly ``epsilon``, adds it to the clean
    embeddings, and measures how much the SAE feature activations change.

    This provides a *lower bound* for adversarial attacks: an attack that
    performs only as well as random noise is not exploiting the loss landscape.

    Parameters
    ----------
    model : Any
        A TransformerLens ``HookedTransformer`` instance (or compatible mock).
        Must expose ``model.embed`` and ``model.run_with_cache`` with
        ``fwd_hooks``.
    sae : Any
        SAE object with ``encode(residual_stream)`` method.
    clean_tokens : torch.Tensor
        Integer token ids of shape ``[batch, seq_len]``.
    target_layer : int
        Transformer layer whose residual stream is fed to the SAE.
    epsilon : float
        L-inf budget.  Noise is scaled so that
        ``||perturbation||_inf == epsilon``.
    n_samples : int, optional
        Number of random perturbations to evaluate.  Default ``10``.
    seed : int, optional
        Random seed for reproducibility.  Default ``42``.
    device : str, optional
        Device string.  Default ``"cuda"``.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"mean_cosine"`` (*float*) -- Mean cosine similarity between clean
          and perturbed SAE activations across samples.
        - ``"std_cosine"`` (*float*) -- Standard deviation of cosine similarity.
        - ``"mean_jaccard"`` (*float*) -- Mean Jaccard index of the active-
          feature support sets.
        - ``"std_jaccard"`` (*float*) -- Standard deviation of Jaccard index.
        - ``"mean_flip_count"`` (*float*) -- Mean number of features that
          changed active/inactive status.
        - ``"std_flip_count"`` (*float*) -- Standard deviation of flip count.

    Notes
    -----
    - Noise is scaled by dividing by its own L-inf norm and multiplying by
      ``epsilon``, so every sample uses exactly the full perturbation budget.
    - Jaccard and flip count use a threshold of ``0.0`` (features are
      considered active if their activation is strictly positive).
    - All computation runs inside ``torch.no_grad()`` -- no gradients are
      tracked.
    """
    torch.manual_seed(seed)
    clean_tokens = clean_tokens.to(device)
    resid_name = f"blocks.{target_layer}.hook_resid_post"

    cosine_scores: list[float] = []
    jaccard_scores: list[float] = []
    flip_counts: list[float] = []

    with torch.no_grad():
        # Clean embeddings and activations
        clean_embeds: torch.Tensor = model.embed(clean_tokens)  # type: ignore[attr-defined]
        clean_embeds = clean_embeds.to(device)

        def _clean_hook(value: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ANN001
            return clean_embeds

        _, clean_cache = model.run_with_cache(  # type: ignore[attr-defined]
            clean_tokens,
            names_filter=resid_name,
            fwd_hooks=[("hook_embed", _clean_hook)],
        )
        clean_resid: torch.Tensor = clean_cache[resid_name].to(device)
        clean_acts: torch.Tensor = sae.encode(clean_resid)  # type: ignore[attr-defined]
        clean_acts_flat = clean_acts.reshape(1, -1)

        for _i in range(n_samples):
            # Sample Gaussian noise and scale to L-inf norm == epsilon
            noise = torch.randn_like(clean_embeds)
            linf_norm = noise.abs().max()
            if linf_norm < 1e-12:
                # Degenerate sample: skip with zero perturbation
                noise = torch.zeros_like(clean_embeds)
            else:
                noise = noise / linf_norm * epsilon

            perturbed_embeds = clean_embeds + noise

            def _pert_hook(value: torch.Tensor, hook: Any) -> torch.Tensor:  # noqa: ANN001
                return perturbed_embeds

            _, pert_cache = model.run_with_cache(  # type: ignore[attr-defined]
                clean_tokens,
                names_filter=resid_name,
                fwd_hooks=[("hook_embed", _pert_hook)],
            )
            pert_resid: torch.Tensor = pert_cache[resid_name].to(device)
            pert_acts: torch.Tensor = sae.encode(pert_resid)  # type: ignore[attr-defined]
            pert_acts_flat = pert_acts.reshape(1, -1)

            # Cosine similarity
            cos_sim = float(
                F.cosine_similarity(clean_acts_flat, pert_acts_flat, dim=-1).mean().item()
            )
            cosine_scores.append(cos_sim)

            # Jaccard and flip count
            jaccard = _jaccard(clean_acts, pert_acts)
            jaccard_scores.append(jaccard)

            flips = float(_feature_flip_count(clean_acts, pert_acts))
            flip_counts.append(flips)

    cos_t = torch.tensor(cosine_scores, dtype=torch.float32)
    jac_t = torch.tensor(jaccard_scores, dtype=torch.float32)
    flip_t = torch.tensor(flip_counts, dtype=torch.float32)

    return {
        "mean_cosine": float(cos_t.mean().item()),
        "std_cosine": float(cos_t.std(correction=1).item() if len(cos_t) > 1 else 0.0),
        "mean_jaccard": float(jac_t.mean().item()),
        "std_jaccard": float(jac_t.std(correction=1).item() if len(jac_t) > 1 else 0.0),
        "mean_flip_count": float(flip_t.mean().item()),
        "std_flip_count": float(flip_t.std(correction=1).item() if len(flip_t) > 1 else 0.0),
    }
