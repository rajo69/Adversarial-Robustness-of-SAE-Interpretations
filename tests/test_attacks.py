"""
test_attacks.py
===============

Pytest unit tests for ``src.attacks``.

All tests run on CPU (``device="cpu"``) so no GPU is required.  The model and
SAE are replaced by lightweight stubs that return real ``torch.Tensor`` objects
so that shape/bound checks are meaningful.  Numerical correctness of the
optimisation is *not* tested here -- only shapes, types, key presence, and
constraint satisfaction (epsilon ball).
"""

import pytest
import torch
import torch.nn.functional as F
from unittest.mock import MagicMock

from src.attacks import (
    output_preserving_attack,
    pgd_attack_sae,
    random_perturbation_baseline,
)

# ---------------------------------------------------------------------------
# Constants shared across tests
# ---------------------------------------------------------------------------

BATCH = 1
SEQ_LEN = 10
HIDDEN_DIM = 64
SAE_WIDTH = 256
VOCAB_SIZE = 100
TARGET_LAYER = 0
EPSILON = 0.05
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_tokens() -> torch.Tensor:
    """Random integer token ids of shape [1, SEQ_LEN]."""
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


@pytest.fixture()
def mock_model(clean_tokens: torch.Tensor) -> MagicMock:
    """Stub TransformerLens model.

    Returns deterministic real tensors so that gradient flow and shape checks
    work correctly.  The ``run_with_cache`` method signature deliberately
    accepts ``**kwargs`` so that ``names_filter`` and ``fwd_hooks`` are silently
    ignored by the mock infrastructure while still exercising the call path in
    ``src.attacks``.
    """
    model = MagicMock()

    # embed() always returns a fresh tensor of the expected shape
    def _embed(tokens: torch.Tensor) -> torch.Tensor:
        return torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)

    model.embed.side_effect = _embed

    # run_with_cache() calls the fwd_hooks to get the (potentially perturbed)
    # embedding, then builds a fake cache using that embedding so that the
    # residual stream is connected to the computational graph.
    resid_key = f"blocks.{TARGET_LAYER}.hook_resid_post"

    def _run_with_cache(tokens, *args, **kwargs):  # noqa: ANN001
        """Simulate a forward pass.

        If a ``fwd_hooks`` list is supplied, call the first hook with a dummy
        value tensor so that the ``perturbed`` closure used in attacks is
        executed and its gradients flow correctly.
        """
        fwd_hooks = kwargs.get("fwd_hooks", [])

        # Determine the embedding to use: if a hook is registered for
        # "hook_embed", call it to retrieve the (possibly perturbed) embedding.
        embed_output = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        for hook_name, hook_fn in fwd_hooks:
            if hook_name == "hook_embed":
                embed_output = hook_fn(embed_output, None)
                break

        # Build residual stream as a simple linear transform of the embedding
        # so that gradients can flow back through it.
        resid = embed_output * 1.0  # identity-ish, keeps grad

        # Logits: [batch, seq, vocab]
        logits = torch.randn(BATCH, SEQ_LEN, VOCAB_SIZE)

        cache = {resid_key: resid}
        return logits, cache

    model.run_with_cache.side_effect = _run_with_cache

    return model


@pytest.fixture()
def mock_sae() -> MagicMock:
    """Stub SAE whose ``encode`` returns a sparse-like tensor."""
    sae = MagicMock()

    def _encode(resid: torch.Tensor) -> torch.Tensor:
        # Use a deterministic linear projection so grad flows through encode
        weight = torch.ones(HIDDEN_DIM, SAE_WIDTH) * 0.01
        # resid: [batch, seq, hidden] -> [batch, seq, sae_width]
        acts = resid @ weight  # broadcasts over batch/seq
        return F.relu(acts)  # sparsify to mimic real SAE

    sae.encode.side_effect = _encode

    return sae


# ---------------------------------------------------------------------------
# Tests for pgd_attack_sae
# ---------------------------------------------------------------------------


class TestPgdAttackSae:
    """Tests for the PGD attack on SAE activations."""

    def test_pgd_output_shape(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """Returned tensor must have the same shape as clean embeddings."""
        clean_embeds = mock_model.embed(clean_tokens)
        result = pgd_attack_sae(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            steps=3,
            device=DEVICE,
        )
        assert result.shape == clean_embeds.shape, (
            f"Expected shape {clean_embeds.shape}, got {result.shape}"
        )

    def test_pgd_epsilon_ball_respected(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """Every element of (output - clean_embed) must be in [-epsilon, epsilon]."""
        # Reset embed side_effect to return a fixed tensor for comparison
        fixed_embeds = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        mock_model.embed.side_effect = lambda tokens: fixed_embeds.clone()

        result = pgd_attack_sae(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            steps=5,
            device=DEVICE,
        )

        delta = result - fixed_embeds
        max_abs = delta.abs().max().item()
        assert max_abs <= EPSILON + 1e-6, (
            f"L-inf constraint violated: max |delta| = {max_abs:.6f} > epsilon = {EPSILON}"
        )

    def test_pgd_returns_different_from_clean(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """The attack should move the embeddings (result != clean embeddings)."""
        fixed_embeds = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        mock_model.embed.side_effect = lambda tokens: fixed_embeds.clone()

        result = pgd_attack_sae(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            steps=5,
            device=DEVICE,
        )

        # The perturbation should be non-zero (attack has had some effect)
        delta_norm = (result - fixed_embeds).abs().max().item()
        assert delta_norm > 1e-9, (
            "Attack produced zero perturbation -- the attack had no effect."
        )


# ---------------------------------------------------------------------------
# Tests for random_perturbation_baseline
# ---------------------------------------------------------------------------


class TestRandomPerturbationBaseline:
    """Tests for the random-noise baseline."""

    EXPECTED_KEYS = {
        "mean_cosine",
        "std_cosine",
        "mean_jaccard",
        "std_jaccard",
        "mean_flip_count",
        "std_flip_count",
    }

    def test_random_baseline_keys(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """Returned dict must contain all expected keys."""
        result = random_perturbation_baseline(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            n_samples=3,
            device=DEVICE,
        )
        assert set(result.keys()) == self.EXPECTED_KEYS, (
            f"Missing or extra keys. Got: {set(result.keys())}"
        )

    def test_random_baseline_cosine_range(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """mean_cosine must be in the valid range [-1, 1]."""
        result = random_perturbation_baseline(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            n_samples=5,
            device=DEVICE,
        )
        mean_cos = result["mean_cosine"]
        assert -1.0 - 1e-6 <= mean_cos <= 1.0 + 1e-6, (
            f"mean_cosine = {mean_cos:.4f} is outside [-1, 1]"
        )

    def test_random_baseline_all_values_are_floats(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """All values in the returned dict must be Python floats."""
        result = random_perturbation_baseline(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            n_samples=3,
            device=DEVICE,
        )
        for key, val in result.items():
            assert isinstance(val, float), (
                f"Expected float for key '{key}', got {type(val)}"
            )


# ---------------------------------------------------------------------------
# Tests for output_preserving_attack
# ---------------------------------------------------------------------------


class TestOutputPreservingAttack:
    """Tests for the output-preserving Lagrangian attack."""

    def test_output_preserving_returns_tuple(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """Return value must be a 2-tuple of (Tensor, dict)."""
        result = output_preserving_attack(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            steps=3,
            device=DEVICE,
        )
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"
        perturbed_embeds, info_dict = result
        assert isinstance(perturbed_embeds, torch.Tensor), (
            f"Expected Tensor as first element, got {type(perturbed_embeds)}"
        )
        assert isinstance(info_dict, dict), (
            f"Expected dict as second element, got {type(info_dict)}"
        )

    def test_output_preserving_info_keys(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """info_dict must contain the four required diagnostic keys."""
        expected_keys = {"final_kl", "final_sae_cosine", "lambda_history", "kl_history"}
        _, info_dict = output_preserving_attack(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            steps=3,
            device=DEVICE,
        )
        assert set(info_dict.keys()) == expected_keys, (
            f"Missing or extra keys. Got: {set(info_dict.keys())}"
        )

    def test_output_preserving_output_shape(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """Perturbed embeddings must have the same shape as clean embeddings."""
        fixed_embeds = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        mock_model.embed.side_effect = lambda tokens: fixed_embeds.clone()

        perturbed_embeds, _ = output_preserving_attack(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            steps=3,
            device=DEVICE,
        )
        assert perturbed_embeds.shape == fixed_embeds.shape, (
            f"Expected shape {fixed_embeds.shape}, got {perturbed_embeds.shape}"
        )

    def test_output_preserving_epsilon_ball_respected(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """L-inf constraint must be satisfied on the returned embeddings."""
        fixed_embeds = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM)
        mock_model.embed.side_effect = lambda tokens: fixed_embeds.clone()

        perturbed_embeds, _ = output_preserving_attack(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            steps=5,
            device=DEVICE,
        )

        delta = perturbed_embeds - fixed_embeds
        max_abs = delta.abs().max().item()
        assert max_abs <= EPSILON + 1e-6, (
            f"L-inf constraint violated: max |delta| = {max_abs:.6f} > epsilon = {EPSILON}"
        )

    def test_output_preserving_history_lengths(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """lambda_history and kl_history must have length equal to steps."""
        n_steps = 5
        _, info_dict = output_preserving_attack(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            steps=n_steps,
            device=DEVICE,
        )
        assert len(info_dict["lambda_history"]) == n_steps, (
            f"lambda_history length {len(info_dict['lambda_history'])} != steps {n_steps}"
        )
        assert len(info_dict["kl_history"]) == n_steps, (
            f"kl_history length {len(info_dict['kl_history'])} != steps {n_steps}"
        )

    def test_output_preserving_final_kl_is_float(
        self,
        mock_model: MagicMock,
        mock_sae: MagicMock,
        clean_tokens: torch.Tensor,
    ) -> None:
        """final_kl and final_sae_cosine must be Python floats."""
        _, info_dict = output_preserving_attack(
            model=mock_model,
            sae=mock_sae,
            clean_tokens=clean_tokens,
            target_layer=TARGET_LAYER,
            epsilon=EPSILON,
            steps=3,
            device=DEVICE,
        )
        assert isinstance(info_dict["final_kl"], float), (
            f"Expected float for 'final_kl', got {type(info_dict['final_kl'])}"
        )
        assert isinstance(info_dict["final_sae_cosine"], float), (
            f"Expected float for 'final_sae_cosine', got {type(info_dict['final_sae_cosine'])}"
        )
