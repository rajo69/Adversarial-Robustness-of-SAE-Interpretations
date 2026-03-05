"""
test_metrics.py
---------------
Unit tests for src/metrics.py.

All tests run on CPU with deterministic inputs so they are fast, hermetic, and
require no GPU or external dependencies.  The fixed random seed ensures
reproducibility across environments.
"""

import pytest
import torch

torch.manual_seed(42)

from src.metrics import (
    compute_all_metrics,
    cosine_similarity_flat,
    feature_flip_count,
    jaccard_active_features,
    kl_divergence_logits,
    rank_correlation_topk,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def nonzero_vector() -> torch.Tensor:
    """A simple 1-D nonzero vector with known values."""
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.fixture()
def random_vector() -> torch.Tensor:
    """A random 1-D vector of length 100."""
    return torch.rand(100)


@pytest.fixture()
def orthogonal_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Two orthogonal 1-D vectors."""
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    return a, b


@pytest.fixture()
def disjoint_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Two vectors whose active features are completely disjoint.

    ``a`` has features 0 and 1 active; ``b`` has features 2 and 3 active.
    """
    a = torch.tensor([1.0, 1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 0.0, 1.0, 1.0])
    return a, b


@pytest.fixture()
def all_zeros() -> torch.Tensor:
    """An all-zero tensor (no active features)."""
    return torch.zeros(8)


@pytest.fixture()
def all_active() -> torch.Tensor:
    """A tensor where all features are active (values = 1.0)."""
    return torch.ones(8)


@pytest.fixture()
def all_inactive() -> torch.Tensor:
    """A tensor where all features are inactive (values = 0.0)."""
    return torch.zeros(8)


@pytest.fixture()
def random_logits_pair() -> tuple[torch.Tensor, torch.Tensor]:
    """Two different 3-D logit tensors [1, seq_len, vocab_size]."""
    torch.manual_seed(0)
    a = torch.randn(1, 5, 50)
    b = torch.randn(1, 5, 50)
    return a, b


@pytest.fixture()
def sae_acts_and_logits() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed SAE activations and logits for compute_all_metrics tests."""
    torch.manual_seed(42)
    clean_acts = torch.rand(1, 8, 64)
    perturbed_acts = torch.rand(1, 8, 64)
    clean_logits = torch.randn(1, 8, 100)
    perturbed_logits = torch.randn(1, 8, 100)
    return clean_acts, perturbed_acts, clean_logits, perturbed_logits


# ---------------------------------------------------------------------------
# cosine_similarity_flat
# ---------------------------------------------------------------------------


class TestCosineSimilarityFlat:
    def test_cosine_similarity_identical(self, nonzero_vector):
        """cosine_similarity_flat(a, a) should be exactly 1.0."""
        result = cosine_similarity_flat(nonzero_vector, nonzero_vector)
        assert abs(result - 1.0) < 1e-5, f"Expected 1.0, got {result}"

    def test_cosine_similarity_opposite(self, nonzero_vector):
        """cosine_similarity_flat(a, -a) should be approximately -1.0."""
        result = cosine_similarity_flat(nonzero_vector, -nonzero_vector)
        assert abs(result - (-1.0)) < 1e-5, f"Expected -1.0, got {result}"

    def test_cosine_similarity_orthogonal(self, orthogonal_pair):
        """Orthogonal vectors should have cosine similarity approximately 0.0."""
        a, b = orthogonal_pair
        result = cosine_similarity_flat(a, b)
        assert abs(result) < 1e-5, f"Expected ~0.0 for orthogonal vectors, got {result}"


# ---------------------------------------------------------------------------
# jaccard_active_features
# ---------------------------------------------------------------------------


class TestJaccardActiveFeatures:
    def test_jaccard_identical(self, nonzero_vector):
        """Jaccard of a tensor with itself should be exactly 1.0."""
        result = jaccard_active_features(nonzero_vector, nonzero_vector)
        assert result == 1.0, f"Expected 1.0, got {result}"

    def test_jaccard_disjoint(self, disjoint_pair):
        """Jaccard of two tensors with no overlapping active features should be 0.0."""
        a, b = disjoint_pair
        result = jaccard_active_features(a, b, threshold=0.0)
        assert result == 0.0, f"Expected 0.0 for disjoint active sets, got {result}"

    def test_jaccard_empty_both(self, all_zeros):
        """Jaccard of two all-zero tensors should be 1.0 (both sets are empty)."""
        result = jaccard_active_features(all_zeros, all_zeros)
        assert result == 1.0, f"Expected 1.0 for two empty feature sets, got {result}"


# ---------------------------------------------------------------------------
# rank_correlation_topk
# ---------------------------------------------------------------------------


class TestRankCorrelationTopk:
    def test_rank_correlation_identical(self, random_vector):
        """Rank correlation of a tensor with itself should be approximately 1.0."""
        result = rank_correlation_topk(random_vector, random_vector, k=10)
        assert abs(result - 1.0) < 1e-5, f"Expected ~1.0, got {result}"

    def test_rank_correlation_min_k(self):
        """rank_correlation_topk should work when the tensor has fewer than k elements."""
        # Only 3 elements but k=50; function should not crash.
        a = torch.tensor([3.0, 1.0, 2.0])
        result = rank_correlation_topk(a, a, k=50)
        assert abs(result - 1.0) < 1e-5, f"Expected ~1.0 for identical small tensor, got {result}"


# ---------------------------------------------------------------------------
# kl_divergence_logits
# ---------------------------------------------------------------------------


class TestKLDivergenceLogits:
    def test_kl_divergence_identical(self):
        """KL divergence of identical logits should be approximately 0.0."""
        torch.manual_seed(7)
        logits = torch.randn(1, 5, 50)
        result = kl_divergence_logits(logits, logits)
        assert abs(result) < 1e-4, f"Expected ~0.0 for identical logits, got {result}"

    def test_kl_divergence_positive(self, random_logits_pair):
        """KL divergence should be non-negative for any pair of distributions."""
        a, b = random_logits_pair
        result = kl_divergence_logits(a, b)
        assert result >= 0.0, f"KL divergence must be >= 0, got {result}"


# ---------------------------------------------------------------------------
# feature_flip_count
# ---------------------------------------------------------------------------


class TestFeatureFlipCount:
    def test_feature_flip_count_no_change(self, random_vector):
        """Flip count of a tensor with itself should be 0."""
        result = feature_flip_count(random_vector, random_vector)
        assert result == 0, f"Expected 0 flips, got {result}"

    def test_feature_flip_count_all_flipped(self, all_active, all_inactive):
        """All-active vs all-inactive: every feature should flip."""
        n_features = all_active.numel()
        result = feature_flip_count(all_active, all_inactive, threshold=0.0)
        assert result == n_features, (
            f"Expected {n_features} flips, got {result}"
        )


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------


class TestComputeAllMetrics:
    EXPECTED_KEYS = {
        "cosine_similarity",
        "jaccard_index",
        "rank_correlation",
        "kl_divergence",
        "feature_flip_count",
        "sae_disruption",
    }

    def test_compute_all_metrics_keys(self, sae_acts_and_logits):
        """compute_all_metrics should return a dict with exactly the 6 expected keys."""
        clean_acts, perturbed_acts, clean_logits, perturbed_logits = sae_acts_and_logits
        result = compute_all_metrics(
            clean_acts, perturbed_acts, clean_logits, perturbed_logits
        )
        assert isinstance(result, dict), "Return value must be a dict."
        assert set(result.keys()) == self.EXPECTED_KEYS, (
            f"Unexpected keys. Got: {set(result.keys())}"
        )

    def test_compute_all_metrics_types(self, sae_acts_and_logits):
        """All values in the returned dict should be float or int."""
        clean_acts, perturbed_acts, clean_logits, perturbed_logits = sae_acts_and_logits
        result = compute_all_metrics(
            clean_acts, perturbed_acts, clean_logits, perturbed_logits
        )
        for key, value in result.items():
            assert isinstance(value, (float, int)), (
                f"Value for key '{key}' has unexpected type {type(value).__name__}."
            )

    def test_compute_all_metrics_sae_disruption_definition(self, sae_acts_and_logits):
        """sae_disruption must equal 1.0 - cosine_similarity."""
        clean_acts, perturbed_acts, clean_logits, perturbed_logits = sae_acts_and_logits
        result = compute_all_metrics(
            clean_acts, perturbed_acts, clean_logits, perturbed_logits
        )
        expected = 1.0 - result["cosine_similarity"]
        assert abs(result["sae_disruption"] - expected) < 1e-6, (
            f"sae_disruption ({result['sae_disruption']}) != "
            f"1 - cosine_similarity ({expected})."
        )

    def test_compute_all_metrics_identical_inputs(self):
        """With identical clean/perturbed tensors, disruption should be 0 and jaccard 1."""
        torch.manual_seed(42)
        acts = torch.rand(1, 8, 64)
        logits = torch.randn(1, 8, 100)
        result = compute_all_metrics(acts, acts, logits, logits)

        assert abs(result["sae_disruption"]) < 1e-5, (
            f"Expected sae_disruption~0 for identical inputs, got {result['sae_disruption']}."
        )
        assert result["jaccard_index"] == 1.0, (
            f"Expected jaccard=1.0 for identical inputs, got {result['jaccard_index']}."
        )
        assert result["feature_flip_count"] == 0, (
            f"Expected 0 flips for identical inputs, got {result['feature_flip_count']}."
        )
