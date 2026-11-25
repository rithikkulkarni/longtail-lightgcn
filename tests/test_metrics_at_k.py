"""Tests for metrics_at_k.py: the first-K-columns trick must equal asking for the top-K list directly."""
import numpy as np
import pytest
import torch

from longtail_lightgcn.metrics import all_metrics, recommend_top_k, target_mask
from longtail_lightgcn.metrics_at_k import check_against_stored, metrics_at_ks

N_USERS, N_ITEMS = 40, 120


@pytest.fixture
def setup():
    g = torch.Generator().manual_seed(0)
    users = torch.randn(N_USERS, 8, generator=g)
    items = torch.randn(N_ITEMS, 8, generator=g)
    rng = np.random.default_rng(0)
    train = (rng.integers(0, N_USERS, 300), rng.integers(0, N_ITEMS, 300))
    test = (rng.integers(0, N_USERS, 200), rng.integers(0, N_ITEMS, 200))
    target = target_mask([test], N_USERS, N_ITEMS, "cpu")
    degree = torch.bincount(torch.as_tensor(train[1]), minlength=N_ITEMS)
    return users, items, train, target, degree


def test_first_k_columns_equal_the_top_k_list(setup):
    users, items, train, _, _ = setup
    top20 = recommend_top_k(users, items, 20, [train])
    for k in (5, 10):
        assert torch.equal(top20[:, :k], recommend_top_k(users, items, k, [train]))


def test_metrics_from_a_prefix_equal_metrics_from_a_direct_top_k(setup):
    users, items, train, target, degree = setup
    top20 = recommend_top_k(users, items, 20, [train])
    by_k = metrics_at_ks(top20, target, degree, N_ITEMS)
    for k in (5, 10, 20):
        direct = all_metrics(recommend_top_k(users, items, k, [train]), target, degree, N_ITEMS)
        assert by_k[k]["recall"] == direct[f"recall@{k}"]
        assert by_k[k]["ndcg"] == direct[f"ndcg@{k}"]
        for name in ("coverage", "gini", "arp", "target_pop"):
            assert by_k[k][name] == direct[name]


def test_recall_and_coverage_never_shrink_as_k_grows(setup):
    users, items, train, target, degree = setup
    by_k = metrics_at_ks(recommend_top_k(users, items, 20, [train]), target, degree, N_ITEMS)
    assert by_k[5]["recall"] <= by_k[10]["recall"] <= by_k[20]["recall"]
    assert by_k[5]["coverage"] <= by_k[10]["coverage"] <= by_k[20]["coverage"]


def test_k_larger_than_the_saved_list_is_an_error(setup):
    users, items, train, target, degree = setup
    with pytest.raises(ValueError):
        metrics_at_ks(recommend_top_k(users, items, 20, [train]), target, degree, N_ITEMS, ks=(30,))


def test_check_against_stored_accepts_a_match_and_rejects_a_mismatch():
    computed = {"recall": 0.25, "ndcg": 0.3, "coverage": 0.6, "gini": 0.85, "arp": 1000.0, "target_pop": 900.0}
    stored = {"recall@20": 0.25, "ndcg@20": 0.3, "coverage": 0.6, "gini": 0.85, "arp": 1000.0, "target_pop": 900.0}
    assert check_against_stored(computed, stored, "k") == 0.0
    with pytest.raises(ValueError, match="recall@20"):
        check_against_stored({**computed, "recall": 0.26}, stored, "k")
