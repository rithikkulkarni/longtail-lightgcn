import math

import numpy as np
import pytest
import torch

from longtail_lightgcn import evaluate
from longtail_lightgcn.metrics import (
    average_recommendation_popularity,
    coverage,
    gini,
    item_exposure,
    ndcg_at_k,
    recall_at_k,
    recommend_top_k,
    target_mask,
    target_popularity,
)

# Toy problem: 3 users, 6 items, K = 3.
RECS = torch.tensor([[0, 1, 2], [1, 0, 4], [2, 3, 4]])
# user0 held-out {1, 4}, user1 held-out {0}, user2 has none
TARGET = target_mask([(np.array([0, 0, 1]), np.array([1, 4, 0]))], 3, 6)
DEGREE = torch.tensor([10, 0, 5, 2, 1, 7])


def test_hand_computed_recall():
    # user0: 1 hit (item 1) of 2 held-out = 0.5; user1: 1 of 1 = 1.0; user2 skipped
    assert math.isclose(recall_at_k(RECS, TARGET), 0.75)


def test_hand_computed_ndcg():
    # user0: hit at rank 2 -> dcg = 1/log2(3); ideal has 2 hits -> idcg = 1 + 1/log2(3)
    # user1: hit at rank 2 -> dcg = 1/log2(3); ideal has 1 hit  -> idcg = 1
    d = 1 / math.log2(3)
    expected = (d / (1 + d) + d / 1) / 2
    assert math.isclose(ndcg_at_k(RECS, TARGET), expected)


def test_perfect_ranking_with_more_targets_than_k():
    # 5 held-out items, list holds 3 of them in the top slots: NDCG = 1 (ideal is capped at K), recall = 3/5
    recs = torch.tensor([[0, 1, 2]])
    target = target_mask([(np.zeros(5, dtype=int), np.arange(5))], 1, 6)
    assert math.isclose(ndcg_at_k(recs, target), 1.0)
    assert math.isclose(recall_at_k(recs, target), 3 / 5)


def test_misses_score_zero_and_rank_matters():
    target = target_mask([(np.array([0]), np.array([5]))], 1, 6)
    assert recall_at_k(torch.tensor([[0, 1, 2]]), target) == 0.0
    assert ndcg_at_k(torch.tensor([[0, 1, 2]]), target) == 0.0
    first = ndcg_at_k(torch.tensor([[5, 0, 1]]), target)
    third = ndcg_at_k(torch.tensor([[0, 1, 5]]), target)
    assert math.isclose(first, 1.0) and math.isclose(third, 0.5)  # 1/log2(4) = 0.5


def test_users_without_targets_are_skipped():
    # adding a user with no held-out items must not change the averages
    recs = torch.cat([RECS, torch.tensor([[3, 4, 5]])])
    target = torch.cat([TARGET, torch.zeros(1, 6, dtype=torch.bool)])
    assert math.isclose(recall_at_k(recs, target), recall_at_k(RECS, TARGET))
    assert math.isclose(ndcg_at_k(recs, target), ndcg_at_k(RECS, TARGET))


def test_exposure_and_coverage_count_never_recommended_items():
    assert item_exposure(RECS, 6).tolist() == [2, 2, 2, 1, 2, 0]  # item 5 is never recommended
    assert math.isclose(coverage(RECS, 6), 5 / 6)
    assert math.isclose(coverage(RECS, 10), 5 / 10)  # denominator is the catalog, not the recommended set


def test_gini_uniform_is_zero():
    assert math.isclose(gini(torch.tensor([3, 3, 3, 3])), 0.0, abs_tol=1e-12)


def test_gini_single_item_gets_everything():
    # (n-1)/n, which approaches 1 as the catalog grows
    assert math.isclose(gini(torch.tensor([0, 0, 0, 10])), 3 / 4)
    assert gini(torch.cat([torch.zeros(3705), torch.tensor([20.0])])) > 0.999


def test_gini_hand_computed_and_order_invariant():
    # sorted [1,2,3]: (-2*1 + 0*2 + 2*3) / (3*6) = 4/18
    assert math.isclose(gini(torch.tensor([1, 2, 3])), 4 / 18)
    assert math.isclose(gini(torch.tensor([3, 1, 2])), 4 / 18)


def test_gini_counts_zero_exposure_items():
    # D-22: [0,0,1,1] is unequal (0.5); dropping the zeros would call [1,1] perfectly equal
    assert math.isclose(gini(torch.tensor([0, 0, 1, 1])), 0.5)


def test_gini_all_zero_raises():
    with pytest.raises(ValueError):
        gini(torch.zeros(5))


def test_hand_computed_arp():
    # per-user means: (10+0+5)/3 = 5, (0+10+1)/3 = 11/3, (5+2+1)/3 = 8/3
    assert math.isclose(average_recommendation_popularity(RECS, DEGREE), (5 + 11 / 3 + 8 / 3) / 3)


def test_target_popularity_is_per_user_then_averaged():
    # user0: (0+1)/2 = 0.5; user1: 10; user2 skipped -> 5.25 (a pooled mean would be 11/3)
    assert math.isclose(target_popularity(TARGET, DEGREE), 5.25)


def test_recommend_top_k_order_and_masking():
    # same toy as test_evaluate: user0 and user2 prefer low item indices, user1 prefers high ones
    items = torch.tensor([[5.0], [4.0], [3.0], [2.0], [1.0], [0.0]])
    users = torch.tensor([[1.0], [-1.0], [1.0]])
    exclude = [(np.array([0]), np.array([0]))]  # user0 has item 0 in train
    recs = recommend_top_k(users, items, 2, exclude)
    assert recs.tolist() == [[1, 2], [5, 4], [0, 1]]
    assert torch.equal(recs, recommend_top_k(users, items, 2, exclude, user_batch=1))


def test_matches_existing_recall_at_k():
    # the new pipeline must agree with the evaluate.py implementation that training uses
    g = torch.Generator().manual_seed(0)
    users, items = torch.randn(40, 8, generator=g), torch.randn(60, 8, generator=g)
    train = (np.repeat(np.arange(40), 3), np.arange(120) % 60)
    val = (np.repeat(np.arange(40), 2), (np.arange(80) * 7 + 1) % 60)
    recs = recommend_top_k(users, items, 10, [train])
    new = recall_at_k(recs, target_mask([val], 40, 60))
    old = evaluate.recall_at_k(users, items, 10, [train], [val])
    assert math.isclose(new, old, rel_tol=1e-6)
