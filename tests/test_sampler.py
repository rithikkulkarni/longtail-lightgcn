from pathlib import Path

import numpy as np
import pytest

from longtail_lightgcn.data.preprocess import load_split
from longtail_lightgcn.sampler import NegativeSampler


def test_never_returns_a_train_positive():
    rng = np.random.default_rng(0)
    n_users, n_items = 20, 40
    users, items = [], []
    for u in range(n_users):
        picked = rng.choice(n_items, size=rng.integers(5, 35), replace=False)  # some very dense users
        users += [u] * len(picked)
        items += list(picked)
    users, items = np.array(users), np.array(items)
    sampler = NegativeSampler(users, items, n_users, n_items, seed=1)

    train = set((users * n_items + items).tolist())
    q = np.repeat(np.arange(n_users), 500)
    neg = sampler.sample(q)
    assert len(neg) == len(q)
    assert not any(int(u) * n_items + int(j) in train for u, j in zip(q, neg))


def test_uniform_over_non_positives():
    # user 0 has items 0..4 in train out of 10, so negatives must be spread evenly over 5..9
    users = np.zeros(5, dtype=int)
    items = np.arange(5)
    sampler = NegativeSampler(users, items, n_users=1, n_items=10, seed=0)
    neg = sampler.sample(np.zeros(50_000, dtype=int))
    counts = np.bincount(neg, minlength=10)
    assert counts[:5].sum() == 0
    assert np.all(np.abs(counts[5:] - 10_000) < 500)  # ~5 sigma


def test_reproducible_with_seed():
    users, items = np.array([0, 0, 1]), np.array([0, 1, 2])
    a = NegativeSampler(users, items, 2, 6, seed=7).sample(np.array([0, 1] * 50))
    b = NegativeSampler(users, items, 2, 6, seed=7).sample(np.array([0, 1] * 50))
    c = NegativeSampler(users, items, 2, 6, seed=8).sample(np.array([0, 1] * 50))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_guard_raises_when_a_user_has_interacted_with_every_item():
    users = np.array([0, 0, 0, 1])
    items = np.array([0, 1, 2, 0])  # user 0 has all 3 items
    with pytest.raises(ValueError, match="every item"):
        NegativeSampler(users, items, n_users=2, n_items=3)


def test_user_with_one_free_item_still_works():
    users = np.array([0, 0])
    items = np.array([0, 1])  # 3 items, only item 2 is a valid negative
    neg = NegativeSampler(users, items, 1, 3, seed=0).sample(np.zeros(200, dtype=int))
    assert (neg == 2).all()


PROCESSED = Path("data/processed/split_0.npz")


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data not built")
def test_real_split_runs_and_is_valid():
    split, n_users, n_items = load_split(PROCESSED)
    sampler = NegativeSampler(split["train_users"], split["train_items"], n_users, n_items, seed=0)
    neg = sampler.sample(split["train_users"])
    train = set((split["train_users"] * n_items + split["train_items"]).tolist())
    assert len(neg) == len(split["train_users"])
    sample = np.random.default_rng(0).choice(len(neg), 20_000, replace=False)
    assert not any(int(split["train_users"][k]) * n_items + int(neg[k]) in train for k in sample)
