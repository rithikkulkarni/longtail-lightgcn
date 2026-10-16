from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from longtail_lightgcn.data.preprocess import (
    build_edges,
    degree_stats,
    holdout_size,
    load_movies,
    load_ratings,
    make_split,
    save_items_csv,
)


def toy_edges(n_users=30, n_items=50, seed=123):
    """Each user rates between 20 and 40 distinct items (like ML-1M's >=20 guarantee)."""
    rng = np.random.default_rng(seed)
    users, items = [], []
    for u in range(n_users):
        k = rng.integers(20, 41)
        picked = rng.choice(n_items, size=k, replace=False)
        users += [u] * k
        items += list(picked)
    return np.array(users), np.array(items)


def pair_set(users, items, n_items):
    return set((users * n_items + items).tolist())


def test_holdout_size_half_up():
    assert holdout_size(20) == 2
    assert holdout_size(24) == 2
    assert holdout_size(25) == 3  # 2.5 rounds UP (Python's round() would give 2)
    assert holdout_size(35) == 4
    assert holdout_size(29) == 3
    # agrees with floor(0.1*n + 0.5) wherever float arithmetic is safe
    for n in range(20, 3000):
        assert holdout_size(n) == int(np.floor(n / 10 + 0.5 + 1e-9))


def test_build_edges_remaps_and_keeps_originals():
    df = pd.DataFrame({"user_id": [10, 10, 30], "movie_id": [500, 7, 7]})
    users, items, user_ids, item_ids = build_edges(df)
    assert list(user_ids) == [10, 30] and list(item_ids) == [7, 500]
    assert list(users) == [0, 0, 1] and list(items) == [1, 0, 0]


def test_build_edges_rejects_duplicates():
    df = pd.DataFrame({"user_id": [1, 1], "movie_id": [2, 2]})
    with pytest.raises(ValueError):
        build_edges(df)


def test_split_partitions_all_edges():
    users, items = toy_edges()
    n_items = 50
    s = make_split(users, items, seed=0)
    train = pair_set(s["train_users"], s["train_items"], n_items)
    val = pair_set(s["val_users"], s["val_items"], n_items)
    test = pair_set(s["test_users"], s["test_items"], n_items)
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == pair_set(users, items, n_items)
    assert len(train) + len(val) + len(test) == len(users)


def test_split_per_user_sizes_and_coverage():
    users, items = toy_edges()
    s = make_split(users, items, seed=0)
    for u in np.unique(users):
        n = (users == u).sum()
        h = holdout_size(n)
        assert (s["val_users"] == u).sum() == h
        assert (s["test_users"] == u).sum() == h
        assert (s["train_users"] == u).sum() == n - 2 * h
        assert h >= 1  # every user appears in val and test


def test_split_reproducible_and_seed_dependent():
    users, items = toy_edges()
    a, b, c = (make_split(users, items, seed=s) for s in (0, 0, 1))
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not np.array_equal(a["test_items"], c["test_items"])


def test_degree_stats_counts_zero_train_items():
    # item 2 has a single edge; force it into test so train degree is 0
    split = {
        "train_items": np.array([0, 0, 1]),
        "val_items": np.array([1]),
        "test_items": np.array([2]),
    }
    items = np.array([0, 0, 1, 1, 2])
    st = degree_stats(items, 3, split)
    assert st["items_with_0_train_edges"] == 1
    assert st["items_with_1_rating_overall"] == 1


def test_items_csv_keeps_unrated_movies_with_flag(tmp_path):
    movies = pd.DataFrame(
        {"movie_id": [1, 2, 3], "title": ["A (1990)", "B (1991)", "C (1992)"], "genres": ["x", "y", "z"]}
    )
    save_items_csv(tmp_path / "items.csv", movies, item_ids=np.array([1, 3]))
    out = pd.read_csv(tmp_path / "items.csv")
    assert len(out) == 3
    assert out["in_catalog"].tolist() == [True, False, True]
    assert out["item_idx"].tolist()[0] == 0 and out["item_idx"].tolist()[2] == 1
    assert pd.isna(out["item_idx"].tolist()[1])


RAW = Path("data/raw/ml-1m")


@pytest.mark.skipif(not (RAW / "ratings.dat").exists(), reason="ML-1M not downloaded")
def test_real_data_counts():
    ratings = load_ratings(RAW)
    movies = load_movies(RAW)
    users, items, user_ids, item_ids = build_edges(ratings)
    assert len(user_ids) == 6040
    assert len(item_ids) == 3706
    assert len(users) == 1_000_209
    assert len(movies) == 3883
    assert users.max() == 6039 and items.max() == 3705
