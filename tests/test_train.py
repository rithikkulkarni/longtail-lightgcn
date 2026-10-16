import math

import numpy as np

from longtail_lightgcn.config import TrainConfig
from longtail_lightgcn.data.preprocess import make_split
from longtail_lightgcn.train import evaluate, train


def planted_dataset(n_groups=4, users_per_group=50, items_per_group=50, seed=0):
    """Users only interact with items of their own group (50% of them), so there is real signal."""
    rng = np.random.default_rng(seed)
    users, items = [], []
    for g in range(n_groups):
        group_items = np.arange(g * items_per_group, (g + 1) * items_per_group)
        for u in range(g * users_per_group, (g + 1) * users_per_group):
            picked = rng.choice(group_items, size=items_per_group // 2, replace=False)
            users += [u] * len(picked)
            items += list(picked)
    n_users, n_items = n_groups * users_per_group, n_groups * items_per_group
    return np.array(users), np.array(items), n_users, n_items


def small_cfg(**kw):
    base = dict(embed_dim=16, n_layers=2, lr=0.01, init_std=0.1, batch_size=256,
                max_epochs=60, eval_every=10, patience=5, k=10, seed=0)
    base.update(kw)
    return TrainConfig(**base)


def run(cfg):
    users, items, nu, ni = planted_dataset()
    split = make_split(users, items, seed=0)
    return train(split, nu, ni, cfg, log=lambda *_: None), split


def test_training_learns_planted_structure():
    # chance Recall@10 is about 10 / 180 = 0.055 (items not in train); a working model is far above it
    result, split = run(small_cfg())
    assert result.best_val_recall > 0.15
    assert result.history[-1]["bpr_loss"] < result.history[0]["bpr_loss"]
    assert result.history[0]["bpr_loss"] < math.log(2) + 0.05  # starts near ln 2 (equal scores)


def test_returns_best_weights_not_last():
    result, split = run(small_cfg())
    again = evaluate(result.model, result.adj, split, 10, "val")
    assert math.isclose(again, result.best_val_recall, rel_tol=1e-6)


def test_early_stopping_triggers_after_patience():
    # lr=0 means embeddings never change, so recall never improves after the first evaluation:
    # eval 1 sets the best, eval 2 is stale -> patience=1 stops at epoch 20
    result, _ = run(small_cfg(lr=0.0, patience=1, eval_every=10, max_epochs=100))
    assert [h["epoch"] for h in result.history] == [10, 20]
    assert result.best_epoch == 10


def test_same_seed_is_reproducible():
    a, _ = run(small_cfg(max_epochs=20))
    b, _ = run(small_cfg(max_epochs=20))
    assert [h["bpr_loss"] for h in a.history] == [h["bpr_loss"] for h in b.history]
    assert a.best_val_recall == b.best_val_recall


def test_test_set_is_only_touched_on_request():
    result, split = run(small_cfg(max_epochs=20))
    val = evaluate(result.model, result.adj, split, 10, "val")
    test = evaluate(result.model, result.adj, split, 10, "test")
    assert 0.0 <= test <= 1.0 and val != test
