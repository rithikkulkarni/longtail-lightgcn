from pathlib import Path

import numpy as np
import pytest
import torch

from longtail_lightgcn.data.adjacency import build_norm_adj
from longtail_lightgcn.data.preprocess import load_split, make_split

# Tiny graph: 3 users, 3 items. Edges: u0-i0, u0-i1, u1-i1, u2-i1.
# Degrees: u0=2, u1=1, u2=1 | i0=1, i1=3, i2=0 (i2 is a degree-0 item).
USERS = np.array([0, 0, 1, 2])
ITEMS = np.array([0, 1, 1, 1])


def expected_tiny():
    """Hand-computed D^-1/2 A D^-1/2. Item i lives at index 3+i."""
    a = np.zeros((6, 6))
    a[0, 3] = 1 / np.sqrt(2 * 1)  # u0-i0
    a[0, 4] = 1 / np.sqrt(2 * 3)  # u0-i1
    a[1, 4] = 1 / np.sqrt(1 * 3)  # u1-i1
    a[2, 4] = 1 / np.sqrt(1 * 3)  # u2-i1
    return a + a.T


def test_tiny_graph_matches_hand_computation():
    adj = build_norm_adj(USERS, ITEMS, n_users=3, n_items=3)
    assert adj.is_sparse and adj.shape == (6, 6)
    np.testing.assert_allclose(adj.to_dense().numpy(), expected_tiny(), atol=1e-6)


def test_symmetric_and_nnz():
    adj = build_norm_adj(USERS, ITEMS, 3, 3)
    dense = adj.to_dense()
    assert torch.equal(dense, dense.T)
    assert adj._nnz() == 2 * len(USERS)  # each edge appears in both directions


def test_degree_zero_node_is_all_zero_and_finite():
    dense = build_norm_adj(USERS, ITEMS, 3, 3).to_dense()
    assert torch.isfinite(dense).all()
    assert dense[5].abs().sum() == 0 and dense[:, 5].abs().sum() == 0  # item 2 -> index 5


def test_adjacency_excludes_val_and_test_edges():
    rng = np.random.default_rng(0)
    n_users, n_items = 30, 50
    users, items = [], []
    for u in range(n_users):
        picked = rng.choice(n_items, size=rng.integers(20, 41), replace=False)
        users += [u] * len(picked)
        items += list(picked)
    split = make_split(np.array(users), np.array(items), seed=0)

    adj = build_norm_adj(split["train_users"], split["train_items"], n_users, n_items).to_dense()
    for k in ("val", "test"):
        u, i = split[f"{k}_users"], split[f"{k}_items"]
        assert (adj[u, n_users + i] == 0).all()
    u, i = split["train_users"], split["train_items"]
    assert (adj[u, n_users + i] > 0).all()


PROCESSED = Path("data/processed/split_0.npz")


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data not built")
def test_real_split_shape_and_nnz():
    split, n_users, n_items = load_split(PROCESSED)
    adj = build_norm_adj(split["train_users"], split["train_items"], n_users, n_items)
    assert adj.shape == (n_users + n_items, n_users + n_items)
    assert adj._nnz() == 2 * len(split["train_users"])
    assert torch.isfinite(adj.values()).all()
