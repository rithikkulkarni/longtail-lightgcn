"""Normalized bipartite adjacency for LightGCN (DESIGN.md D-06, D-07)."""
import numpy as np
import torch


def build_norm_adj(train_users, train_items, n_users: int, n_items: int, gamma: float = 0.0) -> torch.Tensor:
    """Return D^-1/2 A D^-1/2 as a coalesced COO tensor of shape (U+I, U+I).

    A = [[0, R], [R^T, 0]] where R[u, i] = 1 for each TRAIN edge. Users occupy
    indices 0..U-1 and item i sits at index U+i. Only train edges are used, so
    val/test interactions can never leak into propagation.

    Nodes with degree 0 get a normalization factor of 0 instead of 1/sqrt(0) = inf (D-07).

    Intervention A (D-26): gamma > 0 returns S @ A_hat, where S is diagonal with 1 for user rows and
    |N(i)|^-gamma for item row i. This is Eq. 1: the item-update direction is discounted, so the
    matrix is no longer symmetric. gamma = 0 gives S = I, i.e. the baseline matrix exactly.
    """
    users = torch.as_tensor(np.asarray(train_users), dtype=torch.long)
    items = torch.as_tensor(np.asarray(train_items), dtype=torch.long) + n_users
    n = n_users + n_items

    rows = torch.cat([users, items])
    cols = torch.cat([items, users])

    deg = torch.bincount(rows, minlength=n).to(torch.float32)
    d_inv_sqrt = torch.zeros(n)
    nonzero = deg > 0
    d_inv_sqrt[nonzero] = deg[nonzero].pow(-0.5)

    # row scale S: users stay 1; item rows get deg^-gamma (degree-0 rows have no entries, so 0 is never used)
    item_deg, item_nz = deg[n_users:], nonzero[n_users:]
    item_scale = torch.zeros(n_items)
    item_scale[item_nz] = item_deg[item_nz].pow(-gamma)
    row_scale = torch.cat([torch.ones(n_users), item_scale])

    values = d_inv_sqrt[rows] * d_inv_sqrt[cols] * row_scale[rows]
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), values, (n, n)).coalesce()
