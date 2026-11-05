"""Ranking and popularity-bias metrics (DESIGN.md D-21 to D-25).

Every metric takes `recs`, a (n_users, K) LongTensor of recommended item indices ordered best
first, as produced by `recommend_top_k`. Accuracy metrics also take `target`, a dense
(n_users, n_items) bool matrix of held-out items.
"""
import torch

from longtail_lightgcn.evaluate import _edge_mask


@torch.no_grad()
def recommend_top_k(user_final, item_final, k: int, exclude_pairs, user_batch: int = 1024) -> torch.Tensor:
    """Top-K item indices per user over the full catalog, with already-seen items masked out.

    exclude_pairs: list of (users, items) index arrays (validation: [train]; test: [train, val]).
    """
    n_users, n_items = user_final.shape[0], item_final.shape[0]
    exclude = _edge_mask(exclude_pairs, n_users, n_items, user_final.device)
    out = []
    for start in range(0, n_users, user_batch):
        sl = slice(start, start + user_batch)
        scores = (user_final[sl] @ item_final.T).masked_fill(exclude[sl], float("-inf"))
        out.append(scores.topk(k, dim=1).indices)
    return torch.cat(out)


def target_mask(target_pairs, n_users: int, n_items: int, device="cpu") -> torch.Tensor:
    """Dense bool (n_users, n_items) matrix of held-out edges, from a list of (users, items) arrays."""
    return _edge_mask(target_pairs, n_users, n_items, device)


def recall_at_k(recs: torch.Tensor, target: torch.Tensor) -> float:
    """Mean over users with >= 1 held-out item of hits / number of held-out items (D-18)."""
    n_targets = target.sum(dim=1)
    hits = target.gather(1, recs).sum(dim=1)
    has = n_targets > 0
    return (hits[has].double() / n_targets[has].double()).mean().item()


def ndcg_at_k(recs: torch.Tensor, target: torch.Tensor) -> float:
    """Binary-relevance NDCG@K with a log2 discount, averaged over users with >= 1 held-out item.

    The ideal list holds min(K, number of held-out items) hits, so a perfect ranking scores 1.
    """
    k = recs.shape[1]
    discount = 1.0 / torch.log2(torch.arange(2, k + 2, device=recs.device, dtype=torch.float64))
    gains = target.gather(1, recs).double()
    dcg = (gains * discount).sum(dim=1)

    n_targets = target.sum(dim=1)
    has = n_targets > 0
    ideal_hits = n_targets.clamp(max=k).clamp(min=1)  # clamp(min=1) only avoids index -1 for skipped users
    idcg = discount.cumsum(dim=0)[ideal_hits - 1]
    return (dcg[has] / idcg[has]).mean().item()


def item_exposure(recs: torch.Tensor, n_items: int) -> torch.Tensor:
    """How many users' lists contain each item, shape (n_items,); zero for never-recommended items."""
    return torch.bincount(recs.reshape(-1), minlength=n_items)


def coverage(recs: torch.Tensor, n_items: int) -> float:
    """Share of the whole catalog (n_items, D-04) recommended to at least one user."""
    return (item_exposure(recs, n_items) > 0).sum().item() / n_items


def gini(exposure: torch.Tensor) -> float:
    """Gini coefficient of item exposure over ALL catalog items, zeros included (D-22).

    0 = every item exposed equally; (n-1)/n = a single item gets everything.
    Uses G = sum_i (2i - n - 1) x_(i) / (n * sum x) with x sorted ascending and i starting at 1.
    """
    x = exposure.double().sort().values
    n, total = x.numel(), x.sum()
    if total == 0:
        raise ValueError("Gini is undefined when no item is ever recommended.")
    ranks = torch.arange(1, n + 1, device=x.device, dtype=torch.float64)
    return ((2 * ranks - n - 1) * x).sum().div(n * total).item()


def average_recommendation_popularity(recs: torch.Tensor, train_degree: torch.Tensor) -> float:
    """Mean training degree of recommended items (D-23).

    Every user has exactly K recommendations, so per-user-then-averaged equals pooled.
    train_degree: (n_items,) training-edge counts, from the train split only.
    """
    return train_degree[recs].double().mean().item()


def target_popularity(target: torch.Tensor, train_degree: torch.Tensor) -> float:
    """Reference for ARP: mean training degree of each user's held-out items, averaged over users.

    Per user first, then over users with >= 1 held-out item, to mirror how ARP treats each user
    equally. A pooled mean over held-out edges would weight heavy users more.
    """
    n_targets = target.sum(dim=1)
    has = n_targets > 0
    per_user = (target.double() @ train_degree.double())[has] / n_targets[has].double()
    return per_user.mean().item()


def all_metrics(recs: torch.Tensor, target: torch.Tensor, train_degree: torch.Tensor, n_items: int) -> dict:
    """All five proposal metrics plus the held-out popularity reference (D-23), as plain Python floats."""
    k = recs.shape[1]
    return {
        f"recall@{k}": recall_at_k(recs, target),
        f"ndcg@{k}": ndcg_at_k(recs, target),
        "coverage": coverage(recs, n_items),
        "gini": gini(item_exposure(recs, n_items)),
        "arp": average_recommendation_popularity(recs, train_degree),
        "target_pop": target_popularity(target, train_degree),
    }
