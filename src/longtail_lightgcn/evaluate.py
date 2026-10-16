"""Minimal full-ranking Recall@K (DESIGN.md D-13, D-18). Fuller metrics arrive in M3."""
import numpy as np
import torch


def _edge_mask(pairs, n_users: int, n_items: int, device) -> torch.Tensor:
    """Dense (n_users, n_items) boolean matrix, True where an edge exists in any of `pairs`."""
    mask = torch.zeros(n_users, n_items, dtype=torch.bool, device=device)
    for users, items in pairs:
        mask[torch.as_tensor(np.asarray(users), dtype=torch.long, device=device),
             torch.as_tensor(np.asarray(items), dtype=torch.long, device=device)] = True
    return mask


@torch.no_grad()
def recall_at_k(
    user_final: torch.Tensor,
    item_final: torch.Tensor,
    k: int,
    exclude_pairs,
    target_pairs,
    user_batch: int = 1024,
) -> float:
    """Full-ranking Recall@K, averaged over users that have at least one target item.

    For each user: score every item, set the scores of `exclude_pairs` items (already-seen
    edges) to -inf, take the top K, and compute hits / number of target items.

    exclude_pairs / target_pairs: lists of (users, items) index arrays.
      validation: exclude=[train],      target=[val]
      test:       exclude=[train, val], target=[test]
    """
    n_users, n_items = user_final.shape[0], item_final.shape[0]
    device = user_final.device
    exclude = _edge_mask(exclude_pairs, n_users, n_items, device)
    target = _edge_mask(target_pairs, n_users, n_items, device)

    recalls = []
    for start in range(0, n_users, user_batch):
        sl = slice(start, start + user_batch)
        scores = user_final[sl] @ item_final.T
        scores = scores.masked_fill(exclude[sl], float("-inf"))
        top_idx = scores.topk(k, dim=1).indices
        tgt = target[sl]
        n_targets = tgt.sum(dim=1)
        hits = tgt.gather(1, top_idx).sum(dim=1)
        has_targets = n_targets > 0
        recalls.append(hits[has_targets].float() / n_targets[has_targets].float())
    return torch.cat(recalls).mean().item()
