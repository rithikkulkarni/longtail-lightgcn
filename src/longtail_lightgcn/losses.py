"""BPR loss and the batch-level L2 term (DESIGN.md D-15), matching the authors' official code."""
import torch
import torch.nn.functional as F


def pop_weights(train_items, n_items: int, beta: float) -> torch.Tensor:
    """Intervention B (D-27): w(i) = pop(i)^-beta per item, pop = training degree, shape (n_items,).

    Raw weights, no normalization (as in Eq. 2). beta = 0 gives exactly 1 for every item. Degree-0 items
    are never a positive in a training triple, so their weight is never used; the clamp only avoids inf.
    """
    deg = torch.bincount(torch.as_tensor(train_items, dtype=torch.long), minlength=n_items)
    return deg.clamp(min=1).to(torch.float32).pow(-beta)


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """Mean over the batch of w * -log sigmoid(pos - neg); w = 1 when `weights` is None.

    Written as softplus(neg - pos), which equals -log sigmoid(pos - neg) but stays finite
    when the score gap is large (log(sigmoid(x)) underflows to -inf for very negative x).

    `weights` is the per-sample weight of Intervention B (Eq. 2), looked up from the positive item. The
    weighted sum is divided by the batch size, like the baseline's mean, so weights of 1 reproduce it exactly.
    """
    per_sample = F.softplus(neg_scores - pos_scores)
    if weights is not None:
        per_sample = per_sample * weights
    return per_sample.mean()


def l2_reg(user_ego: torch.Tensor, pos_ego: torch.Tensor, neg_ego: torch.Tensor) -> torch.Tensor:
    """0.5 * (sum of squared layer-0 embeddings of the batch's users, positives, negatives) / batch size.

    Only the rows that appear in the batch are penalized (D-15). The caller multiplies this by lambda.
    """
    batch_size = user_ego.shape[0]
    sq = user_ego.pow(2).sum() + pos_ego.pow(2).sum() + neg_ego.pow(2).sum()
    return 0.5 * sq / batch_size
