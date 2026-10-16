"""BPR loss and the batch-level L2 term (DESIGN.md D-15), matching the authors' official code."""
import torch
import torch.nn.functional as F


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Mean over the batch of -log sigmoid(pos - neg).

    Written as softplus(neg - pos), which equals -log sigmoid(pos - neg) but stays finite
    when the score gap is large (log(sigmoid(x)) underflows to -inf for very negative x).
    """
    return F.softplus(neg_scores - pos_scores).mean()


def l2_reg(user_ego: torch.Tensor, pos_ego: torch.Tensor, neg_ego: torch.Tensor) -> torch.Tensor:
    """0.5 * (sum of squared layer-0 embeddings of the batch's users, positives, negatives) / batch size.

    Only the rows that appear in the batch are penalized (D-15). The caller multiplies this by lambda.
    """
    batch_size = user_ego.shape[0]
    sq = user_ego.pow(2).sum() + pos_ego.pow(2).sum() + neg_ego.pow(2).sum()
    return 0.5 * sq / batch_size
