"""Training loop for baseline LightGCN with BPR loss, validation-based early stopping (D-12, D-16)."""
import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from longtail_lightgcn.config import TrainConfig
from longtail_lightgcn.data.adjacency import build_norm_adj
from longtail_lightgcn.evaluate import recall_at_k
from longtail_lightgcn.losses import bpr_loss, l2_reg
from longtail_lightgcn.model import LightGCN
from longtail_lightgcn.sampler import NegativeSampler


@dataclass
class TrainResult:
    model: LightGCN  # holds the BEST weights (by validation Recall@K), not the last ones
    adj: torch.Tensor
    best_epoch: int
    best_val_recall: float
    history: list = field(default_factory=list)  # one dict per evaluation


@torch.no_grad()
def evaluate(model: LightGCN, adj: torch.Tensor, split: dict, k: int, which: str = "val") -> float:
    """Recall@K on validation (exclude train) or test (exclude train + val)."""
    model.eval()
    user_final, item_final = model.propagate(adj)
    train = (split["train_users"], split["train_items"])
    if which == "val":
        exclude, target = [train], [(split["val_users"], split["val_items"])]
    elif which == "test":
        exclude = [train, (split["val_users"], split["val_items"])]
        target = [(split["test_users"], split["test_items"])]
    else:
        raise ValueError(f"which must be 'val' or 'test', got {which!r}")
    return recall_at_k(user_final, item_final, k, exclude, target)


def train(split: dict, n_users: int, n_items: int, cfg: TrainConfig, device: str = "cpu", log=print) -> TrainResult:
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    adj = build_norm_adj(split["train_users"], split["train_items"], n_users, n_items).to(device)
    model = LightGCN(n_users, n_items, cfg.embed_dim, cfg.n_layers, cfg.init_std).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sampler = NegativeSampler(split["train_users"], split["train_items"], n_users, n_items, seed=cfg.seed)

    users_all = torch.as_tensor(split["train_users"], dtype=torch.long, device=device)
    items_all = torch.as_tensor(split["train_items"], dtype=torch.long, device=device)
    n_edges = len(users_all)

    best_recall, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        # one negative per train edge, redrawn every epoch (D-16)
        negs_all = torch.as_tensor(sampler.sample(split["train_users"]), dtype=torch.long, device=device)
        perm = torch.as_tensor(rng.permutation(n_edges), dtype=torch.long, device=device)

        bpr_sum, reg_sum = 0.0, 0.0
        for start in range(0, n_edges, cfg.batch_size):
            idx = perm[start : start + cfg.batch_size]
            u, i, j = users_all[idx], items_all[idx], negs_all[idx]

            user_final, item_final = model.propagate(adj)  # full-graph propagation each step
            pos_scores = model.score(user_final, item_final, u, i)
            neg_scores = model.score(user_final, item_final, u, j)

            bpr = bpr_loss(pos_scores, neg_scores)
            reg = l2_reg(model.user_emb(u), model.item_emb(i), model.item_emb(j))
            loss = bpr + cfg.l2_reg * reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bpr_sum += bpr.item() * len(idx)
            reg_sum += reg.item() * len(idx)

        if epoch % cfg.eval_every == 0 or epoch == cfg.max_epochs:
            val_recall = evaluate(model, adj, split, cfg.k, "val")
            row = {
                "epoch": epoch,
                "bpr_loss": bpr_sum / n_edges,
                "l2_reg": reg_sum / n_edges,
                f"val_recall@{cfg.k}": val_recall,
                "epoch_seconds": time.time() - t0,
            }
            history.append(row)
            log(
                f"epoch {epoch:4d} | bpr {row['bpr_loss']:.4f} | reg {row['l2_reg']:.4f} | "
                f"val recall@{cfg.k} {val_recall:.4f} | {row['epoch_seconds']:.1f}s/epoch"
            )

            if val_recall > best_recall:  # strict improvement
                best_recall, best_epoch, stale = val_recall, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= cfg.patience:
                    log(f"early stop at epoch {epoch}: no improvement for {cfg.patience} evaluations")
                    break

    model.load_state_dict(best_state)
    return TrainResult(model, adj, best_epoch, best_recall, history)
