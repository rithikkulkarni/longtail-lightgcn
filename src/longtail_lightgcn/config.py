"""Hyperparameters, defaulting to the LightGCN authors' README settings (DESIGN.md D-12, D-14, D-19)."""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # model
    embed_dim: int = 64
    n_layers: int = 3
    init_std: float = 0.01  # Normal(0, init_std), as in the authors' official code (D-14)

    # Intervention A: degree-discounted aggregation, gamma in [0, 1]; 0 = baseline (D-26)
    gamma: float = 0.0

    # Intervention B: popularity-aware loss re-weighting, w(i) = pop(i)^-beta, beta in [0, 1]; 0 = baseline (D-27)
    beta: float = 0.0

    # Intervention C: content-based item embeddings, e_i^(0) = e_i^ID + f_theta(x_i); False = baseline (D-28)
    use_content: bool = False

    # optimization
    lr: float = 1e-3
    l2_reg: float = 1e-4  # weight on the batch-level L2 term (D-15)
    batch_size: int = 2048
    max_epochs: int = 1000

    # evaluation / early stopping (authors' schedule, but on validation Recall@K)
    k: int = 20
    eval_every: int = 20
    patience: int = 5  # stop after this many consecutive evaluations without improvement

    # reproducibility
    split_seed: int = 0  # which saved split to load (split_{seed}.npz)
    seed: int = 0  # model init and negative sampling
