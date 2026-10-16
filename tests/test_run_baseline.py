import json

import numpy as np
import torch

from longtail_lightgcn.data.preprocess import make_split, save_split
from longtail_lightgcn.run_baseline import main


def test_end_to_end_on_tiny_split(tmp_path):
    rng = np.random.default_rng(0)
    n_users, n_items = 40, 60
    users, items = [], []
    for u in range(n_users):
        picked = rng.choice(n_items, size=25, replace=False)
        users += [u] * 25
        items += list(picked)
    split = make_split(np.array(users), np.array(items), seed=3)
    save_split(tmp_path / "split_3.npz", split, n_users, n_items)

    summary = main([
        "--data_dir", str(tmp_path), "--out_dir", str(tmp_path / "out"), "--device", "cpu",
        "--split_seed", "3", "--embed_dim", "8", "--n_layers", "2", "--batch_size", "128",
        "--max_epochs", "4", "--eval_every", "2", "--k", "5", "--lr", "0.01",
    ])

    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert saved["config"]["split_seed"] == 3 and saved["config"]["embed_dim"] == 8  # overrides applied
    assert saved["config"]["l2_reg"] == 1e-4  # untouched fields keep the paper default
    assert 0.0 <= saved["test_recall@5"] <= 1.0 and 0.0 <= saved["val_recall@5"] <= 1.0
    assert [h["epoch"] for h in saved["history"]] == [2, 4]
    state = torch.load(tmp_path / "out" / "model.pt")
    assert state["user_emb.weight"].shape == (n_users, 8)
    assert summary["best_epoch"] in (2, 4)
