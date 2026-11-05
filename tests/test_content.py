"""Intervention C: content-based item embeddings, e_i^(0) = e_i^ID + f_theta(x_i) (DESIGN.md D-28)."""
import json

import numpy as np
import pytest
import torch

from longtail_lightgcn.data.adjacency import build_norm_adj
from longtail_lightgcn.data.preprocess import make_split, save_split
from longtail_lightgcn.model import LightGCN
from longtail_lightgcn.run_baseline import build_parser, main
from longtail_lightgcn.train import train
from test_train import planted_dataset, run, small_cfg

# Same tiny graph as test_model: 3 users, 3 items; item 2 has no train edges.
USERS = np.array([0, 0, 1, 2])
ITEMS = np.array([0, 1, 1, 1])
E0 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 3.0], [5.0, 5.0]], dtype=np.float32)
FEATS = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 items, 2 content features


def tiny_adj():
    return build_norm_adj(USERS, ITEMS, 3, 3)


def make_content_model(n_layers=1):
    model = LightGCN(3, 3, embed_dim=2, n_layers=n_layers, item_features=FEATS, mlp_hidden=4)
    with torch.no_grad():
        model.user_emb.weight.copy_(torch.tensor(E0[:3]))
        model.item_emb.weight.copy_(torch.tensor(E0[3:]))
    return model


# ---- model ----

def test_layer0_is_id_plus_mlp_of_features():
    model = make_content_model()
    ego = model.ego_embeddings()
    assert ego.shape == (6, 2)
    assert torch.equal(ego[:3], model.user_emb.weight)  # users are untouched
    assert torch.equal(ego[3:], model.item_emb.weight + model.content_mlp(FEATS))
    assert not torch.equal(ego[3:], model.item_emb.weight)


def test_mlp_is_one_hidden_relu_layer():
    mlp = make_content_model().content_mlp
    assert [type(m).__name__ for m in mlp] == ["Linear", "ReLU", "Linear"]
    assert (mlp[0].in_features, mlp[0].out_features, mlp[2].out_features) == (2, 4, 2)


def test_gradients_reach_the_mlp_and_the_id_embeddings():
    model = make_content_model(n_layers=2)
    uf, itf = model.propagate(tiny_adj())
    model.score(uf, itf, torch.tensor([0]), torch.tensor([1])).sum().backward()
    assert model.item_emb.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.content_mlp.parameters())


def test_isolated_item_still_gets_a_content_embedding():
    # item 2 has no train edges: layers 1..K are zero, so its final embedding is ego / (K+1),
    # and with C the ego part carries the MLP output, not just a near-random ID vector
    k = 3
    model = make_content_model(n_layers=k)
    _, itf = model.propagate(tiny_adj())
    expected = (model.item_emb.weight[2] + model.content_mlp(FEATS)[2]) / (k + 1)
    np.testing.assert_allclose(itf[2].detach().numpy(), expected.detach().numpy(), atol=1e-6)


def test_content_off_is_exactly_the_baseline_model():
    # no MLP is created, so parameters and random init are identical to the baseline, bit for bit
    torch.manual_seed(0)
    base = LightGCN(3, 3, embed_dim=4, n_layers=2)
    torch.manual_seed(0)
    off = LightGCN(3, 3, embed_dim=4, n_layers=2, item_features=None)
    assert off.content_mlp is None and list(off.state_dict()) == list(base.state_dict())
    assert all(torch.equal(a, b) for a, b in zip(base.parameters(), off.parameters()))
    uf0, if0 = base.propagate(tiny_adj())
    uf1, if1 = off.propagate(tiny_adj())
    assert torch.equal(uf0, uf1) and torch.equal(if0, if1)


def test_content_on_keeps_the_same_id_init_and_saves_only_mlp_extras():
    torch.manual_seed(0)
    base = LightGCN(3, 3, embed_dim=4, n_layers=2)
    torch.manual_seed(0)
    on = LightGCN(3, 3, embed_dim=4, n_layers=2, item_features=FEATS)
    assert torch.equal(base.item_emb.weight, on.item_emb.weight)  # the MLP is built after the embeddings
    extra = set(on.state_dict()) - set(base.state_dict())
    assert extra == {"content_mlp.0.weight", "content_mlp.0.bias", "content_mlp.2.weight", "content_mlp.2.bias"}


def test_wrong_feature_row_count_raises():
    with pytest.raises(ValueError):
        LightGCN(3, 5, embed_dim=2, item_features=FEATS)


# ---- training ----

def group_features(n_groups=4, items_per_group=50):
    """Content features for the planted dataset: each item's group as a one-hot (a stand-in for genre)."""
    return np.kron(np.eye(n_groups, dtype=np.float32), np.ones((items_per_group, 1), dtype=np.float32))


def run_content(cfg):
    users, items, nu, ni = planted_dataset()
    split = make_split(users, items, seed=0)
    return train(split, nu, ni, cfg, log=lambda *_: None, item_features=group_features())


def test_content_changes_training_but_still_learns():
    base, _ = run(small_cfg(max_epochs=20))
    content = run_content(small_cfg(max_epochs=20, use_content=True))
    assert [h["bpr_loss"] for h in base.history] != [h["bpr_loss"] for h in content.history]
    full = run_content(small_cfg(use_content=True))
    assert full.best_val_recall > 0.15
    assert any(n.startswith("content_mlp") for n, _ in full.model.named_parameters())


def test_mlp_weights_are_updated_by_training():
    users, items, nu, ni = planted_dataset()
    split = make_split(users, items, seed=0)
    cfg = small_cfg(max_epochs=10, use_content=True)
    torch.manual_seed(cfg.seed)
    before = LightGCN(nu, ni, cfg.embed_dim, cfg.n_layers, cfg.init_std, item_features=torch.tensor(group_features()))
    result = train(split, nu, ni, cfg, log=lambda *_: None, item_features=group_features())
    # same seed, same construction order -> same MLP init; training must have moved it
    assert not torch.equal(before.content_mlp[0].weight, result.model.content_mlp[0].weight)


def test_use_content_without_features_raises():
    users, items, nu, ni = planted_dataset()
    split = make_split(users, items, seed=0)
    with pytest.raises(ValueError):
        train(split, nu, ni, small_cfg(use_content=True), log=lambda *_: None)


# ---- CLI ----

def test_cli_bool_flag_parses_false_as_false():
    parse = build_parser().parse_args
    assert parse([]).use_content is None  # unset keeps the dataclass default
    assert parse(["--use_content", "True"]).use_content is True
    assert parse(["--use_content", "False"]).use_content is False  # type=bool would give True here
    with pytest.raises(SystemExit):
        parse(["--use_content", "maybe"])


def test_end_to_end_with_content_on(tmp_path):
    rng = np.random.default_rng(0)
    n_users, n_items = 40, 60
    users, items = [], []
    for u in range(n_users):
        picked = rng.choice(n_items, size=25, replace=False)
        users += [u] * 25
        items += list(picked)
    split = make_split(np.array(users), np.array(items), seed=3)
    save_split(tmp_path / "split_3.npz", split, n_users, n_items)
    rows = ["movie_id,item_idx,title,genres,in_catalog"]
    for i in range(n_items):
        rows.append(f"{i + 1},{i},Movie {i} ({1950 + i}),{'Comedy' if i % 2 else 'Drama|War'},True")
    (tmp_path / "items.csv").write_text("\n".join(rows) + "\n")

    summary = main([
        "--data_dir", str(tmp_path), "--out_dir", str(tmp_path / "out"), "--device", "cpu",
        "--split_seed", "3", "--embed_dim", "8", "--n_layers", "2", "--batch_size", "128",
        "--max_epochs", "4", "--eval_every", "2", "--k", "5", "--lr", "0.01", "--use_content", "True",
    ])
    saved = json.loads((tmp_path / "out" / "results.json").read_text())
    assert saved["config"]["use_content"] is True
    assert 0.0 <= saved["test_recall@5"] <= 1.0 and summary["best_epoch"] in (2, 4)
    state = torch.load(tmp_path / "out" / "model.pt")
    assert "content_mlp.0.weight" in state and "item_features" not in state
