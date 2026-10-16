import numpy as np
import torch

from longtail_lightgcn.config import TrainConfig
from longtail_lightgcn.data.adjacency import build_norm_adj
from longtail_lightgcn.model import LightGCN

# Same tiny graph as test_adjacency: 3 users, 3 items.
# Edges: u0-i0, u0-i1, u1-i1, u2-i1. Degrees: u0=2, u1=1, u2=1 | i0=1, i1=3, i2=0.
USERS = np.array([0, 0, 1, 2])
ITEMS = np.array([0, 1, 1, 1])

# hand-picked layer-0 embeddings (d=2): rows u0,u1,u2,i0,i1,i2
E0 = np.array(
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 3.0], [5.0, 5.0]],
    dtype=np.float32,
)


def make_model(n_layers):
    model = LightGCN(3, 3, embed_dim=2, n_layers=n_layers)
    with torch.no_grad():
        model.user_emb.weight.copy_(torch.tensor(E0[:3]))
        model.item_emb.weight.copy_(torch.tensor(E0[3:]))
    return model


def tiny_adj():
    return build_norm_adj(USERS, ITEMS, 3, 3)


def test_one_layer_matches_hand_computation():
    # Layer 1 by hand, straight from the formula e_u^(1) = sum_i 1/sqrt(|N(u)||N(i)|) e_i^(0):
    #   u0 = 1/sqrt(2*1) * i0 + 1/sqrt(2*3) * i1
    #   u1 = 1/sqrt(1*3) * i1
    #   i0 = 1/sqrt(1*2) * u0
    #   i1 = 1/sqrt(3*2) * u0 + 1/sqrt(3*1) * u1 + 1/sqrt(3*1) * u2
    #   i2 = 0 (no neighbours)
    i0, i1 = E0[3], E0[4]
    u0, u1, u2 = E0[0], E0[1], E0[2]
    layer1 = {
        "u0": i0 / np.sqrt(2) + i1 / np.sqrt(6),
        "u1": i1 / np.sqrt(3),
        "u2": i1 / np.sqrt(3),
        "i0": u0 / np.sqrt(2),
        "i1": u0 / np.sqrt(6) + u1 / np.sqrt(3) + u2 / np.sqrt(3),
        "i2": np.zeros(2),
    }
    expected = np.stack([layer1[k] for k in ("u0", "u1", "u2", "i0", "i1", "i2")])
    expected_final = (E0 + expected) / 2  # mean of layer 0 and layer 1 (K=1)

    uf, itf = make_model(n_layers=1).propagate(tiny_adj())
    got = torch.cat([uf, itf]).detach().numpy()
    np.testing.assert_allclose(got, expected_final, atol=1e-5)


def test_three_layers_match_matrix_powers_and_mean_combination():
    a = tiny_adj().to_dense().numpy()
    layers = [E0, a @ E0, a @ a @ E0, a @ a @ a @ E0]
    expected = np.mean(layers, axis=0)

    uf, itf = make_model(n_layers=3).propagate(tiny_adj())
    got = torch.cat([uf, itf]).detach().numpy()
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_isolated_item_keeps_only_its_layer0_share():
    # item 2 has no train edges, so layers 1..K are zero and its final embedding is E0 / (K+1)
    k = 3
    _, itf = make_model(n_layers=k).propagate(tiny_adj())
    np.testing.assert_allclose(itf[2].detach().numpy(), E0[5] / (k + 1), atol=1e-6)


def test_scores_are_dot_products_of_final_embeddings():
    model = make_model(n_layers=2)
    uf, itf = model.propagate(tiny_adj())
    s = model.score(uf, itf, torch.tensor([0, 1]), torch.tensor([1, 0]))
    expected = [(uf[0] * itf[1]).sum().item(), (uf[1] * itf[0]).sum().item()]
    np.testing.assert_allclose(s.detach().numpy(), expected, atol=1e-6)


def test_gradients_reach_layer0_embeddings_through_propagation():
    model = make_model(n_layers=2)
    uf, itf = model.propagate(tiny_adj())
    model.score(uf, itf, torch.tensor([0]), torch.tensor([1])).sum().backward()
    assert model.user_emb.weight.grad is not None and model.item_emb.weight.grad is not None
    assert model.user_emb.weight.grad.abs().sum() > 0


def test_init_is_normal_with_configured_std():
    torch.manual_seed(0)
    cfg = TrainConfig()
    model = LightGCN(6040, 3706, cfg.embed_dim, cfg.n_layers, cfg.init_std)
    w = model.ego_embeddings().detach()
    assert w.shape == (6040 + 3706, 64)
    assert abs(w.mean().item()) < 1e-3
    assert abs(w.std().item() - 0.01) < 5e-4
