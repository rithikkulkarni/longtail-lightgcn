import math

import torch

from longtail_lightgcn.losses import bpr_loss, l2_reg


def test_equal_scores_give_log_two():
    s = torch.tensor([0.3, -1.0, 2.0])
    assert math.isclose(bpr_loss(s, s).item(), math.log(2), rel_tol=1e-6)


def test_matches_hand_computed_value():
    # pair 1: pos=2, neg=0 -> -log sigmoid(2) = log(1 + e^-2) = 0.126928...
    # pair 2: pos=0, neg=0 -> log 2 = 0.693147...
    loss = bpr_loss(torch.tensor([2.0, 0.0]), torch.tensor([0.0, 0.0]))
    expected = (math.log(1 + math.exp(-2)) + math.log(2)) / 2
    assert math.isclose(loss.item(), expected, rel_tol=1e-6)


def test_equals_negative_log_sigmoid_formula():
    pos, neg = torch.randn(50), torch.randn(50)
    naive = -torch.log(torch.sigmoid(pos - neg)).mean()
    assert math.isclose(bpr_loss(pos, neg).item(), naive.item(), rel_tol=1e-5)


def test_stable_for_extreme_score_gaps():
    # naive -log(sigmoid(-1000)) is inf in float32; softplus form must stay finite
    wrong = bpr_loss(torch.tensor([0.0]), torch.tensor([1000.0]))
    right = bpr_loss(torch.tensor([1000.0]), torch.tensor([0.0]))
    assert math.isclose(wrong.item(), 1000.0, rel_tol=1e-6)
    assert right.item() == 0.0 or right.item() < 1e-6


def test_gradient_at_equal_scores():
    # d/dpos softplus(neg-pos) = -sigmoid(neg-pos) = -0.5 per pair; mean over 2 pairs -> -0.25
    pos = torch.zeros(2, requires_grad=True)
    neg = torch.zeros(2, requires_grad=True)
    bpr_loss(pos, neg).backward()
    assert torch.allclose(pos.grad, torch.full((2,), -0.25))
    assert torch.allclose(neg.grad, torch.full((2,), 0.25))


def test_l2_reg_matches_hand_computation():
    u = torch.tensor([[1.0, 2.0], [0.0, 0.0]])  # sum sq = 5
    p = torch.tensor([[3.0, 0.0], [0.0, 0.0]])  # 9
    n = torch.tensor([[0.0, 1.0], [0.0, 0.0]])  # 1
    # 0.5 * 15 / batch_size(2) = 3.75
    assert math.isclose(l2_reg(u, p, n).item(), 3.75, rel_tol=1e-6)


def test_l2_reg_gradient_is_embedding_over_batch_size():
    # d/dx [0.5 * x^2 / B] = x / B, so each embedding is pulled toward zero in proportion to its value
    u = torch.tensor([[2.0, 4.0], [1.0, 1.0]], requires_grad=True)
    p = torch.zeros(2, 2, requires_grad=True)
    n = torch.zeros(2, 2, requires_grad=True)
    l2_reg(u, p, n).backward()
    assert torch.allclose(u.grad, u.detach() / 2)
