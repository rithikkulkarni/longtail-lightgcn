import math
from pathlib import Path

import numpy as np
import pytest
import torch

from longtail_lightgcn.data.preprocess import load_split
from longtail_lightgcn.evaluate import recall_at_k

# 3 users, 6 items, 1-d embeddings so scores are easy to read.
# item i has embedding [5,4,3,2,1,0][i]; user0 = +1, user1 = -1, user2 = +1.
#   user0 scores: [5,4,3,2,1,0]       -> ranks items 0,1,2,3,4,5
#   user1 scores: [-5,-4,-3,-2,-1,0]  -> ranks items 5,4,3,2,1,0
ITEMS = torch.tensor([[5.0], [4.0], [3.0], [2.0], [1.0], [0.0]])
USERS = torch.tensor([[1.0], [-1.0], [1.0]])

EXCLUDE = [(np.array([0]), np.array([0]))]  # user0 has item 0 in train
TARGET = [(np.array([0, 0, 1, 1, 1]), np.array([1, 5, 5, 4, 0]))]  # user2 has no targets


def test_hand_computed_recall_at_2():
    # user0: item 0 masked -> top-2 = items 1,2; targets {1,5}: 1 hit / 2 = 0.5
    # user1: top-2 = items 5,4;               targets {5,4,0}: 2 hits / 3 = 0.6667
    # user2: no targets -> skipped from the average
    r = recall_at_k(USERS, ITEMS, 2, EXCLUDE, TARGET)
    assert math.isclose(r, (0.5 + 2 / 3) / 2, rel_tol=1e-6)


def test_exclusion_changes_the_ranking():
    # k=1: with item 0 masked, user0's top-1 is item 1 (a target): recall 1/2.
    #      without the mask, top-1 is item 0 (not a target): recall 0.
    only_user0_target = [(np.array([0, 0]), np.array([1, 5]))]
    masked = recall_at_k(USERS, ITEMS, 1, EXCLUDE, only_user0_target)
    unmasked = recall_at_k(USERS, ITEMS, 1, [(np.array([], dtype=int), np.array([], dtype=int))], only_user0_target)
    assert math.isclose(masked, 0.5) and unmasked == 0.0


def test_multiple_exclusion_sets_are_combined():
    # excluding both item 0 and item 1 for user0: top-1 becomes item 2
    exclude = [(np.array([0]), np.array([0])), (np.array([0]), np.array([1]))]
    target = [(np.array([0]), np.array([2]))]
    assert math.isclose(recall_at_k(USERS, ITEMS, 1, exclude, target), 1.0)


def test_user_batching_gives_same_answer():
    a = recall_at_k(USERS, ITEMS, 2, EXCLUDE, TARGET, user_batch=1)
    b = recall_at_k(USERS, ITEMS, 2, EXCLUDE, TARGET, user_batch=1024)
    assert math.isclose(a, b)


PROCESSED = Path("data/processed/split_0.npz")


@pytest.mark.skipif(not PROCESSED.exists(), reason="processed data not built")
def test_random_embeddings_give_chance_level_recall():
    # Random scores should recall about K / (items not excluded) ~ 20/3600 ~ 0.0056.
    # This catches gross bugs (e.g. leaking targets or forgetting the mask), not subtle ones.
    split, n_users, n_items = load_split(PROCESSED)
    g = torch.Generator().manual_seed(0)
    u = torch.randn(n_users, 16, generator=g)
    i = torch.randn(n_items, 16, generator=g)
    train = (split["train_users"], split["train_items"])
    val = (split["val_users"], split["val_items"])
    r = recall_at_k(u, i, 20, [train], [val])
    assert 0.002 < r < 0.012
