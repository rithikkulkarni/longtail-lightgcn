"""Uniform negative sampling with rejection of train positives (DESIGN.md D-16)."""
import numpy as np


class NegativeSampler:
    """Draw one negative item per (user, positive) pair.

    A negative is uniform over all items, redrawn while the user has it in TRAIN.
    Val/test edges are deliberately not rejected: using them would leak held-out data.
    """

    def __init__(self, train_users, train_items, n_users: int, n_items: int, seed: int = 0):
        train_users = np.asarray(train_users, dtype=np.int64)
        train_items = np.asarray(train_items, dtype=np.int64)

        degree = np.bincount(train_users, minlength=n_users)
        full = np.flatnonzero(degree >= n_items)
        if len(full) > 0:
            raise ValueError(
                f"{len(full)} user(s) (e.g. user {full[0]}) have interacted with every item, "
                "so no valid negative exists and rejection sampling would never finish."
            )

        self.n_items = n_items
        # sorted (user, item) keys let us test "is this pair a train edge?" with binary search
        self._keys = np.sort(train_users * n_items + train_items)
        self._rng = np.random.default_rng(seed)

    def _is_train_edge(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        keys = users * self.n_items + items
        pos = np.searchsorted(self._keys, keys)
        pos = np.minimum(pos, len(self._keys) - 1)  # searchsorted can return len(keys)
        return self._keys[pos] == keys

    def sample(self, users: np.ndarray) -> np.ndarray:
        """Return one negative item for each entry of `users`."""
        users = np.asarray(users, dtype=np.int64)
        neg = self._rng.integers(0, self.n_items, size=len(users))
        bad = self._is_train_edge(users, neg)
        while bad.any():
            idx = np.flatnonzero(bad)
            neg[idx] = self._rng.integers(0, self.n_items, size=len(idx))
            bad[idx] = self._is_train_edge(users[idx], neg[idx])
        return neg
