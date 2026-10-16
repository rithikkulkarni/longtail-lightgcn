"""Parse ML-1M, build the implicit-feedback edge list, split it, and save results.

Decisions implemented here: DESIGN.md D-01 (all ratings -> edges), D-02 (per-user
80/10/10 random split, one split per seed), D-03 (no filtering), D-04 (catalog =
rated movies only), D-05 (.npz splits + CSV metadata), D-09 (half-up rounding).
"""
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_KEYS = ("train", "val", "test")


def load_ratings(data_dir: Path) -> pd.DataFrame:
    """Read ratings.dat. Rating and timestamp are dropped: every rating is an edge (D-01)."""
    df = pd.read_csv(
        Path(data_dir) / "ratings.dat",
        sep="::",
        engine="python",  # the separator is two characters, which the C parser doesn't support
        names=["user_id", "movie_id", "rating", "timestamp"],
        usecols=["user_id", "movie_id"],
    )
    return df


def load_movies(data_dir: Path) -> pd.DataFrame:
    """Read movies.dat (title and raw pipe-separated genre string, no feature engineering)."""
    return pd.read_csv(
        Path(data_dir) / "movies.dat",
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",  # movies.dat is not valid UTF-8
    )


def build_edges(ratings: pd.DataFrame):
    """Remap raw IDs to contiguous indices and return the edge list.

    Returns (users, items, user_ids, item_ids):
      users, items: int64 arrays, one entry per edge, holding 0-based indices.
      user_ids, item_ids: sorted original IDs, so original_id = ids[index].
    Only movies that appear in `ratings` get an index (D-04).
    """
    if ratings.duplicated(["user_id", "movie_id"]).any():
        raise ValueError("Duplicate (user, movie) pairs found; edges must be unique.")

    user_ids, users = np.unique(ratings["user_id"].to_numpy(), return_inverse=True)
    item_ids, items = np.unique(ratings["movie_id"].to_numpy(), return_inverse=True)
    return users.astype(np.int64), items.astype(np.int64), user_ids, item_ids


def holdout_size(n: int) -> int:
    """10% of n rounded half-up (D-09). Integer form of floor(0.1*n + 0.5), avoiding float error."""
    return (n + 5) // 10


def make_split(users: np.ndarray, items: np.ndarray, seed: int) -> dict:
    """Per-user random 80/10/10 split (D-02).

    For each user, shuffle their edges, then give the first holdout_size(n) to val,
    the next holdout_size(n) to test, and the rest to train.
    Returns a dict with keys like "train_users", "train_items", "val_users", ...
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(users, kind="stable")
    users, items = users[order], items[order]
    # boundaries of each user's block in the sorted arrays
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(users)]

    parts = {k: [] for k in SPLIT_KEYS}
    for s, e in zip(starts, ends):
        idx = rng.permutation(np.arange(s, e))
        h = holdout_size(e - s)
        parts["val"].append(idx[:h])
        parts["test"].append(idx[h : 2 * h])
        parts["train"].append(idx[2 * h :])

    split = {}
    for k in SPLIT_KEYS:
        idx = np.concatenate(parts[k])
        split[f"{k}_users"] = users[idx]
        split[f"{k}_items"] = items[idx]
    return split


def save_split(path: Path, split: dict, n_users: int, n_items: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, n_users=n_users, n_items=n_items, **split)


def load_split(path: Path):
    """Return (split_dict, n_users, n_items)."""
    with np.load(path) as f:
        split = {k: f[k] for k in f.files if k not in ("n_users", "n_items")}
        return split, int(f["n_users"]), int(f["n_items"])


def save_id_maps(path: Path, user_ids: np.ndarray, item_ids: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, user_ids=user_ids, item_ids=item_ids)


def save_items_csv(path: Path, movies: pd.DataFrame, item_ids: np.ndarray) -> None:
    """Write metadata for ALL movies in movies.dat, flagging which ones are in the catalog (D-04).

    item_idx is the model's item index; it is empty for the unrated movies that were dropped.
    """
    idx_of = pd.Series(np.arange(len(item_ids)), index=item_ids)
    out = movies.copy()
    out["item_idx"] = out["movie_id"].map(idx_of).astype("Int64")
    out["in_catalog"] = out["item_idx"].notna()
    out = out[["movie_id", "item_idx", "title", "genres", "in_catalog"]]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def degree_stats(items: np.ndarray, n_items: int, split: dict) -> dict:
    """Counts used to decide whether the zero-train-degree rule (D-07) needs revisiting."""
    full_deg = np.bincount(items, minlength=n_items)
    train_deg = np.bincount(split["train_items"], minlength=n_items)
    return {
        "items_with_1_rating_overall": int((full_deg == 1).sum()),
        "items_with_0_train_edges": int((train_deg == 0).sum()),
        "train_edges": len(split["train_items"]),
        "val_edges": len(split["val_items"]),
        "test_edges": len(split["test_items"]),
    }
