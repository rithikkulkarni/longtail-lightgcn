"""Item content features for Intervention C (DESIGN.md D-28): multi-hot genres + normalized release year."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

_YEAR = re.compile(r"\((\d{4})\)\s*$")  # the year is the LAST parenthesized group, e.g. "Seven (Se7en) (1995)"


def parse_year(title: str) -> float:
    """Release year from a title ending in "(YYYY)"; NaN when there is none."""
    m = _YEAR.search(title)
    return float(m.group(1)) if m else float("nan")


def build_item_features(items: pd.DataFrame):
    """Feature matrix for the catalog items, rows ordered by item index.

    `items` is the table written by save_items_csv (columns item_idx, title, genres, in_catalog).
    Returns (features, genre_names): features is float32 (n_catalog_items, n_genres + 1), made of
      - multi-hot over all genres that occur in the table (a movie can have several),
      - release year min-max scaled to [0, 1] over the catalog; a missing year is filled with the
        mean catalog year before scaling.
    """
    genre_names = sorted({g for s in items["genres"] for g in s.split("|")})
    cat = items[items["in_catalog"]].sort_values("item_idx")

    genre_idx = {g: j for j, g in enumerate(genre_names)}
    multi_hot = np.zeros((len(cat), len(genre_names)), dtype=np.float32)
    for row, s in enumerate(cat["genres"]):
        for g in s.split("|"):
            multi_hot[row, genre_idx[g]] = 1.0

    years = cat["title"].map(parse_year).to_numpy(dtype=np.float64)
    years = np.where(np.isnan(years), np.nanmean(years), years)
    span = years.max() - years.min()
    year = ((years - years.min()) / span if span > 0 else np.zeros_like(years)).astype(np.float32)

    return np.concatenate([multi_hot, year[:, None]], axis=1), genre_names


def load_item_features(items_csv: Path) -> np.ndarray:
    """Read items.csv (written by scripts/prepare_data.py) and return the feature matrix."""
    items = pd.read_csv(items_csv, keep_default_na=False, dtype={"genres": str, "title": str})
    items["item_idx"] = pd.to_numeric(items["item_idx"], errors="coerce")
    items["in_catalog"] = items["in_catalog"].astype(str).str.lower().eq("true")
    return build_item_features(items)[0]
