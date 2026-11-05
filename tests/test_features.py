import math

import numpy as np
import pandas as pd

from longtail_lightgcn.data.features import build_item_features, load_item_features, parse_year


def test_parse_year_takes_the_last_parenthesized_group():
    assert parse_year("Toy Story (1995)") == 1995
    assert parse_year("Seven (Se7en) (1995)") == 1995  # alternate title in parentheses comes first
    assert math.isnan(parse_year("No Year Here"))
    assert math.isnan(parse_year("Odd (19955)"))


def tiny_items():
    # rows deliberately out of item_idx order; movie 40 is not in the catalog
    return pd.DataFrame({
        "movie_id": [30, 10, 40, 20],
        "item_idx": [2, 0, np.nan, 1],
        "title": ["C (2000)", "A (1990)", "Unrated (1980)", "B (no year)"],
        "genres": ["Drama", "Comedy|Romance", "Western", "Comedy"],
        "in_catalog": [True, True, False, True],
    })


def test_features_hand_computed():
    feats, genres = build_item_features(tiny_items())
    # genres come from ALL movies (sorted), so the feature width does not depend on the catalog
    assert genres == ["Comedy", "Drama", "Romance", "Western"]
    assert feats.shape == (3, 5) and feats.dtype == np.float32  # catalog items only, ordered by item_idx
    # item 0 = A: Comedy|Romance, year 1990 -> 0
    np.testing.assert_array_equal(feats[0], [1, 0, 1, 0, 0.0])
    # item 1 = B: Comedy, no year -> filled with the mean of 1990 and 2000 = 1995 -> 0.5
    np.testing.assert_array_equal(feats[1], [1, 0, 0, 0, 0.5])
    # item 2 = C: Drama, year 2000 -> 1 (the unrated 1980 movie does not stretch the scale)
    np.testing.assert_array_equal(feats[2], [0, 1, 0, 0, 1.0])


def test_single_year_does_not_divide_by_zero():
    items = tiny_items()
    items["title"] = "Same (1995)"
    feats, _ = build_item_features(items)
    assert np.isfinite(feats).all() and (feats[:, -1] == 0).all()


def test_load_item_features_roundtrips_through_csv(tmp_path):
    items = tiny_items()
    items["item_idx"] = items["item_idx"].astype("Int64")
    path = tmp_path / "items.csv"
    items.to_csv(path, index=False)
    np.testing.assert_array_equal(load_item_features(path), build_item_features(tiny_items())[0])
