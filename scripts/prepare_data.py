"""Download ML-1M, build the edge list, and write N splits plus metadata to data/processed/."""
import argparse
from pathlib import Path

from longtail_lightgcn.data.download import download_ml1m
from longtail_lightgcn.data.preprocess import (
    build_edges,
    degree_stats,
    load_movies,
    load_ratings,
    make_split,
    save_id_maps,
    save_items_csv,
    save_split,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--n-splits", type=int, default=5, help="one split per seed 0..n-1 (D-02)")
    args = ap.parse_args()

    data_dir = download_ml1m(Path(args.raw_dir))
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    users, items, user_ids, item_ids = build_edges(ratings)
    n_users, n_items = len(user_ids), len(item_ids)

    out = Path(args.out_dir)
    save_id_maps(out / "id_maps.npz", user_ids, item_ids)
    save_items_csv(out / "items.csv", movies, item_ids)

    print(f"users={n_users}  items in catalog={n_items} (movies.dat has {len(movies)})  edges={len(users)}")
    for seed in range(args.n_splits):
        split = make_split(users, items, seed)
        save_split(out / f"split_{seed}.npz", split, n_users, n_items)
        print(f"split_{seed}: {degree_stats(items, n_items, split)}")


if __name__ == "__main__":
    main()
