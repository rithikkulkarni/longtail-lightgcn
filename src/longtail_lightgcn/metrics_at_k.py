"""Recompute every test metric at K = 5, 10 and 20 from the saved top-20 lists (no training, no GPU).

Each run saved its best-first test top-20 list to <results_dir>/recs/<key>.npz. The first K columns of a
best-first list ARE the top-K list, so slicing recs[:, :K] and calling the repo's own `all_metrics` gives exact
Recall/NDCG/Coverage/Gini/ARP at that K. Nothing is re-implemented here; only the K changes.

Safety check: before anything is written, the K = 20 numbers recomputed here must match the "test" metrics that
training stored in results.jsonl. A mismatch (wrong split files, wrong recs file, changed metric code) stops the
script without writing the CSV.

Output: <results_dir>/metrics_at_k.csv, one row per (run, K). It is derived data: re-run this after new runs
finish. results.jsonl is only read, never modified.

Run:  python -m longtail_lightgcn.metrics_at_k --results_dir results --data_dir data/processed
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from longtail_lightgcn.data.preprocess import load_split
from longtail_lightgcn.metrics import all_metrics, target_mask
from longtail_lightgcn.run_grid import EPOCHS, read_rows

KS = (5, 10, 20)
TOLERANCE = 1e-6  # absolute; the stored numbers were computed on a GPU, these on CPU
METRICS = ("recall", "ndcg", "coverage", "gini", "arp", "target_pop")


def metrics_at_ks(recs: torch.Tensor, target: torch.Tensor, train_degree: torch.Tensor, n_items: int, ks=KS) -> dict:
    """{K: {metric: value}} for the first K columns of a best-first top-N list (N >= max(ks))."""
    out = {}
    for k in ks:
        if k > recs.shape[1]:
            raise ValueError(f"K={k} is larger than the saved list length {recs.shape[1]}")
        m = all_metrics(recs[:, :k], target, train_degree, n_items)
        out[k] = {name: m[f"{name}@{k}"] if f"{name}@{k}" in m else m[name] for name in METRICS}
    return out


def check_against_stored(computed_at_20: dict, stored_test: dict, key: str, tol: float = TOLERANCE) -> float:
    """Largest absolute difference between the recomputed K=20 metrics and the stored ones; raises above tol."""
    stored = {"recall": stored_test["recall@20"], "ndcg": stored_test["ndcg@20"], "coverage": stored_test["coverage"],
              "gini": stored_test["gini"], "arp": stored_test["arp"], "target_pop": stored_test["target_pop"]}
    worst = 0.0
    for name, want in stored.items():
        diff = abs(computed_at_20[name] - want)
        worst = max(worst, diff)
        if diff > tol:
            raise ValueError(f"{key}: recomputed {name}@20 = {computed_at_20[name]!r} but results.jsonl has {want!r} "
                             f"(diff {diff:.2e} > {tol:.0e}); check the split files and recs file")
    return worst


def build_table(results_dir: Path, data_dir: Path, ks=KS, tol: float = TOLERANCE):
    rows = [r for r in read_rows(Path(results_dir) / "results.jsonl") if r["config"]["max_epochs"] == EPOCHS]
    seen, unique = set(), []
    for r in rows:  # a key should appear once; keep the first if a session was ever repeated
        if r["key"] not in seen:
            seen.add(r["key"])
            unique.append(r)

    splits = {}  # split index -> (target mask, train degree, n_items), loaded once
    table, worst = [], 0.0
    for r in unique:
        cfg = r["config"]
        s = cfg["split_seed"]
        if s not in splits:
            split, n_users, n_items = load_split(Path(data_dir) / f"split_{s}.npz")
            target = target_mask([(split["test_users"], split["test_items"])], n_users, n_items, "cpu")
            degree = torch.bincount(torch.as_tensor(split["train_items"]), minlength=n_items)
            splits[s] = (target, degree, n_items)
        target, degree, n_items = splits[s]
        recs = torch.as_tensor(np.load(Path(results_dir) / "recs" / f"{r['key']}.npz")["recs"].astype(np.int64))
        by_k = metrics_at_ks(recs, target, degree, n_items, ks)
        if 20 in by_k:
            worst = max(worst, check_against_stored(by_k[20], r["test"], r["key"], tol))
        for k, m in by_k.items():
            table.append({"key": r["key"], "label": r["label"], "gamma": cfg["gamma"], "beta": cfg["beta"],
                          "use_content": int(cfg["use_content"]), "split": s, "k": k, **m})
    return pd.DataFrame(table), worst, len(unique)


def main(argv=None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out", default=None, help="default: <results_dir>/metrics_at_k.csv")
    args = ap.parse_args(argv)

    df, worst, n_runs = build_table(Path(args.results_dir), Path(args.data_dir))
    out = Path(args.out) if args.out else Path(args.results_dir) / "metrics_at_k.csv"
    df.to_csv(out, index=False)
    print(f"{n_runs} runs x {len(KS)} K values -> {len(df)} rows in {out}")
    print(f"K=20 check passed: recomputed values match results.jsonl (largest difference {worst:.2e})")
    return df


if __name__ == "__main__":
    main()
