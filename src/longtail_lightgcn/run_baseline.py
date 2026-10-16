"""Train baseline LightGCN on one saved split and write results to disk.

Run:  python -m longtail_lightgcn.run_baseline --split_seed 0 --seed 0 --out_dir results/baseline
Every TrainConfig field can be overridden on the command line (e.g. --lr 0.005 --max_epochs 300).
"""
import argparse
import json
import subprocess
from dataclasses import asdict, fields
from pathlib import Path

import torch

from longtail_lightgcn.config import TrainConfig
from longtail_lightgcn.data.preprocess import load_split
from longtail_lightgcn.train import evaluate, train


def git_commit() -> str:
    """Hash of the code that produced the run, for traceability (empty if not in a git repo)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="results/baseline")
    ap.add_argument("--device", default="auto", help="'auto' uses CUDA when available")
    default_cfg = TrainConfig()
    for f in fields(TrainConfig):  # one flag per config field, defaulting to the dataclass value
        ap.add_argument(f"--{f.name}", type=type(getattr(default_cfg, f.name)), default=None)
    return ap


def main(argv=None) -> dict:
    args = build_parser().parse_args(argv)
    overrides = {f.name: getattr(args, f.name) for f in fields(TrainConfig) if getattr(args, f.name) is not None}
    cfg = TrainConfig(**overrides)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    split, n_users, n_items = load_split(Path(args.data_dir) / f"split_{cfg.split_seed}.npz")
    print(f"device={device}  users={n_users}  items={n_items}  config={asdict(cfg)}")

    result = train(split, n_users, n_items, cfg, device=device)
    # the test set is touched exactly once, with the best (validation-selected) weights
    test_recall = evaluate(result.model, result.adj, split, cfg.k, "test")
    print(f"best epoch {result.best_epoch} | val recall@{cfg.k} {result.best_val_recall:.4f} | test recall@{cfg.k} {test_recall:.4f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": asdict(cfg),
        "git_commit": git_commit(),
        "device": device,
        "best_epoch": result.best_epoch,
        f"val_recall@{cfg.k}": result.best_val_recall,
        f"test_recall@{cfg.k}": test_recall,
        "history": result.history,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    torch.save(result.model.state_dict(), out / "model.pt")
    return summary


if __name__ == "__main__":
    main()
