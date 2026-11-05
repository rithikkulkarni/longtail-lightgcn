"""Experiment grid runner (DESIGN.md D-29).

Stage 1 (split 0): baseline, C alone, A alone at each gamma in DIALS, B alone at each beta in DIALS.
Stage 2: the 8 on/off combinations of A, B, C at the chosen gamma*/beta* (results/dials.json) on 2 splits.

Run:  python -m longtail_lightgcn.run_grid --stage 1 --results_dir results     (next unfinished run)
      python -m longtail_lightgcn.run_grid --stage 2 --max_runs 3
      python -m longtail_lightgcn.run_grid --stage 1 --list                    (what is done / pending)

Each finished run appends one line to <results_dir>/results.jsonl and saves its test top-K lists to
<results_dir>/recs/<key>.npz. A run already in results.jsonl is skipped, so sessions can be repeated or resumed,
and a run shared by both stages (e.g. split 0 of A) is only ever trained once.
"""
import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from longtail_lightgcn.config import TrainConfig
from longtail_lightgcn.data.features import load_item_features
from longtail_lightgcn.data.preprocess import load_split
from longtail_lightgcn.metrics import all_metrics, recommend_top_k, target_mask
from longtail_lightgcn.run_baseline import git_commit
from longtail_lightgcn.train import train

DIALS = (0.25, 0.5, 0.75)  # gamma = 1 and beta = 1 were dropped (D-29)
EPOCHS = 200  # fixed budget for every run (D-29)
EVAL_EVERY = 10
NO_EARLY_STOP = 10**9  # patience so large that the best-validation checkpoint is kept without ever stopping
STAGE2_SPLITS = (0, 1)
# on/off patterns (A, B, C) in the order baseline, A, B, C, AB, AC, BC, ABC
COMBOS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]


@dataclass(frozen=True)
class RunSpec:
    gamma: float = 0.0  # intervention A is on when > 0
    beta: float = 0.0  # intervention B is on when > 0
    use_content: bool = False  # intervention C
    split: int = 0  # saved split index; also the model seed (D-29)

    @property
    def key(self) -> str:
        return f"g{self.gamma:g}_b{self.beta:g}_c{int(self.use_content)}_s{self.split}"

    @property
    def label(self) -> str:
        on = ("A" if self.gamma > 0 else "") + ("B" if self.beta > 0 else "") + ("C" if self.use_content else "")
        return on or "baseline"


def stage1_specs() -> list:
    return ([RunSpec(), RunSpec(use_content=True)]
            + [RunSpec(gamma=g) for g in DIALS] + [RunSpec(beta=b) for b in DIALS])


def stage2_specs(gamma: float, beta: float, splits=STAGE2_SPLITS) -> list:
    return [RunSpec(gamma * a, beta * b, bool(c), s) for s in splits for a, b, c in COMBOS]


def read_rows(path: Path) -> list:
    if not Path(path).exists():
        return []
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"warning: skipping unreadable line {i} of {path}")
    return rows


def append_row(path: Path, row: dict) -> None:
    """Append one JSON line and force it to disk, so a finished run survives a crash right after."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def done_keys(rows: list, budget: int = EPOCHS) -> set:
    """Keys of finished runs at this epoch budget (a smoke run with a different budget never counts)."""
    return {r["key"] for r in rows if r["config"]["max_epochs"] == budget}


def load_dials(results_dir: Path) -> dict:
    path = Path(results_dir) / "dials.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Finish stage 1, run `python -m longtail_lightgcn.select_dials`, "
                         "review the choice, and commit dials.json before stage 2.")
    return json.loads(path.read_text())


def run_one(spec: RunSpec, data_dir: Path, results_dir: Path, device: str, max_epochs: int, features) -> dict:
    cfg = TrainConfig(gamma=spec.gamma, beta=spec.beta, use_content=spec.use_content, split_seed=spec.split,
                      seed=spec.split, max_epochs=max_epochs, eval_every=EVAL_EVERY, patience=NO_EARLY_STOP)
    split, n_users, n_items = load_split(Path(data_dir) / f"split_{spec.split}.npz")
    t0 = time.time()
    result = train(split, n_users, n_items, cfg, device=device, item_features=features)
    train_seconds = time.time() - t0

    # metrics from the best-validation weights; the test set is evaluated once per run
    model = result.model
    model.eval()
    with torch.no_grad():
        user_final, item_final = model.propagate(result.adj)
    train_p = (split["train_users"], split["train_items"])
    val_p = (split["val_users"], split["val_items"])
    test_p = (split["test_users"], split["test_items"])
    degree = torch.bincount(torch.as_tensor(split["train_items"]), minlength=n_items).to(device)
    val_recs = recommend_top_k(user_final, item_final, cfg.k, [train_p])
    test_recs = recommend_top_k(user_final, item_final, cfg.k, [train_p, val_p])
    val = all_metrics(val_recs, target_mask([val_p], n_users, n_items, device), degree, n_items)
    test = all_metrics(test_recs, target_mask([test_p], n_users, n_items, device), degree, n_items)

    recs_dir = Path(results_dir) / "recs"
    recs_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(recs_dir / f"{spec.key}.npz", recs=test_recs.cpu().numpy().astype(np.int16))  # items < 32768

    return {
        "key": spec.key,
        "label": spec.label,
        "config": asdict(cfg),
        "git_commit": git_commit(),
        "device": device,
        "best_epoch": result.best_epoch,
        "train_seconds": train_seconds,
        "val": val,
        "test": test,
        "history": result.history,
    }


def main(argv=None) -> list:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--max_runs", type=int, default=1, help="how many unfinished runs to do this session (D-29: 1)")
    ap.add_argument("--run_id", default=None, help="run one specific key instead of the next unfinished ones")
    ap.add_argument("--max_hours", type=float, default=None,
                    help="do not start a run that would not fit in this many hours (judged by the previous run)")
    ap.add_argument("--max_epochs", type=int, default=EPOCHS, help=f"smoke tests only; the grid uses {EPOCHS}")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--list", action="store_true", help="show done/pending runs and exit")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    results_path = results_dir / "results.jsonl"
    if args.stage == 1:
        specs = stage1_specs()
    else:
        dials = load_dials(results_dir)
        specs = stage2_specs(dials["gamma"], dials["beta"])
    done = done_keys(read_rows(results_path), args.max_epochs)
    pending = [s for s in specs if s.key not in done]

    if args.list:
        for s in specs:
            print(f"{'done   ' if s.key in done else 'pending'}  {s.key:<26} {s.label:<9} split {s.split}")
        print(f"{len(specs) - len(pending)} of {len(specs)} runs done in stage {args.stage}")
        return []
    if args.run_id is not None:
        pending = [s for s in pending if s.key == args.run_id]
        if not pending:
            print(f"{args.run_id!r} is already done or is not part of stage {args.stage}; nothing to do")
            return []
    pending = pending[: args.max_runs]
    if not pending:
        print(f"Stage {args.stage} is complete: all {len(specs)} runs are in {results_path}")
        return []
    if args.max_epochs != EPOCHS:
        print(f"WARNING: --max_epochs {args.max_epochs} is not the grid budget ({EPOCHS}); smoke run only")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    features = load_item_features(Path(args.data_dir) / "items.csv") if any(s.use_content for s in pending) else None
    results_dir.mkdir(parents=True, exist_ok=True)

    written, started, last_hours = [], time.time(), None
    for spec in pending:
        elapsed = (time.time() - started) / 3600
        if args.max_hours is not None and last_hours is not None and elapsed + last_hours > args.max_hours:
            print(f"Stopping: {elapsed:.2f} h used and the last run took {last_hours:.2f} h (limit {args.max_hours} h)")
            break
        print(f"=== run {spec.key} ({spec.label}, split {spec.split}) on {device} ===")
        t0 = time.time()
        row = run_one(spec, args.data_dir, results_dir, device, args.max_epochs, features)
        append_row(results_path, row)
        last_hours = (time.time() - t0) / 3600
        written.append(row)
        print(f"done {spec.key}: best epoch {row['best_epoch']} | test recall {row['test']['recall@20']:.4f} | "
              f"test gini {row['test']['gini']:.4f} | {last_hours:.2f} h")
    return written


if __name__ == "__main__":
    main()
