"""Choose gamma* and beta* from the stage 1 runs (DESIGN.md D-29).

Rule: among the dial values, take the one with the lowest VALIDATION Gini whose VALIDATION Recall@20 is within
`tolerance` (relative) of the stage 1 baseline's. If none qualifies, fall back to the weakest dial (0.25) and warn.
Only validation numbers are read; the test metrics in the same rows are never looked at.

Run:  python -m longtail_lightgcn.select_dials --results_dir results [--tolerance 0.05] [--no_write]
"""
import argparse
import json
from pathlib import Path

from longtail_lightgcn.run_grid import DIALS, EPOCHS, RunSpec, read_rows, stage1_specs

TOLERANCE = 0.05
RECALL = "recall@20"


def select_dials(rows: list, tolerance: float = TOLERANCE, budget: int = EPOCHS) -> dict:
    by_key = {r["key"]: r for r in rows if r["config"]["max_epochs"] == budget}
    missing = [s.key for s in stage1_specs() if s.key not in by_key]
    if missing:
        raise ValueError(f"stage 1 is not finished; missing runs: {missing}")

    baseline_recall = by_key[RunSpec().key]["val"][RECALL]
    floor = (1 - tolerance) * baseline_recall
    out = {"tolerance": tolerance, "baseline_val_recall": baseline_recall, "fallback": {}, "table": {}}
    for name, make in (("gamma", lambda d: RunSpec(gamma=d)), ("beta", lambda d: RunSpec(beta=d))):
        table = []
        for d in DIALS:
            val = by_key[make(d).key]["val"]
            table.append({"dial": d, "val_recall": val[RECALL], "val_gini": val["gini"],
                          "within_tolerance": val[RECALL] >= floor})
        ok = [(t["val_gini"], t["dial"]) for t in table if t["within_tolerance"]]
        out[name] = min(ok)[1] if ok else DIALS[0]  # lowest Gini; ties go to the smaller dial
        out["fallback"][name] = not ok
        out["table"][name] = table
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE)
    ap.add_argument("--no_write", action="store_true", help="print only; do not write dials.json")
    args = ap.parse_args(argv)

    results_dir = Path(args.results_dir)
    choice = select_dials(read_rows(results_dir / "results.jsonl"), args.tolerance)
    print(f"baseline val {RECALL}: {choice['baseline_val_recall']:.4f}  "
          f"(floor at {args.tolerance:.0%} tolerance: {(1 - args.tolerance) * choice['baseline_val_recall']:.4f})")
    for name in ("gamma", "beta"):
        print(f"\n{name}:")
        for t in choice["table"][name]:
            mark = "<- chosen" if t["dial"] == choice[name] else ""
            print(f"  {t['dial']:<5} val recall {t['val_recall']:.4f}  val gini {t['val_gini']:.4f}  "
                  f"{'ok  ' if t['within_tolerance'] else 'FAIL'} {mark}")
        if choice["fallback"][name]:
            print(f"  WARNING: no {name} value is within tolerance; fell back to {DIALS[0]}")
    print(f"\nchosen: gamma* = {choice['gamma']}, beta* = {choice['beta']}")
    if not args.no_write:
        path = results_dir / "dials.json"
        path.write_text(json.dumps(choice, indent=2))
        print(f"wrote {path}")
    return choice


if __name__ == "__main__":
    main()
