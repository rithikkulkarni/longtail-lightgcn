"""Build the report's figures and LaTeX tables from the saved results (no torch needed).

Inputs (all read-only):  results/metrics_at_k.csv   (python -m longtail_lightgcn.metrics_at_k)
                         results/results.jsonl       (validation numbers for the dial sweep)
                         results/dials.json          (gamma*, beta*)

Outputs, for each K in 5, 10, 20:
    report/figures/pareto_k{K}.pdf / .png   Recall@K vs Gini trade-off (test set, mean and per-split points)
    report/tables/results_k{K}.tex          the 8 stage-2 configs: Recall, NDCG, Coverage, Gini, ARP
    report/tables/delta_k{K}.tex            % change in Recall and Gini vs the baseline on the same split
and once:
    report/tables/dial_sweep.tex            stage 1 dial sweep (validation set, split 0, K = 20)

Only splits on which all 8 stage-2 configs have finished are used, so a half-finished split never skews a mean.
Re-run after new runs finish; every file is overwritten.

Run:  python scripts/make_figures.py [--results_dir results] [--out_dir report] [--no_std]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ["baseline", "A", "B", "C", "AB", "AC", "BC", "ABC"]
KS = (5, 10, 20)
BLUE, LIGHT_BLUE, RED, LIGHT_RED = "#1f77b4", "#9ecae1", "#d62728", "#f0a3a3"
# column -> (header, decimals, higher_is_better)
COLUMNS = {"recall": ("Recall@{k}", 4, True), "ndcg": ("NDCG@{k}", 4, True), "coverage": ("Coverage", 3, True),
           "gini": ("Gini", 3, False), "arp": ("ARP", 0, False)}


# ---------------------------------------------------------------- data

def load_stage2(results_dir: Path):
    """Stage-2 rows (the 8 on/off configs at gamma*, beta*) restricted to splits where all 8 finished."""
    df = pd.read_csv(Path(results_dir) / "metrics_at_k.csv")
    dials = json.loads((Path(results_dir) / "dials.json").read_text())
    keep = df["gamma"].isin([0.0, dials["gamma"]]) & df["beta"].isin([0.0, dials["beta"]])
    df = df[keep].copy()
    counts = df[df["k"] == 20].groupby("split")["label"].nunique()
    complete = sorted(counts[counts == len(ORDER)].index)
    skipped = sorted(set(counts.index) - set(complete))
    if not complete:
        raise SystemExit("no split has all 8 stage-2 configs finished yet")
    if skipped:
        print(f"note: splits {skipped} are incomplete and left out of every mean")
    return df[df["split"].isin(complete)], complete, dials


def summarize(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per-config mean and std (over splits) of every metric at K."""
    sub = df[df["k"] == k]
    g = sub.groupby("label")[list(COLUMNS)].agg(["mean", "std"])
    return g.reindex(ORDER)


def percent_changes(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per split: 100 * (config - baseline) / baseline for Recall and Gini; then mean/std over splits."""
    sub = df[df["k"] == k].pivot(index="split", columns="label", values=["recall", "gini"])
    out = {}
    for metric in ("recall", "gini"):
        base = sub[metric]["baseline"]
        for label in ORDER[1:]:
            out[(label, metric)] = 100 * (sub[metric][label] - base) / base
    pc = pd.DataFrame(out)
    return pd.DataFrame({"mean": pc.mean(), "std": pc.std(ddof=1)})


def pareto_frontier(points: dict) -> list:
    """Labels not dominated by any other point (lower Gini AND higher Recall is better), in Gini order."""
    names = list(points)
    keep = []
    for p in names:
        gp, rp = points[p]
        dominated = any(points[q][0] <= gp and points[q][1] >= rp and points[q] != points[p] for q in names if q != p)
        if not dominated:
            keep.append(p)
    return sorted(keep, key=lambda n: points[n][0])


# ---------------------------------------------------------------- LaTeX helpers

def num(x: float, d: int) -> str:
    """Format a number with d decimals."""
    return f"{x:.{d}f}"


def pm(mean: float, std: float, d: int, with_std: bool) -> str:
    """LaTeX cell 'mean +/- std', or the mean alone when std is hidden or undefined."""
    if not with_std or np.isnan(std):
        return num(mean, d)
    return f"{num(mean, d)}{{\\scriptsize\\,$\\pm$\\,{num(std, d)}}}"


def signed(x: float, d: int = 1) -> str:
    """Format a percent change with an explicit plus or minus sign."""
    s = f"{abs(x):.{d}f}"
    return ("$-$" if x < 0 and float(s) != 0 else "+") + s


def write_table(path: Path, body: str) -> None:
    """Write a LaTeX table to disk, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")


# ---------------------------------------------------------------- outputs

def results_table(summary: pd.DataFrame, k: int, n: int, with_std: bool) -> str:
    """LaTeX table of all five metrics for the 8 configurations at one K; best value per column in bold."""
    best = {}
    for col, (_, d, higher) in COLUMNS.items():
        means = summary[(col, "mean")].round(d)
        best[col] = means.max() if higher else means.min()
    head = " & ".join(["Config"] + [h.format(k=k) for h, _, _ in COLUMNS.values()])
    lines = []
    for label in ORDER:
        cells = [label.replace("baseline", "Baseline")]
        for col, (_, d, _) in COLUMNS.items():
            m, s = summary.loc[label, (col, "mean")], summary.loc[label, (col, "std")]
            cell = pm(m, s, d, with_std)
            cells.append(f"\\textbf{{{cell}}}" if round(m, d) == best[col] else cell)
        lines.append(" & ".join(cells) + r" \\")
    std_note = ", mean $\\pm$ std" if with_std else ", mean"
    return "\n".join([
        r"\begin{table}[h]",
        f"  \\caption{{Results on ML-1M (test set{std_note} over {n} splits, $K={k}$). "
        r"Bold marks the best value in each column.}",
        f"  \\label{{tab:results_k{k}}}",
        r"  \centering",
        r"  \begin{tabular}{lccccc}",
        r"    \toprule",
        f"    {head} \\\\",
        r"    \midrule",
        *[f"    {ln}" for ln in lines],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def delta_table(pc: pd.DataFrame, k: int, n: int, with_std: bool) -> str:
    """LaTeX table of percent change in Recall and Gini against the baseline at one K."""
    lines = []
    for label in ORDER[1:]:
        cells = [label]
        for metric in ("recall", "gini"):
            m, s = pc.loc[(label, metric), "mean"], pc.loc[(label, metric), "std"]
            cell = signed(m) + r"\%"
            if with_std and not np.isnan(s):
                cell += f"{{\\scriptsize\\,$\\pm$\\,{s:.1f}}}"
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")
    return "\n".join([
        r"\begin{table}[h]",
        f"  \\caption{{Percent change in Recall@{k} and Gini relative to the baseline on the same split "
        f"(test set, mean over {n} splits" + (r", $\pm$ std" if with_std else "") + ").}",
        f"  \\label{{tab:delta_k{k}}}",
        r"  \centering",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        f"    Config & Recall@{k} ($\\Delta$) & Gini ($\\Delta$) \\\\",
        r"    \midrule",
        *[f"    {ln}" for ln in lines],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def dial_sweep_table(results_dir: Path, dials: dict) -> str:
    """LaTeX table of the stage 1 dial sweep (validation, split 0); chosen dials in bold."""
    rows = {}
    for line in (Path(results_dir) / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["config"]["max_epochs"] == 200 and r["config"]["split_seed"] == 0:
                rows[r["key"]] = r["val"]

    def entry(label, key, selected=False):
        v = rows[key]
        cells = [label, f"{v['recall@20']:.4f}", f"{v['gini']:.3f}"]
        return " & ".join(f"\\textbf{{{c}}}" if selected else c for c in cells) + r" \\"

    gammas = sorted({float(k.split("_")[0][1:]) for k in rows if k.endswith("_b0_c0_s0")} - {0.0})
    betas = sorted({float(k.split("_")[1][1:]) for k in rows if k.startswith("g0_") and k.endswith("_c0_s0")} - {0.0})
    lines = [entry("Baseline", "g0_b0_c0_s0")]
    lines += [entry(f"A ($\\gamma={g:g}$)", f"g{g:g}_b0_c0_s0", g == dials["gamma"]) for g in gammas]
    lines += [entry(f"B ($\\beta={b:g}$)", f"g0_b{b:g}_c0_s0", b == dials["beta"]) for b in betas]
    lines += [entry("C", "g0_b0_c1_s0")]
    return "\n".join([
        r"\begin{table}[h]",
        r"  \caption{Stage 1 dial sweep (validation set, split 0, $K=20$). Bold marks the dial values chosen by the "
        r"rule (lowest validation Gini with validation Recall@20 within 5\% of the baseline). Only the values "
        r"0.25--0.75 were candidates; $\gamma=1$ and $\beta=1$ were run afterwards as sweep end-points.}",
        r"  \label{tab:dial_sweep}",
        r"  \centering",
        r"  \begin{tabular}{lcc}",
        r"    \toprule",
        r"    Config & Recall@20 & Gini \\",
        r"    \midrule",
        *[f"    {ln}" for ln in lines],
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])


def label_offset(label: str, points: dict) -> tuple:
    """Label above-right by default; below-right when another mean sits close by and is at least as high."""
    xs, ys = [p[0] for p in points.values()], [p[1] for p in points.values()]
    xr, yr = (max(xs) - min(xs)) or 1.0, (max(ys) - min(ys)) or 1.0
    gx, gy = points[label]
    for other, (ox, oy) in points.items():
        if other != label and abs(ox - gx) / xr < 0.06 and abs(oy - gy) / yr < 0.06 and oy >= gy:
            return (5, -13)
    return (5, 5)


def pareto_figure(df: pd.DataFrame, k: int, splits: list, out_base: Path) -> None:
    """Plot Recall@K against Gini (means and per-split points) with the Pareto frontier."""
    sub = df[df["k"] == k]
    means = sub.groupby("label")[["gini", "recall"]].mean().reindex(ORDER)
    points = {lab: (means.loc[lab, "gini"], means.loc[lab, "recall"]) for lab in ORDER}
    frontier = pareto_frontier(points)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot([points[f][0] for f in frontier], [points[f][1] for f in frontier], "--", color="gray", lw=1.1,
            label="Pareto frontier (means)", zorder=1)
    for lab in ORDER:
        small = sub[sub["label"] == lab]
        is_base = lab == "baseline"
        c_big, c_small = (RED, LIGHT_RED) if is_base else (BLUE, LIGHT_BLUE)
        gx, gy = points[lab]
        for _, r in small.iterrows():  # thin line from the mean to each split's own point
            ax.plot([gx, r["gini"]], [gy, r["recall"]], "-", color=c_small, lw=0.9, zorder=2)
        ax.scatter(small["gini"], small["recall"], s=16, color=c_small, zorder=3)
        ax.scatter([gx], [gy], s=62, color=c_big, zorder=4)
        ax.annotate("Baseline" if is_base else lab, (gx, gy), textcoords="offset points",
                    xytext=label_offset(lab, points), fontsize=9)
    ax.set_xlabel("Gini of item exposure (lower = less popularity bias)")
    ax.set_ylabel(f"Recall@{k} (higher = more accurate)")
    ax.set_title(f"Accuracy vs. popularity bias (test set, K={k}, {len(splits)} splits)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main(argv=None) -> None:
    """Command-line entry point: write every result table and figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="report")
    ap.add_argument("--no_std", action="store_true", help="tables show means only (as on the slides)")
    args = ap.parse_args(argv)
    out, with_std = Path(args.out_dir), not args.no_std

    df, splits, dials = load_stage2(Path(args.results_dir))
    n = len(splits)
    print(f"stage 2: splits {splits} (n={n}), gamma*={dials['gamma']}, beta*={dials['beta']}")
    for k in KS:
        write_table(out / "tables" / f"results_k{k}.tex", results_table(summarize(df, k), k, n, with_std))
        write_table(out / "tables" / f"delta_k{k}.tex", delta_table(percent_changes(df, k), k, n, with_std))
        pareto_figure(df, k, splits, out / "figures" / f"pareto_k{k}")
    write_table(out / "tables" / "dial_sweep.tex", dial_sweep_table(Path(args.results_dir), dials))
    print(f"wrote {len(KS)} figures and {2 * len(KS) + 1} tables under {out}/")


if __name__ == "__main__":
    main()
