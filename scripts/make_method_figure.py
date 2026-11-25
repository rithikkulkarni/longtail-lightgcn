"""Draw the method overview figure: the LightGCN pipeline and where interventions A, B and C act.

Output: report/figures/method_overview.pdf and .png.   Run:  python scripts/make_method_figure.py
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

GRAY, GRAY_EDGE = "#f2f1ec", "#8a887f"
GREEN, GREEN_EDGE = "#e5f1da", "#4f7f2a"


def box(ax, x, y, w, h, title, body, added=False):
    """Draw one titled stage box; added=True uses the intervention colour."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=GREEN if added else GRAY, edgecolor=GREEN_EDGE if added else GRAY_EDGE,
                                linewidth=1.3 if added else 0.9))
    ax.text(x + w / 2, y + h - 0.27, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(x + w / 2, y + (h - 0.45) / 2, body, ha="center", va="center", fontsize=8.5, linespacing=1.5)


def arrow(ax, start, end, added=False):
    """Draw an arrow between two points; added=True uses the intervention colour."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=1.0,
                                 color=GREEN_EDGE if added else "#444444", shrinkA=0, shrinkB=0))


def main(argv=None) -> None:
    """Command-line entry point: draw and save the method overview figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default="report/figures")
    args = ap.parse_args(argv)

    fig, ax = plt.subplots(figsize=(7.4, 3.1))
    ax.set_xlim(0, 14.8)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # baseline LightGCN pipeline, left to right
    y, h = 2.35, 1.7
    box(ax, 0.1, y, 2.7, h, "Layer 0", "ID embeddings\n$e_u^{(0)},\\; e_i^{(0)}$")
    box(ax, 3.4, y, 2.9, h, "Propagation", "$E^{(k+1)} = \\hat{A}\\,E^{(k)}$\n$k = 0, 1, 2$")
    box(ax, 6.9, y, 2.2, h, "Layer mean", "$e = \\frac{1}{4}\\sum_k e^{(k)}$")
    box(ax, 9.7, y, 2.0, h, "Score", "$y_{ui} = e_u^{\\top} e_i$")
    box(ax, 12.3, y, 2.4, h, "BPR loss", "$-\\ln\\sigma(y_{ui}-y_{uj})$\n+ L2 on layer 0")
    for a, b in ((2.8, 3.4), (6.3, 6.9), (9.1, 9.7), (11.7, 12.3)):
        arrow(ax, (a, y + h / 2), (b, y + h / 2))

    # the three interventions, each attached to the stage it changes
    box(ax, 0.1, 0.1, 3.5, 1.65, "C: content initialization", "$e_i^{(0)} = e_i^{ID} + f_\\theta(x_i)$\n"
        "$x_i$: genres, year", added=True)
    arrow(ax, (1.45, 1.75), (1.45, y), added=True)
    box(ax, 3.2, 4.5, 3.3, 1.65, "A: degree discount", "item update scaled by\n$|N(i)|^{-\\gamma}$", added=True)
    arrow(ax, (4.85, 4.5), (4.85, y + h), added=True)
    box(ax, 11.7, 4.5, 3.0, 1.65, "B: loss re-weighting", "BPR term times\n$w(i) = pop(i)^{-\\beta}$", added=True)
    arrow(ax, (13.5, 4.5), (13.5, y + h), added=True)

    ax.text(9.0, 0.9, "Gray: baseline LightGCN\nGreen: interventions (each on/off independently)",
            ha="center", va="center", fontsize=8.5, color="#444444", linespacing=1.6)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "method_overview.pdf")
    fig.savefig(out / "method_overview.png", dpi=200)
    plt.close(fig)
    print(f"wrote {out / 'method_overview.pdf'}")


if __name__ == "__main__":
    main()
