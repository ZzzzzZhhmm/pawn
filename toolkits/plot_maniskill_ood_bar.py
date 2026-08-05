import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "analysis_outputs" / "derived_csvs" / "maniskill_ood_avg_bar_values.csv"
OUT_PNG = ROOT / "analysis_outputs" / "maniskill_ood_avg_bar.png"
OUT_PDF = ROOT / "analysis_outputs" / "maniskill_ood_avg_bar.pdf"


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["pi0"] = float(row["pi0"])
            row["pi05"] = float(row["pi05"])
            rows.append(row)
    return rows


def style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13,
            "axes.titlesize": 17,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
        }
    )


def annotate_bars(ax, bars):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.45,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )


def main():
    style()
    rows = load_rows(DATA_CSV)

    methods = [r["method"] for r in rows]
    pi0 = np.array([r["pi0"] for r in rows], dtype=float)
    pi05 = np.array([r["pi05"] for r in rows], dtype=float)

    x = np.arange(len(methods))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.3, 4.35), constrained_layout=True)

    color_pi0 = "#5B7DB1"
    color_pi05 = "#8B304D"
    edge = "#333333"

    bars0 = ax.bar(
        x - width / 2,
        pi0,
        width,
        color=color_pi0,
        edgecolor=edge,
        linewidth=0.8,
        label=r"$\pi_0$",
        zorder=3,
    )
    bars1 = ax.bar(
        x + width / 2,
        pi05,
        width,
        color=color_pi05,
        edgecolor=edge,
        linewidth=0.8,
        label=r"$\pi_{0.5}$",
        zorder=3,
    )

    annotate_bars(ax, bars0)
    annotate_bars(ax, bars1)

    ax.set_ylabel("OOD Avg. Success Rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(30, 66.5)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        fancybox=True,
        borderpad=0.35,
        handlelength=1.4,
    )

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")


if __name__ == "__main__":
    main()
