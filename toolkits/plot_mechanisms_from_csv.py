import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


INPUT_DIR = Path("log/analysis_inputs")
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_BASENAME = "object_mechanism_corrected"
PANEL_LABELS = ["(a)", "(b)", "(c)"]

SERIES = [
    {
        "file": "chunk_credit_gain.csv",
        "title": "Chunk Credit Gain",
        "ylabel": "Neutral Loss - Chunk Loss",
    },
    {
        "file": "pair_separation_corrected.csv",
        "title": "Local Pair Separation",
        "ylabel": "Precomputed Pair Gap",
    },
    {
        "file": "block_selectivity_corrected.csv",
        "title": "Block Selectivity",
        "ylabel": "Normalized Selectivity",
    },
]

MAIN_COLOR = "#8B1E3F"


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10.5,
            "axes.labelsize": 12.5,
            "axes.titlesize": 15,
            "legend.fontsize": 10,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.38,
            "grid.linestyle": "--",
            "grid.color": "#D8DCE5",
            "axes.facecolor": "white",
            "axes.linewidth": 1.0,
            "figure.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def _lighten_color(color: str, amount: float = 0.72) -> tuple[float, float, float]:
    base = np.array(to_rgb(color), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    mixed = base * (1.0 - amount) + white * amount
    return tuple(np.clip(mixed, 0.0, 1.0))


def _decorate_axis(ax, panel_label: str):
    ax.text(
        0.02,
        0.98,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#2F3440",
    )
    ax.tick_params(axis="both", which="both", direction="out", width=0.9, color="#5A5F6A")
    ax.set_axisbelow(True)


def _read_series(path: Path):
    steps = []
    raw = []
    smooth = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(float(row["step"])))
            raw_val = row.get("raw_value", "")
            smooth_val = row.get("smooth_value", "")
            raw.append(np.nan if raw_val == "" else float(raw_val))
            smooth.append(np.nan if smooth_val == "" else float(smooth_val))
    return np.array(steps), np.array(raw, dtype=float), np.array(smooth, dtype=float)


def main():
    _style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 3.65))
    raw_color = _lighten_color(MAIN_COLOR, 0.72)

    for idx, (ax, cfg) in enumerate(zip(axes, SERIES)):
        path = INPUT_DIR / cfg["file"]
        if not path.exists():
            raise FileNotFoundError(path)

        steps, raw, smooth = _read_series(path)

        ax.plot(
            steps,
            raw,
            color=raw_color,
            alpha=0.78,
            linewidth=1.1,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            steps,
            smooth,
            color=MAIN_COLOR,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=3,
        )

        ax.set_title(cfg["title"], pad=10)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(cfg["ylabel"])
        _decorate_axis(ax, PANEL_LABELS[idx])

    fig.subplots_adjust(top=0.90, wspace=0.30)

    fig.savefig(OUTPUT_DIR / f"{OUTPUT_BASENAME}.png")
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf")
    plt.close(fig)
    print("Wrote figure to", OUTPUT_DIR / f"{OUTPUT_BASENAME}.png")


if __name__ == "__main__":
    main()
