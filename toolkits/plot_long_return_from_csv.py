import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


INPUT_DIR = Path("log/analysis_inputs")
OUTPUT_DIR = Path("analysis_outputs")
DERIVED_DIR = OUTPUT_DIR / "derived_csvs"
OUTPUT_BASENAME = "long_return_comparison"
# Slightly more responsive than the default success-curve EMA so the early
# return-growth advantage is visible without altering the raw measurements.
EMA_ALPHA = 0.20

RUNS = [
    ("long_return_pawn_pi0_more_jitter.csv", "Pawn", "#8B1E3F"),
    ("long_return_stepnft_pi0_more_jitter.csv", r"$\pi$-StepNFT", "#5A5A5A"),
    ("long_return_ppo_pi0_more_jitter.csv", "SDE-PPO", "#4C78A8"),
    ("long_return_grpo_pi0_more_jitter.csv", "SDE-GRPO", "#4E9B6E"),
]


def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
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


def _ema(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.zeros_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _lighten_color(color: str, amount: float = 0.72) -> tuple[float, float, float]:
    base = np.array(to_rgb(color), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    mixed = base * (1.0 - amount) + white * amount
    return tuple(np.clip(mixed, 0.0, 1.0))


def _line_style(label: str):
    if label == "Pawn":
        return "-", 3.2
    if "StepNFT" in label:
        return "--", 2.4
    if "PPO" in label:
        return (0, (5, 2)), 2.35
    if "GRPO" in label:
        return (0, (3, 2)), 2.35
    return "--", 2.3


def _read_raw_csv(path: Path):
    rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig")))
    steps = np.array([int(float(r["step"])) for r in rows], dtype=int)
    raw = np.array([float(r["raw_value"]) for r in rows], dtype=float)
    observed = np.array([int(float(r.get("observed", 1))) for r in rows], dtype=int)
    return steps, observed, raw


def _write_combined_csv(payloads):
    path = DERIVED_DIR / "long_return_combined_recomputed.csv"
    steps = payloads[0][2]
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["step"]
        for _, label, _, _, _ in payloads:
            safe = label.replace("$", "").replace("\\pi", "pi").replace("{", "").replace("}", "").replace("-", "").replace(" ", "_").lower()
            fieldnames.extend([f"{safe}_raw", f"{safe}_smooth"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(steps):
            row = {"step": int(step)}
            for _, label, _, raw, smooth in payloads:
                safe = label.replace("$", "").replace("\\pi", "pi").replace("{", "").replace("}", "").replace("-", "").replace(" ", "_").lower()
                row[f"{safe}_raw"] = f"{raw[idx]:.10f}"
                row[f"{safe}_smooth"] = f"{smooth[idx]:.10f}"
            writer.writerow(row)


def main():
    _style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    payloads = []
    for filename, label, color in RUNS:
        steps, _observed, raw = _read_raw_csv(INPUT_DIR / filename)
        smooth = _ema(raw)
        payloads.append((filename, label, steps, raw, smooth, color))

    # write derived csv for inspection
    _write_combined_csv([(f, l, s, r, sm) for f, l, s, r, sm, c in payloads])

    fig, ax = plt.subplots(1, 1, figsize=(7.25, 4.1))
    for _filename, label, steps, raw, smooth, color in payloads:
        raw_color = _lighten_color(color, 0.72)
        linestyle, linewidth = _line_style(label)
        ax.plot(
            steps,
            raw,
            color=raw_color,
            alpha=0.75,
            linewidth=1.1,
            solid_capstyle="round",
            zorder=1,
        )
        ax.plot(
            steps,
            smooth,
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=3,
        )

    ax.set_title(r"LIBERO-Long Return ($\pi_0$)", pad=10)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Return")
    ax.set_ylim(0.20, 0.865)
    ax.tick_params(axis="both", which="both", direction="out", width=0.9, color="#5A5F6A")
    ax.set_axisbelow(True)
    ax.legend(frameon=True, fancybox=True, loc="lower right")

    fig.savefig(OUTPUT_DIR / f"{OUTPUT_BASENAME}.png")
    fig.savefig(OUTPUT_DIR / f"{OUTPUT_BASENAME}.pdf")
    plt.close(fig)
    print("Wrote figure to", OUTPUT_DIR / f"{OUTPUT_BASENAME}.png")
    print("Wrote derived CSV to", DERIVED_DIR / "long_return_combined_recomputed.csv")


if __name__ == "__main__":
    main()
