import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb


LOG_FOLDERS = [
    "object_nft_actor_openpi_15ckpt",
    "object_nft_actor_openpi_20ckpt",
    "object_nft_actor_openpi_50ckpt",
    "object_nft_actor_openpi_60-150ckpt",
]

OUTPUT_DIR = Path("analysis_outputs")
CSV_DIR = OUTPUT_DIR / "derived_csvs"
FIG_BASENAME = "object_mechanism_refined"
EMA_ALPHA = 0.18
PANEL_LABELS = ["(a)", "(b)", "(c)"]
MAIN_COLOR = "#8B1E3F"


def _lighten_color(color: str, amount: float = 0.72) -> tuple[float, float, float]:
    base = np.array(to_rgb(color), dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    mixed = base * (1.0 - amount) + white * amount
    return tuple(np.clip(mixed, 0.0, 1.0))


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


def _read_rows() -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for folder in LOG_FOLDERS:
        path = Path("log") / folder / "metrics_full.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                step = int(row["step"])
                rows[step] = row
    return dict(sorted(rows.items()))


def _build_full_series(step_to_val: dict[int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed_steps = np.array(sorted(step_to_val), dtype=int)
    full_steps = np.arange(observed_steps[0], observed_steps[-1] + 1, dtype=int)

    raw_full = np.full(len(full_steps), np.nan, dtype=float)
    for idx, step in enumerate(full_steps):
        if step in step_to_val:
            raw_full[idx] = step_to_val[step]

    filled = np.interp(full_steps.astype(float), observed_steps.astype(float), np.array([step_to_val[s] for s in observed_steps], dtype=float))
    observed = ~np.isnan(raw_full)
    smooth = _ema(filled)
    return full_steps, raw_full, filled, smooth


def _extract_metric_rows(rows: dict[int, dict]) -> dict[str, dict[int, float]]:
    metrics = {
        "chunk_credit_gain": {},
        "pair_separation": {},
        "block_selectivity": {},
    }
    log3 = math.log(3.0)
    for step, row in rows.items():
        chunk_delta = row.get("train/actor/chunk_loss_minus_neutral")
        pair_gap = row.get("train/actor/pair_precompute_gap_mean")
        block_entropy = row.get("train/actor/block_weight_entropy_mean")
        if chunk_delta is not None:
            metrics["chunk_credit_gain"][step] = -float(chunk_delta)
        if pair_gap is not None:
            metrics["pair_separation"][step] = float(pair_gap)
        if block_entropy is not None:
            metrics["block_selectivity"][step] = 1.0 - float(block_entropy) / log3
    return metrics


def _write_metric_csv(metric_name: str, steps: np.ndarray, raw: np.ndarray, filled: np.ndarray, smooth: np.ndarray):
    path = CSV_DIR / f"{metric_name}.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "observed", "raw_value", "filled_value", "smooth_value"],
        )
        writer.writeheader()
        for step, raw_v, filled_v, smooth_v in zip(steps, raw, filled, smooth):
            writer.writerow(
                {
                    "step": int(step),
                    "observed": int(not np.isnan(raw_v)),
                    "raw_value": "" if np.isnan(raw_v) else f"{raw_v:.10f}",
                    "filled_value": f"{filled_v:.10f}",
                    "smooth_value": f"{smooth_v:.10f}",
                }
            )


def _write_combined_csv(metric_payloads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    common_steps = next(iter(metric_payloads.values()))[0]
    path = CSV_DIR / "object_mechanism_combined.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "step",
            "observed",
            "chunk_credit_gain_raw",
            "chunk_credit_gain_filled",
            "chunk_credit_gain_smooth",
            "pair_separation_raw",
            "pair_separation_filled",
            "pair_separation_smooth",
            "block_selectivity_raw",
            "block_selectivity_filled",
            "block_selectivity_smooth",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, step in enumerate(common_steps):
            ccg = metric_payloads["chunk_credit_gain"]
            pair = metric_payloads["pair_separation"]
            block = metric_payloads["block_selectivity"]
            writer.writerow(
                {
                    "step": int(step),
                    "observed": int(not np.isnan(ccg[1][idx])),
                    "chunk_credit_gain_raw": "" if np.isnan(ccg[1][idx]) else f"{ccg[1][idx]:.10f}",
                    "chunk_credit_gain_filled": f"{ccg[2][idx]:.10f}",
                    "chunk_credit_gain_smooth": f"{ccg[3][idx]:.10f}",
                    "pair_separation_raw": "" if np.isnan(pair[1][idx]) else f"{pair[1][idx]:.10f}",
                    "pair_separation_filled": f"{pair[2][idx]:.10f}",
                    "pair_separation_smooth": f"{pair[3][idx]:.10f}",
                    "block_selectivity_raw": "" if np.isnan(block[1][idx]) else f"{block[1][idx]:.10f}",
                    "block_selectivity_filled": f"{block[2][idx]:.10f}",
                    "block_selectivity_smooth": f"{block[3][idx]:.10f}",
                }
            )


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


def _plot_metric(ax, steps: np.ndarray, raw: np.ndarray, smooth: np.ndarray, title: str, ylabel: str, show_legend: bool = False):
    raw_color = _lighten_color(MAIN_COLOR, 0.72)
    ax.plot(
        steps,
        raw,
        color=raw_color,
        alpha=0.75,
        linewidth=1.15,
        solid_capstyle="round",
        label="Object (Ours)" if show_legend else None,
        zorder=1,
    )
    ax.plot(
        steps,
        smooth,
        color=MAIN_COLOR,
        linewidth=3.2,
        solid_capstyle="round",
        label="Object (Ours)",
        zorder=3,
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(frameon=True, fancybox=True)


def _plot_figure(metric_payloads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.55))
    panels = [
        ("chunk_credit_gain", "Chunk Credit Gain", "Neutral Loss - Chunk Loss"),
        ("pair_separation", "Local Pair Separation", "Precomputed Pair Gap"),
        ("block_selectivity", "Block Selectivity", "Normalized Selectivity"),
    ]
    for idx, (ax, (metric_name, title, ylabel)) in enumerate(zip(axes, panels)):
        steps, raw, _filled, smooth = metric_payloads[metric_name]
        _plot_metric(ax, steps, raw, smooth, title, ylabel, show_legend=True)
        _decorate_axis(ax, PANEL_LABELS[idx])
    fig.subplots_adjust(top=0.88, wspace=0.28)
    fig.savefig(OUTPUT_DIR / f"{FIG_BASENAME}.png")
    fig.savefig(OUTPUT_DIR / f"{FIG_BASENAME}.pdf")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    rows = _read_rows()
    if not rows:
        raise RuntimeError("No object metrics_full.jsonl rows found.")

    metric_rows = _extract_metric_rows(rows)
    metric_payloads: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for metric_name, step_to_val in metric_rows.items():
        payload = _build_full_series(step_to_val)
        metric_payloads[metric_name] = payload
        _write_metric_csv(metric_name, *payload)

    _write_combined_csv(metric_payloads)
    _plot_figure(metric_payloads)

    print("Wrote derived CSVs to", CSV_DIR)
    print("Wrote figure to", OUTPUT_DIR / f"{FIG_BASENAME}.png")


if __name__ == "__main__":
    main()
