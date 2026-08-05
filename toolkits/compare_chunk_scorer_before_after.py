import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "log"
OUT_DIR = ROOT / "analysis_outputs"
CSV_DIR = OUT_DIR / "derived_csvs"
OUT_DIR.mkdir(exist_ok=True)
CSV_DIR.mkdir(exist_ok=True)


OLD_LOG = LOG_DIR / "log2.txt"
POST_LOGS = [
    LOG_DIR / "object_nft_actor_openpi_15ckpt" / "metrics_full.jsonl",
    LOG_DIR / "object_nft_actor_openpi_20ckpt" / "metrics_full.jsonl",
    LOG_DIR / "object_nft_actor_openpi_50ckpt" / "metrics_full.jsonl",
    LOG_DIR / "object_nft_actor_openpi_60-150ckpt" / "metrics_full.jsonl",
]

COMMON_MAX_STEP = 25
EMA_ALPHA = 0.22


COLORS = {
    "before": "#6B7280",
    "after": "#8B2740",
}


def ema(values, alpha=EMA_ALPHA):
    if not values:
        return []
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def parse_old_metrics(path: Path):
    rows = []
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in text:
        if "[metrics]" not in line or "{" not in line:
            continue
        try:
            payload = json.loads(line[line.index("{"):])
        except Exception:
            continue
        rows.append(payload)
    rows.sort(key=lambda x: int(x["step"]))
    dedup = {}
    for row in rows:
        dedup[int(row["step"])] = row
    return dedup


def parse_post_metrics(paths):
    dedup = {}
    for order, path in enumerate(paths):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            payload["_order"] = order
            step = int(payload["step"])
            if step not in dedup or payload["_order"] >= dedup[step]["_order"]:
                dedup[step] = payload
    return dedup


def build_series(dedup, key, max_step=COMMON_MAX_STEP, transform=None):
    x = list(range(1, max_step + 1))
    y = []
    observed = []
    for step in x:
        row = dedup.get(step)
        if row is None or key not in row:
            y.append(None)
            observed.append(0)
        else:
            val = row[key]
            if transform is not None:
                val = transform(val)
            y.append(float(val))
            observed.append(1)
    return x, y, observed


def fill_missing(y):
    filled = y[:]
    n = len(filled)
    known = [i for i, v in enumerate(filled) if v is not None]
    if not known:
        return [0.0] * n
    first = known[0]
    for i in range(0, first):
        filled[i] = filled[first]
    last = known[-1]
    for i in range(last + 1, n):
        filled[i] = filled[last]
    for left, right in zip(known, known[1:]):
        if right == left + 1:
            continue
        y0, y1 = filled[left], filled[right]
        gap = right - left
        for idx in range(1, gap):
            t = idx / gap
            filled[left + idx] = (1 - t) * y0 + t * y1
    return [float(v) for v in filled]


def plot_metric_panel(ax, x, smooth_before, smooth_after, title, ylabel, ylim=None, legend=False, zero_line=False):
    ax.plot(
        x, smooth_before, color=COLORS["before"], linewidth=2.4, linestyle="--",
        dash_capstyle="round", label="Before Hard-Failure-Centric Training"
    )
    ax.plot(
        x, smooth_after, color=COLORS["after"], linewidth=2.8,
        solid_capstyle="round", label="With Hard-Failure-Centric Training"
    )

    ax.set_title(title, fontsize=18, pad=10)
    ax.set_xlabel("Training Steps", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, linestyle="--", linewidth=0.9, alpha=0.22)
    ax.tick_params(labelsize=12)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if zero_line:
        ax.axhline(0.0, color="#9ca3af", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend:
        ax.legend(loc="best", fontsize=11, frameon=True)


def main():
    old = parse_old_metrics(OLD_LOG)
    post = parse_post_metrics(POST_LOGS)

    metric_defs = [
        ("success", "env/success_once", None),
        ("chunk_credit_gain", "train/actor/chunk_loss_minus_neutral", lambda v: -v),
        ("pair_saturation", "train/prm/saturated_pair_frac", None),
    ]

    data = {"step": list(range(1, COMMON_MAX_STEP + 1))}
    for prefix, source in [("before", old), ("after", post)]:
        for short_name, key, transform in metric_defs:
            _, raw, observed = build_series(source, key, transform=transform)
            filled = fill_missing(raw)
            smooth = ema(filled)
            data[f"{prefix}_{short_name}_raw"] = raw
            data[f"{prefix}_{short_name}_observed"] = observed
            data[f"{prefix}_{short_name}_filled"] = filled
            data[f"{prefix}_{short_name}_smooth"] = smooth

    csv_path = CSV_DIR / "object_chunk_scorer_before_after.csv"
    fieldnames = list(data.keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for idx in range(COMMON_MAX_STEP):
            writer.writerow([data[field][idx] for field in fieldnames])

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.15))

    plot_metric_panel(
        axes[0], data["step"],
        data["before_success_smooth"],
        data["after_success_smooth"],
        title="Success Rate",
        ylabel="Success Rate",
        ylim=(0.35, 0.82),
        legend=True,
    )
    plot_metric_panel(
        axes[1], data["step"],
        data["before_chunk_credit_gain_smooth"],
        data["after_chunk_credit_gain_smooth"],
        title="Chunk Credit Gain",
        ylabel="Neutral Loss - Chunk Loss",
        ylim=(-0.013, 0.03),
        zero_line=True,
    )
    plot_metric_panel(
        axes[2], data["step"],
        data["before_pair_saturation_smooth"],
        data["after_pair_saturation_smooth"],
        title="PRM Pair Saturation",
        ylabel="Saturated Pair Fraction",
        ylim=(-0.02, 1.02),
    )

    for idx, ax in enumerate(axes):
        ax.text(
            0.02, 0.98, f"({chr(ord('a') + idx)})",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=16, fontweight="bold", color="#1f2937",
        )

    fig.tight_layout(w_pad=2.0)

    png_path = OUT_DIR / "object_chunk_scorer_before_after.png"
    pdf_path = OUT_DIR / "object_chunk_scorer_before_after.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {png_path}")
    print(f"Saved plot: {pdf_path}")


if __name__ == "__main__":
    main()
