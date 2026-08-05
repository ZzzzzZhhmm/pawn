import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from matplotlib.colors import to_rgb

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


"""
Paper-ready plotting template for adaptation curves and mechanism figures.

What this script supports
-------------------------
1. Reading runs from resumed-training folders under `log/`
2. Reading external baseline curves from CSV / JSONL files
3. Merging segmented runs by training step
4. Filling missing steps by linear interpolation for visualization
5. Computing table-ready summary statistics:
   - Peak SR
   - Peak-step
   - T90
6. Plotting:
   - representative adaptation curves for the main paper
   - mechanism curves for our method


Input formats
-------------
A. Folder-based run (our current RL logs)
   The script scans a log folder for files like:
   - metrics_full.jsonl
   - *metrics_full*
   - *pi0*
   and parses each line as JSON.

B. External CSV / JSONL run
   Use this when you later add PPO / GRPO / other baselines.

   Required columns/keys:
   - step
   - success  (or env/success_once)

   Optional columns/keys for mechanism plots:
   - prm_effective_synth_share   (or train/prm/effective_synthetic_weight_share)
   - pair_valid_frac             (or train/actor/pair_valid_frac)
   - block_weight_max            (or train/actor/block_weight_max_mean)

   Example CSV:
   step,success
   1,0.21
   2,0.24
   ...

   Example JSONL:
   {"step": 1, "success": 0.21}
   {"step": 2, "success": 0.24}
"""


# ---------------------------------------------------------------------
# Editable experiment registry
# ---------------------------------------------------------------------
SUCCESS_KEY = "env/success_once"
SUCCESS_ALIASES = ["success", "env/success_once", "sr", "success_rate"]

METRIC_ALIASES = {
    "train/prm/effective_synthetic_weight_share": [
        "train/prm/effective_synthetic_weight_share",
        "prm_effective_synth_share",
    ],
    "train/actor/pair_valid_frac": [
        "train/actor/pair_valid_frac",
        "pair_valid_frac",
    ],
    "train/actor/block_weight_max_mean": [
        "train/actor/block_weight_max_mean",
        "block_weight_max",
    ],
    "ood_avg": [
        "ood_avg",
        "OOD Avg",
        "ood/avg",
    ],
}


# Existing local runs we have already identified.
RUNS = {
    "object_ours_pi0": {
        "label": "Pawn",
        "task": "object",
        "init": "pi0",
        "method": "ours",
        "source": "file",
        "path": "log/analysis_inputs/object_pawn_pi0.csv",
        "color": "#8B1E3F",
    },
    "object_step_pi0": {
        "label": r"$\pi$-StepNFT",
        "task": "object",
        "init": "pi0",
        "method": "stepnft",
        "source": "file",
        "path": "log/analysis_inputs/object_step_pi0.csv",
        "color": "#3C3C3C",
    },
    "object_ppo_pi0": {
        "label": "SDE-PPO",
        "task": "object",
        "init": "pi0",
        "method": "ppo",
        "source": "file",
        "path": "log/analysis_inputs/object_ppo_pi0.csv",
        "color": "#7AA6C2",
    },
    "object_grpo_pi0": {
        "label": "SDE-GRPO",
        "task": "object",
        "init": "pi0",
        "method": "grpo",
        "source": "file",
        "path": "log/analysis_inputs/object_grpo_pi0.csv",
        "color": "#6E9F5A",
    },
    "long_ours_pi0": {
        "label": "Pawn",
        "task": "long",
        "init": "pi0",
        "method": "ours",
        "source": "file",
        "path": "log/analysis_inputs/long_pawn_pi0.csv",
        "color": "#8B1E3F",
    },
    "long_step_pi0": {
        "label": r"$\pi$-StepNFT",
        "task": "long",
        "init": "pi0",
        "method": "stepnft",
        "source": "file",
        "path": "log/analysis_inputs/long_step_pi0.csv",
        "color": "#3C3C3C",
    },
    "long_ppo_pi0": {
        "label": "SDE-PPO",
        "task": "long",
        "init": "pi0",
        "method": "ppo",
        "source": "file",
        "path": "log/analysis_inputs/long_ppo_pi0.csv",
        "color": "#7AA6C2",
    },
    "long_grpo_pi0": {
        "label": "SDE-GRPO",
        "task": "long",
        "init": "pi0",
        "method": "grpo",
        "source": "file",
        "path": "log/analysis_inputs/long_grpo_pi0.csv",
        "color": "#6E9F5A",
    },
    "mani_ours_pi0": {
        "label": "Pawn",
        "task": "maniskill",
        "init": "pi0",
        "method": "ours",
        "source": "file",
        "path": "log/analysis_inputs/mani_pawn_pi0.csv",
        "color": "#8B1E3F",
    },
    "mani_step_pi0": {
        "label": r"$\pi$-StepNFT",
        "task": "maniskill",
        "init": "pi0",
        "method": "stepnft",
        "source": "file",
        "path": "log/analysis_inputs/mani_step_pi0.csv",
        "color": "#3C3C3C",
    },
    "mani_ppo_pi0": {
        "label": "SDE-PPO",
        "task": "maniskill",
        "init": "pi0",
        "method": "ppo",
        "source": "file",
        "path": "log/analysis_inputs/mani_ppo_pi0.csv",
        "color": "#7AA6C2",
    },
    "mani_grpo_pi0": {
        "label": "SDE-GRPO",
        "task": "maniskill",
        "init": "pi0",
        "method": "grpo",
        "source": "file",
        "path": "log/analysis_inputs/mani_grpo_pi0.csv",
        "color": "#6E9F5A",
    },
    "object_mech_pi0": {
        "label": "Object",
        "task": "object",
        "init": "pi0",
        "method": "ours",
        "source": "file",
        "path": "log/analysis_inputs/pawn_mechanism_object_pi0.csv",
        "color": "#A23B72",
    },
    "long_mech_pi0": {
        "label": "Long",
        "task": "long",
        "init": "pi0",
        "method": "ours",
        "source": "file",
        "path": "log/analysis_inputs/pawn_mechanism_long_pi0.csv",
        "color": "#355070",
    },
    "mani_mech_pi0": {
        "label": "ManiSkill",
        "task": "maniskill",
        "init": "pi0",
        "method": "ours",
        "source": "file",
        "path": "log/analysis_inputs/pawn_mechanism_mani_pi0.csv",
        "color": "#1B8A5A",
    },
    "revised_object_ours_pi0": {
        "label": "Pawn",
        "task": "object",
        "init": "pi0",
        "method": "ours",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/object_pawn_pi0.csv",
        "color": "#8B1E3F",
    },
    "revised_object_step_pi0": {
        "label": r"$\pi$-StepNFT",
        "task": "object",
        "init": "pi0",
        "method": "stepnft",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/object_step_pi0.csv",
        "color": "#5A5A5A",
    },
    "revised_object_ppo_pi0": {
        "label": "SDE-PPO",
        "task": "object",
        "init": "pi0",
        "method": "ppo",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/object_ppo_pi0.csv",
        "color": "#4C78A8",
    },
    "revised_object_grpo_pi0": {
        "label": "SDE-GRPO",
        "task": "object",
        "init": "pi0",
        "method": "grpo",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/object_grpo_pi0.csv",
        "color": "#4E9B6E",
    },
    "revised_long_ours_pi0": {
        "label": "Pawn",
        "task": "long",
        "init": "pi0",
        "method": "ours",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/long_pawn_pi0.csv",
        "color": "#8B1E3F",
    },
    "revised_long_step_pi0": {
        "label": r"$\pi$-StepNFT",
        "task": "long",
        "init": "pi0",
        "method": "stepnft",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/long_step_pi0.csv",
        "color": "#5A5A5A",
    },
    "revised_long_ppo_pi0": {
        "label": "SDE-PPO",
        "task": "long",
        "init": "pi0",
        "method": "ppo",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/long_ppo_pi0.csv",
        "color": "#4C78A8",
    },
    "revised_long_grpo_pi0": {
        "label": "SDE-GRPO",
        "task": "long",
        "init": "pi0",
        "method": "grpo",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/long_grpo_pi0.csv",
        "color": "#4E9B6E",
    },
    "revised_mani_ours_pi0": {
        "label": "Pawn",
        "task": "maniskill",
        "init": "pi0",
        "method": "ours",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/mani_pawn_pi0.csv",
        "color": "#8B1E3F",
    },
    "revised_mani_step_pi0": {
        "label": r"$\pi$-StepNFT",
        "task": "maniskill",
        "init": "pi0",
        "method": "stepnft",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/mani_step_pi0.csv",
        "color": "#5A5A5A",
    },
    "revised_mani_ppo_pi0": {
        "label": "SDE-PPO",
        "task": "maniskill",
        "init": "pi0",
        "method": "ppo",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/mani_ppo_pi0.csv",
        "color": "#4C78A8",
    },
    "revised_mani_grpo_pi0": {
        "label": "SDE-GRPO",
        "task": "maniskill",
        "init": "pi0",
        "method": "grpo",
        "source": "file",
        "path": "log/pawn_revised_noisy_curve_csvs/analysis_inputs/mani_grpo_pi0.csv",
        "color": "#4E9B6E",
    },
}


# Main-text figure strategy
# -------------------------
# 1) representative adaptation curves:
#    default recommendation:
#    - Object (pi0)
#    - Goal OR Long (pi0)   <-- current default uses Long
#    - ManiSkill OOD-Avg (pi0)
#
# 2) mechanism figure (ours only):
#    - effective synthetic weight share
#    - pair valid fraction
#    - block selectivity proxy

REPRESENTATIVE_FIGURES = [
    {
        "name": "main_rep_adaptation",
        "panels": [
            {
                    "title": "LIBERO-Object ($\\pi_0$)",
                    "metric": SUCCESS_KEY,
                    "members": [
                        "object_ours_pi0",
                        "object_ppo_pi0",
                        "object_grpo_pi0",
                        "object_step_pi0",
                    ],
                },
                {
                    "title": "LIBERO-Long ($\\pi_0$)",
                    "metric": SUCCESS_KEY,
                    "members": [
                        "long_ours_pi0",
                        "long_ppo_pi0",
                        "long_grpo_pi0",
                        "long_step_pi0",
                    ],
                },
                {
                    "title": "ManiSkill OOD Avg. ($\\pi_0$)",
                    "metric": "ood_avg",
                    "members": [
                        "mani_ours_pi0",
                        "mani_ppo_pi0",
                        "mani_grpo_pi0",
                        "mani_step_pi0",
                    ],
                },
            ],
    }
]

REVISED_SUCCESS_FIGURES = [
    {
        "name": "revised_libero_success",
        "legend_mode": "inside",
        "legend_loc": "lower right",
        "panels": [
            {
                "title": "LIBERO-Object ($\\pi_0$)",
                "metric": SUCCESS_KEY,
                "ylabel": "Success Rate",
                "members": [
                    "revised_object_ours_pi0",
                    "revised_object_step_pi0",
                    "revised_object_ppo_pi0",
                    "revised_object_grpo_pi0",
                ],
            },
            {
                "title": "LIBERO-Long ($\\pi_0$)",
                "metric": SUCCESS_KEY,
                "ylabel": "Success Rate",
                "members": [
                    "revised_long_ours_pi0",
                    "revised_long_step_pi0",
                    "revised_long_ppo_pi0",
                    "revised_long_grpo_pi0",
                ],
            },
        ],
    },
    {
        "name": "revised_maniskill_success",
        "legend_mode": "inside",
        "legend_loc": "lower right",
        "panels": [
            {
                "title": "ManiSkill OOD Avg. ($\\pi_0$)",
                "metric": "ood_avg",
                "ylabel": "OOD Avg. Success Rate",
                "members": [
                    "revised_mani_ours_pi0",
                    "revised_mani_step_pi0",
                    "revised_mani_ppo_pi0",
                    "revised_mani_grpo_pi0",
                ],
            },
        ],
    },
]

MECHANISM_METRICS = {
    "train/prm/effective_synthetic_weight_share": {
        "ylabel": "Effective Synthetic Weight Share",
        "title": "Hard-Failure Contribution",
    },
    "train/actor/pair_valid_frac": {
        "ylabel": "Pair-Valid Fraction",
        "title": "Pair Guidance Activation",
    },
    "train/actor/block_weight_max_mean": {
        "ylabel": "Max Block Weight",
        "title": "Block Selectivity",
    },
}

MECHANISM_FIGURE = {
    "name": "main_mechanism",
    "members": [
        "object_mech_pi0",
        "long_mech_pi0",
        "mani_mech_pi0",
    ]
}


# Smoothing / interpolation settings
INTERPOLATE_MISSING = True
EMA_ALPHA = 0.18
PLATEAU_TAIL_STEPS = 10
OUTPUT_DIR = Path("analysis_outputs")
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)"]


@dataclass
class Curve:
    run_key: str
    label: str
    task: str
    init: str
    method: str
    color: str
    steps: np.ndarray
    raw: dict[str, np.ndarray]
    raw_plot: dict[str, np.ndarray]
    smooth: dict[str, np.ndarray]
    interpolated_mask: np.ndarray


def _resolve_metric(row: dict, canonical_name: str) -> float | None:
    aliases = METRIC_ALIASES.get(canonical_name, [canonical_name])
    if canonical_name == SUCCESS_KEY:
        aliases = SUCCESS_ALIASES
    for alias in aliases:
        if alias in row and row[alias] is not None:
            return row[alias]
    return None


def _find_metric_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    patterns = [
        "metrics_full*.jsonl",
        "*metrics_full*",
        "*pi0*",
    ]
    for pattern in patterns:
        for path in folder.glob(pattern):
            if path.is_file() and not path.name.endswith(".log"):
                files.append(path)
    dedup = []
    seen = set()
    for path in files:
        if path not in seen:
            dedup.append(path)
            seen.add(path)
    return dedup


def _read_jsonl_rows(files: Iterable[Path]) -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for file in files:
        try:
            with file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    step = row.get("step", row.get("global_step"))
                    success = _resolve_metric(row, SUCCESS_KEY)
                    if step is None or success is None:
                        continue
                    merged[int(step)] = row
        except Exception:
            continue
    return dict(sorted(merged.items()))


def _read_csv_or_jsonl_file(path: Path) -> dict[int, dict]:
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl_rows([path])

    merged: dict[int, dict] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = row.get("step") or row.get("global_step")
            if step is None or step == "":
                continue
            step_i = int(float(step))
            # Normalize numeric strings where possible.
            normalized = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    normalized[k] = float(v)
                except ValueError:
                    normalized[k] = v
            merged[step_i] = normalized
    return dict(sorted(merged.items()))


def _interpolate_series(step_to_val: dict[int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.array(sorted(step_to_val), dtype=float)
    vals = np.array([step_to_val[int(s)] for s in steps], dtype=float)
    if len(steps) == 0:
        return np.array([]), np.array([]), np.array([], dtype=bool)
    if not INTERPOLATE_MISSING:
        return steps, vals, np.zeros_like(steps, dtype=bool)

    full_steps = np.arange(int(steps[0]), int(steps[-1]) + 1, dtype=float)
    full_vals = np.interp(full_steps, steps, vals)
    observed = np.isin(full_steps, steps)
    interpolated = ~observed
    return full_steps, full_vals, interpolated


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


def load_curve(run_key: str, cfg: dict) -> Curve | None:
    if cfg["source"] == "folder":
        merged_rows: dict[int, dict] = {}
        for folder_name in cfg["folders"]:
            folder = Path("log") / folder_name
            if not folder.exists():
                continue
            merged_rows.update(_read_jsonl_rows(_find_metric_files(folder)))
    elif cfg["source"] == "file":
        path = Path(cfg["path"])
        if not path.exists():
            return None
        merged_rows = _read_csv_or_jsonl_file(path)
    else:
        raise ValueError(f"Unknown source type: {cfg['source']}")

    if not merged_rows:
        return None

    metric_names = {SUCCESS_KEY, "ood_avg", *MECHANISM_METRICS.keys()}
    raw_series: dict[str, np.ndarray] = {}
    raw_plot_series: dict[str, np.ndarray] = {}
    smooth_series: dict[str, np.ndarray] = {}
    interpolated_mask = None
    full_steps = None

    for metric in metric_names:
        step_to_val = {}
        for step, row in merged_rows.items():
            val = _resolve_metric(row, metric)
            if val is not None:
                step_to_val[step] = float(val)
        if not step_to_val:
            continue

        steps, vals, interp_mask = _interpolate_series(step_to_val)
        if full_steps is None:
            full_steps = steps
            interpolated_mask = interp_mask
        elif len(steps) == len(full_steps) and np.allclose(steps, full_steps):
            pass
        else:
            vals = np.interp(full_steps, steps, vals)
            interp_mask = np.ones_like(full_steps, dtype=bool)

        raw_series[metric] = vals
        smooth_series[metric] = _ema(vals)

    raw_overlay_path = cfg.get("raw_overlay_path")
    if raw_overlay_path:
        overlay_path = Path(raw_overlay_path)
        if overlay_path.exists():
            overlay_rows = _read_csv_or_jsonl_file(overlay_path)
            for metric in list(raw_series.keys()):
                step_to_val = {}
                for step, row in overlay_rows.items():
                    val = _resolve_metric(row, metric)
                    if val is not None:
                        step_to_val[step] = float(val)
                if not step_to_val:
                    continue
                steps, vals, _interp_mask = _interpolate_series(step_to_val)
                if len(steps) == len(full_steps) and np.allclose(steps, full_steps):
                    raw_plot_series[metric] = vals
                else:
                    raw_plot_series[metric] = np.interp(full_steps, steps, vals)

    if full_steps is None:
        return None

    return Curve(
        run_key=run_key,
        label=cfg["label"],
        task=cfg["task"],
        init=cfg["init"],
        method=cfg["method"],
        color=cfg["color"],
        steps=full_steps,
        raw=raw_series,
        raw_plot=raw_plot_series,
        smooth=smooth_series,
        interpolated_mask=interpolated_mask if interpolated_mask is not None else np.zeros_like(full_steps, dtype=bool),
    )


def compute_summary(curve: Curve, metric: str = SUCCESS_KEY) -> dict[str, float | int | str]:
    raw = curve.raw[metric]
    smooth = curve.smooth[metric]
    steps = curve.steps.astype(int)

    peak_raw = float(np.max(raw))
    peak_raw_step = int(steps[int(np.argmax(raw))])

    tail = smooth[-PLATEAU_TAIL_STEPS:] if len(smooth) >= PLATEAU_TAIL_STEPS else smooth
    final_plateau = float(np.mean(tail))
    t90_target = 0.9 * final_plateau
    t90_candidates = steps[smooth >= t90_target]
    t90 = int(t90_candidates[0]) if len(t90_candidates) > 0 else math.nan

    return {
        "run_key": curve.run_key,
        "label": curve.label,
        "task": curve.task,
        "init": curve.init,
        "method": curve.method,
        "step_start": int(steps[0]),
        "step_end": int(steps[-1]),
        "peak_success_raw": peak_raw,
        "peak_success_step": peak_raw_step,
        "final_plateau_smooth": final_plateau,
        "t90": t90,
        "last_success_raw": float(raw[-1]),
        "num_interpolated_steps": int(np.sum(curve.interpolated_mask)),
    }


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


def _line_style(curve: Curve) -> tuple[str, float]:
    if curve.method == "ours":
        return "-", 3.2
    if curve.method == "stepnft":
        return "--", 2.4
    if curve.method == "ppo":
        return (0, (5, 2)), 2.35
    if curve.method == "grpo":
        return (0, (3, 2)), 2.35
    return "--", 2.3


def _plot_single_curve(ax, curve: Curve, metric: str, *, show_raw: bool = True, smooth_zorder: int = 3):
    if metric not in curve.raw:
        return
    raw = curve.raw_plot.get(metric, curve.raw[metric])
    smooth = curve.smooth[metric]
    linestyle, linewidth = _line_style(curve)
    raw_color = _lighten_color(curve.color, 0.72)
    if show_raw:
        ax.plot(
            curve.steps,
            raw,
            color=raw_color,
            alpha=0.75,
            linewidth=1.15,
            solid_capstyle="round",
            zorder=1,
        )
    ax.plot(
        curve.steps,
        smooth,
        color=curve.color,
        linewidth=linewidth,
        linestyle=linestyle,
        label=curve.label,
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=smooth_zorder,
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


def plot_figure_collection(figures: list[dict], curves_by_key: dict[str, Curve], output_dir: Path):
    _style()
    for fig_cfg in figures:
        panels = fig_cfg["panels"]
        legend_mode = fig_cfg.get("legend_mode", "bottom")
        legend_loc = fig_cfg.get("legend_loc", "lower right")
        fig_width = 7.1 * len(panels) if len(panels) > 1 else 7.2
        fig, axes = plt.subplots(1, len(panels), figsize=(fig_width, 4.8), sharey=True)
        if len(panels) == 1:
            axes = [axes]

        legend_handles = None
        legend_labels = None
        for idx, (ax, panel) in enumerate(zip(axes, panels)):
            metric = panel["metric"]
            plotted = 0
            for member in panel["members"]:
                curve = curves_by_key.get(member)
                if curve is None or metric not in curve.raw:
                    continue
                _plot_single_curve(ax, curve, metric)
                plotted += 1
            ax.set_title(panel["title"], pad=12)
            ax.set_xlabel("Training Steps")
            if idx == 0:
                ax.set_ylabel(panel.get("ylabel", "Success Rate"))
            if metric == SUCCESS_KEY or metric == "ood_avg":
                ax.set_ylim(0.0, 1.02)
            _decorate_axis(ax, PANEL_LABELS[idx])
            if plotted == 0:
                ax.text(0.5, 0.5, "Add data here", ha="center", va="center", transform=ax.transAxes, alpha=0.6)
            elif legend_mode == "inside":
                ax.legend(frameon=True, fancybox=True, loc=legend_loc)
            elif legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        if legend_mode != "inside" and legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="lower center",
                ncol=len(legend_labels),
                frameon=True,
                fancybox=True,
                bbox_to_anchor=(0.5, -0.045),
            )

        if legend_mode == "inside":
            fig.subplots_adjust(bottom=0.13, wspace=0.22)
        else:
            fig.subplots_adjust(bottom=0.25, wspace=0.22)

        fig.savefig(output_dir / f"{fig_cfg['name']}.png")
        fig.savefig(output_dir / f"{fig_cfg['name']}.pdf")
        plt.close(fig)


def plot_representative_figures(curves_by_key: dict[str, Curve], output_dir: Path):
    plot_figure_collection(REPRESENTATIVE_FIGURES, curves_by_key, output_dir)


def plot_mechanism_figure(curves_by_key: dict[str, Curve], output_dir: Path):
    _style()
    members = [curves_by_key[m] for m in MECHANISM_FIGURE["members"] if m in curves_by_key]
    if not members:
        return

    metrics = list(MECHANISM_METRICS.items())
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.0, 4.55), sharex=False)
    if len(metrics) == 1:
        axes = [axes]

    legend_handles = None
    legend_labels = None
    for idx, (ax, (metric, meta)) in enumerate(zip(axes, metrics)):
        for curve in members:
            if metric not in curve.smooth:
                continue
            _plot_single_curve(ax, curve, metric, show_raw=True, smooth_zorder=4)
        ax.set_title(meta["title"], pad=12)
        ax.set_xlabel("Training Steps")
        ax.set_ylabel(meta["ylabel"])
        _decorate_axis(ax, PANEL_LABELS[idx])
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=len(legend_labels),
            frameon=True,
            fancybox=True,
            bbox_to_anchor=(0.5, 1.03),
        )

    fig.subplots_adjust(top=0.82, wspace=0.26)

    fig.savefig(output_dir / f"{MECHANISM_FIGURE['name']}.png")
    fig.savefig(output_dir / f"{MECHANISM_FIGURE['name']}.pdf")
    plt.close(fig)


def write_summary(curves: list[Curve], output_dir: Path):
    rows = [compute_summary(curve, SUCCESS_KEY) for curve in curves if SUCCESS_KEY in curve.raw]
    out_path = output_dir / "summary_metrics.csv"
    cols = [
        "run_key",
        "label",
        "task",
        "init",
        "method",
        "step_start",
        "step_end",
        "peak_success_raw",
        "peak_success_step",
        "final_plateau_smooth",
        "t90",
        "last_success_raw",
        "num_interpolated_steps",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== Summary Metrics ===")
    for row in rows:
        print(
            f"{row['label']} [{row['task']}, {row['init']}]: "
            f"peak={row['peak_success_raw']:.4f} @ step {row['peak_success_step']}, "
            f"T90={row['t90']}, plateau={row['final_plateau_smooth']:.4f}, "
            f"interpolated_steps={row['num_interpolated_steps']}"
        )


def main():
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    curves_by_key: dict[str, Curve] = {}
    for run_key, cfg in RUNS.items():
        curve = load_curve(run_key, cfg)
        if curve is not None:
            curves_by_key[run_key] = curve

    if not curves_by_key:
        raise RuntimeError("No valid curves found. Please check RUNS and input paths.")

    curves = list(curves_by_key.values())
    plot_representative_figures(curves_by_key, output_dir)
    plot_figure_collection(REVISED_SUCCESS_FIGURES, curves_by_key, output_dir)
    plot_mechanism_figure(curves_by_key, output_dir)
    write_summary(curves, output_dir)


if __name__ == "__main__":
    main()
