import csv
import json
from pathlib import Path

import numpy as np


LOG_FOLDERS = [
    "object_nft_actor_openpi_15ckpt",
    "object_nft_actor_openpi_20ckpt",
    "object_nft_actor_openpi_50ckpt",
    "object_nft_actor_openpi_60-150ckpt",
]

OUTPUT_DIR = Path("analysis_outputs") / "derived_csvs"
OUTPUT_FILE = OUTPUT_DIR / "object_return.csv"
EMA_ALPHA = 0.18


def _ema(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.zeros_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def _read_rows() -> dict[int, float]:
    step_to_val: dict[int, float] = {}
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
                value = row.get("env/return")
                if value is not None:
                    step_to_val[step] = float(value)
    return dict(sorted(step_to_val.items()))


def _build_series(step_to_val: dict[int, float]):
    observed_steps = np.array(sorted(step_to_val), dtype=int)
    full_steps = np.arange(observed_steps[0], observed_steps[-1] + 1, dtype=int)

    raw_full = np.full(len(full_steps), np.nan, dtype=float)
    for idx, step in enumerate(full_steps):
        if step in step_to_val:
            raw_full[idx] = step_to_val[step]

    filled = np.interp(
        full_steps.astype(float),
        observed_steps.astype(float),
        np.array([step_to_val[s] for s in observed_steps], dtype=float),
    )
    smooth = _ema(filled)
    return full_steps, raw_full, filled, smooth


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    step_to_val = _read_rows()
    if not step_to_val:
        raise RuntimeError("No object env/return values found in metrics_full.jsonl logs.")

    steps, raw, filled, smooth = _build_series(step_to_val)

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
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

    print("Wrote", OUTPUT_FILE)


if __name__ == "__main__":
    main()
