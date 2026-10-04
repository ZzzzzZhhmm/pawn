#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python - <<'PY'
import importlib.util
import sys

if sys.version_info[:2] != (3, 11):
    raise SystemExit("PAWN requires Python 3.11. Activate the RLinf OpenPI environment.")
missing = [
    name for name in ("torch", "openpi", "ray", "hydra", "libero", "mani_skill")
    if importlib.util.find_spec(name) is None
]
if missing:
    raise SystemExit(
        "Missing dependencies: " + ", ".join(missing)
        + ". Prepare the RLinf OpenPI environment before installing PAWN."
    )
PY
# Preserve the simulator/OpenPI stack already installed in the RLinf environment.
python -m pip install -e "$REPO_PATH" --no-deps
