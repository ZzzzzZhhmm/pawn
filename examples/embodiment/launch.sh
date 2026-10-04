#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Expected train or eval}"
shift
case "$MODE" in
  train) ENTRY=train_embodied_agent.py ;;
  eval) ENTRY=eval_embodied_agent.py ;;
  *) printf 'Unsupported mode: %s\n' "$MODE" >&2; exit 2 ;;
esac
CONFIG_NAME="${1:-libero_object_nft_actor_openpi}"
if (($#)); then shift; fi

export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(cd "$EMBODIED_PATH/../.." && pwd)"
export PYTHONPATH="$REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${LIBERO_REPO_PATH:-}" ]]; then
  export PYTHONPATH="$LIBERO_REPO_PATH:$PYTHONPATH"
fi
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HYDRA_FULL_ERROR=1

OUTPUT_ROOT="${PAWN_OUTPUT_DIR:-$REPO_PATH/outputs}"
LOG_DIR="${PAWN_LOG_DIR:-$OUTPUT_ROOT/$(date +'%Y%m%d-%H%M%S')-$MODE-$CONFIG_NAME}"
mkdir -p "$LOG_DIR"
python "$EMBODIED_PATH/$ENTRY" --config-name "$CONFIG_NAME" \
  "runner.logger.log_path=$LOG_DIR" "$@" 2>&1 | tee "$LOG_DIR/console.log"
