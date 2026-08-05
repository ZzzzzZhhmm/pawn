#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export __EGL_VENDOR_LIBRARY_FILENAMES="${REPO_PATH}/10_nvidia.json"
export EGL_PLATFORM=device
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export LIBERO_REPO_PATH="/path/for/LIBERO"
export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi


# Optional ckpt evaluation settings:
#   TIMESTAMP: training run timestamp under logs/
#   EXP_SUBPATH: subpath under logs/<TIMESTAMP>/ (default: maniskill_ppo_openpi/checkpoints)
#   STEP: single global step number (e.g., 120000)
#   STEPS: comma/space separated list of steps (e.g., "120000,140000,160000")
#   MIN_STEP: skip steps below this value when auto-scanning
#   EVAL_NAME: eval output folder name under logs/eval/
TIMESTAMP="${TIMESTAMP:-}"
EXP_SUBPATH="${EXP_SUBPATH:-maniskill_nft_actor_openpi/checkpoints}"
STEP="${STEP:-}"
STEPS="${STEPS:-}"
MIN_STEP="${MIN_STEP:-0}"
EVAL_NAME="${EVAL_NAME:-embodiment_${TIMESTAMP}}"

if [[ -z "${TIMESTAMP}" ]]; then
    LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
    MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
    mkdir -p "${LOG_DIR}"
    CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
    echo ${CMD}
    ${CMD} 2>&1 | tee ${MEGA_LOG_FILE}
    exit 0
fi

CKPT_ROOT="${REPO_PATH}/logs/${TIMESTAMP}/${EXP_SUBPATH}"
if [[ ! -d "${CKPT_ROOT}" ]]; then
    echo "[error] checkpoints not found: ${CKPT_ROOT}"
    exit 1
fi

STEP_DIRS=()
if [[ -n "${STEPS}" ]]; then
    tmp_steps="${STEPS//,/ }"
    read -r -a step_list <<< "${tmp_steps}"
    for step_num in "${step_list[@]}"; do
        STEP_DIRS+=("${CKPT_ROOT}/global_step_${step_num}")
    done
elif [[ -n "${STEP}" ]]; then
    STEP_DIRS+=("${CKPT_ROOT}/global_step_${STEP}")
else
    mapfile -t STEP_DIRS < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name "global_step_*" | sort -Vr)
fi

if [[ ${#STEP_DIRS[@]} -eq 0 ]]; then
    echo "[error] no global_step_* under ${CKPT_ROOT}"
    exit 1
fi

for step_dir in "${STEP_DIRS[@]}"; do
    step_name="$(basename "${step_dir}")"
    step_num="${step_name#global_step_}"
    if [[ "${step_num}" -lt "${MIN_STEP}" ]]; then
        continue
    fi

    CKPT_PATH="${step_dir}/actor/model_state_dict/full_weights.pt"
    if [[ ! -f "${CKPT_PATH}" ]]; then
        echo "[warn] skip ${step_name}: missing ${CKPT_PATH}"
        continue
    fi

    LOG_DIR="${REPO_PATH}/logs/eval/${EVAL_NAME}/${step_name}/$(date +'%Y%m%d-%H:%M:%S')"
    MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
    mkdir -p "${LOG_DIR}"
    CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} runner.ckpt_path=${CKPT_PATH}"
    echo ${CMD}
    ${CMD} 2>&1 | tee ${MEGA_LOG_FILE}
done
