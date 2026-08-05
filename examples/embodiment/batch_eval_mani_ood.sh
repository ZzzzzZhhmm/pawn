#! /bin/bash
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

LOGS_ROOT="${REPO_PATH}/logs"
TIMESTAMP="${TIMESTAMP:-}"
EXP_SUBPATH="${EXP_SUBPATH:-maniskill_nft_actor_openpi/checkpoints}"
CONFIG_NAME="${CONFIG_NAME:-maniskill_nft_actor_openpi_ood}"  # env.eval must be maniskill_ood_template
EVAL_NAME="${EVAL_NAME:-mani_ood_${TIMESTAMP}}"
MIN_STEP="${MIN_STEP:-120}"

CKPT_ROOT="${LOGS_ROOT}/${TIMESTAMP}/${EXP_SUBPATH}"

if [[ ! -d "${CKPT_ROOT}" ]]; then
  echo "[error] checkpoints not found: ${CKPT_ROOT}"
  exit 1
fi

mapfile -t STEP_DIRS < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name "global_step_*" | sort -Vr)

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

  for env_id in \
      "PutOnPlateInScene25VisionImage-v1" "PutOnPlateInScene25VisionTexture03-v1" "PutOnPlateInScene25VisionTexture05-v1" \
      "PutOnPlateInScene25VisionWhole03-v1"  "PutOnPlateInScene25VisionWhole05-v1" \
      "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25Plate-v1" "PutOnPlateInScene25Instruct-v1" \
      "PutOnPlateInScene25MultiCarrot-v1" "PutOnPlateInScene25MultiPlate-v1" \
      "PutOnPlateInScene25Position-v1" "PutOnPlateInScene25EEPose-v1" "PutOnPlateInScene25PositionChangeTo-v1" ; \
  do
      obj_set="test"
      LOG_DIR="${REPO_PATH}/logs/eval/${EVAL_NAME}/${step_name}/$(date +'%Y%m%d-%H:%M:%S')-${env_id}-${obj_set}"
      MEGA_LOG_FILE="${LOG_DIR}/run_ppo.log"
      mkdir -p "${LOG_DIR}"
      CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ \
          --config-name ${CONFIG_NAME} \
          runner.logger.log_path=${LOG_DIR} \
          env.eval.init_params.id=${env_id} \
          env.eval.init_params.obj_set=${obj_set} \
          runner.ckpt_path=${CKPT_PATH}"

      echo "${CMD}" > "${MEGA_LOG_FILE}"
      ${CMD} 2>&1 | tee -a "${MEGA_LOG_FILE}"
  done
done
