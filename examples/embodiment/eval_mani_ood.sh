#!/usr/bin/env bash
set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_NAME="${1:-maniskill_nft_actor_openpi}"
if (($#)); then shift; fi
: "${CKPT_PATH:?Set CKPT_PATH to actor/model_state_dict/full_weights.pt}"
TOTAL_NUM_ENVS="${TOTAL_NUM_ENVS:-320}"
EVAL_ROLLOUT_EPOCH="${EVAL_ROLLOUT_EPOCH:-1}"

for env_id in \
  PutOnPlateInScene25VisionImage-v1 \
  PutOnPlateInScene25VisionTexture03-v1 PutOnPlateInScene25VisionTexture05-v1 \
  PutOnPlateInScene25VisionWhole03-v1 PutOnPlateInScene25VisionWhole05-v1 \
  PutOnPlateInScene25Carrot-v1 PutOnPlateInScene25Plate-v1 \
  PutOnPlateInScene25Instruct-v1 PutOnPlateInScene25MultiCarrot-v1 \
  PutOnPlateInScene25MultiPlate-v1 PutOnPlateInScene25Position-v1 \
  PutOnPlateInScene25EEPose-v1 PutOnPlateInScene25PositionChangeTo-v1
do
  PAWN_LOG_DIR="${PAWN_OUTPUT_DIR:-outputs}/ood/$CONFIG_NAME/$env_id-test" \
    bash "$EMBODIED_PATH/run_eval.sh" "$CONFIG_NAME" \
    env@env.eval=maniskill_ood_template \
    "runner.ckpt_path=$CKPT_PATH" \
    "runner.logger.experiment_name=$env_id-test" \
    "algorithm.eval_rollout_epoch=$EVAL_ROLLOUT_EPOCH" \
    "env.eval.total_num_envs=$TOTAL_NUM_ENVS" \
    "env.eval.init_params.id=$env_id" env.eval.init_params.obj_set=test \
    "$@"
done

for env_id in PutOnPlateInScene25Carrot-v1 \
  PutOnPlateInScene25MultiCarrot-v1 PutOnPlateInScene25MultiPlate-v1
do
  PAWN_LOG_DIR="${PAWN_OUTPUT_DIR:-outputs}/ood/$CONFIG_NAME/$env_id-train" \
    bash "$EMBODIED_PATH/run_eval.sh" "$CONFIG_NAME" \
    env@env.eval=maniskill_ood_template \
    "runner.ckpt_path=$CKPT_PATH" \
    "runner.logger.experiment_name=$env_id-train" \
    "algorithm.eval_rollout_epoch=$EVAL_ROLLOUT_EPOCH" \
    "env.eval.total_num_envs=$TOTAL_NUM_ENVS" \
    "env.eval.init_params.id=$env_id" env.eval.init_params.obj_set=train \
    "$@"
done
