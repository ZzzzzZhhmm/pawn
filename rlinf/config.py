# Copyright 2025 The RLinf Authors.
# Modified for the PAWN OpenPI training release (2026).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import logging
from dataclasses import asdict
from enum import Enum
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig

from rlinf.envs import SupportedEnvType
from rlinf.scheduler.cluster import Cluster
from rlinf.utils.placement import (
    HybridComponentPlacement,
)

logging.getLogger().setLevel(logging.INFO)


class SupportedModel(Enum):
    # Embodied models
    OPENVLA = ("openvla", "embodied")
    OPENVLA_OFT = ("openvla_oft", "embodied")
    OPENPI = ("openpi", "embodied")
    MLP_POLICY = ("mlp_policy", "embodied")
    GR00T = ("gr00t", "embodied")
    CNN_POLICY = ("cnn_policy", "embodied")
    FLOW_POLICY = ("flow_policy", "embodied")

    def __new__(cls, value, category):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.category = category
        return obj


def get_supported_model(model_type: str) -> SupportedModel:
    try:
        return SupportedModel(model_type)
    except ValueError as err:
        supported_models = [e.value for e in SupportedModel]
        raise NotImplementedError(
            f"Model Type: {model_type} not supported. Supported models: {supported_models}"
        ) from err


__all__ = ["build_config"]


def torch_dtype_from_precision(precision: Union[int, str]) -> torch.dtype:
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif precision in [16, "16", "fp16", "16-mixed"]:
        return torch.float16
    elif precision in [32, "32", "fp32", "32-true"]:
        return torch.float32
    elif precision in [None]:
        return None
    else:
        raise ValueError(
            f"Could not parse the precision of `{precision}` to a valid torch.dtype"
        )


@torch.jit.script
def gelu_impl(x):
    """
    OpenAI's gelu implementation.
    """
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


def openai_gelu(x):
    return gelu_impl(x)


try:
    jit_fuser = torch.compile
except Exception:
    jit_fuser = torch.jit.script


@jit_fuser
def squared_relu(x):
    return torch.pow(torch.nn.functional.relu(x), 2)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return (
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)
            + torch.ones_like(x).to(dtype=x.dtype)
        )
    )


def activation_to_func(
    activation: str, openai_gelu: bool = False, onnx_safe: bool = False
) -> Callable:
    """
    Converts an activation function represented as a string to a function.

    Args:
        activation (str): string representation of an activation function, typically gotten from the model config.
        openai_gelu (bool): whether to use the OpenAI GELU implementation. Used with HF compatibility.
        onnx_safe (bool): whether to use the ONNX-compatible implementation of GELU.

    Returns:
        Callable: the activation function.
    """

    supported_activations = [
        "gelu",
        "geglu",
        "reglu",
        "swiglu",
        "squared-relu",
        "fast-geglu",
        "fast-swiglu",
        "fast-reglu",
        "approx-gelu",
    ]

    if activation not in supported_activations:
        raise ValueError(
            f"Unsupported activation {activation}. Supported activations: {supported_activations} "
        )

    # Give openai_gelu precedence over other activations if set, for HF compatibility.
    # Normally this is off and shouldn't affect regular model training.
    if openai_gelu:
        activation_func = openai_gelu
    elif activation in ["gelu", "geglu", "fast-geglu"]:
        activation_func = F.gelu
    elif onnx_safe:
        activation_func = erf_gelu
    elif activation in ["reglu", "fast-reglu"]:
        activation_func = F.relu
    elif activation in ["swiglu", "fast-swiglu"]:
        # SiLU or sigmoid linear unit is the same as swish with beta = 1 (which is what https://arxiv.org/pdf/2002.05202.pdf uses.)
        activation_func = F.silu
    elif activation == "squared-relu":
        activation_func = squared_relu

    return activation_func


def validate_fsdp_cfg(cfg: DictConfig, resume_dir: Optional[str] = None) -> DictConfig:
    def validate_amp_cfg(config: DictConfig) -> DictConfig:
        if "amp" not in config:
            config.amp = {}
        config.amp.enabled = config.amp.get("enabled", False)
        config.amp.precision = config.amp.get("precision", "bf16")
        assert config.amp.precision in ["fp16", "bf16", "fp32"], (
            "fsdp.amp.precision must be one of ['fp16', 'bf16', 'fp32']"
        )
        config.amp.use_grad_scaler = config.amp.get("use_grad_scaler", False)
        return config

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.fsdp_config.strategy = cfg.fsdp_config.get("strategy", "fsdp")

        cfg.fsdp_config.sharding_strategy = cfg.fsdp_config.get(
            "sharding_strategy", "full_shard"
        )

        cfg.fsdp_config.forward_prefetch = cfg.fsdp_config.get(
            "forward_prefetch", False
        )
        cfg.fsdp_config.limit_all_gathers = cfg.fsdp_config.get(
            "limit_all_gathers", False
        )
        cfg.fsdp_config.backward_prefetch = cfg.fsdp_config.get(
            "backward_prefetch", None
        )
        cfg.fsdp_config.use_orig_params = cfg.fsdp_config.get("use_orig_params", False)
        cfg.fsdp_config.use_liger_kernel = cfg.fsdp_config.get(
            "use_liger_kernel", False
        )
        cfg.fsdp_config = validate_amp_cfg(cfg.fsdp_config)

        cfg.fsdp_config.cpu_offload = cfg.fsdp_config.get("cpu_offload", False)
        cfg.fsdp_config.offload_pin_memory = cfg.fsdp_config.get(
            "offload_pin_memory", False
        )
        cfg.fsdp_config.reshard_after_forward = cfg.fsdp_config.get(
            "reshard_after_forward", True
        )
        cfg.fsdp_config.enable_gradient_accumulation = cfg.fsdp_config.get(
            "enable_gradient_accumulation", False
        )

        # Do not force use_orig_params on resume; keep user config to match saved FSDP layout.

        assert cfg.fsdp_config.backward_prefetch in [
            None,
            "pre",
            "post",
        ], "fsdp_config.backward_prefetch must be one of [None, 'pre', 'post']"

        # validate mixed precision config
        assert hasattr(cfg.fsdp_config, "mixed_precision"), (
            "fsdp_config.mixed_precision is required in FSDP actor configuration."
        )

        mixed_precision_config = cfg.fsdp_config.mixed_precision
        mixed_precision_config.param_dtype = mixed_precision_config.get(
            "param_dtype", "bf16"
        )
        mixed_precision_config.reduce_dtype = mixed_precision_config.get(
            "reduce_dtype", "bf16"
        )
        mixed_precision_config.buffer_dtype = mixed_precision_config.get(
            "buffer_dtype", "fp32"
        )

    return cfg


def validate_embodied_cfg(cfg):
    assert get_supported_model(cfg.actor.model.model_type).category == "embodied", (
        f"Model type: '{cfg.actor.model.model_type}' is not an embodied model. "
        f"Supported embodied models: {[e.value for e in SupportedModel if e.category == 'embodied']}."
    )

    # NOTE: Currently we only support actor_critic as PPO algorithm loss, and only support value_head as critic model.
    # This will be updated in the future to support more algorithms and critic models.
    # Check that actor_critic loss requires value_head
    if cfg.algorithm.loss_type == "actor_critic":
        add_value_head = cfg.actor.model.get("add_value_head", False)
        assert add_value_head, (
            f"When using PPO algorithm (algorithm.loss_type='actor_critic'), "
            f"actor.model.add_value_head must be True. "
            f"Current value: {add_value_head}"
        )

    if cfg.algorithm.loss_type == "nft-dual-credit":
        assert cfg.algorithm.adv_type == "dual-credit", (
            "algorithm.loss_type='nft-dual-credit' requires "
            "algorithm.adv_type='dual-credit'."
        )
        assert not cfg.env.train.auto_reset, (
            "Dual-credit requires env.train.auto_reset=False so rollout trajectories "
            "preserve a fixed initial-state group within each collected episode."
        )
        assert not cfg.env.train.ignore_terminations, (
            "Dual-credit requires env.train.ignore_terminations=False so success/fail "
            "outcomes and invalid-chunk masks remain well-defined."
        )
        assert cfg.env.train.group_size == cfg.algorithm.group_size, (
            "Dual-credit requires env.train.group_size to match algorithm.group_size "
            "so each rollout group shares the same task/reset condition."
        )
        store_full_solver_path = bool(
            cfg.actor.model.get("openpi", {}).get("store_full_solver_path", False)
        )
        assert store_full_solver_path, (
            "Dual-credit requires actor.model.openpi.store_full_solver_path=True "
            "to collect all K solver states."
        )
        pos_top_ratio = float(cfg.algorithm.get("prm_pos_top_ratio", 0.9))
        assert 0.0 < pos_top_ratio <= 1.0, (
            "Dual-credit requires algorithm.prm_pos_top_ratio in (0, 1]. "
            f"Current value: {pos_top_ratio}"
        )

    # process num-envs
    component_placement = HybridComponentPlacement(
        cfg, Cluster(cluster_cfg=cfg.cluster)
    )
    stage_num = cfg.rollout.pipeline_stage_num
    env_world_size = component_placement.get_world_size("env")

    if cfg.runner.val_check_interval > 0 or cfg.runner.only_eval:
        assert cfg.env.eval.total_num_envs > 0, (
            "Total number of parallel environments for evaluation must be greater than 0"
        )
        assert cfg.env.eval.total_num_envs % env_world_size == 0, (
            "Total number of parallel environments for evaluation must be divisible by the number of environment processes"
        )
        assert cfg.env.eval.total_num_envs % env_world_size % stage_num == 0, (
            "Total number of parallel environments for evaluation must be divisible by the number of environment processes and the number of pipeline stages"
        )
        assert cfg.env.eval.total_num_envs // env_world_size // stage_num > 0, (
            "env.eval.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
        )
        assert (
            cfg.env.eval.total_num_envs
            // env_world_size
            // stage_num
            % cfg.env.eval.group_size
            == 0
        ), (
            "env.eval.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by the group size"
        )
        assert (
            cfg.env.eval.max_steps_per_rollout_epoch % cfg.actor.model.num_action_chunks
            == 0
        ), (
            "env.eval.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )

    if not cfg.runner.only_eval:
        assert cfg.env.train.total_num_envs > 0, (
            "Total number of parallel environments for training must be greater than 0"
        )
        assert cfg.env.train.total_num_envs % env_world_size == 0, (
            "Total number of parallel environments for training must be divisible by the number of environment processes"
        )
        assert cfg.env.train.total_num_envs % env_world_size % stage_num == 0, (
            "Total number of parallel environments for training must be divisible by the number of environment processes and the number of pipeline stages"
        )
        assert cfg.env.train.total_num_envs // env_world_size // stage_num > 0, (
            "env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num must be greater than 0"
        )
        assert (
            cfg.env.train.total_num_envs
            // env_world_size
            // stage_num
            % cfg.env.train.group_size
            == 0
        ), (
            "env.train.total_num_envs // env_world_size // rollout.pipeline_stage_num must be divisible by the group size"
        )
        assert (
            cfg.env.train.max_steps_per_rollout_epoch
            % cfg.actor.model.num_action_chunks
            == 0
        ), (
            "env.train.max_steps_per_rollout_epoch must be divisible by actor.model.num_action_chunks"
        )

    with open_dict(cfg):
        if (
            SupportedEnvType(cfg.env.train.env_type) == SupportedEnvType.MANISKILL
            or SupportedEnvType(cfg.env.eval.env_type) == SupportedEnvType.MANISKILL
        ):

            def get_robot_control_mode(robot: str):
                if robot == "panda-qpos":
                    return "pd_joint_delta_pos"
                elif robot == "panda-ee-dpos":
                    return "pd_ee_delta_pos"
                elif "google_robot_static" in robot:
                    return "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
                elif "widowx" in robot:
                    return "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
                else:
                    raise NotImplementedError(f"Robot {robot} not supported")

            cfg.env.train.init_params.control_mode = get_robot_control_mode(
                cfg.actor.model.policy_setup
            )
            cfg.env.eval.init_params.control_mode = get_robot_control_mode(
                cfg.actor.model.policy_setup
            )

    return cfg


def validate_cfg(cfg: DictConfig) -> DictConfig:
    """Validate the OpenPI/FSDP training and evaluation configuration."""
    OmegaConf.set_struct(cfg, True)
    assert cfg.runner.task_type == "embodied", "PAWN supports embodied tasks."
    assert cfg.actor.model.model_type == "openpi", "PAWN requires OpenPI."
    assert cfg.actor.training_backend == "fsdp", "PAWN requires FSDP."
    cfg = validate_embodied_cfg(cfg)
    component_placement = HybridComponentPlacement(
        cfg, Cluster(num_nodes=cfg.cluster.num_nodes)
    )
    actor_world_size = component_placement.get_world_size("actor")
    assert (
        cfg.actor.global_batch_size % (cfg.actor.micro_batch_size * actor_world_size)
        == 0
    ), "global_batch_size must be divisible by micro_batch_size * actor_world_size."
    cfg.actor = validate_fsdp_cfg(cfg.actor, cfg.runner.get("resume_dir", None))
    return cfg


def build_config(cls, cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        cfg = asdict(cfg)

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in cfg:
            kwargs[f.name] = cfg.get(f.name)

    return cls(**kwargs)
