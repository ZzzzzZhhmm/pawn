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

from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
    stack_list_of_dict_tensor,
)

ROLLOUT_RESERVED_FORWARD_INPUT_KEYS = {
    "dones",
    "terminations",
    "truncations",
    "rewards",
    "task_ids",
    "reset_state_ids",
    "success_once",
    "prev_logprobs",
    "prev_values",
}


@dataclass(kw_only=True)
class EnvOutput:
    obs: dict[str, Any]
    final_obs: Optional[dict[str, Any]] = None
    dones: Optional[torch.Tensor] = None  # [B]
    terminations: Optional[torch.Tensor] = None  # [B]
    truncations: Optional[torch.Tensor] = None  # [B]
    rewards: Optional[torch.Tensor] = None  # [B]
    task_ids: Optional[torch.Tensor] = None  # [B] or [B, chunk]
    reset_state_ids: Optional[torch.Tensor] = None  # [B] or [B, chunk]
    success_once: Optional[torch.Tensor] = None  # [B]
    chunk_aux: Optional[dict[str, Any]] = None

    intervene_actions: Optional[torch.Tensor] = None  # [B]
    intervene_flags: Optional[torch.Tensor] = None  # [B]

    def __post_init__(self):
        self.obs = put_tensor_device(self.obs, "cpu")
        self.final_obs = (
            put_tensor_device(self.final_obs, "cpu")
            if self.final_obs is not None
            else None
        )
        self.dones = self.dones.cpu().contiguous() if self.dones is not None else None
        self.terminations = (
            self.terminations.cpu().contiguous()
            if self.terminations is not None
            else None
        )
        self.truncations = (
            self.truncations.cpu().contiguous()
            if self.truncations is not None
            else None
        )
        self.rewards = (
            self.rewards.cpu().contiguous() if self.rewards is not None else None
        )
        if self.task_ids is not None:
            self.task_ids = (
                torch.as_tensor(self.task_ids, dtype=torch.long).cpu().contiguous()
            )
        if self.reset_state_ids is not None:
            self.reset_state_ids = (
                torch.as_tensor(self.reset_state_ids, dtype=torch.long)
                .cpu()
                .contiguous()
            )
        if self.success_once is not None:
            self.success_once = torch.as_tensor(self.success_once).cpu().contiguous()
        self.chunk_aux = (
            put_tensor_device(self.chunk_aux, "cpu")
            if self.chunk_aux is not None
            else None
        )
        self.intervene_actions = (
            self.intervene_actions.cpu().contiguous()
            if self.intervene_actions is not None
            else None
        )
        self.intervene_flags = (
            self.intervene_flags.cpu().contiguous()
            if self.intervene_flags is not None
            else None
        )

    def prepare_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_tensor = obs["main_images"] if "main_images" in obs else None
        wrist_image_tensor = obs["wrist_images"] if "wrist_images" in obs else None
        extra_view_image_tensor = (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        )
        states = obs["states"] if "states" in obs else None
        task_descriptions = (
            list(obs["task_descriptions"]) if "task_descriptions" in obs else None
        )

        return {
            "main_images": image_tensor,  # [N_ENV, H, W, C]
            "wrist_images": wrist_image_tensor,  # [N_ENV, H, W, C] or [N_ENV, N_IMG, H, W, C]
            "extra_view_images": extra_view_image_tensor,  # [N_ENV, N_IMG, H, W, C]
            "states": states,
            "task_descriptions": task_descriptions,
        }

    def to_dict(self):
        env_output_dict = {}

        env_output_dict["obs"] = self.prepare_observations(self.obs)
        env_output_dict["final_obs"] = (
            self.prepare_observations(self.final_obs)
            if self.final_obs is not None
            else None
        )
        env_output_dict["dones"] = self.dones
        env_output_dict["terminations"] = self.terminations
        env_output_dict["truncations"] = self.truncations
        env_output_dict["rewards"] = self.rewards
        env_output_dict["task_ids"] = self.task_ids
        env_output_dict["reset_state_ids"] = self.reset_state_ids
        env_output_dict["success_once"] = self.success_once
        env_output_dict["chunk_aux"] = self.chunk_aux
        env_output_dict["intervene_actions"] = self.intervene_actions
        env_output_dict["intervene_flags"] = self.intervene_flags

        return env_output_dict


@dataclass(kw_only=True)
class ChunkStepResult:
    # required
    prev_logprobs: torch.Tensor = None  # [B, action_dim]
    prev_values: torch.Tensor = None  # [B, 1]
    dones: torch.Tensor = None  # [B, 1]
    truncations: torch.Tensor = None  # [B, 1]
    terminations: torch.Tensor = None  # [B, 1]
    rewards: torch.Tensor = None  # [B, 1]
    task_ids: Optional[torch.Tensor] = None  # [B, chunk]
    reset_state_ids: Optional[torch.Tensor] = None  # [B, chunk]
    success_once: Optional[torch.Tensor] = None  # [B]
    forward_inputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if self.prev_logprobs is not None:
            self.prev_logprobs = self.prev_logprobs.cpu().contiguous()
        if self.prev_values is not None:
            self.prev_values = self.prev_values.cpu().contiguous()
        if self.dones is not None:
            self.dones = self.dones.cpu().contiguous()
        if self.terminations is not None:
            self.terminations = self.terminations.cpu().contiguous()
        if self.truncations is not None:
            self.truncations = self.truncations.cpu().contiguous()
        if self.rewards is not None:
            self.rewards = self.rewards.cpu().contiguous()
        if self.task_ids is not None:
            self.task_ids = (
                torch.as_tensor(self.task_ids, dtype=torch.long).cpu().contiguous()
            )
        if self.reset_state_ids is not None:
            self.reset_state_ids = (
                torch.as_tensor(self.reset_state_ids, dtype=torch.long)
                .cpu()
                .contiguous()
            )
        if self.success_once is not None:
            self.success_once = self.success_once.cpu().contiguous()
        if self.forward_inputs:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
            self.forward_inputs = {
                k: v
                for k, v in self.forward_inputs.items()
                if k not in ROLLOUT_RESERVED_FORWARD_INPUT_KEYS
            }


@dataclass(kw_only=True)
class EmbodiedRolloutResult:
    # required
    rollout_epoch: int = None
    prev_logprobs: list[torch.Tensor] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * n_chunk_steps
    prev_values: list[torch.Tensor] = field(
        default_factory=list
    )  # lens is rollout_epoch * (n_chunk_steps + 1) because of the bootstrap value
    dones: list[torch.Tensor] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * (n_chunk_steps + 1) because of the bootstrap value
    terminations: list[torch.Tensor] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * (n_chunk_steps + 1) because of the bootstrap value
    truncations: list[torch.Tensor] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * (n_chunk_steps + 1) because of the bootstrap value
    rewards: list[torch.Tensor] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * n_chunk_steps
    task_ids: list[torch.Tensor] = field(default_factory=list)
    reset_state_ids: list[torch.Tensor] = field(default_factory=list)
    success_once: list[torch.Tensor] = field(default_factory=list)
    forward_inputs: list[dict[str, list[torch.Tensor]]] = field(
        default_factory=list
    )  # lens of results is rollout_epoch * n_chunk_steps
    transitions: list[tuple[dict[str, Any], dict[str, Any]]] = field(
        default_factory=list
    )

    def append_result(self, result: ChunkStepResult):
        if result.prev_logprobs is not None:
            self.prev_logprobs.append(result.prev_logprobs)
        if result.prev_values is not None:
            self.prev_values.append(result.prev_values)
        if result.dones is not None:
            self.dones.append(result.dones)
        if result.truncations is not None:
            self.truncations.append(result.truncations)
        if result.terminations is not None:
            self.terminations.append(result.terminations)
        if result.rewards is not None:
            self.rewards.append(result.rewards)
        if result.task_ids is not None:
            self.task_ids.append(result.task_ids)
        if result.reset_state_ids is not None:
            self.reset_state_ids.append(result.reset_state_ids)
        if result.success_once is not None:
            self.success_once.append(result.success_once)
        if result.forward_inputs:
            self.forward_inputs.append(result.forward_inputs)

    def add_transition(self, obs, next_obs):
        self.transitions.append(
            {
                "obs": put_tensor_device(obs, "cpu"),
                "next_obs": put_tensor_device(next_obs, "cpu"),
            }
        )

    def to_dict(self):
        rollout_result_dict = {}
        rollout_result_dict["prev_logprobs"] = (
            torch.stack(self.prev_logprobs, dim=0).cpu().contiguous()
            if len(self.prev_logprobs) > 0
            else None
        )
        rollout_result_dict["prev_values"] = (
            torch.stack(self.prev_values, dim=0).cpu().contiguous()
            if len(self.prev_values) > 0
            else None
        )
        rollout_result_dict["dones"] = (
            torch.stack(self.dones, dim=0).cpu().contiguous()
            if len(self.dones) > 0
            else None
        )
        rollout_result_dict["terminations"] = (
            torch.stack(self.terminations, dim=0).cpu().contiguous()
            if len(self.terminations) > 0
            else None
        )
        rollout_result_dict["truncations"] = (
            torch.stack(self.truncations, dim=0).cpu().contiguous()
            if len(self.truncations) > 0
            else None
        )
        rollout_result_dict["rewards"] = (
            torch.stack(self.rewards, dim=0).cpu().contiguous()
            if len(self.rewards) > 0
            else None
        )
        rollout_result_dict["task_ids"] = (
            torch.stack(self.task_ids, dim=0).cpu().contiguous()
            if len(self.task_ids) > 0
            else None
        )
        rollout_result_dict["reset_state_ids"] = (
            torch.stack(self.reset_state_ids, dim=0).cpu().contiguous()
            if len(self.reset_state_ids) > 0
            else None
        )
        rollout_result_dict["success_once"] = (
            torch.stack(self.success_once, dim=0).cpu().contiguous()
            if len(self.success_once) > 0
            else None
        )

        merged_forward_inputs = stack_list_of_dict_tensor(self.forward_inputs)
        for k in merged_forward_inputs.keys():
            if k in ROLLOUT_RESERVED_FORWARD_INPUT_KEYS:
                continue
            rollout_result_dict[k] = merged_forward_inputs[k]

        transition_dict = stack_list_of_dict_tensor(self.transitions)
        if len(transition_dict) > 0:
            rollout_result_dict["transitions"] = transition_dict

        assert len(rollout_result_dict["dones"]) == len(
            rollout_result_dict["prev_values"]
        ), "dones and prev_values must have the same length"
        assert (
            len(rollout_result_dict["dones"])
            == len(rollout_result_dict["rewards"]) + self.rollout_epoch
        ), "dones length must be the length of rewards plus rollout_epoch"

        return rollout_result_dict

    def to_splitted_dict(self, split_size) -> list[dict[str, Any]]:
        return split_dict_to_chunk(self.to_dict(), split_size, dim=1)
