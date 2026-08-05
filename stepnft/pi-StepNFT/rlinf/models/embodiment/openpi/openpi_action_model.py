# Copyright 2025 The RLinf Authors.
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

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import jax
import numpy as np
import torch
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules import action_solvers
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead


NON_NFT_SOLVERS = {"flow_sde", "flow_noise", "flow_cps"}
ALL_SOLVERS = {
    "flow_sde",
    "flow_noise",
    "flow_cps",
    "euler",
    "ddim",
    "dpm",
    "flow_grpo",
    "dance",
}


@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    # config for rl
    config_name: str = "pi0_libero"  # pi0_libero, pi05_libero, pi0_maniskill, pi05_maniskill, pi0_metaworld, pi05_metaworld
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    
    # nft相关
    nft_beta: float = 0.1
    use_nft_loss: bool = False
    solver_type: Optional[str] = None  # euler, dpm, ddim, flow_sde, flow_noise, flow_cps, flow_grpo, dance
    solver_eta: float = 0.0
    solver_order: int = 2

    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    
    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    save_stochastic_value_traces: bool = False
    
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token

    def __post_init__(self):
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()
        if self.solver_type is None:
            default_solver = "dpm" if self.use_nft_loss else self.noise_method
            object.__setattr__(self, "solver_type", default_solver)


class OpenPi0ForRLActionPrediction(PI0Pytorch, BasePolicy):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        if getattr(self.config, "solver_type", self.config.noise_method) == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "lm_head",
            # --pi0 only--
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            # --pi05 only--
            "time_mlp_in",
            "time_mlp_out",
        ]

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        super().__init__(config)
        self.sample_actions = sample_actions_func
        self.global_step = 0
        self._validate_solver_config()
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            if self.config.config_name == "pi05_maniskill":
                value_head_hidden_sizes = (1024, 512, 256)
            else:
                value_head_hidden_sizes = (512, 256, 128)
            value_head_activation = "relu"
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation=value_head_activation,
                bias_last=True,
            )
            self.value_head = self.value_head.to(
                dtype=self.action_out_proj.weight.dtype
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        # noise head for flow-noise
        if self.solver_type == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )
            self.noise_head = self.noise_head.to(
                dtype=self.action_out_proj.weight.dtype
            )

        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def set_global_step(self, global_step):
        self.global_step = global_step

    def _validate_solver_config(self):
        solver_type = (self.config.solver_type or self.config.noise_method).lower()
        if self.config.use_nft_loss:
            if solver_type not in ALL_SOLVERS:
                raise ValueError(
                    f"Solver `{solver_type}` 不受支持。可选: {sorted(ALL_SOLVERS)}"
                )
        else:
            if solver_type not in NON_NFT_SOLVERS:
                raise ValueError(
                    f"非 NFT 训练仅支持 {sorted(NON_NFT_SOLVERS)} solver。"
                )
        self.solver_type = solver_type

    def _get_noise_level(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
        else:
            noise_level = self.config.noise_level
        return torch.tensor(noise_level, device=device, dtype=dtype)

    def _predict_critic_value(
        self,
        suffix_out: Optional[torch.Tensor],
        compute_values: bool,
        device: torch.device,
    ) -> torch.Tensor:
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
            and suffix_out is not None
        ):
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            with torch.amp.autocast(device_type="cuda", enabled=False):
                return self.value_head(suffix_out_value.float())[:, 0]
        bsize = 0 if suffix_out is None else suffix_out.shape[0]
        return torch.zeros((bsize,), device=device)

    def _tensor_to_numpy(self, x):
        """Convert tensor to numpy, handling BFloat16/Float16 conversion."""
        if torch.is_tensor(x):
            x_cpu = x.detach().cpu()
            # BFloat16 and Float16 are not supported by numpy, convert to float32
            if x_cpu.dtype in (torch.bfloat16, torch.float16):
                x_cpu = x_cpu.float()
            return np.asarray(x_cpu)
        return x

    def _tensor_to_numpy_single(self, x, index):
        """Convert single tensor element to numpy, handling BFloat16/Float16 conversion."""
        if torch.is_tensor(x):
            x_cpu = x[index].detach().cpu()
            # BFloat16 and Float16 are not supported by numpy, convert to float32
            if x_cpu.dtype in (torch.bfloat16, torch.float16):
                x_cpu = x_cpu.float()
            return np.asarray(x_cpu)
        return x[index]

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = jax.tree.map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # tensor -> numpy (Convert BFloat16/Float16 to float32 for numpy compatibility)
        inputs = jax.tree.map(self._tensor_to_numpy, inputs)
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and transpose
                    else x,
                    sample,
                )
            else:
                sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            transformed_samples.append(transformed_sample)
        # recombine
        inputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        # inputs = jax.tree.map(lambda *x: torch.stack(x, axis=0), inputs)
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: self._tensor_to_numpy_single(x, i), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            if self.config.use_nft_loss or kwargs.get("use_nft_loss", False):
                shared_cache = kwargs.get("shared_cache", None)
                data = kwargs["data"]
                filtered_kwargs = {
                    k: v for k, v in kwargs.items() if k not in {"shared_cache", "data"}
                }
                return self.forward_nft(data, shared_cache=shared_cache, **filtered_kwargs)
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        observation = data["observation"]
        actions = data["actions"]
        return super().forward(observation, actions)

    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        if self.config.use_nft_loss or kwargs.get("use_nft_loss", False):
            shared_cache = kwargs.get("shared_cache", None)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != "shared_cache"}
            return self.forward_nft(data, shared_cache=shared_cache, **filtered_kwargs)
        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        # input transform
        observation = self.input_transform(data, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        # get log prob
        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def forward_nft(self, data, shared_cache=None, **kwargs):
        observation = self.input_transform(data, transpose=False)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        device = None
        if torch.is_tensor(data.get("nft_xt", None)):
            device = data["nft_xt"].device
        elif torch.is_tensor(data.get("observation/state", None)):
            device = data["observation/state"].device
        if device is None:
            device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)

        explicit_inputs = kwargs.get("nft_explicit_inputs", None)
        if explicit_inputs is not None:
            x_t = explicit_inputs["x_t"]
            t = explicit_inputs["timesteps"]
        else:
            if "chains" not in data:
                raise ValueError("NFT forward requires `chains` or `nft_explicit_inputs`.")
            x_0 = data["chains"][:, -1].to(device)
            bsize = x_0.shape[0]
            t = torch.rand((bsize,), device=device)
            t_expanded = t[:, None, None]
            noise = torch.randn_like(x_0)
            x_t = (1 - t_expanded) * x_0 + t_expanded * noise

        if shared_cache is not None:
            past_key_values = shared_cache["past_key_values"]
            prefix_pad_masks = shared_cache["prefix_pad_masks"]
        else:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
            self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
            (_, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        compute_values = kwargs.get("compute_values", False)
        v_theta, suffix_out = self.get_velocity(
            state,
            x_t,
            t,
            prefix_pad_masks,
            past_key_values,
        )
        v_theta = v_theta[:, : self.config.action_chunk, :]

        value_t = None
        if compute_values and hasattr(self, "value_head"):
            value_t = self._predict_critic_value(
                suffix_out=suffix_out,
                compute_values=True,
                device=device,
            )

        result = {
            "v_theta": v_theta,
            "x_t": x_t,
            "timesteps": t,
        }
        if value_t is not None:
            result["values"] = value_t[:, None]
        if shared_cache is None:
            detached_cache = self._detach_kv_cache(past_key_values)
            result["shared_cache"] = {
                "past_key_values": detached_cache,
                "prefix_pad_masks": prefix_pad_masks.detach()
                if torch.is_tensor(prefix_pad_masks)
                else prefix_pad_masks,
            }
        return result

    def _detach_kv_cache(self, kv_cache):
        if kv_cache is None:
            return None
        try:
            from transformers.cache_utils import Cache, DynamicCache

            is_cache_obj = isinstance(kv_cache, Cache)
        except ImportError:
            is_cache_obj = False

        if is_cache_obj or hasattr(kv_cache, "get_seq_length"):
            if hasattr(kv_cache, "key_cache") and hasattr(kv_cache, "value_cache"):
                new_cache = DynamicCache()
                new_cache.key_cache = [k.detach() for k in kv_cache.key_cache]
                new_cache.value_cache = [v.detach() for v in kv_cache.value_cache]
                if hasattr(kv_cache, "_seen_tokens"):
                    new_cache._seen_tokens = kv_cache._seen_tokens
                elif hasattr(kv_cache, "seen_tokens"):
                    new_cache.seen_tokens = kv_cache.seen_tokens
                return new_cache

        if isinstance(kv_cache, tuple):
            return tuple(tuple(x.detach() for x in layer_cache) for layer_cache in kv_cache)
        return kv_cache

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation - ensure float32 to prevent BFloat16 conversion issues
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            state = env_obs["states"]
            if torch.is_tensor(state):
                state = state.to(dtype=torch.float32)
            processed_obs["observation/state"] = state
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        processed_obs[key][sub_key] = sub_value.to(
                            device=device
                        ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        return_obs=True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose=False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"].numpy()

        forward_inputs = {
            "observation/image": env_obs["main_images"],
            "observation/state": env_obs["states"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
        }
        if "chains" in outputs:
            forward_inputs["chains"] = outputs["chains"]
        if "denoise_inds" in outputs:
            forward_inputs["denoise_inds"] = outputs["denoise_inds"]
        if env_obs["wrist_images"] is not None:
            forward_inputs["observation/wrist_image"] = env_obs["wrist_images"]
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        for key in (
            "nft_xt",
            "nft_v",
            "nft_xnext",
            "nft_step_index",
            "nft_noise_level",
        ):
            if key in outputs:
                forward_inputs[key] = outputs[key]

        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        use_nft = getattr(self.config, "use_nft_loss", False)
        solver_type_for_mode = (
            "euler" if (use_nft and mode == "eval") else self.solver_type
        )
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        x_t = noise
        log_probs = []
        value_history = []
        last_value_t = None
        collect_flow_snap = use_nft and solver_type_for_mode == "flow_sde"
        flow_rand_idx = (
            torch.randint(0, self.config.num_steps, (bsize,), device=device)
            if collect_flow_snap
            else None
        )
        flow_xt_snap = flow_v_snap = flow_idx_snap = None
        if collect_flow_snap:
            flow_xt_snap = torch.empty_like(x_t)
            flow_v_snap = torch.empty_like(x_t)
            flow_xnext_snap = torch.empty_like(x_t)
            flow_idx_snap = flow_rand_idx.clone()

        values_vlm = None
        if self.use_vlm_value and compute_values:
            values_vlm = self.get_value_from_vlm(prefix_output)

        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        if use_nft:
            timesteps = torch.linspace(
                1, 0, num_steps + 1, device=device, dtype=x_t.dtype
            )
            solver_state: dict[str, Any] = {}
            last_suffix_out = None
            noise_level = self._get_noise_level(device=device, dtype=x_t.dtype)
            for idx in range(num_steps):
                t_curr = timesteps[idx]
                t_next = timesteps[idx + 1]
                t_input = t_curr.expand(bsize)
                dt = t_next - t_curr
                v_t, suffix_out = self.get_velocity(
                    state,
                    x_t,
                    t_input,
                    prefix_pad_masks,
                    past_key_values,
                )
                if collect_flow_snap and flow_rand_idx is not None:
                    mask = flow_rand_idx == idx
                    if mask.any():
                        flow_xt_snap[mask] = x_t.detach()[mask]
                        flow_v_snap[mask] = v_t.detach()[mask]
                x_t = self._apply_solver_step(
                    solver_type=solver_type_for_mode,
                    x_t=x_t,
                    velocity=v_t,
                    idx=idx,
                    timesteps=timesteps,
                    dt=dt,
                    suffix_out=suffix_out,
                    solver_state=solver_state,
                    noise_level=noise_level,
                )
                if collect_flow_snap and flow_rand_idx is not None:
                    mask = flow_rand_idx == idx
                    if mask.any():
                        flow_xnext_snap[mask] = x_t.detach()[mask]
                last_suffix_out = suffix_out

            x_0 = x_t
            nft_values = torch.zeros((bsize, 1), device=device, dtype=x_0.dtype)
            if compute_values:
                if self.use_vlm_value:
                    if values_vlm is None:
                        raise ValueError(
                            "use_vlm_value=True but values_vlm is not computed."
                        )
                    nft_values = values_vlm[:, None]
                else:
                    nft_values = self._predict_critic_value(
                        suffix_out=last_suffix_out,
                        compute_values=True,
                        device=device,
                    )[:, None]
            dummy_logprobs = torch.zeros(
                (bsize, 1, self.config.action_chunk, self.config.action_env_dim),
                device=device,
                dtype=x_0.dtype,
            )
            nft_traces: dict[str, torch.Tensor] = {}
            if collect_flow_snap and flow_xt_snap is not None and flow_v_snap is not None:
                nft_traces = {
                    "nft_xt": flow_xt_snap,
                    "nft_v": flow_v_snap[:, : self.config.action_chunk],
                    "nft_xnext": flow_xnext_snap,
                    "nft_step_index": flow_idx_snap,
                    "nft_noise_level": noise_level.detach().expand(bsize),
                }
            return {
                "actions": x_0,
                "prev_logprobs": dummy_logprobs,
                "prev_values": nft_values,
                **nft_traces,
            }

        chains = [x_t]
        values = []

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                num_steps,
                compute_values,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        # post process for logprob
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]
        # post process for value
        if self.use_vlm_value and values_vlm is not None:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def _apply_solver_step(
        self,
        solver_type: str,
        x_t: torch.Tensor,
        velocity: torch.Tensor,
        idx: int,
        timesteps: torch.Tensor,
        dt: torch.Tensor,
        suffix_out: Optional[torch.Tensor],
        solver_state: dict[str, Any],
        noise_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        solver_type = solver_type.lower()
        if solver_type == "euler":
            step = action_solvers.euler_step(x_t, velocity, dt)
            return step.sample
        if solver_type == "flow_sde":
            if noise_level is None:
                noise_level = self._get_noise_level(x_t.device, x_t.dtype)
            step = action_solvers.flow_sde_step(
                x_t=x_t,
                velocity=velocity,
                idx=idx,
                timesteps=timesteps,
                noise_level=noise_level,
            )
            return step.mean + self.sample_noise(x_t.shape, x_t.device) * step.std
        if solver_type == "flow_noise":
            if not hasattr(self, "noise_head"):
                raise ValueError("flow_noise solver 需要噪声头 ExploreNoiseNet。")
            step = action_solvers.flow_noise_step(
                x_t=x_t,
                velocity=velocity,
                idx=idx,
                timesteps=timesteps,
                noise_head=self.noise_head,
                suffix_out=suffix_out,
            )
            return step.mean + self.sample_noise(x_t.shape, x_t.device) * step.std
        if solver_type == "flow_cps":
            if noise_level is None:
                noise_level = self._get_noise_level(x_t.device, x_t.dtype)
            step = action_solvers.flow_cps_step(
                x_t=x_t,
                velocity=velocity,
                idx=idx,
                timesteps=timesteps,
                noise_level=noise_level,
            )
            return step.mean + self.sample_noise(x_t.shape, x_t.device) * step.std
        if solver_type == "flow_grpo":
            step = action_solvers.flow_grpo_step(
                x_t=x_t,
                velocity=velocity,
                idx=idx,
                timesteps=timesteps,
                eta=self.config.solver_eta,
            )
            return step.mean + self.sample_noise(x_t.shape, x_t.device) * step.std
        if solver_type == "dance":
            step = action_solvers.dance_step(
                x_t=x_t,
                velocity=velocity,
                idx=idx,
                timesteps=timesteps,
                eta=self.config.solver_eta,
            )
            return step.mean + self.sample_noise(x_t.shape, x_t.device) * step.std
        if solver_type == "ddim":
            step = action_solvers.ddim_step(
                x_t=x_t,
                velocity=velocity,
                idx=idx,
                timesteps=timesteps,
                eta=self.config.solver_eta,
            )
            return step.mean + self.sample_noise(x_t.shape, x_t.device) * step.std
        if solver_type == "dpm":
            if "dpm_state" not in solver_state:
                solver_state["dpm_state"] = action_solvers.DPMState(
                    order=self.config.solver_order
                )
            step = action_solvers.dpm_step(
                x_t=x_t,
                velocity=velocity,
                idx=idx,
                timesteps=timesteps,
                order=self.config.solver_order,
                dpm_state=solver_state["dpm_state"],
            )
            solver_state["dpm_state"] = step.state.get(
                "dpm_state", solver_state["dpm_state"]
            )
            return step.sample
        raise ValueError(f"未识别的 solver 类型：{solver_type}")

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        noise_level = self._get_noise_level(device=device, dtype=x_t.dtype)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device, dtype=x_t.dtype)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device, dtype=x_t.dtype)])
        if isinstance(idx, int):
            idx_tensor = torch.full((bsize,), idx, device=device, dtype=torch.long)
        else:
            idx_tensor = idx.to(device=device, dtype=torch.long)
        # velocity prediction
        v_t, suffix_out = self.get_velocity(
            state,
            x_t,
            timesteps[idx_tensor],
            prefix_pad_masks,
            past_key_values
        )
        
        # value prediction
        value_t = self._predict_critic_value(
            suffix_out=suffix_out,
            compute_values=compute_values,
            device=device,
        )
        if mode == "eval":
            dt = timesteps[idx_tensor + 1] - timesteps[idx_tensor]
            step_result = action_solvers.euler_step(
                x_t=x_t,
                velocity=v_t,
                dt=dt[:, None, None],
            )
            x_t_mean = step_result.sample
            x_t_std = torch.zeros_like(x_t_mean)
        else:
            assert self.config.solver_type in {"flow_sde", "flow_noise"}, (
                "sample_mean_var_val currently only supports flow_sde/flow_noise."
            )
            if self.config.solver_type == "flow_sde":
                step_result = action_solvers.flow_sde_step(
                    x_t=x_t,
                    velocity=v_t,
                    idx=idx,
                    timesteps=timesteps,
                    noise_level=noise_level,
                )
            elif self.config.solver_type == "flow_noise":
                if not hasattr(self, "noise_head"):
                    raise ValueError("flow_noise 需要噪声头 ExploreNoiseNet。")
                step_result = action_solvers.flow_noise_step(
                    x_t=x_t,
                    velocity=v_t,
                    idx=idx,
                    timesteps=timesteps,
                    noise_head=self.noise_head,
                    suffix_out=suffix_out,
                )
            else:
                raise ValueError(f"Invalid noise method: {self.config.solver_type}")
            x_t_mean = step_result.mean
            x_t_std = step_result.std
        return x_t_mean, x_t_std, value_t

    def get_velocity(
        self,
        state,
        x_t,
        timestep,
        prefix_pad_masks,
        past_key_values,
    ):
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            timestep,
        )
        v_t = self.action_out_proj(
            suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        )
        return v_t, suffix_out

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Compute image and language key value cache
        [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # get log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if not self.use_vlm_value:
                chains_values.append(value_t)
        if self.use_vlm_value:
            chains_values.append(self.get_value_from_vlm(prefix_output))
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
