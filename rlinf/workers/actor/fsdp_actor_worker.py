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

import gc
import math
import os
import time
from contextlib import nullcontext
from functools import partial
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp import FSDP
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
    compute_time_decay_weights,
)
from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
    stack_list_of_dict_tensor,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper

_ADV_INPUT_LOG_STATE = {"count": 0}


def _log_terminal_binary_adv_inputs(
    rank: int,
    rewards: torch.Tensor | None,
    dones: torch.Tensor | None,
    success_once: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
) -> None:
    if _ADV_INPUT_LOG_STATE["count"] >= 3:
        return
    _ADV_INPUT_LOG_STATE["count"] += 1
    with torch.no_grad():
        def _stats(t: torch.Tensor | None):
            if t is None:
                return ("<none>", 0, 0.0, 0.0)
            nz = int((t != 0).sum().item())
            t_min = float(t.min().item()) if t.numel() > 0 else 0.0
            t_max = float(t.max().item()) if t.numel() > 0 else 0.0
            return (tuple(t.shape), nz, t_min, t_max)

        r_shape, r_nz, r_min, r_max = _stats(rewards)
        d_shape, d_nz, d_min, d_max = _stats(dones)
        s_shape, s_nz, s_min, s_max = _stats(success_once)
        m_shape, m_nz, m_min, m_max = _stats(loss_mask)
        print(
            "[adv][input] "
            f"rank={rank} rewards_shape={r_shape} rewards_nz={r_nz} "
            f"rewards_min={r_min:.3f} rewards_max={r_max:.3f} "
            f"dones_shape={d_shape} dones_nz={d_nz} dones_min={d_min:.3f} dones_max={d_max:.3f} "
            f"success_once_shape={s_shape} success_once_nz={s_nz} "
            f"success_once_min={s_min:.3f} success_once_max={s_max:.3f} "
            f"loss_mask_shape={m_shape} loss_mask_nz={m_nz} "
            f"loss_mask_min={m_min:.3f} loss_mask_max={m_max:.3f}",
            flush=True,
        )

_TERMINAL_BINARY_LOSS_LOG = {"count": 0}


def _log_terminal_binary_loss_inputs(
    rank: int,
    advantages: torch.Tensor,
    returns: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    adv_clip_max: float | None,
) -> None:
    if _TERMINAL_BINARY_LOSS_LOG["count"] >= 3:
        return
    _TERMINAL_BINARY_LOSS_LOG["count"] += 1
    with torch.no_grad():
        mask = loss_mask if loss_mask is not None else torch.ones_like(advantages)
        mask = mask.to(dtype=torch.bool)
        if mask.shape != advantages.shape:
            mask = mask.expand_as(advantages)
        masked_adv = advantages[mask]
        adv_min = float(masked_adv.min().item()) if masked_adv.numel() > 0 else 0.0
        adv_max = float(masked_adv.max().item()) if masked_adv.numel() > 0 else 0.0
        adv_mean = float(masked_adv.mean().item()) if masked_adv.numel() > 0 else 0.0
        if returns is not None:
            masked_ret = returns[mask]
            ret_min = float(masked_ret.min().item()) if masked_ret.numel() > 0 else 0.0
            ret_max = float(masked_ret.max().item()) if masked_ret.numel() > 0 else 0.0
            ret_unique = (
                torch.unique(masked_ret).detach().cpu().tolist()
                if masked_ret.numel() > 0
                else []
            )
        else:
            ret_min = ret_max = 0.0
            ret_unique = []
        print(
            "[loss][terminal-binary] "
            f"adv_clip_max={adv_clip_max} adv_min={adv_min:.3f} "
            f"adv_max={adv_max:.3f} adv_mean={adv_mean:.3f} "
            f"ret_min={ret_min:.3f} ret_max={ret_max:.3f} ret_unique={ret_unique} "
            f"src={__file__}",
            flush=True,
        )


def nft_return_decay(
    step: int, total_steps: int, base: float = 0.1, target: float = 0.8
) -> float:
    if total_steps <= 0:
        return 1.0

    progress = min(max(step / total_steps, 0.0), 1.0)
    cosine_val = math.cos(progress * math.pi)
    decay = target - (target - base) * 0.5 * (1 + cosine_val)
    return float(decay)


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


def get_nested_k_split_for_specific_keys(nested_dict, num_splits, key_list):
    """
    Get k-split iterator for some keys in nested_dict.
    """
    extra_dict = {}
    for key in key_list:
        if key not in nested_dict.keys():
            continue
        value = nested_dict[key]
        if isinstance(value, dict):
            extra_dict[key] = split_dict_to_chunk(value, num_splits)
        elif isinstance(value, torch.Tensor):
            continue
        else:
            raise NotImplementedError(
                f"Only support dict and tensor type, but got {type(value)}"
            )
    # {key1: [d1, d2, ...], key2: [d1, d2, ...]} -> [{key1: d1, key2: d1}, {key1: d2, key2: d2}, ...]
    extra_list = [
        {k: extra_dict[k][i] for k in extra_dict.keys()} for i in range(num_splits)
    ]
    return extra_list


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            self.cfg.actor.model.encoder_seq_length - self.cfg.data.max_prompt_length
        )
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(self.cfg.algorithm.loss_agg_func)
        self.enable_offload = (
            self.cfg.actor.get("enable_offload", False) and not self.is_pipeline
        )
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.n_mini_batches = self.cfg.algorithm.n_minibatches
        self.task_type = self.cfg.runner.task_type
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "liger_kernel")

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if self.cfg.algorithm.kl_beta > 0 and self.cfg.actor.get(
            "combine_reference_model", True
        ):
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        if hasattr(self, "rollout_state_dict"):
            del self.rollout_state_dict
        clear_memory(sync=False)

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not hasattr(self._strategy, "setup_actor_sync_inference_ranks"):
            if self._rank == 0:
                self.log_info("[sync] inference sync disabled; skipping.")
            return
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device, True)

        self.rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=True
        )

        has_visual = any("visual." in k for k in self.rollout_state_dict.keys())

        state_dict = {}

        if self._weight_dst_rank_in_rollout is not None:
            for k, v in self.rollout_state_dict.items():
                name = k
                if has_visual:
                    if name.startswith("model.language_model."):
                        name = "model." + name[21:]
                    # NOTE:
                    # if transformers version is 4.56.1 or older(not tested),
                    # the following line should be uncommented

                    # elif name.startswith("model."):
                    #     name = name[6:]
                state_dict[name] = reduce_tensor(v) if not self.is_pipeline else v
            if not self.is_pipeline:
                self.send(
                    state_dict,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                )
            else:
                for weight_dst_rank in self._weight_dst_rank_in_rollout:
                    self.send(
                        state_dict,
                        self._rollout_group_name,
                        weight_dst_rank,
                    )

        state_dict.clear()
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    @torch.no_grad()
    def inference_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **multi_modal_inputs,
        )

        logits = outputs.logits
        logits = logits[:, -self.response_len - 1 : -1, :]
        logits = logits / self.cfg.algorithm.sampling_params.temperature

        responses = input_ids[:, -self.response_len :]
        logprobs = compute_logprobs_from_logits(
            logits=logits, target=responses, op_type=self.entropy_op_type
        )
        return logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ) -> None:
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence
            self._load_weight_and_optimizer()

            num_splits = (
                rollout_result.num_sequence
                // self.cfg.algorithm.logprob_forward_micro_batch_size
            )
            micro_batches_iter = get_iterator_k_split(
                batch,
                num_splits=num_splits,
            )
            micro_batches = list(micro_batches_iter)

            prev_logprobs = []
            with self.worker_timer():
                for micro_batch in micro_batches:
                    prev_logprobs.append(self.inference_step(micro_batch).cpu())

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = torch.cat(prev_logprobs)
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = torch.cat(prev_logprobs)

            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None, (
                    "Reference policy state dict is None but compute_ref_logprobs is True"
                )
                ref_logprobs = []
                with cpu_weight_swap(self.model, self.ref_policy_state_dict):
                    for micro_batch in micro_batches:
                        ref_logprobs.append(self.inference_step(micro_batch).cpu())
                    rollout_result.ref_logprobs = torch.cat(ref_logprobs)

            output_channel.put(rollout_result)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt
            micro_batches = get_iterator_k_split(batch, micro_batch_cnt)
            micro_batches_iter = iter(micro_batches)
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
            )
            for k, v in m_batch.items():
                m_batch[k] = v.cuda() if isinstance(v, torch.Tensor) else v

            multi_modal_inputs = {}
            if "multi_modal_inputs" in m_batch.keys():
                for key in m_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                        dim=0,
                    ).cuda()

            input_ids = m_batch["input_ids"]
            attention_mask = m_batch["attention_mask"]
            position_ids = m_batch["position_ids"]
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            with self.amp_context:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                logits: torch.Tensor = output.logits

                logits.div_(self.cfg.algorithm.sampling_params.temperature)

                responses = input_ids[:, -self.response_len :]
                logits = logits[
                    :, -self.response_len - 1 : -1, :
                ]  # (bsz, response_length, vocab_size)
                logprobs = compute_logprobs_from_logits(
                    logits, responses, self.entropy_op_type
                )

                if self.cfg.algorithm.get("importance_sampling_fix", False):
                    rollout_prev_logprobs = prev_logprobs
                    recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                    advantages = advantages * torch.clamp(
                        (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                        min=self.cfg.algorithm.importance_sampling_clip,
                    )

                if self.cfg.algorithm.adv_type == "terminal-binary":
                    _log_terminal_binary_loss_inputs(
                        rank=self._rank,
                        advantages=advantages,
                        returns=m_batch.get("returns", None),
                        loss_mask=loss_mask,
                        adv_clip_max=self.cfg.algorithm.get(
                            "clip_ratio_high", clip_ratio_high
                        ),
                    )

                loss, mbs_metrics_data = policy_loss(
                    loss_type=self.cfg.algorithm.loss_type,
                    loss_agg_func=self.loss_agg_func,
                    logprobs=logprobs,
                    old_logprobs=prev_logprobs,
                    advantages=advantages,
                    clip_ratio_low=clip_ratio_low,
                    clip_ratio_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_mask=loss_mask,
                    task_type=self.task_type,
                )

                entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.calculate_entropy:
                    entropy = compute_entropy_from_logits(
                        logits,
                    )

                    entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                    if self.calculate_entropy_loss:
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.kl_beta > 0 and ref_logprobs is not None:
                    kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                    kl_loss = self.loss_agg_func(kld, loss_mask)
                    loss = loss + kl_loss * self.kl_beta

                # add to log
                # scale loss for gradient accumulation and backprop
                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": loss.detach(),
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()
        return mbs_metrics_list, grad_norm, lr_list

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        self._preserve_frozen_vlm_eval_mode(self.model)
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer():
            for _ in range(self.n_mini_batches):
                metrics, grad_norm, lr_list = self.training_step(
                    batch=train_batch_iterator
                )

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def run_training(self, input_channel: Channel) -> tuple[dict, list]:
        # Get all batches for this DP
        if self.is_pipeline:
            with self.worker_timer():
                return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                metrics, grad_norm, lr_list = self.training_step(batch=mini_batch)

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].cuda(),
                    loss_mask=mask.cuda(),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.cfg.algorithm.get("reinpp_kl_beta", 0.0),
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].cuda()
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].cuda()
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        import warnings

        warnings.filterwarnings(
            "ignore",
            message=".*When using ``NO_SHARD`` for ``ShardingStrategy``.*",
        )
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self.global_step = 0
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        self.ref_model = None
        self.prev_rollout_model = None
        self.prm_model = None
        self._value_head_sync_ready = False
        self._shared_ref_param_names: set[str] = set()
        self._shared_prev_rollout_param_names: set[str] = set()
        self._shared_prm_param_names: set[str] = set()
        self._enable_mem_log = bool(getattr(self.cfg.actor, "enable_mem_log", False))
        self._update_ready = False
        self._student_param_snapshot = None
        self._student_param_snapshot_init = None
        self._watch_param_names = [
            "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.layer_norm1.weight",
        ]
        self._logged_terminal_binary_loss = False
        self._branch_recipe_bank: dict[int, list[dict[str, Any]]] = {}
        self._synthetic_branch_entries: list[dict[str, Any]] = []
        self._last_branch_plan_metrics: dict[str, float] = {}
        self._last_branch_metrics: dict[str, float] = {}
        self._aux_models_offloaded = False
        self._prm_optimizer_state_offloaded = False

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )

    def _log_cuda_memory(self, tag: str) -> None:
        if not self._enable_mem_log or not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
        print(
            f"[Mem] {tag} allocated_gb={allocated:.2f} reserved_gb={reserved:.2f} "
            f"max_allocated_gb={max_alloc:.2f}"
        )

    def _is_primary_progress_rank(self) -> bool:
        return int(getattr(self, "_rank", 0)) == 0

    def _log_progress(self, message: str) -> None:
        if self._is_primary_progress_rank():
            self.log_info(message)

    @staticmethod
    def _should_log_progress(
        current: int,
        total: int,
        last_log_time: float,
        now: float,
        *,
        every_n: int = 0,
        every_s: float = 120.0,
    ) -> bool:
        if total <= 0:
            return False
        if current <= 1 or current >= total:
            return True
        if every_n > 0 and current % every_n == 0:
            return True
        return every_s > 0 and (now - last_log_time) >= every_s

    def _log_vlm_paths(self, model: torch.nn.Module, tag: str) -> None:
        candidates = [
            "paligemma_with_expert",
            "paligemma_with_expert.paligemma",
            "paligemma_with_expert.paligemma.language_model",
            "paligemma_with_expert.paligemma.vision_model",
            "paligemma_with_expert.paligemma.vision_tower",
            "paligemma_with_expert.gemma_expert",
        ]
        found = []
        for path in candidates:
            cur = model
            ok = True
            for part in path.split("."):
                if not hasattr(cur, part):
                    ok = False
                    break
                cur = getattr(cur, part)
            if ok:
                found.append(path)
        print(f"[VLM Path] {tag} candidates: {found}")

        likely = []
        for name, module in model.named_modules():
            cls_name = module.__class__.__name__.lower()
            if any(k in cls_name for k in ("paligemma", "siglip", "vision")):
                likely.append(name)
        if likely:
            print(f"[VLM Path] {tag} named_modules (sample): {likely[:10]}")

    def _preserve_frozen_vlm_eval_mode(self, model: torch.nn.Module) -> None:
        inner = model.module if hasattr(model, "module") else model
        if hasattr(inner, "freeze_vlm"):
            inner.freeze_vlm()

    @torch.no_grad()
    def _move_optimizer_state(
        self,
        optimizer: torch.optim.Optimizer | None,
        device: torch.device | str,
    ) -> None:
        if optimizer is None or not optimizer.state:
            return
        target_device = torch.device(device)
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                state = optimizer.state.get(param, None)
                if not state:
                    continue
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(target_device, non_blocking=True)

    @torch.no_grad()
    def _move_aux_model_nonshared_tensors(
        self,
        model: torch.nn.Module | None,
        shared_param_names: set[str],
        device: torch.device | str,
    ) -> None:
        if model is None:
            return
        target_device = torch.device(device)
        for name, param in model.named_parameters():
            if name in shared_param_names:
                continue
            if param.data is not None and param.data.device != target_device:
                param.data = param.data.to(target_device, non_blocking=True)
            if param.grad is not None and param.grad.device != target_device:
                param.grad = param.grad.to(target_device, non_blocking=True)
        for _, buffer in model.named_buffers():
            if buffer is not None and buffer.device != target_device:
                buffer.data = buffer.data.to(target_device, non_blocking=True)

    def _offload_aux_models_for_actor_loop(self) -> None:
        if self._aux_models_offloaded and self._prm_optimizer_state_offloaded:
            return
        self._move_aux_model_nonshared_tensors(
            self.ref_model,
            self._shared_ref_param_names,
            "cpu",
        )
        self._move_aux_model_nonshared_tensors(
            self.prev_rollout_model,
            self._shared_prev_rollout_param_names,
            "cpu",
        )
        self._move_aux_model_nonshared_tensors(
            self.prm_model,
            self._shared_prm_param_names,
            "cpu",
        )
        self._move_optimizer_state(self.prm_optimizer, "cpu")
        self._aux_models_offloaded = True
        self._prm_optimizer_state_offloaded = True
        clear_memory()

    def _onload_aux_models_for_prm(self) -> None:
        if not self._aux_models_offloaded and not self._prm_optimizer_state_offloaded:
            return
        self._move_aux_model_nonshared_tensors(
            self.ref_model,
            self._shared_ref_param_names,
            self.device,
        )
        self._move_aux_model_nonshared_tensors(
            self.prev_rollout_model,
            self._shared_prev_rollout_param_names,
            self.device,
        )
        self._move_aux_model_nonshared_tensors(
            self.prm_model,
            self._shared_prm_param_names,
            self.device,
        )
        self._move_optimizer_state(self.prm_optimizer, self.device)
        self._aux_models_offloaded = False
        self._prm_optimizer_state_offloaded = False
        clear_memory()

    def _share_vlm_from_student(
        self, target_model: torch.nn.Module, tag: str
    ) -> set[str]:
        shared_names: set[str] = set()
        use_orig_params = bool(
            getattr(self.cfg.actor.fsdp_config, "use_orig_params", False)
        )
        fsdp_use_orig = bool(getattr(self.model, "_use_orig_params", False))
        if isinstance(self.model, FSDP) and not (use_orig_params or fsdp_use_orig):
            share_ctx = FSDP.summon_full_params(
                self.model, writeback=False, recurse=True
            )
            if self._rank == 0:
                print(
                    f"[Memory Opt] {tag} VLM share via summon_full_params "
                    "(use_orig_params=False)."
                )
        else:
            share_ctx = nullcontext()

        with share_ctx:
            student_inner = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            if (
                hasattr(student_inner, "paligemma_with_expert")
                and hasattr(target_model, "paligemma_with_expert")
                and hasattr(student_inner.paligemma_with_expert, "paligemma")
            ):
                target_vlm = target_model.paligemma_with_expert.paligemma

                student_params_alias: dict[str, torch.nn.Parameter] = {}
                for name, param in student_inner.named_parameters():
                    if not name.startswith("paligemma_with_expert.paligemma."):
                        continue
                    suffix = name[len("paligemma_with_expert.paligemma.") :]
                    if suffix.startswith("_fsdp_wrapped_module."):
                        suffix = suffix[len("_fsdp_wrapped_module.") :]
                    suffix = suffix.replace("._fsdp_wrapped_module.", ".")
                    suffix = suffix.replace("._fsdp_wrapped_module", "")
                    student_params_alias.setdefault(suffix, param)
                    if suffix.startswith("model."):
                        student_params_alias.setdefault(suffix[len("model.") :], param)
                    else:
                        student_params_alias.setdefault(f"model.{suffix}", param)

                tied = 0
                missing = 0
                missing_names = []
                for name, target_param in target_vlm.named_parameters():
                    src_param = student_params_alias.get(name, None)
                    if src_param is None or src_param.shape != target_param.shape:
                        missing += 1
                        if len(missing_names) < 5:
                            missing_names.append(name)
                        continue
                    target_param.data = src_param.data
                    target_param.requires_grad = False
                    tied += 1

                shared_ptrs = {p.data_ptr() for p in student_params_alias.values()}
                shared_names = {
                    name
                    for name, p in target_model.named_parameters()
                    if p.data_ptr() in shared_ptrs
                }
                print(
                    f"[Memory Opt] Shared VLM weights between student/{tag}. "
                    f"shared_vlm_params={len(shared_names)} "
                    f"tied={tied} missing_or_mismatch={missing} "
                    f"student_alias_keys={len(student_params_alias)}"
                )
                if missing_names:
                    print(
                        "[Memory Opt] VLM share missing/mismatch sample: "
                        f"{missing_names}"
                    )
            else:
                print(
                    f"[Warning] Could not share VLM weights for {tag}. "
                    "Missing paligemma_with_expert.paligemma."
                )
        return shared_names

    def _filter_shared_state(
        self, state_dict: dict, shared_param_names: set[str]
    ) -> dict:
        if not shared_param_names:
            return state_dict
        shared_prefixes = ("paligemma_with_expert.paligemma.",)
        return {
            name: value
            for name, value in state_dict.items()
            if name not in shared_param_names
            and not any(name.startswith(prefix) for prefix in shared_prefixes)
        }

    def _copy_snapshot_state(
        self,
        target_model: torch.nn.Module,
        source_model: torch.nn.Module,
        shared_param_names: set[str],
        tag: str,
    ) -> None:
        source_state = {
            k: v.detach() if torch.is_tensor(v) else v
            for k, v in source_model.state_dict().items()
        }
        source_state = self._filter_shared_state(source_state, shared_param_names)
        missing, unexpected = target_model.load_state_dict(source_state, strict=False)
        if self._rank == 0 and (missing or unexpected):
            self.log_info(
                f"[{tag}] snapshot copy mismatch: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

    def _maybe_log_terminal_binary_loss_inputs(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor | None,
        loss_mask: torch.Tensor | None,
        adv_clip_max: float | None = None,
    ) -> None:
        if self._logged_terminal_binary_loss:
            return
        if self._rank != 0:
            return
        with torch.no_grad():
            mask = loss_mask if loss_mask is not None else torch.ones_like(advantages)
            mask = mask.to(dtype=torch.bool)
            if mask.shape != advantages.shape:
                mask = mask.expand_as(advantages)
            masked_adv = advantages[mask]
            adv_min = float(masked_adv.min().item()) if masked_adv.numel() > 0 else 0.0
            adv_max = float(masked_adv.max().item()) if masked_adv.numel() > 0 else 0.0
            adv_mean = (
                float(masked_adv.mean().item()) if masked_adv.numel() > 0 else 0.0
            )
            if returns is not None:
                masked_ret = returns[mask]
                ret_min = (
                    float(masked_ret.min().item()) if masked_ret.numel() > 0 else 0.0
                )
                ret_max = (
                    float(masked_ret.max().item()) if masked_ret.numel() > 0 else 0.0
                )
                ret_unique = (
                    torch.unique(masked_ret).detach().cpu().tolist()
                    if masked_ret.numel() > 0
                    else []
                )
            else:
                ret_min = ret_max = 0.0
                ret_unique = []
            print(
                "[loss][terminal-binary] "
                f"adv_clip_max={adv_clip_max} adv_min={adv_min:.3f} "
                f"adv_max={adv_max:.3f} adv_mean={adv_mean:.3f} "
                f"ret_min={ret_min:.3f} ret_max={ret_max:.3f} ret_unique={ret_unique}",
                flush=True,
            )
        self._logged_terminal_binary_loss = True


    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        if self.cfg.algorithm.loss_type.startswith("nft"):
            with open_dict(self.cfg):
                if self.cfg.actor.model.get("add_value_head", False):
                    if self.cfg.actor.fsdp_config.get("wrap_value_head", True):
                        self.cfg.actor.fsdp_config.wrap_value_head = False
                        if self._rank == 0:
                            print(
                                "[FSDP] Disabling value_head auto-wrap to avoid shape writeback errors."
                            )

        self.setup_model_and_optimizer()
        self._log_cuda_memory("init/after_setup")

        if self.cfg.algorithm.kl_beta > 0 or self.cfg.algorithm.loss_type.startswith(
            "nft"
        ):
            import gc

            ref_model = get_model(self.cfg.actor.model)
            if ref_model is None:
                ref_model = super().model_provider_func()

            if self.cfg.runner.get("ckpt_path", None):
                model_dict = torch.load(self.cfg.runner.ckpt_path)
                ref_model.load_state_dict(model_dict)

            gc.collect()
            print("[Memory Opt] Moving Action Expert to GPU...")
            ref_model.to(self.device)
            self._log_cuda_memory("init/after_ref_model_to_gpu")

            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

            self.ref_model = ref_model

            share_vlm = bool(getattr(self.cfg.actor, "share_vlm_with_ref", True))
            if share_vlm:
                self._shared_ref_param_names = self._share_vlm_from_student(
                    self.ref_model, "ref"
                )
                if len(self._shared_ref_param_names) == 0:
                    print(
                        "[Warning] VLM share attempted for ref_model but no shared params detected."
                    )
                else:
                    sample_names = sorted(self._shared_ref_param_names)[:5]
                    print(
                        "[Memory Opt] Shared VLM param sample: "
                        f"{sample_names}"
                    )
            else:
                print("[Memory Opt] VLM share disabled via actor.share_vlm_with_ref.")
            self._log_cuda_memory("init/after_vlm_share")

            torch.cuda.empty_cache()
            self._log_cuda_memory("init/after_empty_cache")
            try:
                student_param_ptrs = {p.data_ptr() for p in self.model.parameters()}
                ref_param_ptrs = {p.data_ptr() for p in self.ref_model.parameters()}
                shared_param_count = len(student_param_ptrs & ref_param_ptrs)
                print(
                    "[Memory Opt] Reference Model loaded. "
                    f"Shared parameter count with student: {shared_param_count}"
                )
            except Exception as e:
                print(f"[Memory Opt] Reference Model loaded. Share check failed: {e}")
            self._log_vlm_paths(self.model, "student")
            self._log_vlm_paths(self.ref_model, "ref")

        # --- PRM model for dual-credit ---
        self.prm_model = None
        self.prm_optimizer = None
        if self.cfg.algorithm.loss_type == "nft-dual-credit":
            import gc as _gc
            prev_model = get_model(self.cfg.actor.model)
            if prev_model is None:
                prev_model = super().model_provider_func()
            if self.cfg.runner.get("ckpt_path", None):
                prev_model.load_state_dict(torch.load(self.cfg.runner.ckpt_path))
            _gc.collect()
            prev_model.to(self.device)
            prev_model.eval()
            for p in prev_model.parameters():
                p.requires_grad = False
            self.prev_rollout_model = prev_model
            if share_vlm:
                self._shared_prev_rollout_param_names = self._share_vlm_from_student(
                    self.prev_rollout_model, "prev_rollout"
                )
            self._copy_snapshot_state(
                self.prev_rollout_model,
                self.ref_model,
                self._shared_prev_rollout_param_names,
                tag="init/prev_from_ref",
            )

            prm_model = get_model(self.cfg.actor.model)
            if prm_model is None:
                prm_model = super().model_provider_func()
            if self.cfg.runner.get("ckpt_path", None):
                prm_model.load_state_dict(torch.load(self.cfg.runner.ckpt_path))
            _gc.collect()
            prm_model.to(self.device)
            prm_model.train()
            self._preserve_frozen_vlm_eval_mode(prm_model)
            self.prm_model = prm_model

            # share VLM weights with student (same pattern as ref_model)
            if share_vlm:
                self._shared_prm_param_names = self._share_vlm_from_student(
                    prm_model, "prm"
                )

            # only optimise action expert params (not shared VLM)
            prm_params = [p for p in self.prm_model.parameters() if p.requires_grad]
            prm_lr = self.cfg.algorithm.get("prm_lr", 1e-5)
            self.prm_optimizer = torch.optim.AdamW(prm_params, lr=prm_lr, weight_decay=0.01)
            if self._rank == 0:
                print(f"[Dual-Credit] PRM model created. trainable params={sum(p.numel() for p in prm_params)}")
            self._log_cuda_memory("init/after_prm_model")

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _ref_checkpoint_path(self, base_path: str) -> str:
        return os.path.join(base_path, "ref_model.pt")

    def _prev_rollout_checkpoint_path(self, base_path: str) -> str:
        return os.path.join(base_path, "prev_rollout_model.pt")

    def _prm_checkpoint_path(self, base_path: str) -> str:
        return os.path.join(base_path, "prm_model.pt")

    def _prm_optimizer_checkpoint_path(self, base_path: str) -> str:
        return os.path.join(base_path, "prm_optimizer.pt")

    def _resume_signature_path(self, base_path: str) -> str:
        return os.path.join(base_path, "resume_signature.pt")

    def _runtime_state_path(self, base_path: str) -> str:
        return os.path.join(base_path, "runtime_state.pt")

    @staticmethod
    def _serialize_branch_recipe_bank(
        bank: dict[int, list[dict[str, Any]]],
    ) -> dict[int, list[dict[str, Any]]]:
        serialized: dict[int, list[dict[str, Any]]] = {}
        for task_id, recipes in bank.items():
            task_key = int(task_id)
            task_bucket: list[dict[str, Any]] = []
            for recipe in recipes:
                item: dict[str, Any] = {}
                for key, value in recipe.items():
                    if torch.is_tensor(value):
                        item[key] = value.detach().cpu().contiguous()
                    else:
                        item[key] = value
                task_bucket.append(item)
            serialized[task_key] = task_bucket
        return serialized

    @staticmethod
    def _deserialize_branch_recipe_bank(
        state: dict[Any, list[dict[str, Any]]] | None,
    ) -> dict[int, list[dict[str, Any]]]:
        restored: dict[int, list[dict[str, Any]]] = {}
        if not state:
            return restored
        for task_id, recipes in state.items():
            try:
                task_key = int(task_id)
            except Exception:
                continue
            bucket: list[dict[str, Any]] = []
            for recipe in recipes:
                item: dict[str, Any] = {}
                for key, value in recipe.items():
                    if torch.is_tensor(value):
                        item[key] = value.detach().cpu().float().contiguous()
                    else:
                        item[key] = value
                if "delta_action" in item and not torch.is_tensor(item["delta_action"]):
                    item["delta_action"] = torch.as_tensor(
                        item["delta_action"], dtype=torch.float32
                    ).cpu().contiguous()
                bucket.append(item)
            restored[task_key] = bucket
        return restored

    def _build_state_signature(self, state_dict: dict, max_keys: int = 8) -> dict:
        signature = {}
        if not state_dict:
            return signature
        keys = sorted(state_dict.keys())[:max_keys]
        for name in keys:
            value = state_dict[name]
            if not torch.is_tensor(value):
                continue
            tensor = value.detach().float().cpu()
            signature[name] = {
                "shape": tuple(tensor.shape),
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()),
            }
        return signature

    def _compare_signature(self, saved: dict, current: dict, tol: float = 1e-3) -> list:
        mismatches = []
        for name, saved_stats in saved.items():
            cur_stats = current.get(name)
            if cur_stats is None:
                mismatches.append(f"{name}: missing_current")
                continue
            if saved_stats.get("shape") != cur_stats.get("shape"):
                mismatches.append(f"{name}: shape")
                continue
            for field in ("mean", "std"):
                saved_val = saved_stats.get(field)
                cur_val = cur_stats.get(field)
                if saved_val is None or cur_val is None:
                    mismatches.append(f"{name}: {field}_missing")
                    break
                if abs(saved_val - cur_val) > tol:
                    mismatches.append(f"{name}: {field}")
                    break
        return mismatches

    def save_checkpoint(self, save_path: str, global_steps: int) -> None:
        super().save_checkpoint(save_path, global_steps)
        if not self.cfg.algorithm.loss_type.startswith("nft"):
            return
        if self.ref_model is None:
            return
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self._rank == 0:
            ref_state = {
                k: v.detach().cpu() if torch.is_tensor(v) else v
                for k, v in self.ref_model.state_dict().items()
            }
            ref_state = self._filter_shared_state(ref_state, self._shared_ref_param_names)
            torch.save(ref_state, self._ref_checkpoint_path(save_path))
            prev_state = None
            if self.prev_rollout_model is not None:
                prev_state = {
                    k: v.detach().cpu() if torch.is_tensor(v) else v
                    for k, v in self.prev_rollout_model.state_dict().items()
                }
                prev_state = self._filter_shared_state(
                    prev_state, self._shared_prev_rollout_param_names
                )
                torch.save(prev_state, self._prev_rollout_checkpoint_path(save_path))
            prm_state = None
            if self.prm_model is not None:
                prm_state = {
                    k: v.detach().cpu() if torch.is_tensor(v) else v
                    for k, v in self.prm_model.state_dict().items()
                }
                prm_state = self._filter_shared_state(
                    prm_state, self._shared_prm_param_names
                )
                torch.save(prm_state, self._prm_checkpoint_path(save_path))
            if self.prm_optimizer is not None:
                torch.save(
                    self.prm_optimizer.state_dict(),
                    self._prm_optimizer_checkpoint_path(save_path),
                )
            actor_state = self.model.state_dict()
            signature = {
                "actor": self._build_state_signature(actor_state),
                "ref_model": self._build_state_signature(ref_state),
            }
            if prev_state is not None:
                signature["prev_rollout_model"] = self._build_state_signature(prev_state)
            if prm_state is not None:
                signature["prm_model"] = self._build_state_signature(prm_state)
            torch.save(signature, self._resume_signature_path(save_path))
            runtime_state = {
                "branch_recipe_bank": self._serialize_branch_recipe_bank(
                    self._branch_recipe_bank
                ),
                "optimizer_steps": int(getattr(self, "optimizer_steps", 0)),
                "global_step": int(getattr(self, "global_step", global_steps)),
                "critic_warmup_finished": bool(
                    getattr(self, "_critic_warmup_finished", True)
                ),
                "value_head_frozen": bool(getattr(self, "value_head_frozen", False)),
            }
            torch.save(runtime_state, self._runtime_state_path(save_path))
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)
        if not self.cfg.algorithm.loss_type.startswith("nft"):
            return
        if self.ref_model is None:
            return
        ref_path = self._ref_checkpoint_path(load_path)
        if not os.path.exists(ref_path):
            if self._rank == 0:
                self.log_info(f"[resume] ref_model checkpoint not found: {ref_path}")
            return
        ref_state = torch.load(ref_path, map_location="cpu")
        ref_state = self._filter_shared_state(ref_state, self._shared_ref_param_names)
        missing, unexpected = self.ref_model.load_state_dict(ref_state, strict=False)
        if self._rank == 0 and (missing or unexpected):
            self.log_info(
                "[resume] ref_model state dict mismatch: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            shared_cnt = len(self._shared_ref_param_names)
            extra_missing = max(len(missing) - shared_cnt, 0)
            self.log_info(
                "[resume] ref_model load stats: "
                f"ref_state_keys={len(ref_state)} "
                f"ref_model_params={sum(1 for _ in self.ref_model.parameters())} "
                f"shared_vlm_params={shared_cnt} extra_missing={extra_missing}"
            )
        if self.prev_rollout_model is not None:
            prev_path = self._prev_rollout_checkpoint_path(load_path)
            if os.path.exists(prev_path):
                prev_state = torch.load(prev_path, map_location="cpu")
                prev_state = self._filter_shared_state(
                    prev_state, self._shared_prev_rollout_param_names
                )
                self.prev_rollout_model.load_state_dict(prev_state, strict=False)
            elif self._rank == 0:
                self.log_info(
                    f"[resume] prev_rollout_model checkpoint not found: {prev_path}"
                )
        if self.prm_model is not None:
            prm_path = self._prm_checkpoint_path(load_path)
            if os.path.exists(prm_path):
                prm_state = torch.load(prm_path, map_location="cpu")
                prm_state = self._filter_shared_state(
                    prm_state, self._shared_prm_param_names
                )
                self.prm_model.load_state_dict(prm_state, strict=False)
            elif self._rank == 0:
                self.log_info(f"[resume] prm_model checkpoint not found: {prm_path}")
            prm_opt_path = self._prm_optimizer_checkpoint_path(load_path)
            if self.prm_optimizer is not None and os.path.exists(prm_opt_path):
                self.prm_optimizer.load_state_dict(
                    torch.load(prm_opt_path, map_location="cpu")
                )
                for state in self.prm_optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(self.device)
        signature_path = self._resume_signature_path(load_path)
        if self._rank == 0 and os.path.exists(signature_path):
            saved_signature = torch.load(signature_path, map_location="cpu")
            actor_sig = self._build_state_signature(self.model.state_dict())
            ref_sig = self._build_state_signature(self.ref_model.state_dict())
            prev_sig = (
                self._build_state_signature(self.prev_rollout_model.state_dict())
                if self.prev_rollout_model is not None
                else {}
            )
            prm_sig = (
                self._build_state_signature(self.prm_model.state_dict())
                if self.prm_model is not None
                else {}
            )
            actor_mismatches = self._compare_signature(
                saved_signature.get("actor", {}), actor_sig
            )
            ref_mismatches = self._compare_signature(
                saved_signature.get("ref_model", {}), ref_sig
            )
            prev_mismatches = self._compare_signature(
                saved_signature.get("prev_rollout_model", {}), prev_sig
            )
            prm_mismatches = self._compare_signature(
                saved_signature.get("prm_model", {}), prm_sig
            )
            if actor_mismatches or ref_mismatches or prev_mismatches or prm_mismatches:
                self.log_info(
                    "[resume] signature mismatch: "
                    f"actor={actor_mismatches[:5]} "
                    f"ref_model={ref_mismatches[:5]} "
                    f"prev_rollout_model={prev_mismatches[:5]} "
                    f"prm_model={prm_mismatches[:5]}"
                )
            else:
                self.log_info(
                    "[resume] signature check passed for actor/ref_model/"
                    "prev_rollout_model/prm_model"
                )
        runtime_state_path = self._runtime_state_path(load_path)
        if os.path.exists(runtime_state_path):
            runtime_state = torch.load(runtime_state_path, map_location="cpu")
            self._branch_recipe_bank = self._deserialize_branch_recipe_bank(
                runtime_state.get("branch_recipe_bank")
            )
            self.optimizer_steps = int(runtime_state.get("optimizer_steps", 0))
            self.global_step = int(runtime_state.get("global_step", self.global_step))
            self._critic_warmup_finished = bool(
                runtime_state.get(
                    "critic_warmup_finished", getattr(self, "_critic_warmup_finished", True)
                )
            )
            should_freeze_value_head = bool(
                runtime_state.get("value_head_frozen", getattr(self, "value_head_frozen", False))
            )
            if should_freeze_value_head:
                self._maybe_freeze_value_head()
                self.value_head_frozen = True
            if self._rank == 0:
                self.log_info(
                    "[resume] runtime state restored: "
                    f"branch_bank_tasks={len(self._branch_recipe_bank)} "
                    f"branch_bank_entries={sum(len(v) for v in self._branch_recipe_bank.values())} "
                    f"optimizer_steps={self.optimizer_steps} "
                    f"global_step={self.global_step}"
                )
        elif self._rank == 0:
            self.log_info(
                f"[resume] runtime state checkpoint not found: {runtime_state_path}"
            )

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        self._log_cuda_memory("sync/before_send")
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        if (
            getattr(self.cfg.algorithm, "loss_type", "") == "nft-actor-critic"
            and self.ref_model is not None
        ):
            if self._value_head_sync_ready:
                if isinstance(self.model, FSDP) and not getattr(
                    self.model, "_is_root", False
                ):
                    if self._rank == 0:
                        self.log_info(
                            "[sync] skip value_head hard-copy (FSDP root not initialized yet)"
                        )
                else:
                    student_inner = (
                        self.model.module if hasattr(self.model, "module") else self.model
                    )
                    student_vh = getattr(student_inner, "value_head", None)
                    ref_vh = getattr(self.ref_model, "value_head", None)
                    if student_vh is not None and ref_vh is not None:
                        ref_vh.load_state_dict(student_vh.state_dict())
                        self.log_info(
                            "[sync] hard-copied value_head parameters from student to ref_model"
                        )
            else:
                if self._rank == 0:
                    self.log_info(
                        "[sync] value_head hard-copy skipped (training not started yet)"
                    )

        if (
            self.cfg.algorithm.loss_type.startswith("nft")
            and self.ref_model is not None
        ):
            state_dict = self.ref_model.state_dict()
        else:
            state_dict = self.get_model_state_dict(
                cpu_offload=False, full_state_dict=True
            )

        sync_to_cpu = bool(getattr(self.cfg.rollout, "sync_weights_to_cpu", True))
        if sync_to_cpu:
            state_dict = {
                k: v.detach().to(device="cpu", non_blocking=True).contiguous()
                if torch.is_tensor(v)
                else v
                for k, v in state_dict.items()
            }
        for rank in self._weight_dst_rank_in_rollout:
            self.send(
                state_dict,
                self._rollout_group_name,
                rank,
                async_op=True,
                options=self._sync_weight_comm_options,
            )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()
        self._log_cuda_memory("sync/after_send")

    def recv_rollout_batch(self, input_channel: Channel) -> None:
        """
        Receive rollout batch from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for _ in range(split_num):
            recv_list.append(input_channel.get())

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        self.rollout_batch = cat_list_of_dict_tensor(recv_list, dim=1)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)
            primitive_loss_mask = loss_mask.clone()

            if self.cfg.algorithm.reward_type == "chunk_level":
                rollout_batch["primitive_chunk_mask"] = primitive_loss_mask
                rollout_batch["chunk_valid_fraction"] = primitive_loss_mask.float().mean(
                    dim=-1, keepdim=True
                )
                rollout_batch["chunk_full_mask"] = primitive_loss_mask.all(
                    dim=-1, keepdim=True
                )
                rollout_batch["chunk_partial_mask"] = (
                    primitive_loss_mask.any(dim=-1, keepdim=True)
                    & ~primitive_loss_mask.all(dim=-1, keepdim=True)
                )
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        use_time_decay = (
            self.cfg.algorithm.loss_type.startswith("nft")
            and self.cfg.algorithm.get("use_nft_time_decay", False)
        )
        if use_time_decay and rollout_batch.get("loss_mask", None) is not None:
            gamma = float(self.cfg.algorithm.get("nft_time_decay_gamma", 0.9))
            epsilon = float(self.cfg.algorithm.get("nft_time_decay_epsilon", 0.1))
            rollout_batch["time_decay_weights"] = compute_time_decay_weights(
                rollout_batch["loss_mask"], gamma=gamma, epsilon=epsilon
            )

        return rollout_batch

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """

        if self.cfg.algorithm.adv_type == "terminal-binary":
            _log_terminal_binary_adv_inputs(
                rank=self._rank,
                rewards=self.rollout_batch.get("rewards", None),
                dones=self.rollout_batch.get("dones", None),
                success_once=self.rollout_batch.get("success_once", None),
                loss_mask=self.rollout_batch.get("loss_mask", None),
            )

        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "success_once": self.rollout_batch.get("success_once", None),
            "task_ids": self.rollout_batch.get("task_ids", None),
            "reset_state_ids": self.rollout_batch.get("reset_state_ids", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "prm_pos_top_ratio": self.cfg.algorithm.get("prm_pos_top_ratio", 0.9),
            "prm_pos_min_count": self.cfg.algorithm.get("prm_pos_min_count", 1),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
            "rollout_epoch": self.cfg.algorithm.get("rollout_epoch", 1),
            "adv_clip_max": self.cfg.algorithm.get("clip_ratio_high", 1.0),
        }

        if self.cfg.algorithm.adv_type == "dual-credit":
            kwargs["_rollout_batch"] = self.rollout_batch

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch["loss_mask"] = kwargs["loss_mask"]
        else:
            self.rollout_batch.pop("loss_mask", None)
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch["loss_mask_sum"] = kwargs["loss_mask_sum"]
        else:
            self.rollout_batch.pop("loss_mask_sum", None)

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        if "avg_success_done_step" in self.rollout_batch:
            val = self.rollout_batch["avg_success_done_step"]
            if isinstance(val, torch.Tensor):
                val = val.item()
            rollout_metrics["avg_success_done_step"] = val

        return rollout_metrics

    # ------------------------------------------------------------------
    # Dual-Credit: PRM training (Phase C) and chunk gate (Phase D)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_nft_forward_context(data_dict: dict[str, Any]) -> dict[str, Any]:
        """Keep only observation/context tensors that OpenPI NFT forward consumes."""
        keep_keys = {"tokenized_prompt", "tokenized_prompt_mask"}
        return {
            k: v
            for k, v in data_dict.items()
            if isinstance(k, str) and ("/" in k or k in keep_keys)
        }

    @staticmethod
    def _repeat_nft_forward_context(
        data_dict: dict[str, Any],
        repeat_factor: int,
        batch_size: int,
    ) -> dict[str, Any]:
        """Repeat NFT forward context on the batch dimension."""
        if repeat_factor == 1:
            return data_dict

        repeated = {}
        for key, value in data_dict.items():
            if not isinstance(value, torch.Tensor):
                repeated[key] = value
                continue
            if value.ndim == 0:
                repeated[key] = value
                continue
            if value.shape[0] != batch_size:
                raise ValueError(
                    "dual-credit expected NFT context tensor batch dim "
                    f"{batch_size} for key `{key}`, got {tuple(value.shape)}"
                )
            repeated[key] = value.repeat_interleave(repeat_factor, dim=0).contiguous()
        return repeated

    @staticmethod
    def _repeat_nft_shared_cache(
        shared_cache: dict[str, Any] | None,
        repeat_factor: int,
    ) -> dict[str, Any] | None:
        """Repeat prefix KV cache on the batch dimension to match flattened solver steps."""
        if shared_cache is None or repeat_factor == 1:
            return shared_cache

        def _repeat_cache_value(value):
            if value is None:
                return None
            if torch.is_tensor(value):
                return value.repeat_interleave(repeat_factor, dim=0).contiguous()
            if isinstance(value, tuple):
                return tuple(_repeat_cache_value(item) for item in value)
            if isinstance(value, list):
                return [_repeat_cache_value(item) for item in value]
            if hasattr(value, "key_cache") and hasattr(value, "value_cache"):
                try:
                    repeated_cache = type(value)()
                except Exception:
                    from transformers.cache_utils import DynamicCache

                    repeated_cache = DynamicCache()
                repeated_cache.key_cache = [
                    _repeat_cache_value(item) for item in value.key_cache
                ]
                repeated_cache.value_cache = [
                    _repeat_cache_value(item) for item in value.value_cache
                ]
                if hasattr(value, "_seen_tokens"):
                    repeated_cache._seen_tokens = value._seen_tokens
                elif hasattr(value, "seen_tokens"):
                    repeated_cache.seen_tokens = value.seen_tokens
                return repeated_cache
            return value

        return {
            key: _repeat_cache_value(value) for key, value in shared_cache.items()
        }

    def _build_nft_shared_cache(
        self,
        model: torch.nn.Module,
        data_dict: dict[str, Any],
    ) -> dict[str, Any] | None:
        inner = model.module if hasattr(model, "module") else model
        if not hasattr(inner, "build_nft_shared_cache"):
            return None
        with torch.no_grad():
            with self.amp_context:
                return inner.build_nft_shared_cache(data_dict)

    def _forward_nft_all_steps_block(
        self,
        model: torch.nn.Module,
        data_context: dict[str, Any],
        shared_cache_base: dict[str, Any] | None,
        nft_xt_all: torch.Tensor,
        schedule: torch.Tensor,
        step_indices: list[int],
    ) -> torch.Tensor:
        """Run an NFT all-steps forward for a contiguous/specified solver-step block."""

        batch_size = nft_xt_all.shape[0]
        block_steps = len(step_indices)
        xt_block = nft_xt_all[:, step_indices].reshape(
            batch_size * block_steps, *nft_xt_all.shape[2:]
        )
        step_idx_tensor = torch.tensor(
            step_indices, device=self.device, dtype=torch.long
        )
        step_idx_flat = (
            step_idx_tensor.unsqueeze(0).expand(batch_size, block_steps).reshape(-1)
        )
        t_flat = schedule[step_idx_flat]
        data_expanded = self._repeat_nft_forward_context(
            data_context,
            block_steps,
            batch_size,
        )
        shared_cache_expanded = self._repeat_nft_shared_cache(
            shared_cache_base,
            block_steps,
        )
        with self.amp_context:
            out_block = model(
                data=data_expanded,
                use_nft_loss=True,
                compute_values=False,
                nft_explicit_inputs={
                    "x_t": xt_block,
                    "timesteps": t_flat,
                },
                use_cache=False,
                shared_cache=shared_cache_expanded,
            )
        chunk_size = out_block["v_theta"].shape[1]
        return out_block["v_theta"].reshape(batch_size, block_steps, chunk_size, -1)

    def _forward_nft_all_steps_safe(
        self,
        model: torch.nn.Module,
        data_dict: dict[str, Any],
        nft_xt_all: torch.Tensor,
        schedule: torch.Tensor,
        *,
        step_indices: list[int] | None = None,
    ) -> torch.Tensor:
        """Compute v_theta for all solver steps with OOM-safe solver-step splitting."""

        if step_indices is None:
            step_indices = list(range(int(nft_xt_all.shape[1])))
        data_context = self._extract_nft_forward_context(data_dict)
        shared_cache_base = self._build_nft_shared_cache(model, data_context)
        initial_step_block = min(len(step_indices), 2)
        self._actor_fused_step_max_block = max(
            int(getattr(self, "_actor_fused_step_max_block", 0)),
            initial_step_block,
        )

        def _run_block(indices: list[int]) -> torch.Tensor:
            try:
                return self._forward_nft_all_steps_block(
                    model,
                    data_context,
                    shared_cache_base,
                    nft_xt_all,
                    schedule,
                    indices,
                )
            except torch.OutOfMemoryError:
                if len(indices) <= 1:
                    raise
                self._actor_fused_step_split_fallback_count += 1
                torch.cuda.empty_cache()
                mid = len(indices) // 2
                left = _run_block(indices[:mid])
                right = _run_block(indices[mid:])
                return torch.cat([left, right], dim=1)

        if len(step_indices) <= initial_step_block:
            return _run_block(step_indices)

        outputs = []
        for block_start in range(0, len(step_indices), initial_step_block):
            block_indices = step_indices[block_start : block_start + initial_step_block]
            outputs.append(_run_block(block_indices))
        return torch.cat(outputs, dim=1)

    def _slice_train_batch(
        self,
        value: Any,
        start: int,
        end: int,
        batch_size: int,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            if value.ndim >= 1 and value.shape[0] == batch_size:
                return value[start:end]
            if value.ndim >= 2 and value.shape[1] == batch_size:
                return value[:, start:end]
            return value
        if isinstance(value, dict):
            return {
                key: self._slice_train_batch(item, start, end, batch_size)
                for key, item in value.items()
            }
        if isinstance(value, list):
            if len(value) == batch_size:
                return value[start:end]
            return value
        return value

    def _split_train_batch_in_half(
        self,
        data_dict: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        batch_size = int(data_dict["advantages"].shape[0])
        mid = batch_size // 2
        if mid <= 0 or mid >= batch_size:
            raise ValueError(f"Cannot split train batch of size {batch_size}")
        left = self._slice_train_batch(data_dict, 0, mid, batch_size)
        right = self._slice_train_batch(data_dict, mid, batch_size, batch_size)
        return left, right

    def _accumulate_weighted_metric_sums(
        self,
        metric_sums: dict[str, float],
        metrics_data: dict[str, Any],
        weight: float,
    ) -> None:
        for key, value in metrics_data.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    continue
                scalar = float(value.detach().item())
            elif isinstance(value, (int, float, bool, np.number)):
                scalar = float(value)
            else:
                continue
            metric_sums[key] = metric_sums.get(key, 0.0) + weight * scalar

    def _finalize_weighted_metric_sums(
        self,
        metric_sums: dict[str, float],
        total_weight: float,
    ) -> dict[str, float]:
        if total_weight <= 0:
            return {}
        return {key: value / total_weight for key, value in metric_sums.items()}

    @staticmethod
    def _linear_schedule_value(
        step: int,
        start_step: int,
        end_step: int,
        start_value: float,
        end_value: float,
    ) -> float:
        if end_step <= start_step:
            return float(end_value)
        if step < start_step:
            return float(start_value)
        if step >= end_step:
            return float(end_value)
        alpha = float(step - start_step) / float(end_step - start_step)
        return float((1.0 - alpha) * start_value + alpha * end_value)

    def _get_guidance_schedule_values(self) -> dict[str, float]:
        start_step = int(self.cfg.algorithm.get("guidance_lambda_start_step", 20))
        end_step = int(self.cfg.algorithm.get("guidance_lambda_end_step", 50))

        lambda_phi_final = float(self.cfg.algorithm.get("lambda_phi", 0.5))
        lambda_phi_init = float(
            self.cfg.algorithm.get("lambda_phi_init", lambda_phi_final)
        )
        lambda_pair_final = float(
            self.cfg.algorithm.get("pair_guidance_lambda", 0.2)
        )
        lambda_pair_init = float(
            self.cfg.algorithm.get("pair_guidance_lambda_init", lambda_pair_final)
        )

        return {
            "lambda_phi": self._linear_schedule_value(
                self.global_step,
                start_step,
                end_step,
                lambda_phi_init,
                lambda_phi_final,
            ),
            "lambda_pair": self._linear_schedule_value(
                self.global_step,
                start_step,
                end_step,
                lambda_pair_init,
                lambda_pair_final,
            ),
            "pair_delta_thresh": float(
                self.cfg.algorithm.get("pair_guidance_delta_thresh", 0.1)
            ),
            "schedule_progress": float(
                min(
                    1.0,
                    max(
                        0.0,
                        (
                            float(self.global_step - start_step)
                            / float(max(end_step - start_step, 1))
                        ),
                    ),
                )
            ),
        }

    def _compute_dual_credit_subbatch_loss(
        self,
        data: dict[str, Any],
        schedule: torch.Tensor,
        *,
        compute_values: bool,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        v_old = data.get("nft_v", None)
        x_t_input = data.get("nft_xt", None)
        x_next_input = data.get("nft_xnext", None)
        step_indices = data.get("nft_step_index", None)
        noise_level_for_loss = data.get("nft_noise_level", None)

        num_steps = self.cfg.actor.model.num_steps
        t = schedule[step_indices.long()]

        dual_credit_enabled = self.cfg.algorithm.loss_type == "nft-dual-credit"
        nft_xt_all = data.get("nft_xt_all", None)
        nft_xnext_all = data.get("nft_xnext_all", None)
        nft_v_all = data.get("nft_v_all", None)
        v_theta_all = None
        values = None

        fused_dual_credit_forward = (
            dual_credit_enabled
            and not compute_values
            and nft_xt_all is not None
            and nft_xnext_all is not None
        )

        if fused_dual_credit_forward:
            batch_size = nft_xt_all.shape[0]
            v_theta_all = self._forward_nft_all_steps_safe(
                self.model,
                data,
                nft_xt_all,
                schedule,
            )
            chunk_size = v_theta_all.shape[2]
            gather_step_idx = step_indices.reshape(batch_size, -1)[:, 0].long()
            batch_arange = torch.arange(batch_size, device=self.device)
            v_theta = v_theta_all[batch_arange, gather_step_idx]
        else:
            with self.amp_context:
                output_dict = self.model(
                    data=data,
                    use_nft_loss=True,
                    compute_values=compute_values,
                    compute_noise_stats=True,
                    nft_explicit_inputs={"x_t": x_t_input, "timesteps": t},
                    use_cache=False,
                    shared_cache=None,
                )

            v_theta = output_dict["v_theta"]
            values = output_dict.get("values", None)
            chunk_size = v_theta.shape[1]

        x_t_loss = x_t_input[:, :chunk_size, :]
        x_next_loss = x_next_input[:, :chunk_size, :]

        prev_values = data.get("prev_values", None)
        if prev_values is not None and prev_values.dim() > 1:
            prev_values = prev_values[:, :1]
        returns = data.get("returns", None)
        if returns is not None and returns.dim() > 1:
            returns = returns[:, :1]

        if self.cfg.algorithm.adv_type == "terminal-binary":
            self._maybe_log_terminal_binary_loss_inputs(
                advantages=data["advantages"],
                returns=returns,
                loss_mask=data.get("loss_mask", None),
                adv_clip_max=self.cfg.algorithm.get("clip_ratio_high", 5.0),
            )

        loss_type = "nft-actor-critic" if compute_values else "nft-actor"
        if self.cfg.algorithm.loss_type == "nft-dual-credit":
            loss_type = "nft-dual-credit"
        guidance_schedule = self._get_guidance_schedule_values()
        kwargs = {
            "loss_type": loss_type,
            "task_type": self.cfg.runner.task_type,
            "v_theta": v_theta,
            "v_old": v_old,
            "x_t": x_t_loss,
            "x_next": x_next_loss,
            "schedule": schedule,
            "step_indices": step_indices,
            "total_denoise_steps": num_steps,
            "noise_level": noise_level_for_loss,
            "advantages": data["advantages"],
            "loss_mask": data.get("loss_mask", None),
            "loss_mask_sum": data.get("loss_mask_sum", None),
            "time_decay_weights": data.get("time_decay_weights", None),
            "beta": self.cfg.algorithm.get("nft_beta", 1.0),
            "kl_beta": self.cfg.algorithm.get("kl_beta", 0.0),
            "adv_clip_max": self.cfg.algorithm.get("clip_ratio_high", 1.0),
            "task_ids": data.get("task_ids", None),
            "values": values,
            "returns": returns,
            "prev_values": prev_values,
            "value_clip": self.cfg.algorithm.get("value_clip", None),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "max_episode_steps": self.cfg.env.train.max_episode_steps,
            "critic_warmup": self._is_in_critic_warmup(),
        }

        if loss_type == "nft-dual-credit":
            if v_theta_all is None and nft_xt_all is not None:
                v_theta_all = self._forward_nft_all_steps_safe(
                    self.model,
                    data,
                    nft_xt_all,
                    schedule,
                )

            kwargs.update(
                {
                    "v_theta_all": v_theta_all,
                    "v_old_all": nft_v_all,
                    "nft_xt_all": nft_xt_all,
                    "nft_xnext_all": nft_xnext_all,
                    "chunk_gate": data.get("chunk_gate", None),
                    "y_phi": data.get("y_phi", None),
                    "enable_chunk_term": self.cfg.algorithm.get(
                        "enable_chunk_term", True
                    ),
                    "enable_gate": self.cfg.algorithm.get("enable_gate", True),
                    "enable_block_nft": self.cfg.algorithm.get(
                        "enable_block_nft", False
                    ),
                    "enable_pair_guidance": self.cfg.algorithm.get(
                        "enable_pair_guidance", False
                    ),
                    "lambda_phi": guidance_schedule["lambda_phi"],
                    "lambda_pair": guidance_schedule["lambda_pair"],
                    "lambda_tr": self.cfg.algorithm.get("lambda_tr", 0.001),
                    "block_nft_axis_mix": self.cfg.algorithm.get(
                        "block_nft_axis_mix", 0.5
                    ),
                    "block_nft_tau_block": self.cfg.algorithm.get(
                        "block_nft_tau_block", 1.0
                    ),
                    "block_nft_tau_axis": self.cfg.algorithm.get(
                        "block_nft_tau_axis", 1.0
                    ),
                    "block_nft_guidance_mode": self.cfg.algorithm.get(
                        "block_nft_guidance_mode", "score"
                    ),
                    "pair_guidance_kappa": self.cfg.algorithm.get(
                        "pair_guidance_kappa", 1.0
                    ),
                    "pair_guidance_good_xnext": data.get(
                        "pair_guidance_good_xnext", None
                    ),
                    "pair_guidance_bad_xnext": data.get(
                        "pair_guidance_bad_xnext", None
                    ),
                    "pair_guidance_weight": data.get(
                        "pair_guidance_weight", None
                    ),
                    "pair_guidance_mask": data.get("pair_guidance_mask", None),
                    "pair_guidance_score_gap": data.get(
                        "pair_guidance_score_gap", None
                    ),
                    "guidance_schedule_progress": guidance_schedule[
                        "schedule_progress"
                    ],
                }
            )
        return policy_loss(**kwargs)

    def _run_dual_credit_subbatch_safe(
        self,
        data: dict[str, Any],
        schedule: torch.Tensor,
        *,
        compute_values: bool,
        sync_on_last_leaf: bool,
    ) -> dict[str, float]:
        batch_cap = int(self.cfg.algorithm.get("dual_credit_inner_batch_cap", 8))

        def _run_recursive(
            sub_data: dict[str, Any],
            sync_last: bool,
        ) -> tuple[dict[str, float], float]:
            batch_size = int(sub_data["advantages"].shape[0])
            self._actor_dual_credit_inner_min_leaf = min(
                int(getattr(self, "_actor_dual_credit_inner_min_leaf", batch_size)),
                batch_size,
            )
            if batch_size > batch_cap:
                left_data, right_data = self._split_train_batch_in_half(sub_data)
                left_metrics, left_weight = _run_recursive(left_data, False)
                right_metrics, right_weight = _run_recursive(right_data, sync_last)
                merged = dict(left_metrics)
                for key, value in right_metrics.items():
                    merged[key] = merged.get(key, 0.0) + value
                return merged, left_weight + right_weight

            try:
                loss, metrics_data = self._compute_dual_credit_subbatch_loss(
                    sub_data,
                    schedule,
                    compute_values=compute_values,
                )
                raw_loss = loss.detach()
                scaled_loss = loss / self.gradient_accumulation
                if scaled_loss.requires_grad:
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=sync_last,
                    )
                    with backward_ctx:
                        self.grad_scaler.scale(scaled_loss).backward()
                else:
                    metrics_data["actor/no_grad_leaf"] = 1.0

                total_loss_for_log = metrics_data.get("actor/total_loss", raw_loss)
                if torch.is_tensor(total_loss_for_log):
                    total_loss_for_log = total_loss_for_log.detach().item()
                metrics_data["loss"] = float(total_loss_for_log)
                metrics_data["loss_scaled"] = scaled_loss.detach().item()
                metric_sums: dict[str, float] = {}
                self._accumulate_weighted_metric_sums(
                    metric_sums,
                    metrics_data,
                    float(batch_size),
                )
                return metric_sums, float(batch_size)
            except torch.OutOfMemoryError:
                if batch_size <= 1:
                    raise
                self._actor_dual_credit_inner_split_fallback_count += 1
                gc.collect()
                torch.cuda.empty_cache()
                left_data, right_data = self._split_train_batch_in_half(sub_data)
                left_metrics, left_weight = _run_recursive(left_data, False)
                right_metrics, right_weight = _run_recursive(right_data, sync_last)
                merged = dict(left_metrics)
                for key, value in right_metrics.items():
                    merged[key] = merged.get(key, 0.0) + value
                return merged, left_weight + right_weight

        metric_sums, total_weight = _run_recursive(data, sync_on_last_leaf)
        return self._finalize_weighted_metric_sums(metric_sums, total_weight)

    def _slice_chunk_forward_context_batch(
        self,
        data_dict: dict[str, Any],
        anchor_specs: list[dict[str, int]],
        *,
        n_chunk_steps: int,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, Any]:
        """Gather the local observation context for anchor chunks."""
        chunk_indices = [int(spec["chunk_idx"]) for spec in anchor_specs]
        batch_indices = [int(spec["batch_idx"]) for spec in anchor_specs]
        gathered: dict[str, Any] = {}
        for key, value in data_dict.items():
            if not isinstance(value, torch.Tensor):
                gathered[key] = value
                continue
            idx_chunk = torch.as_tensor(chunk_indices, device=value.device, dtype=torch.long)
            idx_batch = torch.as_tensor(batch_indices, device=value.device, dtype=torch.long)
            if value.ndim >= 2 and value.shape[0] == n_chunk_steps and value.shape[1] == batch_size:
                gathered[key] = put_tensor_device(value[idx_chunk, idx_batch], device)
            elif value.ndim >= 1 and value.shape[0] == batch_size:
                gathered[key] = put_tensor_device(value.index_select(0, idx_batch), device)
            else:
                gathered[key] = put_tensor_device(value, device)
        return gathered

    def _run_pair_guidance_precompute(self, schedule: torch.Tensor) -> dict[str, float]:
        if not bool(self.cfg.algorithm.get("enable_pair_guidance", False)):
            return {}
        if self.prm_model is None:
            return {}
        self._onload_aux_models_for_prm()

        rollout_model = self.ref_model if self.ref_model is not None else self.model
        rollout_inner = rollout_model.module if hasattr(rollout_model, "module") else rollout_model
        if not hasattr(rollout_inner, "sample_nft_suffix_candidates"):
            return {}

        ref_model_for_score = self.prev_rollout_model or self.ref_model
        if ref_model_for_score is None:
            return {}

        nft_xt_all = self.rollout_batch.get("nft_xt_all", None)
        nft_step_index = self.rollout_batch.get("nft_step_index", None)
        advantages = self.rollout_batch.get("advantages", None)
        loss_mask = self.rollout_batch.get("loss_mask", None)
        noise_level = self.rollout_batch.get("nft_noise_level", None)
        if nft_xt_all is None or nft_step_index is None or advantages is None:
            return {}

        device = self.device
        n_chunk_steps, batch_size = nft_xt_all.shape[:2]
        nft_xt_all_device = put_tensor_device(nft_xt_all, device)
        chunk_mask = self._coerce_chunk_mask(
            put_tensor_device(loss_mask, device) if loss_mask is not None else None,
            n_chunk_steps=n_chunk_steps,
            batch_size=batch_size,
        )
        if chunk_mask is None:
            chunk_mask = torch.ones(
                n_chunk_steps, batch_size, device=device, dtype=torch.bool
            )
        step_index_values = self._coerce_chunk_values(
            put_tensor_device(nft_step_index, device),
            n_chunk_steps=n_chunk_steps,
            batch_size=batch_size,
        )
        if step_index_values is None:
            return {}
        step_index_values = step_index_values.long()
        noise_level_values = None
        if noise_level is not None:
            noise_level_values = self._coerce_chunk_values(
                put_tensor_device(noise_level, device),
                n_chunk_steps=n_chunk_steps,
                batch_size=batch_size,
            )

        anchor_specs: list[dict[str, int]] = []
        adv_device = put_tensor_device(advantages, device)
        for batch_idx in range(batch_size):
            valid_idx = torch.nonzero(chunk_mask[:, batch_idx], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            traj_label = adv_device[valid_idx[0], batch_idx].reshape(-1)[0].sign().item()
            if traj_label >= 0:
                chunk_idx = int(valid_idx[-1].item())
            else:
                tail_start = max(int(math.floor(0.75 * max(valid_idx.numel() - 1, 0))), 0)
                chunk_idx = int(valid_idx[tail_start:][-1].item())
            step_idx = int(step_index_values[chunk_idx, batch_idx].item())
            if step_idx < 0 or step_idx >= int(self.cfg.actor.model.num_steps):
                continue
            anchor_specs.append(
                {
                    "batch_idx": batch_idx,
                    "chunk_idx": chunk_idx,
                    "step_idx": step_idx,
                }
            )

        output_good = torch.zeros(
            n_chunk_steps,
            batch_size,
            *nft_xt_all.shape[3:],
            dtype=nft_xt_all.dtype,
        )
        output_bad = torch.zeros_like(output_good)
        output_weight = torch.zeros(n_chunk_steps, batch_size, dtype=torch.float32)
        output_gap = torch.zeros_like(output_weight)
        output_mask = torch.zeros(n_chunk_steps, batch_size, dtype=torch.bool)

        if len(anchor_specs) == 0:
            self.rollout_batch["pair_guidance_good_xnext"] = output_good
            self.rollout_batch["pair_guidance_bad_xnext"] = output_bad
            self.rollout_batch["pair_guidance_weight"] = output_weight
            self.rollout_batch["pair_guidance_score_gap"] = output_gap
            self.rollout_batch["pair_guidance_mask"] = output_mask
            guidance_schedule = self._get_guidance_schedule_values()
            return {
                "actor/pair_precompute_s": 0.0,
                "actor/pair_selected_traj_frac": 0.0,
                "actor/pair_candidate_valid_frac": 0.0,
                "actor/pair_precompute_gap_mean": 0.0,
                "actor/pair_precompute_weight_mean": 0.0,
                "actor/pair_delta_thresh_eff": guidance_schedule[
                    "pair_delta_thresh"
                ],
            }

        pair_num_samples = max(
            2, int(self.cfg.algorithm.get("pair_guidance_num_samples", 4))
        )
        guidance_schedule = self._get_guidance_schedule_values()
        delta_thresh = float(guidance_schedule["pair_delta_thresh"])
        prm_beta = float(self.cfg.algorithm.get("prm_beta", 0.5))
        precompute_start = time.perf_counter()
        full_context = self._extract_nft_forward_context(self.rollout_batch)
        grouped_specs: dict[int, list[dict[str, int]]] = {}
        for spec in anchor_specs:
            grouped_specs.setdefault(spec["step_idx"], []).append(spec)

        pair_gap_values: list[float] = []
        pair_weight_values: list[float] = []
        valid_pairs = 0
        pair_group_batch_size = 2

        with torch.no_grad():
            self.prm_model.eval()
            ref_model_for_score.eval()
            rollout_model.eval()
            for step_idx, group_specs in grouped_specs.items():
                for group_start in range(0, len(group_specs), pair_group_batch_size):
                    subgroup_specs = group_specs[
                        group_start : group_start + pair_group_batch_size
                    ]
                    context_batch = self._slice_chunk_forward_context_batch(
                        full_context,
                        subgroup_specs,
                        n_chunk_steps=n_chunk_steps,
                        batch_size=batch_size,
                        device=device,
                    )
                    shared_cache = self._build_nft_shared_cache(
                        rollout_model, context_batch
                    )
                    idx_chunk = torch.as_tensor(
                        [spec["chunk_idx"] for spec in subgroup_specs],
                        device=device,
                        dtype=torch.long,
                    )
                    idx_batch = torch.as_tensor(
                        [spec["batch_idx"] for spec in subgroup_specs],
                        device=device,
                        dtype=torch.long,
                    )
                    start_x_t = nft_xt_all_device[
                        idx_chunk,
                        idx_batch,
                        step_idx,
                    ]
                    group_noise = None
                    if noise_level_values is not None:
                        group_noise = noise_level_values[idx_chunk, idx_batch]

                    with self.amp_context:
                        sampled = rollout_inner.sample_nft_suffix_candidates(
                            context_batch,
                            start_x_t,
                            step_idx,
                            num_candidates=pair_num_samples,
                            shared_cache=shared_cache,
                            noise_level=group_noise,
                        )
                    group_size = len(subgroup_specs)
                    suffix_xt = sampled["nft_xt_all"].reshape(
                        group_size * pair_num_samples,
                        sampled["nft_xt_all"].shape[2],
                        *sampled["nft_xt_all"].shape[3:],
                    )
                    suffix_xnext = sampled["nft_xnext_all"].reshape(
                        group_size * pair_num_samples,
                        sampled["nft_xnext_all"].shape[2],
                        *sampled["nft_xnext_all"].shape[3:],
                    )
                    suffix_step_idx = sampled["nft_step_index_all"].reshape(
                        group_size * pair_num_samples,
                        sampled["nft_step_index_all"].shape[2],
                    )
                    suffix_noise = sampled["nft_noise_level"].reshape(
                        group_size * pair_num_samples
                    )
                    context_rep = self._repeat_nft_forward_context(
                        context_batch,
                        pair_num_samples,
                        group_size,
                    )
                    shared_cache_rep = self._repeat_nft_shared_cache(
                        shared_cache,
                        pair_num_samples,
                    )
                    E_prm = self._compute_energy_for_model(
                        self.prm_model,
                        suffix_xt,
                        suffix_xnext,
                        schedule,
                        suffix_noise,
                        data_dict=context_rep,
                        shared_cache=shared_cache_rep,
                        step_indices_all=suffix_step_idx,
                    )
                    E_ref = self._compute_energy_for_model(
                        ref_model_for_score,
                        suffix_xt,
                        suffix_xnext,
                        schedule,
                        suffix_noise,
                        data_dict=context_rep,
                        shared_cache=shared_cache_rep,
                        step_indices_all=suffix_step_idx,
                    )
                    raw_scores = (
                        -0.5 * prm_beta * (E_prm - E_ref).mean(dim=1)
                    ).reshape(group_size, pair_num_samples)
                    score_norm = torch.tanh(raw_scores)
                    best_idx = raw_scores.argmax(dim=1)
                    worst_idx = raw_scores.argmin(dim=1)
                    best_scores = score_norm.gather(1, best_idx.unsqueeze(1)).squeeze(1)
                    worst_scores = score_norm.gather(1, worst_idx.unsqueeze(1)).squeeze(1)
                    gap = best_scores - worst_scores
                    weight = (
                        (gap - delta_thresh) / max(1.0 - delta_thresh, 1.0e-6)
                    ).clamp(min=0.0, max=1.0)

                    first_xnext = sampled["first_xnext"]
                    for local_idx, spec in enumerate(subgroup_specs):
                        gap_val = float(gap[local_idx].item())
                        weight_val = float(weight[local_idx].item())
                        pair_gap_values.append(gap_val)
                        pair_weight_values.append(weight_val)
                        if weight_val <= 0.0:
                            continue
                        valid_pairs += 1
                        best_local = int(best_idx[local_idx].item())
                        worst_local = int(worst_idx[local_idx].item())
                        chunk_idx = spec["chunk_idx"]
                        batch_idx = spec["batch_idx"]
                        output_good[chunk_idx, batch_idx] = (
                            first_xnext[local_idx, best_local].detach().cpu()
                        )
                        output_bad[chunk_idx, batch_idx] = (
                            first_xnext[local_idx, worst_local].detach().cpu()
                        )
                        output_weight[chunk_idx, batch_idx] = weight_val
                        output_gap[chunk_idx, batch_idx] = gap_val
                        output_mask[chunk_idx, batch_idx] = True

        elapsed = time.perf_counter() - precompute_start
        self.rollout_batch["pair_guidance_good_xnext"] = output_good
        self.rollout_batch["pair_guidance_bad_xnext"] = output_bad
        self.rollout_batch["pair_guidance_weight"] = output_weight
        self.rollout_batch["pair_guidance_score_gap"] = output_gap
        self.rollout_batch["pair_guidance_mask"] = output_mask
        self._log_progress(
            "[train][pair] precompute "
            f"step={self.global_step} anchors={len(anchor_specs)} valid={valid_pairs} "
            f"elapsed={elapsed:.1f}s gap_mean={float(np.mean(pair_gap_values)) if pair_gap_values else 0.0:.4f}"
        )
        return {
            "actor/pair_precompute_s": elapsed,
            "actor/pair_selected_traj_frac": float(len(anchor_specs)) / max(batch_size, 1),
            "actor/pair_candidate_valid_frac": float(valid_pairs) / max(len(anchor_specs), 1),
            "actor/pair_precompute_gap_mean": float(np.mean(pair_gap_values)) if pair_gap_values else 0.0,
            "actor/pair_precompute_weight_mean": float(np.mean(pair_weight_values)) if pair_weight_values else 0.0,
            "actor/pair_delta_thresh_eff": delta_thresh,
        }

    @staticmethod
    def _coerce_chunk_mask(
        mask: torch.Tensor | None,
        n_chunk_steps: int,
        batch_size: int,
    ) -> torch.Tensor | None:
        """Convert rollout/train masks to chunk-level shape [n_chunk_steps, batch_size]."""
        if mask is None:
            return None

        mask = mask.bool()

        if mask.ndim == 3:
            if mask.shape[:2] == (n_chunk_steps, batch_size):
                return mask.any(dim=-1)
            if mask.shape[0] == batch_size and mask.shape[1] == n_chunk_steps:
                return mask.any(dim=-1).transpose(0, 1)

        if mask.ndim == 2:
            if mask.shape == (n_chunk_steps, batch_size):
                return mask
            if mask.shape == (batch_size, n_chunk_steps):
                return mask.transpose(0, 1)
            if mask.shape[1] == batch_size and mask.shape[0] % n_chunk_steps == 0:
                chunk_span = mask.shape[0] // n_chunk_steps
                return mask.reshape(n_chunk_steps, chunk_span, batch_size).any(dim=1)
            if mask.shape[0] == batch_size and mask.shape[1] % n_chunk_steps == 0:
                chunk_span = mask.shape[1] // n_chunk_steps
                return (
                    mask.reshape(batch_size, n_chunk_steps, chunk_span)
                    .permute(1, 0, 2)
                    .any(dim=-1)
                )

        if mask.ndim == 1 and mask.shape[0] == batch_size:
            return mask.view(1, batch_size).expand(n_chunk_steps, batch_size)

        raise ValueError(
            "dual-credit PRM expected loss_mask compatible with "
            f"[n_chunk_steps={n_chunk_steps}, batch_size={batch_size}], got {tuple(mask.shape)}"
        )

    @staticmethod
    def _coerce_chunk_values(
        values: torch.Tensor | None,
        n_chunk_steps: int,
        batch_size: int,
    ) -> torch.Tensor | None:
        """Convert chunk-aligned tensors to shape [n_chunk_steps, batch_size]."""
        if values is None:
            return None

        if values.ndim == 3 and values.shape[:2] == (n_chunk_steps, batch_size):
            if values.shape[2] == 1:
                return values.squeeze(-1)
            return values.float().mean(dim=-1)

        if values.ndim == 2:
            if values.shape == (n_chunk_steps, batch_size):
                return values
            if values.shape == (batch_size, n_chunk_steps):
                return values.transpose(0, 1)

        if values.ndim == 1 and values.shape[0] == batch_size:
            return values.view(1, batch_size).expand(n_chunk_steps, batch_size)

        raise ValueError(
            "dual-credit PRM expected chunk-aligned values compatible with "
            f"[n_chunk_steps={n_chunk_steps}, batch_size={batch_size}], got {tuple(values.shape)}"
        )

    @staticmethod
    def _coerce_chunk_sequence(
        values: torch.Tensor | None,
        n_chunk_steps: int,
        batch_size: int,
        chunk_size_hint: int,
    ) -> torch.Tensor | None:
        """Convert chunk-aligned trajectory tensors to [n_chunk_steps, batch_size, chunk, dim]."""
        if values is None:
            return None

        if values.ndim == 4:
            if values.shape[:2] == (n_chunk_steps, batch_size):
                return values.contiguous()
            if values.shape[:2] == (batch_size, n_chunk_steps):
                return values.permute(1, 0, 2, 3).contiguous()

        if values.ndim == 3 and chunk_size_hint > 0:
            if (
                values.shape[:2] == (n_chunk_steps, batch_size)
                and values.shape[-1] % chunk_size_hint == 0
            ):
                return values.reshape(
                    n_chunk_steps, batch_size, chunk_size_hint, -1
                ).contiguous()
            if (
                values.shape[:2] == (batch_size, n_chunk_steps)
                and values.shape[-1] % chunk_size_hint == 0
            ):
                return (
                    values.reshape(batch_size, n_chunk_steps, chunk_size_hint, -1)
                    .permute(1, 0, 2, 3)
                    .contiguous()
                )

        return None

    @staticmethod
    def _squeeze_step_batch_dim(data_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert stacked step dict tensors from [T, 1, ...] to [T, ...]."""
        squeezed = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                squeezed[key] = EmbodiedFSDPActor._squeeze_step_batch_dim(value)
            elif isinstance(value, torch.Tensor):
                if value.ndim >= 2 and value.shape[1] == 1:
                    squeezed[key] = value.squeeze(1).contiguous()
                else:
                    squeezed[key] = value.contiguous()
            else:
                squeezed[key] = value
        return squeezed

    @staticmethod
    def _branch_valid_fraction_from_dones(dones: torch.Tensor) -> torch.Tensor:
        dones = torch.as_tensor(dones, dtype=torch.bool)
        if dones.ndim != 2:
            raise ValueError(
                "dual-credit branch suffix expected dones shape [T, num_action_chunks], "
                f"got {tuple(dones.shape)}"
            )
        valid = (dones.to(torch.int64).cumsum(dim=1) == 0).float()
        return valid.mean(dim=1)

    def _append_branch_recipe(self, sample: dict[str, Any]) -> None:
        task_id = int(sample.get("task_id", -1))
        if task_id < 0:
            return
        bank_limit = int(self.cfg.algorithm.get("branch_recipe_bank_size", 64))
        if bank_limit <= 0:
            return
        bucket = self._branch_recipe_bank.setdefault(task_id, [])
        bucket.append(
            {
                "rel_branch_pos": float(sample["branch_chunk_idx"])
                / max(float(sample["source_valid_len"] - 1), 1.0),
                "delta_action": sample["delta_action"].detach().cpu().contiguous(),
                "delta_norm": float(sample.get("delta_norm", 0.0)),
                "scale_used": float(
                    sample.get(
                        "scale_used",
                        self.cfg.algorithm.get("branch_delta_scale", 0.5),
                    )
                ),
                "last_used_step": int(self.global_step),
                "use_count": 0,
                "source_kind": str(sample.get("source_kind", "natural")),
            }
        )
        if len(bucket) > bank_limit:
            del bucket[: len(bucket) - bank_limit]

    def _prepare_branch_sample(
        self, branch_result: dict[str, Any]
    ) -> dict[str, Any] | None:
        suffix_forward_inputs = branch_result.get("suffix_forward_inputs", [])
        suffix_dones = branch_result.get("suffix_dones", [])
        if len(suffix_forward_inputs) == 0 or len(suffix_forward_inputs) != len(suffix_dones):
            return None

        merged = stack_list_of_dict_tensor(suffix_forward_inputs, dim=0)
        merged = self._squeeze_step_batch_dim(merged)
        if "nft_xt_all" not in merged or "nft_xnext_all" not in merged:
            return None

        suffix_dones_tensor = torch.stack(
            [torch.as_tensor(x, dtype=torch.bool).view(-1) for x in suffix_dones], dim=0
        )
        suffix_valid_fraction = self._branch_valid_fraction_from_dones(suffix_dones_tensor)
        if float(suffix_valid_fraction.sum().item()) <= 0.0:
            return None

        delta_action = torch.as_tensor(
            branch_result.get("delta_action"), dtype=torch.float32
        ).cpu()
        return {
            "tau_plus": int(branch_result["tau_plus"]),
            "task_id": int(branch_result.get("task_id", -1)),
            "reset_state_id": int(branch_result.get("reset_state_id", -1)),
            "branch_chunk_idx": int(branch_result["branch_chunk_idx"]),
            "source_kind": str(branch_result.get("source_kind", "natural")),
            "source_valid_len": int(branch_result.get("source_valid_len", 0)),
            "scale_used": float(branch_result.get("scale_used", 0.0)),
            "suffix_len": int(suffix_valid_fraction.shape[0]),
            "suffix_data": merged,
            "suffix_valid_fraction": suffix_valid_fraction.contiguous(),
            "delta_action": delta_action.contiguous(),
            "delta_norm": float(delta_action.norm().item()),
        }

    def plan_branch_rollouts(self) -> list[dict[str, Any]]:
        """Plan lightweight near-miss branch rollouts for PRM-only augmentation."""
        self._synthetic_branch_entries = []
        self._last_branch_metrics = {}
        self._last_branch_plan_metrics = {}

        if not self.cfg.algorithm.get("enable_branch_rollout", False):
            return []

        preference_group_specs = self.rollout_batch.get("preference_group_specs", [])
        if not preference_group_specs:
            return []

        executed_action = self.rollout_batch.get("executed_action", None)
        nft_xt_all = self.rollout_batch.get("nft_xt_all", None)
        if executed_action is None or nft_xt_all is None:
            return []

        n_chunk_steps, batch_size = nft_xt_all.shape[:2]
        chunk_size_hint = int(getattr(self.cfg.actor.model, "num_action_chunks", 1))
        action_seq = self._coerce_chunk_sequence(
            executed_action,
            n_chunk_steps=n_chunk_steps,
            batch_size=batch_size,
            chunk_size_hint=chunk_size_hint,
        )
        if action_seq is None:
            action_seq = self._coerce_chunk_sequence(
                self.rollout_batch.get("action", None),
                n_chunk_steps=n_chunk_steps,
                batch_size=batch_size,
                chunk_size_hint=chunk_size_hint,
            )
        if action_seq is None:
            return []

        chunk_mask = self._coerce_chunk_mask(
            self.rollout_batch.get("loss_mask", None),
            n_chunk_steps=n_chunk_steps,
            batch_size=batch_size,
        )
        if chunk_mask is None:
            valid_fraction = self._coerce_chunk_values(
                self.rollout_batch.get("chunk_valid_fraction", None),
                n_chunk_steps=n_chunk_steps,
                batch_size=batch_size,
            )
            if valid_fraction is not None:
                chunk_mask = valid_fraction > 0
        if chunk_mask is None:
            chunk_mask = torch.ones(
                n_chunk_steps, batch_size, dtype=torch.bool, device=action_seq.device
            )
        chunk_mask = chunk_mask.detach().cpu()
        action_seq = action_seq.detach().cpu().float()

        branch_plans_per_group = int(self.cfg.algorithm.get("branch_plans_per_group", 1))
        branch_delta_scale = float(self.cfg.algorithm.get("branch_delta_scale", 0.5))
        branch_jitter_scale = float(self.cfg.algorithm.get("branch_jitter_scale", 0.05))
        rng = np.random.default_rng(
            int(self.cfg.actor.seed)
            + 1009 * int(self._rank)
            + 1000003 * int(self.global_step)
        )

        candidate_plans: list[dict[str, Any]] = []
        planned_from_bank = 0
        planned_from_natural = 0
        similarity_sum = 0.0
        branch_idx_sum = 0.0
        suffix_horizon_sum = 0.0
        delta_norm_sum = 0.0
        action_shift_norm_sum = 0.0
        action_shift_rel_sum = 0.0
        eligible_groups = 0

        def _valid_len(traj_idx: int) -> int:
            return int(chunk_mask[:, traj_idx].sum().item())

        @staticmethod
        def _branch_window(valid_len: int) -> tuple[int, int] | None:
            if valid_len < 6:
                return None
            min_prefix = max(1, int(math.floor(0.35 * valid_len)))
            max_by_ratio = int(math.floor(0.85 * valid_len))
            max_by_suffix = valid_len - 5
            max_branch = min(max_by_ratio, max_by_suffix)
            max_branch = min(max_branch, valid_len - 2)
            if max_branch < min_prefix:
                min_prefix = 1
                max_branch = valid_len - 2
            if max_branch < min_prefix:
                return None
            return min_prefix, max_branch

        def _score_failure_pair(tau_plus: int, tau_minus: int) -> dict[str, Any] | None:
            valid_len = min(_valid_len(tau_plus), _valid_len(tau_minus))
            if valid_len < 3:
                return None
            plus = action_seq[:valid_len, tau_plus].reshape(valid_len, -1)
            minus = action_seq[:valid_len, tau_minus].reshape(valid_len, -1)
            plus_norm = plus.norm(dim=-1).clamp_min(1.0e-6)
            minus_norm = minus.norm(dim=-1).clamp_min(1.0e-6)
            cosine = (plus * minus).sum(dim=-1) / (plus_norm * minus_norm)
            diff = (minus - plus).norm(dim=-1) / plus_norm
            window = _branch_window(valid_len)
            if window is None:
                return None
            min_branch, max_branch = window
            late_weight = torch.linspace(0.75, 1.0, valid_len)
            weighted_diff = diff * late_weight
            window_scores = weighted_diff[min_branch : max_branch + 1]
            branch_idx = min_branch + int(torch.argmax(window_scores).item())
            similarity = float(cosine.mean().item())
            early_penalty = float(diff[: max(valid_len // 3, 1)].mean().item())
            branch_score = similarity - 0.35 * early_penalty + 0.15 * (
                branch_idx / max(valid_len - 1, 1)
            )
            return {
                "tau_minus": int(tau_minus),
                "branch_idx": int(branch_idx),
                "branch_score": float(branch_score),
                "similarity": float(similarity),
                "delta_action": (
                    action_seq[branch_idx, tau_minus] - action_seq[branch_idx, tau_plus]
                ).contiguous(),
            }

        def _sample_recipe(task_id: int, valid_len: int) -> dict[str, Any] | None:
            recipes = self._branch_recipe_bank.get(int(task_id), [])
            if len(recipes) == 0 or valid_len < 3:
                return None
            ordered = sorted(
                recipes,
                key=lambda item: (
                    int(item.get("use_count", 0)),
                    -int(item.get("last_used_step", 0)),
                ),
            )
            recipe = ordered[0]
            branch_idx = int(
                round(float(recipe.get("rel_branch_pos", 0.5)) * max(valid_len - 1, 1))
            )
            window = _branch_window(valid_len)
            if window is None:
                return None
            min_branch, max_branch = window
            branch_idx = max(min_branch, min(branch_idx, max_branch))
            return {
                "tau_minus": -1,
                "branch_idx": int(branch_idx),
                "branch_score": 0.0,
                "similarity": 0.0,
                "delta_action": recipe["delta_action"].detach().cpu().float().contiguous(),
                "preferred_scale": float(
                    recipe.get(
                        "scale_used",
                        self.cfg.algorithm.get("branch_delta_scale", 0.5),
                    )
                ),
                "_recipe_ref": recipe,
                "source_kind": "bank",
            }

        for spec in preference_group_specs:
            pos_candidates = [int(x) for x in spec.get("pos_candidates", [])]
            if len(pos_candidates) == 0:
                continue
            eligible_groups += 1
            tau_plus = int(pos_candidates[min(len(pos_candidates) // 2, len(pos_candidates) - 1)])
            valid_len_plus = _valid_len(tau_plus)
            if valid_len_plus < 3:
                continue

            fail_candidates = [int(x) for x in spec.get("fail_candidates", [])]
            natural_choice = None
            if fail_candidates:
                scored = []
                for tau_minus in fail_candidates:
                    item = _score_failure_pair(tau_plus, tau_minus)
                    if item is not None:
                        scored.append(item)
                if scored:
                    natural_choice = max(
                        scored,
                        key=lambda item: (
                            float(item["branch_score"]),
                            float(item["similarity"]),
                            int(item["branch_idx"]),
                        ),
                    )

            task_id = (
                int(spec["group_key"][0])
                if len(spec.get("group_key", ())) > 0
                else -1
            )
            group_choices = []
            if natural_choice is not None:
                group_choices.append(natural_choice)
            if natural_choice is None or branch_plans_per_group > 1:
                recipe_choice = _sample_recipe(task_id, valid_len_plus)
            else:
                recipe_choice = None
            if recipe_choice is not None:
                group_choices.append(recipe_choice)
            if not group_choices:
                continue
            group_choices = sorted(
                group_choices,
                key=lambda item: (
                    -float(item.get("branch_score", 0.0)),
                    -float(item.get("similarity", 0.0)),
                    str(item.get("source_kind", "natural")) == "bank",
                ),
            )[: max(branch_plans_per_group, 0)]

            for choice in group_choices:
                branch_idx = int(choice["branch_idx"])
                suffix_horizon = max(valid_len_plus - branch_idx - 1, 0)
                if suffix_horizon < 1:
                    continue

                base_action = action_seq[branch_idx, tau_plus]
                delta_action = choice["delta_action"].float()
                plan_delta_scale = float(choice.get("preferred_scale", branch_delta_scale))
                plan_delta_scale = max(0.05, min(plan_delta_scale, 1.0))
                delta_scale = max(
                    float(delta_action.norm().item())
                    / max(delta_action.numel() ** 0.5, 1.0),
                    1.0e-3,
                )
                jitter = torch.as_tensor(
                    rng.normal(size=tuple(delta_action.shape)),
                    dtype=delta_action.dtype,
                ) * (branch_jitter_scale * delta_scale)
                branch_action = (
                    base_action + plan_delta_scale * delta_action + jitter
                ).clamp_(-1.0, 1.0)
                action_shift_norm = float((branch_action - base_action).norm().item())
                action_shift_rel = action_shift_norm / max(
                    float(base_action.norm().item()), 1.0e-6
                )

                reset_state_id = (
                    int(spec["group_key"][1])
                    if len(spec.get("group_key", ())) > 1
                    else -1
                )
                source_kind = str(choice.get("source_kind", "natural"))
                if source_kind == "bank" and isinstance(choice.get("_recipe_ref"), dict):
                    recipe = choice["_recipe_ref"]
                    recipe["use_count"] = int(recipe.get("use_count", 0)) + 1
                    recipe["last_used_step"] = int(self.global_step)
                candidate_plans.append(
                    {
                        "plan_id": int(
                            self.global_step * 100000
                            + self._rank * 1000
                            + len(candidate_plans)
                        ),
                        "actor_rank": int(self._rank),
                        "tau_plus": int(tau_plus),
                        "tau_minus": int(choice.get("tau_minus", -1)),
                        "task_id": int(task_id),
                        "reset_state_id": int(reset_state_id),
                        "branch_chunk_idx": int(branch_idx),
                        "source_valid_len": int(valid_len_plus),
                        "suffix_horizon": int(suffix_horizon),
                        "prefix_actions": action_seq[:branch_idx, tau_plus].contiguous(),
                        "base_action": base_action.contiguous(),
                        "branch_action": branch_action.contiguous(),
                        "branch_jitter": jitter.contiguous(),
                        "branch_delta_scale": float(plan_delta_scale),
                        "delta_action": delta_action.contiguous(),
                        "source_kind": source_kind,
                        "plan_score": float(choice.get("branch_score", 0.0)),
                        "similarity": float(choice.get("similarity", 0.0)),
                        "action_shift_norm": float(action_shift_norm),
                        "action_shift_rel": float(action_shift_rel),
                    }
                )
                planned_from_bank += int(source_kind == "bank")
                planned_from_natural += int(source_kind != "bank")
                similarity_sum += float(choice.get("similarity", 0.0))
                branch_idx_sum += float(branch_idx)
                suffix_horizon_sum += float(suffix_horizon)
                delta_norm_sum += float(delta_action.norm().item())
                action_shift_norm_sum += float(action_shift_norm)
                action_shift_rel_sum += float(action_shift_rel)

        selected_plans = candidate_plans
        selected_bank = sum(int(item.get("source_kind") == "bank") for item in selected_plans)
        selected_natural = sum(
            int(item.get("source_kind") != "bank") for item in selected_plans
        )
        selected_similarity_sum = sum(
            float(item.get("similarity", 0.0)) for item in selected_plans
        )
        selected_branch_idx_sum = sum(
            float(item.get("branch_chunk_idx", 0)) for item in selected_plans
        )
        selected_rel_branch_pos_sum = sum(
            float(item.get("branch_chunk_idx", 0))
            / max(float(item.get("source_valid_len", 1) - 1), 1.0)
            for item in selected_plans
        )
        selected_suffix_horizon_sum = sum(
            float(item.get("suffix_horizon", 0)) for item in selected_plans
        )
        selected_delta_norm_sum = sum(
            float(torch.as_tensor(item.get("delta_action")).norm().item())
            for item in selected_plans
        )
        selected_action_shift_norm_sum = sum(
            float(item.get("action_shift_norm", 0.0)) for item in selected_plans
        )
        selected_action_shift_rel_sum = sum(
            float(item.get("action_shift_rel", 0.0)) for item in selected_plans
        )
        self._last_branch_plan_metrics = {
            "branch/planned": float(len(selected_plans)),
            "branch/eligible_groups": float(eligible_groups),
            "branch/planned_from_natural": float(selected_natural),
            "branch/planned_from_bank": float(selected_bank),
            "branch/plan_similarity_mean": similarity_sum / max(len(candidate_plans), 1),
            "branch/plan_branch_idx_mean": branch_idx_sum / max(len(candidate_plans), 1),
            "branch/plan_suffix_horizon_mean": suffix_horizon_sum
            / max(len(candidate_plans), 1),
            "branch/plan_delta_norm_mean": delta_norm_sum / max(len(candidate_plans), 1),
            "branch/plan_action_shift_norm_mean": action_shift_norm_sum
            / max(len(candidate_plans), 1),
            "branch/plan_action_shift_rel_mean": action_shift_rel_sum
            / max(len(candidate_plans), 1),
            "branch/selected_similarity_mean": selected_similarity_sum
            / max(len(selected_plans), 1),
            "branch/selected_branch_idx_mean": selected_branch_idx_sum
            / max(len(selected_plans), 1),
            "branch/selected_rel_branch_pos_mean": selected_rel_branch_pos_sum
            / max(len(selected_plans), 1),
            "branch/selected_suffix_horizon_mean": selected_suffix_horizon_sum
            / max(len(selected_plans), 1),
            "branch/selected_delta_norm_mean": selected_delta_norm_sum
            / max(len(selected_plans), 1),
            "branch/selected_action_shift_norm_mean": selected_action_shift_norm_sum
            / max(len(selected_plans), 1),
            "branch/selected_action_shift_rel_mean": selected_action_shift_rel_sum
            / max(len(selected_plans), 1),
            "branch/plan_selection_frac": float(len(selected_plans)) / max(
                len(candidate_plans), 1
            ),
            "branch/plans_per_group": float(branch_plans_per_group),
        }
        self._log_progress(
            "[train][branch][plan] "
            f"step={self.global_step} selected={len(selected_plans)}/{len(candidate_plans)} "
            f"eligible_groups={eligible_groups} bank={selected_bank} natural={selected_natural}"
        )
        return selected_plans

    def load_branch_results(
        self, branch_results_by_rank: dict[int, list[dict[str, Any]]]
    ) -> dict[str, float]:
        local_results = list(branch_results_by_rank.get(int(self._rank), []))
        self._synthetic_branch_entries = []

        accepted = 0
        source_bank = 0
        suffix_len_sum = 0.0
        delta_norm_sum = 0.0
        generated_success = 0
        generated_fail = 0
        immediate_success = 0
        terminated = 0
        generated_suffix_len_sum = 0.0
        success_suffix_len_sum = 0.0
        fail_suffix_len_sum = 0.0
        search_attempts_sum = 0.0
        scale_used_sum = 0.0
        result_action_shift_norm_sum = 0.0
        result_action_shift_rel_sum = 0.0
        found_failure = 0
        for result in local_results:
            is_success = bool(result.get("success", not bool(result.get("failed", False))))
            is_failed = bool(result.get("failed", not is_success))
            generated_success += int(is_success)
            generated_fail += int(is_failed)
            immediate_success += int(bool(result.get("immediate_success", False)))
            terminated += int(bool(result.get("terminated", False)))
            result_suffix_len = float(result.get("suffix_len", 0))
            generated_suffix_len_sum += result_suffix_len
            if is_success:
                success_suffix_len_sum += result_suffix_len
            else:
                fail_suffix_len_sum += result_suffix_len
            search_attempts_sum += float(result.get("scale_attempts", 1))
            scale_used_sum += float(result.get("scale_used", 0.0))
            result_action_shift_norm_sum += float(result.get("action_shift_norm", 0.0))
            result_action_shift_rel_sum += float(result.get("action_shift_rel", 0.0))
            found_failure += int(bool(result.get("scale_found_failure", is_failed)))
            if not bool(result.get("failed", False)):
                continue
            sample = self._prepare_branch_sample(result)
            if sample is None:
                continue
            self._synthetic_branch_entries.append(sample)
            self._append_branch_recipe(sample)
            accepted += 1
            source_bank += int(sample["source_kind"] == "bank")
            suffix_len_sum += float(sample["suffix_len"])
            delta_norm_sum += float(sample["delta_norm"])

        self._last_branch_metrics = dict(self._last_branch_plan_metrics)
        self._last_branch_metrics.update(
            {
                "branch/generated": float(len(local_results)),
                "branch/accepted": float(accepted),
                "branch/accept_rate": float(accepted) / max(len(local_results), 1),
                "branch/generated_fail": float(generated_fail),
                "branch/generated_success": float(generated_success),
                "branch/generated_fail_rate": float(generated_fail)
                / max(len(local_results), 1),
                "branch/generated_success_rate": float(generated_success)
                / max(len(local_results), 1),
                "branch/immediate_success_rate": float(immediate_success)
                / max(len(local_results), 1),
                "branch/terminated_rate": float(terminated) / max(len(local_results), 1),
                "branch/generated_suffix_len_mean": generated_suffix_len_sum
                / max(len(local_results), 1),
                "branch/success_suffix_len_mean": success_suffix_len_sum
                / max(generated_success, 1),
                "branch/fail_suffix_len_mean": fail_suffix_len_sum
                / max(generated_fail, 1),
                "branch/search_attempts_mean": search_attempts_sum
                / max(len(local_results), 1),
                "branch/search_found_failure_rate": float(found_failure)
                / max(len(local_results), 1),
                "branch/scale_used_mean": scale_used_sum / max(len(local_results), 1),
                "branch/result_action_shift_norm_mean": result_action_shift_norm_sum
                / max(len(local_results), 1),
                "branch/result_action_shift_rel_mean": result_action_shift_rel_sum
                / max(len(local_results), 1),
                "branch/source_bank_frac": float(source_bank) / max(accepted, 1),
                "branch/suffix_len_mean": suffix_len_sum / max(accepted, 1),
                "branch/delta_norm_mean": delta_norm_sum / max(accepted, 1),
                "branch/recipe_bank_size": float(
                    sum(len(v) for v in self._branch_recipe_bank.values())
                ),
            }
        )
        self._log_progress(
            "[train][branch][load] "
            f"step={self.global_step} generated={len(local_results)} accepted={accepted} "
            f"fail_rate={float(generated_fail) / max(len(local_results), 1):.3f} "
            f"immediate_success_rate={float(immediate_success) / max(len(local_results), 1):.3f} "
            f"bank_size={int(self._last_branch_metrics['branch/recipe_bank_size'])}"
        )
        return self._last_branch_metrics

    def _compute_energy_for_model(
        self,
        model,
        nft_xt_all,
        nft_xnext_all,
        schedule,
        noise_level,
        data_dict=None,
        shared_cache=None,
        step_indices_all: torch.Tensor | None = None,
        return_chunk_embed: bool = False,
    ):
        """Run *model* on all K solver steps and return energy [B, K].

        ``nft_xt_all``: [B, K, horizon, dim]
        ``data_dict``: observation context with batch dim B.
        """
        from rlinf.algorithms.losses import compute_flow_sde_energy, _prepare_schedule_params

        fwd_data = data_dict if data_dict is not None else self._prm_fwd_data
        B, K = nft_xt_all.shape[:2]
        xt_flat = nft_xt_all.reshape(B * K, *nft_xt_all.shape[2:])
        xnext_flat = nft_xnext_all.reshape(B * K, *nft_xnext_all.shape[2:])
        if step_indices_all is None:
            step_idx_flat = (
                torch.arange(K, device=xt_flat.device, dtype=torch.long)
                .unsqueeze(0)
                .expand(B, K)
                .reshape(-1)
            )
        else:
            if step_indices_all.shape[:2] != (B, K):
                raise ValueError(
                    "dual-credit energy expected step_indices_all shape "
                    f"{(B, K)}, got {tuple(step_indices_all.shape)}"
                )
            step_idx_flat = step_indices_all.to(
                device=xt_flat.device, dtype=torch.long
            ).reshape(-1)
        if noise_level is not None and torch.is_tensor(noise_level) and noise_level.ndim > 0:
            if noise_level.shape[0] != B:
                raise ValueError(
                    "dual-credit PRM expected noise_level batch dim "
                    f"{B}, got {tuple(noise_level.shape)}"
                )
            noise_level_flat = noise_level.repeat_interleave(K)
        else:
            noise_level_flat = noise_level
        t_flat, delta_flat, sigma_flat = _prepare_schedule_params(
            schedule,
            step_idx_flat,
            noise_level_flat,
            xt_flat,
        )
        t_input_flat = schedule[step_idx_flat]

        shared_cache_base = shared_cache
        if shared_cache_base is None:
            with torch.no_grad():
                shared_cache_base = self._build_nft_shared_cache(model, fwd_data)
        shared_cache_flat = self._repeat_nft_shared_cache(shared_cache_base, K)
        fwd_data_flat = self._repeat_nft_forward_context(fwd_data, K, B)

        with self.amp_context:
            output = model(
                data=fwd_data_flat,
                use_nft_loss=True,
                compute_values=False,
                nft_explicit_inputs={"x_t": xt_flat, "timesteps": t_input_flat},
                use_cache=False,
                shared_cache=shared_cache_flat,
                return_chunk_embed=return_chunk_embed,
            )
        v_flat = output["v_theta"]
        chunk_size = v_flat.shape[1]
        E_flat = compute_flow_sde_energy(
            v_flat,
            xt_flat[:, :chunk_size],
            xnext_flat[:, :chunk_size],
            t_flat,
            delta_flat,
            sigma_flat,
        )
        if E_flat.ndim != 1:
            raise ValueError(
                "dual-credit PRM expected scalar energy per chunk-step, got "
                f"E_flat shape={tuple(E_flat.shape)} from xt_flat={tuple(xt_flat.shape)}, "
                f"xnext_flat={tuple(xnext_flat.shape)}, v_flat={tuple(v_flat.shape)}"
            )
        E = E_flat.reshape(B, K)
        if not return_chunk_embed:
            return E

        chunk_embed_flat = output.get("chunk_embed", None)
        chunk_token_embed_flat = output.get("chunk_token_embed", None)
        if chunk_embed_flat is None:
            raise ValueError(
                "dual-credit semantic pair mining expected `chunk_embed` in NFT forward output"
            )
        if chunk_embed_flat.ndim != 2 or chunk_embed_flat.shape[0] != B * K:
            raise ValueError(
                "dual-credit semantic pair mining expected chunk_embed shape "
                f"[(B*K)={B * K}, D], got {tuple(chunk_embed_flat.shape)}"
            )
        if chunk_token_embed_flat is None:
            raise ValueError(
                "dual-credit semantic pair mining expected `chunk_token_embed` in NFT forward output"
            )
        if chunk_token_embed_flat.ndim != 3 or chunk_token_embed_flat.shape[0] != B * K:
            raise ValueError(
                "dual-credit semantic pair mining expected chunk_token_embed shape "
                f"[(B*K)={B * K}, A, D], got {tuple(chunk_token_embed_flat.shape)}"
            )
        chunk_embed = chunk_embed_flat.reshape(B, K, chunk_embed_flat.shape[-1])
        chunk_token_embed = chunk_token_embed_flat.reshape(
            B,
            K,
            chunk_token_embed_flat.shape[1],
            chunk_token_embed_flat.shape[2],
        )
        return E, chunk_embed, chunk_token_embed

    def _run_prm_training(self, schedule):
        """Phase C: train implicit PRM with trajectory-level DPO."""
        self._onload_aux_models_for_prm()
        preference_pairs = self.rollout_batch.get("preference_pairs", [])
        candidate_preference_pairs = list(preference_pairs)
        preference_group_specs = self.rollout_batch.get("preference_group_specs", [])
        synthetic_branch_entries = list(getattr(self, "_synthetic_branch_entries", []))
        ref_model_for_prm = self.prev_rollout_model or self.ref_model
        if (
            (not preference_pairs and not preference_group_specs and not synthetic_branch_entries)
            or self.prm_model is None
            or ref_model_for_prm is None
        ):
            return {}

        nft_xt_all = self.rollout_batch.get("nft_xt_all", None)
        nft_xnext_all = self.rollout_batch.get("nft_xnext_all", None)
        noise_level = self.rollout_batch.get("nft_noise_level", None)
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if nft_xt_all is None or nft_xnext_all is None:
            return {}

        device = self.device

        # nft_xt_all shape: [n_chunk_steps, batch_size, K, chunk, dim]
        # Keep the 2D (n_chunk_steps, batch_size) structure for trajectory-level operations
        # For energy computation, we process per-trajectory: select batch index, flatten chunks
        n_chunk_steps, batch_size = nft_xt_all.shape[:2]
        prm_beta = self.cfg.algorithm.get("prm_beta", 0.5)
        prm_epochs = self.cfg.algorithm.get("prm_update_epochs", 2)
        prm_energy_batch_size = max(
            1, int(self.cfg.algorithm.get("prm_energy_batch_size", 1))
        )
        prm_score_mode = str(
            self.cfg.algorithm.get("prm_score_mode", "prefix_weighted_mean")
        ).lower()
        prm_pair_mode = str(
            self.cfg.algorithm.get("prm_pair_mode", "dual_head_hard_topk")
        ).lower()
        prm_hard_negative_k = int(
            self.cfg.algorithm.get("prm_hard_negative_k", 2)
        )
        prm_hard_negative_semantic_k = int(
            self.cfg.algorithm.get(
                "prm_hard_negative_semantic_k",
                1 if prm_hard_negative_k > 0 else 0,
            )
        )
        prm_hard_negative_semantic_k = max(prm_hard_negative_semantic_k, 0)
        prm_hard_negative_quality_k = int(
            self.cfg.algorithm.get(
                "prm_hard_negative_quality_k",
                max(prm_hard_negative_k - prm_hard_negative_semantic_k, 0),
            )
        )
        prm_hard_negative_quality_k = max(prm_hard_negative_quality_k, 0)
        if prm_hard_negative_semantic_k + prm_hard_negative_quality_k > prm_hard_negative_k:
            prm_hard_negative_quality_k = max(
                prm_hard_negative_k - prm_hard_negative_semantic_k, 0
            )
        prm_hard_negative_shortlist_k = int(
            self.cfg.algorithm.get(
                "prm_hard_negative_shortlist_k",
                max(prm_hard_negative_k, 4),
            )
        )
        prm_hard_negative_shortlist_k = max(
            prm_hard_negative_shortlist_k, prm_hard_negative_k, 1
        )
        prm_hard_negative_fail_reuse_cap = int(
            self.cfg.algorithm.get("prm_hard_negative_fail_reuse_cap", 0)
        )
        prm_pair_embed_source = str(
            self.cfg.algorithm.get("prm_pair_embed_source", "ref_suffix")
        ).lower()
        prm_pair_sim_metric = str(
            self.cfg.algorithm.get("prm_pair_sim_metric", "cosine")
        ).lower()
        prm_semantic_tail_weight = float(
            self.cfg.algorithm.get("prm_semantic_tail_weight", 0.5)
        )
        prm_semantic_early_penalty_weight = float(
            self.cfg.algorithm.get("prm_semantic_early_penalty_weight", 0.75)
        )
        prm_semantic_early_threshold = float(
            self.cfg.algorithm.get("prm_semantic_early_threshold", 0.85)
        )
        prm_semantic_early_prefix_frac = float(
            self.cfg.algorithm.get("prm_semantic_early_prefix_frac", 0.4)
        )
        prm_quality_semantic_min = float(
            self.cfg.algorithm.get("prm_quality_semantic_min", 0.8)
        )
        prm_quality_early_penalty_max = float(
            self.cfg.algorithm.get("prm_quality_early_penalty_max", 0.08)
        )
        prm_quality_mid_target = float(
            self.cfg.algorithm.get("prm_quality_mid_target", 0.9)
        )
        prm_quality_mid_width = float(
            self.cfg.algorithm.get("prm_quality_mid_width", 0.08)
        )
        prm_quality_weight_midband = float(
            self.cfg.algorithm.get("prm_quality_weight_midband", 1.0)
        )
        prm_quality_weight_action_path = float(
            self.cfg.algorithm.get("prm_quality_weight_action_path", 1.0)
        )
        prm_quality_weight_action_reversal = float(
            self.cfg.algorithm.get("prm_quality_weight_action_reversal", 0.75)
        )
        prm_quality_weight_action_gripper = float(
            self.cfg.algorithm.get("prm_quality_weight_action_gripper", 0.5)
        )
        prm_quality_weight_action_continuity = float(
            self.cfg.algorithm.get("prm_quality_weight_action_continuity", 0.5)
        )
        prm_quality_weight_state_path = float(
            self.cfg.algorithm.get("prm_quality_weight_state_path", 1.0)
        )
        prm_quality_weight_state_reversal = float(
            self.cfg.algorithm.get("prm_quality_weight_state_reversal", 0.75)
        )
        prm_quality_weight_state_gripper = float(
            self.cfg.algorithm.get("prm_quality_weight_state_gripper", 0.5)
        )
        prm_pair_weighting = bool(self.cfg.algorithm.get("prm_pair_weighting", True))
        prm_pair_weight_min = float(
            self.cfg.algorithm.get("prm_pair_weight_min", 0.05)
        )
        prm_pair_weight_min = min(max(prm_pair_weight_min, 0.0), 1.0)
        prm_pair_weight_rho = max(
            0.0, float(self.cfg.algorithm.get("prm_pair_weight_rho", 1.0))
        )
        prm_margin_weight_temp = max(
            1.0e-6, float(self.cfg.algorithm.get("prm_margin_weight_temp", 5.0))
        )
        prm_margin_weight_gamma = max(
            0.0, float(self.cfg.algorithm.get("prm_margin_weight_gamma", 1.0))
        )
        prm_synthetic_pair_weight = max(
            0.0, float(self.cfg.algorithm.get("prm_synthetic_pair_weight", 1.0))
        )
        if prm_pair_mode not in {
            "all_pairs",
            "semantic_hard_topk",
            "dual_head_hard_topk",
        }:
            raise ValueError(
                f"Unsupported prm_pair_mode={prm_pair_mode}. "
                "Expected one of: all_pairs, semantic_hard_topk, dual_head_hard_topk."
            )
        log_interval_s = float(
            self.cfg.algorithm.get("train_progress_log_interval_s", 120.0)
        )
        prm_start_time = time.perf_counter()
        self._log_progress(
            "[train][prm] start "
            f"step={self.global_step} candidate_pairs={len(preference_pairs)} "
            f"synthetic_pairs={len(synthetic_branch_entries)} "
            f"batch_size={batch_size} n_chunk_steps={n_chunk_steps} "
            f"prm_epochs={prm_epochs} energy_batch={prm_energy_batch_size} "
            f"score_mode={prm_score_mode} "
            f"pair_mode={prm_pair_mode} hard_k={prm_hard_negative_k} "
            f"semantic_k={prm_hard_negative_semantic_k} "
            f"quality_k={prm_hard_negative_quality_k} "
            f"shortlist_k={prm_hard_negative_shortlist_k} "
            f"fail_reuse_cap={prm_hard_negative_fail_reuse_cap} "
            f"pair_weighting={prm_pair_weighting} weight_min={prm_pair_weight_min:.3f} "
            f"margin_temp={prm_margin_weight_temp:.3f}"
        )

        nft_xt_all = nft_xt_all.to(device)
        nft_xnext_all = nft_xnext_all.to(device)
        if noise_level is not None:
            noise_level = noise_level.to(device)
            # noise_level: [n_chunk_steps, batch_size] -> keep as is
        chunk_loss_mask = None
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)
            chunk_loss_mask = self._coerce_chunk_mask(
                loss_mask, n_chunk_steps=n_chunk_steps, batch_size=batch_size
            )
        chunk_valid_fraction = self._coerce_chunk_values(
            put_tensor_device(self.rollout_batch.get("chunk_valid_fraction", None), device)
            if self.rollout_batch.get("chunk_valid_fraction", None) is not None
            else None,
            n_chunk_steps=n_chunk_steps,
            batch_size=batch_size,
        )
        chunk_partial_mask = self._coerce_chunk_values(
            put_tensor_device(self.rollout_batch.get("chunk_partial_mask", None), device)
            if self.rollout_batch.get("chunk_partial_mask", None) is not None
            else None,
            n_chunk_steps=n_chunk_steps,
            batch_size=batch_size,
        )
        if chunk_valid_fraction is not None:
            chunk_valid_fraction = chunk_valid_fraction.to(device=device, dtype=nft_xt_all.dtype)
        if chunk_partial_mask is not None:
            chunk_partial_mask = chunk_partial_mask.to(device=device).bool()

        # build full-batch data dict for model forward (observation context)
        full_data = self._extract_nft_forward_context({
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.rollout_batch.items()
            if k not in ("preference_pairs", "preference_group_specs")
        })
        self._prm_fwd_data = full_data

        def _slice_data_flat(data_dict, batch_idx):
            """Extract all chunks of trajectory batch_idx, flattened to [n_chunk_steps, ...]."""
            sliced = {}
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 2 and v.shape[1] == batch_size:
                    # [n_chunk_steps, batch_size, ...] -> [n_chunk_steps, ...]
                    sliced[k] = v[:, batch_idx]
                elif isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == batch_size:
                    # [batch_size, ...] -> [1, ...] then expand
                    sliced[k] = v[batch_idx:batch_idx+1].expand(n_chunk_steps, *v.shape[1:])
                else:
                    sliced[k] = v
            return sliced

        def _slice_data_flat_batch(data_dict, batch_indices: list[int]):
            """Extract a trajectory mini-batch, flattened as [B_traj * n_chunk_steps, ...]."""
            if len(batch_indices) == 0:
                raise ValueError("dual-credit PRM expected non-empty batch_indices")
            idx = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
            n_traj = int(idx.numel())
            sliced = {}
            for k, v in data_dict.items():
                if (
                    isinstance(v, torch.Tensor)
                    and v.ndim >= 2
                    and v.shape[0] == n_chunk_steps
                    and v.shape[1] == batch_size
                ):
                    # [H, B, ...] -> [B_traj, H, ...] -> [B_traj * H, ...]
                    selected = v.index_select(1, idx)
                    permute_order = [1, 0] + list(range(2, selected.ndim))
                    selected = selected.permute(*permute_order).contiguous()
                    sliced[k] = selected.reshape(
                        n_traj * n_chunk_steps, *selected.shape[2:]
                    )
                elif (
                    isinstance(v, torch.Tensor)
                    and v.ndim >= 1
                    and v.shape[0] == batch_size
                ):
                    # [B, ...] -> [B_traj, H, ...] -> [B_traj * H, ...]
                    selected = v.index_select(0, idx)
                    expanded = selected.unsqueeze(1).expand(
                        n_traj, n_chunk_steps, *selected.shape[1:]
                    )
                    sliced[k] = expanded.contiguous().reshape(
                        n_traj * n_chunk_steps, *selected.shape[1:]
                    )
                else:
                    sliced[k] = v
            return sliced

        def _compute_traj_energy_batch(
            model,
            batch_indices: list[int],
            data_dict,
            *,
            return_chunk_embed: bool = False,
        ):
            """Compute energies for several trajectories without changing PRM math."""
            if len(batch_indices) == 0:
                raise ValueError("dual-credit PRM expected non-empty batch_indices")
            idx = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
            n_traj = int(idx.numel())

            # [H, B, K, ...] -> [B_traj, H, K, ...] -> [B_traj * H, K, ...]
            xt = nft_xt_all.index_select(1, idx)
            xn = nft_xnext_all.index_select(1, idx)
            permute_order = [1, 0] + list(range(2, xt.ndim))
            xt = xt.permute(*permute_order).contiguous().reshape(
                n_traj * n_chunk_steps, *nft_xt_all.shape[2:]
            )
            xn = xn.permute(*permute_order).contiguous().reshape(
                n_traj * n_chunk_steps, *nft_xnext_all.shape[2:]
            )
            if (
                noise_level is not None
                and torch.is_tensor(noise_level)
                and noise_level.ndim == 2
            ):
                nl = (
                    noise_level.index_select(1, idx)
                    .permute(1, 0)
                    .contiguous()
                    .reshape(n_traj * n_chunk_steps)
                )
            else:
                nl = noise_level
            sliced = _slice_data_flat_batch(data_dict, batch_indices)
            output = self._compute_energy_for_model(
                model,
                xt,
                xn,
                schedule,
                nl,
                data_dict=sliced,
                return_chunk_embed=return_chunk_embed,
            )
            if not return_chunk_embed:
                return output.reshape(n_traj, n_chunk_steps, *output.shape[1:])

            E, chunk_embed, chunk_token_embed = output
            return (
                E.reshape(n_traj, n_chunk_steps, *E.shape[1:]),
                chunk_embed.reshape(
                    n_traj, n_chunk_steps, *chunk_embed.shape[1:]
                ),
                chunk_token_embed.reshape(
                    n_traj, n_chunk_steps, *chunk_token_embed.shape[1:]
                ),
            )

        prm_energy_split_fallback_count = 0
        prm_energy_split_max_batch = 0
        prm_pair_batch_fallback_count = 0
        prm_traj_chunk_split_fallback_count = 0
        prm_traj_chunk_split_max_block = 0
        prm_traj_chunk_block_size = 4

        def _concat_energy_outputs(outputs, *, return_chunk_embed: bool):
            if not return_chunk_embed:
                return torch.cat(outputs, dim=0)

            energy_list = [item[0] for item in outputs]
            chunk_embed_list = [item[1] for item in outputs]
            chunk_token_embed_list = [item[2] for item in outputs]
            return (
                torch.cat(energy_list, dim=0),
                torch.cat(chunk_embed_list, dim=0),
                torch.cat(chunk_token_embed_list, dim=0),
            )

        def _compute_traj_energy_batch_safe(
            model,
            batch_indices: list[int],
            data_dict,
            *,
            return_chunk_embed: bool = False,
        ):
            nonlocal prm_energy_split_fallback_count
            nonlocal prm_energy_split_max_batch
            nonlocal prm_pair_batch_fallback_count

            if len(batch_indices) == 0:
                raise ValueError("dual-credit PRM expected non-empty batch_indices")

            try:
                return _compute_traj_energy_batch(
                    model,
                    batch_indices,
                    data_dict,
                    return_chunk_embed=return_chunk_embed,
                )
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if len(batch_indices) == 1:
                    raise

                prm_energy_split_fallback_count += 1
                prm_energy_split_max_batch = max(
                    prm_energy_split_max_batch, len(batch_indices)
                )
                if len(batch_indices) == 2:
                    prm_pair_batch_fallback_count += 1

                mid = len(batch_indices) // 2
                left = _compute_traj_energy_batch_safe(
                    model,
                    batch_indices[:mid],
                    data_dict,
                    return_chunk_embed=return_chunk_embed,
                )
                right = _compute_traj_energy_batch_safe(
                    model,
                    batch_indices[mid:],
                    data_dict,
                    return_chunk_embed=return_chunk_embed,
                )
                return _concat_energy_outputs(
                    [left, right],
                    return_chunk_embed=return_chunk_embed,
                )

        def _compute_traj_energy(
            model,
            batch_idx,
            data_dict,
            *,
            return_chunk_embed: bool = False,
        ):
            """Compute energy for all chunks of one trajectory."""
            # nft_xt_all[:, batch_idx] -> [n_chunk_steps, K, chunk, dim]
            xt = nft_xt_all[:, batch_idx]
            xn = nft_xnext_all[:, batch_idx]
            nl = noise_level[:, batch_idx] if noise_level is not None and noise_level.ndim == 2 else noise_level
            sliced = _slice_data_flat(data_dict, batch_idx)
            return self._compute_energy_for_model(
                model,
                xt,
                xn,
                schedule,
                nl,
                data_dict=sliced,
                return_chunk_embed=return_chunk_embed,
            )

        def _slice_traj_chunk_context_range(
            traj_data_dict: dict[str, Any],
            start_idx: int,
            end_idx: int,
        ) -> dict[str, Any]:
            sliced = {}
            for k, v in traj_data_dict.items():
                if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == n_chunk_steps:
                    sliced[k] = v[start_idx:end_idx]
                else:
                    sliced[k] = v
            return sliced

        def _aggregate_energy_block_score(
            E_block: torch.Tensor,
            E_ref_block: torch.Tensor,
            weights_block: torch.Tensor,
            *,
            denom: torch.Tensor | None,
        ) -> torch.Tensor | None:
            valid = weights_block > 0
            if not valid.any():
                return None
            r_block = -(prm_beta / 2.0) * (E_block - E_ref_block).sum(dim=1)
            if prm_score_mode == "weighted_sum":
                return (r_block * weights_block).sum()
            assert denom is not None
            return (r_block[valid] * weights_block[valid]).sum() / denom

        def _compute_traj_chunk_block_score(
            model,
            xt_all: torch.Tensor,
            xn_all: torch.Tensor,
            nl_all,
            traj_data_dict: dict[str, Any],
            E_ref_traj: torch.Tensor,
            weights: torch.Tensor,
            start_idx: int,
            end_idx: int,
            *,
            denom: torch.Tensor | None,
            requires_grad: bool,
        ) -> torch.Tensor | None:
            nonlocal prm_traj_chunk_split_fallback_count
            nonlocal prm_traj_chunk_split_max_block

            weights_block = weights[start_idx:end_idx]
            if not (weights_block > 0).any():
                return None

            xt = xt_all[start_idx:end_idx]
            xn = xn_all[start_idx:end_idx]
            if (
                nl_all is not None
                and torch.is_tensor(nl_all)
                and nl_all.ndim > 0
                and nl_all.shape[0] == n_chunk_steps
            ):
                nl = nl_all[start_idx:end_idx]
            else:
                nl = nl_all
            sliced_data = _slice_traj_chunk_context_range(
                traj_data_dict, start_idx, end_idx
            )

            try:
                if requires_grad:
                    E_block = self._compute_energy_for_model(
                        model,
                        xt,
                        xn,
                        schedule,
                        nl,
                        data_dict=sliced_data,
                    )
                else:
                    with torch.no_grad():
                        E_block = self._compute_energy_for_model(
                            model,
                            xt,
                            xn,
                            schedule,
                            nl,
                            data_dict=sliced_data,
                        )
            except torch.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                block_len = end_idx - start_idx
                if block_len <= 1:
                    raise
                prm_traj_chunk_split_fallback_count += 1
                prm_traj_chunk_split_max_block = max(
                    prm_traj_chunk_split_max_block, block_len
                )
                mid_idx = start_idx + block_len // 2
                left = _compute_traj_chunk_block_score(
                    model,
                    xt_all,
                    xn_all,
                    nl_all,
                    traj_data_dict,
                    E_ref_traj,
                    weights,
                    start_idx,
                    mid_idx,
                    denom=denom,
                    requires_grad=requires_grad,
                )
                right = _compute_traj_chunk_block_score(
                    model,
                    xt_all,
                    xn_all,
                    nl_all,
                    traj_data_dict,
                    E_ref_traj,
                    weights,
                    mid_idx,
                    end_idx,
                    denom=denom,
                    requires_grad=requires_grad,
                )
                if left is None:
                    return right
                if right is None:
                    return left
                return left + right

            return _aggregate_energy_block_score(
                E_block,
                E_ref_traj[start_idx:end_idx],
                weights_block,
                denom=denom,
            )

        def _compute_traj_score_chunkwise(
            model,
            batch_idx: int,
            data_dict: dict[str, Any],
            E_ref_traj: torch.Tensor,
            weights: torch.Tensor,
            *,
            requires_grad: bool,
        ) -> torch.Tensor | None:
            valid = weights > 0
            if not valid.any():
                return None
            denom = None
            if prm_score_mode != "weighted_sum":
                denom = weights[valid].sum().clamp_min(1.0e-6)

            xt_all = nft_xt_all[:, batch_idx]
            xn_all = nft_xnext_all[:, batch_idx]
            if noise_level is not None and torch.is_tensor(noise_level) and noise_level.ndim == 2:
                nl_all = noise_level[:, batch_idx]
            else:
                nl_all = noise_level
            traj_data = _slice_data_flat(data_dict, batch_idx)

            score = None
            for start_idx in range(0, n_chunk_steps, prm_traj_chunk_block_size):
                end_idx = min(start_idx + prm_traj_chunk_block_size, n_chunk_steps)
                block_score = _compute_traj_chunk_block_score(
                    model,
                    xt_all,
                    xn_all,
                    nl_all,
                    traj_data,
                    E_ref_traj,
                    weights,
                    start_idx,
                    end_idx,
                    denom=denom,
                    requires_grad=requires_grad,
                )
                if block_score is None:
                    continue
                score = block_score if score is None else score + block_score
            return score

        def _backward_traj_score_chunkwise(
            model,
            batch_idx: int,
            data_dict: dict[str, Any],
            E_ref_traj: torch.Tensor,
            weights: torch.Tensor,
            *,
            gradient: torch.Tensor,
        ) -> bool:
            valid = weights > 0
            if not valid.any():
                return False

            denom = None
            if prm_score_mode != "weighted_sum":
                denom = weights[valid].sum().clamp_min(1.0e-6)

            xt_all = nft_xt_all[:, batch_idx]
            xn_all = nft_xnext_all[:, batch_idx]
            if (
                noise_level is not None
                and torch.is_tensor(noise_level)
                and noise_level.ndim == 2
            ):
                nl_all = noise_level[:, batch_idx]
            else:
                nl_all = noise_level
            traj_data = _slice_data_flat(data_dict, batch_idx)

            block_ranges: list[tuple[int, int]] = []
            for start_idx in range(0, n_chunk_steps, prm_traj_chunk_block_size):
                end_idx = min(start_idx + prm_traj_chunk_block_size, n_chunk_steps)
                if (weights[start_idx:end_idx] > 0).any():
                    block_ranges.append((start_idx, end_idx))
            if not block_ranges:
                return False

            grad_scalar = gradient.detach()
            no_sync_fn = getattr(model, "no_sync", None)
            last_block_idx = len(block_ranges) - 1
            any_backward = False
            for block_idx, (start_idx, end_idx) in enumerate(block_ranges):
                sync_ctx = nullcontext()
                if callable(no_sync_fn) and block_idx < last_block_idx:
                    sync_ctx = no_sync_fn()
                with sync_ctx:
                    block_score = _compute_traj_chunk_block_score(
                        model,
                        xt_all,
                        xn_all,
                        nl_all,
                        traj_data,
                        E_ref_traj,
                        weights,
                        start_idx,
                        end_idx,
                        denom=denom,
                        requires_grad=True,
                    )
                    if block_score is None:
                        continue
                    block_score.backward(
                        gradient=grad_scalar.to(
                            device=block_score.device, dtype=block_score.dtype
                        )
                    )
                    any_backward = True
            return any_backward

        def _compute_synthetic_score(
            model,
            sample: dict[str, Any],
            E_ref_synth: torch.Tensor,
            weights: torch.Tensor,
            *,
            requires_grad: bool,
        ) -> torch.Tensor | None:
            valid = weights > 0
            if not valid.any():
                return None
            if requires_grad:
                E_phi = _compute_synthetic_energy(model, sample)
            else:
                with torch.no_grad():
                    E_phi = _compute_synthetic_energy(model, sample)
            r_chunk = -(prm_beta / 2.0) * (E_phi - E_ref_synth).sum(dim=1)
            if prm_score_mode == "weighted_sum":
                return (r_chunk * weights).sum()
            denom = weights[valid].sum().clamp_min(1.0e-6)
            return (r_chunk[valid] * weights[valid]).sum() / denom

        def _compute_synthetic_energy(
            model,
            sample: dict[str, Any],
        ):
            suffix_data = sample["suffix_data"]
            xt = put_tensor_device(suffix_data["nft_xt_all"], device)
            xn = put_tensor_device(suffix_data["nft_xnext_all"], device)
            nl = put_tensor_device(suffix_data.get("nft_noise_level", None), device)
            fwd_data = self._extract_nft_forward_context(
                {
                    k: put_tensor_device(v, device) if isinstance(v, torch.Tensor) else v
                    for k, v in suffix_data.items()
                }
            )
            return self._compute_energy_for_model(
                model,
                xt,
                xn,
                schedule,
                nl,
                data_dict=fwd_data,
            )

        def _get_chunk_weights(batch_idx: int) -> torch.Tensor:
            if chunk_valid_fraction is not None:
                weights = chunk_valid_fraction[:, batch_idx]
            elif chunk_loss_mask is not None:
                weights = chunk_loss_mask[:, batch_idx].float()
            else:
                weights = torch.ones(
                    n_chunk_steps, device=device, dtype=nft_xt_all.dtype
                )
            return weights.to(device=device, dtype=nft_xt_all.dtype)

        def _truncate_weights_to_budget(weights: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
            """Keep the earliest prefix whose total effective weight matches `budget`."""
            clipped = torch.zeros_like(weights)
            remaining = budget.to(dtype=weights.dtype)
            for idx in range(weights.shape[0]):
                current = weights[idx]
                if float(remaining.item()) <= 0.0:
                    break
                take = torch.minimum(current, remaining)
                clipped[idx] = take
                remaining = remaining - take
            return clipped

        def _aggregate_chunk_score(
            r_chunk: torch.Tensor, weights: torch.Tensor
        ) -> torch.Tensor | None:
            valid = weights > 0
            if not valid.any():
                return None

            if prm_score_mode == "weighted_sum":
                return (r_chunk * weights).sum()

            denom = weights[valid].sum().clamp_min(1.0e-6)
            return (r_chunk[valid] * weights[valid]).sum() / denom

        action_chunk_size = int(
            getattr(self.cfg.actor.model, "num_action_chunks", 1)
        )

        def _align_chunk_sequence_tensor(
            values: torch.Tensor | None,
            *,
            chunk_size_hint: int,
        ) -> torch.Tensor | None:
            if values is None or not isinstance(values, torch.Tensor):
                return None
            values = values.detach().cpu().float()
            if values.ndim == 4:
                if values.shape[:2] == (n_chunk_steps, batch_size):
                    return values.contiguous()
                if values.shape[:2] == (batch_size, n_chunk_steps):
                    return values.permute(1, 0, 2, 3).contiguous()
            if values.ndim == 3 and chunk_size_hint > 0:
                if (
                    values.shape[:2] == (n_chunk_steps, batch_size)
                    and values.shape[-1] % chunk_size_hint == 0
                ):
                    return values.reshape(
                        n_chunk_steps, batch_size, chunk_size_hint, -1
                    ).contiguous()
                if (
                    values.shape[:2] == (batch_size, n_chunk_steps)
                    and values.shape[-1] % chunk_size_hint == 0
                ):
                    return (
                        values.reshape(batch_size, n_chunk_steps, chunk_size_hint, -1)
                        .permute(1, 0, 2, 3)
                        .contiguous()
                    )
            return None

        def _align_chunk_mask_tensor(
            mask: torch.Tensor | None,
            *,
            chunk_size_hint: int,
        ) -> torch.Tensor | None:
            if mask is None or not isinstance(mask, torch.Tensor):
                return None
            mask = mask.detach().cpu().bool()
            if mask.ndim == 3:
                if mask.shape[:2] == (n_chunk_steps, batch_size):
                    return mask.contiguous()
                if mask.shape[:2] == (batch_size, n_chunk_steps):
                    return mask.permute(1, 0, 2).contiguous()
            if mask.ndim == 4 and mask.shape[-1] == 1:
                return _align_chunk_mask_tensor(
                    mask.squeeze(-1), chunk_size_hint=chunk_size_hint
                )
            if mask.ndim == 2 and chunk_size_hint > 0:
                if mask.shape == (n_chunk_steps, batch_size * chunk_size_hint):
                    return mask.reshape(n_chunk_steps, batch_size, chunk_size_hint)
                if mask.shape == (batch_size, n_chunk_steps * chunk_size_hint):
                    return (
                        mask.reshape(batch_size, n_chunk_steps, chunk_size_hint)
                        .permute(1, 0, 2)
                        .contiguous()
                    )
            return None

        def _extract_first_last_valid(
            sequence: torch.Tensor,
            mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if sequence.shape[:3] != mask.shape:
                raise ValueError(
                    "dual-credit expected sequence/mask prefix match, got "
                    f"{tuple(sequence.shape)} vs {tuple(mask.shape)}"
                )
            if sequence.shape[2] == 0:
                zeros = torch.zeros(
                    *sequence.shape[:2],
                    sequence.shape[-1],
                    dtype=sequence.dtype,
                    device=sequence.device,
                )
                valid_any = torch.zeros(
                    *sequence.shape[:2], dtype=torch.bool, device=sequence.device
                )
                return zeros, zeros.clone(), valid_any

            valid_any = mask.any(dim=2)
            first_idx = mask.float().argmax(dim=2)
            last_idx = sequence.shape[2] - 1 - mask.flip(dims=[2]).float().argmax(dim=2)
            gather_first = first_idx.unsqueeze(-1).unsqueeze(-1).expand(
                *first_idx.shape, 1, sequence.shape[-1]
            )
            gather_last = last_idx.unsqueeze(-1).unsqueeze(-1).expand(
                *last_idx.shape, 1, sequence.shape[-1]
            )
            first = sequence.gather(2, gather_first).squeeze(2)
            last = sequence.gather(2, gather_last).squeeze(2)
            first = torch.where(valid_any.unsqueeze(-1), first, torch.zeros_like(first))
            last = torch.where(valid_any.unsqueeze(-1), last, torch.zeros_like(last))
            return first, last, valid_any

        def _compute_path_inefficiency(
            sequence: torch.Tensor,
            mask: torch.Tensor,
            motion_dim: int,
            *,
            eps: float = 1.0e-6,
        ) -> torch.Tensor:
            if motion_dim <= 0 or sequence.shape[2] <= 1:
                return torch.zeros(
                    sequence.shape[:2],
                    dtype=sequence.dtype,
                    device=sequence.device,
                )
            motion = sequence[..., :motion_dim]
            delta = motion[:, :, 1:] - motion[:, :, :-1]
            valid_pair = mask[:, :, 1:] & mask[:, :, :-1]
            step_len = delta.norm(dim=-1) * valid_pair.float()
            path_len = step_len.sum(dim=-1)
            first, last, valid_any = _extract_first_last_valid(motion, mask)
            net_len = (last - first).norm(dim=-1) * valid_any.float()
            ineff = (path_len - net_len).clamp_min(0.0) / path_len.clamp_min(eps)
            valid_count = mask.sum(dim=2)
            ineff = torch.where(valid_count >= 2, ineff, torch.zeros_like(ineff))
            return ineff.clamp(0.0, 1.0)

        def _compute_reversal_score(
            sequence: torch.Tensor,
            mask: torch.Tensor,
            motion_dim: int,
        ) -> torch.Tensor:
            if motion_dim <= 0 or sequence.shape[2] <= 2:
                return torch.zeros(
                    sequence.shape[:2],
                    dtype=sequence.dtype,
                    device=sequence.device,
                )
            motion = sequence[..., :motion_dim]
            delta = motion[:, :, 1:] - motion[:, :, :-1]
            valid_pair = mask[:, :, 1:] & mask[:, :, :-1]
            if delta.shape[2] <= 1:
                return torch.zeros(
                    sequence.shape[:2],
                    dtype=sequence.dtype,
                    device=sequence.device,
                )
            d0 = delta[:, :, :-1]
            d1 = delta[:, :, 1:]
            valid_triplet = valid_pair[:, :, :-1] & valid_pair[:, :, 1:]
            cos = torch.nn.functional.cosine_similarity(d0, d1, dim=-1, eps=1.0e-6)
            reversal = torch.relu(-cos) * valid_triplet.float()
            denom = valid_triplet.float().sum(dim=-1).clamp_min(1.0)
            score = reversal.sum(dim=-1) / denom
            has_triplet = valid_triplet.any(dim=-1)
            return torch.where(has_triplet, score, torch.zeros_like(score))

        def _compute_gripper_oscillation(
            sequence: torch.Tensor,
            mask: torch.Tensor,
            *,
            eps: float = 1.0e-6,
        ) -> torch.Tensor:
            if sequence.shape[-1] <= 0 or sequence.shape[2] <= 1:
                return torch.zeros(
                    sequence.shape[:2],
                    dtype=sequence.dtype,
                    device=sequence.device,
                )
            grip = sequence[..., -1:]
            delta = (grip[:, :, 1:] - grip[:, :, :-1]).abs().squeeze(-1)
            valid_pair = mask[:, :, 1:] & mask[:, :, :-1]
            total_var = (delta * valid_pair.float()).sum(dim=-1)
            first, last, valid_any = _extract_first_last_valid(grip, mask)
            net_var = (last - first).abs().squeeze(-1) * valid_any.float()
            osc = (total_var - net_var).clamp_min(0.0) / total_var.clamp_min(eps)
            valid_count = mask.sum(dim=2)
            osc = torch.where(valid_count >= 2, osc, torch.zeros_like(osc))
            return osc.clamp(0.0, 1.0)

        def _compute_chunk_continuity(
            sequence: torch.Tensor,
            mask: torch.Tensor,
            motion_dim: int,
            *,
            eps: float = 1.0e-6,
        ) -> torch.Tensor:
            continuity = torch.zeros(
                sequence.shape[:2],
                dtype=sequence.dtype,
                device=sequence.device,
            )
            if motion_dim <= 0 or sequence.shape[0] <= 1:
                return continuity
            motion = sequence[..., :motion_dim]
            first, last, valid_any = _extract_first_last_valid(motion, mask)
            prev_last = last[:-1]
            curr_first = first[1:]
            valid_link = valid_any[:-1] & valid_any[1:]
            jump = (curr_first - prev_last).norm(dim=-1)
            scale = curr_first.norm(dim=-1) + prev_last.norm(dim=-1)
            continuity_score = jump / (jump + scale + eps)
            continuity[1:] = torch.where(
                valid_link, continuity_score, torch.zeros_like(continuity_score)
            )
            return continuity.clamp(0.0, 1.0)

        def _aggregate_prefix_feature(
            feature_values: torch.Tensor,
            weights: torch.Tensor,
        ) -> float | None:
            valid = weights > 0
            if not valid.any():
                return None
            denom = weights[valid].sum().clamp_min(1.0e-6)
            return float((feature_values[valid] * weights[valid]).sum().item() / denom.item())

        def _get_chunk_weights_cpu(batch_idx: int) -> torch.Tensor:
            if chunk_valid_fraction_cpu is not None:
                weights = chunk_valid_fraction_cpu[:, batch_idx]
            elif chunk_loss_mask_cpu is not None:
                weights = chunk_loss_mask_cpu[:, batch_idx].float()
            else:
                weights = torch.ones(n_chunk_steps, dtype=torch.float32)
            return weights.to(dtype=torch.float32)

        def _compute_pair_semantic_stats(
            tau_plus: int,
            tau_minus: int,
            traj_chunk_token_embeds: torch.Tensor,
        ) -> dict[str, Any] | None:
            if prm_pair_sim_metric != "cosine":
                raise ValueError(
                    f"Unsupported prm_pair_sim_metric={prm_pair_sim_metric}. Expected `cosine`."
                )

            weights_plus_full = _get_chunk_weights_cpu(tau_plus)
            weights_minus_full = _get_chunk_weights_cpu(tau_minus)
            if float(weights_plus_full.sum().item()) <= 0.0 or float(
                weights_minus_full.sum().item()
            ) <= 0.0:
                return None

            shared_budget = torch.minimum(
                weights_plus_full.sum(), weights_minus_full.sum()
            )
            if float(shared_budget.item()) <= 0.0:
                return None

            weights_plus = _truncate_weights_to_budget(weights_plus_full, shared_budget)
            weights_minus = _truncate_weights_to_budget(weights_minus_full, shared_budget)
            common_weights = torch.minimum(weights_plus, weights_minus)
            valid = common_weights > 0
            if not valid.any():
                return None

            plus_tokens = traj_chunk_token_embeds[tau_plus]
            minus_tokens = traj_chunk_token_embeds[tau_minus]
            sim_per_token = torch.nn.functional.cosine_similarity(
                plus_tokens,
                minus_tokens,
                dim=-1,
                eps=1.0e-6,
            )
            sim_per_chunk = sim_per_token.mean(dim=-1)
            denom = common_weights[valid].sum().clamp_min(1.0e-6)
            semantic_mean = float(
                (sim_per_chunk[valid] * common_weights[valid]).sum().item() / denom.item()
            )

            valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
            prefix_count = max(
                1,
                int(math.ceil(prm_semantic_early_prefix_frac * valid_idx.numel())),
            )
            prefix_idx = valid_idx[:prefix_count]
            tail_idx = valid_idx[-prefix_count:]

            prefix_weights = common_weights[prefix_idx]
            tail_weights = common_weights[tail_idx]
            prefix_denom = prefix_weights.sum().clamp_min(1.0e-6)
            tail_denom = tail_weights.sum().clamp_min(1.0e-6)
            prefix_sim = sim_per_chunk[prefix_idx]
            tail_sim = sim_per_chunk[tail_idx]
            prefix_mean = float(
                (prefix_sim * prefix_weights).sum().item() / prefix_denom.item()
            )
            tail_mean = float((tail_sim * tail_weights).sum().item() / tail_denom.item())
            early_penalty = float(
                (
                    torch.relu(prm_semantic_early_threshold - prefix_sim) * prefix_weights
                ).sum().item()
                / prefix_denom.item()
            )
            semantic_midband = float(
                math.exp(
                    -0.5
                    * (
                        (semantic_mean - prm_quality_mid_target)
                        / max(prm_quality_mid_width, 1.0e-6)
                    )
                    ** 2
                )
            )
            semantic_score = (
                semantic_mean
                + prm_semantic_tail_weight * tail_mean
                - prm_semantic_early_penalty_weight * early_penalty
            )
            return {
                "common_weights": common_weights,
                "shared_budget": float(shared_budget.item()),
                "semantic_mean": semantic_mean,
                "semantic_prefix_mean": prefix_mean,
                "semantic_tail_mean": tail_mean,
                "semantic_early_penalty": early_penalty,
                "semantic_midband": semantic_midband,
                "semantic_score": float(semantic_score),
            }

        def _make_pair_select_rng(
            group_key: tuple[int, ...],
            tau_plus: int,
            *,
            salt: int = 0,
        ):
            seed = (
                int(self.cfg.actor.seed)
                + 1009 * int(self._rank)
                + 1000003 * int(self.global_step)
            )
            modulus = (1 << 63) - 1
            for value in group_key:
                seed = (
                    seed * 1000003
                    + int(value)
                    + 0x9E3779B97F4A7C15
                ) % modulus
            seed = (
                seed * 1000003
                + int(tau_plus)
                + int(salt)
                + 0xBF58476D1CE4E5B9
            ) % modulus
            return np.random.default_rng(seed if seed > 0 else 1)

        def _compute_rank_scores(
            candidates: list[dict[str, Any]],
            key: str,
        ) -> dict[int, float]:
            if len(candidates) == 0:
                return {}
            values = [float(item.get(key, 0.0)) for item in candidates]
            if max(values) - min(values) <= 1.0e-8:
                return {int(item["tau_minus"]): 1.0 for item in candidates}
            ordered = sorted(
                candidates,
                key=lambda item: (-float(item.get(key, 0.0)), int(item["tau_minus"])),
            )
            if len(ordered) == 1:
                return {int(ordered[0]["tau_minus"]): 1.0}
            rank_scores = {}
            denom = max(len(ordered) - 1, 1)
            for rank, item in enumerate(ordered):
                rank_scores[int(item["tau_minus"])] = 1.0 - rank / denom
            return rank_scores

        def _choose_ranked_failures(
            ranked_candidates: list[dict[str, Any]],
            *,
            target_k: int,
            group_key: tuple[int, ...],
            tau_plus: int,
            salt: int,
            fail_usage_count: dict[int, int],
            already_selected: set[int],
        ) -> tuple[list[dict[str, Any]], int, int, bool]:
            if target_k <= 0 or len(ranked_candidates) == 0:
                return [], 0, 0, False

            filtered = [
                item
                for item in ranked_candidates
                if int(item["tau_minus"]) not in already_selected
            ]
            if len(filtered) == 0:
                return [], 0, 0, False

            shortlist_k = min(
                len(filtered),
                max(target_k, prm_hard_negative_shortlist_k),
            )
            shortlist = filtered[:shortlist_k]
            eligible_shortlist = shortlist
            cap_limited = False
            if prm_hard_negative_fail_reuse_cap > 0:
                eligible_shortlist = [
                    item
                    for item in shortlist
                    if fail_usage_count.get(int(item["tau_minus"]), 0)
                    < prm_hard_negative_fail_reuse_cap
                ]
                if len(eligible_shortlist) < min(target_k, len(shortlist)):
                    cap_limited = True

            chosen: list[dict[str, Any]] = []
            if eligible_shortlist:
                sample_k = min(target_k, len(eligible_shortlist))
                if sample_k >= len(eligible_shortlist):
                    chosen = list(eligible_shortlist[:sample_k])
                else:
                    rng = _make_pair_select_rng(
                        group_key, tau_plus, salt=salt
                    )
                    sampled_idx = rng.choice(
                        len(eligible_shortlist), size=sample_k, replace=False
                    )
                    chosen = [eligible_shortlist[int(idx)] for idx in sampled_idx]
                    chosen.sort(
                        key=lambda item: (
                            -float(item.get("semantic_score", 0.0)),
                            int(item["tau_minus"]),
                        )
                    )

            backfill_count = 0
            if len(chosen) < target_k:
                chosen_ids = {int(item["tau_minus"]) for item in chosen}
                for item in filtered[shortlist_k:]:
                    tau_minus = int(item["tau_minus"])
                    if tau_minus in chosen_ids or tau_minus in already_selected:
                        continue
                    if (
                        prm_hard_negative_fail_reuse_cap > 0
                        and fail_usage_count.get(tau_minus, 0)
                        >= prm_hard_negative_fail_reuse_cap
                    ):
                        cap_limited = True
                        continue
                    chosen.append(item)
                    chosen_ids.add(tau_minus)
                    backfill_count += 1
                    if len(chosen) >= target_k:
                        break

            return chosen, shortlist_k, backfill_count, cap_limited

        chunk_loss_mask_cpu = (
            chunk_loss_mask.detach().cpu() if chunk_loss_mask is not None else None
        )
        chunk_valid_fraction_cpu = (
            chunk_valid_fraction.detach().cpu()
            if chunk_valid_fraction is not None
            else None
        )
        primitive_chunk_mask_cpu = _align_chunk_mask_tensor(
            self.rollout_batch.get("primitive_chunk_mask", None),
            chunk_size_hint=action_chunk_size,
        )
        executed_action_cpu = _align_chunk_sequence_tensor(
            self.rollout_batch.get("executed_action", None),
            chunk_size_hint=action_chunk_size,
        )
        if executed_action_cpu is None:
            executed_action_cpu = _align_chunk_sequence_tensor(
                self.rollout_batch.get("action", None),
                chunk_size_hint=action_chunk_size,
            )
        chunk_state_trace_cpu = _align_chunk_sequence_tensor(
            self.rollout_batch.get("chunk_state_trace", None),
            chunk_size_hint=action_chunk_size,
        )

        if primitive_chunk_mask_cpu is None and executed_action_cpu is not None:
            primitive_chunk_mask_cpu = torch.ones(
                executed_action_cpu.shape[:3], dtype=torch.bool
            )
        if primitive_chunk_mask_cpu is None and chunk_state_trace_cpu is not None:
            primitive_chunk_mask_cpu = torch.ones(
                chunk_state_trace_cpu.shape[:3], dtype=torch.bool
            )

        quality_feature_tensors: dict[str, torch.Tensor] = {}
        action_prior_available = 0.0
        state_prior_available = 0.0

        if (
            executed_action_cpu is not None
            and primitive_chunk_mask_cpu is not None
            and executed_action_cpu.shape[:3] == primitive_chunk_mask_cpu.shape
        ):
            action_prior_available = 1.0
            action_motion_dim = (
                min(executed_action_cpu.shape[-1] - 1, 3)
                if executed_action_cpu.shape[-1] > 1
                else executed_action_cpu.shape[-1]
            )
            quality_feature_tensors["action_path"] = _compute_path_inefficiency(
                executed_action_cpu,
                primitive_chunk_mask_cpu,
                action_motion_dim,
            )
            quality_feature_tensors["action_reversal"] = _compute_reversal_score(
                executed_action_cpu,
                primitive_chunk_mask_cpu,
                action_motion_dim,
            )
            quality_feature_tensors["action_gripper"] = _compute_gripper_oscillation(
                executed_action_cpu,
                primitive_chunk_mask_cpu,
            )
            quality_feature_tensors["action_continuity"] = _compute_chunk_continuity(
                executed_action_cpu,
                primitive_chunk_mask_cpu,
                action_motion_dim,
            )

        if (
            chunk_state_trace_cpu is not None
            and primitive_chunk_mask_cpu is not None
            and chunk_state_trace_cpu.shape[:3] == primitive_chunk_mask_cpu.shape
        ):
            state_prior_available = 1.0
            state_motion_dim = min(chunk_state_trace_cpu.shape[-1], 3)
            quality_feature_tensors["state_path"] = _compute_path_inefficiency(
                chunk_state_trace_cpu,
                primitive_chunk_mask_cpu,
                state_motion_dim,
            )
            quality_feature_tensors["state_reversal"] = _compute_reversal_score(
                chunk_state_trace_cpu,
                primitive_chunk_mask_cpu,
                state_motion_dim,
            )
            quality_feature_tensors["state_gripper"] = _compute_gripper_oscillation(
                chunk_state_trace_cpu,
                primitive_chunk_mask_cpu,
            )

        # Pre-compute energies under the frozen rollout snapshot used as the PRM reference.
        ref_model_for_prm.eval()
        E_ref_all = []
        ref_chunk_token_embed_all = []
        ref_start_time = time.perf_counter()
        last_ref_log_time = ref_start_time
        ref_log_every = max(1, batch_size // 4)
        with torch.no_grad():
            for batch_start in range(0, batch_size, prm_energy_batch_size):
                batch_end = min(batch_start + prm_energy_batch_size, batch_size)
                batch_indices = list(range(batch_start, batch_end))
                if prm_pair_mode in {"semantic_hard_topk", "dual_head_hard_topk"}:
                    if prm_pair_embed_source != "ref_suffix":
                        raise ValueError(
                            "dual-credit hard negative mining currently supports "
                            f"only prm_pair_embed_source=`ref_suffix`, got {prm_pair_embed_source}"
                        )
                    E_ref_b, _, chunk_token_embed_b = _compute_traj_energy_batch_safe(
                        ref_model_for_prm,
                        batch_indices,
                        full_data,
                        return_chunk_embed=True,
                    )
                    E_ref_all.append(E_ref_b)
                    ref_chunk_token_embed_all.append(
                        chunk_token_embed_b[:, :, -1].detach().cpu()
                    )
                else:
                    E_ref_all.append(
                        _compute_traj_energy_batch_safe(
                            ref_model_for_prm, batch_indices, full_data
                        )
                    )
                current = batch_end
                now = time.perf_counter()
                if self._should_log_progress(
                    current,
                    batch_size,
                    last_ref_log_time,
                    now,
                    every_n=ref_log_every,
                    every_s=log_interval_s,
                ):
                    elapsed = now - ref_start_time
                    self._log_progress(
                        "[train][prm][ref] "
                        f"{current}/{batch_size} traj "
                        f"elapsed={elapsed:.1f}s avg_per_traj={elapsed / max(current, 1):.2f}s"
                    )
                    last_ref_log_time = now
        E_ref_all = torch.cat(E_ref_all, dim=0)  # [batch_size, n_chunk_steps, K]
        if len(ref_chunk_token_embed_all) > 0:
            ref_chunk_token_embed_all = torch.cat(
                ref_chunk_token_embed_all, dim=0
            )  # [B, H, A, D]
        else:
            ref_chunk_token_embed_all = None
        ref_elapsed = time.perf_counter() - ref_start_time
        synthetic_ref_energies = []
        if len(synthetic_branch_entries) > 0:
            synth_ref_start_time = time.perf_counter()
            with torch.no_grad():
                for sample in synthetic_branch_entries:
                    synthetic_ref_energies.append(
                        _compute_synthetic_energy(ref_model_for_prm, sample)
                    )
            self._log_progress(
                "[train][prm][branch-ref] "
                f"samples={len(synthetic_ref_energies)} "
                f"elapsed={time.perf_counter() - synth_ref_start_time:.1f}s"
            )

        pair_selection_start_time = time.perf_counter()
        all_pair_similarity_sum = 0.0
        all_pair_similarity_sq_sum = 0.0
        all_pair_similarity_min = None
        all_pair_similarity_max = None
        all_pair_similarity_count = 0
        selected_pair_similarity_sum = 0.0
        selected_pair_similarity_sq_sum = 0.0
        selected_pair_similarity_min = None
        selected_pair_similarity_max = None
        selected_pair_similarity_count = 0
        selected_fails_per_positive = []
        selected_unique_fails_per_group = []
        shortlist_sizes = []
        shortlist_selected_count = 0
        backfill_selected_count = 0
        reuse_cap_limited_positive_count = 0
        reuse_cap_limited_group_count = 0
        fail_reuse_mean_per_group = []
        fail_reuse_max_per_group = []
        semantic_score_all_sum = 0.0
        semantic_score_selected_sum = 0.0
        semantic_score_selected_count = 0
        semantic_tail_all_sum = 0.0
        semantic_tail_selected_sum = 0.0
        semantic_early_penalty_all_sum = 0.0
        semantic_early_penalty_selected_sum = 0.0
        semantic_midband_all_sum = 0.0
        semantic_midband_selected_sum = 0.0
        quality_gate_pass_count = 0
        quality_candidate_count = 0
        quality_score_sum = 0.0
        quality_score_count = 0
        quality_score_selected_sum = 0.0
        quality_score_selected_count = 0
        semantic_head_selected_count = 0
        quality_head_selected_count = 0
        fallback_head_selected_count = 0
        candidate_fails_per_positive = []
        selected_feature_excess_sums = {
            "action_path": 0.0,
            "action_reversal": 0.0,
            "action_gripper": 0.0,
            "action_continuity": 0.0,
            "state_path": 0.0,
            "state_reversal": 0.0,
            "state_gripper": 0.0,
        }
        all_feature_excess_sums = {
            key: 0.0 for key in selected_feature_excess_sums
        }
        candidate_pairs_total = len(candidate_preference_pairs)
        selected_preference_pairs = preference_pairs
        selected_pair_static_hard_scores = [1.0 for _ in selected_preference_pairs]
        selected_pair_group_ids = [None for _ in selected_preference_pairs]

        def _compute_static_pair_hardness(item: dict[str, Any]) -> float:
            """Detached pair prior: high when the failure is similar but locally flawed."""
            semantic_midband = min(
                max(float(item.get("semantic_midband", 0.0)), 0.0), 1.0
            )
            semantic_mean = min(
                max(float(item.get("semantic_mean", semantic_midband)), 0.0), 1.0
            )
            semantic_hard = min(
                max(0.5 * semantic_midband + 0.5 * semantic_mean, 0.0), 1.0
            )
            quality_hard = float(item.get("quality_score", 0.0))
            if quality_hard <= 0.0:
                quality_hard = semantic_hard
            quality_hard = min(max(quality_hard, 0.0), 1.0)
            return float(math.sqrt(max(semantic_hard * quality_hard, 0.0)))

        def _calibrate_static_pair_hardness(
            raw_scores: list[float],
            group_ids: list[tuple[int, ...] | None],
        ) -> tuple[list[float], list[float], list[float], list[float]]:
            """Expand compressed raw hardness into a wider, group-relative prior.

            Raw hardness scores from pair mining are useful for ordering, but in
            practice they are often numerically compressed into a narrow band.
            We therefore remap them within each rollout group using:
            1. rank percentile among selected natural pairs in the group
            2. min-max spread within the same group
            3. a fixed sigmoid to make easy pairs clearly small and hard pairs
               clearly large without introducing extra config surface
            """
            if len(raw_scores) == 0:
                return [], [], [], []

            clipped = [min(max(float(score), 0.0), 1.0) for score in raw_scores]
            calibrated = [0.5 for _ in clipped]
            percentiles = [0.5 for _ in clipped]
            rank_signals = [0.5 for _ in clipped]
            grouped_indices: dict[tuple[int, ...] | tuple[str], list[int]] = {}
            for idx, group_id in enumerate(group_ids):
                key = group_id if group_id is not None else ("__global__",)
                grouped_indices.setdefault(key, []).append(idx)

            static_sigmoid_temp = 0.12
            for indices in grouped_indices.values():
                group_values = [clipped[idx] for idx in indices]
                if len(indices) == 1:
                    spread_norms = [0.5]
                else:
                    ranked_local = sorted(
                        range(len(indices)),
                        key=lambda local_idx: (
                            group_values[local_idx],
                            indices[local_idx],
                        ),
                    )
                    for order, local_idx in enumerate(ranked_local):
                        percentiles[indices[local_idx]] = (order + 0.5) / len(indices)
                    min_value = min(group_values)
                    max_value = max(group_values)
                    if max_value - min_value <= 1.0e-6:
                        spread_norms = [0.5 for _ in group_values]
                    else:
                        spread_norms = [
                            (value - min_value) / (max_value - min_value)
                            for value in group_values
                        ]

                for local_idx, idx in enumerate(indices):
                    if len(indices) == 1:
                        percentiles[idx] = 0.5
                    rank_signal = 0.5 * percentiles[idx] + 0.5 * spread_norms[local_idx]
                    rank_signals[idx] = rank_signal
                    calibrated[idx] = 1.0 / (
                        1.0
                        + math.exp(-(rank_signal - 0.5) / static_sigmoid_temp)
                    )

            return clipped, calibrated, percentiles, rank_signals

        if prm_pair_mode in {"semantic_hard_topk", "dual_head_hard_topk"} and preference_group_specs:
            selected_preference_pairs = []
            selected_pair_static_hard_scores = []
            selected_pair_group_ids = []
            candidate_pairs_total = 0
            for spec in preference_group_specs:
                group_key = tuple(int(x) for x in spec.get("group_key", ()))
                pos_candidates = [int(x) for x in spec.get("pos_candidates", [])]
                fail_candidates = [int(x) for x in spec.get("fail_candidates", [])]
                if len(pos_candidates) == 0 or len(fail_candidates) == 0:
                    continue
                candidate_pairs_total += len(pos_candidates) * len(fail_candidates)
                selected_fails_in_group = set()
                fail_usage_count = {int(tau_minus): 0 for tau_minus in fail_candidates}
                group_cap_limited = False

                for tau_plus in pos_candidates:
                    candidate_fails_per_positive.append(len(fail_candidates))
                    scored_failures = []
                    for tau_minus in fail_candidates:
                        semantic_stats = _compute_pair_semantic_stats(
                            tau_plus, tau_minus, ref_chunk_token_embed_all
                        )
                        if semantic_stats is None:
                            continue
                        semantic_mean = float(semantic_stats["semantic_mean"])
                        semantic_score = float(semantic_stats["semantic_score"])
                        semantic_tail = float(semantic_stats["semantic_tail_mean"])
                        semantic_early_penalty = float(
                            semantic_stats["semantic_early_penalty"]
                        )
                        semantic_midband = float(semantic_stats["semantic_midband"])
                        common_weights = semantic_stats["common_weights"]

                        all_pair_similarity_sum += semantic_mean
                        all_pair_similarity_sq_sum += semantic_mean * semantic_mean
                        all_pair_similarity_count += 1
                        if (
                            all_pair_similarity_min is None
                            or semantic_mean < all_pair_similarity_min
                        ):
                            all_pair_similarity_min = semantic_mean
                        if (
                            all_pair_similarity_max is None
                            or semantic_mean > all_pair_similarity_max
                        ):
                            all_pair_similarity_max = semantic_mean

                        semantic_score_all_sum += semantic_score
                        semantic_tail_all_sum += semantic_tail
                        semantic_early_penalty_all_sum += semantic_early_penalty
                        semantic_midband_all_sum += semantic_midband

                        feature_excess = {}
                        for feature_name, feature_tensor in quality_feature_tensors.items():
                            plus_feature = _aggregate_prefix_feature(
                                feature_tensor[:, tau_plus], common_weights
                            )
                            minus_feature = _aggregate_prefix_feature(
                                feature_tensor[:, tau_minus], common_weights
                            )
                            if plus_feature is None or minus_feature is None:
                                excess = 0.0
                            else:
                                excess = max(minus_feature - plus_feature, 0.0)
                            feature_excess[feature_name] = float(excess)
                            all_feature_excess_sums[feature_name] += float(excess)

                        quality_gate = (
                            semantic_mean >= prm_quality_semantic_min
                            and semantic_early_penalty <= prm_quality_early_penalty_max
                        )
                        quality_candidate_count += 1
                        quality_gate_pass_count += int(quality_gate)
                        scored_failures.append(
                            {
                                "tau_minus": int(tau_minus),
                                "semantic_mean": semantic_mean,
                                "semantic_score": semantic_score,
                                "semantic_tail_mean": semantic_tail,
                                "semantic_early_penalty": semantic_early_penalty,
                                "semantic_midband": semantic_midband,
                                "quality_gate": quality_gate,
                                "shared_budget": float(semantic_stats["shared_budget"]),
                                "feature_excess": feature_excess,
                                "quality_score": 0.0,
                            }
                        )

                    if len(scored_failures) == 0:
                        selected_fails_per_positive.append(0)
                        continue

                    semantic_ranked = sorted(
                        scored_failures,
                        key=lambda item: (
                            -float(item["semantic_score"]),
                            -float(item["semantic_mean"]),
                            int(item["tau_minus"]),
                        ),
                    )

                    quality_ranked = []
                    if prm_pair_mode == "dual_head_hard_topk":
                        gated_failures = [
                            item for item in scored_failures if bool(item["quality_gate"])
                        ]
                        if gated_failures:
                            feature_weights = {
                                "semantic_midband": prm_quality_weight_midband,
                                "action_path": (
                                    prm_quality_weight_action_path
                                    if "action_path" in quality_feature_tensors
                                    else 0.0
                                ),
                                "action_reversal": (
                                    prm_quality_weight_action_reversal
                                    if "action_reversal" in quality_feature_tensors
                                    else 0.0
                                ),
                                "action_gripper": (
                                    prm_quality_weight_action_gripper
                                    if "action_gripper" in quality_feature_tensors
                                    else 0.0
                                ),
                                "action_continuity": (
                                    prm_quality_weight_action_continuity
                                    if "action_continuity" in quality_feature_tensors
                                    else 0.0
                                ),
                                "state_path": (
                                    prm_quality_weight_state_path
                                    if "state_path" in quality_feature_tensors
                                    else 0.0
                                ),
                                "state_reversal": (
                                    prm_quality_weight_state_reversal
                                    if "state_reversal" in quality_feature_tensors
                                    else 0.0
                                ),
                                "state_gripper": (
                                    prm_quality_weight_state_gripper
                                    if "state_gripper" in quality_feature_tensors
                                    else 0.0
                                ),
                            }
                            rank_scores = {
                                "semantic_midband": _compute_rank_scores(
                                    gated_failures, "semantic_midband"
                                )
                            }
                            for feature_name in selected_feature_excess_sums:
                                feature_candidates = [
                                    {
                                        "tau_minus": int(item["tau_minus"]),
                                        feature_name: float(
                                            item["feature_excess"].get(feature_name, 0.0)
                                        ),
                                    }
                                    for item in gated_failures
                                ]
                                rank_scores[feature_name] = _compute_rank_scores(
                                    feature_candidates, feature_name
                                )
                            for item in gated_failures:
                                tau_minus = int(item["tau_minus"])
                                score_num = 0.0
                                score_den = 0.0
                                for feature_name, weight in feature_weights.items():
                                    if weight <= 0.0:
                                        continue
                                    rank_score = rank_scores.get(feature_name, {}).get(
                                        tau_minus, 0.0
                                    )
                                    score_num += weight * rank_score
                                    score_den += weight
                                item["quality_score"] = score_num / max(score_den, 1.0e-6)
                                quality_score_sum += float(item["quality_score"])
                                quality_score_count += 1
                            quality_ranked = sorted(
                                gated_failures,
                                key=lambda item: (
                                    -float(item["quality_score"]),
                                    -float(item["semantic_midband"]),
                                    -float(item["semantic_mean"]),
                                    int(item["tau_minus"]),
                                ),
                            )

                    target_k = len(scored_failures)
                    if prm_hard_negative_k > 0:
                        target_k = min(target_k, prm_hard_negative_k)
                    if target_k <= 0:
                        selected_fails_per_positive.append(0)
                        continue

                    if prm_pair_mode == "semantic_hard_topk":
                        semantic_target = target_k
                        quality_target = 0
                    else:
                        semantic_target = min(prm_hard_negative_semantic_k, target_k)
                        quality_target = min(
                            prm_hard_negative_quality_k,
                            max(target_k - semantic_target, 0),
                        )

                    chosen_items: list[tuple[dict[str, Any], str]] = []
                    chosen_fail_ids: set[int] = set()
                    positive_cap_limited = False

                    chosen_semantic, shortlist_k_sem, backfill_sem, cap_limited_sem = (
                        _choose_ranked_failures(
                            semantic_ranked,
                            target_k=semantic_target,
                            group_key=group_key,
                            tau_plus=int(tau_plus),
                            salt=17,
                            fail_usage_count=fail_usage_count,
                            already_selected=chosen_fail_ids,
                        )
                    )
                    shortlist_sizes.append(shortlist_k_sem)
                    shortlist_selected_count += len(chosen_semantic) - backfill_sem
                    backfill_selected_count += backfill_sem
                    if cap_limited_sem:
                        positive_cap_limited = True
                    for item in chosen_semantic:
                        chosen_items.append((item, "semantic"))
                        chosen_fail_ids.add(int(item["tau_minus"]))

                    if quality_target > 0 and len(quality_ranked) > 0:
                        chosen_quality, shortlist_k_q, backfill_q, cap_limited_q = (
                            _choose_ranked_failures(
                                quality_ranked,
                                target_k=quality_target,
                                group_key=group_key,
                                tau_plus=int(tau_plus),
                                salt=29,
                                fail_usage_count=fail_usage_count,
                                already_selected=chosen_fail_ids,
                            )
                        )
                        shortlist_sizes.append(shortlist_k_q)
                        shortlist_selected_count += len(chosen_quality) - backfill_q
                        backfill_selected_count += backfill_q
                        if cap_limited_q:
                            positive_cap_limited = True
                        for item in chosen_quality:
                            chosen_items.append((item, "quality"))
                            chosen_fail_ids.add(int(item["tau_minus"]))

                    if len(chosen_items) < target_k:
                        fallback_ranked = sorted(
                            scored_failures,
                            key=lambda item: (
                                -max(
                                    float(item["semantic_score"]),
                                    float(item.get("quality_score", 0.0)),
                                ),
                                -float(item["semantic_mean"]),
                                int(item["tau_minus"]),
                            ),
                        )
                        chosen_fallback, shortlist_k_fb, backfill_fb, cap_limited_fb = (
                            _choose_ranked_failures(
                                fallback_ranked,
                                target_k=max(target_k - len(chosen_items), 0),
                                group_key=group_key,
                                tau_plus=int(tau_plus),
                                salt=43,
                                fail_usage_count=fail_usage_count,
                                already_selected=chosen_fail_ids,
                            )
                        )
                        shortlist_sizes.append(shortlist_k_fb)
                        shortlist_selected_count += len(chosen_fallback) - backfill_fb
                        backfill_selected_count += backfill_fb
                        if cap_limited_fb:
                            positive_cap_limited = True
                        for item in chosen_fallback:
                            chosen_items.append((item, "fallback"))
                            chosen_fail_ids.add(int(item["tau_minus"]))

                    if positive_cap_limited:
                        reuse_cap_limited_positive_count += 1
                        group_cap_limited = True

                    selected_fails_per_positive.append(len(chosen_items))
                    for item, head_name in chosen_items:
                        tau_minus = int(item["tau_minus"])
                        semantic_mean = float(item["semantic_mean"])
                        selected_preference_pairs.append((int(tau_plus), tau_minus))
                        selected_pair_static_hard_scores.append(
                            _compute_static_pair_hardness(item)
                        )
                        selected_pair_group_ids.append(group_key)
                        selected_fails_in_group.add(tau_minus)
                        fail_usage_count[tau_minus] = fail_usage_count.get(tau_minus, 0) + 1
                        selected_pair_similarity_sum += semantic_mean
                        selected_pair_similarity_sq_sum += semantic_mean * semantic_mean
                        selected_pair_similarity_count += 1
                        if (
                            selected_pair_similarity_min is None
                            or semantic_mean < selected_pair_similarity_min
                        ):
                            selected_pair_similarity_min = semantic_mean
                        if (
                            selected_pair_similarity_max is None
                            or semantic_mean > selected_pair_similarity_max
                        ):
                            selected_pair_similarity_max = semantic_mean
                        semantic_score_selected_sum += float(item["semantic_score"])
                        semantic_score_selected_count += 1
                        semantic_tail_selected_sum += float(item["semantic_tail_mean"])
                        semantic_early_penalty_selected_sum += float(
                            item["semantic_early_penalty"]
                        )
                        semantic_midband_selected_sum += float(item["semantic_midband"])
                        if bool(item.get("quality_gate", False)):
                            quality_score_selected_sum += float(item.get("quality_score", 0.0))
                            quality_score_selected_count += 1
                        for feature_name, feature_value in item["feature_excess"].items():
                            selected_feature_excess_sums[feature_name] += float(feature_value)
                        if head_name == "semantic":
                            semantic_head_selected_count += 1
                        elif head_name == "quality":
                            quality_head_selected_count += 1
                        else:
                            fallback_head_selected_count += 1

                selected_unique_fails_per_group.append(len(selected_fails_in_group))
                if group_cap_limited:
                    reuse_cap_limited_group_count += 1
                if selected_fails_in_group:
                    usage_values = [
                        float(fail_usage_count[tau_minus])
                        for tau_minus in selected_fails_in_group
                    ]
                    fail_reuse_mean_per_group.append(
                        float(sum(usage_values) / len(usage_values))
                    )
                    fail_reuse_max_per_group.append(float(max(usage_values)))

        if (
            prm_pair_mode in {"semantic_hard_topk", "dual_head_hard_topk"}
            and len(selected_preference_pairs) == 0
        ):
            self._log_progress(
                "[train][prm][pair_select] hard negative mining produced zero pairs; "
                "falling back to candidate preference pairs"
            )
            selected_preference_pairs = candidate_preference_pairs
            selected_pair_static_hard_scores = [1.0 for _ in selected_preference_pairs]
            selected_pair_group_ids = [None for _ in selected_preference_pairs]

        preference_pairs = selected_preference_pairs
        if len(selected_pair_static_hard_scores) != len(preference_pairs):
            selected_pair_static_hard_scores = [1.0 for _ in preference_pairs]
        if len(selected_pair_group_ids) != len(preference_pairs):
            selected_pair_group_ids = [None for _ in preference_pairs]
        (
            selected_pair_static_hard_raw_scores,
            selected_pair_static_hard_scores,
            selected_pair_static_percentiles,
            selected_pair_static_rank_signals,
        ) = _calibrate_static_pair_hardness(
            selected_pair_static_hard_scores,
            selected_pair_group_ids,
        )
        pair_selection_elapsed = time.perf_counter() - pair_selection_start_time
        if len(preference_pairs) == 0 and len(synthetic_branch_entries) == 0:
            return {}

        pair_selection_frac = len(preference_pairs) / max(candidate_pairs_total, 1)
        all_fail_similarity_mean = (
            all_pair_similarity_sum / max(all_pair_similarity_count, 1)
        )
        all_fail_similarity_std = 0.0
        if all_pair_similarity_count > 0:
            all_fail_similarity_std = max(
                all_pair_similarity_sq_sum / all_pair_similarity_count
                - all_fail_similarity_mean * all_fail_similarity_mean,
                0.0,
            ) ** 0.5
        hard_neg_similarity_mean = (
            selected_pair_similarity_sum / max(selected_pair_similarity_count, 1)
        )
        hard_neg_similarity_std = 0.0
        if selected_pair_similarity_count > 0:
            hard_neg_similarity_std = max(
                selected_pair_similarity_sq_sum / selected_pair_similarity_count
                - hard_neg_similarity_mean * hard_neg_similarity_mean,
                0.0,
            ) ** 0.5
        hard_negative_k_effective = (
            float(sum(selected_fails_per_positive) / len(selected_fails_per_positive))
            if selected_fails_per_positive
            else 0.0
        )
        group_selected_fail_mean = (
            float(sum(selected_unique_fails_per_group) / len(selected_unique_fails_per_group))
            if selected_unique_fails_per_group
            else 0.0
        )
        shortlist_size_mean = (
            float(sum(shortlist_sizes) / len(shortlist_sizes))
            if shortlist_sizes
            else 0.0
        )
        shortlist_sample_frac = shortlist_selected_count / max(
            len(preference_pairs), 1
        )
        backfill_sample_frac = backfill_selected_count / max(len(preference_pairs), 1)
        fail_reuse_mean = (
            float(sum(fail_reuse_mean_per_group) / len(fail_reuse_mean_per_group))
            if fail_reuse_mean_per_group
            else 0.0
        )
        fail_reuse_max = (
            float(max(fail_reuse_max_per_group))
            if fail_reuse_max_per_group
            else 0.0
        )
        semantic_score_all_mean = (
            semantic_score_all_sum / max(all_pair_similarity_count, 1)
        )
        semantic_score_selected_mean = (
            semantic_score_selected_sum / max(semantic_score_selected_count, 1)
        )
        semantic_tail_all_mean = (
            semantic_tail_all_sum / max(all_pair_similarity_count, 1)
        )
        semantic_tail_selected_mean = (
            semantic_tail_selected_sum / max(semantic_score_selected_count, 1)
        )
        semantic_early_penalty_all_mean = (
            semantic_early_penalty_all_sum / max(all_pair_similarity_count, 1)
        )
        semantic_early_penalty_selected_mean = (
            semantic_early_penalty_selected_sum / max(semantic_score_selected_count, 1)
        )
        semantic_midband_all_mean = (
            semantic_midband_all_sum / max(all_pair_similarity_count, 1)
        )
        semantic_midband_selected_mean = (
            semantic_midband_selected_sum / max(semantic_score_selected_count, 1)
        )
        quality_gate_pass_frac = quality_gate_pass_count / max(quality_candidate_count, 1)
        quality_score_mean = quality_score_sum / max(quality_score_count, 1)
        quality_score_selected_mean = (
            quality_score_selected_sum / max(quality_score_selected_count, 1)
        )
        candidate_fails_per_positive_mean = (
            float(sum(candidate_fails_per_positive) / len(candidate_fails_per_positive))
            if candidate_fails_per_positive
            else 0.0
        )
        semantic_head_pair_frac = semantic_head_selected_count / max(
            len(preference_pairs), 1
        )
        quality_head_pair_frac = quality_head_selected_count / max(
            len(preference_pairs), 1
        )
        fallback_head_pair_frac = fallback_head_selected_count / max(
            len(preference_pairs), 1
        )
        feature_excess_all_mean = {
            key: value / max(all_pair_similarity_count, 1)
            for key, value in all_feature_excess_sums.items()
        }
        feature_excess_selected_mean = {
            key: value / max(len(preference_pairs), 1)
            for key, value in selected_feature_excess_sums.items()
        }
        reuse_cap_limited_positive_frac = (
            reuse_cap_limited_positive_count / max(len(selected_fails_per_positive), 1)
        )
        reuse_cap_limited_group_frac = (
            reuse_cap_limited_group_count
            / max(len(selected_unique_fails_per_group), 1)
        )
        self._log_progress(
            "[train][prm][pair_select] "
            f"mode={prm_pair_mode} selected_pairs={len(preference_pairs)}/{candidate_pairs_total} "
            f"frac={pair_selection_frac:.3f} hard_sim_mean={hard_neg_similarity_mean:.4f} "
            f"all_sim_mean={all_fail_similarity_mean:.4f} shortlist_mean={shortlist_size_mean:.2f} "
            f"semantic_score_sel={semantic_score_selected_mean:.4f} "
            f"quality_gate_frac={quality_gate_pass_frac:.3f} "
            f"quality_score_sel={quality_score_selected_mean:.4f} "
            f"shortlist_frac={shortlist_sample_frac:.3f} backfill_frac={backfill_sample_frac:.3f} "
            f"fail_reuse_mean={fail_reuse_mean:.2f} fail_reuse_max={fail_reuse_max:.1f} "
            f"cap_limited_pos_frac={reuse_cap_limited_positive_frac:.3f} "
            f"elapsed={pair_selection_elapsed:.1f}s"
        )

        def _compute_dpo_pair_weight(
            pair_margin: torch.Tensor,
            static_hard_score: float,
            *,
            is_synthetic: bool = False,
        ) -> tuple[torch.Tensor, float, float]:
            """Detached importance weight for PRM DPO pairs.

            Static hardness comes from pair mining; dynamic hardness comes from
            the current detached DPO margin. Synthetic near-miss pairs are kept
            at an explicit fixed weight by default so rare generated negatives
            are not washed out by natural easy pairs.
            """
            if not prm_pair_weighting:
                one = torch.ones_like(pair_margin)
                return one, 1.0, 1.0
            if is_synthetic:
                weight = torch.ones_like(pair_margin) * prm_synthetic_pair_weight
                return weight, 1.0, 1.0

            static_component = min(max(float(static_hard_score), 0.0), 1.0)
            if prm_pair_weight_rho != 1.0:
                static_component = static_component ** prm_pair_weight_rho

            # Normalize sigmoid so margin=0 (undecided) keeps full dynamic weight,
            # while large positive margins are downweighted as already-easy pairs.
            margin_component_tensor = (
                2.0 * torch.sigmoid(-pair_margin.detach() / prm_margin_weight_temp)
            ).clamp(0.0, 1.0)
            if prm_margin_weight_gamma != 1.0:
                margin_component_tensor = margin_component_tensor.pow(
                    prm_margin_weight_gamma
                )
            raw = margin_component_tensor * static_component
            weight = prm_pair_weight_min + (1.0 - prm_pair_weight_min) * raw
            return (
                weight,
                static_component,
                float(margin_component_tensor.detach().item()),
            )

        # Train PRM
        total_loss_sum = 0.0
        unweighted_loss_sum = 0.0
        correct = 0
        total_pairs = 0
        pair_margin_sum = 0.0
        pair_margin_per_chunk_sum = 0.0
        pair_margin_sq_sum = 0.0
        pair_margin_min = None
        pair_margin_max = None
        pair_margin_gt_5 = 0
        pair_margin_gt_10 = 0
        saturated_pair_count = 0
        score_plus_sum = 0.0
        score_minus_sum = 0.0
        valid_chunks_plus_sum = 0.0
        valid_chunks_minus_sum = 0.0
        shared_budget_sum = 0.0
        shared_budget_sq_sum = 0.0
        minus_retained_frac_sum = 0.0
        synthetic_pair_count = 0
        synthetic_loss_sum = 0.0
        synthetic_unweighted_loss_sum = 0.0
        synthetic_margin_sum = 0.0
        pair_weight_sum = 0.0
        pair_weight_sq_sum = 0.0
        pair_weight_min = None
        pair_weight_max = None
        pair_weight_low_count = 0
        pair_weight_le_0p2_count = 0
        static_hard_sum = 0.0
        margin_weight_sum = 0.0
        natural_pair_weight_sum = 0.0
        natural_pair_weight_count = 0
        synthetic_pair_weight_sum = 0.0
        static_hard_raw_sum = 0.0
        static_hard_raw_sq_sum = 0.0
        static_hard_raw_min = None
        static_hard_raw_max = None
        static_hard_percentile_sum = 0.0
        static_hard_percentile_sq_sum = 0.0
        static_hard_rank_signal_sum = 0.0
        static_hard_rank_signal_sq_sum = 0.0
        static_hard_natural_sum = 0.0
        static_hard_natural_sq_sum = 0.0
        static_hard_low_count = 0
        static_hard_high_count = 0
        static_hard_natural_count = 0
        self.prm_model.train()
        self._preserve_frozen_vlm_eval_mode(self.prm_model)
        prm_trainable_params = [
            p for p in self.prm_model.parameters() if p.requires_grad
        ]
        pair_train_start_time = time.perf_counter()
        total_pair_iters = prm_epochs * (
            len(preference_pairs) + len(synthetic_branch_entries)
        )
        pair_iter_count = 0
        pair_log_every = max(1, total_pair_iters // 6)
        last_pair_log_time = pair_train_start_time
        for prm_epoch_idx in range(prm_epochs):
            self._log_progress(
                "[train][prm][dpo] "
                f"epoch={prm_epoch_idx + 1}/{prm_epochs} "
                f"pairs={len(preference_pairs)} synthetic={len(synthetic_branch_entries)}"
            )
            for pair_idx, (tau_plus, tau_minus) in enumerate(preference_pairs):
                E_ref_plus = E_ref_all[tau_plus]   # [n_chunk_steps, K]
                E_ref_minus = E_ref_all[tau_minus]
                weights_plus_full = _get_chunk_weights(tau_plus)
                weights_minus_full = _get_chunk_weights(tau_minus)
                if float(weights_plus_full.sum().item()) <= 0.0 or float(
                    weights_minus_full.sum().item()
                ) <= 0.0:
                    continue
                if prm_score_mode == "prefix_weighted_mean":
                    shared_budget = torch.minimum(
                        weights_plus_full.sum(), weights_minus_full.sum()
                    )
                    weights_plus = _truncate_weights_to_budget(
                        weights_plus_full, shared_budget
                    )
                    weights_minus = _truncate_weights_to_budget(
                        weights_minus_full, shared_budget
                    )
                elif prm_score_mode == "weighted_mean":
                    shared_budget = torch.minimum(
                        weights_plus_full.sum(), weights_minus_full.sum()
                    )
                    weights_plus = weights_plus_full
                    weights_minus = weights_minus_full
                elif prm_score_mode == "weighted_sum":
                    shared_budget = torch.minimum(
                        weights_plus_full.sum(), weights_minus_full.sum()
                    )
                    weights_plus = weights_plus_full
                    weights_minus = weights_minus_full
                else:
                    raise ValueError(
                        f"Unsupported prm_score_mode={prm_score_mode}. "
                        "Expected one of: weighted_mean, prefix_weighted_mean, weighted_sum."
                    )

                score_plus_det = _compute_traj_score_chunkwise(
                    self.prm_model,
                    tau_plus,
                    full_data,
                    E_ref_plus,
                    weights_plus,
                    requires_grad=False,
                )
                score_minus_det = _compute_traj_score_chunkwise(
                    self.prm_model,
                    tau_minus,
                    full_data,
                    E_ref_minus,
                    weights_minus,
                    requires_grad=False,
                )
                if score_plus_det is None or score_minus_det is None:
                    continue
                valid_plus_count = float((weights_plus > 0).sum().item())
                valid_minus_count = float((weights_minus > 0).sum().item())
                pair_margin = (score_plus_det - score_minus_det).detach().float()

                loss_unweighted = -torch.nn.functional.logsigmoid(
                    pair_margin
                )
                static_hard_score = (
                    selected_pair_static_hard_scores[pair_idx]
                    if pair_idx < len(selected_pair_static_hard_scores)
                    else 1.0
                )
                static_hard_raw_score = (
                    selected_pair_static_hard_raw_scores[pair_idx]
                    if pair_idx < len(selected_pair_static_hard_raw_scores)
                    else static_hard_score
                )
                static_hard_percentile = (
                    selected_pair_static_percentiles[pair_idx]
                    if pair_idx < len(selected_pair_static_percentiles)
                    else 0.5
                )
                static_hard_rank_signal = (
                    selected_pair_static_rank_signals[pair_idx]
                    if pair_idx < len(selected_pair_static_rank_signals)
                    else static_hard_percentile
                )
                pair_weight, static_component, margin_component = (
                    _compute_dpo_pair_weight(
                        pair_margin,
                        static_hard_score,
                        is_synthetic=False,
                    )
                )

                self.prm_optimizer.zero_grad(set_to_none=True)
                pair_grad_coeff = (
                    pair_weight.detach() * torch.sigmoid(-pair_margin)
                ).detach()
                plus_ok = _backward_traj_score_chunkwise(
                    self.prm_model,
                    tau_plus,
                    full_data,
                    E_ref_plus,
                    weights_plus,
                    gradient=(-pair_grad_coeff),
                )
                if not plus_ok:
                    continue
                minus_ok = _backward_traj_score_chunkwise(
                    self.prm_model,
                    tau_minus,
                    full_data,
                    E_ref_minus,
                    weights_minus,
                    gradient=pair_grad_coeff,
                )
                if not minus_ok:
                    self.prm_optimizer.zero_grad(set_to_none=True)
                    continue
                torch.nn.utils.clip_grad_norm_(prm_trainable_params, 1.0)
                self.prm_optimizer.step()
                loss = pair_weight * loss_unweighted

                total_loss_sum += loss.item()
                unweighted_loss_sum += loss_unweighted.item()
                correct += int((score_plus_det > score_minus_det).item())
                total_pairs += 1
                pair_weight_item = float(pair_weight.detach().item())
                pair_weight_sum += pair_weight_item
                pair_weight_sq_sum += pair_weight_item * pair_weight_item
                if pair_weight_min is None or pair_weight_item < pair_weight_min:
                    pair_weight_min = pair_weight_item
                if pair_weight_max is None or pair_weight_item > pair_weight_max:
                    pair_weight_max = pair_weight_item
                pair_weight_low_count += int(pair_weight_item <= 0.1)
                pair_weight_le_0p2_count += int(pair_weight_item <= 0.2)
                static_hard_sum += float(static_component)
                margin_weight_sum += float(margin_component)
                natural_pair_weight_sum += pair_weight_item
                natural_pair_weight_count += 1
                static_hard_raw_item = float(static_hard_raw_score)
                static_hard_raw_sum += static_hard_raw_item
                static_hard_raw_sq_sum += static_hard_raw_item * static_hard_raw_item
                if (
                    static_hard_raw_min is None
                    or static_hard_raw_item < static_hard_raw_min
                ):
                    static_hard_raw_min = static_hard_raw_item
                if (
                    static_hard_raw_max is None
                    or static_hard_raw_item > static_hard_raw_max
                ):
                    static_hard_raw_max = static_hard_raw_item
                static_hard_percentile_sum += float(static_hard_percentile)
                static_hard_percentile_sq_sum += float(static_hard_percentile) ** 2
                static_hard_rank_signal_sum += float(static_hard_rank_signal)
                static_hard_rank_signal_sq_sum += float(static_hard_rank_signal) ** 2
                static_hard_natural_sum += float(static_component)
                static_hard_natural_sq_sum += float(static_component) ** 2
                static_hard_low_count += int(float(static_component) <= 0.2)
                static_hard_high_count += int(float(static_component) >= 0.8)
                static_hard_natural_count += 1
                pair_margin_item = float(pair_margin.item())
                pair_margin_sum += pair_margin_item
                pair_margin_sq_sum += pair_margin_item * pair_margin_item
                pair_margin_per_chunk_sum += pair_margin_item / max(
                    (valid_plus_count + valid_minus_count) / 2.0, 1.0
                )
                if pair_margin_min is None or pair_margin_item < pair_margin_min:
                    pair_margin_min = pair_margin_item
                if pair_margin_max is None or pair_margin_item > pair_margin_max:
                    pair_margin_max = pair_margin_item
                pair_margin_gt_5 += int(pair_margin_item > 5.0)
                pair_margin_gt_10 += int(pair_margin_item > 10.0)
                saturated_pair_count += int(loss_unweighted.item() < 1.0e-3)
                score_plus_sum += float(score_plus_det.detach().item())
                score_minus_sum += float(score_minus_det.detach().item())
                valid_chunks_plus_sum += valid_plus_count
                valid_chunks_minus_sum += valid_minus_count
                shared_budget_item = float(shared_budget.detach().item())
                shared_budget_sum += shared_budget_item
                shared_budget_sq_sum += shared_budget_item * shared_budget_item
                minus_budget_total = float(weights_minus_full.sum().item())
                minus_retained_frac_sum += shared_budget_item / max(
                    minus_budget_total, 1.0e-6
                )
                pair_iter_count += 1

                now = time.perf_counter()
                if self._should_log_progress(
                    pair_iter_count,
                    total_pair_iters,
                    last_pair_log_time,
                    now,
                    every_n=pair_log_every,
                    every_s=log_interval_s,
                ):
                    elapsed = now - pair_train_start_time
                    self._log_progress(
                        "[train][prm][dpo] "
                        f"{pair_iter_count}/{total_pair_iters} pair_updates "
                        f"elapsed={elapsed:.1f}s avg_per_pair={elapsed / max(pair_iter_count, 1):.2f}s "
                        f"effective_pairs={total_pairs}"
                    )
                    last_pair_log_time = now

            for synth_idx, sample in enumerate(synthetic_branch_entries):
                tau_plus = int(sample["tau_plus"])
                branch_idx = int(sample["branch_chunk_idx"])
                pos_start = branch_idx + 1
                if pos_start >= n_chunk_steps:
                    continue

                E_ref_plus_full = E_ref_all[tau_plus]
                E_ref_minus = synthetic_ref_energies[synth_idx]

                synth_len = int(sample["suffix_len"])
                available = min(synth_len, n_chunk_steps - pos_start)
                if available <= 0:
                    continue

                weights_plus_full = _get_chunk_weights(tau_plus)[
                    pos_start : pos_start + available
                ]
                weights_minus_full = put_tensor_device(
                    sample["suffix_valid_fraction"][:available], device
                ).to(dtype=weights_plus_full.dtype)
                if float(weights_plus_full.sum().item()) <= 0.0 or float(
                    weights_minus_full.sum().item()
                ) <= 0.0:
                    continue

                shared_budget = torch.minimum(
                    weights_plus_full.sum(), weights_minus_full.sum()
                )
                if prm_score_mode == "prefix_weighted_mean":
                    weights_plus = _truncate_weights_to_budget(
                        weights_plus_full, shared_budget
                    )
                    weights_minus = _truncate_weights_to_budget(
                        weights_minus_full, shared_budget
                    )
                elif prm_score_mode in {"weighted_mean", "weighted_sum"}:
                    weights_plus = weights_plus_full
                    weights_minus = weights_minus_full
                else:
                    raise ValueError(
                        f"Unsupported prm_score_mode={prm_score_mode}. "
                        "Expected one of: weighted_mean, prefix_weighted_mean, weighted_sum."
                    )

                weights_plus_traj = weights_plus_full.new_zeros(n_chunk_steps)
                weights_plus_traj[pos_start : pos_start + available] = weights_plus
                score_plus_det = _compute_traj_score_chunkwise(
                    self.prm_model,
                    tau_plus,
                    full_data,
                    E_ref_plus_full,
                    weights_plus_traj,
                    requires_grad=False,
                )
                score_minus_det = _compute_synthetic_score(
                    self.prm_model,
                    sample,
                    E_ref_minus,
                    weights_minus,
                    requires_grad=False,
                )
                if score_plus_det is None or score_minus_det is None:
                    continue

                pair_margin = (score_plus_det - score_minus_det).detach().float()
                loss_unweighted = -torch.nn.functional.logsigmoid(
                    pair_margin
                )
                pair_weight, static_component, margin_component = (
                    _compute_dpo_pair_weight(
                        pair_margin,
                        1.0,
                        is_synthetic=True,
                    )
                )

                self.prm_optimizer.zero_grad(set_to_none=True)
                pair_grad_coeff = (
                    pair_weight.detach() * torch.sigmoid(-pair_margin)
                ).detach()
                plus_ok = _backward_traj_score_chunkwise(
                    self.prm_model,
                    tau_plus,
                    full_data,
                    E_ref_plus_full,
                    weights_plus_traj,
                    gradient=(-pair_grad_coeff),
                )
                if not plus_ok:
                    continue
                score_minus = _compute_synthetic_score(
                    self.prm_model,
                    sample,
                    E_ref_minus,
                    weights_minus,
                    requires_grad=True,
                )
                if score_minus is None:
                    self.prm_optimizer.zero_grad(set_to_none=True)
                    continue
                score_minus.backward(
                    gradient=pair_grad_coeff.to(
                        device=score_minus.device, dtype=score_minus.dtype
                    )
                )
                torch.nn.utils.clip_grad_norm_(prm_trainable_params, 1.0)
                self.prm_optimizer.step()
                loss = pair_weight * loss_unweighted

                valid_plus_count = float((weights_plus > 0).sum().item())
                valid_minus_count = float((weights_minus > 0).sum().item())
                total_loss_sum += loss.item()
                unweighted_loss_sum += loss_unweighted.item()
                correct += int((score_plus_det > score_minus_det).item())
                total_pairs += 1
                synthetic_pair_count += 1
                synthetic_loss_sum += float(loss.item())
                synthetic_unweighted_loss_sum += float(loss_unweighted.item())
                pair_weight_item = float(pair_weight.detach().item())
                pair_weight_sum += pair_weight_item
                pair_weight_sq_sum += pair_weight_item * pair_weight_item
                if pair_weight_min is None or pair_weight_item < pair_weight_min:
                    pair_weight_min = pair_weight_item
                if pair_weight_max is None or pair_weight_item > pair_weight_max:
                    pair_weight_max = pair_weight_item
                pair_weight_low_count += int(pair_weight_item <= 0.1)
                pair_weight_le_0p2_count += int(pair_weight_item <= 0.2)
                static_hard_sum += float(static_component)
                margin_weight_sum += float(margin_component)
                synthetic_pair_weight_sum += pair_weight_item
                pair_margin_item = float(pair_margin.item())
                synthetic_margin_sum += pair_margin_item
                pair_margin_sum += pair_margin_item
                pair_margin_sq_sum += pair_margin_item * pair_margin_item
                pair_margin_per_chunk_sum += pair_margin_item / max(
                    (valid_plus_count + valid_minus_count) / 2.0, 1.0
                )
                if pair_margin_min is None or pair_margin_item < pair_margin_min:
                    pair_margin_min = pair_margin_item
                if pair_margin_max is None or pair_margin_item > pair_margin_max:
                    pair_margin_max = pair_margin_item
                pair_margin_gt_5 += int(pair_margin_item > 5.0)
                pair_margin_gt_10 += int(pair_margin_item > 10.0)
                saturated_pair_count += int(loss_unweighted.item() < 1.0e-3)
                score_plus_sum += float(score_plus_det.detach().item())
                score_minus_sum += float(score_minus_det.detach().item())
                valid_chunks_plus_sum += valid_plus_count
                valid_chunks_minus_sum += valid_minus_count
                shared_budget_item = float(shared_budget.detach().item())
                shared_budget_sum += shared_budget_item
                shared_budget_sq_sum += shared_budget_item * shared_budget_item
                minus_budget_total = float(weights_minus_full.sum().item())
                minus_retained_frac_sum += shared_budget_item / max(
                    minus_budget_total, 1.0e-6
                )
                pair_iter_count += 1

                now = time.perf_counter()
                if self._should_log_progress(
                    pair_iter_count,
                    total_pair_iters,
                    last_pair_log_time,
                    now,
                    every_n=pair_log_every,
                    every_s=log_interval_s,
                ):
                    elapsed = now - pair_train_start_time
                    self._log_progress(
                        "[train][prm][dpo] "
                        f"{pair_iter_count}/{total_pair_iters} pair_updates "
                        f"elapsed={elapsed:.1f}s avg_per_pair={elapsed / max(pair_iter_count, 1):.2f}s "
                        f"effective_pairs={total_pairs}"
                    )
                    last_pair_log_time = now

        accuracy = correct / max(total_pairs, 1)
        avg_loss = total_loss_sum / max(total_pairs, 1)
        avg_unweighted_loss = unweighted_loss_sum / max(total_pairs, 1)
        avg_pair_margin = pair_margin_sum / max(total_pairs, 1)
        avg_pair_margin_per_chunk = pair_margin_per_chunk_sum / max(total_pairs, 1)
        avg_pair_margin_sq = pair_margin_sq_sum / max(total_pairs, 1)
        pair_margin_std = max(avg_pair_margin_sq - avg_pair_margin**2, 0.0) ** 0.5
        pair_margin_gt_5_frac = pair_margin_gt_5 / max(total_pairs, 1)
        pair_margin_gt_10_frac = pair_margin_gt_10 / max(total_pairs, 1)
        saturated_pair_frac = saturated_pair_count / max(total_pairs, 1)
        score_plus_mean = score_plus_sum / max(total_pairs, 1)
        score_minus_mean = score_minus_sum / max(total_pairs, 1)
        valid_chunks_plus_mean = valid_chunks_plus_sum / max(total_pairs, 1)
        valid_chunks_minus_mean = valid_chunks_minus_sum / max(total_pairs, 1)
        shared_budget_mean = shared_budget_sum / max(total_pairs, 1)
        shared_budget_sq_mean = shared_budget_sq_sum / max(total_pairs, 1)
        shared_budget_std = max(
            shared_budget_sq_mean - shared_budget_mean**2, 0.0
        ) ** 0.5
        minus_retained_frac_mean = minus_retained_frac_sum / max(total_pairs, 1)
        synthetic_pair_frac = synthetic_pair_count / max(total_pairs, 1)
        synthetic_loss_mean = synthetic_loss_sum / max(synthetic_pair_count, 1)
        synthetic_unweighted_loss_mean = synthetic_unweighted_loss_sum / max(
            synthetic_pair_count, 1
        )
        synthetic_margin_mean = synthetic_margin_sum / max(synthetic_pair_count, 1)
        pair_weight_mean = pair_weight_sum / max(total_pairs, 1)
        pair_weight_sq_mean = pair_weight_sq_sum / max(total_pairs, 1)
        pair_weight_std = max(pair_weight_sq_mean - pair_weight_mean**2, 0.0) ** 0.5
        pair_weight_low_frac = pair_weight_low_count / max(total_pairs, 1)
        pair_weight_le_0p2_frac = pair_weight_le_0p2_count / max(total_pairs, 1)
        static_hard_mean = static_hard_sum / max(total_pairs, 1)
        margin_weight_mean = margin_weight_sum / max(total_pairs, 1)
        natural_pair_weight_mean = natural_pair_weight_sum / max(
            natural_pair_weight_count, 1
        )
        synthetic_pair_weight_mean = synthetic_pair_weight_sum / max(
            synthetic_pair_count, 1
        )
        static_hard_raw_mean = static_hard_raw_sum / max(static_hard_natural_count, 1)
        static_hard_raw_sq_mean = static_hard_raw_sq_sum / max(
            static_hard_natural_count, 1
        )
        static_hard_raw_std = max(
            static_hard_raw_sq_mean - static_hard_raw_mean**2, 0.0
        ) ** 0.5
        static_hard_percentile_mean = static_hard_percentile_sum / max(
            static_hard_natural_count, 1
        )
        static_hard_percentile_sq_mean = static_hard_percentile_sq_sum / max(
            static_hard_natural_count, 1
        )
        static_hard_percentile_std = max(
            static_hard_percentile_sq_mean - static_hard_percentile_mean**2, 0.0
        ) ** 0.5
        static_hard_rank_signal_mean = static_hard_rank_signal_sum / max(
            static_hard_natural_count, 1
        )
        static_hard_rank_signal_sq_mean = static_hard_rank_signal_sq_sum / max(
            static_hard_natural_count, 1
        )
        static_hard_rank_signal_std = max(
            static_hard_rank_signal_sq_mean - static_hard_rank_signal_mean**2, 0.0
        ) ** 0.5
        static_hard_natural_mean = static_hard_natural_sum / max(
            static_hard_natural_count, 1
        )
        static_hard_natural_sq_mean = static_hard_natural_sq_sum / max(
            static_hard_natural_count, 1
        )
        static_hard_natural_std = max(
            static_hard_natural_sq_mean - static_hard_natural_mean**2, 0.0
        ) ** 0.5
        static_hard_low_frac = static_hard_low_count / max(static_hard_natural_count, 1)
        static_hard_high_frac = static_hard_high_count / max(static_hard_natural_count, 1)
        effective_synthetic_weight_share = synthetic_pair_weight_sum / max(
            pair_weight_sum, 1.0e-6
        )
        pair_train_elapsed = time.perf_counter() - pair_train_start_time

        # Compute chunk rewards for ALL chunks of ALL trajectories (Phase D prep)
        self.prm_model.eval()
        all_chunk_rewards = []
        reward_start_time = time.perf_counter()
        last_reward_log_time = reward_start_time
        reward_log_every = max(1, batch_size // 4)
        with torch.no_grad():
            for batch_start in range(0, batch_size, prm_energy_batch_size):
                batch_end = min(batch_start + prm_energy_batch_size, batch_size)
                batch_indices = list(range(batch_start, batch_end))
                E_phi_batch = _compute_traj_energy_batch_safe(
                    self.prm_model, batch_indices, full_data
                )
                E_ref_batch = E_ref_all[batch_start:batch_end]
                r_batch = -(prm_beta / 2.0) * (
                    E_phi_batch - E_ref_batch
                ).sum(dim=2)  # [B_traj, n_chunk_steps]
                for local_idx, b in enumerate(batch_indices):
                    r_b = r_batch[local_idx]
                    weights_b = _get_chunk_weights(b)
                    r_b = r_b * weights_b
                    if chunk_loss_mask is not None:
                        r_b = torch.where(
                            chunk_loss_mask[:, b], r_b, torch.zeros_like(r_b)
                        )
                    all_chunk_rewards.append(r_b)
                current = batch_end
                now = time.perf_counter()
                if self._should_log_progress(
                    current,
                    batch_size,
                    last_reward_log_time,
                    now,
                    every_n=reward_log_every,
                    every_s=log_interval_s,
                ):
                    elapsed = now - reward_start_time
                    self._log_progress(
                        "[train][prm][reward] "
                        f"{current}/{batch_size} traj "
                        f"elapsed={elapsed:.1f}s avg_per_traj={elapsed / max(current, 1):.2f}s"
                    )
                    last_reward_log_time = now
        # [batch_size, n_chunk_steps] -> [n_chunk_steps, batch_size] to match rollout_batch convention
        chunk_rewards = torch.stack(all_chunk_rewards, dim=0).T  # [n_chunk_steps, batch_size]
        self.rollout_batch["chunk_rewards"] = chunk_rewards.cpu()
        reward_elapsed = time.perf_counter() - reward_start_time
        prm_total_elapsed = time.perf_counter() - prm_start_time

        def _masked_reward_stats(
            values: torch.Tensor, mask: torch.Tensor | None = None
        ) -> tuple[float, float, float]:
            if mask is not None:
                values = values[mask]
            else:
                values = values.reshape(-1)
            if values.numel() == 0:
                return 0.0, 0.0, 0.0
            mean = float(values.mean().item())
            median = float(values.median().item())
            std = float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0
            return mean, median, std

        if chunk_loss_mask is not None and chunk_loss_mask.any():
            r_chunk_mean = chunk_rewards[chunk_loss_mask].mean().item()
        else:
            r_chunk_mean = chunk_rewards.mean().item()
        r_chunk_median, r_chunk_std = 0.0, 0.0
        if chunk_loss_mask is not None and chunk_loss_mask.any():
            _, r_chunk_median, r_chunk_std = _masked_reward_stats(
                chunk_rewards, chunk_loss_mask
            )
        else:
            _, r_chunk_median, r_chunk_std = _masked_reward_stats(chunk_rewards)

        traj_labels = None
        advantages = self.rollout_batch.get("advantages", None)
        if isinstance(advantages, torch.Tensor):
            traj_adv = advantages.to(device)
            while traj_adv.ndim > 2 and traj_adv.shape[-1] == 1:
                traj_adv = traj_adv.squeeze(-1)
            if traj_adv.ndim == 1 and traj_adv.shape[0] == batch_size:
                traj_labels = traj_adv.sign()
            elif traj_adv.ndim == 2:
                if traj_adv.shape[1] == batch_size:
                    pass
                elif traj_adv.shape[0] == batch_size:
                    traj_adv = traj_adv.T
                else:
                    traj_adv = None
                if traj_adv is not None:
                    pos_any = (traj_adv > 0).any(dim=0)
                    neg_any = (traj_adv < 0).any(dim=0)
                    traj_labels = torch.zeros(
                        batch_size, device=device, dtype=chunk_rewards.dtype
                    )
                    traj_labels = torch.where(
                        pos_any,
                        torch.ones_like(traj_labels),
                        traj_labels,
                    )
                    traj_labels = torch.where(
                        neg_any & ~pos_any,
                        -torch.ones_like(traj_labels),
                        traj_labels,
                    )

        valid_chunks_per_traj = (
            chunk_loss_mask.sum(dim=0).float()
            if chunk_loss_mask is not None
            else torch.full(
                (batch_size,),
                float(n_chunk_steps),
                device=device,
                dtype=chunk_rewards.dtype,
            )
        )
        effective_chunks_per_traj = (
            chunk_valid_fraction.sum(dim=0)
            if chunk_valid_fraction is not None
            else valid_chunks_per_traj.clone()
        )
        valid_chunks_per_traj_mean = float(valid_chunks_per_traj.mean().item())
        effective_chunks_per_traj_mean = float(effective_chunks_per_traj.mean().item())
        valid_chunks_success_mean = 0.0
        valid_chunks_fail_mean = 0.0
        effective_chunks_success_mean = 0.0
        effective_chunks_fail_mean = 0.0
        r_chunk_success_mean = 0.0
        r_chunk_fail_mean = 0.0
        r_chunk_success_median = 0.0
        r_chunk_fail_median = 0.0
        r_chunk_success_std = 0.0
        r_chunk_fail_std = 0.0
        chunk_valid_fraction_mean = (
            float(chunk_valid_fraction[chunk_valid_fraction > 0].mean().item())
            if chunk_valid_fraction is not None and (chunk_valid_fraction > 0).any()
            else 0.0
        )
        partial_chunk_frac = (
            float(chunk_partial_mask[chunk_loss_mask].float().mean().item())
            if chunk_partial_mask is not None
            and chunk_loss_mask is not None
            and chunk_loss_mask.any()
            else 0.0
        )
        partial_chunk_success_frac = 0.0
        partial_chunk_fail_frac = 0.0
        if traj_labels is not None:
            success_traj_mask = traj_labels > 0
            fail_traj_mask = traj_labels < 0
            if success_traj_mask.any():
                valid_chunks_success_mean = float(
                    valid_chunks_per_traj[success_traj_mask].mean().item()
                )
                effective_chunks_success_mean = float(
                    effective_chunks_per_traj[success_traj_mask].mean().item()
                )
                success_chunk_mask = success_traj_mask.unsqueeze(0).expand_as(chunk_rewards)
                if chunk_loss_mask is not None:
                    success_chunk_mask = success_chunk_mask & chunk_loss_mask
                (
                    r_chunk_success_mean,
                    r_chunk_success_median,
                    r_chunk_success_std,
                ) = _masked_reward_stats(chunk_rewards, success_chunk_mask)
                if chunk_partial_mask is not None and success_chunk_mask.any():
                    partial_chunk_success_frac = float(
                        chunk_partial_mask[success_chunk_mask].float().mean().item()
                    )
            if fail_traj_mask.any():
                valid_chunks_fail_mean = float(
                    valid_chunks_per_traj[fail_traj_mask].mean().item()
                )
                effective_chunks_fail_mean = float(
                    effective_chunks_per_traj[fail_traj_mask].mean().item()
                )
                fail_chunk_mask = fail_traj_mask.unsqueeze(0).expand_as(chunk_rewards)
                if chunk_loss_mask is not None:
                    fail_chunk_mask = fail_chunk_mask & chunk_loss_mask
                (
                    r_chunk_fail_mean,
                    r_chunk_fail_median,
                    r_chunk_fail_std,
                ) = _masked_reward_stats(chunk_rewards, fail_chunk_mask)
                if chunk_partial_mask is not None and fail_chunk_mask.any():
                    partial_chunk_fail_frac = float(
                        chunk_partial_mask[fail_chunk_mask].float().mean().item()
                    )

        self._log_progress(
            "[train][prm] done "
            f"total={prm_total_elapsed:.1f}s ref={ref_elapsed:.1f}s "
            f"select={pair_selection_elapsed:.1f}s dpo={pair_train_elapsed:.1f}s "
            f"reward={reward_elapsed:.1f}s effective_pairs={total_pairs} "
            f"acc={accuracy:.4f} weight_mean={pair_weight_mean:.3f} "
            f"static_raw={static_hard_raw_mean:.3f}->cal={static_hard_natural_mean:.3f} "
            f"oom_splits={prm_energy_split_fallback_count} "
            f"traj_chunk_splits={prm_traj_chunk_split_fallback_count} "
            f"synth_w_share={effective_synthetic_weight_share:.3f} "
            f"loss={avg_loss:.4f}/{avg_unweighted_loss:.4f}"
        )

        return {
            "prm/loss": avg_loss,
            "prm/unweighted_loss": avg_unweighted_loss,
            "prm/accuracy": accuracy,
            "prm/num_pairs": total_pairs,
            "prm/energy_batch_size": float(prm_energy_batch_size),
            "prm/pair_weight_mean": pair_weight_mean,
            "prm/pair_weight_std": pair_weight_std,
            "prm/pair_weight_min": pair_weight_min if pair_weight_min is not None else 0.0,
            "prm/pair_weight_max": pair_weight_max if pair_weight_max is not None else 0.0,
            "prm/pair_weight_low_frac": pair_weight_low_frac,
            "prm/pair_weight_le_0p2_frac": pair_weight_le_0p2_frac,
            "prm/static_hard_mean": static_hard_mean,
            "prm/static_hard_natural_mean": static_hard_natural_mean,
            "prm/static_hard_natural_std": static_hard_natural_std,
            "prm/static_hard_raw_mean": static_hard_raw_mean,
            "prm/static_hard_raw_std": static_hard_raw_std,
            "prm/static_hard_raw_min": (
                static_hard_raw_min if static_hard_raw_min is not None else 0.0
            ),
            "prm/static_hard_raw_max": (
                static_hard_raw_max if static_hard_raw_max is not None else 0.0
            ),
            "prm/static_hard_percentile_mean": static_hard_percentile_mean,
            "prm/static_hard_percentile_std": static_hard_percentile_std,
            "prm/static_hard_rank_signal_mean": static_hard_rank_signal_mean,
            "prm/static_hard_rank_signal_std": static_hard_rank_signal_std,
            "prm/static_hard_low_frac": static_hard_low_frac,
            "prm/static_hard_high_frac": static_hard_high_frac,
            "prm/margin_weight_mean": margin_weight_mean,
            "prm/natural_pair_weight_mean": natural_pair_weight_mean,
            "prm/synthetic_pair_weight_mean": synthetic_pair_weight_mean,
            "prm/effective_synthetic_weight_share": effective_synthetic_weight_share,
            "prm/candidate_pairs": candidate_pairs_total,
            "prm/selected_pair_frac": pair_selection_frac,
            "prm/hard_negative_k_effective": hard_negative_k_effective,
            "prm/hard_shortlist_k_effective": shortlist_size_mean,
            "prm/hard_shortlist_sample_frac": shortlist_sample_frac,
            "prm/hard_backfill_sample_frac": backfill_sample_frac,
            "prm/group_selected_fail_mean": group_selected_fail_mean,
            "prm/group_selected_fail_reuse_mean": fail_reuse_mean,
            "prm/group_selected_fail_reuse_max": fail_reuse_max,
            "prm/reuse_cap_limited_positive_frac": reuse_cap_limited_positive_frac,
            "prm/reuse_cap_limited_group_frac": reuse_cap_limited_group_frac,
            "prm/all_fail_similarity_mean": all_fail_similarity_mean,
            "prm/all_fail_similarity_std": all_fail_similarity_std,
            "prm/all_fail_similarity_min": (
                all_pair_similarity_min if all_pair_similarity_min is not None else 0.0
            ),
            "prm/all_fail_similarity_max": (
                all_pair_similarity_max if all_pair_similarity_max is not None else 0.0
            ),
            "prm/hard_neg_similarity_mean": hard_neg_similarity_mean,
            "prm/hard_neg_similarity_std": hard_neg_similarity_std,
            "prm/hard_neg_similarity_min": (
                selected_pair_similarity_min
                if selected_pair_similarity_min is not None
                else 0.0
            ),
            "prm/hard_neg_similarity_max": (
                selected_pair_similarity_max
                if selected_pair_similarity_max is not None
                else 0.0
            ),
            "prm/hard_over_all_similarity_gap": (
                hard_neg_similarity_mean - all_fail_similarity_mean
            ),
            "prm/candidate_fails_per_positive_mean": candidate_fails_per_positive_mean,
            "prm/action_prior_available": action_prior_available,
            "prm/state_prior_available": state_prior_available,
            "prm/semantic_score_all_mean": semantic_score_all_mean,
            "prm/semantic_score_selected_mean": semantic_score_selected_mean,
            "prm/semantic_tail_all_mean": semantic_tail_all_mean,
            "prm/semantic_tail_selected_mean": semantic_tail_selected_mean,
            "prm/semantic_early_penalty_all_mean": semantic_early_penalty_all_mean,
            "prm/semantic_early_penalty_selected_mean": semantic_early_penalty_selected_mean,
            "prm/semantic_midband_all_mean": semantic_midband_all_mean,
            "prm/semantic_midband_selected_mean": semantic_midband_selected_mean,
            "prm/quality_gate_pass_frac": quality_gate_pass_frac,
            "prm/quality_score_mean": quality_score_mean,
            "prm/quality_score_selected_mean": quality_score_selected_mean,
            "prm/semantic_head_pair_frac": semantic_head_pair_frac,
            "prm/quality_head_pair_frac": quality_head_pair_frac,
            "prm/fallback_head_pair_frac": fallback_head_pair_frac,
            "prm/action_path_excess_all_mean": feature_excess_all_mean["action_path"],
            "prm/action_reversal_excess_all_mean": feature_excess_all_mean["action_reversal"],
            "prm/action_gripper_excess_all_mean": feature_excess_all_mean["action_gripper"],
            "prm/action_continuity_excess_all_mean": feature_excess_all_mean["action_continuity"],
            "prm/state_path_excess_all_mean": feature_excess_all_mean["state_path"],
            "prm/state_reversal_excess_all_mean": feature_excess_all_mean["state_reversal"],
            "prm/state_gripper_excess_all_mean": feature_excess_all_mean["state_gripper"],
            "prm/action_path_excess_selected_mean": feature_excess_selected_mean["action_path"],
            "prm/action_reversal_excess_selected_mean": feature_excess_selected_mean["action_reversal"],
            "prm/action_gripper_excess_selected_mean": feature_excess_selected_mean["action_gripper"],
            "prm/action_continuity_excess_selected_mean": feature_excess_selected_mean["action_continuity"],
            "prm/state_path_excess_selected_mean": feature_excess_selected_mean["state_path"],
            "prm/state_reversal_excess_selected_mean": feature_excess_selected_mean["state_reversal"],
            "prm/state_gripper_excess_selected_mean": feature_excess_selected_mean["state_gripper"],
            "prm/group_count": self.rollout_batch.get("preference_group_count", 0),
            "prm/groups_total": self.rollout_batch.get("preference_groups_total", 0),
            "prm/groups_with_pairs": self.rollout_batch.get(
                "preference_groups_with_pairs", 0
            ),
            "prm/groups_no_success": self.rollout_batch.get(
                "preference_groups_no_success", 0
            ),
            "prm/groups_no_failure": self.rollout_batch.get(
                "preference_groups_no_failure", 0
            ),
            "prm/group_success_mean": self.rollout_batch.get(
                "preference_group_success_mean", 0.0
            ),
            "prm/group_success_min": self.rollout_batch.get(
                "preference_group_success_min", 0
            ),
            "prm/group_success_max": self.rollout_batch.get(
                "preference_group_success_max", 0
            ),
            "prm/group_failure_mean": self.rollout_batch.get(
                "preference_group_failure_mean", 0.0
            ),
            "prm/group_failure_min": self.rollout_batch.get(
                "preference_group_failure_min", 0
            ),
            "prm/group_failure_max": self.rollout_batch.get(
                "preference_group_failure_max", 0
            ),
            "prm/group_topk_mean": self.rollout_batch.get(
                "preference_group_topk_mean", 0.0
            ),
            "prm/group_topk_min": self.rollout_batch.get(
                "preference_group_topk_min", 0
            ),
            "prm/group_topk_max": self.rollout_batch.get(
                "preference_group_topk_max", 0
            ),
            "prm/group_topk_eq_1_frac": self.rollout_batch.get(
                "preference_group_topk_eq_1_frac", 0.0
            ),
            "prm/group_topk_eq_all_success_frac": self.rollout_batch.get(
                "preference_group_topk_eq_all_success_frac", 0.0
            ),
            "prm/group_pairs_mean": self.rollout_batch.get(
                "preference_group_pairs_mean", 0.0
            ),
            "prm/group_pairs_max": self.rollout_batch.get(
                "preference_group_pairs_max", 0
            ),
            "prm/r_chunk_mean": r_chunk_mean,
            "prm/r_chunk_median": r_chunk_median,
            "prm/r_chunk_std": r_chunk_std,
            "prm/r_chunk_success_mean": r_chunk_success_mean,
            "prm/r_chunk_fail_mean": r_chunk_fail_mean,
            "prm/r_chunk_success_median": r_chunk_success_median,
            "prm/r_chunk_fail_median": r_chunk_fail_median,
            "prm/r_chunk_success_std": r_chunk_success_std,
            "prm/r_chunk_fail_std": r_chunk_fail_std,
            "prm/pair_margin_mean": avg_pair_margin,
            "prm/pair_margin_per_chunk_mean": avg_pair_margin_per_chunk,
            "prm/pair_margin_std": pair_margin_std,
            "prm/pair_margin_min": pair_margin_min if pair_margin_min is not None else 0.0,
            "prm/pair_margin_max": pair_margin_max if pair_margin_max is not None else 0.0,
            "prm/pair_margin_pos_frac": accuracy,
            "prm/pair_margin_gt_5_frac": pair_margin_gt_5_frac,
            "prm/pair_margin_gt_10_frac": pair_margin_gt_10_frac,
            "prm/pair_batch_fallback_count": float(prm_pair_batch_fallback_count),
            "prm/energy_split_fallback_count": float(prm_energy_split_fallback_count),
            "prm/energy_split_max_batch": float(prm_energy_split_max_batch),
            "prm/traj_chunk_split_fallback_count": float(
                prm_traj_chunk_split_fallback_count
            ),
            "prm/traj_chunk_split_max_block": float(prm_traj_chunk_split_max_block),
            "prm/saturated_pair_frac": saturated_pair_frac,
            "prm/score_plus_mean": score_plus_mean,
            "prm/score_minus_mean": score_minus_mean,
            "prm/synthetic_pair_count": synthetic_pair_count,
            "prm/synthetic_pair_frac": synthetic_pair_frac,
            "prm/synthetic_loss_mean": synthetic_loss_mean,
            "prm/synthetic_unweighted_loss_mean": synthetic_unweighted_loss_mean,
            "prm/synthetic_margin_mean": synthetic_margin_mean,
            "prm/shared_budget_mean": shared_budget_mean,
            "prm/shared_budget_std": shared_budget_std,
            "prm/minus_retained_frac_mean": minus_retained_frac_mean,
            "prm/valid_chunks_plus_mean": valid_chunks_plus_mean,
            "prm/valid_chunks_minus_mean": valid_chunks_minus_mean,
            "prm/valid_chunks_per_traj_mean": valid_chunks_per_traj_mean,
            "prm/effective_chunks_per_traj_mean": effective_chunks_per_traj_mean,
            "prm/valid_chunks_success_mean": valid_chunks_success_mean,
            "prm/valid_chunks_fail_mean": valid_chunks_fail_mean,
            "prm/effective_chunks_success_mean": effective_chunks_success_mean,
            "prm/effective_chunks_fail_mean": effective_chunks_fail_mean,
            "prm/chunk_valid_fraction_mean": chunk_valid_fraction_mean,
            "prm/partial_chunk_frac": partial_chunk_frac,
            "prm/partial_chunk_success_frac": partial_chunk_success_frac,
            "prm/partial_chunk_fail_frac": partial_chunk_fail_frac,
            "actor/prm_ref_precompute_s": ref_elapsed,
            "actor/prm_pair_select_s": pair_selection_elapsed,
            "actor/prm_pair_train_s": pair_train_elapsed,
            "actor/prm_reward_infer_s": reward_elapsed,
            "actor/prm_total_s": prm_total_elapsed,
        }

    def _compute_chunk_gates(self):
        """Phase D: compute chunk gates and y_phi from chunk rewards."""
        from rlinf.algorithms.losses import compute_chunk_gate

        chunk_rewards = self.rollout_batch.get("chunk_rewards", None)
        advantages = self.rollout_batch.get("advantages", None)
        loss_mask = self.rollout_batch.get("loss_mask", None)
        if chunk_rewards is None or advantages is None:
            return

        # Flatten to 1D for gate computation
        cr_flat = chunk_rewards.reshape(-1)

        # y_tau: trajectory-level label, flatten to match
        adv_flat = advantages.reshape(-1, *advantages.shape[2:])
        if adv_flat.ndim >= 2:
            y_tau = adv_flat[:, 0].sign()
        else:
            y_tau = adv_flat.sign()

        tau_phi = self.cfg.algorithm.get("gate_tau_phi", 1.0)
        eta_min = self.cfg.algorithm.get("gate_eta_min", 0.1)
        eta_max = self.cfg.algorithm.get("gate_eta_max", 1.0)
        valid_flat = None
        if loss_mask is not None:
            chunk_loss_mask = self._coerce_chunk_mask(
                loss_mask.to(chunk_rewards.device),
                n_chunk_steps=chunk_rewards.shape[0],
                batch_size=chunk_rewards.shape[1],
            )
            valid_flat = chunk_loss_mask.reshape(-1).bool()

        gate, y_phi = compute_chunk_gate(
            cr_flat,
            y_tau,
            valid_mask=valid_flat,
            tau_phi=tau_phi,
            eta_min=eta_min,
            eta_max=eta_max,
        )
        # Store in same shape as chunk_rewards for correct shuffle
        self.rollout_batch["chunk_gate"] = gate.reshape(chunk_rewards.shape)
        self.rollout_batch["y_phi"] = y_phi.reshape(chunk_rewards.shape)

    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        run_training_start = time.perf_counter()
        self._log_cuda_memory("train/start")
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        self._log_progress(
            "[train] run_training start "
            f"step={self.global_step} rollout_size={rollout_size}"
        )
        gate_elapsed = 0.0
        pair_precompute_metrics = {}
        shuffle_elapsed = 0.0
        actor_loop_elapsed = 0.0
        ema_elapsed = 0.0
        self._actor_fused_step_split_fallback_count = 0
        self._actor_fused_step_max_block = 0
        self._actor_dual_credit_inner_split_fallback_count = 0
        self._actor_dual_credit_inner_min_leaf = 10**9

        # --- Dual-Credit Phase C & D (before main training loop) ---
        prm_metrics = {}
        if self.cfg.algorithm.loss_type == "nft-dual-credit" and self.prm_model is not None:
            num_steps = self.cfg.actor.model.num_steps
            schedule = torch.linspace(1, 0, num_steps + 1, device=self.device)
            prm_metrics = self._run_prm_training(schedule)
            gate_start_time = time.perf_counter()
            self._compute_chunk_gates()
            gate_elapsed = time.perf_counter() - gate_start_time
            self._log_progress(
                "[train][gate] done "
                f"step={self.global_step} elapsed={gate_elapsed:.1f}s"
            )
            pair_precompute_metrics = self._run_pair_guidance_precompute(schedule)
            self._offload_aux_models_for_actor_loop()
            gc.collect()
            torch.cuda.empty_cache()
            self._log_cuda_memory("train/before_actor_loop_after_aux")

        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        shuffle_start_time = time.perf_counter()
        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )
        shuffle_elapsed = time.perf_counter() - shuffle_start_time
        self._log_progress(
            "[train][actor] rollout batch reshaped "
            f"elapsed={shuffle_elapsed:.1f}s"
        )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        global_batches_per_epoch = rollout_size // batch_size_per_rank
        total_global_batches = update_epoch * global_batches_per_epoch
        global_batch_count = 0
        actor_log_interval_s = float(
            self.cfg.algorithm.get("train_progress_log_interval_s", 120.0)
        )
        empty_cache_every_n_batches = int(
            self.cfg.algorithm.get("empty_cache_every_n_batches", 0)
        )
        actor_batch_log_every = max(1, total_global_batches // 6)
        last_actor_log_time = time.perf_counter()
        actor_loop_start_time = time.perf_counter()
        self._log_progress(
            "[train][actor] loop start "
            f"update_epoch={update_epoch} global_batches_per_epoch={global_batches_per_epoch} "
            f"gradient_accumulation={self.gradient_accumulation} "
            f"empty_cache_every_n_batches={empty_cache_every_n_batches}"
        )
        for epoch_idx in range(update_epoch):
            epoch_start_time = time.perf_counter()
            self._log_progress(
                "[train][actor] epoch start "
                f"{epoch_idx + 1}/{update_epoch}"
            )
            rollout_dataloader_iter = get_iterator_k_split(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for global_batch_idx, train_global_batch in enumerate(
                rollout_dataloader_iter, start=1
            ):
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = get_iterator_k_split(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad(set_to_none=True)
                for idx, data in enumerate(train_micro_batch):
                    data = put_tensor_device(
                        data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                    )
                    if self.cfg.algorithm.loss_type.startswith("nft"):
                        num_steps = self.cfg.actor.model.num_steps
                        x_t_input = data.get("nft_xt", None)
                        schedule = torch.linspace(
                            1,
                            0,
                            num_steps + 1,
                            device=self.device,
                            dtype=x_t_input.dtype,
                        )
                        compute_values = self.cfg.actor.model.get(
                            "add_value_head", False
                        )
                        if self.cfg.algorithm.loss_type == "nft-dual-credit":
                            metrics_data = self._run_dual_credit_subbatch_safe(
                                data,
                                schedule,
                                compute_values=compute_values,
                                sync_on_last_leaf=(idx + 1)
                                == self.gradient_accumulation,
                            )
                        else:
                            v_old = data.get("nft_v", None)
                            x_next_input = data.get("nft_xnext", None)
                            step_indices = data.get("nft_step_index", None)
                            noise_level_for_loss = data.get("nft_noise_level", None)
                            t = schedule[step_indices.long()]

                            with self.amp_context:
                                output_dict = self.model(
                                    data=data,
                                    use_nft_loss=True,
                                    compute_values=compute_values,
                                    compute_noise_stats=True,
                                    nft_explicit_inputs={"x_t": x_t_input, "timesteps": t},
                                    use_cache=False,
                                    shared_cache=None,
                                )

                            v_theta = output_dict["v_theta"]
                            values = output_dict.get("values", None)
                            chunk_size = v_theta.shape[1]
                            x_t_loss = x_t_input[:, :chunk_size, :]
                            x_next_loss = x_next_input[:, :chunk_size, :]

                            prev_values = data.get("prev_values", None)
                            if prev_values is not None and prev_values.dim() > 1:
                                prev_values = prev_values[:, :1]
                            returns = data.get("returns", None)
                            if returns is not None and returns.dim() > 1:
                                returns = returns[:, :1]

                            if self.cfg.algorithm.adv_type == "terminal-binary":
                                self._maybe_log_terminal_binary_loss_inputs(
                                    advantages=data["advantages"],
                                    returns=returns,
                                    loss_mask=data.get("loss_mask", None),
                                    adv_clip_max=self.cfg.algorithm.get(
                                        "clip_ratio_high", 5.0
                                    ),
                                )

                            kwargs = {
                                "loss_type": "nft-actor-critic"
                                if compute_values
                                else "nft-actor",
                                "task_type": self.cfg.runner.task_type,
                                "v_theta": v_theta,
                                "v_old": v_old,
                                "x_t": x_t_loss,
                                "x_next": x_next_loss,
                                "schedule": schedule,
                                "step_indices": step_indices,
                                "total_denoise_steps": num_steps,
                                "noise_level": noise_level_for_loss,
                                "advantages": data["advantages"],
                                "loss_mask": data.get("loss_mask", None),
                                "loss_mask_sum": data.get("loss_mask_sum", None),
                                "time_decay_weights": data.get("time_decay_weights", None),
                                "beta": self.cfg.algorithm.get("nft_beta", 1.0),
                                "kl_beta": self.cfg.algorithm.get("kl_beta", 0.0),
                                "adv_clip_max": self.cfg.algorithm.get(
                                    "clip_ratio_high", 1.0
                                ),
                                "task_ids": data.get("task_ids", None),
                                "values": values,
                                "returns": returns,
                                "prev_values": prev_values,
                                "value_clip": self.cfg.algorithm.get("value_clip", None),
                                "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                                "max_episode_steps": self.cfg.env.train.max_episode_steps,
                                "critic_warmup": self._is_in_critic_warmup(),
                            }
                            loss, metrics_data = policy_loss(**kwargs)

                            raw_loss = loss.detach()
                            loss /= self.gradient_accumulation
                            backward_ctx = self.before_micro_batch(
                                self.model,
                                is_last_micro_batch=(idx + 1)
                                == self.gradient_accumulation,
                            )
                            with backward_ctx:
                                self.grad_scaler.scale(loss).backward()

                            total_loss_for_log = metrics_data.get(
                                "actor/total_loss", raw_loss
                            )
                            if torch.is_tensor(total_loss_for_log):
                                total_loss_for_log = total_loss_for_log.detach().item()
                            metrics_data["loss"] = float(total_loss_for_log)
                            metrics_data["loss_scaled"] = loss.detach().item()
                    else:
                        advantages = data["advantages"]
                        prev_logprobs = data["prev_logprobs"]
                        returns = data.get("returns", None)
                        prev_values = data.get("prev_values", None)
                        loss_mask = data.get("loss_mask", None)
                        loss_mask_sum = data.get("loss_mask_sum", None)

                        if SupportedModel(self.cfg.actor.model.model_type) in [
                            SupportedModel.OPENVLA,
                            SupportedModel.OPENVLA_OFT,
                        ]:
                            data["temperature"] = (
                                self.cfg.algorithm.sampling_params.temperature_train
                            )
                            data["top_k"] = self.cfg.algorithm.sampling_params.top_k

                        compute_values = (
                            True if self.cfg.algorithm.adv_type == "gae" else False
                        )

                        with self.amp_context:
                            output_dict = self.model(
                                data=data,
                                compute_logprobs=True,
                                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                                compute_values=compute_values,
                                use_cache=False,
                            )

                        if SupportedModel(self.cfg.actor.model.model_type) in [
                            SupportedModel.GR00T
                        ]:
                            prev_logprobs = output_dict["prev_logprobs"]

                        if self.cfg.algorithm.adv_type == "terminal-binary":
                            self._maybe_log_terminal_binary_loss_inputs(
                                advantages=advantages,
                                returns=returns,
                                loss_mask=loss_mask,
                                adv_clip_max=self.cfg.algorithm.clip_ratio_high,
                            )

                        kwargs = {
                            "loss_type": self.cfg.algorithm.loss_type,
                            "logprob_type": self.cfg.algorithm.logprob_type,
                            "reward_type": self.cfg.algorithm.reward_type,
                            "single_action_dim": self.cfg.actor.model.get(
                                "action_dim", 7
                            ),
                            "logprobs": output_dict["logprobs"],
                            "values": output_dict.get("values", None),
                            "old_logprobs": prev_logprobs,
                            "advantages": advantages,
                            "returns": returns,
                            "prev_values": prev_values,
                            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                            "value_clip": self.cfg.algorithm.get("value_clip", None),
                            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                            "loss_mask": loss_mask,
                            "loss_mask_sum": loss_mask_sum,
                            "max_episode_steps": self.cfg.env.train.max_episode_steps,
                            "task_type": self.cfg.runner.task_type,
                            "critic_warmup": self._is_in_critic_warmup(),
                        }
                        loss, metrics_data = policy_loss(**kwargs)

                        entropy_loss = torch.tensor(
                            0.0, device=torch.cuda.current_device()
                        )
                        if (
                            self.cfg.algorithm.entropy_bonus > 0
                            and not kwargs["critic_warmup"]
                        ):
                            entropy = output_dict["entropy"]
                            entropy = reshape_entropy(
                                entropy,
                                entropy_type=self.cfg.algorithm.entropy_type,
                                action_dim=self.cfg.actor.model.get("action_dim", 7),
                                batch_size=output_dict["logprobs"].shape[0],
                            )
                            entropy_loss = masked_mean(entropy, mask=loss_mask)
                            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                        metrics_data["entropy_loss"] = entropy_loss.detach().item()

                        backward_ctx = self.before_micro_batch(
                            self.model,
                            is_last_micro_batch=(idx + 1)
                            == self.gradient_accumulation,
                        )
                        raw_loss = loss.detach()
                        loss /= self.gradient_accumulation
                        with backward_ctx:
                            self.grad_scaler.scale(loss).backward()

                        total_loss_for_log = metrics_data.get("actor/total_loss", raw_loss)
                        if torch.is_tensor(total_loss_for_log):
                            total_loss_for_log = total_loss_for_log.detach().item()
                        metrics_data["loss"] = float(total_loss_for_log)
                        metrics_data["loss_scaled"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                if (
                    empty_cache_every_n_batches > 0
                    and (global_batch_count + 1) % empty_cache_every_n_batches == 0
                ):
                    torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                self._update_ready = True
                self._log_cuda_memory("train/after_optimizer_step")
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                    "actor/empty_cache_every_n_batches": float(
                        empty_cache_every_n_batches
                    ),
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
                global_batch_count += 1
                now = time.perf_counter()
                if self._should_log_progress(
                    global_batch_count,
                    total_global_batches,
                    last_actor_log_time,
                    now,
                    every_n=actor_batch_log_every,
                    every_s=actor_log_interval_s,
                ):
                    elapsed = now - actor_loop_start_time
                    self._log_progress(
                        "[train][actor] progress "
                        f"global_batch={global_batch_count}/{total_global_batches} "
                        f"epoch={epoch_idx + 1}/{update_epoch} "
                        f"epoch_batch={global_batch_idx}/{global_batches_per_epoch} "
                        f"elapsed={elapsed:.1f}s avg_per_global_batch={elapsed / max(global_batch_count, 1):.1f}s"
                    )
                    last_actor_log_time = now
            epoch_elapsed = time.perf_counter() - epoch_start_time
            self._log_progress(
                "[train][actor] epoch done "
                f"{epoch_idx + 1}/{update_epoch} elapsed={epoch_elapsed:.1f}s"
            )
        actor_loop_elapsed = time.perf_counter() - actor_loop_start_time
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        self._log_cuda_memory("train/after_lr_step")

        decay = 1.0
        ema_start_time = time.perf_counter()
        if (
            self.cfg.algorithm.loss_type.startswith("nft")
            and self.ref_model is not None
        ):
            with torch.no_grad():
                if self.prev_rollout_model is not None:
                    self._copy_snapshot_state(
                        self.prev_rollout_model,
                        self.ref_model,
                        self._shared_prev_rollout_param_names,
                        tag="snapshot/prev_from_ref",
                    )
                total_steps = self.cfg.algorithm.get(
                    "decay_epochs",
                    self.cfg.runner.get("max_epochs", self.cfg.runner.get("max_steps", 0)),
                )
                base_decay = self.cfg.algorithm.get("base", 0.1)
                target_decay = self.cfg.algorithm.get("target", 0.8)
                decay = nft_return_decay(
                    self.global_step,
                    total_steps,
                    base=base_decay,
                    target=target_decay,
                )

                if decay < 1.0:
                    alpha = 1.0 - decay
                    try:
                        student_sd = self.get_model_state_dict(
                            cpu_offload=False, full_state_dict=True
                        )
                    except AssertionError as exc:
                        if self._rank == 0:
                            self.log_info(
                                "[EMA] state_dict assertion, falling back to named_parameters. "
                                f"error={exc}"
                            )

                        def _normalize_name(name: str) -> str:
                            if name.startswith("_fsdp_wrapped_module."):
                                name = name[len("_fsdp_wrapped_module.") :]
                            name = name.replace("._fsdp_wrapped_module.", ".")
                            name = name.replace("._fsdp_wrapped_module", "")
                            return name

                        student_sd = {}
                        for name, param in self.model.named_parameters():
                            norm = _normalize_name(name)
                            student_sd.setdefault(norm, param.detach())
                            if norm.startswith("model."):
                                student_sd.setdefault(norm[len("model.") :], param.detach())
                            else:
                                student_sd.setdefault(f"model.{norm}", param.detach())

                    shape_mismatch_cnt = 0
                    updated = 0
                    mismatch_samples = []
                    for name, tgt in self.ref_model.named_parameters():
                        if (
                            name in self._shared_ref_param_names
                            or "value_head" in name
                            or "paligemma." in name
                            or "u_encoder" in name
                        ):
                            continue
                        src = student_sd.get(name, None)
                        if src is None:
                            continue
                        if src.shape != tgt.shape:
                            shape_mismatch_cnt += 1
                            if len(mismatch_samples) < 5:
                                mismatch_samples.append(
                                    f"{name}: src={tuple(src.shape)} tgt={tuple(tgt.shape)}"
                                )
                            continue
                        tgt.data.lerp_(src.to(tgt.device), alpha)
                        updated += 1

                    try:
                        student_ptrs = {p.data_ptr() for p in self.model.parameters()}
                        leaked_shared = [
                            n
                            for n, p in self.ref_model.named_parameters()
                            if p.data_ptr() in student_ptrs
                            and n not in self._shared_ref_param_names
                        ]
                    except Exception:
                        leaked_shared = []
                    if updated > 0:
                        print(
                            f"[EMA] updated {updated} policy parameters with decay={decay:.4f}",
                            flush=True,
                        )
                    else:
                        print(
                            "[EMA] no policy parameter was updated in this step "
                            f"(decay={decay:.4f}, optimizer_steps={self.optimizer_steps}, "
                            f"shape_mismatch={shape_mismatch_cnt})",
                            flush=True,
                        )
                    if shape_mismatch_cnt > 0 and mismatch_samples:
                        print(
                            "[EMA] shape mismatch samples: " + "; ".join(mismatch_samples),
                            flush=True,
                        )
                    if leaked_shared:
                        print(
                            "[EMA] shared params not tracked (sample): "
                            + ", ".join(leaked_shared[:5]),
                            flush=True,
                        )
        ema_elapsed = time.perf_counter() - ema_start_time


        clear_memory()

        def _to_scalar_list(val):
            if isinstance(val, list):
                items = val
            else:
                items = [val]
            scalars = []
            for x in items:
                if torch.is_tensor(x):
                    scalars.append(float(x.detach().mean().item()))
                else:
                    scalars.append(float(np.mean(x)))
            return scalars

        mean_metric_dict = {}
        for key, value in metrics.items():
            scalars = _to_scalar_list(value)
            mean_metric_dict[key] = (
                float(np.mean(scalars)) if len(scalars) > 0 else 0.0
            )
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        stage_timing_metrics = {
            "actor/gate_s": gate_elapsed,
            "actor/rollout_reformat_s": shuffle_elapsed,
            "actor/main_loop_s": actor_loop_elapsed,
            "actor/ema_s": ema_elapsed,
            "actor/total_training_s": time.perf_counter() - run_training_start,
            "actor/num_global_batches": float(total_global_batches),
            "actor/gradient_accumulation": float(self.gradient_accumulation),
            "actor/fused_step_split_fallback_count": float(
                self._actor_fused_step_split_fallback_count
            ),
            "actor/fused_step_max_block": float(self._actor_fused_step_max_block),
            "actor/dual_credit_inner_split_fallback_count": float(
                self._actor_dual_credit_inner_split_fallback_count
            ),
            "actor/dual_credit_inner_min_leaf": float(
                0
                if self._actor_dual_credit_inner_min_leaf == 10**9
                else self._actor_dual_credit_inner_min_leaf
            ),
            "actor/dual_credit_inner_batch_cap": float(
                self.cfg.algorithm.get("dual_credit_inner_batch_cap", 8)
            ),
        }
        stage_timing_metrics = all_reduce_dict(
            stage_timing_metrics, op=torch.distributed.ReduceOp.MAX
        )
        mean_metric_dict.update(stage_timing_metrics)
        if self.cfg.algorithm.loss_type.startswith("nft"):
            mean_metric_dict["actor/nft_decay"] = decay
        if prm_metrics:
            prm_stage_timing_metrics = {
                k: v for k, v in prm_metrics.items() if k.startswith("actor/")
            }
            prm_scalar_metrics = {
                k: v for k, v in prm_metrics.items() if not k.startswith("actor/")
            }
            if prm_stage_timing_metrics:
                prm_stage_timing_metrics = all_reduce_dict(
                    prm_stage_timing_metrics, op=torch.distributed.ReduceOp.MAX
                )
                mean_metric_dict.update(prm_stage_timing_metrics)
            if prm_scalar_metrics:
                prm_scalar_metrics = all_reduce_dict(
                    prm_scalar_metrics, op=torch.distributed.ReduceOp.AVG
                )
                mean_metric_dict.update(prm_scalar_metrics)
        if pair_precompute_metrics:
            pair_precompute_metrics = all_reduce_dict(
                pair_precompute_metrics, op=torch.distributed.ReduceOp.AVG
            )
            mean_metric_dict.update(pair_precompute_metrics)
        if self._last_branch_metrics:
            branch_metrics = all_reduce_dict(
                self._last_branch_metrics, op=torch.distributed.ReduceOp.AVG
            )
            mean_metric_dict.update(branch_metrics)

        self._log_progress(
            "[train] run_training done "
            f"step={self.global_step} total={mean_metric_dict['actor/total_training_s']:.1f}s "
            f"main_loop={mean_metric_dict['actor/main_loop_s']:.1f}s "
            f"prm={mean_metric_dict.get('actor/prm_total_s', 0.0):.1f}s"
        )

        self._synthetic_branch_entries = []
        return mean_metric_dict

    def set_global_step(self, global_step) -> None:
        """
        Set the global step for the model, if needed.
        """
        super().set_global_step(global_step)
        self.global_step = int(global_step)
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
        if self.global_step > 0:
            self._value_head_sync_ready = True
