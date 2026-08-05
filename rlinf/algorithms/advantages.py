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
from typing import Optional

import torch

from rlinf.algorithms.registry import register_advantage
from rlinf.algorithms.utils import kl_penalty, safe_normalize
from rlinf.utils.utils import masked_mean

_TERMINAL_BINARY_LOG_STATE = {"count": 0}


def _maybe_log_terminal_binary_adv(
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    success_label: torch.Tensor,
    adv_clip_max: float,
    used_success_once: bool,
) -> None:
    if _TERMINAL_BINARY_LOG_STATE["count"] >= 3:
        return
    _TERMINAL_BINARY_LOG_STATE["count"] += 1
    with torch.no_grad():
        mask = loss_mask if loss_mask is not None else torch.ones_like(advantages)
        mask = mask.to(dtype=torch.bool)
        if mask.shape != advantages.shape:
            mask = mask.expand_as(advantages)
        masked_adv = advantages[mask]
        masked_ret = returns[mask]
        if success_label.ndim == 2 and success_label.shape == advantages.shape:
            traj_mask = mask.any(dim=0)
            success_per_traj = (success_label > 0).any(dim=0) & traj_mask
            success_count = int(success_per_traj.sum().item())
            total_count = int(traj_mask.sum().item())
            label_unique = (
                torch.unique(success_per_traj.to(dtype=success_label.dtype))
                .detach()
                .cpu()
                .tolist()
            )
        else:
            masked_label = success_label[mask]
            label_unique = (
                torch.unique(masked_label).detach().cpu().tolist()
                if masked_label.numel() > 0
                else []
            )
            success_count = int((masked_label > 0).sum().item())
            total_count = int(masked_label.numel())
        adv_min = float(masked_adv.min().item()) if masked_adv.numel() > 0 else 0.0
        adv_max = float(masked_adv.max().item()) if masked_adv.numel() > 0 else 0.0
        adv_mean = float(masked_adv.mean().item()) if masked_adv.numel() > 0 else 0.0
        ret_min = float(masked_ret.min().item()) if masked_ret.numel() > 0 else 0.0
        ret_max = float(masked_ret.max().item()) if masked_ret.numel() > 0 else 0.0
        print(
            "[adv][terminal-binary] "
            f"used_success_once={used_success_once} adv_clip_max={adv_clip_max} "
            f"adv_min={adv_min:.3f} adv_max={adv_max:.3f} adv_mean={adv_mean:.3f} "
            f"ret_min={ret_min:.3f} ret_max={ret_max:.3f} "
            f"labels={label_unique} success_traj={success_count}/{total_count} src={__file__}",
            flush=True,
        )


@register_advantage("gae")
def compute_gae_advantages_and_returns(
    rewards: torch.Tensor,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    values: Optional[torch.Tensor] = None,
    normalize_advantages: bool = True,
    normalize_returns: bool = False,
    loss_mask: Optional[torch.Tensor] = None,
    dones: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate advantages and returns for Proximal Policy Optimization (PPO).
    NOTE: currently this function does not support auto-reset.

    This function implements Generalized Advantage Estimation (GAE) to compute
    advantages and returns for PPO training. The advantages are normalized
    using mean and standard deviation for stable training.

    Args:
        rewards (torch.Tensor): Rewards per timestep. Shape: [seq_len, bsz].
        values (torch.Tensor): Value function estimates. Shape: [seq_len, bsz].
        dones (torch.Tensor): Done flags (1 if episode ended, else 0).
        gamma (float, optional): Discount factor. Defaults to 1.0.
        gae_lambda (float, optional): GAE smoothing factor. Defaults to 1.0.
        normalize_advantages (bool, optional): Whether to normalize advantages. Defaults to True.
        normalize_returns (bool, optional): Whether to normalize returns. Defaults to False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (advantages, returns)
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    gae = 0

    critic_free = values is None
    if critic_free:
        gae_lambda = 1
        gamma = 1

    for step in reversed(range(T)):
        if critic_free:
            delta = rewards[step]
        else:
            delta = (
                rewards[step]
                + gamma * values[step + 1] * (~dones[step + 1])
                - values[step]
            )

        gae = delta + gamma * gae_lambda * (~dones[step + 1]) * gae
        returns[step] = gae if critic_free else gae + values[step]

    advantages = returns - values[:-1] if not critic_free else returns

    if normalize_advantages:
        advantages = safe_normalize(advantages, loss_mask=loss_mask)
    if normalize_returns:
        returns = safe_normalize(returns, loss_mask=loss_mask)

    return advantages, returns


@register_advantage("grpo")
def compute_grpo_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    **kwargs,
):
    """
    Compute GRPO advantages.

    Args:
        rewards (torch.Tensor): Reward or score values. Shape: [num_groups, group_size]
        loss_mask (torch.Tensor): Loss mask for valid entries. Shape: [num_groups, group_size]
        group_size (int): Group size for advantage computation.

    Returns:
        torch.Tensor: advantages
    """
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True).expand_as(
        grouped_rewards
    )

    advantages = grouped_rewards - grouped_reward_mean
    advantages = advantages / (grouped_reward_std + 1e-6)

    advantages = (torch.zeros_like(loss_mask) + advantages.view(1, -1)) * loss_mask

    return advantages, None


@register_advantage("reinpp")
def compute_reinpp_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    use_reinpp_baseline: bool = False,
    kl_beta: float = 0.0,
    logprob=None,
    ref_logprob=None,
    kl_penalty_type: str = "",
    **kwargs,
):
    """
    Compute advantages for reinforce++ and reinforce++ baseline.

    Args:
        rewards (torch.Tensor): The reward or score values.
        loss_mask (torch.Tensor): The loss mask for valid entries.
        group_size (int): The group size for advantage computation.
        use_reinpp_baseline (bool, optional): Whether to use reinforce++ baseline.
        kl_beta (float, optional): KL penalty coefficient.
        logprob (optional): Log probability of current policy.
        ref_logprob (optional): Log probability of reference policy.
        kl_penalty_type (str, optional): Type of KL penalty.

    Returns:
        torch.Tensor: advantages
    """
    # first group baseline for reinforce++ baseline
    if use_reinpp_baseline:
        grouped_rewards = rewards.view(-1, group_size)  # [num_prompt, group_size]
        grouped_rewards -= grouped_rewards.mean(dim=1, keepdims=True)
        rewards = grouped_rewards.view(-1)  # [B]

    # build the reward matrix
    r_matrix = torch.zeros_like(loss_mask).float()  # [L, B]
    seq_length = loss_mask.size(0)
    mask_flipped = loss_mask.long().fliplr()
    eos_positions = mask_flipped.argmax(
        dim=0, keepdim=True
    )  # position of last True in original mask
    eos_indices = seq_length - 1 - eos_positions  # [1, B]

    r_matrix = r_matrix.scatter_(dim=0, index=eos_indices, src=rewards)  # [L, B]

    # add kl penalty
    if kl_beta > 0:
        kld = kl_penalty(logprob, ref_logprob, kl_penalty=kl_penalty_type)  # [L, B]
        r_matrix -= kl_beta * kld

    # compute return
    ret_matrix = torch.cumsum(r_matrix.flip(dims=[0]), dim=0).flip(dims=[0])

    # normalize
    advantages = ret_matrix.clone()

    mean = masked_mean(advantages, loss_mask)
    var = masked_mean((advantages - mean).pow(2), loss_mask)
    rstd = var.clamp(min=1e-8).rsqrt()

    advantages = (advantages - mean) * rstd

    return advantages, None


@register_advantage("grpo-nft")
def compute_grpo_nft_advantages(
    rewards: torch.Tensor,
    loss_mask: torch.Tensor,
    group_size: int,
    epsilon: float = 1e-6,
    failure_threshold: float = 1e-3,
    success_threshold: float = 0.99,
    virtual_min_reward: float = 0.0,
    virtual_max_reward: float = 1.0,
    penalty_scale: float = 1.0,
    reward_scale: float = 1.0,
    **kwargs,
):
    grouped_rewards = rewards.view(-1, group_size)

    grouped_reward_mean = grouped_rewards.mean(dim=-1, keepdim=True)
    grouped_reward_std = grouped_rewards.std(dim=-1, keepdim=True)

    is_low_variance = grouped_reward_std < epsilon
    is_all_failure = is_low_variance & (grouped_reward_mean < failure_threshold)
    is_all_success = is_low_variance & (grouped_reward_mean > success_threshold)

    normal_advantages = (grouped_rewards - grouped_reward_mean) / (
        grouped_reward_std + epsilon
    )

    group_size_float = float(group_size)
    augmented = group_size_float + 1.0

    mean_fail = (group_size_float * virtual_min_reward + virtual_max_reward) / augmented
    mean_sq_fail = (
        group_size_float * (virtual_min_reward**2) + (virtual_max_reward**2)
    ) / augmented
    var_fail = mean_sq_fail - mean_fail**2
    std_fail = torch.sqrt(
        torch.tensor(var_fail, device=rewards.device).clamp(min=epsilon)
    )
    base_penalty = (virtual_min_reward - mean_fail) / (std_fail + epsilon)
    dynamic_penalty_val = base_penalty * penalty_scale

    mean_succ = (group_size_float * virtual_max_reward + virtual_min_reward) / augmented
    mean_sq_succ = (
        group_size_float * (virtual_max_reward**2) + (virtual_min_reward**2)
    ) / augmented
    var_succ = mean_sq_succ - mean_succ**2
    std_succ = torch.sqrt(
        torch.tensor(var_succ, device=rewards.device).clamp(min=epsilon)
    )
    base_reward = (virtual_max_reward - mean_succ) / (std_succ + epsilon)
    dynamic_reward_val = base_reward * reward_scale

    failure_mask_expanded = is_all_failure.expand_as(grouped_rewards)
    success_mask_expanded = is_all_success.expand_as(grouped_rewards)

    penalty_tensor = torch.tensor(
        dynamic_penalty_val, device=rewards.device, dtype=rewards.dtype
    )
    reward_tensor = torch.tensor(
        dynamic_reward_val, device=rewards.device, dtype=rewards.dtype
    )

    advantages = torch.where(failure_mask_expanded, penalty_tensor, normal_advantages)
    advantages = torch.where(success_mask_expanded, reward_tensor, advantages)

    advantages = advantages.view(-1)
    if loss_mask is not None:
        advantages = advantages.view(1, -1) * loss_mask

    return advantages, None


@register_advantage("value-as-score")
def compute_value_as_score_advantages(
    rewards: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    values: torch.Tensor,
    normalize_advantages: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values is None:
        raise ValueError("compute_value_as_score_advantages requires `values` in kwargs.")

    def _squeeze_last_dim(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1) if x.ndim >= 3 and x.shape[-1] == 1 else x

    rewards = _squeeze_last_dim(rewards)
    values = _squeeze_last_dim(values)
    if loss_mask is not None:
        loss_mask = _squeeze_last_dim(loss_mask)

    if loss_mask is not None and loss_mask.ndim == 2:
        time_len, batch_size = loss_mask.shape
    elif values.ndim == 2:
        time_len, batch_size = values.shape
    elif values.ndim == 1:
        time_len, batch_size = 1, values.shape[0]
    else:
        raise ValueError(
            f"compute_value_as_score_advantages: unsupported values shape {values.shape}"
        )

    def _align_to_time_first(
        t: Optional[torch.Tensor], name: str
    ) -> Optional[torch.Tensor]:
        if t is None:
            return None
        t = _squeeze_last_dim(t)
        if t.ndim == 2 and t.shape[1] == batch_size:
            if t.shape[0] == time_len:
                return t
            if t.shape[0] > time_len:
                return t[:time_len]
            last_row = t[-1:].clone()
            pad_rows = last_row.expand(time_len - t.shape[0], -1)
            t = torch.cat([t, pad_rows], dim=0)
            return t
        if t.ndim == 2 and t.shape[0] == batch_size:
            t = t.transpose(0, 1)
            if t.shape[0] == time_len:
                return t
            if t.shape[0] > time_len:
                return t[:time_len]
            last_row = t[-1:].clone()
            pad_rows = last_row.expand(time_len - t.shape[0], -1)
            t = torch.cat([t, pad_rows], dim=0)
            return t
        if t.ndim == 2 and t.shape == (batch_size, 1):
            return t.expand(batch_size, time_len).transpose(0, 1)
        if t.ndim == 1 and t.shape[0] == batch_size:
            return t.view(1, batch_size).expand(time_len, batch_size)
        if t.numel() >= time_len * batch_size:
            return t.reshape(-1, batch_size)[:time_len]
        raise ValueError(
            f"compute_value_as_score_advantages: cannot align {name}.shape={t.shape} "
            f"to time_len={time_len}, batch_size={batch_size}"
        )

    rewards = _align_to_time_first(rewards, "rewards")
    values = _align_to_time_first(values, "values")
    loss_mask = _align_to_time_first(loss_mask, "loss_mask")

    score_probs = torch.sigmoid(values)

    if loss_mask is not None and loss_mask.shape == rewards.shape:
        mask_float = loss_mask.float()
        lengths = mask_float.sum(dim=0).clamp(min=1).long() - 1
        idx = lengths.unsqueeze(0).expand(1, batch_size)
        terminal_reward = rewards.gather(0, idx).squeeze(0)
    else:
        terminal_reward = rewards[-1]

    terminal_reward = terminal_reward.view(1, batch_size).expand(time_len, batch_size)
    returns = (terminal_reward > 0).to(
        dtype=score_probs.dtype, device=score_probs.device
    )

    advantages = score_probs

    if loss_mask is not None:
        advantages = advantages * loss_mask.float()
        returns = returns * loss_mask.float()

    return advantages, returns


@register_advantage("terminal-binary")
def compute_terminal_binary_advantages(
    rewards: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    adv_clip_max: float = 1.0,
    dones: Optional[torch.Tensor] = None,
    success_once: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Broadcast terminal reward to every step and map success/failure to fixed NFT labels.

    - advantages: +adv_clip_max for success (reward>0), -adv_clip_max for failure.
      This ensures NFT r=1/y=1 for success and r=0/y=-1 for failure after loss scaling.
    - returns: 1 for success else 0, broadcast to all steps.
    """

    def _squeeze_last_dim(x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(-1) if x.ndim >= 3 and x.shape[-1] == 1 else x

    rewards = _squeeze_last_dim(rewards)
    if loss_mask is not None:
        loss_mask = _squeeze_last_dim(loss_mask)

    # Infer time_len / batch_size (prefer loss_mask if available to respect padding).
    if loss_mask is not None and loss_mask.ndim == 2:
        time_len, batch_size = loss_mask.shape
    elif rewards.ndim == 2:
        time_len, batch_size = rewards.shape
    elif rewards.ndim == 1:
        time_len, batch_size = 1, rewards.shape[0]
    else:
        raise ValueError(f"compute_terminal_binary_advantages: unsupported rewards shape {rewards.shape}")

    def _align_to_time_first(t: Optional[torch.Tensor], name: str) -> Optional[torch.Tensor]:
        if t is None:
            return None
        t = _squeeze_last_dim(t)
        if name == "success_once" and t.ndim == 2:
            if t.shape[0] == time_len + 1 and t.shape[1] == batch_size:
                return t[1:time_len + 1]
            if t.shape[1] == time_len + 1 and t.shape[0] == batch_size:
                return t[:, 1:time_len + 1].transpose(0, 1)
        if t.ndim == 2 and t.shape[1] == batch_size:
            if t.shape[0] == time_len:
                return t
            if t.shape[0] > time_len:
                return t[:time_len]
            last_row = t[-1:].clone()
            pad_rows = last_row.expand(time_len - t.shape[0], -1)
            t = torch.cat([t, pad_rows], dim=0)
            return t
        if t.ndim == 2 and t.shape[0] == batch_size:
            t = t.transpose(0, 1)
            if t.shape[0] == time_len:
                return t
            if t.shape[0] > time_len:
                return t[:time_len]
            last_row = t[-1:].clone()
            pad_rows = last_row.expand(time_len - t.shape[0], -1)
            t = torch.cat([t, pad_rows], dim=0)
            return t
        if t.ndim == 2 and t.shape == (batch_size, 1):
            return t.expand(batch_size, time_len).transpose(0, 1)
        if t.ndim == 1 and t.shape[0] == batch_size:
            return t.view(1, batch_size).expand(time_len, batch_size)
        if t.numel() >= time_len * batch_size:
            return t.reshape(-1, batch_size)[:time_len]
        raise ValueError(
            f"compute_terminal_binary_advantages: cannot align {name}.shape={t.shape} to time_len={time_len}, batch_size={batch_size}"
        )

    rewards = _align_to_time_first(rewards, "rewards")
    loss_mask = _align_to_time_first(loss_mask, "loss_mask")

    success_once = _align_to_time_first(success_once, "success_once")

    used_success_once = success_once is not None
    if success_once is not None:
        success_steps = success_once.to(dtype=rewards.dtype, device=rewards.device)
        if success_steps.ndim == 2:
            success_per_batch = (success_steps > 0).any(dim=0)
        elif success_steps.ndim == 1:
            success_per_batch = success_steps > 0
        else:
            success_per_batch = (success_steps > 0).reshape(-1, batch_size).any(dim=0)
        success_labels = success_per_batch.view(1, batch_size).expand(
            time_len, batch_size
        )
    else:
        if loss_mask is not None and loss_mask.shape == rewards.shape:
            mask_float = loss_mask.float()
            lengths = mask_float.sum(dim=0).clamp(min=1).long() - 1
            idx = lengths.unsqueeze(0).expand(1, batch_size)
            terminal_reward = rewards.gather(0, idx).squeeze(0)
        else:
            terminal_reward = rewards[-1]
        success_labels = terminal_reward.view(1, batch_size).expand(time_len, batch_size)

    success_label = (success_labels > 0).to(dtype=rewards.dtype, device=rewards.device)
    base = success_label * 2 - 1  # success -> 1, failure -> -1
    adv_scale = torch.as_tensor(
        adv_clip_max if adv_clip_max > 0 else 5.0, device=rewards.device, dtype=rewards.dtype
    )
    advantages = base * adv_scale
    advantages = advantages.view(time_len, batch_size)
    returns = success_label.view(time_len, batch_size)

    if loss_mask is not None:
        mask_float = loss_mask.float()
        advantages = advantages * mask_float
        returns = returns * mask_float

    _maybe_log_terminal_binary_adv(
        rewards=rewards,
        advantages=advantages,
        returns=returns,
        loss_mask=loss_mask,
        success_label=success_label,
        adv_clip_max=adv_clip_max,
        used_success_once=used_success_once,
    )

    return advantages, returns


@register_advantage("dual-credit")
def compute_dual_credit_advantages(
    rewards: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    success_once: Optional[torch.Tensor] = None,
    group_size: int = 4,
    adv_clip_max: float = 1.0,
    task_ids: Optional[torch.Tensor] = None,
    reset_state_ids: Optional[torch.Tensor] = None,
    prm_pos_top_ratio: float = 0.9,
    prm_pos_min_count: int = 1,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct trajectory labels and PRM preference pairs for dual-credit RL.

    Preference groups are identified by semantic rollout metadata when available:
    trajectories sharing the same ``(task_id, reset_state_id)`` are grouped
    together. Inside each group, all failure trajectories are paired against the
    earliest-success subset consisting of
    ``min(num_success, max(prm_pos_min_count, floor(prm_pos_top_ratio * num_success)))``
    trajectories.

    We keep both:
    - ``preference_pairs``: the full Cartesian candidate set, preserved as a
      deterministic fallback / diagnostic baseline.
    - ``preference_group_specs``: compact group-level candidate metadata used by
      downstream PRM training to perform semantic hard-negative mining without
      changing the trajectory-level labels or actor losses.
    """

    def _squeeze(x):
        return x.squeeze(-1) if x is not None and x.ndim >= 3 and x.shape[-1] == 1 else x

    def _extract_traj_ids(
        meta: Optional[torch.Tensor], name: str
    ) -> Optional[torch.Tensor]:
        if meta is None:
            return None
        meta = _squeeze(meta)
        if meta.ndim == 1:
            if meta.shape[0] != batch_size:
                raise ValueError(
                    f"dual-credit: {name} shape {meta.shape} incompatible with batch_size={batch_size}"
                )
            return meta.to(dtype=torch.long)
        if meta.ndim != 2:
            raise ValueError(
                f"dual-credit: unsupported {name} shape {meta.shape}, expected [T, B] or [B]"
            )
        if meta.shape[1] != batch_size:
            if meta.shape[0] == batch_size and meta.shape[1] == time_len:
                meta = meta.T
            else:
                raise ValueError(
                    f"dual-credit: {name} shape {meta.shape} incompatible with time_len={time_len}, batch_size={batch_size}"
                )

        traj_ids = torch.empty(batch_size, device=meta.device, dtype=torch.long)
        for b in range(batch_size):
            if loss_mask is not None:
                valid_steps = loss_mask[:, b].bool()
            else:
                valid_steps = torch.ones(time_len, device=meta.device, dtype=torch.bool)
            vals = meta[:, b][valid_steps]
            if vals.numel() == 0:
                vals = meta[:, b][:1]
            if vals.numel() > 1 and torch.any(vals != vals[0]):
                raise ValueError(
                    f"dual-credit: {name} varies within trajectory {b}, expected a fixed rollout-group identifier"
                )
            traj_ids[b] = vals[0].long()
        return traj_ids

    rewards = _squeeze(rewards)
    if loss_mask is not None:
        loss_mask = _squeeze(loss_mask)

    if loss_mask is not None and loss_mask.ndim == 2:
        time_len, batch_size = loss_mask.shape
    elif rewards.ndim == 2:
        time_len, batch_size = rewards.shape
    elif rewards.ndim == 1:
        time_len, batch_size = 1, rewards.shape[0]
    else:
        raise ValueError(f"dual-credit: unsupported rewards shape {rewards.shape}")

    if not (0.0 < float(prm_pos_top_ratio) <= 1.0):
        raise ValueError(
            f"dual-credit: prm_pos_top_ratio must be in (0, 1], got {prm_pos_top_ratio}"
        )
    if int(prm_pos_min_count) < 1:
        raise ValueError(
            f"dual-credit: prm_pos_min_count must be >= 1, got {prm_pos_min_count}"
        )

    # --- determine per-trajectory success ---
    if success_once is not None:
        so = _squeeze(success_once)
        if so.ndim == 2:
            if so.shape[0] == time_len + 1:
                so = so[1:]
            elif so.shape[1] == time_len + 1:
                so = so[:, 1:].T
            if so.shape[0] != time_len:
                so = so[:time_len]
            success_per_batch = (so > 0).any(dim=0)
            first_success = torch.where(
                so > 0,
                torch.arange(time_len, device=so.device).unsqueeze(1).expand_as(so),
                torch.full_like(so, time_len, dtype=torch.long),
            ).min(dim=0).values
        else:
            success_per_batch = so > 0
            first_success = torch.zeros(batch_size, device=rewards.device, dtype=torch.long)
    else:
        terminal_reward = rewards[-1] if rewards.ndim == 2 else rewards
        success_per_batch = terminal_reward > 0
        first_success = torch.zeros(batch_size, device=rewards.device, dtype=torch.long)

    traj_task_ids = _extract_traj_ids(task_ids, "task_ids")
    traj_reset_state_ids = _extract_traj_ids(reset_state_ids, "reset_state_ids")

    # --- build preference pairs per semantic group ---
    preference_pairs = []
    preference_group_specs = []
    y_tau = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)
    groups_no_success = 0
    groups_no_failure = 0
    groups_with_pairs = 0
    group_success_counts = []
    group_failure_counts = []
    group_topk_counts = []
    group_pairs_counts = []
    group_topk_eq_one = 0
    group_topk_eq_all_success = 0

    group_to_indices: dict[tuple[int, ...], list[int]] = {}
    if traj_task_ids is not None and traj_reset_state_ids is not None:
        for idx in range(batch_size):
            key = (int(traj_task_ids[idx].item()), int(traj_reset_state_ids[idx].item()))
            group_to_indices.setdefault(key, []).append(idx)
    elif traj_task_ids is not None:
        for idx in range(batch_size):
            key = (int(traj_task_ids[idx].item()),)
            group_to_indices.setdefault(key, []).append(idx)
    else:
        if batch_size % group_size != 0:
            raise ValueError(
                f"dual-credit requires batch_size divisible by group_size, got "
                f"batch_size={batch_size}, group_size={group_size}"
            )
        num_groups = batch_size // group_size
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group_to_indices[(g,)] = list(range(start, end))

    for group_key, indices in group_to_indices.items():
        grp_idx = torch.as_tensor(indices, device=rewards.device, dtype=torch.long)
        grp_success = success_per_batch[grp_idx]
        y_tau[grp_idx] = torch.where(
            grp_success,
            torch.ones_like(grp_success, dtype=rewards.dtype),
            -torch.ones_like(grp_success, dtype=rewards.dtype),
        )

        succ_idx = grp_idx[grp_success]
        fail_idx = grp_idx[~grp_success]
        succ_count = int(succ_idx.numel())
        fail_count = int(fail_idx.numel())
        group_success_counts.append(succ_count)
        group_failure_counts.append(fail_count)

        if succ_idx.numel() == 0:
            groups_no_success += 1
        if fail_idx.numel() == 0:
            groups_no_failure += 1

        pos_candidates = succ_idx
        top_k = 0
        if succ_idx.numel() > 0:
            succ_order = first_success[succ_idx].argsort()
            succ_idx = succ_idx[succ_order]
            top_k = min(
                succ_count,
                max(
                    int(prm_pos_min_count),
                    math.floor(float(prm_pos_top_ratio) * succ_count),
                ),
            )
            pos_candidates = succ_idx[:top_k]
            group_topk_counts.append(int(top_k))

        preference_group_specs.append(
            {
                "group_key": tuple(int(x) for x in group_key),
                "pos_candidates": pos_candidates.tolist(),
                "fail_candidates": fail_idx.tolist(),
                "succ_count": succ_count,
                "fail_count": fail_count,
                "top_k": int(top_k),
                "has_pair_candidates": bool(
                    succ_idx.numel() > 0 and fail_idx.numel() > 0 and top_k > 0
                ),
            }
        )

        if succ_idx.numel() == 0 or fail_idx.numel() == 0 or top_k <= 0:
            continue

        group_pairs_counts.append(int(top_k * fail_count))
        groups_with_pairs += 1
        group_topk_eq_one += int(top_k == 1)
        group_topk_eq_all_success += int(top_k == succ_count)

        for tau_plus in pos_candidates.tolist():
            for tau_minus in fail_idx.tolist():
                preference_pairs.append((tau_plus, tau_minus))

    # --- broadcast to [time_len, batch_size] ---
    advantages = (y_tau * adv_clip_max).unsqueeze(0).expand(time_len, batch_size)
    returns = success_per_batch.float().unsqueeze(0).expand(time_len, batch_size)

    if loss_mask is not None:
        mask_float = loss_mask.float()
        advantages = advantages * mask_float
        returns = returns * mask_float

    rb = kwargs.get("_rollout_batch", None)
    if rb is not None:
        rb["preference_pairs"] = preference_pairs
        rb["preference_group_specs"] = preference_group_specs
        rb["preference_group_count"] = len(group_to_indices)
        rb["preference_groups_total"] = len(group_to_indices)
        rb["preference_groups_with_pairs"] = groups_with_pairs
        rb["preference_groups_no_success"] = groups_no_success
        rb["preference_groups_no_failure"] = groups_no_failure
        rb["preference_group_success_mean"] = (
            float(sum(group_success_counts) / len(group_success_counts))
            if group_success_counts
            else 0.0
        )
        rb["preference_group_success_min"] = (
            int(min(group_success_counts)) if group_success_counts else 0
        )
        rb["preference_group_success_max"] = (
            int(max(group_success_counts)) if group_success_counts else 0
        )
        rb["preference_group_failure_mean"] = (
            float(sum(group_failure_counts) / len(group_failure_counts))
            if group_failure_counts
            else 0.0
        )
        rb["preference_group_failure_min"] = (
            int(min(group_failure_counts)) if group_failure_counts else 0
        )
        rb["preference_group_failure_max"] = (
            int(max(group_failure_counts)) if group_failure_counts else 0
        )
        rb["preference_group_topk_mean"] = (
            float(sum(group_topk_counts) / len(group_topk_counts))
            if group_topk_counts
            else 0.0
        )
        rb["preference_group_topk_min"] = int(min(group_topk_counts)) if group_topk_counts else 0
        rb["preference_group_topk_max"] = int(max(group_topk_counts)) if group_topk_counts else 0
        rb["preference_group_topk_eq_1_frac"] = (
            float(group_topk_eq_one / groups_with_pairs) if groups_with_pairs > 0 else 0.0
        )
        rb["preference_group_topk_eq_all_success_frac"] = (
            float(group_topk_eq_all_success / groups_with_pairs)
            if groups_with_pairs > 0
            else 0.0
        )
        rb["preference_group_pairs_mean"] = (
            float(sum(group_pairs_counts) / len(group_pairs_counts))
            if group_pairs_counts
            else 0.0
        )
        rb["preference_group_pairs_max"] = (
            int(max(group_pairs_counts)) if group_pairs_counts else 0
        )

    return advantages, returns
