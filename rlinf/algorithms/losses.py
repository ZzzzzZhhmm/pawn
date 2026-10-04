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

import math
from collections.abc import Sequence
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from rlinf.algorithms.registry import register_policy_loss
from rlinf.algorithms.utils import huber_loss
from rlinf.utils.utils import masked_mean, masked_mean_ratio


def compute_ppo_actor_loss(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    clip_ratio_c: Optional[float] = None,
    loss_agg_func: Optional[Callable[..., torch.Tensor]] = masked_mean,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    critic_warmup: Optional[bool] = False,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.FloatTensor): Log probabilities of actions.
        old_logprobs (torch.FloatTensor): Old log probabilities of actions.
        clip_ratio_low (float): Lower bound of clipping ratio.
        clip_ratio_high (float): Upper bound of clipping ratio.
        advantages (torch.FloatTensor): GAE (normalized) advantages.
        loss_mask (Optional[torch.BoolTensor], optional): Mask for valid entries. Defaults to None.
        clip_ratio_c (Optional[float], optional): Optional clipping coefficient. Defaults to None.
        loss_agg_func (callable, optional): Aggregation function (e.g., masked_mean). Defaults to None.
        max_episode_steps (Optional[int], optional): Max episode length for normalization. Defaults to None.

    Returns:
        Tuple[torch.Tensor, Dict]: (actor_loss, metrics_dict)
    """

    loss_mask_ratio = None

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    if loss_mask is None:
        loss_mask = torch.ones_like(logprobs).bool()

    assert logprobs.dtype == torch.float32
    assert old_logprobs.dtype == torch.float32
    assert advantages.dtype == torch.float32

    loss_mask_count = loss_mask.count_nonzero() or 1
    # For numerical stability.
    ratio = torch.where(loss_mask, torch.exp(logprobs - old_logprobs), 0)
    approx_kl = torch.where(loss_mask, (logprobs - old_logprobs).detach(), 0.0)

    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    policy_loss1 = -advantages * ratio
    policy_loss2 = -advantages * clipped_ratio

    clip_mask = policy_loss1.detach() < policy_loss2.detach()

    policy_loss = torch.max(policy_loss1, policy_loss2)
    if clip_ratio_c is not None:
        assert clip_ratio_c > 1.0, clip_ratio_c
        policy_loss3 = torch.sign(advantages) * clip_ratio_c * advantages
        dual_clip_mask = policy_loss3.detach() < policy_loss.detach()
        policy_loss = torch.min(policy_loss, policy_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)

    policy_loss = loss_agg_func(
        policy_loss, loss_mask, loss_mask_ratio
    )  # default max_episode_steps is None

    clip_mask = policy_loss1.detach() < policy_loss2.detach()
    dual_clip_mask.logical_and_(loss_mask)

    clip_fraction = clip_mask.logical_and_(loss_mask).count_nonzero() / loss_mask_count
    approx_kl = -approx_kl.sum() / loss_mask_count

    dual_cliped_ratio = torch.where(dual_clip_mask, ratio, 0)

    if critic_warmup:
        # Preserve the computation graph while zeroing gradients during warmup.
        policy_loss = policy_loss * 0.0

    # Compile metrics for logging
    ratio_for_metrics = ratio.detach()
    clipped_ratio_for_metrics = clipped_ratio.detach()
    dual_cliped_ratio_for_metrics = dual_cliped_ratio.detach()
    loss_mask_for_metrics = loss_mask

    # Only broadcast when ratio has action_dim dimension and loss_mask's last dim is 1
    # This handles token_level mode: ratio [bsz, num_chunks, action_dim], loss_mask [bsz, num_chunks, 1]
    if len(ratio.shape) > 2 and loss_mask.shape[-1] == 1 and ratio.shape[-1] > 1:
        # Broadcast loss_mask to match ratio's shape for metrics computation
        loss_mask_for_metrics = loss_mask.expand_as(ratio)

    metrics_data = {
        "actor/policy_loss": policy_loss.detach(),
        "actor/ratio": masked_mean(ratio_for_metrics, loss_mask_for_metrics),
        "actor/clipped_ratio": masked_mean(
            clipped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/dual_cliped_ratio": masked_mean(
            dual_cliped_ratio_for_metrics, loss_mask_for_metrics
        ),
        "actor/approx_kl": approx_kl.detach(),
        "actor/clip_fraction": clip_fraction.detach(),
    }
    return policy_loss, metrics_data


def compute_ppo_critic_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    prev_values: torch.Tensor,
    value_clip: float,
    huber_delta: float,
    loss_mask: Optional[torch.Tensor] = None,
    max_episode_steps: Optional[int] = None,
    loss_mask_sum: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO critic loss function.

    Args:
        values (torch.Tensor): Current value predictions.
        returns (torch.Tensor): Return values.
        prev_values (torch.Tensor): Previous value predictions.
        value_clip (float): Value clipping threshold.
        huber_delta (float): Huber loss delta parameter.

    Returns:
        Tuple[torch.Tensor, Dict]: (critic_loss, metrics_dict)
    """
    loss_mask_ratio = None
    loss_agg_func = masked_mean

    if (
        max_episode_steps is not None
        and loss_mask_sum is not None
        and loss_mask is not None
    ):
        loss_mask_ratio = (loss_mask_sum * 1.0) / max_episode_steps
        loss_agg_func = masked_mean_ratio

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -value_clip, value_clip
    )  # [bsz, ] | [bsz, chunk-step]

    value_loss_original = huber_loss(
        returns - values, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss_clipped = huber_loss(
        returns - value_pred_clipped, huber_delta
    )  # [bsz, ] | [bsz, chunk-step]
    value_loss = torch.max(value_loss_original, value_loss_clipped)
    value_loss = loss_agg_func(value_loss, loss_mask, loss_mask_ratio)

    value_clip_indicator = (value_pred_clipped - prev_values).abs() > value_clip
    value_clip_ratio = value_clip_indicator.float().mean()

    # explained variance
    if loss_mask is not None:
        masked_returns = returns[loss_mask]
        masked_values = values[loss_mask]
    else:
        masked_returns = returns
        masked_values = values

    var_returns = torch.var(masked_returns)
    if torch.isnan(var_returns) or var_returns == 0:
        explained_variance = torch.tensor(float("nan"), device=returns.device)
    else:
        var_diff = torch.var(masked_returns - masked_values)
        if torch.isnan(var_diff):
            explained_variance = torch.tensor(float("nan"), device=returns.device)
        else:
            explained_variance = 1 - var_diff / var_returns

    # Compile metrics for logging
    metrics_data = {
        "critic/value_loss": value_loss.detach().item(),
        "critic/value_clip_ratio": value_clip_ratio.detach().item(),
        "critic/explained_variance": explained_variance.detach().item(),
    }
    return value_loss, metrics_data


@register_policy_loss("actor_critic")
def compute_ppo_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute PPO actor loss function.

    Args:
        logprobs (torch.Tensor): Log probabilities of actions
        values (torch.Tensor): Current value predictions
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values
        returns (torch.Tensor): Return values
        prev_values (torch.Tensor): Previous value predictions
        clip_ratio_low (float): Lower clipping ratio for PPO
        clip_ratio_high (float): Upper clipping ratio for PPO
        value_clip (float): Value clipping threshold
        huber_delta (float): Huber loss delta parameter

    Returns:
        Tuple[torch.Tensor, Dict]: Loss and metrics dictionary
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    critic_loss, critic_metrics_data = compute_ppo_critic_loss(**kwargs)

    loss = actor_loss + critic_loss
    metrics_data.update(actor_metrics_data)
    metrics_data.update(critic_metrics_data)

    return loss, metrics_data


@register_policy_loss("actor")
def compute_grpo_actor_loss_fn(**kwargs) -> tuple[torch.Tensor, dict]:
    """
    Compute actor loss for Group Relative Policy Optimization (GRPO).

    This function implements the PPO-style actor loss with clipping for GRPO.
    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppotrainer.py#L1122

    Args:
        log_prob (torch.Tensor): Current log probabilities
        old_log_prob (torch.Tensor): Previous log probabilities
        advantages (torch.Tensor): Advantage values of shape
        clip_ratio_high (float): Upper clipping ratio for PPO
        clip_ratio_low (float): Lower clipping ratio for PPO
        loss_mask (Optional[torch.Tensor]): Mask tensor of shape to apply to the loss

    Returns:
        Tuple[torch.Tensor, Dict]: Policy gradient loss and metrics dictionary containing:
            - actor/loss: Total actor loss
            - actor/policy_loss: Policy gradient loss
            - actor/clip_fraction: Fraction of clipped policy gradient loss
            - actor/ppo_kl: Approximate KL divergence
    """
    metrics_data = {}
    actor_loss, actor_metrics_data = compute_ppo_actor_loss(**kwargs)
    metrics_data.update(actor_metrics_data)

    return actor_loss, metrics_data


@register_policy_loss("nft-actor")
def compute_nft_actor_loss(
    v_theta: torch.Tensor,
    v_old: torch.Tensor,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    schedule: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    step_indices: Optional[torch.Tensor] = None,
    total_denoise_steps: Optional[int] = None,
    noise_level: Optional[torch.Tensor | float] = None,
    std_epsilon: float = 1e-4,
    beta: float = 1.0,
    kl_beta: float = 0.0001,
    adv_clip_max: float = 1.0,
    critic_warmup: bool = False,
    x0_target: Optional[torch.Tensor] = None,
    use_x0_target: bool = False,
    loss_form: str = "dpo",
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    def _align_to_steps(
        x: torch.Tensor | None, target_shape: Sequence[int]
    ) -> torch.Tensor | None:
        if x is None:
            return None
        target_b, target_steps = target_shape
        if x.shape == (target_b, target_steps):
            return x
        if x.ndim == 1 and x.shape[0] == target_b:
            return x.unsqueeze(1).expand(target_b, target_steps)
        if x.ndim == 2 and x.shape[0] == target_b and x.shape[1] == 1:
            return x.expand(target_b, target_steps)
        if x.numel() == target_b * target_steps:
            return x.reshape(target_b, target_steps)
        raise ValueError(f"Cannot align tensor of shape {x.shape} to steps {target_shape}")

    def _masked_mean_per_traj(
        x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is None:
            traj_mean = x.mean(dim=1)
            valid_mask = torch.ones_like(traj_mean, dtype=torch.bool)
            return traj_mean, valid_mask
        mask = mask.float()
        x = x * mask
        valid = mask.sum(dim=1)
        valid_clamped = valid.clamp_min(1.0)
        traj_sum = x.sum(dim=1)
        valid_mask = valid > 0
        traj_mean = torch.where(
            valid_mask, traj_sum / valid_clamped, torch.zeros_like(traj_sum)
        )
        return traj_mean, valid_mask

    def _pad_right_ndim(x: torch.Tensor, target_ndim: int) -> torch.Tensor:
        while x.ndim < target_ndim:
            x = x.unsqueeze(-1)
        return x

    batch_size, n_steps = x_t.shape[:2]
    step_shape = (batch_size, n_steps)

    advantages = _align_to_steps(advantages, step_shape)
    loss_mask = _align_to_steps(loss_mask, step_shape)

    if advantages is None:
        raise ValueError("NFT loss requires `advantages`.")
    if loss_mask is None:
        loss_mask_float = None
    else:
        loss_mask_float = loss_mask.float()

    if step_indices is None or total_denoise_steps is None or noise_level is None:
        raise ValueError(
            "step_indices, total_denoise_steps, and noise_level must be provided for NFT loss."
        )

    advantages_clip = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    normalized_advantages_clip = (advantages_clip / adv_clip_max) / 2.0 + 0.5
    r = torch.clamp(normalized_advantages_clip, 0, 1)
    y = r * 2.0 - 1.0

    v_old = v_old.detach()
    delta_v = v_theta - v_old

    dims_v = tuple(range(2, delta_v.ndim))
    delta_norm = delta_v.norm(dim=dims_v, keepdim=True) + 1e-8
    max_drift = float(kwargs.get("max_drift", 0.5))
    clip_coef = (max_drift / delta_norm).clamp(max=1.0)

    delta_v_clipped = delta_v * clip_coef
    v_pos = v_old + beta * delta_v_clipped
    v_neg = v_old - beta * delta_v_clipped

    dims = tuple(range(2, x_t.ndim))
    idx = step_indices.long()
    t_cur = schedule[idx]
    t_next = schedule[idx + 1]
    delta = t_cur - t_next

    t_bc = _pad_right_ndim(t_cur, x_t.ndim)
    delta_bc = _pad_right_ndim(delta, x_t.ndim)

    denom = schedule.clone()
    denom[0] = denom[1]
    sigma_base = torch.sqrt(schedule / (1 - denom))[:-1]
    sigma_i = _pad_right_ndim(sigma_base[idx], x_t.ndim)
    nl_tensor = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
    sigma_i = sigma_i * _pad_right_ndim(nl_tensor, sigma_i.ndim)

    std_t = torch.sqrt(delta_bc.clamp_min(0)) * sigma_i
    std_t_detached = std_t.detach()

    if use_x0_target:
        if x0_target is None:
            raise ValueError("use_x0_target=True requires `x0_target` to be provided.")
        if x0_target.shape != x_t.shape:
            raise ValueError(
                f"x0_target shape {x0_target.shape} must match x_t shape {x_t.shape}."
            )
        x0_pos = x_t - t_bc * v_pos
        x0_neg = x_t - t_bc * v_neg
        var = std_t_detached**2 + std_epsilon
        E_pos = ((x0_pos - x0_target) ** 2 / var).sum(dim=dims)
        E_neg = ((x0_neg - x0_target) ** 2 / var).sum(dim=dims)
        delta_E = E_pos - E_neg
    else:
        def _flow_mean(x_cur: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
            x0_pred = x_cur - velocity * t_bc
            x1_pred = x_cur + velocity * (1 - t_bc)
            x0_weight = torch.ones_like(t_bc) - (t_bc - delta_bc)
            x1_weight = t_bc - delta_bc - sigma_i**2 * delta_bc / (2 * t_bc)
            return x0_pred * x0_weight + x1_pred * x1_weight

        mean_pos = _flow_mean(x_t, v_pos)
        mean_neg = _flow_mean(x_t, v_neg)
        var = std_t_detached**2 + std_epsilon
        E_pos = ((x_next - mean_pos) ** 2 / var).sum(dim=dims)
        E_neg = ((x_next - mean_neg) ** 2 / var).sum(dim=dims)
        delta_E = E_pos - E_neg

    dpo_beta = float(kwargs.get("dpo_beta", 1.0))
    logit = (dpo_beta / 2.0) * y * delta_E

    if loss_form == "weighted":
        L_step = r * E_pos + (1 - r) * E_neg
        traj_loss, traj_valid = _masked_mean_per_traj(L_step, loss_mask_float)
    else:
        L_step = F.softplus(logit)
        traj_loss, traj_valid = _masked_mean_per_traj(L_step, loss_mask_float)
    nft_loss = traj_loss.sum() / traj_valid.sum().clamp_min(1.0)

    kl_loss_per_sample = torch.mean((v_theta - v_old) ** 2, dim=dims)
    kl_per_traj, kl_valid = _masked_mean_per_traj(kl_loss_per_sample, loss_mask_float)
    kl_loss = kl_per_traj.sum() / kl_valid.sum().clamp_min(1.0)

    total_loss = nft_loss + kl_beta * kl_loss
    if critic_warmup:
        # Preserve the computation graph while zeroing gradients during warmup.
        total_loss = total_loss * 0.0

    with torch.no_grad():
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        adv_clip_frac = (advantages.abs() >= adv_clip_max).float().mean()
        r_mean = r.mean()
        r_std = r.std()
        y_abs_mean = y.abs().mean()
        y_sat_frac = ((r < 0.05) | (r > 0.95)).float().mean()

        delta_v_norm = delta_v.norm(dim=dims_v)
        delta_v_clipped_norm = delta_v_clipped.norm(dim=dims_v)
        clip_frac = (clip_coef < 1).float().mean()
        clip_coef_mean = clip_coef.mean()

        std_mean = std_t_detached.mean()
        std_min = std_t_detached.min()
        std_max = std_t_detached.max()
        z2_mean = (
            ((x0_pos - x0_target) / (std_t_detached + std_epsilon)).pow(2).mean()
            if use_x0_target
            else ((x_next - mean_pos) / (std_t_detached + std_epsilon)).pow(2).mean()
        )
        finite_frac = torch.isfinite(delta_E).float().mean()

        logit_mean = logit.mean()
        logit_std = logit.std()
        margin_mean = (-logit).mean()
        pref_acc = (logit < 0).float().mean()
        y_abs = y.abs()
        mask_strong = y_abs > 0.3
        pref_acc_strong = (
            (logit[mask_strong] < 0).float().mean()
            if mask_strong.any()
            else torch.tensor(0.0, device=x_t.device)
        )
        pref_acc_weighted = (
            ((logit < 0).float() * y_abs).sum() / (y_abs.sum() + 1e-8)
        )
        deltaE_pos_mean = (
            delta_E[y > 0].mean()
            if (y > 0).any()
            else torch.tensor(0.0, device=x_t.device)
        )
        deltaE_neg_mean = (
            delta_E[y < 0].mean()
            if (y < 0).any()
            else torch.tensor(0.0, device=x_t.device)
        )
        E_pos_mean = E_pos.mean()
        E_neg_mean = E_neg.mean()
        delta_E_mean = delta_E.mean()

        kl_raw = kl_per_traj.sum() / kl_valid.sum().clamp_min(1.0)
        kl_weighted = kl_beta * kl_raw
        kl_ratio = kl_weighted / (nft_loss + 1e-8)

    metrics_data = {
        "actor/nft_loss": nft_loss.detach(),
        "actor/kl_loss": kl_loss.detach(),
        "actor/total_loss": total_loss.detach(),
        "actor/adv_mean": adv_mean,
        "actor/adv_std": adv_std,
        "actor/adv_clip_frac": adv_clip_frac,
        "actor/r_mean": r_mean,
        "actor/r_std": r_std,
        "actor/y_abs_mean": y_abs_mean,
        "actor/y_sat_frac": y_sat_frac,
        "actor/delta_v_norm_mean": delta_v_norm.mean(),
        "actor/delta_v_clipped_norm_mean": delta_v_clipped_norm.mean(),
        "actor/clip_coef_mean": clip_coef_mean,
        "actor/clip_frac": clip_frac,
        "actor/std_mean": std_mean,
        "actor/std_min": std_min,
        "actor/std_max": std_max,
        "actor/z2_mean": z2_mean,
        "actor/finite_frac": finite_frac,
        "actor/logit_mean": logit_mean,
        "actor/logit_std": logit_std,
        "actor/margin_mean": margin_mean,
        "actor/pref_acc": pref_acc,
        "actor/pref_acc_strong": pref_acc_strong,
        "actor/pref_acc_weighted": pref_acc_weighted,
        "actor/deltaE_pos_mean": deltaE_pos_mean,
        "actor/deltaE_neg_mean": deltaE_neg_mean,
        "actor/E_pos_mean": E_pos_mean,
        "actor/E_neg_mean": E_neg_mean,
        "actor/delta_E_mean": delta_E_mean,
        "actor/kl_raw": kl_raw,
        "actor/kl_weighted": kl_weighted,
        "actor/kl_ratio": kl_ratio,
    }
    return total_loss, metrics_data


@register_policy_loss("nft-actor-critic")
def compute_nft_actor_critic_loss(**kwargs) -> tuple[torch.Tensor, dict]:
    values = kwargs.get("values", None)
    returns = kwargs.get("returns", None)
    loss_mask = kwargs.get("loss_mask", None)
    loss_mask_sum = kwargs.get("loss_mask_sum", None)

    actor_loss, actor_metrics = compute_nft_actor_loss(**kwargs)

    critic_loss = torch.tensor(0.0, device=actor_loss.device, dtype=actor_loss.dtype)
    critic_metrics: dict = {}
    prev_values = kwargs.get("prev_values", None)
    value_clip = kwargs.get("value_clip", None)
    huber_delta = kwargs.get("huber_delta", None)
    have_critic_inputs = (
        values is not None
        and returns is not None
        and prev_values is not None
        and value_clip is not None
        and huber_delta is not None
    )
    if have_critic_inputs:
        critic_loss, critic_metrics = compute_ppo_critic_loss(
            values=values,
            returns=returns,
            prev_values=prev_values,
            value_clip=value_clip,
            huber_delta=huber_delta,
            loss_mask=loss_mask,
            loss_mask_sum=loss_mask_sum,
        )

    total_loss = actor_loss + critic_loss
    metrics_data = {**actor_metrics, **critic_metrics}
    metrics_data["actor/total_loss"] = actor_loss.detach()
    if have_critic_inputs:
        metrics_data["critic/value_loss_total"] = critic_loss.detach()

    return total_loss, metrics_data


# ---------------------------------------------------------------------------
# Dual-Credit Implicit Chunk Reward helpers and loss
# ---------------------------------------------------------------------------


def compute_flow_sde_energy(
    v: torch.Tensor,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    t_cur: torch.Tensor,
    delta: torch.Tensor,
    sigma_i: torch.Tensor,
    std_epsilon: float = 1e-4,
) -> torch.Tensor:
    """Compute variance-normalized energy E = ||x_next - mu||^2 / var.

    Returns [B] energy per sample, summed over all non-batch dimensions.
    """
    ndim = x_t.ndim
    t_bc = t_cur
    delta_bc = delta
    while t_bc.ndim < ndim:
        t_bc = t_bc.unsqueeze(-1)
    while delta_bc.ndim < ndim:
        delta_bc = delta_bc.unsqueeze(-1)
    sig = sigma_i
    while sig.ndim < ndim:
        sig = sig.unsqueeze(-1)

    x0_pred = x_t - v * t_bc
    x1_pred = x_t + v * (1 - t_bc)
    x0_weight = torch.ones_like(t_bc) - (t_bc - delta_bc)
    x1_weight = t_bc - delta_bc - sig ** 2 * delta_bc / (2 * t_bc)
    mean = x0_pred * x0_weight + x1_pred * x1_weight

    std_t = torch.sqrt(delta_bc.clamp_min(0)) * sig
    var = std_t.detach() ** 2 + std_epsilon
    dims = tuple(range(1, ndim))
    return ((x_next - mean) ** 2 / var).sum(dim=dims)


def compute_chunk_gate(
    r_chunk: torch.Tensor,
    y_tau: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    tau_phi: float = 1.0,
    eta_min: float = 0.1,
    eta_max: float = 1.0,
    epsilon: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Robust normalisation of chunk rewards and outcome-consistency gate.

    Returns (gate, y_phi), both detached.
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(r_chunk, dtype=torch.bool)
    else:
        valid_mask = valid_mask.to(dtype=torch.bool, device=r_chunk.device)

    y_phi = torch.zeros_like(r_chunk)
    g = torch.zeros_like(r_chunk)

    if valid_mask.any():
        valid_rewards = r_chunk[valid_mask]
        med = valid_rewards.median()
        mad = (valid_rewards - med).abs().median()
        y_phi_valid = torch.tanh(
            (valid_rewards - med) / (tau_phi * (mad + epsilon))
        )
        g_valid = eta_min + (eta_max - eta_min) * (
            1.0 + y_tau[valid_mask] * y_phi_valid
        ) / 2.0
        y_phi[valid_mask] = y_phi_valid
        g[valid_mask] = g_valid
    return g.detach(), y_phi.detach()


def _prepare_schedule_params(schedule, step_indices, noise_level, x_t):
    """Extract t_cur, delta, sigma_i from schedule for given step indices."""
    idx = step_indices.long()
    t_cur = schedule[idx]
    t_next = schedule[idx + 1]
    delta = t_cur - t_next

    denom = schedule.clone()
    denom[0] = denom[1]
    sigma_base = torch.sqrt(schedule / (1 - denom))[:-1]
    sigma_i = sigma_base[idx]
    nl = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
    sigma_i = sigma_i * nl
    return t_cur, delta, sigma_i


_BLOCK_NFT_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("tr", (0, 1, 2)),
    ("rot", (3, 4, 5)),
    ("gr", (6,)),
)
_BLOCK_NFT_AXIS_NAMES: tuple[str, ...] = ("x", "y", "z", "rx", "ry", "rz", "gr")


def _safe_normalize_weights(weights: torch.Tensor, epsilon: float = 1.0e-6) -> torch.Tensor:
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(epsilon)
    return weights / denom


def _safe_weight_entropy(weights: torch.Tensor, epsilon: float = 1.0e-8) -> torch.Tensor:
    probs = weights.clamp_min(epsilon)
    return -(probs * probs.log()).sum(dim=-1)


def _clip_nft_delta_v(
    v_theta: torch.Tensor,
    v_old: torch.Tensor,
    max_drift: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    v_old_det = v_old.detach()
    delta_v = v_theta - v_old_det
    dims_v = tuple(range(2, delta_v.ndim))
    delta_norm = delta_v.norm(dim=dims_v, keepdim=True) + 1e-8
    clip_coef = (max_drift / delta_norm).clamp(max=1.0)
    delta_v_clipped = delta_v * clip_coef
    return v_old_det, delta_v_clipped, clip_coef


def _project_last_dim(tensor: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
    if len(indices) == 0:
        return torch.zeros_like(tensor)
    mask = torch.zeros(tensor.shape[-1], device=tensor.device, dtype=tensor.dtype)
    valid_indices = [idx for idx in indices if 0 <= idx < tensor.shape[-1]]
    if len(valid_indices) == 0:
        return torch.zeros_like(tensor)
    mask[valid_indices] = 1.0
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(0)
    return tensor * mask


def _compute_projected_delta_E(
    v_old_det: torch.Tensor,
    delta_v_clipped: torch.Tensor,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    schedule: torch.Tensor,
    step_indices: torch.Tensor,
    noise_level: Optional[torch.Tensor | float],
    beta: float,
    std_epsilon: float,
    indices: Sequence[int],
) -> torch.Tensor:
    projected_delta = _project_last_dim(delta_v_clipped, indices)
    v_pos = v_old_det + beta * projected_delta
    v_neg = v_old_det - beta * projected_delta
    t_cur, delta_t, sigma_i = _prepare_schedule_params(
        schedule, step_indices, noise_level, x_t
    )
    E_pos = compute_flow_sde_energy(
        v_pos, x_t, x_next, t_cur, delta_t, sigma_i, std_epsilon
    )
    E_neg = compute_flow_sde_energy(
        v_neg, x_t, x_next, t_cur, delta_t, sigma_i, std_epsilon
    )
    return E_pos - E_neg


def _combine_block_semantic_margin(
    delta_E_block: torch.Tensor,
    delta_E_axis: torch.Tensor,
    guidance_signal: torch.Tensor,
    tau_block: float,
    tau_axis: float,
    axis_mix: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tau_block = max(float(tau_block), 1.0e-6)
    tau_axis = max(float(tau_axis), 1.0e-6)
    axis_mix = float(min(max(axis_mix, 0.0), 1.0))
    guidance_signal = guidance_signal.reshape(delta_E_block.shape[0])

    block_logits = F.softplus(
        -(guidance_signal.unsqueeze(1) * delta_E_block) / tau_block
    )
    block_weights = _safe_normalize_weights(block_logits)
    axis_weights = torch.zeros_like(delta_E_axis)

    semantic_margin = torch.zeros_like(guidance_signal)
    for block_idx, (_, axis_indices) in enumerate(_BLOCK_NFT_GROUPS):
        axis_delta = delta_E_axis[:, list(axis_indices)]
        if len(axis_indices) == 1:
            axis_weight = torch.ones_like(axis_delta)
            axis_component = axis_delta[:, 0]
        else:
            axis_logits = F.softplus(
                -(guidance_signal.unsqueeze(1) * axis_delta) / tau_axis
            )
            axis_weight = _safe_normalize_weights(axis_logits)
            axis_component = (axis_weight * axis_delta).sum(dim=1)
        axis_weights[:, list(axis_indices)] = axis_weight
        block_component = (1.0 - axis_mix) * delta_E_block[:, block_idx] + axis_mix * axis_component
        semantic_margin = semantic_margin + block_weights[:, block_idx] * block_component

    return semantic_margin, block_weights, axis_weights


def _compute_block_guidance_signal(
    y_tau: torch.Tensor,
    y_phi: torch.Tensor,
    gate: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    mode = str(mode).lower()
    outcome_sign = torch.where(
        y_tau >= 0,
        torch.ones_like(y_tau),
        -torch.ones_like(y_tau),
    )
    if mode == "outcome":
        return outcome_sign
    if mode == "score":
        return gate * y_phi
    if mode == "blended":
        return 0.5 * outcome_sign + 0.5 * (gate * y_phi)
    raise ValueError(f"Unsupported block_nft_guidance_mode: {mode}")


@register_policy_loss("nft-dual-credit")
def compute_nft_dual_credit_loss(
    v_theta: torch.Tensor,
    v_old: torch.Tensor,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    schedule: torch.Tensor,
    advantages: torch.Tensor,
    step_indices: Optional[torch.Tensor] = None,
    noise_level: Optional[torch.Tensor | float] = None,
    total_denoise_steps: Optional[int] = None,
    loss_mask: Optional[torch.Tensor] = None,
    # Full K-step data for L_chunk
    v_theta_all: Optional[torch.Tensor] = None,
    v_old_all: Optional[torch.Tensor] = None,
    nft_xt_all: Optional[torch.Tensor] = None,
    nft_xnext_all: Optional[torch.Tensor] = None,
    # Chunk credit signals
    chunk_gate: Optional[torch.Tensor] = None,
    y_phi: Optional[torch.Tensor] = None,
    pair_guidance_good_xnext: Optional[torch.Tensor] = None,
    pair_guidance_bad_xnext: Optional[torch.Tensor] = None,
    pair_guidance_weight: Optional[torch.Tensor] = None,
    pair_guidance_mask: Optional[torch.Tensor] = None,
    pair_guidance_score_gap: Optional[torch.Tensor] = None,
    # Ablation switches
    enable_chunk_term: bool = True,
    enable_gate: bool = True,
    enable_block_nft: bool = False,
    enable_pair_guidance: bool = False,
    # Weights
    lambda_phi: float = 0.5,
    lambda_pair: float = 0.0,
    lambda_tr: float = 0.001,
    block_nft_axis_mix: float = 0.5,
    block_nft_tau_block: float = 1.0,
    block_nft_tau_axis: float = 1.0,
    block_nft_guidance_mode: str = "score",
    pair_guidance_kappa: float = 1.0,
    guidance_schedule_progress: float = 1.0,
    # Standard NFT params
    std_epsilon: float = 1e-4,
    beta: float = 1.0,
    adv_clip_max: float = 1.0,
    critic_warmup: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, dict]:
    """Dual-credit mirrored loss: L_traj + lambda_phi * L_chunk + lambda_tr * L_TR."""

    batch_size = x_t.shape[0]

    def _to_sample_mask(mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch_size, device=x_t.device, dtype=torch.bool)
        mask = mask.to(device=x_t.device)
        if mask.shape[0] != batch_size:
            raise ValueError(
                f"dual-credit loss expected loss_mask batch dim {batch_size}, got {mask.shape}"
            )
        while mask.ndim > 1:
            mask = mask.any(dim=-1)
        return mask.to(dtype=torch.bool)

    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return x[mask].mean()
        # Preserve the autograd graph even when this leaf/sub-batch has no
        # valid elements. Returning a fresh detached zero can make some ranks
        # skip backward while others still participate, which later shows up
        # as an FSDP/NCCL collective timeout.
        return x.sum() * 0.0

    def _masked_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.any():
            return x[mask].std(unbiased=False)
        return x.sum() * 0.0

    valid_mask = _to_sample_mask(loss_mask)

    # --- y_tau from advantages (binary ±1) ---
    y_tau = advantages[:, 0].sign() if advantages.ndim == 2 else advantages.sign()
    y_tau = y_tau.reshape(batch_size)

    # --- velocity delta and clipping (same as baseline NFT) ---
    max_drift = float(kwargs.get("max_drift", 0.5))
    v_old_det, delta_v_clipped, clip_coef = _clip_nft_delta_v(
        v_theta, v_old, max_drift
    )
    delta_v = v_theta - v_old_det
    dims_v = tuple(range(2, delta_v.ndim))
    v_pos = v_old_det + beta * delta_v_clipped
    v_neg = v_old_det - beta * delta_v_clipped

    # ===================== L_traj (single sampled step) =====================
    t_cur, delta_t, sigma_i = _prepare_schedule_params(
        schedule, step_indices, noise_level, x_t
    )
    E_pos_traj = compute_flow_sde_energy(
        v_pos, x_t, x_next, t_cur, delta_t, sigma_i, std_epsilon
    )
    E_neg_traj = compute_flow_sde_energy(
        v_neg, x_t, x_next, t_cur, delta_t, sigma_i, std_epsilon
    )
    delta_E_traj = E_pos_traj - E_neg_traj
    traj_softplus_input = 0.5 * y_tau * delta_E_traj
    L_traj = F.softplus(traj_softplus_input)
    traj_loss = _masked_mean(L_traj, valid_mask)

    # ===================== L_chunk (all K steps, averaged) =====================
    chunk_loss = torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
    delta_E_bar = torch.zeros(batch_size, device=x_t.device, dtype=x_t.dtype)
    delta_E_plain_bar = torch.zeros(batch_size, device=x_t.device, dtype=x_t.dtype)
    delta_E_sem = torch.zeros(batch_size, device=x_t.device, dtype=x_t.dtype)
    chunk_softplus_input = torch.zeros(batch_size, device=x_t.device, dtype=x_t.dtype)
    block_weights = torch.zeros(batch_size, len(_BLOCK_NFT_GROUPS), device=x_t.device, dtype=x_t.dtype)
    axis_weights = torch.zeros(batch_size, len(_BLOCK_NFT_AXIS_NAMES), device=x_t.device, dtype=x_t.dtype)
    block_guidance_signal_metric = torch.zeros(
        batch_size, device=x_t.device, dtype=x_t.dtype
    )
    delta_E_block_metric = torch.zeros(
        batch_size, len(_BLOCK_NFT_GROUPS), device=x_t.device, dtype=x_t.dtype
    )
    delta_E_axis_metric = torch.zeros(
        batch_size, len(_BLOCK_NFT_AXIS_NAMES), device=x_t.device, dtype=x_t.dtype
    )

    has_chunk_data = (
        enable_chunk_term
        and v_theta_all is not None
        and v_old_all is not None
        and nft_xt_all is not None
        and nft_xnext_all is not None
        and y_phi is not None
    )

    if has_chunk_data:
        K = nft_xt_all.shape[1]
        y_phi = y_phi.reshape(batch_size)
        g = (
            chunk_gate.reshape(batch_size)
            if (enable_gate and chunk_gate is not None)
            else torch.ones_like(y_phi)
        )
        delta_E_plain_sum = torch.zeros_like(delta_E_plain_bar)
        delta_E_block_sum = None
        delta_E_axis_sum = None
        for j in range(K):
            idx_j = torch.full((batch_size,), j, device=x_t.device, dtype=torch.long)
            chunk_size_j = v_old_all[:, j].shape[1]
            x_t_j = nft_xt_all[:, j, :chunk_size_j]
            x_next_j = nft_xnext_all[:, j, :chunk_size_j]
            t_j, delta_j, sigma_j = _prepare_schedule_params(
                schedule, idx_j, noise_level, x_t_j
            )
            v_old_j, dv_clipped_j, _ = _clip_nft_delta_v(
                v_theta_all[:, j], v_old_all[:, j], max_drift
            )
            v_pos_j = v_old_j + beta * dv_clipped_j
            v_neg_j = v_old_j - beta * dv_clipped_j

            E_pos_j = compute_flow_sde_energy(
                v_pos_j, x_t_j, x_next_j,
                t_j, delta_j, sigma_j, std_epsilon,
            )
            E_neg_j = compute_flow_sde_energy(
                v_neg_j, x_t_j, x_next_j,
                t_j, delta_j, sigma_j, std_epsilon,
            )
            delta_E_plain_sum = delta_E_plain_sum + (E_pos_j - E_neg_j)
            if enable_block_nft and v_theta_all.shape[-1] >= len(_BLOCK_NFT_AXIS_NAMES):
                delta_E_block_j = torch.stack(
                    [
                        _compute_projected_delta_E(
                            v_old_j,
                            dv_clipped_j,
                            x_t_j,
                            x_next_j,
                            schedule,
                            idx_j,
                            noise_level,
                            beta,
                            std_epsilon,
                            axis_indices,
                        )
                        for _, axis_indices in _BLOCK_NFT_GROUPS
                    ],
                    dim=1,
                )
                delta_E_axis_j = torch.stack(
                    [
                        _compute_projected_delta_E(
                            v_old_j,
                            dv_clipped_j,
                            x_t_j,
                            x_next_j,
                            schedule,
                            idx_j,
                            noise_level,
                            beta,
                            std_epsilon,
                            (axis_idx,),
                        )
                        for axis_idx in range(len(_BLOCK_NFT_AXIS_NAMES))
                    ],
                    dim=1,
                )
                if delta_E_block_sum is None:
                    delta_E_block_sum = delta_E_block_j
                    delta_E_axis_sum = delta_E_axis_j
                else:
                    delta_E_block_sum = delta_E_block_sum + delta_E_block_j
                    delta_E_axis_sum = delta_E_axis_sum + delta_E_axis_j

        delta_E_plain_bar = delta_E_plain_sum / float(K)
        delta_E_bar = delta_E_plain_bar
        if delta_E_block_sum is not None and delta_E_axis_sum is not None:
            delta_E_block = delta_E_block_sum / float(K)
            delta_E_axis = delta_E_axis_sum / float(K)
            delta_E_block_metric = delta_E_block
            delta_E_axis_metric = delta_E_axis
            block_guidance_signal = _compute_block_guidance_signal(
                y_tau, y_phi, g, block_nft_guidance_mode
            )
            block_guidance_signal_metric = block_guidance_signal
            delta_E_sem, block_weights, axis_weights = _combine_block_semantic_margin(
                delta_E_block,
                delta_E_axis,
                block_guidance_signal,
                block_nft_tau_block,
                block_nft_tau_axis,
                block_nft_axis_mix,
            )
            delta_E_bar = delta_E_sem

        chunk_softplus_input = 0.5 * y_phi * delta_E_bar
        L_chunk = g * F.softplus(chunk_softplus_input)
        chunk_loss = _masked_mean(L_chunk, valid_mask)

    # ===================== L_pair (local good-vs-bad suffix guidance) =====================
    pair_loss = torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
    pair_softplus_input = torch.zeros(batch_size, device=x_t.device, dtype=x_t.dtype)
    pair_weight_eff = torch.zeros(batch_size, device=x_t.device, dtype=x_t.dtype)
    pair_score_gap_eff = torch.zeros(batch_size, device=x_t.device, dtype=x_t.dtype)
    delta_E_pair_good = torch.zeros(batch_size, device=x_t.device, dtype=x_t.dtype)
    delta_E_pair_bad = torch.zeros(batch_size, device=x_t.device, dtype=x_t.dtype)
    pair_valid_mask = torch.zeros(batch_size, device=x_t.device, dtype=torch.bool)

    has_pair_data = (
        enable_pair_guidance
        and pair_guidance_good_xnext is not None
        and pair_guidance_bad_xnext is not None
        and pair_guidance_weight is not None
        and pair_guidance_mask is not None
    )
    if has_pair_data:
        pair_weight_eff = pair_guidance_weight.reshape(batch_size).to(
            device=x_t.device, dtype=x_t.dtype
        )
        pair_valid_mask = (
            valid_mask
            & pair_guidance_mask.reshape(batch_size).to(device=x_t.device, dtype=torch.bool)
            & (pair_weight_eff > 0)
        )
        if pair_guidance_score_gap is not None:
            pair_score_gap_eff = pair_guidance_score_gap.reshape(batch_size).to(
                device=x_t.device, dtype=x_t.dtype
            )

        pair_chunk_size = v_old.shape[1]
        x_next_good = pair_guidance_good_xnext[:, :pair_chunk_size].to(
            device=x_t.device, dtype=x_t.dtype
        )
        x_next_bad = pair_guidance_bad_xnext[:, :pair_chunk_size].to(
            device=x_t.device, dtype=x_t.dtype
        )

        if enable_block_nft and y_phi is not None and x_t.shape[-1] >= len(_BLOCK_NFT_AXIS_NAMES):
            g_pair = (
                chunk_gate.reshape(batch_size)
                if (enable_gate and chunk_gate is not None)
                else torch.ones(batch_size, device=x_t.device, dtype=x_t.dtype)
            )
            pair_guidance_signal = _compute_block_guidance_signal(
                y_tau,
                y_phi.reshape(batch_size),
                g_pair,
                block_nft_guidance_mode,
            )
            pair_block_good = torch.stack(
                [
                    _compute_projected_delta_E(
                        v_old_det,
                        delta_v_clipped,
                        x_t,
                        x_next_good,
                        schedule,
                        step_indices,
                        noise_level,
                        beta,
                        std_epsilon,
                        axis_indices,
                    )
                    for _, axis_indices in _BLOCK_NFT_GROUPS
                ],
                dim=1,
            )
            pair_axis_good = torch.stack(
                [
                    _compute_projected_delta_E(
                        v_old_det,
                        delta_v_clipped,
                        x_t,
                        x_next_good,
                        schedule,
                        step_indices,
                        noise_level,
                        beta,
                        std_epsilon,
                        (axis_idx,),
                    )
                    for axis_idx in range(len(_BLOCK_NFT_AXIS_NAMES))
                ],
                dim=1,
            )
            pair_block_bad = torch.stack(
                [
                    _compute_projected_delta_E(
                        v_old_det,
                        delta_v_clipped,
                        x_t,
                        x_next_bad,
                        schedule,
                        step_indices,
                        noise_level,
                        beta,
                        std_epsilon,
                        axis_indices,
                    )
                    for _, axis_indices in _BLOCK_NFT_GROUPS
                ],
                dim=1,
            )
            pair_axis_bad = torch.stack(
                [
                    _compute_projected_delta_E(
                        v_old_det,
                        delta_v_clipped,
                        x_t,
                        x_next_bad,
                        schedule,
                        step_indices,
                        noise_level,
                        beta,
                        std_epsilon,
                        (axis_idx,),
                    )
                    for axis_idx in range(len(_BLOCK_NFT_AXIS_NAMES))
                ],
                dim=1,
            )
            delta_E_pair_good, _, _ = _combine_block_semantic_margin(
                pair_block_good,
                pair_axis_good,
                pair_guidance_signal,
                block_nft_tau_block,
                block_nft_tau_axis,
                block_nft_axis_mix,
            )
            delta_E_pair_bad, _, _ = _combine_block_semantic_margin(
                pair_block_bad,
                pair_axis_bad,
                pair_guidance_signal,
                block_nft_tau_block,
                block_nft_tau_axis,
                block_nft_axis_mix,
            )
        else:
            t_pair, delta_pair, sigma_pair = _prepare_schedule_params(
                schedule, step_indices, noise_level, x_t
            )
            E_good_pos = compute_flow_sde_energy(
                v_pos, x_t, x_next_good, t_pair, delta_pair, sigma_pair, std_epsilon
            )
            E_good_neg = compute_flow_sde_energy(
                v_neg, x_t, x_next_good, t_pair, delta_pair, sigma_pair, std_epsilon
            )
            E_bad_pos = compute_flow_sde_energy(
                v_pos, x_t, x_next_bad, t_pair, delta_pair, sigma_pair, std_epsilon
            )
            E_bad_neg = compute_flow_sde_energy(
                v_neg, x_t, x_next_bad, t_pair, delta_pair, sigma_pair, std_epsilon
            )
            delta_E_pair_good = E_good_pos - E_good_neg
            delta_E_pair_bad = E_bad_pos - E_bad_neg

        pair_softplus_input = 0.5 * pair_guidance_kappa * (
            delta_E_pair_good - delta_E_pair_bad
        )
        pair_loss_raw = pair_weight_eff * F.softplus(pair_softplus_input)
        pair_loss = _masked_mean(pair_loss_raw, pair_valid_mask)

    # ===================== L_TR (trust-region KL) =====================
    kl_dims = tuple(range(2, v_theta.ndim))
    kl_loss_per_sample = (delta_v ** 2).mean(dim=kl_dims)
    kl_loss = _masked_mean(kl_loss_per_sample, valid_mask)

    # ===================== Total =====================
    total_loss = (
        traj_loss
        + lambda_phi * chunk_loss
        + lambda_pair * pair_loss
        + lambda_tr * kl_loss
    )
    if critic_warmup:
        # Preserve the computation graph while zeroing gradients during warmup.
        total_loss = total_loss * 0.0
    # Keep a zero-gradient anchor to the current forward graph. This makes
    # distributed backward structurally consistent even when every valid term
    # in a leaf/sub-batch is masked out.
    if not total_loss.requires_grad:
        total_loss = total_loss + v_theta.sum() * 0.0

    with torch.no_grad():
        ln2 = torch.tensor(math.log(2.0), device=x_t.device, dtype=x_t.dtype)
        gate_mean = (
            _masked_mean(chunk_gate.reshape(batch_size), valid_mask)
            if chunk_gate is not None
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        gate_std = (
            _masked_std(chunk_gate.reshape(batch_size), valid_mask)
            if chunk_gate is not None
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        y_phi_mean = (
            _masked_mean(y_phi.reshape(batch_size), valid_mask)
            if y_phi is not None
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        y_phi_std = (
            _masked_std(y_phi.reshape(batch_size), valid_mask)
            if y_phi is not None
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        gate_consistency = (
            (y_phi.reshape(batch_size)[valid_mask].sign() == y_tau[valid_mask].sign())
            .float()
            .mean()
            if y_phi is not None and valid_mask.any()
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        traj_softplus_input_std = _masked_std(traj_softplus_input, valid_mask)
        traj_softplus_input_pos_frac = (
            (traj_softplus_input[valid_mask] > 0).float().mean()
            if valid_mask.any()
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        chunk_softplus_input_mean = (
            _masked_mean(chunk_softplus_input, valid_mask)
            if y_phi is not None
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        chunk_softplus_input_std = (
            _masked_std(chunk_softplus_input, valid_mask)
            if y_phi is not None
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        chunk_softplus_input_pos_frac = (
            (chunk_softplus_input[valid_mask] > 0).float().mean()
            if y_phi is not None and valid_mask.any()
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        chunk_softplus_input_abs_mean = (
            _masked_mean(chunk_softplus_input.abs(), valid_mask)
            if y_phi is not None
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        chunk_neutral_loss = (
            _masked_mean(g * torch.full_like(g, float(math.log(2.0))), valid_mask)
            if has_chunk_data
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        pair_valid_frac = pair_valid_mask.float().mean()
        pair_weight_mean = (
            _masked_mean(pair_weight_eff, pair_valid_mask)
            if pair_valid_mask.any()
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        pair_score_gap_mean = (
            _masked_mean(pair_score_gap_eff, pair_valid_mask)
            if pair_valid_mask.any()
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        pair_softplus_input_mean = (
            _masked_mean(pair_softplus_input, pair_valid_mask)
            if pair_valid_mask.any()
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        block_weight_mean = (
            block_weights[valid_mask].mean(dim=0)
            if valid_mask.any() and block_weights.numel() > 0
            else torch.zeros(len(_BLOCK_NFT_GROUPS), device=x_t.device, dtype=x_t.dtype)
        )
        block_weight_entropy_mean = (
            _masked_mean(_safe_weight_entropy(block_weights), valid_mask)
            if valid_mask.any() and block_weights.numel() > 0
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        block_weight_max_mean = (
            _masked_mean(block_weights.max(dim=1).values, valid_mask)
            if valid_mask.any() and block_weights.numel() > 0
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        block_delta_std_mean = (
            _masked_mean(delta_E_block_metric.std(dim=1), valid_mask)
            if valid_mask.any() and delta_E_block_metric.numel() > 0
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        axis_weight_mean = (
            axis_weights[valid_mask].mean(dim=0)
            if valid_mask.any() and axis_weights.numel() > 0
            else torch.zeros(len(_BLOCK_NFT_AXIS_NAMES), device=x_t.device, dtype=x_t.dtype)
        )
        axis_weight_entropy_mean = (
            _masked_mean(_safe_weight_entropy(axis_weights[:, :3]), valid_mask)
            if valid_mask.any() and axis_weights.numel() > 0
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )
        axis_delta_std_mean = (
            _masked_mean(delta_E_axis_metric.std(dim=1), valid_mask)
            if valid_mask.any() and delta_E_axis_metric.numel() > 0
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        )

    metrics_data = {
        "actor/traj_loss": traj_loss.detach(),
        "actor/chunk_loss": chunk_loss.detach(),
        "actor/pair_loss": pair_loss.detach(),
        "actor/kl_loss": kl_loss.detach(),
        "actor/total_loss": total_loss.detach(),
        "actor/delta_E_traj_mean": _masked_mean(delta_E_traj, valid_mask).detach(),
        "actor/delta_E_bar_mean": _masked_mean(delta_E_bar, valid_mask).detach(),
        "actor/delta_E_plain_bar_mean": _masked_mean(
            delta_E_plain_bar, valid_mask
        ).detach(),
        "actor/delta_E_sem_mean": _masked_mean(delta_E_sem, valid_mask).detach(),
        "actor/block_guidance_abs_mean": _masked_mean(
            block_guidance_signal_metric.abs(), valid_mask
        ).detach(),
        "actor/block_weight_entropy_mean": block_weight_entropy_mean.detach(),
        "actor/block_weight_max_mean": block_weight_max_mean.detach(),
        "actor/block_delta_std_mean": block_delta_std_mean.detach(),
        "actor/axis_weight_entropy_mean": axis_weight_entropy_mean.detach(),
        "actor/axis_delta_std_mean": axis_delta_std_mean.detach(),
        "actor/traj_softplus_input_mean": _masked_mean(
            traj_softplus_input, valid_mask
        ).detach(),
        "actor/traj_softplus_input_std": traj_softplus_input_std.detach(),
        "actor/traj_softplus_input_pos_frac": traj_softplus_input_pos_frac.detach(),
        "actor/traj_signed_margin_mean": _masked_mean(
            y_tau * delta_E_traj, valid_mask
        ).detach(),
        "actor/traj_loss_minus_ln2": (traj_loss - ln2).detach(),
        "actor/chunk_softplus_input_mean": chunk_softplus_input_mean.detach(),
        "actor/chunk_softplus_input_std": chunk_softplus_input_std.detach(),
        "actor/chunk_softplus_input_pos_frac": chunk_softplus_input_pos_frac.detach(),
        "actor/chunk_softplus_input_abs_mean": chunk_softplus_input_abs_mean.detach(),
        "actor/chunk_signed_margin_mean": (
            _masked_mean(y_phi.reshape(batch_size) * delta_E_bar, valid_mask).detach()
            if y_phi is not None
            else torch.tensor(0.0, device=x_t.device, dtype=x_t.dtype)
        ),
        "actor/chunk_neutral_loss": chunk_neutral_loss.detach(),
        "actor/chunk_loss_minus_neutral": (chunk_loss - chunk_neutral_loss).detach(),
        "actor/pair_valid_frac": pair_valid_frac.detach(),
        "actor/pair_weight_mean": pair_weight_mean.detach(),
        "actor/pair_score_gap_mean": pair_score_gap_mean.detach(),
        "actor/pair_softplus_input_mean": pair_softplus_input_mean.detach(),
        "actor/pair_delta_good_mean": _masked_mean(
            delta_E_pair_good, pair_valid_mask
        ).detach(),
        "actor/pair_delta_bad_mean": _masked_mean(
            delta_E_pair_bad, pair_valid_mask
        ).detach(),
        "actor/lambda_phi_eff": torch.tensor(
            float(lambda_phi), device=x_t.device, dtype=x_t.dtype
        ),
        "actor/lambda_pair_eff": torch.tensor(
            float(lambda_pair), device=x_t.device, dtype=x_t.dtype
        ),
        "actor/guidance_schedule_progress": torch.tensor(
            float(guidance_schedule_progress), device=x_t.device, dtype=x_t.dtype
        ),
        "actor/block_tr_weight_mean": block_weight_mean[0].detach(),
        "actor/block_rot_weight_mean": block_weight_mean[1].detach(),
        "actor/block_gr_weight_mean": block_weight_mean[2].detach(),
        "actor/axis_x_weight_mean": axis_weight_mean[0].detach(),
        "actor/axis_y_weight_mean": axis_weight_mean[1].detach(),
        "actor/axis_z_weight_mean": axis_weight_mean[2].detach(),
        "actor/axis_rx_weight_mean": axis_weight_mean[3].detach(),
        "actor/axis_ry_weight_mean": axis_weight_mean[4].detach(),
        "actor/axis_rz_weight_mean": axis_weight_mean[5].detach(),
        "actor/axis_gr_weight_mean": axis_weight_mean[6].detach(),
        "actor/gate_mean": gate_mean,
        "actor/gate_std": gate_std,
        "actor/y_phi_mean": y_phi_mean,
        "actor/y_phi_std": y_phi_std,
        "actor/gate_consistency": gate_consistency,
        "actor/clip_coef_mean": _masked_mean(clip_coef, valid_mask).detach(),
    }
    return total_loss, metrics_data
