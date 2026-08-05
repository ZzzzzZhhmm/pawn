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
        policy_loss = torch.tensor(0.0, device=policy_loss.device)

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
        total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype)

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
