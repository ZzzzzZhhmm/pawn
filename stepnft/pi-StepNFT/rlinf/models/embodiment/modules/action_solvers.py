import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch

try:
    from diffusers.utils.torch_utils import randn_tensor as _randn_tensor
except ImportError:  # pragma: no cover

    def _randn_tensor(
        shape,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device=device, dtype=dtype)


@dataclass
class ActionSolverStepResult:
    """Container describing the outcome of a solver step."""

    sample: Optional[torch.Tensor] = None  # sampled next latent/state
    mean: Optional[torch.Tensor] = None  # conditional mean of the next state
    std: Optional[torch.Tensor] = None  # conditional std/scale for the transition
    log_prob: Optional[torch.Tensor] = None  # log-probability of the sampled point
    state: dict[str, Any] = field(default_factory=dict)  # mutable solver-specific cache


def _expand_step_index(
    idx: int | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalize scalar/tensor indices to a tensor defined on the right device."""

    if isinstance(idx, int):
        return torch.full((batch_size,), idx, device=device, dtype=torch.long)
    return idx.to(device=device, dtype=torch.long)


@dataclass
class _RectifiedFlowTerms:
    """Small struct caching broadcasted temporal parameters for rectified-flow updates."""

    schedule: torch.Tensor
    idx_tensor: torch.Tensor
    t_bc: torch.Tensor
    delta_bc: torch.Tensor
    dt_bc: torch.Tensor


def _prepare_rectified_terms(
    x_t: torch.Tensor,
    idx: int | torch.Tensor,
    timesteps: torch.Tensor,
) -> _RectifiedFlowTerms:
    """Pre-compute tensors shared by rectified-flow solvers (broadcasted timestamps/deltas)."""

    device = x_t.device
    schedule = timesteps.to(device=device, dtype=x_t.dtype)
    idx_tensor = _expand_step_index(idx, x_t.shape[0], device)
    t_cur = schedule[idx_tensor]
    t_next = schedule[idx_tensor + 1]
    delta = t_cur - t_next
    dt = t_next - t_cur
    return _RectifiedFlowTerms(
        schedule=schedule,
        idx_tensor=idx_tensor,
        t_bc=t_cur[:, None, None].expand_as(x_t),
        delta_bc=delta[:, None, None].expand_as(x_t),
        dt_bc=dt[:, None, None].expand_as(x_t),
    )


def _rectified_predictions(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    t_bc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x0, x1) predictions given velocity and broadcasted timestamps."""

    x0_pred = x_t - velocity * t_bc
    x1_pred = x_t + velocity * (1 - t_bc)
    return x0_pred, x1_pred


def _expand_dt(dt: float | torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar dt to match the action tensor."""

    if not torch.is_tensor(dt):
        dt = torch.as_tensor(dt, device=reference.device, dtype=reference.dtype)
    while dt.ndim < reference.ndim:
        dt = dt.unsqueeze(-1)
    return dt.expand_as(reference)


def euler_step(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    dt: float | torch.Tensor,
    state: Optional[dict[str, Any]] = None,
) -> ActionSolverStepResult:
    """First-order Euler solver."""

    dt_tensor = _expand_dt(dt, x_t)
    sample = x_t + velocity * dt_tensor
    return ActionSolverStepResult(sample=sample, state=state or {})


def flow_sde_step(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    idx: int | torch.Tensor,
    timesteps: torch.Tensor,
    noise_level: float | torch.Tensor,
) -> ActionSolverStepResult:
    terms = _prepare_rectified_terms(x_t, idx, timesteps)
    x0_pred, x1_pred = _rectified_predictions(x_t, velocity, terms.t_bc)
    noise_level_tensor = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
    sigmas = (
        noise_level_tensor
        * torch.sqrt(
            terms.schedule
            / (1 - torch.where(terms.schedule == 1, terms.schedule[1], terms.schedule))
        )[:-1]
    )
    sigma_i = sigmas[terms.idx_tensor][:, None, None].expand_as(x_t)
    x0_weight = torch.ones_like(terms.t_bc) - (terms.t_bc - terms.delta_bc)
    x1_weight = terms.t_bc - terms.delta_bc - sigma_i**2 * terms.delta_bc / (2 * terms.t_bc)
    std = torch.sqrt(terms.delta_bc) * sigma_i
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    return ActionSolverStepResult(mean=mean, std=std)


def flow_noise_step(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    idx: int | torch.Tensor,
    timesteps: torch.Tensor,
    noise_head,
    suffix_out: torch.Tensor,
) -> ActionSolverStepResult:
    terms = _prepare_rectified_terms(x_t, idx, timesteps)
    x0_pred, x1_pred = _rectified_predictions(x_t, velocity, terms.t_bc)
    x0_weight = 1 - (terms.t_bc - terms.delta_bc)
    x1_weight = terms.t_bc - terms.delta_bc
    std = noise_head(suffix_out)
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    return ActionSolverStepResult(mean=mean, std=std)


def flow_cps_step(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    idx: int | torch.Tensor,
    timesteps: torch.Tensor,
    noise_level: float | torch.Tensor,
) -> ActionSolverStepResult:
    terms = _prepare_rectified_terms(x_t, idx, timesteps)
    x0_pred, x1_pred = _rectified_predictions(x_t, velocity, terms.t_bc)
    noise_level_tensor = torch.as_tensor(noise_level, device=x_t.device, dtype=x_t.dtype)
    pi = torch.tensor(math.pi, device=x_t.device, dtype=x_t.dtype)
    cos_term = torch.cos(pi * noise_level_tensor / 2)
    sin_term = torch.sin(pi * noise_level_tensor / 2)
    x0_weight = torch.ones_like(terms.t_bc) - (terms.t_bc - terms.delta_bc)
    x1_weight = (terms.t_bc - terms.delta_bc) * cos_term
    std = (terms.t_bc - terms.delta_bc) * sin_term
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    return ActionSolverStepResult(mean=mean, std=std)


def flow_grpo_step(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    idx: int,
    timesteps: torch.Tensor,
    eta: float,
    sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> ActionSolverStepResult:
    device = x_t.device
    sigma = timesteps[idx].to(device)
    sigma_prev = timesteps[idx + 1].to(device)
    sigma_max = timesteps[1].item()
    dt = sigma_prev - sigma

    std_dev_t = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma))) * eta
    mean = (
        x_t * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + velocity * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )
    std = std_dev_t * torch.sqrt(-1 * dt)

    if sample is None:
        noise = _randn_tensor(
            velocity.shape,
            generator=generator,
            device=device,
            dtype=velocity.dtype,
        )
        sample = mean + std * noise

    log_prob = (
        -((sample.detach() - mean) ** 2) / (2 * (std**2))
        - torch.log(std)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=device)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return ActionSolverStepResult(sample=sample, mean=mean, std=std, log_prob=log_prob)


def dance_step(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    idx: int,
    timesteps: torch.Tensor,
    eta: float,
    sample: Optional[torch.Tensor] = None,
) -> ActionSolverStepResult:
    sigma = timesteps[idx]
    dsigma = timesteps[idx + 1] - sigma
    mean = x_t + dsigma * velocity

    pred_original = x_t - sigma * velocity
    delta_t = sigma - timesteps[idx + 1]
    std_dev_t = eta * math.sqrt(delta_t)

    score_estimate = -(x_t - pred_original * (1 - sigma)) / sigma**2
    log_term = -0.5 * eta**2 * score_estimate
    mean = mean + log_term * dsigma

    if sample is None:
        sample = mean + torch.randn_like(mean) * std_dev_t

    log_prob = -((sample.detach().to(torch.float32) - mean.to(torch.float32)) ** 2) / (
        2 * (std_dev_t**2)
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    std = torch.full_like(mean, std_dev_t, dtype=mean.dtype)
    return ActionSolverStepResult(sample=sample, mean=mean, std=std, log_prob=log_prob)


def _velocity_to_x0_pred(
    velocity: torch.Tensor,
    x_t: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    sigma_t = sigmas[step_index]
    return x_t - sigma_t * velocity


def ddim_step(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    idx: int,
    timesteps: torch.Tensor,
    eta: float,
    noise: Optional[torch.Tensor] = None,
) -> ActionSolverStepResult:
    x0_pred = _velocity_to_x0_pred(velocity, x_t, timesteps, step_index=idx)
    prev_sample, mean, std_dev_t, dt_sqrt = ddim_update(
        x0_pred=x0_pred,
        sigmas=timesteps.to(torch.float64),
        step_index=idx,
        sample=x_t,
        noise=noise,
        eta=eta,
    )
    log_prob = (
        -((prev_sample.detach() - mean) ** 2) / (2 * ((std_dev_t * dt_sqrt) ** 2))
        - torch.log(std_dev_t * dt_sqrt)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi, device=x_t.device)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return ActionSolverStepResult(
        sample=prev_sample,
        mean=mean,
        std=std_dev_t * dt_sqrt,
        log_prob=log_prob,
    )


@dataclass
class DPMState:
    """Keeps track of historical model outputs for multi-step DPM solvers."""

    order: int
    x0_preds: List[torch.Tensor] = field(default_factory=list)
    lower_order_nums: int = 0

    def __post_init__(self):
        self.x0_preds = [None] * self.order

    def update(self, x0_pred: torch.Tensor):
        for i in range(self.order - 1):
            self.x0_preds[i] = self.x0_preds[i + 1]
        self.x0_preds[-1] = x0_pred

    def update_lower_order(self):
        if self.lower_order_nums < self.order:
            self.lower_order_nums += 1


def dpm_step(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    idx: int,
    timesteps: torch.Tensor,
    order: int,
    dpm_state: DPMState,
) -> ActionSolverStepResult:
    n_steps = timesteps.shape[0] - 1
    step_list = list(range(n_steps))
    lower_order_final = idx == len(step_list) - 1
    lower_order_second = (idx == len(step_list) - 2) and len(step_list) < 15

    x0_pred = _velocity_to_x0_pred(velocity, x_t, timesteps, step_index=idx)
    dpm_state.update(x0_pred)

    sample_fp32 = x_t.to(torch.float32)

    if order == 1 or dpm_state.lower_order_nums < 1 or lower_order_final:
        if idx == 0 or lower_order_final:
            prev_sample, _, _, _ = ddim_update(
                x0_pred,
                timesteps.to(torch.float64),
                idx,
                sample_fp32,
                eta=0.0,
            )
        else:
            prev_sample = dpm_solver_first_order_update(
                x0_pred,
                timesteps.to(torch.float64),
                idx,
                sample_fp32,
            )
    elif order == 2 or dpm_state.lower_order_nums < 2 or lower_order_second:
        prev_sample = multistep_dpm_solver_second_order_update(
            dpm_state.x0_preds,
            timesteps.to(torch.float64),
            idx,
            sample_fp32,
        )
    else:
        raise ValueError("Unsupported DPM order.")

    dpm_state.update_lower_order()
    prev_sample = prev_sample.to(x0_pred.dtype)
    return ActionSolverStepResult(sample=prev_sample, state={"dpm_state": dpm_state})


def ddim_update(
    x0_pred: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    sample: Optional[torch.Tensor] = None,
    noise: Optional[torch.Tensor] = None,
    eta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    t, s = sigmas[step_index + 1], sigmas[step_index]
    std_dev_t = eta * t
    dt_sqrt = torch.sqrt(1.0 - t**2 * (1 - s) ** 2 / (s**2 * (1 - t) ** 2))
    rho_t = std_dev_t * dt_sqrt
    noise_pred = (sample - (1 - s) * x0_pred) / s
    if noise is None:
        noise = torch.randn_like(x0_pred)
    mean = (1 - t) * x0_pred + torch.sqrt(t**2 - rho_t**2) * noise_pred
    prev_sample = mean + rho_t * noise
    return prev_sample, mean, std_dev_t, dt_sqrt


def dpm_solver_first_order_update(
    x0_pred: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    sample: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sigma_t, sigma_s = sigmas[step_index + 1], sigmas[step_index]
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = _sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

    h = lambda_t - lambda_s
    return (sigma_t / sigma_s) * sample - (alpha_t * (torch.exp(-h) - 1.0)) * x0_pred


def multistep_dpm_solver_second_order_update(
    x0_pred_list: List[torch.Tensor],
    sigmas: torch.Tensor,
    step_index: int,
    sample: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sigma_t, sigma_s0, sigma_s1 = (
        sigmas[step_index + 1],
        sigmas[step_index],
        sigmas[step_index - 1],
    )
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = _sigma_to_alpha_sigma_t(sigma_s1)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)
    m0, m1 = x0_pred_list[-1], x0_pred_list[-2]

    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    D0, D1 = m0, (1.0 / r0) * (m0 - m1)
    return (
        (sigma_t / sigma_s0) * sample
        - (alpha_t * (torch.exp(-h) - 1.0)) * D0
        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * D1
    )


def _sigma_to_alpha_sigma_t(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_t = 1 - sigma
    sigma_t = sigma
    return alpha_t, sigma_t


__all__ = [
    "ActionSolverStepResult",
    "euler_step",
    "flow_sde_step",
    "flow_noise_step",
    "flow_cps_step",
    "flow_grpo_step",
    "dance_step",
    "ddim_step",
    "dpm_step",
    "DPMState",
]

