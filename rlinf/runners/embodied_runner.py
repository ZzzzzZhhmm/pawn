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

import json
import os
import time
from typing import TYPE_CHECKING, Optional, Union

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.data.replay_buffer import SACReplayBuffer
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor", "EmbodiedSACFSDPPolicy", "AsyncEmbodiedSACFSDPPolicy"
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        demo_buffer: Optional[SACReplayBuffer] = None,
        critic=None,
        reward=None,
        run_timer=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.demo_buffer = demo_buffer
        self.critic = critic
        self.reward = reward

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.demo_buffer is not None:
            self.demo_data_channel = Channel.create("DemoBufferChannel")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.metric_logger = MetricLogger(cfg)
        self.metrics_jsonl_path = os.path.join(
            self.metric_logger.log_path, "metrics_full.jsonl"
        )
        self.runner_events_path = os.path.join(
            self.metric_logger.log_path, "runner_events.log"
        )

    def _append_jsonl_record(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.metrics_jsonl_path), exist_ok=True)
        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def _append_runner_event(self, message: str) -> None:
        os.makedirs(os.path.dirname(self.runner_events_path), exist_ok=True)
        with open(self.runner_events_path, "a", encoding="utf-8") as fp:
            fp.write(message.rstrip("\n") + "\n")

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        self.actor.init_worker().wait()
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()
        self.env.set_max_steps(self.max_steps).wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def send_demo_buffer(self):
        if self.demo_buffer is not None:
            sub_demo_buffer_ls = self.demo_buffer.split_to_dict(self.actor._world_size)

            for sub_demo_buffer in sub_demo_buffer_ls:
                self.demo_data_channel.put(sub_demo_buffer, async_op=True)
            self.actor.recv_demo_data(self.demo_data_channel).wait()

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=5000,
        )
        self.send_demo_buffer()
        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            self.env.set_global_step(self.global_step)
            runner_msg = f"[runner] step={self.global_step} start sync_weights"
            print(runner_msg, flush=True)
            self._append_runner_event(runner_msg)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    self.update_rollout_weights()
                runner_msg = (
                    f"[runner] step={self.global_step} start generate_rollouts"
                )
                print(runner_msg, flush=True)
                self._append_runner_event(runner_msg)
                with self.timer("generate_rollouts"):
                    env_handle: Handle = self.env.interact(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.env_channel,
                        output_channel=self.rollout_channel,
                        actor_channel=self.actor_channel,
                    )
                    self.actor.recv_rollout_batch(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()
                runner_msg = (
                    f"[runner] step={self.global_step} rollout finished; "
                    f"start cal_adv_and_returns"
                )
                print(runner_msg, flush=True)
                self._append_runner_event(runner_msg)

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )
                runner_msg = (
                    f"[runner] step={self.global_step} advantages ready; "
                    f"prepare branch_rollouts/actor_training"
                )
                print(runner_msg, flush=True)
                self._append_runner_event(runner_msg)

                if self.cfg.algorithm.get("enable_branch_rollout", False):
                    with self.timer("plan_branch_rollouts"):
                        branch_plan_lists = self.actor.plan_branch_rollouts().wait()
                    branch_plans = []
                    for rank_plans in branch_plan_lists:
                        if rank_plans:
                            branch_plans.extend(rank_plans)
                    if branch_plans:
                        runner_msg = (
                            f"[runner] step={self.global_step} start branch_rollouts "
                            f"plans={len(branch_plans)}"
                        )
                        print(runner_msg, flush=True)
                        self._append_runner_event(runner_msg)
                        with self.timer("generate_branch_rollouts"):
                            branch_result_lists = self.rollout.generate_branch_rollouts(
                                branch_plans
                            ).wait()
                        branch_results_by_rank = {}
                        for worker_results in branch_result_lists:
                            if not worker_results:
                                continue
                            for item in worker_results:
                                actor_rank = int(item.get("actor_rank", -1))
                                if actor_rank < 0:
                                    continue
                                branch_results_by_rank.setdefault(actor_rank, []).append(
                                    item
                                )
                        with self.timer("load_branch_rollouts"):
                            self.actor.load_branch_results(branch_results_by_rank).wait()
                        runner_msg = (
                            f"[runner] step={self.global_step} branch_rollouts done "
                            f"results={sum(len(v) for v in branch_results_by_rank.values())}"
                        )
                        print(runner_msg, flush=True)
                        self._append_runner_event(runner_msg)

                # actor training.
                with self.timer("actor_training"):
                    actor_training_start = time.perf_counter()
                    actor_training_metrics = self.actor.run_training().wait()
                    actor_training_elapsed = time.perf_counter() - actor_training_start
                runner_msg = (
                    f"[runner] step={self.global_step} actor_training finished "
                    f"elapsed={actor_training_elapsed:.1f}s"
                )
                print(runner_msg, flush=True)
                self._append_runner_event(runner_msg)

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                eval_metrics = {}
                if run_val:
                    with self.timer("eval"):
                        self.update_rollout_weights()
                        eval_metrics = self.evaluate()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

            env_results_list = [
                results for results in env_handle.wait() if results is not None
            ]
            env_metrics = compute_evaluate_metrics(env_results_list)
            env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}

            rollout_metrics = {
                f"rollout/{k}": v for k, v in actor_rollout_metrics[0].items()
            }

            training_metrics = {
                f"train/{k}": v for k, v in actor_training_metrics[0].items()
            }

            self.metric_logger.log(env_metrics, _step)
            self.metric_logger.log(rollout_metrics, _step)
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(eval_metrics)
            logging_metrics.update(env_metrics)
            logging_metrics.update(rollout_metrics)
            logging_metrics.update(training_metrics)

            def _to_jsonable(val):
                try:
                    if hasattr(val, "item"):
                        return val.item()
                except Exception:
                    pass
                return val

            full_metrics_record = {
                "step": int(_step + 1),
                **{k: _to_jsonable(v) for k, v in logging_metrics.items()},
            }
            self._append_jsonl_record(full_metrics_record)

            metrics_summary = {
                "step": int(_step + 1),
                "time/step": _to_jsonable(logging_metrics.get("time/step", None)),
                "time/generate_rollouts": _to_jsonable(
                    logging_metrics.get("time/generate_rollouts", None)
                ),
                "time/actor_training": _to_jsonable(
                    logging_metrics.get("time/actor_training", None)
                ),
                "env/success_once": _to_jsonable(
                    logging_metrics.get("env/success_once", None)
                ),
                "env/return": _to_jsonable(logging_metrics.get("env/return", None)),
                "train/loss": _to_jsonable(logging_metrics.get("train/loss", None)),
                "train/actor/traj_loss": _to_jsonable(
                    logging_metrics.get("train/actor/traj_loss", None)
                ),
                "train/actor/chunk_loss": _to_jsonable(
                    logging_metrics.get("train/actor/chunk_loss", None)
                ),
                "train/prm/loss": _to_jsonable(
                    logging_metrics.get("train/prm/loss", None)
                ),
            }
            print(
                "[metrics-summary] "
                + json.dumps(metrics_summary, ensure_ascii=True, sort_keys=True),
                flush=True,
            )

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

        self.metric_logger.finish()

    def _save_checkpoint(self):
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
