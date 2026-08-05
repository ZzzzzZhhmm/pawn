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
import typing

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics

if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self):
        self.rollout.init_worker().wait()
        self.env.init_worker().wait()

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
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        def _to_jsonable(value):
            if hasattr(value, "tolist"):
                return value.tolist()
            if isinstance(value, dict):
                return {k: _to_jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_jsonable(v) for v in value]
            return value

        log_path = self.cfg.runner.logger.log_path
        os.makedirs(log_path, exist_ok=True)
        metrics_path = os.path.join(log_path, "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(eval_metrics), f, indent=2, ensure_ascii=False)

        self.metric_logger.finish()
