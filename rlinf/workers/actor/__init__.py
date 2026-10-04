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

from omegaconf import DictConfig

from rlinf.scheduler.worker.worker import Worker


def get_actor_worker(cfg: DictConfig) -> type[Worker]:
    if cfg.actor.training_backend != "fsdp":
        raise ValueError(f"Unsupported training backend: {cfg.actor.training_backend}")
    from .fsdp_actor_worker import EmbodiedFSDPActor
    return EmbodiedFSDPActor
