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

from enum import Enum


class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    LIBERO = "libero"


def get_env_cls(env_type: str, env_cfg=None, enable_offload=False):
    """Resolve a supported PAWN environment without importing other simulators."""
    env_type = SupportedEnvType(env_type)
    if env_type == SupportedEnvType.MANISKILL:
        if enable_offload:
            from rlinf.envs.maniskill.maniskill_offload_env import (
                ManiskillOffloadEnv as ManiskillEnv,
            )
        else:
            from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
        return ManiskillEnv

    from rlinf.envs.libero.libero_env import LiberoEnv

    return LiberoEnv
