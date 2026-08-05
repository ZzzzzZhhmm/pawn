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

import os
import tempfile
import warnings

from omegaconf import DictConfig, OmegaConf, open_dict


class _TensorboardLogger:
    def __init__(self, log_path):
        from torch.utils.tensorboard import SummaryWriter

        self.log_path = log_path
        self.writer = SummaryWriter(log_path)
        self._disabled = False
        self._warned = False

    def _disable(self, exc: Exception, operation: str) -> None:
        if not self._warned:
            warnings.warn(
                "[MetricLogger] tensorboard logging disabled after "
                f"{operation} failed at {self.log_path}: {type(exc).__name__}: {exc}"
            )
            self._warned = True
        self._disabled = True
        writer = self.writer
        self.writer = None
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

    def log(self, data: dict[str, float], step: int) -> None:
        if self._disabled or self.writer is None:
            return
        for key, value in data.items():
            try:
                self.writer.add_scalar(key, value, step)
            except Exception as exc:
                self._disable(exc, f"add_scalar({key})")
                return

    def finish(self):
        if self.writer is None:
            return
        try:
            self.writer.close()
        except Exception as exc:
            self._disable(exc, "close")


class MetricLogger:
    supported_logger = ["wandb", "swanlab", "tensorboard"]

    def __init__(self, cfg: DictConfig):
        logger_cfg = cfg.runner.logger

        log_path = logger_cfg.get("log_path", "logs")
        project_name = logger_cfg.get("project_name", "rlinf")
        experiment_name = logger_cfg.get("experiment_name", "default")

        logger_backends = logger_cfg.get("logger_backends", ["tensorboard"])
        if isinstance(logger_backends, str):
            self.logger_backends = [logger_backends]
        elif logger_backends is None:
            self.logger_backends = []
        else:
            self.logger_backends = logger_backends

        wandb_proxy = logger_cfg.get("wandb_proxy", None)
        swanlab_mode = logger_cfg.get("swanlab_mode", "cloud")
        if len(self.logger_backends) > 0:
            assert all(
                backend in self.supported_logger for backend in self.logger_backends
            ), f"Unsupported logger backend: {self.logger_backends}"

        self.logger = {}
        self.log_path = self._resolve_log_path(
            cfg=cfg, requested_log_path=log_path, experiment_name=experiment_name
        )
        log_path = self.log_path
        config = OmegaConf.to_container(cfg, resolve=True)

        if "wandb" in self.logger_backends:
            try:
                import wandb

                wandb_log_path = os.path.join(log_path, "wandb")
                os.makedirs(wandb_log_path, exist_ok=True)

                settings = None
                if wandb_proxy:
                    settings = wandb.Settings(https_proxy=wandb_proxy)
                wandb.init(
                    project=project_name,
                    name=experiment_name,
                    config=config,
                    settings=settings,
                    dir=wandb_log_path,
                )
                self.logger["wandb"] = wandb
            except OSError as exc:
                warnings.warn(
                    f"[MetricLogger] disabled wandb logging at {log_path} due to OSError: {exc}"
                )

        if "swanlab" in self.logger_backends:
            try:
                import swanlab

                swanlab_log_path = os.path.join(log_path, "swanlab")
                os.makedirs(swanlab_log_path, exist_ok=True)

                swanlab.init(
                    project=project_name,
                    experiment_name=experiment_name,
                    config=config,
                    logdir=swanlab_log_path,
                    mode=swanlab_mode,
                )
                self.logger["swanlab"] = swanlab
            except OSError as exc:
                warnings.warn(
                    f"[MetricLogger] disabled swanlab logging at {log_path} due to OSError: {exc}"
                )

        if "tensorboard" in self.logger_backends:
            try:
                tensorboard_log_path = os.path.join(log_path, "tensorboard")
                os.makedirs(tensorboard_log_path, exist_ok=True)

                config_yaml_path = os.path.join(tensorboard_log_path, "config.yaml")
                OmegaConf.save(cfg, config_yaml_path, resolve=True)

                self.logger["tensorboard"] = _TensorboardLogger(tensorboard_log_path)
            except OSError as exc:
                warnings.warn(
                    f"[MetricLogger] disabled tensorboard logging at {log_path} due to OSError: {exc}"
                )

    def _resolve_log_path(
        self, cfg: DictConfig, requested_log_path: str, experiment_name: str
    ) -> str:
        last_error = None
        for candidate in self._get_log_path_candidates(
            requested_log_path=requested_log_path,
            experiment_name=experiment_name,
        ):
            try:
                self._probe_writable_dir(candidate)
                if os.path.abspath(candidate) != os.path.abspath(requested_log_path):
                    warnings.warn(
                        "[MetricLogger] requested log_path is not writable; "
                        f"falling back from {requested_log_path} to {candidate}"
                    )
                with open_dict(cfg):
                    cfg.runner.logger.log_path = candidate
                return candidate
            except OSError as exc:
                last_error = exc

        raise last_error

    def _get_log_path_candidates(
        self, requested_log_path: str, experiment_name: str
    ) -> list[str]:
        candidates = [requested_log_path]
        run_name = os.path.basename(os.path.normpath(requested_log_path))
        if not run_name:
            run_name = experiment_name or "rlinf"

        fallback_roots = []
        env_root = os.environ.get("RLINF_FALLBACK_LOG_ROOT", None)
        if env_root:
            fallback_roots.append(env_root)
        fallback_roots.append(os.path.join(tempfile.gettempdir(), "rlinf_logs"))

        seen = {os.path.abspath(requested_log_path)}
        for root in fallback_roots:
            candidate = os.path.join(root, run_name)
            candidate_abs = os.path.abspath(candidate)
            if candidate_abs not in seen:
                candidates.append(candidate)
                seen.add(candidate_abs)
        return candidates

    @staticmethod
    def _probe_writable_dir(log_path: str) -> None:
        os.makedirs(log_path, exist_ok=True)
        probe_path = os.path.join(log_path, ".rlinf_write_probe")
        with open(probe_path, "w", encoding="utf-8") as fp:
            fp.write("probe\n")
        os.remove(probe_path)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                try:
                    logger_instance.log(data=data, step=step)
                except Exception as exc:
                    warnings.warn(
                        "[MetricLogger] backend "
                        f"{default_backend} failed during log and will be ignored: "
                        f"{type(exc).__name__}: {exc}"
                    )

    def log_table(self, df_data, name, step):
        if "wandb" in self.logger_backends:
            table = self.logger["wandb"].Table(dataframe=df_data)
            self.logger["wandb"].log({name: table}, step=step)
        else:
            raise ValueError(f"Unsupported log table for {self.logger_backends}")

    def __del__(self):
        try:
            self.finish()
        except Exception:
            pass

    def finish(self):
        for logger_instance in self.logger.values():
            try:
                logger_instance.finish()
            except Exception as exc:
                warnings.warn(
                    "[MetricLogger] backend finish failed and was ignored: "
                    f"{type(exc).__name__}: {exc}"
                )
