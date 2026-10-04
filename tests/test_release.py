# Copyright 2026 The PAWN Authors.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the public release's composition and import surface."""

import ast
import importlib.util
import re
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "examples" / "embodiment" / "config"
CONFIGS = sorted(p.stem for p in CONFIG_DIR.glob("*nft_actor_openpi*.yaml"))


@pytest.mark.parametrize("name", CONFIGS)
def test_training_config(name, monkeypatch):
    monkeypatch.setenv("PAWN_MODEL_PATH", "checkpoints/test-policy")
    monkeypatch.setenv("PAWN_OUTPUT_DIR", "outputs/test")
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=name)
    OmegaConf.resolve(cfg)
    assert cfg.actor.model.model_path == "checkpoints/test-policy"
    assert cfg.rollout.model.model_path == cfg.actor.model.model_path
    assert cfg.runner.logger.log_path == "outputs/test"
    assert cfg.algorithm.loss_type == "nft-dual-credit"
    assert cfg.algorithm.adv_type == "dual-credit"
    assert cfg.actor.model.openpi.store_full_solver_path
    assert cfg.actor.model.openpi.train_expert_only
    assert cfg.actor.model.use_proprio
    assert cfg.algorithm.enable_block_nft
    assert cfg.algorithm.enable_pair_guidance
    assert cfg.algorithm.enable_branch_rollout
    assert cfg.env.train.group_size == cfg.algorithm.group_size == 32
    assert not cfg.env.train.auto_reset
    assert not cfg.env.train.ignore_terminations
    for world_size in (4, 8):
        assert (
            cfg.actor.global_batch_size % (cfg.actor.micro_batch_size * world_size) == 0
        )
    assert cfg.env.train.total_num_envs % (2 * cfg.algorithm.group_size) == 0
    assert (
        cfg.env.train.max_steps_per_rollout_epoch % cfg.actor.model.num_action_chunks
        == 0
    )
    expected = "pi05" if name.endswith("_pi05") else "pi0"
    assert cfg.actor.model.openpi.config_name.startswith(expected + "_")


@pytest.mark.parametrize(
    "name",
    [
        "maniskill_nft_actor_openpi",
        "maniskill_nft_actor_openpi_pi05",
    ],
)
def test_maniskill_ood_override(name):
    with initialize_config_dir(version_base="1.1", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name=name,
            overrides=[
                "env@env.eval=maniskill_ood_template",
                "env.eval.init_params.id=PutOnPlateInScene25VisionImage-v1",
                "env.eval.init_params.obj_set=test",
            ],
        )
    OmegaConf.resolve(cfg)
    assert cfg.env.train.init_params.id == "PutOnPlateInScene25Main-v3"
    assert cfg.env.eval.init_params.obs_mode == "rgb+segmentation"
    assert cfg.env.eval.init_params.id == "PutOnPlateInScene25VisionImage-v1"
    assert cfg.env.eval.video_cfg.save_video


def test_local_imports_resolve():
    missing = []
    for path in (ROOT / "rlinf").rglob("*.py"):
        parts = path.relative_to(ROOT).with_suffix("").parts
        package = ".".join(parts[:-1])
        if path.name == "__init__.py":
            package = ".".join(parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    name = importlib.util.resolve_name("." * node.level + name, package)
                names = [name]
            for name in names:
                if not name.startswith("rlinf."):
                    continue
                target = ROOT.joinpath(*name.split("."))
                if not target.is_dir() and not target.with_suffix(".py").is_file():
                    missing.append((str(path.relative_to(ROOT)), name))
    assert not missing, missing


def test_release_scope():
    assert len(CONFIGS) == 10
    assert not any(p.is_file() for p in (ROOT / "stepnft").rglob("*"))
    for path in CONFIG_DIR.rglob("*.yaml"):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r":\s*['\"]?(?:[A-Za-z]:[/\\]|/(?!/))", source)
