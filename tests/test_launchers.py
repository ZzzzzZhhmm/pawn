# Copyright 2026 The PAWN Authors.
# SPDX-License-Identifier: Apache-2.0
"""Exercise shell argument forwarding and failures without loading GPU models."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
if os.name == "nt":
    candidate = (
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe"
    )
    if candidate.is_file():
        BASH = str(candidate)


@pytest.mark.skipif(BASH is None, reason="Bash is unavailable")
@pytest.mark.parametrize("mode", ["train", "eval"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_launcher(tmp_path, mode, exit_code):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python"
    fake_python.write_text(
        '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\nexit "${FAKE_EXIT_CODE:-0}"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = str(bin_dir) + os.pathsep + env["PATH"]
    env["PAWN_LOG_DIR"] = str(tmp_path / "logs")
    env["FAKE_EXIT_CODE"] = str(exit_code)
    script = "run_embodiment.sh" if mode == "train" else "run_eval.sh"
    result = subprocess.run(
        [
            BASH,
            str(ROOT / "examples/embodiment" / script),
            "libero_object_nft_actor_openpi",
            "runner.max_epochs=1",
            "runner.logger.experiment_name=two words",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == exit_code, result.stderr
    entry = "train_embodied_agent.py" if mode == "train" else "eval_embodied_agent.py"
    assert entry in result.stdout
    assert "--config-name\nlibero_object_nft_actor_openpi\n" in result.stdout
    assert "runner.logger.experiment_name=two words\n" in result.stdout
    assert (tmp_path / "logs/console.log").is_file()
