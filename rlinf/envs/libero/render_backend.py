import os
from typing import Any


def configure_libero_render_backend(
    seed_offset: int,
    worker_info: Any = None,
    env_cfg: Any = None,
) -> None:
    """Apply safe headless rendering defaults for LIBERO subprocesses.

    We use setdefault so explicit launcher / cluster env vars always win.
    """

    mujoco_gl = "egl"
    pyopengl_platform = "egl"
    egl_platform = None
    if env_cfg is not None:
        mujoco_gl = env_cfg.get("mujoco_gl", mujoco_gl)
        pyopengl_platform = env_cfg.get(
            "pyopengl_platform", pyopengl_platform
        )
        egl_platform = env_cfg.get("egl_platform", egl_platform)

    os.environ.setdefault("MUJOCO_GL", str(mujoco_gl))
    os.environ.setdefault("PYOPENGL_PLATFORM", str(pyopengl_platform))
    if egl_platform is not None:
        os.environ.setdefault("EGL_PLATFORM", str(egl_platform))

    if "MUJOCO_EGL_DEVICE_ID" in os.environ:
        return

    available_accelerators = getattr(worker_info, "available_accelerators", None)
    if available_accelerators:
        os.environ.setdefault(
            "MUJOCO_EGL_DEVICE_ID", str(available_accelerators[0])
        )
    else:
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(seed_offset))
