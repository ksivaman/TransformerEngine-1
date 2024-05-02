# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FW agnostic user-end APIs"""

import ctypes
import os
import platform
from pathlib import Path

import transformer_engine


def get_te_path():
    """Find Transformer Engine install path using pip"""
    return Path(transformer_engine.__path__[0]).parent


def _get_sys_extension():
    system = platform.system()
    if system == "Linux":
        extension = "so"
    elif system == "Darwin":
        extension = "dylib"
    elif system == "Windows":
        extension = "dll"
    else:
        raise RuntimeError(f"Unsupported operating system ({system})")

    return extension


def _load_library():
    """Load shared library with Transformer Engine C extensions"""

    so_dir = get_te_path() / "transformer_engine"

    return ctypes.CDLL(
        so_dir / f"libtransformer_engine.{_get_sys_extension()}",
        mode=ctypes.RTLD_GLOBAL,
    )


def _load_userbuffers():
    """Load shared library with userbuffers"""

    so_dir = get_te_path() / "transformer_engine"
    so_file = so_dir / f"libtransformer_engine_userbuffers.{_get_sys_extension()}"

    if so_file.exists():
        return ctypes.CDLL(so_file, mode=ctypes.RTLD_GLOBAL)
    return None


# if not hasattr(os.environ, "PIP_DEFAULT_TIMEOUT"):
if "TE_PROJECT_BUILDING" not in os.environ:
    _TE_LIB_CTYPES = _load_library()
    _UB_LIB_CTYPES = _load_userbuffers()
