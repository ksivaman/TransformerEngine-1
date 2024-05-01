# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import ctypes
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Tuple, Union

import setuptools
from paddle.utils.cpp_extension import BuildExtension
from setuptools.command.build_ext import build_ext

try:
    import transformer_engine  # noqa: F401
except ImportError as e:
    raise RuntimeError("The package `transformer_engine` must be installed in order to build this package.") from e

from transformer_engine.build_tools.build_ext import get_build_ext
from transformer_engine.build_tools.utils import found_cmake, found_ninja, remove_dups
from transformer_engine.te_version import te_version

CMakeBuildExtension = get_build_ext(BuildExtension, dlfw="paddle")

try:
    import paddle  # noqa: F401
except ImportError as e:
    raise RuntimeError("This package needs Paddle Paddle to build.") from e


def setup_paddle_extension() -> setuptools.Extension:
    """Setup CUDA extension for Paddle support"""

    # Source files
    root_path = Path(__file__).resolve().parent
    src_dir = root_path / "transformer_engine" / "paddle" / "csrc"
    sources = [
        src_dir / "extensions.cu",
        src_dir / "common.cpp",
        src_dir / "custom_ops.cu",
    ]

    # Header files
    include_dirs = [
        root_path / "transformer_engine" / "common" / "include",
        root_path / "transformer_engine" / "paddle" / "csrc",
        root_path / "transformer_engine",
    ]

    # Compiler flags
    cxx_flags = ["-O3"]
    nvcc_flags = [
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]

    # Version-dependent CUDA options
    try:
        version = cuda_version()
    except FileNotFoundError:
        print("Could not determine CUDA Toolkit version")
    else:
        if version >= (11, 2):
            nvcc_flags.extend(["--threads", "4"])
        if version >= (11, 0):
            nvcc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
        if version >= (11, 8):
            nvcc_flags.extend(["-gencode", "arch=compute_90,code=sm_90"])

    # Construct Paddle CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from paddle.utils.cpp_extension import CUDAExtension

    ext = CUDAExtension(
        sources=sources,
        include_dirs=include_dirs,
        libraries=["transformer_engine"],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    )
    ext.name = "transformer_engine_paddle_pd_"
    return ext


if __name__ == "__main__":
    # Extensions
    ext_modules = [setup_paddle_extension()]

    # Configure package
    setuptools.setup(
        name="transformer_engine_paddle",
        version=te_version(),
        packages=[],
        description="Transformer acceleration library - Paddle Paddle Lib",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        install_requires=["paddlepaddle-gpu"],
        test_requires=["numpy"],
        license_files=("LICENSE",),
    )
