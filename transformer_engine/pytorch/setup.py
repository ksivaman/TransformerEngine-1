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
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension

try:
    import transformer_engine  # noqa: F401
except ImportError as e:
    raise RuntimeError("The package `transformer_engine` must be installed in order to build this package.") from e

from transformer_engine.build_tools.build_ext import get_build_ext
from transformer_engine.build_tools.utils import (
    all_files_in_dir,
    cuda_version,
    found_cmake,
    found_ninja,
    remove_dups,
    userbuffers_enabled,
)
from transformer_engine.te_version import te_version

CMakeBuildExtension = get_build_ext(BuildExtension, dlfw="torch")

try:
    import torch  # noqa: F401
except ImportError as e:
    raise RuntimeError("This package needs Torch to build.") from e


def setup_pytorch_extension() -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    src_dir = Path("transformer_engine/pytorch/csrc")
    extensions_dir = src_dir / "extensions"
    sources = [
        src_dir / "common.cu",
        src_dir / "ts_fp8_op.cpp",
        # We need to compile system.cpp because the pytorch extension uses
        # transformer_engine::getenv. This is a workaround to avoid direct
        # linking with libtransformer_engine.so, as the pre-built PyTorch
        # wheel from conda or PyPI was not built with CXX11_ABI, and will
        # cause undefined symbol issues.
        Path("transformer_engine/common/util/system.cpp"),
    ] + all_files_in_dir(extensions_dir)

    # Header files
    include_dirs = [
        Path("transformer_engine/common/include"),
        Path("transformer_engine/pytorch/csrc"),
        Path("transformer_engine"),
        Path("3rdparty/cudnn-frontend/include"),
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

    # userbuffers support
    if userbuffers_enabled():
        if os.getenv("MPI_HOME"):
            mpi_home = Path(os.getenv("MPI_HOME"))
            include_dirs.append(mpi_home / "include")
        cxx_flags.append("-DNVTE_WITH_USERBUFFERS")
        nvcc_flags.append("-DNVTE_WITH_USERBUFFERS")

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CUDAExtension

    return CUDAExtension(
        name="transformer_engine_extensions",
        sources=sources,
        include_dirs=include_dirs,
        # libraries=["transformer_engine"], ### TODO (tmoon) Debug linker errors
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    )


if __name__ == "__main__":
    # Extensions
    ext_modules = [setup_pytorch_extension()]

    # if "paddle" in frameworks():
    #     ext_modules.append(setup_paddle_extension())

    # Configure package
    setuptools.setup(
        name="transformer_engine_torch",
        version=te_version(),
        packages=[],
        description="Transformer acceleration library - Torch Lib",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        install_requires=["torch", "flash-attn>=2.0.6,<=2.4.2,!=2.0.9,!=2.1.0"],
        test_requires=["numpy", "onnxruntime", "torchvision"],
        license_files=("LICENSE",),
    )
