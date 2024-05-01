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
from setuptools.command.build_ext import build_ext as BuildExtension

try:
    import transformer_engine  # noqa: F401
except ImportError as e:
    raise RuntimeError("The package `transformer_engine` must be installed in order to build this package.") from e

from transformer_engine.build_tools.build_ext import get_build_ext
from transformer_engine.build_tools.utils import (
    found_cmake,
    found_ninja,
    found_pybind11,
    remove_dups,
)
from transformer_engine.te_version import te_version

CMakeBuildExtension = get_build_ext(BuildExtension, dlfw="jax")

try:
    import jax  # noqa: F401
except ImportError as e:
    raise RuntimeError("This package needs JAX to build.") from e

if __name__ == "__main__":
    # Extensions
    ext_modules = [setup_paddle_extension()]

    setup_reqs = list()
    if not found_pybind11():
        setup_reqs.append("pybind11")

    # Configure package
    setuptools.setup(
        name="transformer_engine_jax",
        version=te_version(),
        packages=[],
        description="Transformer acceleration library - Jax Lib",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        install_requires=["jax", "flax>=0.7.1"],
        test_requires=["numpy", "praxis"],
        setup_requires=setup_reqs,
        license_files=("LICENSE",),
    )
