# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script for TE jax extensions."""
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext as BuildExtension

try:
    import transformer_engine  # noqa: F401
except ImportError as e:
    raise RuntimeError("The package `transformer_engine` must be installed in order to build this package.") from e

from transformer_engine.build_tools.build_ext import get_build_ext
from transformer_engine.build_tools.utils import found_pybind11
from transformer_engine.te_version import te_version

CMakeBuildExtension = get_build_ext(BuildExtension, dlfw="jax")

try:
    import jax  # noqa: F401
except ImportError as e:
    raise RuntimeError("This package needs JAX to build.") from e

# Project directory root
root_path: Path = Path(__file__).resolve().parent.parent.parent


if __name__ == "__main__":
    setup_reqs = list()
    if not found_pybind11():
        setup_reqs.append("pybind11")

    # Configure package
    setuptools.setup(
        name="transformer_engine_jax",
        version=te_version(),
        packages=[],
        description="Transformer acceleration library - Jax Lib",
        ext_modules=[],
        cmdclass={"build_ext": CMakeBuildExtension},
        install_requires=["jax", "flax>=0.7.1"],
        test_requires=["numpy", "praxis"],
        setup_requires=setup_reqs,
        license_files=(root_path / "LICENSE",),
    )
