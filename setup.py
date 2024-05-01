# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

from pathlib import Path
from typing import List, Tuple

import setuptools
from setuptools.command.build_ext import build_ext as BuildExtension

from transformer_engine.build_tools.build_ext import CMakeExtension, get_build_ext
from transformer_engine.build_tools.utils import (
    found_cmake,
    found_ninja,
    remove_dups,
    userbuffers_enabled,
)
from transformer_engine.te_version import te_version

CMakeBuildExtension = get_build_ext(BuildExtension)


def setup_common_extension() -> CMakeExtension:
    """Setup CMake extension for common library

    Also builds JAX or userbuffers support if needed.

    """
    cmake_flags = []
    if userbuffers_enabled():
        cmake_flags.append("-DNVTE_WITH_USERBUFFERS=ON")

    # Project directory root
    root_path = Path(__file__).resolve().parent
    return CMakeExtension(
        name="transformer_engine",
        cmake_path=root_path / Path("transformer_engine"),
        cmake_flags=cmake_flags,
    )


def setup_requirements() -> Tuple[List[str], List[str], List[str]]:
    """Setup Python dependencies

    Returns dependencies for build, runtime, and testing.
    """

    # Common requirements
    setup_reqs: List[str] = []
    install_reqs: List[str] = [
        "pydantic",
        "importlib-metadata>=1.0; python_version<'3.8'",
    ]
    test_reqs: List[str] = ["pytest"]

    # Requirements that may be installed outside of Python
    if not found_cmake():
        setup_reqs.append("cmake>=3.18")
    if not found_ninja():
        setup_reqs.append("ninja")

    return [remove_dups(reqs) for reqs in [setup_reqs, install_reqs, test_reqs]]


if __name__ == "__main__":
    # Dependencies
    setup_requires, install_requires, test_requires = setup_requirements()

    __version__ = te_version()

    # Configure package
    setuptools.setup(
        name="transformer_engine",
        version=__version__,
        packages=setuptools.find_packages(
            include=["transformer_engine", "transformer_engine.*"],
        ),
        extras_require={
            "jax": [f"transformer_engine_jax=={__version__}"],
            "torch": [f"transformer_engine_torch=={__version__}"],
            "paddle": [f"transformer_engine_paddle=={__version__}"],
            "test": test_requires,
        },
        description="Transformer acceleration library",
        ext_modules=[setup_common_extension()],
        cmdclass={"build_ext": CMakeBuildExtension},
        setup_requires=setup_requires,
        install_requires=install_requires,
        license_files=("LICENSE",),
    )
