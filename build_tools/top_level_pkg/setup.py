# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script for empty top-level package for dependency management."""

import setuptools
from pathlib import Path

if __name__ == "__main__":
    root_path = Path(__file__).resolve().parent.parent.parent
    with open(root_path / "build_tools" / "VERSION.txt", "r") as f:
        __version__ = f.readline().strip()
    with open(root_path / "README.rst", encoding="utf-8") as f:
        long_description = f.read()

    setuptools.setup(
        name="transformer_engine",
        version=__version__,
        description=("Transformer acceleration library."),
        long_description=long_description,
        long_description_content_type='text/x-rst',
        install_requires=[f"transformer_engine_cu12=={__version__}"],
        extras_require={
            "pytorch": [f"transformer_engine_torch=={__version__}"],
            "jax": [f"transformer_engine_jax=={__version__}"],
            "paddle": [f"transformer_engine_paddle=={__version__}"],
        },
    )

