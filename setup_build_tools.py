# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import os
import contextlib
import pathlib
from typing import Generator
import shutil

import setuptools


@contextlib.contextmanager
def temp_project_dir(name: str) -> Generator[pathlib.Path, None, None]:
    project_dir = pathlib.Path(name)

    if project_dir.exists():
        raise RuntimeError(
            f"Impossible to proceed, folder: `{project_dir}` already exists."
        )

    project_dir.mkdir(parents=True, exist_ok=False)

    try:
        yield project_dir
    except:
        shutil.rmtree(project_dir)
        raise
    shutil.rmtree(project_dir)


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        path = pathlib.Path(path)
        for filename in filenames:
            paths.append(str(path / filename).replace(f"{directory}/", ""))
    return paths


if __name__ == "__main__":
    with open("transformer_engine/VERSION", "r") as f:
        __version__ = f.readline().strip()

    __package_name__ = "transformer_engine_build_tools"

    with temp_project_dir(f"{__package_name__}/") as tmp_dir:
        with pathlib.Path(tmp_dir / "__init__.py").open(mode="w") as f:
            f.write(f"__version__ = '{__version__}'")

        shutil.copy("transformer_engine/VERSION", tmp_dir)
        shutil.copy("transformer_engine/te_version.py", tmp_dir)

        shutil.copytree("transformer_engine/common/util", tmp_dir / "common/util")
        shutil.copy("transformer_engine/common/nvtx.h", tmp_dir / "common")
        shutil.copy("transformer_engine/common/common.h", tmp_dir / "common")

        shutil.copytree("transformer_engine/common/include", tmp_dir / "include")
        shutil.copytree("transformer_engine/build_tools", tmp_dir / "build_tools")
        shutil.copytree("3rdparty/cudnn-frontend", tmp_dir / "cudnn-frontend")

        # Configure package
        setuptools.setup(
            name=__package_name__,
            version=__version__,
            packages=[__package_name__],
            description="Transformer acceleration library - Build Requirements",
            license_files=("LICENSE",),
            package_data={
                __package_name__: package_files(tmp_dir),
            },
            include_package_data=True,
        )
