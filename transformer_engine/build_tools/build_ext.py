# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script."""

import ctypes
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional

import setuptools

from transformer_engine.build_tools.utils import (
    cmake_bin,
    debug_build_enabled,
    found_ninja,
)


class CMakeExtension(setuptools.Extension):
    """CMake extension module"""

    def __init__(
        self,
        name: str,
        cmake_path: Path,
        cmake_flags: Optional[List[str]] = None,
    ) -> None:
        super().__init__(name, sources=[])  # No work for base class
        self.cmake_path: Path = cmake_path
        self.cmake_flags: List[str] = [] if cmake_flags is None else cmake_flags

    def _build_cmake(self, build_dir: Path, install_dir: Path) -> None:
        # Make sure paths are str
        _cmake_bin = str(cmake_bin())
        cmake_path = str(self.cmake_path)
        build_dir = str(build_dir)
        install_dir = str(install_dir)

        # CMake configure command
        build_type = "Debug" if debug_build_enabled() else "Release"
        configure_command = [
            _cmake_bin,
            "-S",
            cmake_path,
            "-B",
            build_dir,
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-DPython_INCLUDE_DIR={sysconfig.get_path('include')}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        ]
        configure_command += self.cmake_flags
        if found_ninja():
            configure_command.append("-GNinja")
        try:
            import pybind11
        except ImportError:
            pass
        else:
            pybind11_dir = Path(pybind11.__file__).resolve().parent
            pybind11_dir = pybind11_dir / "share" / "cmake" / "pybind11"
            configure_command.append(f"-Dpybind11_DIR={pybind11_dir}")

        # CMake build and install commands
        build_command = [_cmake_bin, "--build", build_dir]
        install_command = [_cmake_bin, "--install", build_dir]

        # Run CMake commands
        for command in [configure_command, build_command, install_command]:
            print(f"Running command {' '.join(command)}")
            try:
                subprocess.run(command, cwd=build_dir, check=True)
            except (CalledProcessError, OSError) as e:
                raise RuntimeError(f"Error when running CMake: {e}")


def get_build_ext(extension_cls: setuptools.Extension, dlfw: Optional[str] = None):
    if dlfw is not None and dlfw not in ["jax", "torch", "paddle"]:
        raise ValueError(f"Unexpected value received for `dlfw`: {dlfw}")

    class _CMakeBuildExtension(extension_cls):
        """Setuptools command with support for CMake extension modules"""

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)

        def run(self) -> None:
            # Build CMake extensions
            for ext in self.extensions:
                if isinstance(ext, CMakeExtension):
                    print(f"Building CMake extension {ext.name}")
                    # Set up incremental builds for CMake extensions
                    setup_dir = Path(__file__).resolve().parent
                    build_dir = setup_dir / "build" / "cmake"
                    build_dir.mkdir(
                        parents=True, exist_ok=True
                    )  # Ensure the directory exists
                    package_path = Path(self.get_ext_fullpath(ext.name))
                    install_dir = package_path.resolve().parent
                    ext._build_cmake(
                        build_dir=build_dir,
                        install_dir=install_dir,
                    )

            # Paddle requires linker search path for libtransformer_engine.so
            paddle_ext = None
            if dlfw == "paddle":
                for ext in self.extensions:
                    if "paddle" in ext.name:
                        ext.library_dirs.append(self.build_lib)
                        paddle_ext = ext
                        break

            # Build non-CMake extensions as usual
            all_extensions = self.extensions
            self.extensions = [
                ext for ext in self.extensions if not isinstance(ext, CMakeExtension)
            ]
            super().run()
            self.extensions = all_extensions

            # Manually write stub file for Paddle extension
            if paddle_ext is not None:
                # Load libtransformer_engine.so to avoid linker errors
                for path in Path(self.build_lib).iterdir():
                    if path.name.startswith("libtransformer_engine."):
                        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)

                # Figure out stub file path
                module_name = paddle_ext.name
                assert module_name.endswith(
                    "_pd_"
                ), "Expected Paddle extension module to end with '_pd_'"
                stub_name = module_name[:-4]  # remove '_pd_'
                stub_path = os.path.join(self.build_lib, stub_name + ".py")

                # Figure out library name
                # Note: This library doesn't actually exist. Paddle
                # internally reinserts the '_pd_' suffix.
                so_path = self.get_ext_fullpath(module_name)
                _, so_ext = os.path.splitext(so_path)
                lib_name = stub_name + so_ext

                # Write stub file
                print(f"Writing Paddle stub for {lib_name} into file {stub_path}")
                from paddle.utils.cpp_extension.extension_utils import custom_write_stub

                custom_write_stub(lib_name, stub_path)

    return _CMakeBuildExtension
