# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70 80 89 90)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(transformer_engine_jax_ LANGUAGES CUDA CXX)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 4")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G")
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/")
find_package(CUDAToolkit REQUIRED cublas nvToolsExt)
find_package(CUDNN REQUIRED cudnn)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Check for cuDNN frontend API
set(CUDNN_FRONTEND_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cudnn-frontend/include")
if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}")
    message(FATAL_ERROR
            "Could not find cuDNN frontend API. "
            "Try running 'git submodule update --init --recursive' "
            "within the Transformer Engine source.")
endif()

include_directories("${PROJECT_SOURCE_DIR}/..")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../common/include")
include_directories("${CUDNN_FRONTEND_INCLUDE_DIR}")

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    transformer_engine_jax
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extensions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/modules.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/utils.cu
)

target_link_libraries(transformer_engine_jax PRIVATE CUDA::cudart CUDA::cublas CUDA::cublasLt)
install(TARGETS transformer_engine_jax DESTINATION .)
