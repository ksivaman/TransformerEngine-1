/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_HEADERS_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_HEADERS_H_


#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


#ifdef NVTE_LIBTORCH_STABLE_ABI


// Original headers

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
// #include <c10/macros/Macros.h>
// #include <c10/util/Float8_e4m3fn.h>
// #include <c10/util/Float8_e5m2.h>
#include <torch/cuda.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "c10/util/ArrayRef.h"

// Replacements

#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/csrc/stable/accelerator.h>


// To add.

// #include <torch/csrc/stable/tensor.h>
// #include <torch/csrc/stable/library.h>
// #include <torch/csrc/stable/ops.h>
// #include <torch/headeronly/core/ScalarType.h>

#else

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/macros/Macros.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <torch/cuda.h>
#include <torch/torch.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "c10/util/ArrayRef.h"

#endif


cudaStream_t get_current_cuda_stream();


#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_HEADERS_H_