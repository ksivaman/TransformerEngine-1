/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_HEADERS_H_
#define TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_HEADERS_H_

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/native/DispatchStub.h>
#include <c10/macros/Macros.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Float8_e5m2.h>
#include <torch/cuda.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "c10/util/ArrayRef.h"

#endif  // TRANSFORMER_ENGINE_PYTORCH_CSRC_TORCH_HEADERS_H_