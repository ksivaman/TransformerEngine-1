/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <ATen/cuda/CUDAContext.h>

#include <vector>

#include "common/common.h"
#include "extensions.h"

namespace transformer_engine::pytorch {

namespace {

at::Tensor collect_pointers_in_device_tensor(const std::vector<uint64_t>& host_ptrs,
                                             const at::Device& device, cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(host_ptrs.size());
  auto out = at::empty({count}, at::TensorOptions().dtype(at::kLong).device(device));
  auto out_nvte = makeTransformerEngineTensor(out);
  nvte_convert_pointers_to_tensor(host_ptrs.data(), out_nvte.data(), count, stream);
  return out;
}

}  // namespace

std::vector<at::Tensor> convert_host_pointers_to_tensor(
    std::vector<std::vector<at::Tensor>> tensor_lists) {
  std::vector<at::Tensor> outputs;
  outputs.reserve(tensor_lists.size());
  auto stream = at::cuda::getCurrentCUDAStream();

  for (const auto& tensor_list : tensor_lists) {
    NVTE_CHECK(!tensor_list.empty(), "Tensor list is empty.");
    const auto& first_tensor = tensor_list[0];
    NVTE_CHECK(first_tensor.is_cuda(), "Tensor list must be on CUDA.");
    const auto device = first_tensor.device();
    const int64_t count = static_cast<int64_t>(tensor_list.size());
    std::vector<uint64_t> host_ptrs(count);
    for (int64_t i = 0; i < count; ++i) {
      host_ptrs[i] = reinterpret_cast<uintptr_t>(tensor_list[static_cast<size_t>(i)].data_ptr());
    }
    outputs.push_back(collect_pointers_in_device_tensor(host_ptrs, device, stream));
  }

  return outputs;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> get_device_pointer_for_data_and_scales(
    std::vector<at::Tensor> data_tensors, std::vector<at::Tensor> scale_tensors, bool swizzle,
    bool rowwise, transformer_engine::DType data_dtype, int64_t pad_data_to_multiple) {
  const size_t num_tensors = data_tensors.size();
  NVTE_CHECK(num_tensors > 0, "data_tensors must not be empty.");
  NVTE_CHECK(num_tensors == scale_tensors.size(),
             "data_tensors and scale_tensors must have the same size.");
  NVTE_CHECK(data_tensors[0].is_cuda(), "data_tensors must be on CUDA.");
  const auto device = data_tensors[0].device();
  auto stream = at::cuda::getCurrentCUDAStream();

  NVTE_CHECK(data_tensors[0].dim() == 2,
             "data_tensors elements must be 2D, got dim=", data_tensors[0].dim());

  const int64_t orig_dim0 = data_tensors[0].size(0);
  const int64_t dim1 = data_tensors[0].size(1);

  // Optionally pad the first dimension so that cuDNN's tile_atom_to_shape_SF
  // produces the full BlockScaledBasicChunk atom layout.
  const int64_t padded_dim0 =
      (pad_data_to_multiple > 0)
          ? ((orig_dim0 + pad_data_to_multiple - 1) / pad_data_to_multiple) * pad_data_to_multiple
          : orig_dim0;
  const bool needs_data_padding = (padded_dim0 != orig_dim0);

  NVTEShape data_shape{};
  data_shape.ndim = 2;
  data_shape.data[0] = static_cast<size_t>(padded_dim0);
  data_shape.data[1] = static_cast<size_t>(dim1);

  at::Tensor padded_data_keepalive;
  std::vector<uint64_t> data_host_ptrs(num_tensors);

  if (needs_data_padding) {
    const auto elem_size = data_tensors[0].element_size();
    const size_t per_expert_bytes = static_cast<size_t>(padded_dim0) * dim1 * elem_size;
    const size_t total_bytes = num_tensors * per_expert_bytes;
    padded_data_keepalive =
        at::zeros({static_cast<int64_t>(total_bytes)},
                  at::TensorOptions().dtype(at::kByte).device(device));
    uint8_t* buf = reinterpret_cast<uint8_t*>(padded_data_keepalive.data_ptr());

    for (size_t i = 0; i < num_tensors; ++i) {
      uint8_t* expert_ptr = buf + i * per_expert_bytes;
      at::Tensor padded_view;
      if (rowwise) {
        padded_view = at::from_blob(expert_ptr, {padded_dim0, dim1},
                                    {dim1 * elem_size, elem_size},
                                    at::TensorOptions().dtype(data_tensors[i].dtype())
                                        .device(device));
      } else {
        padded_view = at::from_blob(expert_ptr, {padded_dim0, dim1},
                                    {elem_size, padded_dim0 * elem_size},
                                    at::TensorOptions().dtype(data_tensors[i].dtype())
                                        .device(device));
      }
      padded_view.narrow(0, 0, orig_dim0).copy_(data_tensors[i]);
      data_host_ptrs[i] = reinterpret_cast<uintptr_t>(expert_ptr);
    }
  } else {
    padded_data_keepalive =
        at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
    for (size_t i = 0; i < num_tensors; ++i) {
      data_host_ptrs[i] = reinterpret_cast<uintptr_t>(data_tensors[i].data_ptr());
    }
  }

  // Swizzle scales and collect scale pointers
  at::Tensor swizzled_scales_keepalive;
  std::vector<uint64_t> scale_host_ptrs(num_tensors);

  if (swizzle) {
    NVTEScalingMode scaling_mode;
    transformer_engine::DType scale_dtype;
    if (is_fp8_dtype(data_dtype)) {
      scaling_mode = NVTE_MXFP8_1D_SCALING;
      scale_dtype = transformer_engine::DType::kFloat8E8M0;
    } else if (is_fp4_dtype(data_dtype)) {
      scaling_mode = NVTE_NVFP4_1D_SCALING;
      scale_dtype = transformer_engine::DType::kFloat8E4M3;
    } else {
      NVTE_ERROR("data_dtype must be an FP8 or FP4 type for swizzling.");
    }

    // Compute output buffer size for swizzled scales (16B aligned per tensor)
    std::vector<size_t> output_offsets;
    size_t output_bytes = 0;
    for (size_t i = 0; i < num_tensors; ++i) {
      const size_t scale_numel = static_cast<size_t>(scale_tensors[i].numel());
      const size_t dtype_bits = transformer_engine::pytorch::typeToNumBits(scale_dtype);
      output_bytes = roundup(output_bytes, 16);
      output_offsets.push_back(output_bytes);
      output_bytes += ceildiv(scale_numel * dtype_bits, 8);
    }

    // Allocate single buffer for all swizzled scales
    swizzled_scales_keepalive =
        allocateSpace(std::vector<size_t>{output_bytes}, transformer_engine::DType::kByte, false);
    uint8_t* output_dptr = reinterpret_cast<uint8_t*>(getDataPtr(swizzled_scales_keepalive));

    // Build TensorWrapper input/output pairs and get scale shapes
    std::vector<transformer_engine::TensorWrapper> inputs_nvte, outputs_nvte;
    inputs_nvte.reserve(num_tensors);
    outputs_nvte.reserve(num_tensors);
    for (size_t i = 0; i < num_tensors; ++i) {
      inputs_nvte.emplace_back(scaling_mode);
      outputs_nvte.emplace_back(scaling_mode);
      auto& input_nvte = inputs_nvte.back();
      auto& output_nvte = outputs_nvte.back();
      output_nvte.set_with_gemm_swizzled_scales(true);

      NVTEShape scale_shape = convertTorchShape(scale_tensors[i].sizes());
      void* scale_ptr = scale_tensors[i].data_ptr();
      uint8_t* out_scale_ptr = output_dptr + output_offsets[i];

      if (rowwise) {
        input_nvte.set_rowwise_data(nullptr, data_dtype, data_shape);
        input_nvte.set_rowwise_scale_inv(scale_ptr, scale_dtype, scale_shape);
        output_nvte.set_rowwise_data(nullptr, data_dtype, data_shape);
        output_nvte.set_rowwise_scale_inv(out_scale_ptr, scale_dtype, scale_shape);
      } else {
        input_nvte.set_columnwise_data(nullptr, data_dtype, data_shape);
        input_nvte.set_columnwise_scale_inv(scale_ptr, scale_dtype, scale_shape);
        output_nvte.set_columnwise_data(nullptr, data_dtype, data_shape);
        output_nvte.set_columnwise_scale_inv(out_scale_ptr, scale_dtype, scale_shape);
      }
    }

    // Pack raw NVTETensors and launch swizzle kernel
    std::vector<NVTETensor> inputs_raw, outputs_raw;
    inputs_raw.reserve(num_tensors);
    outputs_raw.reserve(num_tensors);
    for (auto& t : inputs_nvte) inputs_raw.push_back(t.data());
    for (auto& t : outputs_nvte) outputs_raw.push_back(t.data());

    nvte_multi_tensor_swizzle_scaling_factors(inputs_raw.data(), outputs_raw.data(), num_tensors,
                                              stream);

    // Collect swizzled scale pointers
    for (size_t i = 0; i < num_tensors; ++i) {
      scale_host_ptrs[i] = reinterpret_cast<uintptr_t>(output_dptr + output_offsets[i]);
    }
  } else {
    swizzled_scales_keepalive = at::empty({0}, at::TensorOptions().dtype(at::kByte).device(device));
    for (size_t i = 0; i < num_tensors; ++i) {
      scale_host_ptrs[i] = reinterpret_cast<uintptr_t>(scale_tensors[i].data_ptr());
    }
  }

  // Convert pointer arrays to device tensors
  auto data_ptrs = collect_pointers_in_device_tensor(data_host_ptrs, device, stream);
  auto scale_ptrs = collect_pointers_in_device_tensor(scale_host_ptrs, device, stream);

  return {std::move(data_ptrs), std::move(scale_ptrs), std::move(swizzled_scales_keepalive),
          std::move(padded_data_keepalive)};
}

}  // namespace transformer_engine::pytorch
