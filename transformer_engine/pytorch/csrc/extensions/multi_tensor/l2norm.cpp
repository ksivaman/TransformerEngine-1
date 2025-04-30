/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"


std::tuple<at::Tensor, at::Tensor> multi_tensor_l2norm_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);
  auto output = at::zeros({320}, float_options);

  at::Tensor output_per_tensor;
  at::Tensor ret_per_tensor;

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = at::zeros({ntensors * max_chunks_per_tensor}, float_options);
    ret_per_tensor = at::empty({ntensors}, float_options);
  } else {
    ret_per_tensor = at::empty({0}, float_options);
  }

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      tensor_lists[0][0].scalar_type(), 0, "multi_tensor_l2norm_cuda",
      multi_tensor_apply<1>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                            L2NormFunctor<scalar_t_0>(), output.data_ptr<float>(),
                            per_tensor ? output_per_tensor.data_ptr<float>() : nullptr, per_tensor,
                            max_chunks_per_tensor);)

  AT_CUDA_CHECK(cudaGetLastError());
  // AT_CUDA_CHECK(cudaDeviceSynchronize());

  // This involves one more small kernel launches, but will be negligible end to end.
  // I could get rid of these by hacking the functor + multi tensor harness with persistence
  // logic, but keeping it simple for now
  auto ret = at::empty({1}, output.options());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  auto stream = at::cuda::getCurrentCUDAStream();
  cleanup<<<per_tensor ? ntensors : 1, 512, 0, stream>>>(
      output.data_ptr<float>(), per_tensor ? output_per_tensor.data_ptr<float>() : nullptr,
      ret.data_ptr<float>(), per_tensor ? ret_per_tensor.data_ptr<float>() : nullptr, per_tensor,
      max_chunks_per_tensor);

  return std::tuple<at::Tensor, at::Tensor>(ret, ret_per_tensor);
}

std::tuple<at::Tensor, at::Tensor> multi_tensor_unscale_l2norm_cuda(
    int chunk_size, at::Tensor noop_flag, std::vector<std::vector<at::Tensor>> tensor_lists,
    at::Tensor inv_scale, at::optional<bool> per_tensor_python) {
  bool per_tensor = per_tensor_python.has_value() ? per_tensor_python.value() : false;

  auto float_options = tensor_lists[0][0].options().dtype(at::kFloat);
  auto output = at::zeros({320}, float_options);

  at::Tensor output_per_tensor;
  at::Tensor ret_per_tensor;

  int ntensors = tensor_lists[0].size();
  int max_chunks_per_tensor = -1;

  // Create output tensors for multi scale L2 norm kernel.
  if (per_tensor) {
    for (int t = 0; t < ntensors; t++) {
      int max_chunks_this_tensor = (tensor_lists[0][t].numel() + chunk_size - 1) / chunk_size;
      if (max_chunks_this_tensor > max_chunks_per_tensor)
        max_chunks_per_tensor = max_chunks_this_tensor;
    }
    output_per_tensor = at::zeros({ntensors * max_chunks_per_tensor}, float_options);
    ret_per_tensor = at::empty({ntensors}, float_options);
  } else {
    ret_per_tensor = at::empty({0}, float_options);
  }
  auto ret = at::empty({1}, output.options());

  

  DISPATCH_FLOAT_HALF_AND_BFLOAT(
      tensor_lists[0][0].scalar_type(), 0, "multi_tensor_unscale_l2norm_cuda",
      multi_tensor_apply<1>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                            UnscaleL2NormFunctor<scalar_t_0>(), inv_scale.data_ptr<float>(),
                            output.data_ptr<float>(),
                            per_tensor ? output_per_tensor.data_ptr<float>() : nullptr, per_tensor,
                            max_chunks_per_tensor);)

  AT_CUDA_CHECK(cudaGetLastError());
  // AT_CUDA_CHECK(cudaDeviceSynchronize());

  // This involves one more small kernel launches, but will be negligible end to end.
  // I could get rid of these by hacking the functor + multi tensor harness with persistence
  // logic, but keeping it simple for now
  const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
  auto stream = at::cuda::getCurrentCUDAStream();
  cleanup<<<per_tensor ? ntensors : 1, 512, 0, stream>>>(
      output.data_ptr<float>(), per_tensor ? output_per_tensor.data_ptr<float>() : nullptr,
      ret.data_ptr<float>(), per_tensor ? ret_per_tensor.data_ptr<float>() : nullptr, per_tensor,
      max_chunks_per_tensor);

  return std::tuple<at::Tensor, at::Tensor>(ret, ret_per_tensor);
}
