/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

void multi_tensor_scale_cuda(int chunk_size, at::Tensor noop_flag,
                             std::vector<std::vector<at::Tensor>> at_tensor_lists, float scale) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);

  size_t outer_size = at_tensor_lists.size();
  NVTE_CHECK(outer_size > 0, "Outer list is empty");
  size_t inner_size = at_tensor_lists[0].size();

  for (const auto& list : at_tensor_lists) {
      NVTE_CHECK(list.size() == inner_size, "Inconsistent inner list sizes");
  }

  // Flattened storage
  std::vector<transformer_engine::TensorWrapper> flat_wrappers;
  flat_wrappers.reserve(outer_size * inner_size);
  std::vector<NVTETensor*> tensor_ptrs_final;
  tensor_ptrs_final.reserve(outer_size);

  for (const auto& row : at_tensor_lists) {
      tensor_ptrs_final.push_back(reinterpret_cast<NVTETensor*>(flat_wrappers.data() + flat_wrappers.size()));
      for (const auto& t : row) {
          flat_wrappers.push_back(makeTransformerEngineTensor(t));
      }
  }

  // Now extract actual NVTETensor values
  std::vector<NVTETensor> flat_nvtes;
  flat_nvtes.reserve(flat_wrappers.size());
  for (const auto& te : flat_wrappers) {
      flat_nvtes.push_back(te.data());
  }

  // Rebuild tensor_ptrs to point to correct locations in flat_nvtes
  tensor_ptrs_final.clear();
  for (size_t i = 0; i < outer_size; ++i) {
      tensor_ptrs_final.push_back(&flat_nvtes[i * inner_size]);
  }

  nvte_multi_tensor_scale_cuda(chunk_size, noop_flag_cu.data(), tensor_ptrs_final.data(), outer_size,
                               inner_size, scale, at::cuda::getCurrentCUDAStream());
}
