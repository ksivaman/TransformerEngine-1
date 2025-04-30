/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"


void multi_tensor_adam_cuda(int chunk_size, at::Tensor noop_flag,
                            std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                            const float beta1, const float beta2, const float epsilon,
                            const int step, const int mode, const int bias_correction,
                            const float weight_decay) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [tensor_lists_ptr, num_lists, num_tensors] = makeTransformerEngineTensor(tensor_lists);

  nvte_multi_tensor_adam_cuda(
    chunk_size, noop_flag_cu.data(), tensor_lists_ptr, num_lists, num_tensors,
    lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay
  );
}

void multi_tensor_adam_param_remainder_cuda(int chunk_size, at::Tensor noop_flag,
                                            std::vector<std::vector<at::Tensor>> tensor_lists,
                                            const float lr, const float beta1, const float beta2,
                                            const float epsilon, const int step, const int mode,
                                            const int bias_correction, const float weight_decay) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [tensor_lists_ptr, num_lists, num_tensors] = makeTransformerEngineTensor(tensor_lists);

  nvte_multi_tensor_adam_param_remainder_cuda(
    chunk_size, noop_flag_cu.data(), tensor_lists_ptr, num_lists, num_tensors,
    lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay
  );
}

void multi_tensor_adam_fp8_cuda(int chunk_size, at::Tensor noop_flag,
                                std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
                                const float beta1, const float beta2, const float epsilon,
                                const int step, const int mode, const int bias_correction,
                                const float weight_decay, DType fp8_dtype) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [tensor_lists_ptr, num_lists, num_tensors] = makeTransformerEngineTensor(tensor_lists);

  nvte_multi_tensor_adam_fp8_cuda(
    chunk_size, noop_flag_cu.data(), tensor_lists_ptr, num_lists, num_tensors,
    lr, beta1, beta2, epsilon, step, mode, bias_correction, weight_decay,
    static_cast<NVTEDType>(fp8_dtype)
  );
}

void multi_tensor_adam_capturable_cuda(int chunk_size, at::Tensor noop_flag,
                                       std::vector<std::vector<at::Tensor>> tensor_lists,
                                       at::Tensor lr, const float beta1, const float beta2,
                                       const float epsilon, at::Tensor step, const int mode,
                                       const int bias_correction, const float weight_decay,
                                       at::Tensor inv_scale) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [tensor_lists_ptr, num_lists, num_tensors] = makeTransformerEngineTensor(tensor_lists);
  auto lr_cu = makeTransformerEngineTensor(lr);
  auto step_cu = makeTransformerEngineTensor(step);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_adam_capturable_cuda(
    chunk_size, noop_flag_cu.data(), tensor_lists_ptr, num_lists, num_tensors,
    lr.data(), beta1, beta2, epsilon, step.data(), mode, bias_correction, weight_decay,
    inv_scale_cu.data()
  );
}

void multi_tensor_adam_capturable_master_cuda(int chunk_size, at::Tensor noop_flag,
                                              std::vector<std::vector<at::Tensor>> tensor_lists,
                                              at::Tensor lr, const float beta1, const float beta2,
                                              const float epsilon, at::Tensor step, const int mode,
                                              const int bias_correction, const float weight_decay,
                                              at::Tensor inv_scale) {
  using namespace transformer_engine;
  using namespace transformer_engine::pytorch;

  auto noop_flag_cu = makeTransformerEngineTensor(noop_flag);
  auto [tensor_lists_ptr, num_lists, num_tensors] = makeTransformerEngineTensor(tensor_lists);
  auto lr_cu = makeTransformerEngineTensor(lr);
  auto step_cu = makeTransformerEngineTensor(step);
  auto inv_scale_cu = makeTransformerEngineTensor(inv_scale);

  nvte_multi_tensor_adam_capturable_master_cuda(
    chunk_size, noop_flag_cu.data(), tensor_lists_ptr, num_lists, num_tensors,
    lr.data(), beta1, beta2, epsilon, step.data(), mode, bias_correction, weight_decay,
    inv_scale_cu.data()
  );
}
