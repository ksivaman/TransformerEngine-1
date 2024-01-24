# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for GEMM extensions"""
import os
from typing import Optional, Tuple, Union

import torch
import transformer_engine_extensions as tex
from transformer_engine.common.recipe import Format

from ..fp8 import _default_sf_compute
from ..constants import TE_DType
from ..utils import assert_dim_for_fp8_exec


__all__ = ['gemm', 'fp8_gemm']


def fp8_gemm(
    A: torch.Tensor,
    A_scale_inv: torch.Tensor,
    A_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    A_dtype: tex.DType,
    B: torch.Tensor,
    B_scale_inv: torch.Tensor,
    B_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    B_dtype: tex.DType,
    out_dtype: torch.dtype,
    workspace: torch.Tensor,
    gelu: bool = False,
    accumulate: bool = False,
    out: Optional[torch.Tensor] = None,
    out_index = None,
    fp8_meta_tensor: tex.FP8TensorMeta = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
    ub_algo: tex.UbufOverlapAlgo = None,
    ub: Union[tex.UbufCommOverlap, tex.UbufP2PCommOverlap] = None,
    extra_output_tensor: torch.Tensor = None,
) -> torch.Tensor:
    """TN layout GEMM with fp8 inputs."""

    empty_tensor = torch.Tensor()
    if D_dtype is not None and D_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
        assert fp8_meta_tensor is not None and out_index is not None
    assert_dim_for_fp8_exec(A)
    assert_dim_for_fp8_exec(B)

    if out is None:
        out = torch.empty(
            B.shape[0],
            A.shape[0],
            dtype=out_dtype,
            device="cuda",
        )

    # Use bfloat16 as default bias_dtype
    bias_dtype = torch.bfloat16 if bias is None else bias.dtype
    if gelu:
        gelu_input = torch.empty_like(out, dtype=bias_dtype)
    else:
        gelu_input = empty_tensor
    bias_dtype = TE_DType[bias_dtype]

    out_dtype = TE_DType[out.dtype] if D_dtype is None else D_dtype

    args = (
        A,
        A_scale_inv,
        A_fp8_tensor,
        A_dtype,
        True,  # transa
        B,
        B_scale_inv,
        B_fp8_tensor,
        B_dtype,
        False,  # transb
        out,
        empty_tensor if out_index is None else fp8_meta_tensor.scale[out_index],
        out_dtype,
        empty_tensor if out_index is None else fp8_meta_tensor.amax_history[0][out_index],
        bias if use_bias else empty_tensor,
        bias_dtype,
        gelu_input,  # this is pre_gelu_out
        False,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator)
    fn = torch.ops.tex_ts.te_gemm_ts
    if ub_algo is not None:
        assert ub is not None, 'ub object is None!'
        if ub_algo == tex.UbufOverlapAlgo.BULK_OVERLAP_AG:
            fn = ub.bulk_overlap
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (1, extra_output_tensor,))
        elif ub_algo == tex.UbufOverlapAlgo.BULK_OVERLAP_RS:
            fn = ub.bulk_overlap
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (0, extra_output_tensor,))
        elif ub_algo == tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG:
            fn = ub.split_overlap_ag
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.UbufOverlapAlgo.ATOMIC_GEMM_AG:
            fn = ub.atomic_gemm_overlap_ag
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS:
            fn = ub.split_overlap_rs
            assert (
                extra_output_tensor is not None
            ), 'SPLIT_PIPELINED_RS requires extra output tensor'
            args = tuple(args + (True, extra_output_tensor,))
        elif ub_algo == tex.UbufOverlapAlgo.ATOMIC_GEMM_RS:
            fn = ub.atomic_gemm_overlap_rs
            assert (
                extra_output_tensor is not None
            ), 'ATOMIC_GEMM_RS requires extra output tensor'
            args = tuple(args + (True, extra_output_tensor,))
    _ = fn(*args)

    return out, gelu_input


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    dtype: torch.dtype,
    workspace: torch.Tensor,
    gelu: bool = False,
    gelu_input: Optional[torch.Tensor] = None,
    grad: bool = False,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    ub_algo: tex.UbufOverlapAlgo = None,
    ub: tex.UbufCommOverlap = None,
    extra_output_tensor: torch.Tensor = None,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Non FP8 GEMM."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"

    if out is None:
        out = torch.empty(
            B.shape[1] if transb else B.shape[0],
            A.shape[0] if transa else A.shape[1],
            dtype=dtype,
            device="cuda",
        )

    # Optional current scaling FP8 recipe for debug mode.
    fp8_current_scaling_recipe = os.getenv("NVTE_FP8_CURRENT_SCALING_RECIPE", "")
    if fp8_current_scaling_recipe in ["E4M3", "HYBRID"]:
        assert ub_algo is None, "Userbuf unsupported with current scaling."
        assert (
            bool(int(os.getenv("NVTE_BIAS_GELU_NVFUSION", "1")))
        ), "GEMM-gelu fusion not available for FP8."

        return fp8_gemm_current_scaling(
            A, B, workspace, out, fp8_current_scaling_recipe,
            grad, accumulate, layout, bias, use_bias)

    empty_tensor = torch.Tensor()

    if gelu and not grad:
        gelu_input = torch.empty_like(out, dtype=dtype)
    elif not gelu:
        gelu_input = empty_tensor

    if grad and use_bias:
        grad_bias = torch.empty(B.shape[1], dtype=out.dtype, device="cuda")
    else:
        grad_bias = empty_tensor

    bias = bias if use_bias else empty_tensor

    assert A.dtype == dtype and B.dtype == dtype, \
        f'Expected dtype={dtype}, but found A.dtype={A.dtype} and B.dtype={B.dtype}'
    input_dtype = TE_DType[dtype]
    output_dtype = TE_DType[out.dtype]
    if use_bias:
        bias_dtype = TE_DType[grad_bias.dtype] if grad else TE_DType[bias.dtype]
    else:
        bias_dtype = output_dtype

    args = (
        A,
        empty_tensor,
        -1, # unused FP8 index
        input_dtype,
        transa,
        B,
        empty_tensor,
        -1, # unused FP8 index
        input_dtype,
        transb,
        out,
        empty_tensor, # out_scale
        output_dtype,
        empty_tensor, # out_amax
        grad_bias if grad else bias,
        bias_dtype,
        gelu_input,
        grad,
        workspace,
        workspace.shape[0],
        accumulate,
        False,  # use_split_accumulator
    )
    fn = torch.ops.tex_ts.te_gemm_ts
    if ub_algo is not None:
        assert ub is not None, 'ub object is None!'
        if ub_algo == tex.UbufOverlapAlgo.BULK_OVERLAP_AG:
            fn = ub.bulk_overlap
            args = tuple(args + (1, empty_tensor))
        elif ub_algo == tex.UbufOverlapAlgo.BULK_OVERLAP_RS:
            fn = ub.bulk_overlap
            args = tuple(args + (0, empty_tensor))
        elif ub_algo == tex.UbufOverlapAlgo.SPLIT_PIPELINED_AG:
            fn = ub.split_overlap_ag
            extra_output_tensor = (
                empty_tensor if extra_output_tensor is None else extra_output_tensor
            )
            args = tuple(args + (extra_output_tensor,))
        elif ub_algo == tex.UbufOverlapAlgo.SPLIT_PIPELINED_RS:
            fn = ub.split_overlap_rs
            assert (
                extra_output_tensor is not None
            ), 'SPLIT_PIPELINED_RS requires extra output tensor'
            args = tuple(args + (False, extra_output_tensor,))
    _ = fn(*args)

    return out, grad_bias, gelu_input


def fp8_cast(tensor, recipe, grad_tensor, margin=0):
    assert tensor.dtype in (torch.float, torch.float16, torch.bfloat16), "Unsupported tensor type."
    assert tensor.is_cuda, "Must be a GPU tensor."

    if recipe == "HYBRID" and grad_tensor:
        fp8_dtype = tex.DType.kFloat8E5M2
        fp8_max = Format.E5M2.value.max_fwd
    else:
        fp8_dtype = tex.DType.kFloat8E4M3
        fp8_max = Format.E4M3.value.max_fwd

    amax = torch.max(torch.abs(tensor)).float()
    one = torch.ones(1, device="cuda")

    scale = _default_sf_compute(amax, one, fp8_max, margin)
    scale_inv = 1.0 / scale

    fp8_tensor = tex.cast_to_fp8(tensor, scale, amax, scale_inv, fp8_dtype)
    return fp8_tensor, fp8_dtype, scale_inv


def fp8_gemm_current_scaling(
    A: torch.Tensor,
    B: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
    recipe: str,
    grad: bool = False,
    accumulate: bool = False,
    layout: str = "TN",
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Non FP8 GEMM."""

    empty_tensor = torch.Tensor()

    if grad and use_bias:
        # Unfused bgrad
        grad_bias = B.sum(0).squeeze(0)
    else:
        grad_bias = empty_tensor

    A = A.t().contiguous() if layout[0] == "N" else A
    B = B.t().contiguous() if layout[1] == "T" else B

    bias = bias if (use_bias and bias is not None) else empty_tensor

    output_dtype = TE_DType[out.dtype]
    if use_bias:
        bias_dtype = TE_DType[grad_bias.dtype] if grad else TE_DType[bias.dtype]
    else:
        bias_dtype = output_dtype

    # Prepare FP8.
    A_fp8, A_dtype, A_scale_inv = fp8_cast(A, recipe, False)
    B_fp8, B_dtype, B_scale_inv = fp8_cast(B, recipe, grad) # Only B is grad tensor

    _ = torch.ops.tex_ts.te_gemm_ts(
        A_fp8,
        A_scale_inv,
        0,
        A_dtype,
        True,  # transa
        B_fp8,
        B_scale_inv,
        0,
        B_dtype,
        False,  # transb
        out,
        empty_tensor, # out_scale
        output_dtype,
        empty_tensor, # out_amax
        bias,
        bias_dtype,
        empty_tensor, # gelu_input
        False, # grad
        workspace,
        workspace.shape[0],
        accumulate,
        grad,  # use_split_accumulator
    )

    return out, grad_bias, empty_tensor
