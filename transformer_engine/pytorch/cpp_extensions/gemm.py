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

print("NVTE_FP8_CURRENT_SCALING_RECIPE: ", os.getenv("NVTE_FP8_CURRENT_SCALING_RECIPE", ""))
print("NVTE_DEBUG_DGRAD_CURR_AMAX_GRADIENTS: ", os.getenv("NVTE_DEBUG_DGRAD_CURR_AMAX_GRADIENTS", "0"))
print("NVTE_DEBUG_DGRAD_CURR_AMAX_WEIGHTS: ", os.getenv("NVTE_DEBUG_DGRAD_CURR_AMAX_WEIGHTS", "0"))
print("NVTE_DEBUG_AMAX_CORRECTION: ", os.getenv("NVTE_DEBUG_AMAX_CORRECTION", "0"))
print("NVTE_DEBUG_AMAX_CORRECTION_SATURATION: ", os.getenv("NVTE_DEBUG_AMAX_CORRECTION_SATURATION", "0"))
print("NVTE_DEBUG_FP8_MARGIN" , os.getenv("NVTE_DEBUG_FP8_MARGIN", "0"))

import random
print("NVTE_DEBUG_FP8_MANTISSA_SWITCH" , os.getenv("NVTE_DEBUG_FP8_MANTISSA_SWITCH", "0"))

margin_printed = False
layernorm_linear_printed = False
linear_printed = False
mantissa_switch_printed = False
__all__ = ['gemm', 'fp8_gemm', 'fp8_gemm_experimental']


def fp8_gemm_experimental(
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
    weights_bf16: torch.Tensor = None,
    gradients_bf16: torch.Tensor = None,
    fp8_meta_info = None,
    layer_name = None,
) -> torch.Tensor:
    """TN layout GEMM with fp8 inputs."""
    
    global layernorm_linear_printed
    global linear_printed
    assert ub_algo is None, "Userbuf unsupported with fp8 gemm experimental."
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


    fp8_current_scaling_recipe = os.getenv("NVTE_FP8_CURRENT_SCALING_RECIPE", "")
    fp8_current_scaling_gradients = int(os.getenv("NVTE_DEBUG_DGRAD_CURR_AMAX_GRADIENTS", "0")) and (gradients_bf16 is not None)
    fp8_current_scaling_weights = int(os.getenv("NVTE_DEBUG_DGRAD_CURR_AMAX_WEIGHTS", "0")) and (weights_bf16 is not None)
    fp8_amax_correction = int(os.getenv("NVTE_DEBUG_AMAX_CORRECTION", "0"))
    fp8_amax_correction_saturation = int(os.getenv("NVTE_DEBUG_AMAX_CORRECTION_SATURATION", "0"))
    fp8_mantissa_switch = int(os.getenv("NVTE_DEBUG_FP8_MANTISSA_SWITCH", "0"))
    # DGRAD LAYOUT is "NN" in layernorm_linear and linear layers
    if fp8_current_scaling_recipe in ["E4M3", "HYBRID"] and (fp8_current_scaling_gradients or fp8_current_scaling_weights):
        assert (
            bool(int(os.getenv("NVTE_BIAS_GELU_NVFUSION", "1")))
        ), "GEMM-gelu fusion not available for FP8."

        if fp8_current_scaling_weights:
            # Layout is "NN" for dgrad
            #print("WEIGHTS CURRENT SCALING ON")
            weights_bf16_t = weights_bf16.t().contiguous()
            weights_fp8, weights_dtype, weights_scale_inv = fp8_cast(weights_bf16_t, fp8_current_scaling_recipe, False)
            weights_fp8_tensor = 0
        else:
            #print("WEIGHTS CURRENT SCALING OFF")
            weights_fp8 = A
            weights_dtype = A_dtype
            weights_scale_inv = A_scale_inv
            weights_fp8_tensor = A_fp8_tensor

        if fp8_current_scaling_gradients:
            # Layout is NN for dgrad
            # Prepare FP8.
            #print("GRADIENTS CURRENT SCALING ON")
            if not layernorm_linear_printed and layer_name == "layernorm_linear":
                print("CURRENT SCALING DGRAD GRADIENT SELECTED FOR LAYERNORM LINEAR")
                layernorm_linear_printed = True
            if not linear_printed and layer_name == "linear":
                print("CURRENT SCALING DGRAD GRADIENT SELECTED FOR LINEAR")
                linear_printed = True
            if fp8_amax_correction > 0:
                curr_amax = torch.max(torch.abs(gradients_bf16)).float()
                hist_amax = torch.max(fp8_meta_info['scaling_bwd'].amax_history[1:], dim=0).values[0]
                #print("Base Size: ", torch.max(fp8_meta_info['scaling_bwd'].amax_history, dim=0))
                #print("Leave First: ", torch.max(fp8_meta_info['scaling_bwd'].amax_history[1:], dim=0))
                #print(f"CURR AMAX={curr_amax}, HIST AMAX={hist_amax}")
                if (hist_amax / curr_amax) >= (2**fp8_amax_correction):
                    #print(f"Using Current Amax: hist_amax={hist_amax}, curr_amax={curr_amax}, Correction Factor Threshold={2**fp8_amax_correction}")
                    gradients_fp8, gradients_dtype, gradients_scale_inv = fp8_cast(gradients_bf16, fp8_current_scaling_recipe, True) # Only B is grad tensor
                    gradients_fp8_tensor = 0
                else:
                    gradients_fp8 = B
                    gradients_dtype = B_dtype
                    gradients_scale_inv = B_scale_inv
                    gradients_fp8_tensor = B_fp8_tensor
            elif fp8_amax_correction_saturation > 0:
                curr_amax = torch.max(torch.abs(gradients_bf16)).float()
                hist_amax = torch.max(fp8_meta_info['scaling_bwd'].amax_history, dim=0).values[0]
                #print(f"CURR AMAX={curr_amax}, HIST AMAX={hist_amax}")
                if curr_amax > hist_amax:
                    print(f"Using Current Amax: hist_amax={hist_amax}, curr_amax={curr_amax}, Correction for Saturation")
                    gradients_fp8, gradients_dtype, gradients_scale_inv = fp8_cast(gradients_bf16, fp8_current_scaling_recipe, True) # Only B is grad tensor
                    gradients_fp8_tensor = 0
                else:
                    gradients_fp8 = B
                    gradients_dtype = B_dtype
                    gradients_scale_inv = B_scale_inv
                    gradients_fp8_tensor = B_fp8_tensor    
            else:
                gradients_fp8, gradients_dtype, gradients_scale_inv = fp8_cast(gradients_bf16, fp8_current_scaling_recipe, True) # Only B is grad tensor
                gradients_fp8_tensor = 0
        else:
            #print("GRADIENTS CURRENT SCALING OFF")
            gradients_fp8 = B
            gradients_dtype = B_dtype
            gradients_scale_inv = B_scale_inv
            gradients_fp8_tensor = B_fp8_tensor

        args = (
            weights_fp8,
            weights_scale_inv,
            weights_fp8_tensor,
            weights_dtype,
            True,  # transa
            gradients_fp8,
            gradients_scale_inv,
            gradients_fp8_tensor,
            gradients_dtype,
            False,  # transb
            out,
            empty_tensor if out_index is None else fp8_meta_tensor.scale[out_index],
            out_dtype,
            empty_tensor if out_index is None else fp8_meta_tensor.amax_history[0][out_index],
            bias if use_bias else empty_tensor,
            bias_dtype,
            gelu_input,
            False, # grad
            workspace,
            workspace.shape[0],
            accumulate,
            use_split_accumulator)
    elif fp8_mantissa_switch:
        global mantissa_switch_printed
        if not mantissa_switch_printed:
            print(F"MANTISSA SWITCH ENABLED FOR DGRAD")
            mantissa_switch_printed = True
        if not layernorm_linear_printed and layer_name == "layernorm_linear":
            print("MANTISSA SWITCH SELECTED FOR LAYERNORM LINEAR")
            layernorm_linear_printed = True
        if not linear_printed and layer_name == "linear":
            print("MANTISSA SWITCH SELECTED FOR LINEAR")
            linear_printed = True
        hist_amax = torch.max(fp8_meta_info['scaling_bwd'].amax_history[1:], dim=0).values[0]
        gradients_fp8, gradients_dtype, gradients_scale_inv = fp8_cast_mantissa_switch(gradients_bf16, fp8_current_scaling_recipe, True, hist_amax) # Only B is grad tenso
        gradients_fp8_tensor = 0
        args = (
            A,
            A_scale_inv,
            A_fp8_tensor,
            A_dtype,
            True,  # transa
            gradients_fp8,
            gradients_scale_inv,
            gradients_fp8_tensor,
            gradients_dtype,
            False,  # transb
            out,
            empty_tensor if out_index is None else fp8_meta_tensor.scale[out_index],
            out_dtype,
            empty_tensor if out_index is None else fp8_meta_tensor.amax_history[0][out_index],
            bias if use_bias else empty_tensor,
            bias_dtype,
            gelu_input,
            False, # grad
            workspace,
            workspace.shape[0],
            accumulate,
            use_split_accumulator)
    else:
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
    _ = fn(*args)

    return out, gelu_input

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
    global margin_printed

    if recipe == "HYBRID" and grad_tensor:
        fp8_dtype = tex.DType.kFloat8E5M2
        fp8_max = Format.E5M2.value.max_fwd
    else:
        fp8_dtype = tex.DType.kFloat8E4M3
        fp8_max = Format.E4M3.value.max_fwd

    if (margin == 0) and (int(os.getenv("NVTE_DEBUG_FP8_MARGIN", "0")) != 0):
        margin = int(os.getenv("NVTE_DEBUG_FP8_MARGIN", "0"))
        if not margin_printed:
            print(F"MARGIN SETTING CALLED = {margin}")
            margin_printed = True

    amax = torch.max(torch.abs(tensor)).float()
    one = torch.ones(1, device="cuda")

    scale = _default_sf_compute(amax, one, fp8_max, margin)
    scale_inv = 1.0 / scale

    fp8_tensor = tex.cast_to_fp8(tensor, scale, amax, scale_inv, fp8_dtype)
    return fp8_tensor, fp8_dtype, scale_inv

def sf_mantissa_switch_compute(
    amax: torch.Tensor,
    scale: torch.Tensor,
    fp8_max: float,
    margin: int,
) -> torch.Tensor:
    """Default function to convert amax to scaling factor."""
    # sf = (fp8_max / amax) / (2 ** margin)
    # sf = torch.where(amax > 0.0, sf, scale)
    # sf = torch.where(torch.isfinite(amax), sf, scale)
    exp = torch.floor(torch.log2(fp8_max / amax)) - margin
    #print(f"amax_shape: {amax.shape}, exp: {exp}")
    sf = torch.round(torch.pow(2, torch.abs(exp)))
    sf = torch.where(exp < 0, 1 / sf, sf)
    #cnt = sf.numel()
    #for i in range(cnt):
    extra_sf = random.choice([1.     , 0.96875, 0.9375 , 0.90625, 0.875  , 0.84375, 0.8125 ,
        0.78125, 0.75   , 0.71875, 0.6875 , 0.65625, 0.625  , 0.59375,
        0.5625 , 0.53125])
    sf = sf * extra_sf
    #sf[i] = sf[i] * extra_sf
    sf = torch.where(amax > 0.0, sf, scale)
    sf = torch.where(torch.isfinite(amax), sf, scale)
    return sf


def fp8_cast_mantissa_switch(tensor, recipe, grad_tensor, hist_amax, margin=0):
    assert tensor.dtype in (torch.float, torch.float16, torch.bfloat16), "Unsupported tensor type."
    assert tensor.is_cuda, "Must be a GPU tensor."

    if recipe == "HYBRID" and grad_tensor:
        fp8_dtype = tex.DType.kFloat8E5M2
        fp8_max = Format.E5M2.value.max_fwd
    else:
        fp8_dtype = tex.DType.kFloat8E4M3
        fp8_max = Format.E4M3.value.max_fwd

    one = torch.ones(1, device="cuda")

    scale = sf_mantissa_switch_compute(hist_amax, one, fp8_max, margin)
    scale_inv = 1.0 / scale

    fp8_tensor = tex.cast_to_fp8(tensor, scale, hist_amax, scale_inv, fp8_dtype)
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
