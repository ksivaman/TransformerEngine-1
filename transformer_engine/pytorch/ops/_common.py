# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper functions used in fusible operations."""

from __future__ import annotations
import functools
from importlib.metadata import PackageNotFoundError, version as get_pkg_version
from typing import Optional

import torch
from packaging.version import Version as PkgVersion

import transformer_engine_torch as tex

from transformer_engine_torch import FP8TensorMeta
from ..torch_version import torch_version
from ..quantization import FP8GlobalStateManager
from ..tensor.float8_tensor import Float8Tensor
from ..quantized_tensor import QuantizedTensorStorage
from ..utils import canonicalize_dtype
from ..tensor import Quantizer
from ..tensor.grouped_tensor import GroupedTensor


@functools.lru_cache(maxsize=1)
def _nvidia_cudnn_frontend_supports_scaled_clamped_qgeglu() -> bool:
    """Check cuDNN FE min version with fixed numerics for qgeglu."""
    try:
        return PkgVersion(get_pkg_version("nvidia-cudnn-frontend")) >= PkgVersion("1.23.0")
    except PackageNotFoundError:
        return False


@functools.lru_cache(maxsize=1)
def _nvidia_cudnn_frontend_supports_wgrad() -> bool:
    """Check cuDNN FE min version for grouped GEMM wgrad kernel."""
    try:
        return PkgVersion(get_pkg_version("nvidia-cudnn-frontend")) >= PkgVersion("1.23.0")
    except PackageNotFoundError:
        return False


def is_quantized_tensor(tensor: torch.Tensor | QuantizedTensorStorage) -> bool:
    """Check if tensor is a quantized tensor"""
    return isinstance(tensor, QuantizedTensorStorage)


def maybe_dequantize(
    tensor: torch.Tensor | QuantizedTensorStorage, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize tensor to given dtype or just convert if not a quantized tensor"""
    if is_quantized_tensor(tensor):
        return tensor.dequantize(dtype=dtype)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def maybe_autocast_dtype(
    *,
    device_type: str = "cuda",
    default_dtype: Optional[torch.dtype] = None,
) -> torch.dtype:
    """Get autocast dtype if enabled"""

    if torch_version() >= (2, 4, 3):
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    else:
        if torch.is_autocast_enabled():
            return torch.get_autocast_gpu_dtype()
    return canonicalize_dtype(default_dtype)


def get_fp8_meta_from_fp8_tensor(tensor: Float8Tensor) -> tuple[FP8TensorMeta, int]:
    """Get FP8TensorMeta object and index corresponding to Float8Tensor

    Constructs FP8TensorMeta if needed.

    """

    # Check if tensor already has FP8 metadata
    if tensor._fp8_meta is not None:
        key = FP8GlobalStateManager.get_meta_tensor_key(
            forward=tensor._fp8_meta_forward,
        )
        return tensor._fp8_meta[key], tensor._fp8_meta_index

    # Create FP8TensorMeta class
    fp8_meta = FP8TensorMeta()
    fp8_meta.scale = tensor._scale_inv.reciprocal()
    fp8_meta.amax_history = torch.empty(1, 1, dtype=torch.float32, device=tensor.device)
    fp8_meta.scale_inv = tensor._scale_inv
    return fp8_meta, 0


@functools.lru_cache(maxsize=None)
def _get_trivial_tensor_offsets(M: int, K: int, device_index: int) -> torch.Tensor:
    device = f"cuda:{device_index}"
    return torch.cat([
        torch.zeros(1, dtype=torch.int64, device=device),
        torch.full((1,), M * K, dtype=torch.int64, device=device),
    ])


def validate_grouped_mlp_dims(fc1, glu_op, fc2) -> None:
    """Validate FC1 / scaled GLU / FC2 dimensions for fused grouped MLP."""

    if fc1.in_features % 64 != 0 or fc1.out_features % 64 != 0:
        raise ValueError(
            f"Unsupported dims for FC1 (num_groups={fc1.num_groups}, "
            f"in_features={fc1.in_features}, out_features={fc1.out_features})."
        )
    if fc2.in_features % 64 != 0 or fc2.out_features % 64 != 0:
        raise ValueError(
            f"Unsupported dims for FC2 (num_groups={fc2.num_groups}, "
            f"in_features={fc2.in_features}, out_features={fc2.out_features})."
        )
    if fc1.out_features != 2 * fc2.in_features or fc1.num_groups != fc2.num_groups:
        raise ValueError(
            f"FC1 (num_groups={fc1.num_groups}, in_features={fc1.in_features}, "
            f"out_features={fc1.out_features}) "
            f"and FC2 (num_groups={fc2.num_groups}, in_features={fc2.in_features}, "
            f"out_features={fc2.out_features}) do not match."
        )
    if glu_op.glu_interleave_size != 32:
        raise ValueError(
            "Fused kernel requires 32-wide GLU interleaving, "
            f"but got glu_interleave_size={glu_op.glu_interleave_size}."
        )


@functools.lru_cache(maxsize=None)
def _get_trivial_split_points(M: int, device_index: int) -> torch.Tensor:
    """Return a single-element int32 tensor [M] — the split point for a single group."""
    return torch.tensor([M], dtype=torch.int32, device=f"cuda:{device_index}")


def group_quantize(
    tensor: torch.Tensor,
    quantizer: "Quantizer",
    num_groups: int,
    split_sizes,
    tensor_offsets=None,
) -> "GroupedTensor":
    """Quantize tensor into a GroupedTensor.

    For num_groups==1, uses tex.quantize (single-tensor path) which calls
    nvte_quantize_with_hadamard_transform when with_rht=True, avoiding the
    heavier group_row_col_rht_gemm_device_graph_safe grouped kernel.
    For num_groups>1, uses tex.group_quantize (original behavior).
    """
    if num_groups != 1:
        return tex.group_quantize(tensor, quantizer, num_groups, split_sizes)

    # Single-group path: tex.quantize handles with_rht via nvte_quantize_with_hadamard_transform
    quantized = tex.quantize(tensor, quantizer)
    rowwise_data = getattr(quantized, "_rowwise_data", None)
    rowwise_scale = getattr(quantized, "_rowwise_scale_inv", None)
    columnwise_data = getattr(quantized, "_columnwise_data", None)
    columnwise_scale = getattr(quantized, "_columnwise_scale_inv", None)
    amax = getattr(quantized, "_amax_rowwise", None)
    amax_columnwise = getattr(quantized, "_amax_columnwise", None)

    if split_sizes is None:
        split_sizes = torch.full((1,), tensor.shape[0], dtype=torch.int64, device=tensor.device)

    M = tensor.shape[0]
    # RHT may pad K; derive K_padded from quantized data (mirrors NVFP4Tensor.shape property:
    # byte_shape[-1] * 2). M is never padded by RHT.
    if rowwise_data is not None:
        K = rowwise_data.shape[-1] * 2  # FP4x2: each element holds 2 FP4 values
    elif columnwise_data is not None:
        K = columnwise_data.shape[0]    # columnwise_data shape is [K, M//2]
    else:
        K = tensor.shape[-1]

    if getattr(quantizer, "optimize_for_gemm", False):
        tex.swizzle_scales_for_gemm_(quantized)
        rowwise_scale = getattr(quantized, "_rowwise_scale_inv", None)
        columnwise_scale = getattr(quantized, "_columnwise_scale_inv", None)

    if tensor_offsets is None:
        tensor_offsets = _get_trivial_tensor_offsets(M, K, tensor.device.index)

    with_gemm_swizzled_scales = getattr(quantizer, "optimize_for_gemm", False)
    grouped = GroupedTensor(
        shape=(M, K),
        dtype=tensor.dtype,
        quantizer=quantizer,
        num_tensors=1,
        data=rowwise_data.reshape(-1) if rowwise_data is not None else None,
        columnwise_data=columnwise_data.reshape(-1) if columnwise_data is not None else None,
        scale_inv=rowwise_scale.reshape(-1) if rowwise_scale is not None else None,
        columnwise_scale_inv=columnwise_scale.reshape(-1) if columnwise_scale is not None else None,
        amax=None,
        columnwise_amax=None,
        scale=None,
        first_dims=split_sizes,
        last_dims=None,
        tensor_offsets=tensor_offsets,
        offsets=None,
        scale_inv_offsets=None,
        columnwise_scale_inv_offsets=None,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )
    if amax is not None:
        grouped.amax = amax
    if amax_columnwise is not None:
        grouped.columnwise_amax = amax_columnwise
    return grouped


def make_grouped_tensor_from_buffers(
    *,
    num_groups: int,
    data: torch.Tensor,
    split_sizes: torch.Tensor,
    columnwise_data: torch.Tensor = None,
    scale_inv: torch.Tensor = None,
    columnwise_scale_inv: torch.Tensor = None,
    tensor_offsets: torch.Tensor = None,
    logical_last_dim: int,
    dtype: torch.dtype,
    quantizer: Quantizer = None,
    with_gemm_swizzled_scales: bool = False,
) -> GroupedTensor:
    """Build GroupedTensor from FC1+SwiGLU / dSwiGLU kernel outputs.

    Scales are already in GEMM swizzled layout.
    """
    if tensor_offsets is None:
        tensor_offsets = GroupedTensor.make_tensor_offsets(split_sizes, logical_last_dim)
    logical_first_dim = data.shape[0] if data is not None else columnwise_data.shape[0]
    ndim = data.ndim if data is not None else columnwise_data.ndim
    if ndim == 1:
        logical_first_dim = logical_first_dim // logical_last_dim
    return GroupedTensor(
        shape=(logical_first_dim, logical_last_dim),
        dtype=dtype,
        quantizer=quantizer,
        num_tensors=num_groups,
        data=data,
        columnwise_data=columnwise_data,
        scale_inv=scale_inv,
        columnwise_scale_inv=columnwise_scale_inv,
        amax=None,
        columnwise_amax=None,
        scale=None,
        first_dims=split_sizes,
        last_dims=None,
        tensor_offsets=tensor_offsets,
        offsets=None,
        scale_inv_offsets=None,
        columnwise_scale_inv_offsets=None,
        with_gemm_swizzled_scales=with_gemm_swizzled_scales,
    )


def fuse_grouped_mlp_ops(
    ops,
    *,
    recipe,
    fused_op_cls,
):
    """Sliding-window fusion for GroupedLinear + scaled GLU + GroupedLinear.

    Parameters
    ----------
    ops : list of FusibleOperation
        Operations to scan.
    recipe : Recipe or None
        Quantization recipe.
    fused_op_cls : type
        Fused operation class with ``is_supported()`` classmethod and
        constructor accepting ``fc1``, ``glu_op``, ``fc2`` keyword args. The
        ``glu_op`` must be :class:`~transformer_engine.pytorch.ops.basic.swiglu.ScaledSwiGLU`
        or :class:`~transformer_engine.pytorch.ops.basic.swiglu.ScaledClampedQGeGLU`.
        May also expose ``is_fc1_bias_supported()`` and/or
        ``is_fc2_bias_supported()`` classmethods for bias eligibility.

    Returns
    -------
    list of FusibleOperation
        Updated operations with matched triples replaced by fused ops.
    """
    from .basic import (  # pylint: disable=import-outside-toplevel
        GroupedLinear,
        ScaledClampedQGeGLU,
        ScaledSwiGLU,
    )

    if not fused_op_cls.is_supported():
        return ops
    if recipe is None or not recipe.mxfp8():
        return ops

    fc1_bias_ok = (
        not hasattr(fused_op_cls, "is_fc1_bias_supported") or fused_op_cls.is_fc1_bias_supported()
    )
    fc2_bias_ok = (
        not hasattr(fused_op_cls, "is_fc2_bias_supported") or fused_op_cls.is_fc2_bias_supported()
    )

    out = []
    window, ops = ops[:3], ops[3:]
    while len(window) == 3:

        matches_pattern = True
        if not (
            isinstance(window[0], GroupedLinear)
            and isinstance(window[1], (ScaledSwiGLU, ScaledClampedQGeGLU))
            and isinstance(window[2], GroupedLinear)
        ):
            matches_pattern = False
        elif isinstance(window[1], ScaledClampedQGeGLU) and (
            abs(window[1]._clamped.alpha - 1.702) > 0.001
            or not _nvidia_cudnn_frontend_supports_scaled_clamped_qgeglu()
        ):
            matches_pattern = False
        elif window[0].num_groups != window[2].num_groups:
            matches_pattern = False
        elif (
            window[0].in_features % 64 != 0
            or window[0].out_features % 64 != 0
            or window[2].in_features % 64 != 0
            or window[2].out_features % 64 != 0
        ):
            matches_pattern = False
        elif window[1].glu_interleave_size != 32:
            matches_pattern = False
        elif window[0].has_bias and not fc1_bias_ok:
            matches_pattern = False
        elif window[2].has_bias and not fc2_bias_ok:
            matches_pattern = False

        if matches_pattern:
            op = fused_op_cls(
                fc1=window[0],
                swiglu=window[1],
                fc2=window[2],
            )
            window = [op]
        else:
            out.extend(window[:-2])
            window = window[-2:]

        out.extend(window[:-3])
        window = window[-3:]
        while ops and len(window) < 3:
            window.append(ops[0])
            ops = ops[1:]

    out.extend(window)
    return out
