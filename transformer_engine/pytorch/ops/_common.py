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

from transformer_engine_torch import FP8TensorMeta
from ..torch_version import torch_version
from ..quantization import FP8GlobalStateManager
from ..tensor.float8_tensor import Float8Tensor
from ..quantized_tensor import QuantizedTensorStorage
from ..utils import canonicalize_dtype


@functools.lru_cache(maxsize=1)
def _nvidia_cudnn_frontend_supports_scaled_clamped_qgeglu() -> bool:
    """Check cuDNN FE min version with fixed numerics for qgeglu."""
    return True


@functools.lru_cache(maxsize=1)
def _nvidia_cudnn_frontend_supports_wgrad() -> bool:
    """Check cuDNN FE min version for grouped GEMM wgrad kernel."""
    return True


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


def pad_grouped_weight_scale_inv_for_swizzle(
    grouped_weight,
    *,
    rows_per_tensor: int,
    cols_per_tensor: int,
    num_tensors: int,
    rowwise: bool = False,
    columnwise: bool = False,
) -> None:
    """Pad per-expert MXFP8 scale buffers to the layout expected by the grouped GEMM swizzle.

    The grouped MXFP8 quantize kernel writes per-expert scales into a single contiguous
    buffer in a "compact" layout where each expert occupies exactly its un-padded row
    count: stride between experts is ``rows_per_tensor * scale_cols``.
    ``tex.grouped_swizzle_for_gemm`` (and the grouped GEMM kernels it feeds) instead
    require a "per-tensor padded" layout where each expert's scales are padded to a
    multiple of 128 rows (``round_up(rows_per_tensor, 128)``) before being stacked.

    When ``rows_per_tensor`` is already a multiple of 128 the two layouts coincide and
    nothing is done. Otherwise the swizzle would otherwise fail with::

        Assertion failed: input->scale_inv.numel() == input->num_tensors * scale_elems

    This helper rewrites ``scale_inv`` and/or ``columnwise_scale_inv`` of
    ``grouped_weight`` in place into the per-tensor padded layout, zero-filling the
    inserted padding rows. It must be called on a shallow ``copy()`` of the weight
    (never the original, which is saved for backward).
    """
    # pylint: disable=import-outside-toplevel
    from ..constants import MXFP8_BLOCK_SCALING_SIZE

    if rows_per_tensor % 128 == 0:
        return

    padded_rows = ((rows_per_tensor + 127) // 128) * 128

    def _pad_first_dim(buffer: torch.Tensor, per_tensor_rows: int, padded_per_tensor_rows: int):
        """Reshape ``buffer`` to (num_tensors, per_tensor_rows, scale_cols), zero-pad
        the middle dim to ``padded_per_tensor_rows``, and return a contiguous flat view."""
        per_tensor_compact = per_tensor_rows * num_tensors
        # The C++ allocator may have rounded the total scale-row count up; ignore the
        # trailing slack since the kernel only writes ``per_tensor_compact`` rows.
        total_elems = buffer.numel()
        if total_elems % per_tensor_compact != 0:
            raise RuntimeError(
                "Unexpected grouped MXFP8 scale buffer size "
                f"({total_elems}) for num_tensors={num_tensors}, "
                f"per_tensor_rows={per_tensor_rows}."
            )
        scale_cols = total_elems // per_tensor_compact
        compact = buffer.view(num_tensors, per_tensor_rows, scale_cols)
        padded = torch.nn.functional.pad(
            compact, (0, 0, 0, padded_per_tensor_rows - per_tensor_rows)
        ).contiguous()
        return padded.view(-1)

    if rowwise and grouped_weight.scale_inv is not None:
        grouped_weight.scale_inv = _pad_first_dim(
            grouped_weight.scale_inv,
            per_tensor_rows=rows_per_tensor,
            padded_per_tensor_rows=padded_rows,
        )

    if columnwise and grouped_weight.columnwise_scale_inv is not None:
        scale_rows_per_tensor = rows_per_tensor // MXFP8_BLOCK_SCALING_SIZE
        padded_scale_rows = padded_rows // MXFP8_BLOCK_SCALING_SIZE
        grouped_weight.columnwise_scale_inv = _pad_first_dim(
            grouped_weight.columnwise_scale_inv,
            per_tensor_rows=scale_rows_per_tensor,
            padded_per_tensor_rows=padded_scale_rows,
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
