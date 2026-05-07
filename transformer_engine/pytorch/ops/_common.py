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
def _cudnn_frontend_version_supported() -> bool:
    """Check cuDNN frontend is at least 1.23.0.

    All grouped MLP fused-kernel features require cuDNN frontend 1.23.0.
    """
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


def validate_grouped_mlp_dims(fc1, act_op, fc2) -> None:
    """Validate FC1 / scaled activation / FC2 dimensions for fused grouped MLP.

    Supports both gated (SwiGLU/QGeGLU) and non-gated (SReLU) activations.
    For gated activations FC1's output features must be twice FC2's input
    features; for non-gated activations they must be equal.
    """
    from .basic import (  # pylint: disable=import-outside-toplevel
        ScaledClampedQGeGLU,
        ScaledSReLU,
        ScaledSwiGLU,
    )

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

    is_gated = isinstance(act_op, (ScaledSwiGLU, ScaledClampedQGeGLU))
    expected_fc1_out = (2 if is_gated else 1) * fc2.in_features
    if fc1.out_features != expected_fc1_out or fc1.num_groups != fc2.num_groups:
        raise ValueError(
            f"FC1 (num_groups={fc1.num_groups}, in_features={fc1.in_features}, "
            f"out_features={fc1.out_features}) "
            f"and FC2 (num_groups={fc2.num_groups}, in_features={fc2.in_features}, "
            f"out_features={fc2.out_features}) do not match for "
            f"{type(act_op).__name__} (expected FC1 out={expected_fc1_out})."
        )
    if is_gated and act_op.glu_interleave_size != 32:
        raise ValueError(
            "Fused kernel requires 32-wide GLU interleaving, "
            f"but got glu_interleave_size={act_op.glu_interleave_size}."
        )
    if not isinstance(act_op, (ScaledSwiGLU, ScaledClampedQGeGLU, ScaledSReLU)):
        raise ValueError(
            f"Unsupported activation type {type(act_op).__name__} for fused grouped MLP."
        )


def fuse_grouped_mlp_ops(
    ops,
    *,
    recipe,
    fused_op_cls,
):
    """Sliding-window fusion for GroupedLinear + scaled activation + GroupedLinear.

    Parameters
    ----------
    ops : list of FusibleOperation
        Operations to scan.
    recipe : Recipe or None
        Quantization recipe.
    fused_op_cls : type
        Fused operation class with constructor accepting ``fc1``, ``act_op``,
        ``fc2`` keyword args. Must expose:

        * ``supported_activation_types`` (tuple of activation classes the
          fused op can fuse).
        * ``is_supported()`` classmethod for environment / hardware /
          frontend gating.
        * ``is_activation_supported(act_op)`` classmethod for per-activation
          kernel availability.
        * ``fc1_out_factor(act_op)`` classmethod returning the ratio between
          FC1 ``out_features`` and FC2 ``in_features`` for that activation.

    Returns
    -------
    list of FusibleOperation
        Updated operations with matched triples replaced by fused ops.
    """
    from .basic import (  # pylint: disable=import-outside-toplevel
        GroupedLinear,
        ScaledClampedQGeGLU,
    )

    if not fused_op_cls.is_supported():
        return ops
    if recipe is None or not recipe.mxfp8():
        return ops

    activation_types = fused_op_cls.supported_activation_types

    out = []
    window, ops = ops[:3], ops[3:]
    while len(window) == 3:

        matches_pattern = True
        if not (
            isinstance(window[0], GroupedLinear)
            and isinstance(window[1], activation_types)
            and isinstance(window[2], GroupedLinear)
        ):
            matches_pattern = False
        elif not fused_op_cls.is_activation_supported(window[1]):
            matches_pattern = False
        elif isinstance(window[1], ScaledClampedQGeGLU) and (
            abs(window[1]._clamped.alpha - 1.702) > 0.001
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
        elif (
            window[0].out_features
            != fused_op_cls.fc1_out_factor(window[1]) * window[2].in_features
        ):
            matches_pattern = False
        elif (
            hasattr(window[1], "glu_interleave_size")
            and window[1].glu_interleave_size != 32
        ):
            matches_pattern = False

        if matches_pattern:
            op = fused_op_cls(
                fc1=window[0],
                act_op=window[1],
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
