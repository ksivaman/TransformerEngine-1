# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for scaled SReLU."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

import transformer_engine_torch as tex
from ...cpu_offload import is_cpu_offload_enabled, mark_activation_offload
from ...tensor import Quantizer
from ...utils import clear_tensor_data
from ..op import BasicOperation, OperationContext
from .._common import maybe_dequantize

__all__ = ["ScaledSReLU"]


class ScaledSReLU(BasicOperation):
    r"""SReLU with per-row post-scaling
    (matches cuDNN grouped GEMM ``grouped_gemm_srelu`` kernel).

    The squared ReLU activation,

    .. math::

       \text{SReLU}(x) = \max(x^2, 0),

    is applied element-wise. The result is multiplied by a per-row scale
    tensor passed as an extra input. If the activation output has shape
    ``(d_1, ..., d_n)``, the scale tensor has shape ``(d_1, ..., d_{n-1})``.

    This op is intended as the middle operation in a fused grouped MLP
    (``GroupedLinear`` + ``ScaledSReLU`` + ``GroupedLinear``) and shares
    the post-scaling contract with :class:`ScaledSwiGLU` /
    :class:`ScaledClampedQGeGLU`.

    """

    num_extra_inputs: int = 1

    def op_forward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_forward` instead of `op_forward`."
        )

    def op_backward(self, *args, **kwargs) -> None:
        raise RuntimeError(
            f"{self.__class__.__name__} operation has "
            f"{self.num_extra_inputs} extra tensor inputs "
            f"and {self.num_extra_outputs} extra tensor outputs. "
            "It overrides `fuser_backward` instead of `op_backward`."
        )

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:
        extra_input = basic_op_extra_inputs[0][0]

        # Determine compute dtype
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        elif isinstance(input_, torch.Tensor):
            dtype = input_.dtype
        else:
            dtype = extra_input.dtype

        # Make sure inputs are in correct dtype
        input_ = maybe_dequantize(input_, dtype)
        scales = maybe_dequantize(extra_input, dtype)

        srelu_out = tex.srelu(input_, None)
        out = srelu_out * scales.unsqueeze(-1)

        # Save state for backward pass
        ctx = basic_op_ctxs[0]
        if ctx.requires_grad:
            if is_cpu_offload_enabled():
                mark_activation_offload(input_)
            ctx.input_requires_grad = True
            ctx.extra_input_requires_grad = extra_input.requires_grad
            ctx.dtype = dtype
            ctx.save_for_backward(
                input_,
                scales if ctx.input_requires_grad else None,
            )

        return out, [()]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        Iterable[Iterable[Optional[torch.Tensor]]],
        Iterable[Iterable[Optional[torch.Tensor]]],
    ]:
        ctx = basic_op_ctxs[0]
        input_, scales = ctx.saved_tensors
        input_ = maybe_dequantize(input_, ctx.dtype)
        if scales is not None:
            scales = maybe_dequantize(scales, ctx.dtype)
        grad_output = maybe_dequantize(grad_output, ctx.dtype)

        # Compute input grad
        grad_input = None
        if ctx.input_requires_grad:
            grad_srelu_out = grad_output * scales.unsqueeze(-1)
            grad_input = tex.dsrelu(grad_srelu_out, input_, None)

        # Compute scales grad by recomputing SReLU
        grad_extra_input = None
        if ctx.extra_input_requires_grad:
            srelu_out = tex.srelu(input_, None)
            grad_extra_input = torch.linalg.vecdot(srelu_out, grad_output)

        # Clear input tensor if possible
        clear_tensor_data(ctx.saved_tensors[0])  # input_

        return grad_input, [()], [(grad_extra_input,)]
