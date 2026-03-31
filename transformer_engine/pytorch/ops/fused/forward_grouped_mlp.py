# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for MoE grouped MLP."""

from __future__ import annotations
from collections.abc import Callable, Iterable
import functools
from typing import Any, Optional

import torch
from cuda.bindings import driver as cuda

import transformer_engine_torch as tex
from ...cpp_extensions import general_grouped_gemm_for_grouped_tensor
from ...module._common import noop_cat
from ...quantization import Recipe
from ...tensor import NVFP4Quantizer, Quantizer
from ...utils import get_device_compute_capability
from ...tensor.grouped_tensor import GroupedTensor
from ...constants import MXFP8_BLOCK_SCALING_SIZE, NVFP4_BLOCK_SCALING_SIZE
from ..basic import GroupedLinear, ScaledSwiGLU
from ..fuser import register_forward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext




from .._common import (
    is_quantized_tensor,
    make_grouped_tensor_from_buffers,
    maybe_dequantize,
)

global_alpha_tensor = None


class ForwardGroupedMLP_CuTeGEMMSwiGLU_BlockScaled(FusedOperation):
    """Fused op for MXFP8/NVFP4 GroupedLinear + ScaledSwiGLU + GroupedLinear

    Uses experimental CuTe DSL kernel from cuDNN front-end.

    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_swiglu_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, SwiGLU, and post-multiplication."""
        from cudnn import grouped_gemm_swiglu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_swiglu_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether this fused operation is supported on the current system."""
        if get_device_compute_capability() < (10, 0):
            # Kernel requires SM100+
            return False
        try:
            # Make sure kernel is available
            cls.grouped_gemm_swiglu_kernel()
        except ImportError:
            return False
        return True

    def __init__(
        self,
        *,
        fc1: GroupedLinear,
        swiglu: ScaledSwiGLU,
        fc2: GroupedLinear,
    ) -> None:
        super().__init__((fc1, swiglu, fc2))
        self._mxfp8_alpha_tensor: Optional[torch.Tensor] = None
        self._mxfp8_norm_const_tensor: Optional[torch.Tensor] = None
        # Check for unsupported configurations
        if not self.is_supported():
            self.grouped_gemm_swiglu_kernel()  # Try triggering import error
            raise RuntimeError(f"{self.__class__.__name__} is not supported on this system.")
        if fc1.in_features % 256 != 0 or fc1.out_features % 256 != 0:
            raise ValueError(
                f"Unsupported dims for FC1 (num_groups={fc1.num_groups}, "
                f"in_features={fc1.in_features}, out_features={fc1.out_features})."
            )
        if fc2.in_features % 256 != 0 or fc2.out_features % 256 != 0:
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
        if fc1.has_bias or fc2.has_bias:
            raise ValueError("Fused kernel does not support bias.")
        if swiglu.glu_interleave_size != 32:
            raise ValueError(
                "Fused kernel requires 32-wide GLU interleaving, "
                f"but got glu_interleave_size={swiglu.glu_interleave_size}."
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
        # Get basic operations
        fc1_op, _, fc2_op = self.basic_ops
        fc1_ctx, swiglu_ctx, fc2_ctx = basic_op_ctxs

        import sys as _sys, os as _os
        if int(_os.getenv("SLURM_PROCID", 0)) == 0:
            _sys.__stdout__.write(
                f"[FUSER_FWD] ForwardGroupedMLP_CuTeGEMMSwiGLU_BlockScaled.fuser_forward called "
                f"fc1.num_groups={fc1_op.num_groups} fc1.in={fc1_op.in_features} fc1.out={fc1_op.out_features}\n"
            )
            _sys.__stdout__.flush()

        # Tensor properties
        in_shape = list(input_.size())
        assert len(in_shape) == 2, f"Expected 2D input tensor, got shape={in_shape}."
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)

        num_groups = fc1_op.num_groups
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_parameter else fc1_op.weight0
        fc2_weight_param = fc2_op.weight if fc2_op.single_grouped_parameter else fc2_op.weight0
        device = fc1_weight_param.device
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")
        else:
            dtype = fc1_weight_param.dtype

        # Check which grads are required
        requires_grad = any(ctx.requires_grad for ctx in basic_op_ctxs)
        input_requires_grad = requires_grad
        weight_requires_grad = requires_grad and (
            fc1_weight_param.requires_grad or fc2_weight_param.requires_grad
        )

        # Quantizers
        fc1_input_quantizers = [None] * num_groups
        fc1_weight_quantizer = fc1_op.get_quantizer("forward", 1)
        fc1_grad_output_quantizers = [None] * num_groups
        fc2_input_quantizers = [None] * num_groups
        fc2_weight_quantizer = fc2_op.get_quantizer("forward", 1)
        fc2_grad_output_quantizers = [None] * num_groups
        for idx in range(num_groups):
            fc1_input_quantizers[idx] = fc1_op.get_quantizer("forward", 2 * idx)
            fc1_grad_output_quantizers[idx] = fc1_op.get_quantizer("backward", idx)
            fc2_input_quantizers[idx] = fc2_op.get_quantizer("forward", 2 * idx)
            fc2_grad_output_quantizers[idx] = fc2_op.get_quantizer("backward", idx)
        use_nvfp4 = isinstance(fc1_input_quantizers[0], NVFP4Quantizer) or \
                    type(fc1_weight_param).__name__ == 'NVFP4Tensor'

        # Extract split sizes from extra input
        fc1_split_sizes = basic_op_extra_inputs[0][0]
        fc2_split_sizes = basic_op_extra_inputs[2][0]
        if (
            fc1_split_sizes.size() != fc2_split_sizes.size()
            or fc1_split_sizes.data_ptr() != fc2_split_sizes.data_ptr()
        ):
            raise RuntimeError(
                f"{self.__class__.__name__} got different split points for FC1 and FC2."
            )
        split_sizes = fc1_split_sizes
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")
        split_sizes = split_sizes.to(dtype=torch.int64, device=device)
        split_points = torch.cumsum(split_sizes, 0, dtype=torch.int)
        fc1_x_tensor_offsets = GroupedTensor.make_tensor_offsets(split_sizes, fc1_weight_shape[1])
        fc2_x_tensor_offsets = GroupedTensor.make_tensor_offsets(split_sizes, fc2_weight_shape[1])

        # Extract post-scales from extra input
        scales = basic_op_extra_inputs[1][0]

        # Prepare FC1 grouped weight tensor for fused kernels.
        # Support both:
        #  - single_grouped_parameter=True: op.weight is already a GroupedTensor
        #  - single_grouped_parameter=False: pack per-group weights into a GroupedTensor
        import sys as _wdbg_sys, os as _wdbg_os
        _wdbg_rank0 = int(_wdbg_os.getenv("SLURM_PROCID", 0)) == 0
        if fc1_op.single_grouped_parameter:
            if not isinstance(fc1_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC1 expected GroupedTensor weight with single_grouped_parameter=True."
                )
            if fc1_op.weight.quantizer is not None:
                if _wdbg_rank0:
                    _wdbg_sys.stderr.write(
                        f"[WEIGHT_QUANT_FWD] FC1 single_grouped_parameter=True: "
                        f"weight.quantizer IS NOT None ({type(fc1_op.weight.quantizer).__name__}) "
                        f"=> weight HAS PENDING QUANTIZER, updating and using GroupedTensor as-is "
                        f"(lazy quantization, NOT re-quantizing raw data now); "
                        f"weight.rowwise_data={'None' if fc1_op.weight.rowwise_data is None else fc1_op.weight.rowwise_data.shape}\n"
                    ); _wdbg_sys.stderr.flush()
                fc1_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                fc1_op.weight.quantizer = fc1_weight_quantizer
                grouped_fc1_weight = fc1_op.weight
            else:
                if fc1_op.weight.rowwise_data is None:
                    raise RuntimeError("FC1 grouped weight has no rowwise_data to quantize.")
                if _wdbg_rank0:
                    _wdbg_sys.stderr.write(
                        f"[WEIGHT_QUANT_FWD] FC1 single_grouped_parameter=True: "
                        f"weight.quantizer IS None => QUANTIZING FRESH via tex.group_quantize "
                        f"from rowwise_data shape={fc1_op.weight.rowwise_data.shape} "
                        f"dtype={fc1_op.weight.rowwise_data.dtype}\n"
                    ); _wdbg_sys.stderr.flush()
                fc1_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                grouped_fc1_weight = tex.group_quantize(
                    fc1_op.weight.rowwise_data.view(fc1_op.weight.logical_shape),
                    fc1_weight_quantizer,
                    num_groups,
                    None,
                )
        else:
            fc1_weights = [getattr(fc1_op, f"weight{idx}") for idx in range(num_groups)]
            quantized_fc1_weights = []
            for idx, weight in enumerate(fc1_weights):
                quantizer = fc1_op.get_quantizer("forward", 2 * idx + 1)
                if not is_quantized_tensor(weight):
                    if _wdbg_rank0:
                        _wdbg_sys.stderr.write(
                            f"[WEIGHT_QUANT_FWD] FC1 group={idx}: weight NOT quantized "
                            f"(type={type(weight).__name__} shape={weight.shape} dtype={weight.dtype}) "
                            f"=> QUANTIZING FRESH via quantizer({type(quantizer).__name__})\n"
                        ); _wdbg_sys.stderr.flush()
                    quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                    quantized_fc1_weights.append(quantizer(weight))
                else:
                    # Weight is already quantized. Use as-is.
                    if _wdbg_rank0:
                        _wdbg_sys.stderr.write(
                            f"[WEIGHT_QUANT_FWD] FC1 group={idx}: weight ALREADY QUANTIZED "
                            f"(type={type(weight).__name__}) => APPENDING AS-IS (no re-quantization)\n"
                        ); _wdbg_sys.stderr.flush()
                    quantized_fc1_weights.append(weight)
            grouped_fc1_weight = quantized_fc1_weights
            if use_nvfp4:
                # NVFP4 discrete A_list grouped GEMM requires amax pointers to be contiguous.
                row_amaxes = [getattr(weight, "_amax_rowwise", None) for weight in grouped_fc1_weight]
                if all(amax is not None for amax in row_amaxes):
                    packed_row_amax = torch.cat([amax.view(-1) for amax in row_amaxes], dim=0).contiguous()
                    for idx, weight in enumerate(grouped_fc1_weight):
                        weight._amax_rowwise = packed_row_amax[idx : idx + 1]
                col_amaxes = [getattr(weight, "_amax_columnwise", None) for weight in grouped_fc1_weight]
                if all(amax is not None for amax in col_amaxes):
                    packed_col_amax = torch.cat([amax.view(-1) for amax in col_amaxes], dim=0).contiguous()
                    for idx, weight in enumerate(grouped_fc1_weight):
                        weight._amax_columnwise = packed_col_amax[idx : idx + 1]

        # Prepare FC2 grouped weight tensor for fused kernels.
        if fc2_op.single_grouped_parameter:
            if not isinstance(fc2_op.weight, GroupedTensor):
                raise RuntimeError(
                    "FC2 expected GroupedTensor weight with single_grouped_parameter=True."
                )
            if fc2_op.weight.quantizer is not None:
                if _wdbg_rank0:
                    _wdbg_sys.stderr.write(
                        f"[WEIGHT_QUANT_FWD] FC2 single_grouped_parameter=True: "
                        f"weight.quantizer IS NOT None ({type(fc2_op.weight.quantizer).__name__}) "
                        f"=> weight HAS PENDING QUANTIZER, updating and using GroupedTensor as-is "
                        f"(lazy quantization, NOT re-quantizing raw data now); "
                        f"weight.rowwise_data={'None' if fc2_op.weight.rowwise_data is None else fc2_op.weight.rowwise_data.shape}\n"
                    ); _wdbg_sys.stderr.flush()
                fc2_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                fc2_op.weight.quantizer = fc2_weight_quantizer
                grouped_fc2_weight = fc2_op.weight
            else:
                if fc2_op.weight.rowwise_data is None:
                    raise RuntimeError("FC2 grouped weight has no rowwise_data to quantize.")
                if _wdbg_rank0:
                    _wdbg_sys.stderr.write(
                        f"[WEIGHT_QUANT_FWD] FC2 single_grouped_parameter=True: "
                        f"weight.quantizer IS None => QUANTIZING FRESH via tex.group_quantize "
                        f"from rowwise_data shape={fc2_op.weight.rowwise_data.shape} "
                        f"dtype={fc2_op.weight.rowwise_data.dtype}\n"
                    ); _wdbg_sys.stderr.flush()
                fc2_weight_quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                grouped_fc2_weight = tex.group_quantize(
                    fc2_op.weight.rowwise_data.view(fc2_op.weight.logical_shape),
                    fc2_weight_quantizer,
                    num_groups,
                    None,
                )
        else:
            fc2_weights = [getattr(fc2_op, f"weight{idx}") for idx in range(num_groups)]
            quantized_fc2_weights = []
            for idx, weight in enumerate(fc2_weights):
                quantizer = fc2_op.get_quantizer("forward", 2 * idx + 1)
                quantizer.set_usage(rowwise=True, columnwise=input_requires_grad)
                if not is_quantized_tensor(weight):
                    if _wdbg_rank0:
                        _wdbg_sys.stderr.write(
                            f"[WEIGHT_QUANT_FWD] FC2 group={idx}: weight NOT quantized "
                            f"(type={type(weight).__name__} shape={weight.shape} dtype={weight.dtype}) "
                            f"=> QUANTIZING FRESH via quantizer({type(quantizer).__name__})\n"
                        ); _wdbg_sys.stderr.flush()
                    quantized_fc2_weights.append(quantizer(weight))
                else:
                    # Weight is already quantized. Use as-is.
                    if _wdbg_rank0:
                        _wdbg_sys.stderr.write(
                            f"[WEIGHT_QUANT_FWD] FC2 group={idx}: weight ALREADY QUANTIZED "
                            f"(type={type(weight).__name__}) => APPENDING AS-IS (no re-quantization)\n"
                        ); _wdbg_sys.stderr.flush()
                    quantized_fc2_weights.append(weight)
            grouped_fc2_weight = quantized_fc2_weights
            if use_nvfp4:
                # NVFP4 discrete A_list grouped GEMM requires amax pointers to be contiguous.
                row_amaxes = [getattr(weight, "_amax_rowwise", None) for weight in grouped_fc2_weight]
                if all(amax is not None for amax in row_amaxes):
                    packed_row_amax = torch.cat([amax.view(-1) for amax in row_amaxes], dim=0).contiguous()
                    for idx, weight in enumerate(grouped_fc2_weight):
                        weight._amax_rowwise = packed_row_amax[idx : idx + 1]
                col_amaxes = [getattr(weight, "_amax_columnwise", None) for weight in grouped_fc2_weight]
                if all(amax is not None for amax in col_amaxes):
                    packed_col_amax = torch.cat([amax.view(-1) for amax in col_amaxes], dim=0).contiguous()
                    for idx, weight in enumerate(grouped_fc2_weight):
                        weight._amax_columnwise = packed_col_amax[idx : idx + 1]

        # Some wrapper-copy paths may drop grouped storage metadata; enforce defaults.
        if getattr(grouped_fc1_weight, "with_gemm_swizzled_scales", None) is None and isinstance(
            grouped_fc1_weight, GroupedTensor
        ):
            grouped_fc1_weight.with_gemm_swizzled_scales = False
        if getattr(grouped_fc2_weight, "with_gemm_swizzled_scales", None) is None and isinstance(
            grouped_fc2_weight, GroupedTensor
        ):
            # Grouped MLP fused path expects FC2 grouped GEMM operands to use
            # GEMM-swizzled block scales. Some wrapper-copy paths can drop this
            # metadata bit even when buffers are already in swizzled layout.
            grouped_fc2_weight.with_gemm_swizzled_scales = True

        # Group-quantize input tensor and convert dtypes if needed
        fc1_x = maybe_dequantize(input_, dtype)
        import sys as _fsys_x, os as _os_x
        if int(_os_x.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
            _fsys_x.stderr.write(
                f"[EXP fc1_x input] shape={fc1_x.shape} min={fc1_x.float().min().item():.4g} max={fc1_x.float().max().item():.4g}\n"
                f"  row0[:8]={fc1_x[0,:8].float().tolist()}\n"
            ); _fsys_x.stderr.flush()
        for quantizer in fc1_input_quantizers:
            quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
            quantizer.optimize_for_gemm = True
            if use_nvfp4:
                quantizer.with_rht = True
                quantizer.with_post_rht_amax = True
        grouped_fc1_x = tex.group_quantize(fc1_x, fc1_input_quantizers[0], num_groups, split_sizes)
        if not torch.cuda.is_current_stream_capturing():
            import sys as _fsys0; _fsys0.stderr.write(
                f"[FWD fc1_x] rowwise_data shape={grouped_fc1_x.rowwise_data.shape} "
                f"scale_inv shape={grouped_fc1_x.scale_inv.shape} "
                f"with_gemm_swizzled_scales={getattr(grouped_fc1_x, 'with_gemm_swizzled_scales', 'N/A')}\n"
            ); _fsys0.stderr.flush()
        data_dtype = torch.float4_e2m1fn_x2 if use_nvfp4 else torch.float8_e4m3fn
        scale_view_dtype = torch.float8_e4m3fn if use_nvfp4 else torch.float8_e8m0fnu
        sf_vec_size = NVFP4_BLOCK_SCALING_SIZE if use_nvfp4 else MXFP8_BLOCK_SCALING_SIZE
        data_k = in_shape[1] // 2 if use_nvfp4 else in_shape[1]
        weight_k = fc1_weight_shape[1] // 2 if use_nvfp4 else fc1_weight_shape[1]
        k_sf_divisor = 2 * sf_vec_size if use_nvfp4 else 4 * sf_vec_size

        # Pack data tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (1, sum(m), k)
        # Scale actual shape: (1, sum(m)/128, k/128, 32 (block row),
        #  4 (block row), 4 (block col))
        # Data logical shape: (sum(m), k, 1)
        # Scale logical shape: (32 (block row), 4 (block row),
        #   sum(m)/128, 4 (block col), k/128, 1)
        # For NVFP4, rowwise_data is byte-packed (K/2 storage). Reinterpret first,
        # then reshape to logical (M, K) dimensions.
        fc1_x_data = grouped_fc1_x.rowwise_data.view(dtype=data_dtype)
        fc1_x_data = fc1_x_data.view(in_shape[0], data_k)
        fc1_x_data = fc1_x_data.unsqueeze(0).permute(1, 2, 0)
        fc1_x_scales = grouped_fc1_x.scale_inv
        import sys as _fsys3, os as _os3
        if use_nvfp4 and int(_os3.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
            _raw_si = grouped_fc1_x.scale_inv
            _fsys3.stderr.write(
                f"[FWD fc1_x scale_inv RAW] dtype={_raw_si.dtype} shape={_raw_si.shape} "
                f"first4_bytes={_raw_si.view(-1)[:4].tolist()}\n"
            ); _fsys3.stderr.flush()
            # Also print fc1_x fp4 data first byte and what it dequantizes to
            _raw_data = grouped_fc1_x.rowwise_data.view(-1)[:4].tolist()
            _fsys3.stderr.write(f"[FWD fc1_x raw_data first4_bytes]={_raw_data}\n"); _fsys3.stderr.flush()
            # Print _amax_rowwise (global amax used for fp8-normalization)
            _amax_x = getattr(grouped_fc1_x, 'amax', None)
            if _amax_x is None:
                _amax_x = getattr(grouped_fc1_x, '_amax_rowwise', None)
            if _amax_x is not None:
                _amax_x0 = _amax_x.view(-1)[0].float().item()
                _scale0 = _raw_si.view(-1)[:1].view(torch.float8_e4m3fn).float().item()
                _fsys3.stderr.write(
                    f"[FWD fc1_x amax] shape={_amax_x.shape} val={_amax_x.float().view(-1).tolist()}\n"
                    f"  => true_dequant[block0] = scale_fp8[0] * amax / (fp8_max*fp4_max) = "
                    f"{_scale0:.4f} * "
                    f"{_amax_x0:.4f} / (448*6) = "
                    f"{_scale0 * _amax_x0 / (448*6):.6f}\n"
                    f"  vs 1/scale_fp8[0] = {1.0/_scale0:.6f}\n"
                ); _fsys3.stderr.flush()
            else:
                _fsys3.stderr.write("[FWD fc1_x _amax_rowwise] NOT FOUND\n"); _fsys3.stderr.flush()
        fc1_x_scales = fc1_x_scales.view(dtype=scale_view_dtype)
        if use_nvfp4:
            # Capture activation amax for alpha computation.
            # Pass scale_inv (S_dec) directly to the kernel without pre-conversion,
            # matching the baseline cuBLAS path (alpha = amax_A * amax_B / (fp4_max^2 * fp8_max^2)).
            _amax_x = getattr(grouped_fc1_x, 'amax', None)
            if _amax_x is None:
                _amax_x = getattr(grouped_fc1_x, '_amax_rowwise')
            _amax_x = _amax_x.float().view(-1)
            if _amax_x.numel() != num_groups:
                # Single global amax (or unexpected size): broadcast global max to all groups.
                _amax_x = _amax_x.amax().view(1).expand(num_groups).contiguous()
        if not torch.cuda.is_current_stream_capturing():
            import sys as _fsys3b; _fsys3b.stderr.write(
                f"[FWD fc1_x_scales after view dtype] shape={fc1_x_scales.shape} "
                f"min={fc1_x_scales.float().min().item():.6f} max={fc1_x_scales.float().max().item():.6f}\n"
            ); _fsys3b.stderr.flush()
        if use_nvfp4 and getattr(grouped_fc1_x, 'with_gemm_swizzled_scales', False):
            # RHT kernel with optimize_for_gemm=True writes scales directly in
            # SwizzledSFALayout (cuDNN-compatible format). Only kernel-format permute needed.
            fc1_x_scales = fc1_x_scales.view(
                1, in_shape[0] // 128, data_k // k_sf_divisor, 32, 4, 4,
            )
            fc1_x_scales = fc1_x_scales.permute(3, 4, 1, 5, 2, 0)
        else:
            # Unswizzled TE format: convert unswizzled → swizzled, then to kernel format.
            fc1_x_scales = fc1_x_scales.view(
                1, in_shape[0] // 128, 4, 32, data_k // k_sf_divisor, 4,
            )
            fc1_x_scales = fc1_x_scales.permute(0, 1, 4, 3, 2, 5).contiguous()
            fc1_x_scales = fc1_x_scales.permute(3, 4, 1, 5, 2, 0)

        # Pack weight tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (num_groups, n, k)
        # Scale actual shape: (num_groups, n/128, k/128, 32 (block row),
        #  4 (block row), 4 (block col))
        # Data logical shape: (n, k, num_groups)
        # Scale logical shape: (32 (block row), 4 (block row), n/128,
        #   4 (block col), k/128, num_groups)
        fc1_w_data = (
            grouped_fc1_weight.rowwise_data
            if fc1_op.single_grouped_parameter
            else noop_cat([w._rowwise_data for w in grouped_fc1_weight])
        )
        fc1_w_data = fc1_w_data.view(dtype=data_dtype)
        fc1_w_data = fc1_w_data.view(num_groups, fc1_weight_shape[0], weight_k)
        fc1_w_data = fc1_w_data.permute(1, 2, 0)
        _fc1_w_scales_raw = (
            grouped_fc1_weight.scale_inv
            if fc1_op.single_grouped_parameter
            else noop_cat([w._rowwise_scale_inv for w in grouped_fc1_weight])
        )
        if not torch.cuda.is_current_stream_capturing():
            import sys as _fsys; _fsys.stderr.write(
                f"[FWD fc1_w_scales_raw] dtype={_fc1_w_scales_raw.dtype} shape={_fc1_w_scales_raw.shape} "
                f"numel={_fc1_w_scales_raw.numel()} expected_numel={num_groups * fc1_weight_shape[0]//128 * 4 * 32 * (weight_k//k_sf_divisor) * 4}\n"
                f"  scale_view_dtype={scale_view_dtype} weight_k={weight_k} k_sf_divisor={k_sf_divisor} fc1_weight_shape={fc1_weight_shape}\n"
            ); _fsys.stderr.flush()
        fc1_w_scales = _fc1_w_scales_raw
        # Print weight _amax_rowwise for the first weight (global amax used for fp8-normalization)
        import sys as _fsys_w, os as _os_w
        if use_nvfp4 and int(_os_w.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
            _w0 = grouped_fc1_weight if fc1_op.single_grouped_parameter else grouped_fc1_weight[0]
            _amax_w = getattr(_w0, '_amax_rowwise', None)
            _w_s0 = _fc1_w_scales_raw.view(-1)[:1].view(torch.float8_e4m3fn).float().item()
            if _amax_w is not None:
                _fsys_w.stderr.write(
                    f"[FWD fc1_w _amax_rowwise] val={_amax_w.float().item():.6f}\n"
                    f"  => true_dequant[block0] = scale_fp8[0] * amax / (fp8_max*fp4_max) = "
                    f"{_w_s0:.4f} * {_amax_w.float().item():.4f} / (448*6) = "
                    f"{_w_s0 * _amax_w.float().item() / (448*6):.6f}\n"
                    f"  vs 1/scale_fp8[0] = {1.0/_w_s0:.6f}\n"
                ); _fsys_w.stderr.flush()
            else:
                _fsys_w.stderr.write(f"[FWD fc1_w _amax_rowwise] NOT FOUND  scale_fp8[0]={_w_s0:.4f}\n"); _fsys_w.stderr.flush()
        fc1_w_scales = fc1_w_scales.view(dtype=scale_view_dtype)
        if use_nvfp4:
            # Capture weight amax for alpha computation.
            # Pass scale_inv (S_dec) directly to the kernel without pre-conversion.
            if fc1_op.single_grouped_parameter:
                _amax_w = (grouped_fc1_weight.amax if grouped_fc1_weight.amax is not None
                           else getattr(grouped_fc1_weight, '_amax_rowwise')).float().view(-1)
                if _amax_w.numel() != num_groups:
                    # Single global amax (or unexpected size): broadcast global max to all groups.
                    _amax_w = _amax_w.amax().view(1).expand(num_groups).contiguous()
                else:
                    _amax_w = _amax_w.contiguous()
            else:
                # Per-group weights: stack individual amaxes into (num_groups,) tensor
                _amax_w = torch.cat([_w._amax_rowwise.float().view(-1) for _w in grouped_fc1_weight]).contiguous()
        if not torch.cuda.is_current_stream_capturing():
            import sys as _fsys; _fsys.stderr.write(
                f"[FWD fc1_w_scales after view dtype] shape={fc1_w_scales.shape} "
                f"min={fc1_w_scales.float().min().item():.6f} max={fc1_w_scales.float().max().item():.6f}\n"
            ); _fsys.stderr.flush()
        fc1_w_scales = fc1_w_scales.view(
            num_groups,
            fc1_weight_shape[0] // 128,
            4,
            32,
            weight_k // k_sf_divisor,
            4,
        )  # Unswizzled layout
        fc1_w_scales = fc1_w_scales.permute(
            0, 1, 4, 3, 2, 5
        ).contiguous()  # Convert to swizzled layout
        fc1_w_scales = fc1_w_scales.permute(3, 4, 1, 5, 2, 0)

        # Kernel scaling factors
        if use_nvfp4:
            # Baseline cuBLAS approach: alpha = amax_A * amax_B / (fp4_max^2 * fp8_max^2)
            # Mirrors nvte_nvfp4_compute_per_tensor_scale: alpha *= amax_A * amax_B / (6^2 * 448^2)
            _nvfp4_fp4_max = 6.0
            _nvfp4_fp8_max = 448.0
            alpha_tensor = (_amax_x * _amax_w / (_nvfp4_fp4_max ** 2 * _nvfp4_fp8_max ** 2)).contiguous().to(torch.float32)
            norm_const_tensor_arg = None
            if int(_os.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
                import sys as _fsys_alpha
                _fsys_alpha.stderr.write(
                    f"[FWD alpha_tensor per-group] amax_x={_amax_x.tolist()} amax_w={_amax_w.tolist()} alpha={alpha_tensor.tolist()}\n"
                ); _fsys_alpha.stderr.flush()
        else:
            alpha_tensor, norm_const_tensor = self._get_kernel_constants(
                num_groups=num_groups, dtype=dtype, device=device
            )
            norm_const_tensor_arg = norm_const_tensor
        current_stream = cuda.CUstream(  # pylint: disable=c-extension-no-member
            torch.cuda.current_stream().cuda_stream
        )

        _use_unfused_fwd = _os.getenv("TE_NVFP4_UNFUSED_FWD") == "1"
        _unfused_fwd_rank0 = int(_os.getenv("SLURM_PROCID", 0)) == 0

        if _use_unfused_fwd:
            # Unfused path: FC1 GEMM + manual SwiGLU + FC2 input quantize. Bypasses fused kernel.
            # Step 1: FC1 GEMM
            _fc1_out = torch.zeros(in_shape[0], fc1_weight_shape[0], dtype=dtype, device=device)
            _grouped_fc1_out = make_grouped_tensor_from_buffers(
                num_groups=num_groups,
                data=_fc1_out,
                split_sizes=split_sizes,
                dtype=dtype,
                logical_last_dim=fc1_weight_shape[0],
            )
            general_grouped_gemm_for_grouped_tensor(
                grouped_fc1_weight,
                grouped_fc1_x,
                _grouped_fc1_out,
                layout="TN",
                accumulate=False,
            )
            if _unfused_fwd_rank0 and not torch.cuda.is_current_stream_capturing():
                _sys.__stdout__.write(
                    f"[FWD_UNFUSED_fc1_out]  nan={_fc1_out.isnan().any().item()}"
                    f" min={_fc1_out.min().item():.4g} max={_fc1_out.abs().max().item():.4g}"
                    f" norm={_fc1_out.norm().item():.4g}\n"
                )
                _sys.__stdout__.flush()
            # swiglu_in = FC1 output (bfloat16), saved for backward dSwiGLU
            swiglu_in = _fc1_out
            # Step 2: Manual SwiGLU — out = x * silu(gate)
            _half_fwd = fc1_weight_shape[0] // 2
            _x_half = _fc1_out[:, :_half_fwd].float()
            _gate = _fc1_out[:, _half_fwd:].float()
            _swiglu_out = _x_half * _gate * torch.sigmoid(_gate)
            # Apply per-group routing scales
            _offset_fwd = 0
            for _gi_fwd, _sz_fwd in enumerate(split_sizes.tolist()):
                _swiglu_out[_offset_fwd:_offset_fwd + _sz_fwd] *= scales[_gi_fwd].float()
                _offset_fwd += _sz_fwd
            _swiglu_out = _swiglu_out.to(dtype)
            if _unfused_fwd_rank0 and not torch.cuda.is_current_stream_capturing():
                _sys.__stdout__.write(
                    f"[FWD_UNFUSED_swiglu_out]  nan={_swiglu_out.isnan().any().item()}"
                    f" min={_swiglu_out.min().item():.4g} max={_swiglu_out.abs().max().item():.4g}"
                    f" norm={_swiglu_out.norm().item():.4g}\n"
                )
                _sys.__stdout__.flush()
            # Package outputs in kernel-compatible dict so existing extraction code runs unchanged.
            # c_tensor: (M, H, 1) → after permute(2,0,1)+view gives (M, H) = swiglu_in
            # d_tensor: (M, H/2, 1) → after permute(2,0,1)+view gives (M, H/2) = fc2_in
            fc1_kernel_out = {
                "c_tensor": _fc1_out.view(in_shape[0], fc1_weight_shape[0], 1),
                "d_tensor": _swiglu_out.view(in_shape[0], fc2_weight_shape[1], 1),
            }

        else:
            # Fused kernel for FC1 + SwiGLU + post-scale
            fc1_kernel_out = self.grouped_gemm_swiglu_kernel()(
            fc1_x_data,
            fc1_w_data,
            fc1_x_scales,
            fc1_w_scales,
            split_points,
            alpha_tensor,  # alpha_tensor
            norm_const_tensor=norm_const_tensor_arg,
            prob_tensor=scales.detach().reshape(-1, 1, 1),
            acc_dtype=torch.float32,
            c_dtype=torch.bfloat16,
            d_dtype=torch.bfloat16 if use_nvfp4 else torch.float8_e4m3fn,
            cd_major="n",
            sf_vec_size=sf_vec_size,
            current_stream=current_stream,
            discrete_col_sfd=not use_nvfp4,
        )

        # Unpack kernel outputs
        # Note: Fused kernel outputs tensors with non-contiguous
        # logical dims.
        # Row-wise data logical shape: (sum(m_splits), k, 1)
        # Row-wise scale logical shape: (32 (block row), 4 (block row),
        #   sum(m_splits)/128, 4 (block col), k/128, 1)
        # Column-wise data logical shape: (sum(m_splits), k, 1)
        # Column-wise scale logical shape: (32 (block col), 4 (block col),
        #   k/128, 4 (block row), sum(m_splits)/128, 1)
        swiglu_in = fc1_kernel_out["c_tensor"]
        if not torch.cuda.is_current_stream_capturing():
            import sys as _fsys2; _fsys2.stderr.write(
                f"[FWD c_tensor (post-fc1, pre-swiglu)] shape={swiglu_in.shape} dtype={swiglu_in.dtype} "
                f"min={swiglu_in.float().min().item():.4e} max={swiglu_in.float().max().item():.4e}\n"
            ); _fsys2.stderr.flush()
        swiglu_in = swiglu_in.permute(2, 0, 1)
        swiglu_in = swiglu_in.view(in_shape[0], fc1_weight_shape[0])
        import sys as _fsys2b, os as _os2b
        if int(_os2b.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
            _fsys2b.stderr.write(
                f"[EXP c_tensor row0[:8]]={swiglu_in[0,:8].float().tolist()}\n"
            ); _fsys2b.stderr.flush()
            _offset_c = 0
            for _gi_c, _sz_c in enumerate(split_sizes.tolist()):
                if _sz_c > 0:
                    _grp_c = swiglu_in[_offset_c:_offset_c + _sz_c].float()
                    _fsys2b.stderr.write(
                        f"[FWD c_tensor group={_gi_c} sz={_sz_c}]"
                        f" row0[:8]={_grp_c[0, :8].tolist()}"
                        f" norm={_grp_c.norm().item():.4g} min={_grp_c.min().item():.4g} max={_grp_c.max().item():.4g}\n"
                    )
                else:
                    _fsys2b.stderr.write(f"[FWD c_tensor group={_gi_c} sz=0 (empty)]\n")
                _offset_c += _sz_c
            _fsys2b.stderr.flush()
        if use_nvfp4 and int(_os2b.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
            # ---- MANUAL VERIFICATION BLOCK ----
            # Decode A row 0 fp4 values and W row 0 fp4 values.
            # Show first 2 scale blocks (32 elements) and their partial dot product.
            # Compare with kernel's c_tensor[0, 0].
            import sys as _fv, os as _ov
            # fp4_e2m1fn nibble decode lookup (signed 4-bit e2m1fn)
            _fp4lut = torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
                dtype=torch.float32, device='cpu'
            )
            def _decode_fp4x2_row(raw_uint8_flat, row_idx, K_packed):
                """Decode one row of fp4x2-packed data. Returns float32 tensor of shape (2*K_packed,)."""
                row = raw_uint8_flat[row_idx * K_packed : (row_idx + 1) * K_packed].cpu()
                lo = (row & 0xF).long()   # even-indexed fp4 values
                hi = ((row >> 4) & 0xF).long()  # odd-indexed fp4 values
                decoded = torch.empty(2 * K_packed, dtype=torch.float32)
                decoded[0::2] = _fp4lut[lo]
                decoded[1::2] = _fp4lut[hi]
                return decoded

            # A row 0 (input activation after RHT+quant)
            _a_raw = grouped_fc1_x.rowwise_data.view(-1)  # uint8
            _a_row0_fp4 = _decode_fp4x2_row(_a_raw, 0, data_k)  # shape (K=8192,)

            # W row 0 (first output neuron weights; W stored as (N, K_packed) fp4x2)
            _w_raw = (
                grouped_fc1_weight.rowwise_data
                if fc1_op.single_grouped_parameter
                else noop_cat([w._rowwise_data for w in grouped_fc1_weight])
            ).view(-1)
            _w_row0_fp4 = _decode_fp4x2_row(_w_raw, 0, weight_k)  # shape (K=8192,)

            # Scale for A row 0: scale_inv stored as (M/128, K/16) in TE unswizzled or swizzled.
            # For simplicity, read raw scale_inv bytes and find scales for row 0 blocks 0 and 1.
            # scale_inv raw = uint8, 4194304 elements = (M * K / 16) fp8_e4m3fn values.
            # In TE NVF4, scales are organized as (M/128, K/sf_divisor, 32, 4, 4)
            # or swizzled. Read first few to see scale for A[0, block 0..1].
            # Use the ALREADY-INVERTED scales from fc1_x_scales (after inversion above).
            # fc1_x_scales after inversion = dequant scales, shape depends on fast path.
            # --- Just read the raw scale_inv and compute dequant scale directly ---
            _a_scale_raw = grouped_fc1_x.scale_inv.view(-1).cpu()  # uint8 flat on CPU
            # First few bytes: scales for some row/block (offset depends on TE layout).
            # Read first 8 bytes as fp8_e4m3fn to get representative scales.
            _a_s_fp8 = _a_scale_raw[:8].view(torch.float8_e4m3fn).float()
            _a_s_dequant = 1.0 / _a_s_fp8  # dequant = 1/quant_scale

            _w_scale_raw = _fc1_w_scales_raw.view(-1).cpu()  # uint8 flat on CPU
            _w_s_fp8 = _w_scale_raw[:8].view(torch.float8_e4m3fn).float()
            _w_s_dequant = 1.0 / _w_s_fp8

            # Partial dot product: first 16 elements (block 0) using scale[0]
            _K_partial = sf_vec_size  # 16 elements per block
            _partial_a = _a_row0_fp4[:_K_partial] * _a_s_dequant[0].item()  # dequant A block 0
            _partial_w = _w_row0_fp4[:_K_partial] * _w_s_dequant[0].item()  # dequant W block 0
            _partial_dot = (_partial_a * _partial_w).sum().item()

            # Full approximate dot product over all K (using first scale per group-of-16)
            # Block each 16 elements with the corresponding scale (rough, scale offset unknown)
            _a_row0_dequant_approx = _a_row0_fp4.clone()
            _w_row0_dequant_approx = _w_row0_fp4.clone()

            _fv.stderr.write(
                f"[VERIFY] Original x[0,:8]={fc1_x[0,:8].float().tolist()}\n"
                f"[VERIFY] A_fp4 row0[0:32]={_a_row0_fp4[:32].tolist()}\n"
                f"[VERIFY] A scale_inv[0:8] as fp8={_a_s_fp8.tolist()} -> dequant={_a_s_dequant.tolist()}\n"
                f"[VERIFY] W_fp4 row0[0:32]={_w_row0_fp4[:32].tolist()}\n"
                f"[VERIFY] W scale_inv[0:8] as fp8={_w_s_fp8.tolist()} -> dequant={_w_s_dequant.tolist()}\n"
                f"[VERIFY] Partial dot (block 0, k=0..15): sum(A*sA * W*sW) = {_partial_dot:.6f}\n"
                f"[VERIFY] Kernel c_tensor[0,0] = {swiglu_in[0,0].float().item():.6f}\n"
            ); _fv.stderr.flush()
            # ---- END MANUAL VERIFICATION BLOCK ----
        if use_nvfp4:
            fc2_in = fc1_kernel_out["d_tensor"]
            if not torch.cuda.is_current_stream_capturing():
                import sys as _fsys2; _fsys2.stderr.write(
                    f"[FWD d_tensor (post-swiglu, fc2 input)] shape={fc2_in.shape} dtype={fc2_in.dtype} "
                    f"min={fc2_in.float().min().item():.4e} max={fc2_in.float().max().item():.4e}\n"
                ); _fsys2.stderr.flush()
            fc2_in = fc2_in.permute(2, 0, 1)
            fc2_in = fc2_in.view(in_shape[0], fc2_weight_shape[1]).contiguous()
            if int(_os2b.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
                import sys as _fsys_d
                _offset_d = 0
                for _gi_d, _sz_d in enumerate(split_sizes.tolist()):
                    if _sz_d > 0:
                        _grp_d = fc2_in[_offset_d:_offset_d + _sz_d].float()
                        _fsys_d.stderr.write(
                            f"[FWD d_tensor group={_gi_d} sz={_sz_d}]"
                            f" row0[:8]={_grp_d[0, :8].tolist()}"
                            f" norm={_grp_d.norm().item():.4g} max={_grp_d.abs().max().item():.4g}\n"
                        )
                    else:
                        _fsys_d.stderr.write(f"[FWD d_tensor group={_gi_d} sz=0 (empty)]\n")
                    _offset_d += _sz_d
                _fsys_d.stderr.flush()
            for quantizer in fc2_input_quantizers:
                quantizer.set_usage(rowwise=True, columnwise=weight_requires_grad)
                quantizer.optimize_for_gemm = True
                if use_nvfp4:
                    quantizer.with_rht = True
                    quantizer.with_post_rht_amax = True
            grouped_fc2_x = tex.group_quantize(fc2_in, fc2_input_quantizers[0], num_groups, split_sizes)
        else:
            fc2_in_row_data = fc1_kernel_out["d_tensor"]
            fc2_in_row_data = fc2_in_row_data.permute(2, 0, 1)
            fc2_in_row_data = fc2_in_row_data.view(in_shape[0], fc2_weight_shape[1]).contiguous()
            fc2_in_row_scale = fc1_kernel_out["sfd_row_tensor"]
            fc2_in_row_scale = fc2_in_row_scale.permute(5, 2, 4, 0, 1, 3)

            fc2_in_col_data = fc1_kernel_out["d_col_tensor"]
            fc2_in_col_data = fc2_in_col_data.permute(2, 0, 1)
            fc2_in_col_data = fc2_in_col_data.view(in_shape[0], fc2_weight_shape[1]).contiguous()
            fc2_in_col_scale = fc1_kernel_out["sfd_col_tensor"]
            fc2_in_col_scale = fc2_in_col_scale.permute(5, 2, 4, 0, 1, 3)
            # Repack columnwise scales on GPU to preserve group ordering.

            # FC2 input scales are already swizzled/optimized for GEMM.
            grouped_fc2_x = make_grouped_tensor_from_buffers(
                num_groups=num_groups,
                data=fc2_in_row_data.reshape(-1),
                columnwise_data=fc2_in_col_data.reshape(-1),
                scale_inv=fc2_in_row_scale.reshape(-1),
                columnwise_scale_inv=fc2_in_col_scale.reshape(-1),
                split_sizes=split_sizes,
                logical_last_dim=fc2_weight_shape[1],
                dtype=dtype,
                quantizer=fc2_input_quantizers[0],
                with_gemm_swizzled_scales=True,
                tensor_offsets=fc2_x_tensor_offsets,
            )

        # FC2 GEMM
        fc2_out_shape = in_shape[:-1] + [fc2_weight_shape[0]]
        fc2_out = torch.empty(fc2_out_shape, dtype=dtype, device=device)
        grouped_fc2_out = make_grouped_tensor_from_buffers(
            num_groups=num_groups,
            data=fc2_out,
            split_sizes=split_sizes,
            dtype=fc2_out.dtype,
            logical_last_dim=fc2_weight_shape[0],
        )

        # Fused grouped GEMM requires block scales in GEMM-swizzled layout for
        # both operands. Some construction paths drop this metadata bit.
        if use_nvfp4 and getattr(grouped_fc2_weight, "with_gemm_swizzled_scales", None) not in (True, None):
            grouped_fc2_weight.with_gemm_swizzled_scales = True
        if use_nvfp4 and getattr(grouped_fc2_x, "with_gemm_swizzled_scales", None) not in (True, None):
            grouped_fc2_x.with_gemm_swizzled_scales = True

        general_grouped_gemm_for_grouped_tensor(
            grouped_fc2_weight,
            grouped_fc2_x,
            grouped_fc2_out,
            layout="TN",
            accumulate=False,
        )
        import sys as _fsys_fc2out, os as _os_fc2out
        if int(_os_fc2out.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
            _fc2out_f = fc2_out.detach().float()
            _fsys_fc2out.__stdout__.write(
                f"[FWD_FC2_OUT] shape={list(fc2_out.shape)}"
                f"  nan={_fc2out_f.isnan().any().item()}"
                f"  min={_fc2out_f.min().item():.4g} max={_fc2out_f.max().item():.4g}"
                f"  norm={_fc2out_f.norm().item():.4g}"
                f"  row0[:8]={_fc2out_f[0, :8].tolist()}\n"
            )
            _fsys_fc2out.__stdout__.flush()
            _offset_fc2 = 0
            for _gi_fc2, _sz_fc2 in enumerate(split_sizes.tolist()):
                if _sz_fc2 > 0:
                    _grp_fc2 = _fc2out_f[_offset_fc2:_offset_fc2 + _sz_fc2]
                    _fsys_fc2out.__stdout__.write(
                        f"[FWD_FC2_OUT group={_gi_fc2} sz={_sz_fc2}]"
                        f" row0[:8]={_grp_fc2[0, :8].tolist()}"
                        f" norm={_grp_fc2.norm().item():.4g}"
                        f" min={_grp_fc2.min().item():.4g} max={_grp_fc2.max().item():.4g}\n"
                    )
                else:
                    _fsys_fc2out.__stdout__.write(f"[FWD_FC2_OUT group={_gi_fc2} sz=0 (empty)]\n")
                _offset_fc2 += _sz_fc2
            _fsys_fc2out.__stdout__.flush()

        # Prepare input tensors for backward pass
        if not weight_requires_grad:
            grouped_fc1_x = None
            grouped_fc2_x = None

        # Save state for backward pass
        if requires_grad:
            if grouped_fc1_x is not None:
                grouped_fc1_x.columnwise_data.grouped_name = "fc1_columnwise_data"
                grouped_fc1_x.columnwise_data.logical_shape = grouped_fc1_x.logical_shape
                grouped_fc1_x.columnwise_scale_inv.grouped_name = "fc1_columnwise_scale_inv"
                grouped_fc1_x.columnwise_scale_inv.logical_shape = grouped_fc1_x.logical_shape
                fc1_input_tensors = (
                    None,  # data
                    grouped_fc1_x.columnwise_data,  # columnwise_data
                    None,  # scale_inv
                    grouped_fc1_x.columnwise_scale_inv,  # columnwise_scale_inv
                    fc1_x_tensor_offsets,  # tensor_offsets
                    grouped_fc1_x.amax,  # rowwise global amax (NVFP4 2-level scale dequant)
                    grouped_fc1_x.columnwise_amax,  # columnwise global amax (NVFP4 2-level scale dequant)
                )
            else:
                fc1_input_tensors = (None, None, None, None, None, None, None)
            # FC1
            if fc1_op.single_grouped_parameter:
                fc1_ctx.save_for_backward(
                    split_sizes, split_points, grouped_fc1_weight, *fc1_input_tensors
                )
            else:
                fc1_ctx.save_for_backward(
                    split_sizes, split_points, *grouped_fc1_weight, *fc1_input_tensors
                )
            fc1_ctx.with_quantized_compute = True
            fc1_ctx.input_quantizers = fc1_input_quantizers
            fc1_ctx.weight_quantizer = fc1_weight_quantizer
            fc1_ctx.grad_output_quantizers = fc1_grad_output_quantizers
            fc1_ctx.grad_input_quantizers = None
            fc1_ctx.dtype = dtype
            fc1_ctx.input_requires_grad = input_requires_grad
            fc1_ctx.weight_requires_grad = weight_requires_grad

            # Scaled SwiGLU
            swiglu_in.grouped_name = "swiglu_in"
            scales.grouped_name = "scales"
            swiglu_ctx.save_for_backward(swiglu_in, scales)
            swiglu_ctx.input_requires_grad = True
            swiglu_ctx.extra_input_requires_grad = True
            swiglu_ctx.dtype = dtype

            # FC2 state
            if grouped_fc2_x is not None:
                grouped_fc2_x.columnwise_data.grouped_name = "fc2_columnwise_data"
                grouped_fc2_x.columnwise_data.logical_shape = grouped_fc2_x.logical_shape
                grouped_fc2_x.columnwise_scale_inv.grouped_name = "fc2_columnwise_scale_inv"
                grouped_fc2_x.columnwise_scale_inv.logical_shape = grouped_fc2_x.logical_shape
                fc2_input_tensors = (
                    None,  # data
                    grouped_fc2_x.columnwise_data,  # columnwise_data
                    None,  # scale_inv
                    grouped_fc2_x.columnwise_scale_inv,  # columnwise_scale_inv
                    fc2_x_tensor_offsets,  # tensor_offsets
                    grouped_fc2_x.amax,  # rowwise global amax (NVFP4 2-level scale dequant)
                    grouped_fc2_x.columnwise_amax,  # columnwise global amax (NVFP4 2-level scale dequant)
                )
            else:
                fc2_input_tensors = (None, None, None, None, None, None, None)

            if fc2_op.single_grouped_parameter:
                fc2_ctx.save_for_backward(split_sizes, grouped_fc2_weight, *fc2_input_tensors)
            else:
                fc2_ctx.save_for_backward(split_sizes, *grouped_fc2_weight, *fc2_input_tensors)

            fc2_ctx.with_quantized_compute = True
            fc2_ctx.input_quantizers = fc2_input_quantizers
            fc2_ctx.weight_quantizer = fc2_weight_quantizer
            fc2_ctx.grad_output_quantizers = fc2_grad_output_quantizers
            fc2_ctx.grad_input_quantizers = None
            fc2_ctx.dtype = dtype
            fc2_ctx.input_requires_grad = input_requires_grad
            fc2_ctx.weight_requires_grad = weight_requires_grad

        return fc2_out, [(), (), ()]

    def _get_kernel_constants(
        self,
        *,
        num_groups: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global global_alpha_tensor
        alpha_tensor = self._mxfp8_alpha_tensor
        norm_const_tensor = self._mxfp8_norm_const_tensor
        if (
            alpha_tensor is None
            or alpha_tensor.numel() != num_groups
            or alpha_tensor.dtype != dtype
            or alpha_tensor.device != device
        ):
            if (
                global_alpha_tensor is None
                or global_alpha_tensor.numel() != num_groups
                or global_alpha_tensor.dtype != dtype
                or global_alpha_tensor.device != device
            ):
                global_alpha_tensor = torch.ones(num_groups, dtype=dtype, device=device)
            alpha_tensor = global_alpha_tensor
            norm_const_tensor = alpha_tensor[:1]
            self._mxfp8_alpha_tensor = alpha_tensor
            self._mxfp8_norm_const_tensor = norm_const_tensor
        elif (
            norm_const_tensor is None
            or norm_const_tensor.numel() != 1
            or norm_const_tensor.dtype != dtype
            or norm_const_tensor.device != device
        ):
            norm_const_tensor = alpha_tensor[:1]
            self._mxfp8_norm_const_tensor = norm_const_tensor
        return alpha_tensor, norm_const_tensor


def fuse_forward_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply operation fusion for forward pass.

    Parameters
    ----------
    ops : list of FusibleOperation
        Forward pass operations.
    recipe : Recipe, optional
        Quantization recipe.

    Returns
    -------
    ops : list of FusibleOperation
        Updated forward pass operations

    """

    # Return immediately if fused kernel is not supported
    if not ForwardGroupedMLP_CuTeGEMMSwiGLU_BlockScaled.is_supported():
        return ops

    # Check if recipe is supported
    if recipe is None:
        return ops
    if not (recipe.mxfp8() or recipe.nvfp4()):
        return ops

    # Scan through ops, fusing if possible
    out = []
    window, ops = ops[:3], ops[3:]
    while len(window) == 3:

        # Check if window matches pattern
        matches_pattern = True
        if not (
            isinstance(window[0], GroupedLinear)
            and isinstance(window[1], ScaledSwiGLU)
            and isinstance(window[2], GroupedLinear)
        ):
            matches_pattern = False
        elif window[0].has_bias or window[2].has_bias:
            matches_pattern = False
        elif window[0].num_groups != window[2].num_groups:
            matches_pattern = False
        elif (
            window[0].in_features % 256 != 0
            or window[0].out_features % 256 != 0
            or window[2].in_features % 256 != 0
            or window[2].out_features % 256 != 0
        ):
            matches_pattern = False
        elif window[1].glu_interleave_size != 32:
            matches_pattern = False

        if matches_pattern:
            # Construct fused op if window matches pattern
            op = ForwardGroupedMLP_CuTeGEMMSwiGLU_BlockScaled(
                fc1=window[0],
                swiglu=window[1],
                fc2=window[2],
            )
            window = [op]
        else:
            # Shift window if window doesn't match pattern
            out.extend(window[:-2])
            window = window[-2:]

        # Adjust window to expected size
        out.extend(window[:-3])
        window = window[-3:]
        while ops and len(window) < 3:
            window.append(ops[0])
            ops = ops[1:]

    # Return list of ops
    out.extend(window)
    return out


# Register fusion if available
if ForwardGroupedMLP_CuTeGEMMSwiGLU_BlockScaled.is_supported():
    register_forward_fusion(fuse_forward_ops, prepend=True)
