# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fused operation for MoE grouped MLP."""

from __future__ import annotations
from collections.abc import Callable
import functools
import math
from typing import Optional

import torch
from cuda.bindings import driver as cuda

import transformer_engine_torch as tex
from ...cpp_extensions import (
    general_grouped_gemm_for_grouped_tensor,
)
from ...module._common import noop_cat
from ...module.base import get_dummy_wgrad
from ...quantization import Recipe
from ...tensor import NVFP4Quantizer
from ...tensor.grouped_tensor import GroupedTensor
from ...utils import clear_tensor_data, get_device_compute_capability
from ...constants import MXFP8_BLOCK_SCALING_SIZE, NVFP4_BLOCK_SCALING_SIZE
from ..basic import GroupedLinear, ScaledSwiGLU
from ..fuser import register_backward_fusion
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    make_grouped_tensor_from_buffers,
    maybe_dequantize,
)

global_alpha_tensor = None


class BackwardGroupedMLP_CuTeGEMMDSwiGLU_BlockScaled(FusedOperation):
    """Fused op for MXFP8/NVFP4 GroupedLinear + ScaledSwiGLU + GroupedLinear

    Uses experimental CuTe DSL kernel from cuDNN front-end.

    """

    @classmethod
    @functools.lru_cache(maxsize=None)
    def grouped_gemm_dswiglu_kernel(cls) -> Callable:
        """Fused kernel for grouped GEMM, SwiGLU backward, and scale grad."""
        from cudnn import grouped_gemm_dswiglu_wrapper_sm100  # pylint: disable=no-name-in-module

        return grouped_gemm_dswiglu_wrapper_sm100

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_supported(cls) -> bool:
        """Whether this fused operation is supported on the current system."""
        if get_device_compute_capability() < (10, 0):
            # Kernel requires SM100+
            return False
        try:
            # Make sure kernel is available
            cls.grouped_gemm_dswiglu_kernel()
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
            self.grouped_gemm_dswiglu_kernel()  # Try triggering import error
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

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        **unused,  # pylint: disable=unused-argument
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[()]],
    ]:

        import sys as _sys; _sys.stderr.write("[BWD_ENTRY] fuser_backward entered\n"); _sys.stderr.flush()
        print("[BWD_ENTRY] fuser_backward entered", flush=True)
        # Get basic operations
        fc1_op, _, fc2_op = self.basic_ops
        fc1_ctx, swiglu_ctx, fc2_ctx = basic_op_ctxs

        # Tensor properties
        out_shape = list(grad_output.size())
        assert len(out_shape) == 2, f"Expected 2D grad output tensor, got shape={out_shape}."
        fc1_weight_shape = (fc1_op.out_features, fc1_op.in_features)
        fc2_weight_shape = (fc2_op.out_features, fc2_op.in_features)
        num_groups = fc1_op.num_groups
        fc1_weight_param = fc1_op.weight if fc1_op.single_grouped_parameter else fc1_op.weight0
        device = fc1_weight_param.device
        dtype = fc1_ctx.dtype

        # Saved tensors from FC1 forward
        saved_tensors = fc1_ctx.saved_tensors
        split_sizes, split_points, saved_tensors = (
            saved_tensors[0],
            saved_tensors[1],
            saved_tensors[2:],
        )

        if fc1_op.single_grouped_parameter:
            grouped_fc1_weight, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            grouped_fc1_weight, saved_tensors = (
                saved_tensors[:num_groups],
                saved_tensors[num_groups:],
            )

        (
            fc1_x_data,
            fc1_x_col_data,
            fc1_x_scale,
            fc1_x_col_scale,
            fc1_x_tensor_offsets,
            fc1_x_amax,
            fc1_x_col_amax,
        ), saved_tensors = (
            saved_tensors[:7],
            saved_tensors[7:],
        )

        # Saved tensors from scaled SwiGLU forward
        swiglu_in, scales = swiglu_ctx.saved_tensors

        # Saved tensors from FC2 forward
        saved_tensors = fc2_ctx.saved_tensors
        _, saved_tensors = saved_tensors[0], saved_tensors[1:]  # Assume same split sizes as FC1
        if fc2_op.single_grouped_parameter:
            grouped_fc2_weight, saved_tensors = saved_tensors[0], saved_tensors[1:]
        else:
            grouped_fc2_weight, saved_tensors = (
                saved_tensors[:num_groups],
                saved_tensors[num_groups:],
            )

        (
            fc2_x_data,
            fc2_x_col_data,
            fc2_x_scale,
            fc2_x_col_scale,
            fc2_x_tensor_offsets,
            fc2_x_amax,
            fc2_x_col_amax,
        ), saved_tensors = (
            saved_tensors[:7],
            saved_tensors[7:],
        )

        # Group splits
        if int(split_sizes.numel()) != num_groups:
            raise ValueError(f"Expected {num_groups} splits, but got {int(split_sizes.numel())}.")
        split_sizes = split_sizes.to(dtype=torch.int64, device=device)
        split_points = split_points.to(dtype=torch.int, device=device)

        grouped_fc1_x = None
        if fc1_ctx.weight_requires_grad:
            grouped_fc1_x = make_grouped_tensor_from_buffers(
                num_groups=num_groups,
                data=fc1_x_data,
                columnwise_data=fc1_x_col_data,
                scale_inv=fc1_x_scale,
                columnwise_scale_inv=fc1_x_col_scale,
                split_sizes=split_sizes,
                logical_last_dim=fc1_weight_shape[1],
                dtype=dtype,
                quantizer=fc1_ctx.input_quantizers[0],
                with_gemm_swizzled_scales=True,
                tensor_offsets=fc1_x_tensor_offsets,
            )
            # Restore NVFP4 global amaxes needed by general_grouped_gemm_for_grouped_tensor
            # for 2-level scale dequantization: true_scale = scale_fp8 * amax / (fp8_max * fp4_max)
            if fc1_x_amax is not None:
                grouped_fc1_x.amax = fc1_x_amax
            if fc1_x_col_amax is not None:
                grouped_fc1_x.columnwise_amax = fc1_x_col_amax

        grouped_fc2_x = None
        if fc2_ctx.weight_requires_grad:
            grouped_fc2_x = make_grouped_tensor_from_buffers(
                num_groups=num_groups,
                data=fc2_x_data,
                columnwise_data=fc2_x_col_data,
                scale_inv=fc2_x_scale,
                columnwise_scale_inv=fc2_x_col_scale,
                split_sizes=split_sizes,
                logical_last_dim=fc2_weight_shape[1],
                dtype=dtype,
                quantizer=fc2_ctx.input_quantizers[0],
                with_gemm_swizzled_scales=True,
                tensor_offsets=fc2_x_tensor_offsets,
            )
            # Restore NVFP4 global amaxes needed by general_grouped_gemm_for_grouped_tensor
            # for 2-level scale dequantization: true_scale = scale_fp8 * amax / (fp8_max * fp4_max)
            if fc2_x_amax is not None:
                grouped_fc2_x.amax = fc2_x_amax
            if fc2_x_col_amax is not None:
                grouped_fc2_x.columnwise_amax = fc2_x_col_amax

        # Split grad output tensor and convert dtypes if needed
        fc2_dy = maybe_dequantize(grad_output, dtype)
        _fc2_w = grouped_fc2_weight if fc2_op.single_grouped_parameter else grouped_fc2_weight[0]
        use_nvfp4 = isinstance(fc2_ctx.grad_output_quantizers[0], NVFP4Quantizer) or type(_fc2_w).__name__ == 'NVFP4Tensor'
        import sys as _sys2, os as _os2; _sys2.stderr.write(f"[BWD use_nvfp4={use_nvfp4}] grad_out_q type={type(fc2_ctx.grad_output_quantizers[0]).__name__}  fc2_w type={type(_fc2_w).__name__}  single_grouped={fc2_op.single_grouped_parameter}\n"); _sys2.stderr.flush()
        if int(_os2.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
            _g = fc2_dy.detach().float()
            _sys2.__stdout__.write(
                f"[BWD_FUSED_GRAD_IN]  nan={_g.isnan().any().item()} min={_g.min().item():.4g} max={_g.abs().max().item():.4g} norm={_g.norm().item():.4g}\n"
            )
            _sys2.__stdout__.flush()
            if use_nvfp4:
                # Print per-block amax for fc2_dy, split by group
                _nvfp4_block = NVFP4_BLOCK_SCALING_SIZE
                _row_offset = 0
                for _gi, _sz in enumerate(split_sizes.tolist()):
                    if _sz == 0:
                        _sys2.__stdout__.write(f"[BWD_FUSED_GRAD_IN amax_block] group={_gi} sz=0 (empty)\n")
                        continue
                    _g_grp = _g[_row_offset : _row_offset + _sz]  # (sz, K)
                    # Compute per-block amax over each 16-element run along K dimension
                    _g_flat = _g_grp.reshape(-1, _nvfp4_block)  # (sz*K/16, 16)
                    _blk_amaxes = _g_flat.abs().amax(dim=1)     # (sz*K/16,)
                    _grp_amax = _blk_amaxes.max().item()
                    _sys2.__stdout__.write(
                        f"[BWD_FUSED_GRAD_IN amax_block] group={_gi} sz={_sz}"
                        f"  num_blocks={_blk_amaxes.numel()}"
                        f"  block_amax_min={_blk_amaxes.min().item():.4f}"
                        f"  block_amax_max={_blk_amaxes.max().item():.4f}"
                        f"  global_amax={_grp_amax:.4f}"
                        f"  first8_block_amaxes={_blk_amaxes[:8].tolist()}\n"
                    )
                    _row_offset += _sz
                _sys2.__stdout__.flush()
        for quantizer in fc2_ctx.grad_output_quantizers:
            quantizer.set_usage(rowwise=True, columnwise=fc2_ctx.weight_requires_grad)
            quantizer.optimize_for_gemm = True
            if use_nvfp4:
                quantizer.with_rht = True
                quantizer.with_post_rht_amax = True
        grouped_fc2_dy = tex.group_quantize(
            fc2_dy, fc2_ctx.grad_output_quantizers[0], num_groups, split_sizes
        )
        data_dtype = torch.float4_e2m1fn_x2 if use_nvfp4 else torch.float8_e4m3fn
        scale_view_dtype = torch.float8_e4m3fn if use_nvfp4 else torch.float8_e8m0fnu
        sf_vec_size = NVFP4_BLOCK_SCALING_SIZE if use_nvfp4 else MXFP8_BLOCK_SCALING_SIZE
        data_k = out_shape[1] // 2 if use_nvfp4 else out_shape[1]
        weight_k = fc2_weight_shape[1] // 2 if use_nvfp4 else fc2_weight_shape[1]
        k_sf_divisor = 2 * sf_vec_size if use_nvfp4 else 4 * sf_vec_size

        def _pack_nvfp4_amax_list(tensors: object) -> None:
            """Ensure discrete NVFP4 tensor list uses contiguous per-group amax buffers."""
            if not (use_nvfp4 and isinstance(tensors, list) and tensors):
                return
            row_amaxes = [getattr(tensor, "_amax_rowwise", None) for tensor in tensors]
            if all(amax is not None for amax in row_amaxes):
                packed_row_amax = torch.cat([amax.view(-1) for amax in row_amaxes], dim=0).contiguous()
                for idx, tensor in enumerate(tensors):
                    tensor._amax_rowwise = packed_row_amax[idx : idx + 1]
            col_amaxes = [getattr(tensor, "_amax_columnwise", None) for tensor in tensors]
            if all(amax is not None for amax in col_amaxes):
                packed_col_amax = torch.cat([amax.view(-1) for amax in col_amaxes], dim=0).contiguous()
                for idx, tensor in enumerate(tensors):
                    tensor._amax_columnwise = packed_col_amax[idx : idx + 1]

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
        fc2_dy_data = grouped_fc2_dy.rowwise_data.view(dtype=data_dtype)
        fc2_dy_data = fc2_dy_data.view(out_shape[0], data_k)
        fc2_dy_data = fc2_dy_data.unsqueeze(0).permute(1, 2, 0)
        fc2_dy_scales = grouped_fc2_dy.scale_inv
        fc2_dy_scales = fc2_dy_scales.view(dtype=scale_view_dtype)
        if use_nvfp4:
            _nvfp4_fp8_max = 448.0
            _nvfp4_fp4_max = 6.0
            _amax_fc2_dy = getattr(grouped_fc2_dy, 'amax', None)
            if _amax_fc2_dy is None:
                _amax_fc2_dy = getattr(grouped_fc2_dy, '_amax_rowwise')
            _amax_fc2_dy = _amax_fc2_dy.float().view(-1)
            if int(_os2.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
                _raw_dy_si = fc2_dy_scales.view(-1)[:4].tolist()
                _amax_fc2_dy0 = _amax_fc2_dy.view(-1)[0].item()
                _scale0 = fc2_dy_scales.view(-1)[:1].view(torch.float8_e4m3fn).float().item()
                _sys2.__stdout__.write(
                    f"[BWD_FUSED_fc2_dy]  amax={_amax_fc2_dy.tolist()}"
                    f"  scale_inv_raw[:4]={_raw_dy_si}"
                    f"  scales_after_convert[block0]={_scale0 * _amax_fc2_dy0 / (_nvfp4_fp8_max * _nvfp4_fp4_max):.6f}\n"
                )
                _sys2.__stdout__.flush()
            if _amax_fc2_dy.numel() > 1:
                _total_scales = fc2_dy_scales.numel()
                _total_rows = split_sizes.sum().float()
                _scales_per_group = (split_sizes.float() / _total_rows * _total_scales).long()
                _scales_per_group[-1] = _total_scales - _scales_per_group[:-1].sum()
                _amax_fc2_dy_expanded = torch.repeat_interleave(_amax_fc2_dy, _scales_per_group)
            else:
                _amax_fc2_dy_expanded = _amax_fc2_dy
            fc2_dy_scales = (fc2_dy_scales.float() * (_amax_fc2_dy_expanded / (_nvfp4_fp8_max * _nvfp4_fp4_max))).to(scale_view_dtype)
        if use_nvfp4 and getattr(grouped_fc2_dy, 'with_gemm_swizzled_scales', False):
            # RHT kernel with optimize_for_gemm=True writes scales directly in
            # SwizzledSFALayout (cuDNN-compatible format). Direct view, no permute needed.
            fc2_dy_scales = fc2_dy_scales.view(
                32, 4, out_shape[0] // 128, 4, data_k // k_sf_divisor, 1,
            )
        else:
            # Unswizzled TE format: permute to cuDNN format
            fc2_dy_scales = fc2_dy_scales.view(
                1,
                out_shape[0] // 128,
                data_k // k_sf_divisor,
                32,
                4,
                4,
            )
            fc2_dy_scales = fc2_dy_scales.permute(3, 4, 1, 5, 2, 0)

        # Pack weight tensors
        # Note: Fused kernel expects tensor with non-contiguous
        # logical dims.
        # Data actual shape: (num_groups, k, n)
        # Scale actual shape: (num_groups, n/128, k/128, 32 (block col),
        #  4 (block col), 4 (block row))
        # Data logical shape: (n, k, num_groups)
        # Scale logical shape: (32 (block col), 4 (block col), n/128,
        #   4 (block row), k/128, num_groups)
        fc2_w_data = (
            grouped_fc2_weight.columnwise_data
            if fc2_op.single_grouped_parameter
            else noop_cat([w._columnwise_data for w in grouped_fc2_weight])
        )
        import sys as _sys3; _sys3.stderr.write(f"[BWD fc2_w_data] shape={fc2_w_data.shape} dtype={fc2_w_data.dtype} weight_k={weight_k} fc2_weight_shape={fc2_weight_shape}\n"); _sys3.stderr.flush()
        fc2_w_data = fc2_w_data.view(dtype=data_dtype)
        if use_nvfp4:
            # columnwise_data is (in_features=n, out_features/2=k_packed) — already (n, k_packed)
            # Need (n, k_packed, l) with k-major strides (k_packed, 1, n*k_packed)
            # so that logical fp4 strides = (k_fp4, 1, n*k_fp4).
            fc2_w_data = fc2_w_data.reshape(num_groups, fc2_weight_shape[1], fc2_weight_shape[0] // 2)
            fc2_w_data = fc2_w_data.permute(1, 2, 0)
        else:
            fc2_w_data = fc2_w_data.view(num_groups, fc2_weight_shape[0], weight_k)
            fc2_w_data = fc2_w_data.permute(2, 1, 0)
        fc2_w_scales = (
            grouped_fc2_weight.columnwise_scale_inv
            if fc2_op.single_grouped_parameter
            else noop_cat([w._columnwise_scale_inv for w in grouped_fc2_weight])
        )
        fc2_w_scales = fc2_w_scales.view(dtype=scale_view_dtype)
        if use_nvfp4:
            _w2_for_amax = grouped_fc2_weight if fc2_op.single_grouped_parameter else grouped_fc2_weight[0]
            if fc2_op.single_grouped_parameter:
                # GroupedTensor uses .columnwise_amax (not ._amax_columnwise which is on individual NVFP4Tensor)
                _amax_fc2_w_col = (grouped_fc2_weight.columnwise_amax
                                   if grouped_fc2_weight.columnwise_amax is not None
                                   else getattr(grouped_fc2_weight, '_amax_columnwise')).float()
            else:
                # Per-group weights: gather each group's columnwise amax into a vector
                _col_amaxes = []
                import sys as _dbg_sys
                for _gi2, _w in enumerate(grouped_fc2_weight):
                    _a = getattr(_w, '_amax_columnwise', None)
                    _dbg_sys.stderr.write(
                        f"[BWD_fc2_w_col_amax] group={_gi2} _amax_columnwise="
                        f"{'None' if _a is None else _a.float().view(-1).tolist()}\n"
                    ); _dbg_sys.stderr.flush()
                    if _a is None:
                        _a = torch.zeros(1, dtype=torch.float32, device=fc2_w_scales.device)
                    _col_amaxes.append(_a.float().view(-1)[0])
                _amax_fc2_w_col = torch.stack(_col_amaxes)
            _amax_fc2_w_col = _amax_fc2_w_col.view(-1)
            if int(_os2.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
                _raw_w2_si = fc2_w_scales.view(-1)[:4].tolist()
                _amax_fc2_w_col0 = _amax_fc2_w_col.view(-1)[0].item()
                _scale0 = fc2_w_scales.view(-1)[:1].view(torch.float8_e4m3fn).float().item()
                _sys2.__stdout__.write(
                    f"[BWD_FUSED_fc2_w]  _amax_columnwise={_amax_fc2_w_col.tolist()}"
                    f"  _amax_rowwise={getattr(_w2_for_amax, '_amax_rowwise', 'N/A')}"
                    f"  col_scale_inv_raw[:4]={_raw_w2_si}"
                    f"  scales_after_convert[block0]={_scale0 * _amax_fc2_w_col0 / (_nvfp4_fp8_max * _nvfp4_fp4_max):.6f}\n"
                )
                _sys2.__stdout__.flush()
            if _amax_fc2_w_col.numel() > 1:
                _sf_per_group_w2 = fc2_w_scales.numel() // _amax_fc2_w_col.numel()
                _amax_fc2_w_col_expanded = _amax_fc2_w_col.repeat_interleave(_sf_per_group_w2)
            else:
                _amax_fc2_w_col_expanded = _amax_fc2_w_col
            _fc2_w_scales_shape = fc2_w_scales.shape
            fc2_w_scales = (fc2_w_scales.float().reshape(-1) * (_amax_fc2_w_col_expanded / (_nvfp4_fp8_max * _nvfp4_fp4_max))).to(scale_view_dtype).reshape(_fc2_w_scales_shape)
        if use_nvfp4:
            # For NVFP4 columnwise: n=in_features=28672, k_packed=out_features/2=4096
            weight_k_col = fc2_weight_shape[0] // 2  # packed k = out_features/2
            fc2_w_scales = fc2_w_scales.view(
                num_groups,
                fc2_weight_shape[1] // 128,  # n dim = in_features/128 = 224
                4,
                weight_k_col // k_sf_divisor,  # k scale dim = 4096/32 = 128
                4,
                32,
            )  # Unswizzled layout
        else:
            fc2_w_scales = fc2_w_scales.view(
                num_groups,
                fc2_weight_shape[0] // 128,
                4,
                weight_k // k_sf_divisor,
                4,
                32,
            )  # Unswizzled layout
        fc2_w_scales = fc2_w_scales.permute(
            0, 3, 1, 5, 4, 2
        ).contiguous()  # Convert to swizzled layout
        fc2_w_scales = fc2_w_scales.permute(3, 4, 2, 5, 1, 0)

        # Kernel scaling factors
        alpha_tensor, norm_const_tensor = self._get_kernel_constants(
            num_groups=num_groups, dtype=dtype, device=device
        )
        norm_const_tensor_arg = None if use_nvfp4 else norm_const_tensor
        current_stream = cuda.CUstream(  # pylint: disable=c-extension-no-member
            torch.cuda.current_stream().cuda_stream
        )

        # Fused kernel for FC2 dgrad + dSwiGLU + grad scale
        fc2_dgrad_kernel_out = self.grouped_gemm_dswiglu_kernel()(
            fc2_dy_data,
            fc2_w_data,
            swiglu_in.unsqueeze(0).permute(1, 2, 0),
            fc2_dy_scales,
            fc2_w_scales,
            split_points,
            alpha_tensor,  # alpha_tensor
            alpha_tensor,  # beta_tensor
            scales.detach().to(dtype=torch.float32).reshape(-1, 1, 1),
            norm_const_tensor=norm_const_tensor_arg,
            d_dtype=torch.bfloat16 if use_nvfp4 else torch.float8_e4m3fn,
            cd_major="n",
            sf_vec_size=sf_vec_size,
            current_stream=current_stream,
            discrete_col_sfd=not use_nvfp4,
        )

        # Unpack kernel outputs
        # Note: Fused kernel outputs tensors with non-contiguous
        # logical dims.
        # Row-wise data logical shape: (sum(m), k, 1)
        # Row-wise scale logical shape: (32 (block row), 4 (block row),
        #   sum(m)/128, 4 (block col), k/128, 1)
        # Column-wise data logical shape: (k, sum(m), 1)
        # Column-wise scale logical shape: (32 (block col), 4 (block col),
        #   k/128, 4 (block row), sum(m)/128, 1)
        fc1_dy_row_data = fc2_dgrad_kernel_out["d_row_tensor"]
        if int(_os2.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
            _kout = fc1_dy_row_data.detach().float()
            _sys2.__stdout__.write(
                f"[BWD_FUSED_kernel_out d_row]  nan={_kout.isnan().any().item()} min={_kout.min().item():.4g} max={_kout.abs().max().item():.4g}\n"
            )
            _sys2.__stdout__.flush()
        fc1_dy_row_data = fc1_dy_row_data.permute(2, 0, 1)
        fc1_dy_row_data = fc1_dy_row_data.view(out_shape[0], fc1_weight_shape[0]).contiguous()

        grad_scales = fc2_dgrad_kernel_out["dprob_tensor"]
        grad_scales = grad_scales.view(-1).to(dtype=dtype)

        # FC1 grad output for dgrad and wgrad GEMMs
        if use_nvfp4:
            for quantizer in fc1_ctx.grad_output_quantizers:
                quantizer.set_usage(rowwise=True, columnwise=fc1_ctx.weight_requires_grad)
                quantizer.optimize_for_gemm = True
                quantizer.with_rht = True
                quantizer.with_post_rht_amax = True
            grouped_fc1_dy = tex.group_quantize(
                fc1_dy_row_data,
                fc1_ctx.grad_output_quantizers[0],
                num_groups,
                split_sizes,
            )
        else:
            fc1_dy_row_scale = fc2_dgrad_kernel_out["sfd_row_tensor"]
            fc1_dy_row_scale = fc1_dy_row_scale.permute(5, 2, 4, 0, 1, 3)
            fc1_dy_row_scale = fc1_dy_row_scale.view(
                out_shape[0], fc1_weight_shape[0] // 32
            ).contiguous()
            fc1_dy_col_data = fc2_dgrad_kernel_out["d_col_tensor"]
            fc1_dy_col_data = fc1_dy_col_data.permute(2, 0, 1)
            fc1_dy_col_data = fc1_dy_col_data.view(out_shape[0], fc1_weight_shape[0]).contiguous()
            fc1_dy_col_scale = fc2_dgrad_kernel_out["sfd_col_tensor"]
            fc1_dy_col_scale = fc1_dy_col_scale.permute(5, 2, 4, 0, 1, 3)
            fc1_dy_col_scale = fc1_dy_col_scale.reshape(-1)
            grouped_fc1_dy = make_grouped_tensor_from_buffers(
                num_groups=num_groups,
                data=fc1_dy_row_data,
                columnwise_data=fc1_dy_col_data,
                scale_inv=fc1_dy_row_scale,
                columnwise_scale_inv=fc1_dy_col_scale,
                split_sizes=split_sizes,
                logical_last_dim=fc1_weight_shape[0],
                dtype=dtype,
                quantizer=fc1_ctx.grad_output_quantizers[0],
                with_gemm_swizzled_scales=True,
            )

        # FC2 wgrad GEMM
        fc2_packed_wgrad = None
        fc2_weight_grads: list[Optional[torch.Tensor]]
        if fc2_op.single_grouped_parameter:
            fc2_weight_grads = [None]
        else:
            fc2_weight_grads = [None] * num_groups
        if fc2_ctx.weight_requires_grad:

            # Initialize grad buffers
            accumulate_into_main_grad = False
            if fc2_op.single_grouped_parameter:
                grouped_fc2_wgrad = None
                weight_param = fc2_op.weight
                if fc2_op._accumulate_into_main_grad:
                    # Megatron-LM wgrad fusion
                    # Note: Get grad tensors from params so we can
                    # accumulate directly into it.
                    if hasattr(weight_param, "__fsdp_param__"):
                        weight_param.main_grad = weight_param.get_main_grad()
                    main_grad = weight_param.main_grad
                    grouped_shape = (num_groups, *fc2_weight_shape)
                    if main_grad.shape != grouped_shape:
                        if main_grad.numel() != math.prod(grouped_shape):
                            raise RuntimeError(
                                "Grouped MLP fused backward expected FC2 main_grad to have "
                                f"shape {grouped_shape} or matching numel, "
                                f"but got shape {tuple(main_grad.shape)}"
                            )
                        # Keep aliasing with weight.main_grad; do not allow implicit copies.
                        try:
                            main_grad = main_grad.view(grouped_shape)
                        except RuntimeError as e:
                            raise RuntimeError(
                                "Grouped MLP fused backward requires FC2 main_grad to be viewable"
                                f" as {grouped_shape} without copy, but got shape"
                                f" {tuple(main_grad.shape)} and stride {tuple(main_grad.stride())}"
                            ) from e
                    accumulate_into_main_grad = not getattr(
                        weight_param, "overwrite_main_grad", False
                    )
                    if accumulate_into_main_grad:
                        grouped_fc2_wgrad = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                            num_tensors=num_groups,
                            tensor_shape=fc2_weight_shape,
                            rowwise_data=main_grad,
                            dtype=main_grad.dtype,
                        )

                if grouped_fc2_wgrad is None:
                    grouped_fc2_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                        num_tensors=num_groups,
                        shapes=[fc2_weight_shape] * num_groups,
                        quantizer=None,
                        device=device,
                        dtype=dtype,
                    )

                # Launch GEMM
                # A=grouped_input, B=grouped_fc2_dy; B's scales are GEMM-swizzled (see group_quantize above).
                general_grouped_gemm_for_grouped_tensor(
                    grouped_fc2_x,
                    grouped_fc2_dy,
                    grouped_fc2_wgrad,
                    layout="NT",
                    accumulate=accumulate_into_main_grad,
                )
                fc2_packed_wgrad = grouped_fc2_wgrad.rowwise_data.view(
                    num_groups, *fc2_weight_shape
                )
                if accumulate_into_main_grad and hasattr(weight_param, "grad_added_to_main_grad"):
                    weight_param.grad_added_to_main_grad = True
                    fc2_packed_wgrad = get_dummy_wgrad(
                        list(weight_param.size()),
                        weight_param.dtype,
                        zero=getattr(weight_param, "zero_out_wgrad", False),
                    )
            else:
                if fc2_op._accumulate_into_main_grad:
                    for idx in range(num_groups):
                        weight_param = getattr(fc2_op, f"weight{idx}")
                        if hasattr(weight_param, "__fsdp_param__"):
                            weight_param.main_grad = weight_param.get_main_grad()
                        fc2_weight_grads[idx] = weight_param.main_grad
                    accumulate_into_main_grad = not getattr(
                        fc2_op.weight0, "overwrite_main_grad", False
                    )
                else:
                    for idx in range(num_groups):
                        fc2_weight_grads[idx] = torch.empty(
                            fc2_weight_shape, dtype=dtype, device=device
                        )

                general_grouped_gemm_for_grouped_tensor(
                    grouped_fc2_x,
                    grouped_fc2_dy,
                    fc2_weight_grads,
                    layout="NT",
                    accumulate=accumulate_into_main_grad,
                )
                if accumulate_into_main_grad:
                    for idx in range(num_groups):
                        weight_param = getattr(fc2_op, f"weight{idx}")
                        if hasattr(weight_param, "grad_added_to_main_grad"):
                            weight_param.grad_added_to_main_grad = True
                            fc2_weight_grads[idx] = get_dummy_wgrad(
                                list(weight_param.size()),
                                weight_param.dtype,
                                zero=getattr(weight_param, "zero_out_wgrad", False),
                            )

        # Clear FC2 input tensor if possible
        if grouped_fc2_x is not None:
            clear_tensor_data(
                grouped_fc2_x.data,
                grouped_fc2_x.columnwise_data,
                grouped_fc2_x.scale_inv,
                grouped_fc2_x.columnwise_scale_inv,
            )

        # FC1 dgrad GEMM
        grad_input = None
        if fc1_ctx.input_requires_grad:
            _pack_nvfp4_amax_list(grouped_fc1_weight)
            if (
                use_nvfp4
                and hasattr(grouped_fc1_weight, "with_gemm_swizzled_scales")
                and grouped_fc1_weight.with_gemm_swizzled_scales is not True
            ):
                grouped_fc1_weight.with_gemm_swizzled_scales = True
            if (
                use_nvfp4
                and hasattr(grouped_fc1_dy, "with_gemm_swizzled_scales")
                and grouped_fc1_dy.with_gemm_swizzled_scales is not True
            ):
                grouped_fc1_dy.with_gemm_swizzled_scales = True
            # Launch GEMM
            in_shape = out_shape[:-1] + [fc1_weight_shape[1]]
            grad_input = torch.empty(in_shape, dtype=dtype, device=device)
            grouped_grad_input = make_grouped_tensor_from_buffers(
                num_groups=num_groups,
                data=grad_input,
                split_sizes=split_sizes,
                dtype=grad_input.dtype,
                logical_last_dim=fc1_weight_shape[1],
            )

            general_grouped_gemm_for_grouped_tensor(
                grouped_fc1_weight,
                grouped_fc1_dy,
                grouped_grad_input,
                layout="NN",
                accumulate=False,
            )

        # FC1 wgrad GEMM
        fc1_packed_wgrad = None
        fc1_weight_grads: list[Optional[torch.Tensor]]
        if fc1_op.single_grouped_parameter:
            fc1_weight_grads = [None]
        else:
            fc1_weight_grads = [None] * num_groups
        if fc1_ctx.weight_requires_grad:

            # Initialize grad buffers
            accumulate_into_main_grad = False
            if fc1_op.single_grouped_parameter:
                grouped_fc1_wgrad = None
                weight_param = fc1_op.weight
                if fc1_op._accumulate_into_main_grad:
                    # Megatron-LM wgrad fusion
                    # Note: Get grad tensors from params so we can
                    # accumulate directly into it.
                    if hasattr(weight_param, "__fsdp_param__"):
                        weight_param.main_grad = weight_param.get_main_grad()
                    main_grad = weight_param.main_grad
                    grouped_shape = (num_groups, *fc1_weight_shape)
                    if main_grad.shape != grouped_shape:
                        if main_grad.numel() != math.prod(grouped_shape):
                            raise RuntimeError(
                                "Grouped MLP fused backward expected FC1 main_grad to have "
                                f"shape {grouped_shape} or matching numel, "
                                f"but got shape {tuple(main_grad.shape)}"
                            )
                        # Keep aliasing with weight.main_grad; do not allow implicit copies.
                        try:
                            main_grad = main_grad.view(grouped_shape)
                        except RuntimeError as e:
                            raise RuntimeError(
                                "Grouped MLP fused backward requires FC1 main_grad to be viewable"
                                f" as {grouped_shape} without copy, but got shape"
                                f" {tuple(main_grad.shape)} and stride {tuple(main_grad.stride())}"
                            ) from e
                    accumulate_into_main_grad = not getattr(
                        weight_param, "overwrite_main_grad", False
                    )
                    if accumulate_into_main_grad:
                        grouped_fc1_wgrad = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                            num_tensors=num_groups,
                            tensor_shape=fc1_weight_shape,
                            rowwise_data=main_grad,
                            dtype=main_grad.dtype,
                        )

                if grouped_fc1_wgrad is None:
                    grouped_fc1_wgrad = GroupedTensor.make_grouped_tensor_with_shapes(
                        num_tensors=num_groups,
                        shapes=[fc1_weight_shape] * num_groups,
                        quantizer=None,
                        device=device,
                        dtype=dtype,
                    )

                # Launch GEMM
                general_grouped_gemm_for_grouped_tensor(
                    grouped_fc1_x,
                    grouped_fc1_dy,
                    grouped_fc1_wgrad,
                    layout="NT",
                    accumulate=accumulate_into_main_grad,
                )
                fc1_packed_wgrad = grouped_fc1_wgrad.rowwise_data.view(
                    num_groups, *fc1_weight_shape
                )
                if accumulate_into_main_grad and hasattr(weight_param, "grad_added_to_main_grad"):
                    weight_param.grad_added_to_main_grad = True
                    fc1_packed_wgrad = get_dummy_wgrad(
                        list(weight_param.size()),
                        weight_param.dtype,
                        zero=getattr(weight_param, "zero_out_wgrad", False),
                    )
            else:
                if fc1_op._accumulate_into_main_grad:
                    for idx in range(num_groups):
                        weight_param = getattr(fc1_op, f"weight{idx}")
                        if hasattr(weight_param, "__fsdp_param__"):
                            weight_param.main_grad = weight_param.get_main_grad()
                        fc1_weight_grads[idx] = weight_param.main_grad
                    accumulate_into_main_grad = not getattr(
                        fc1_op.weight0, "overwrite_main_grad", False
                    )
                else:
                    for idx in range(num_groups):
                        fc1_weight_grads[idx] = torch.empty(
                            fc1_weight_shape, dtype=dtype, device=device
                        )

                general_grouped_gemm_for_grouped_tensor(
                    grouped_fc1_x,
                    grouped_fc1_dy,
                    fc1_weight_grads,
                    layout="NT",
                    accumulate=accumulate_into_main_grad,
                )
                if accumulate_into_main_grad:
                    for idx in range(num_groups):
                        weight_param = getattr(fc1_op, f"weight{idx}")
                        if hasattr(weight_param, "grad_added_to_main_grad"):
                            weight_param.grad_added_to_main_grad = True
                            fc1_weight_grads[idx] = get_dummy_wgrad(
                                list(weight_param.size()),
                                weight_param.dtype,
                                zero=getattr(weight_param, "zero_out_wgrad", False),
                            )

        # Clear FC1 input tensor if possible
        if grouped_fc1_x is not None:
            clear_tensor_data(
                grouped_fc1_x.data,
                grouped_fc1_x.columnwise_data,
                grouped_fc1_x.scale_inv,
                grouped_fc1_x.columnwise_scale_inv,
            )

        # Construct param grads in parameter registration order.
        if fc1_op.single_grouped_parameter:
            fc1_weight_grads = [fc1_packed_wgrad] if fc1_packed_wgrad is not None else [None]
        if fc2_op.single_grouped_parameter:
            fc2_weight_grads = [fc2_packed_wgrad] if fc2_packed_wgrad is not None else [None]

        import sys as _bwd_sys, os as _bwd_os
        _bwd_step = getattr(self, '_dbg_bwd_step', 0)
        if int(_bwd_os.getenv("SLURM_PROCID", 0)) == 0 and not torch.cuda.is_current_stream_capturing():
            _fc1w = (grouped_fc1_weight if fc1_op.single_grouped_parameter else grouped_fc1_weight[0])
            _fc2w = (grouped_fc2_weight if fc2_op.single_grouped_parameter else grouped_fc2_weight[0])
            _gi = grad_input.float() if grad_input is not None else None
            _bwd_sys.__stdout__.write(
                f"[BWD step={_bwd_step} L=?]"
                f"  grad_input={'None' if _gi is None else f'nan={_gi.isnan().any().item()} min={_gi.min().item():.4g} max={_gi.max().item():.4g}'}"
                f"  fc1w type={type(_fc1w).__name__}"
                f"  fc2w type={type(_fc2w).__name__}"
                f"  fc1_wgrad={'None' if fc1_packed_wgrad is None else f'nan={fc1_packed_wgrad.float().isnan().any().item()} max={fc1_packed_wgrad.float().abs().max().item():.4g}'}"
                f"  fc2_wgrad={'None' if fc2_packed_wgrad is None else f'nan={fc2_packed_wgrad.float().isnan().any().item()} max={fc2_packed_wgrad.float().abs().max().item():.4g}'}\n"
            )
            _bwd_sys.__stdout__.flush()
        self._dbg_bwd_step = _bwd_step + 1

        return (
            grad_input,
            [fc1_weight_grads, (), fc2_weight_grads],
            [(None,), (grad_scales,), (None,)],
        )

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

        return alpha_tensor, norm_const_tensor


def fuse_backward_ops(
    ops: list[FusibleOperation],
    *,
    recipe: Optional[Recipe] = None,
    **unused,  # pylint: disable=unused-argument
) -> list[FusibleOperation]:
    """Apply operation fusion for backward pass.

    Parameters
    ----------
    ops : list of FusibleOperation
        Forward pass operations.
    recipe : Recipe, optional
        Quantization recipe.

    Returns
    -------
    ops : list of FusibleOperation
        Updated backward pass operations

    """

    # Return immediately if fused kernel is not supported
    if not BackwardGroupedMLP_CuTeGEMMDSwiGLU_BlockScaled.is_supported():
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
            op = BackwardGroupedMLP_CuTeGEMMDSwiGLU_BlockScaled(
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
if BackwardGroupedMLP_CuTeGEMMDSwiGLU_BlockScaled.is_supported():
    register_backward_fusion(fuse_backward_ops, prepend=True)
