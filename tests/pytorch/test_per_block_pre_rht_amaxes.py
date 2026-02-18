# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
import torch

from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
import transformer_engine_torch as tex


def inspect_tensor(pt_path: str):
    data = torch.load(pt_path, map_location="cpu")

    if isinstance(data, torch.Tensor):
        tensor = data
    elif isinstance(data, dict):
        tensor = next((v for v in data.values() if isinstance(v, torch.Tensor)), None)
        if tensor is None:
            raise ValueError("No tensor found inside the .pt dictionary")
    else:
        raise TypeError(f"Unsupported .pt content type: {type(data)}")

    bf16_amax = tensor.bfloat16().abs().amax().item()
    print(f"Shape: {tuple(tensor.shape)}")
    print(f"Min value: {tensor.min().item()}")
    print(f"Max value: {tensor.max().item()}")
    print(f"BF16 amax: {bf16_amax}")

    return tensor.to(device="cuda", dtype=torch.bfloat16), bf16_amax


def num_block_amax_elements(shape: tuple) -> int:
    rows = math.prod(shape[:-1])
    cols = shape[-1]
    return ((cols + 127) // 128) * ((rows + 127) // 128)


BLOCK_SIZE = 128


def get_block_abs_values_sorted_descending(x: torch.Tensor, block_idx: int) -> torch.Tensor:
    """Return the block's abs values (float32) sorted descending, for block index block_idx."""
    if x.dim() != 2:
        x = x.view(-1, x.shape[-1])
    rows, cols = x.shape
    num_blocks_y = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_x = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    by = block_idx // num_blocks_x
    bx = block_idx % num_blocks_x
    r0, r1 = by * BLOCK_SIZE, min((by + 1) * BLOCK_SIZE, rows)
    c0, c1 = bx * BLOCK_SIZE, min((bx + 1) * BLOCK_SIZE, cols)
    block = x[r0:r1, c0:c1].float().abs()
    pad_rows = BLOCK_SIZE - (r1 - r0)
    pad_cols = BLOCK_SIZE - (c1 - c0)
    if pad_rows > 0 or pad_cols > 0:
        block = torch.nn.functional.pad(block, (0, pad_cols, 0, pad_rows), value=0.0)
    flat = block.reshape(-1)
    return torch.sort(flat, descending=True).values


def kernel_value_rank_in_block(x: torch.Tensor, block_idx: int, kernel_amax: float) -> tuple:
    """
    Check if kernel_amax exists in the block and its 1-based rank (1 = largest).
    Returns (value_exists, rank). rank is None if value not found in block.
    """
    sorted_vals = get_block_abs_values_sorted_descending(x, block_idx)
    matches = sorted_vals == kernel_amax
    if not matches.any():
        return False, None
    rank = matches.nonzero(as_tuple=True)[0][0].item() + 1
    return True, rank


def pytorch_block_amax(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 2:
        x = x.view(-1, x.shape[-1])
    rows, cols = x.shape
    num_blocks_y = (rows + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_blocks_x = (cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    a = x.float().abs()
    pad_rows = num_blocks_y * BLOCK_SIZE - rows
    pad_cols = num_blocks_x * BLOCK_SIZE - cols
    if pad_rows > 0 or pad_cols > 0:
        a = torch.nn.functional.pad(a, (0, pad_cols, 0, pad_rows), value=0.0)
    a = (
        a.view(
            num_blocks_y,
            BLOCK_SIZE,
            num_blocks_x,
            BLOCK_SIZE,
        )
        .permute(0, 2, 1, 3)
        .reshape(num_blocks_y * num_blocks_x, BLOCK_SIZE * BLOCK_SIZE)
    )
    return a.max(dim=1).values.to(torch.float32)


def main() -> None:
    device = torch.device("cuda")
    num_repeats = 100000

    t1, t1_bf16_amax = inspect_tensor(
        "/files/dnm6/failed_inputs_set1/input_pool0-0373_rank_6_iteration_6_19677.pt"
    )
    t2, t2_bf16_amax = inspect_tensor(
        "/files/dnm6/failed_inputs_set1/input_pool0-0373_rank_7_iteration_6_19629.pt"
    )
    t3, t3_bf16_amax = inspect_tensor(
        "/files/dnm6/failed_inputs_set2/input_rank_6_iteration_20_step_18.pt"
    )
    t4, t4_bf16_amax = inspect_tensor(
        "/files/dnm6/failed_inputs_set2/input_rank_7_iteration_20_step_14.pt"
    )

    quantizer = NVFP4Quantizer(
        with_rht=True,
        with_post_rht_amax=True,
        with_2d_quantization=False,
        stochastic_rounding=False,
        with_amax_reduction=False,
    )
    quantizer.rowwise_usage = True
    quantizer.columnwise_usage = False

    tensors = [
        ("t1", t1, t1_bf16_amax),
        ("t2", t2, t2_bf16_amax),
        ("t3", t3, t3_bf16_amax),
        ("t4", t4, t4_bf16_amax),
    ]
    all_mismatch_indices = {}

    for step in range(num_repeats):
        for name, tensor, amax in tensors:
            num_blocks = num_block_amax_elements(tensor.shape)
            block_amax = torch.zeros(num_blocks, dtype=torch.float32, device=device)
            qtensor = tex.quantize(tensor, quantizer, block_amax_out=block_amax)
            with torch.no_grad():
                pytorch_block_amax_out = pytorch_block_amax(tensor)
            if amax != qtensor._amax_rowwise.item():
                print(
                    f"{name}: Wrong total amax. Expected {amax} but got"
                    f" {qtensor._amax_rowwise.item()}"
                )

            # block_amax = block_amax.sort().values
            # pytorch_block_amax_out = pytorch_block_amax_out.sort().values

            mismatch_mask = block_amax != pytorch_block_amax_out
            mismatch_indices = mismatch_mask.nonzero(as_tuple=True)[0]
            if mismatch_indices.numel() > 0:
                key = f"{name}_step{step}"
                # Store (index, kernel_amax, correct_amax) for each mismatch
                all_mismatch_indices[key] = [
                    (int(i), block_amax[i].item(), pytorch_block_amax_out[i].item())
                    for i in mismatch_indices
                ]

    name_to_tensor = {name: tensor for name, tensor, _ in tensors}

    if all_mismatch_indices:
        print("Block amax mismatch (kernel vs PyTorch ref):")
        for key, index_value_pairs in all_mismatch_indices.items():
            name = key.rsplit("_step", 1)[0]
            tensor = name_to_tensor[name]
            for idx, kernel_amax, correct_amax in index_value_pairs:
                exists, rank = kernel_value_rank_in_block(tensor, idx, kernel_amax)
                if exists:
                    rank_str = f"rank in block (1=largest): {rank}"
                else:
                    rank_str = "value NOT found in block"
                print(
                    f"  {key} index {idx}: kernel amax = {kernel_amax}, correct amax ="
                    f" {correct_amax}; kernel value {rank_str}"
                )
        assert False, f"Block amax mismatches in {len(all_mismatch_indices)} run(s)"
    print("All block amax comparisons passed.")


if __name__ == "__main__":
    main()
