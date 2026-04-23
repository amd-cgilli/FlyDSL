#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Test and benchmark harness for the 4-wave FP8 blockscale preshuffle GEMM.

Usage:
    PYTHONPATH=./ python tests/kernels/test_preshuffle_gemm_fp8_4wave.py \
        -M 16384 -N 16384 -K 16384 --num_iters 20
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl.compiler as flyc
from flydsl.runtime.device import get_rocm_arch
from kernels.preshuffle_gemm_fp8_4wave import compile_fp8_gemm_4wave
from tests.test_common import run_perftest, verify_output
from tests.utils import shuffle_weight

DTYPE_FP8 = torch.float8_e4m3fn

BLOCK_SHAPE = (128, 128)  # (scale_block_n, scale_block_k)


def run_torch_blockscale(x, weight, x_scale, w_scale, block_shape=BLOCK_SHAPE,
                         dtype=torch.bfloat16):
    """Torch reference for blockscale GEMM."""
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k

    # Dequant A: x_f32[m, k] = x[m, k] * x_scale[m, scale_k] (broadcast over block)
    x_f32 = (
        x.to(x_scale.dtype).view(m, k // block_shape_k, block_shape_k)
        * x_scale.unsqueeze(-1)
    )
    x_f32 = x_f32.view(m, k)

    # Dequant B: weight_f32[n, k] = weight[n, k] * w_scale[scale_n, scale_k] (broadcast)
    w_scale_expanded = (
        w_scale.view(-1, 1)
        .repeat(1, block_shape_n * block_shape_k)
        .view(scale_n, scale_k, block_shape_n, block_shape_k)
        .permute(0, 2, 1, 3)
        .reshape(scale_n * block_shape_n, scale_k * block_shape_k)
    )
    w_scale_expanded = w_scale_expanded[:n, :k]
    weight_f32 = weight.to(w_scale_expanded.dtype) * w_scale_expanded

    out = F.linear(x_f32.to(torch.float32), weight_f32.to(torch.float32))
    return out.to(dtype)


def test_fp8_gemm_4wave(
    M,
    N,
    K,
    *,
    tile_m=64,
    tile_n=128,
    tile_k=128,
    num_iters=20,
    num_warmup=3,
    test_graph=False,
):
    """Run the 4-wave FP8 blockscale GEMM and validate against torch reference."""
    arch = str(get_rocm_arch())
    if not arch.startswith("gfx95"):
        print(f"Skipping: gfx950 required, got {arch}")
        return

    block_shape_n, block_shape_k = BLOCK_SHAPE
    scale_k = K // block_shape_k
    scale_n = N // block_shape_n

    print("=" * 80)
    print(
        f"4-Wave FP8 Blockscale GEMM  M={M}, N={N}, K={K}  "
        f"(tile={tile_m}x{tile_n}x{tile_k}, scale_block=128x128)"
    )
    print("=" * 80)

    device = torch.device("cuda")

    # ---- Compile kernel ----
    launch_fn = compile_fp8_gemm_4wave(M=M, N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
    print("Kernel compiled.")

    # ---- Prepare data ----
    x = (torch.rand((M, K), dtype=torch.float16, device=device) / 10).to(DTYPE_FP8)
    weight = (torch.rand((N, K), dtype=torch.float16, device=device) / 10).to(DTYPE_FP8)

    # Block scales
    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device=device)

    # Reference
    c_ref = run_torch_blockscale(x, weight, x_scale, w_scale, dtype=torch.float32)

    # Preshuffle B
    b_shuffled = shuffle_weight(weight, layout=(16, 16))

    # Transpose scale_a to [scale_k, M] and flatten
    x_scale_t = x_scale.transpose(0, 1).contiguous().view(-1)
    w_scale_flat = w_scale.contiguous().view(-1)

    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

    def _gemm_args(c, a, b, sa, sb):
        return (
            c.contiguous().view(-1),
            a.contiguous().view(-1),
            b.contiguous().view(-1),
            sa.contiguous().view(-1),
            sb.contiguous().view(-1),
            M,
            N,
            torch.cuda.current_stream(),
        )

    # ---- Compile JIT ----
    compiled_fn = flyc.compile(
        launch_fn,
        *_gemm_args(c_out, x, b_shuffled, x_scale_t, w_scale_flat),
    )

    def launch_kernel(c, a, b, sa, sb):
        compiled_fn(*_gemm_args(c, a, b, sa, sb))

    # ---- Benchmark ----
    num_iters = max(2, int(num_iters))
    _, us = run_perftest(
        launch_kernel,
        c_out,
        x,
        b_shuffled,
        x_scale_t,
        w_scale_flat,
        num_iters=num_iters,
        num_warmup=int(num_warmup),
        testGraph=test_graph,
    )
    torch.cuda.synchronize()

    # ---- Validate ----
    c_out_f32 = c_out.to(torch.float32)
    ok = verify_output(c_out_f32, c_ref, rtol=1e-2, atol=0.01)

    # ---- Metrics ----
    size_a = M * K
    size_b = N * K
    size_c = M * N * 2
    scales_bytes = (M * scale_k + scale_n * scale_k) * 4
    bytes_moved = size_a + size_b + size_c + scales_bytes
    flops = 2 * M * N * K
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")
    print(f"[flyc] Correct: {ok}")
    if not ok:
        diff = (c_out_f32 - c_ref).abs()
        print(f"  max error: {diff.max().item():.4f}")
        print(f"  mean error: {diff.mean().item():.4f}")
        print(f"  ref[:4,:4]:\n{c_ref[:4, :4]}")
        print(f"  out[:4,:4]:\n{c_out_f32[:4, :4]}")

    assert ok, "Output mismatch!"
    print("PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-wave FP8 blockscale GEMM benchmark")
    parser.add_argument("-M", type=int, default=16384)
    parser.add_argument("-N", type=int, default=16384)
    parser.add_argument("-K", type=int, default=16384)
    parser.add_argument("--tile_m", type=int, default=64)
    parser.add_argument("--tile_n", type=int, default=128)
    parser.add_argument("--tile_k", type=int, default=128)
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--num_warmup", type=int, default=1000)
    parser.add_argument("--test_graph", "-tg", action="store_true", default=False)
    args = parser.parse_args()

    torch.set_default_device("cuda")
    test_fp8_gemm_4wave(
        args.M,
        args.N,
        args.K,
        tile_m=args.tile_m,
        tile_n=args.tile_n,
        tile_k=args.tile_k,
        num_iters=args.num_iters,
        num_warmup=args.num_warmup,
        test_graph=args.test_graph,
    )
