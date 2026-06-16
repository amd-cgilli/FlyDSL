# SPDX-License-Identifier: Apache-2.0
"""Standalone benchmark / smoke test for the 4-wave interleaved MXFP8 GEMM.

Builds inputs the same way as ``bench_mxfp8_linear_standalone`` (MXFP8 E4M3
quantization with E8M0 block scales), compiles both the GEMM and the
scale-repack launchers from ``kernels.mxfp8_gemm_4w`` with ``flyc.compile``,
then times the GEMM with ``triton.testing.do_bench``.

Run with the default M=N=K=8192:

    python3 tests/kernels/bench_mxfp8_gemm_4w.py
"""

import argparse
import os
import sys

import torch
import triton

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import flydsl.compiler as flyc
from kernels.mxfp8_gemm_4w import compile_mxfp8_gemm_4w
from tests.test_common import verify_output

MXFP8_BLOCK_SIZE = 32
BLOCK_K = 128
FP8_DTYPE = torch.float8_e4m3fn
SCALE_DTYPE = torch.uint8


def mxfp8_e4m3_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` [..., K] to FP8 E4M3 values + E8M0 (uint8) block scales."""
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    orig_shape = x.shape
    xb = x.reshape(*orig_shape[:-1], orig_shape[-1] // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    amax = xb.abs().amax(dim=-1)
    amax = torch.where(amax == 0, torch.ones_like(amax), amax)
    scale_exp = (torch.floor(torch.log2(amax)) - 8.0).clamp(-127, 128)
    e8m0 = (scale_exp + 127.0).to(torch.int32).to(SCALE_DTYPE)
    q = (xb * torch.exp2(-scale_exp).unsqueeze(-1)).to(FP8_DTYPE)
    return q.reshape(orig_shape), e8m0


def mxfp8_dequantize(q: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    """Reconstruct an fp32 tensor from FP8 values + E8M0 (uint8) block scales."""
    dim, k = q.shape
    scale = torch.exp2(e8m0.to(torch.float32) - 127.0)  # [dim, k//32]
    qf = q.to(torch.float32).reshape(dim, k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    return (qf * scale.unsqueeze(-1)).reshape(dim, k)


def pack_scales_iter_major(scale_raw: torch.Tensor, k_iters: int) -> torch.Tensor:
    """Pack [dim, K//32] uint8 E8M0 scales into iteration-major [k_iters, dim] uint32.

    Each iteration covers BLOCK_K=128 K-elements = 4 blocks of 32, so the 4
    consecutive uint8 block scales are packed little-endian into one uint32.
    """
    dim, scale_k = scale_raw.shape
    assert scale_k == k_iters * 4, (scale_k, k_iters)
    s = scale_raw.to(torch.int64).reshape(dim, k_iters, 4)
    shift = torch.arange(4, device=s.device, dtype=torch.int64) * 8
    packed = (s << shift).sum(dim=-1)  # [dim, k_iters]
    return packed.t().contiguous().to(torch.int32)  # [k_iters, dim]


def make_runner(M, N, K, x_dtype):
    assert K % BLOCK_K == 0
    device = "cuda"
    k_iters = K // BLOCK_K

    x = torch.randn(M, K, dtype=x_dtype, device=device)
    w = torch.randn(N, K, dtype=x_dtype, device=device)
    x_q, x_scale = mxfp8_e4m3_quantize(x)  # [M,K] fp8, [M,K//32] u8
    w_q, w_scale = mxfp8_e4m3_quantize(w)  # [N,K] fp8, [N,K//32] u8
    x_q, w_q = x_q.contiguous(), w_q.contiguous()

    # Iteration-major packed scales [k_iters, dim] uint32, repacked on-device
    # into the MFMA layout by the repack launcher.
    sa_iter = pack_scales_iter_major(x_scale.contiguous(), k_iters)  # [k_iters, M]
    sb_iter = pack_scales_iter_major(w_scale.contiguous(), k_iters)  # [k_iters, N]
    sa_mfma = torch.empty_like(sa_iter)
    sb_mfma = torch.empty_like(sb_iter)

    c_out = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    launch_gemm, launch_repack = compile_mxfp8_gemm_4w(M=M, N=N, K=K)
    stream = torch.cuda.current_stream()

    def _gemm_args():
        return (
            x_q.view(-1),
            w_q.view(-1),
            c_out.view(-1),
            sa_mfma,
            sb_mfma,
            stream,
        )

    def _repack_a_args():
        return (sa_iter, sa_mfma, M, k_iters, stream)

    def _repack_b_args():
        return (sb_iter, sb_mfma, N, k_iters, stream)

    repack_a = flyc.compile(launch_repack, *_repack_a_args())
    repack_b = flyc.compile(launch_repack, *_repack_b_args())
    gemm = flyc.compile(launch_gemm, *_gemm_args())

    def run():
        repack_a(*_repack_a_args())
        repack_b(*_repack_b_args())
        gemm(*_gemm_args())

    # Reference from the dequantized fp8 inputs (matches what the kernel sees).
    x_deq = mxfp8_dequantize(x_q, x_scale)  # [M, K]
    w_deq = mxfp8_dequantize(w_q, w_scale)  # [N, K]
    c_ref = (x_deq @ w_deq.t()).to(torch.float32)

    # Verify accuracy before handing back the benchmark closure.
    run()
    torch.cuda.synchronize()
    assert verify_output(c_out.to(torch.float32), c_ref, rtol=0.1, atol=0.1)

    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-M", type=int, default=8192)
    ap.add_argument("-N", type=int, default=8192)
    ap.add_argument("-K", type=int, default=8192)
    args = ap.parse_args()

    M, N, K = args.M, args.N, args.K

    run = make_runner(M, N, K, torch.bfloat16)

    us = triton.testing.do_bench(run) * 1e3
    tflops = 2 * M * N * K / (us * 1e-6) / 1e12

    print(f"{'shape (MxNxK)':>18} {'dtype':>10} {'latency(us)':>12} {'TFLOPS':>9}")
    print(f"{f'{M}x{N}x{K}':>18} {args.dtype:>10} {us:>12.2f} {tflops:>9.2f}")


if __name__ == "__main__":
    main()
