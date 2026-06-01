# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""BF16 GEMM correctness + perf harness.

Computes ``C = A @ B_T.T`` with bf16 A/B inputs, f32 output, and a k-major
(row-major, K contiguous) layout for both operands. Kernel implementation
lives in ``kernels/bf16_gemm.py``.
"""

import os
import sys

import pytest
import torch

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.bf16_gemm_256_256_64_32x16 import compile_bf16_gemm_32x16  # noqa: E402
from kernels.bf16_gemm_256_256_64_16x32 import compile_bf16_gemm_16x32  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


DEFAULT_BENCH_ITERS = 100
DEFAULT_BENCH_WARMUP = 20


def _run_torch(a, b_t, dtype=torch.float32):
    """Reference GEMM: C = A @ B_T.T accumulated in f32."""
    c = torch.mm(a.to(torch.float32), b_t.to(torch.float32).T)
    return c.to(dtype)


def _bench_bf16_gemm(
    M: int,
    N: int,
    K: int,
    *,
    num_warmups: int = DEFAULT_BENCH_WARMUP,
    num_iters: int = DEFAULT_BENCH_ITERS,
    use_32x16: bool = False,
):
    device = torch.device("cuda")

    # k-major (row-major, K contiguous) bf16 operands.
    a = torch.rand(M, K, device=device, dtype=torch.float32).uniform_(-1, 1).to(torch.bfloat16)
    b_t = torch.rand(N, K, device=device, dtype=torch.float32).uniform_(-1, 1).to(torch.bfloat16)
    c_out = torch.empty((M, N), device=device, dtype=torch.bfloat16)

    a = a.contiguous()
    b_t = b_t.contiguous()

    c_ref_f32 = _run_torch(a, b_t)

    if use_32x16:
        launch_fn = compile_bf16_gemm_32x16(K=K)
        print(f"\n[bf16_gemm 32x16] M={M} N={N} K={K}")
    else:
        launch_fn = compile_bf16_gemm_16x32(K=K)
        print(f"\n[bf16_gemm 16x32] M={M} N={N} K={K}")

    def _args(c, a_, b_):
        return (
            a_.contiguous().view(-1),
            b_.contiguous().view(-1),
            c.contiguous().view(-1),
            M,
            N,
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out, a, b_t))

    def _launch(c, a_, b_):
        compiled(*_args(c, a_, b_))

    num_iters = max(2, int(num_iters))
    _, us = run_perftest(
        _launch,
        c_out,
        a,
        b_t,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()

    assert verify_output(c_out.to(torch.float32), c_ref_f32, rtol=0.1, atol=0.1)

    flops = 2 * M * N * K
    bytes_moved = (M * K * 2) + (N * K * 2) + (M * N * 4)
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s")

    return tflops


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BF16 GEMM benchmark")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("--num_iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--num_warmups", type=int, default=DEFAULT_BENCH_WARMUP)
    parser.add_argument("--use_32x16", action="store_true")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    # for s in [1024, 2048, 4096, 8192, 16384]:
    #     _bench_bf16_gemm(s, s, s, num_warmups=DEFAULT_BENCH_WARMUP, num_iters=DEFAULT_BENCH_ITERS)

    try:
        _bench_bf16_gemm(
            M=args.M,
            N=args.N,
            K=args.K,
            num_warmups=args.num_warmups,
            num_iters=args.num_iters,
            use_32x16=args.use_32x16
        )
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
