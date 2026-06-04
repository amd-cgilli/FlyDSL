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
import torch.nn.functional as F

import flydsl.compiler as flyc

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels.bf16_gemm_256_256_64_32x16 import compile_bf16_gemm_32x16  # noqa: E402
from kernels.bf16_gemm_256_256_64_16x32 import compile_bf16_gemm_16x32  # noqa: E402
from kernels.bf16_gemm_256_256_64_16x32_4w import compile_bf16_gemm_16x32_4w  # noqa: E402
from tests.test_common import run_perftest, verify_output  # noqa: E402

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


DEFAULT_BENCH_ITERS = 200
DEFAULT_BENCH_WARMUP = 100


def _run_torch(a, b_t, dtype=torch.float32):
    """Reference GEMM: C = A @ B_T.T accumulated in f32."""
    c = torch.mm(a.to(torch.float32), b_t.to(torch.float32).T)
    return c.to(dtype)

def _bench_torch(a, b, num_warmups, num_iters) -> float:
    m, _ = a.shape
    n, _ = b.shape
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

    def _launch(_out):
        F.linear(a, b, out=_out)

    _, us = run_perftest(_launch, out, num_iters=num_iters, num_warmup=num_warmups)
    torch.cuda.synchronize()
    return us

def tflops(m, n, k, us):
    flops = 2 * m * n * k
    return flops / (us / 1e6) / 1e12

def _bench_bf16_gemm(
    M: int,
    N: int,
    K: int,
    *,
    num_warmups: int = DEFAULT_BENCH_WARMUP,
    num_iters: int = DEFAULT_BENCH_ITERS,
    use_32x16: bool = False,
    use_4wave: bool = False,
    vs_torch: bool = False,
):
    device = torch.device("cuda")

    # k-major (row-major, K contiguous) bf16 operands.
    a = torch.rand(M, K, device=device, dtype=torch.float32).uniform_(-1, 1).to(torch.bfloat16)
    b_t = torch.rand(N, K, device=device, dtype=torch.float32).uniform_(-1, 1).to(torch.bfloat16)
    c_out = torch.empty((M, N), device=device, dtype=torch.bfloat16)

    a = a.contiguous()
    b_t = b_t.contiguous()

    c_ref_f32 = _run_torch(a, b_t)

    if use_4wave:
        launch_fn = compile_bf16_gemm_16x32_4w(K=K)
        print(f'\n[bf16_gemm 16x32 4Wave] M={M} N={N} K={K}')
    else:
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
    _, us_fly = run_perftest(
        _launch,
        c_out,
        a,
        b_t,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()

    assert verify_output(c_out.to(torch.float32), c_ref_f32, rtol=0.1, atol=0.1)

    tflops_fly = tflops(M, N, K, us_fly)
    print(f"[FLYDSL] Throughput: {us_fly:.1f} us, {tflops_fly:.2f} TFLOPS")

    if vs_torch:
        us_torch = _bench_torch(a, b_t, num_warmups=num_warmups, num_iters=num_iters)
        tflops_torch = tflops(M, N, K, us_torch)
        print(f"[PyTorch] Throughput: {us_torch:.1f} us, {tflops_torch:.2f} TFLOPS")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BF16 GEMM benchmark")
    parser.add_argument("-M", type=int, default=4096)
    parser.add_argument("-N", type=int, default=4096)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("--num_iters", type=int, default=DEFAULT_BENCH_ITERS)
    parser.add_argument("--num_warmups", type=int, default=DEFAULT_BENCH_WARMUP)
    parser.add_argument("--use_32x16", action="store_true")
    parser.add_argument("--use_4w", action="store_true")
    parser.add_argument("--vs_torch", action="store_true")
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
            use_32x16=args.use_32x16,
            vs_torch=args.vs_torch,
            use_4wave=args.use_4w
        )
    except pytest.skip.Exception as e:
        print(f"Skipped: {e}")
