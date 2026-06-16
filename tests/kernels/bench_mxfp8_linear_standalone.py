# SPDX-License-Identifier: Apache-2.0
"""Benchmark ONLY the MXFP8 ``tl.dot_scaled`` Triton GEMM kernel.

Times the GEMM in isolation (inputs pre-quantized once, outside the timed
region) and reports shape, dtype, latency, and TFLOPS. Targets AMD CDNA4
(gfx950) native microscaling. Output goes to stdout as a table, or to CSV
with ``--csv <path>``.
"""

import argparse
import csv
import sys

import torch
import triton
import triton.language as tl

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

MXFP8_BLOCK_SIZE = 32
FP8_DTYPE = torch.float8_e4m3fn
SCALE_DTYPE = torch.uint8

# (m, n, k) shapes to sweep.
SHAPES = [
    (3, 6144, 2048),
    (3, 2560, 6144),
    (3, 1536, 6144),
    (3, 6144, 768),
    (2, 6144, 2048),
    (2, 2560, 6144),
    (2, 1536, 6144),
    (2, 6144, 768),
    (4, 6144, 2048),
    (1, 6144, 2048),
    (4, 2560, 6144),
    (4, 1536, 6144),
    (4, 6144, 768),
    (1, 2560, 6144),
    (1, 1536, 6144),
    (1, 6144, 768),
    (1024, 6144, 2048),
    (2268, 6144, 2048),
    (357, 6144, 2048),
    (137, 6144, 2048),
    (164, 6144, 2048),
    (117, 6144, 2048),
    (8192, 2560, 6144),
    (8192, 1536, 6144),
    (8192, 6144, 768),
    (1024, 2560, 6144),
    (1024, 1536, 6144),
    (1024, 6144, 768),
    (2268, 2560, 6144),
    (2268, 1536, 6144),
    (2268, 6144, 768),
    (357, 2560, 6144),
    (357, 1536, 6144),
    (357, 6144, 768),
    (137, 2560, 6144),
    (137, 1536, 6144),
    (137, 6144, 768),
    (164, 2560, 6144),
    (164, 1536, 6144),
    (164, 6144, 768),
    (117, 2560, 6144),
    (117, 1536, 6144),
    (117, 6144, 768),
    (3, 2304, 6144),
    (3, 6144, 6144),
    (3, 6144, 3072),
    (8192, 2304, 6144),
    (8192, 6144, 6144),
    (8192, 6144, 3072),
    (1024, 2304, 6144),
    (1024, 6144, 6144),
    (1024, 6144, 3072),
]


@triton.jit
def _mxfp8_linear_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_xsm,
    stride_xsk,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_sk = tl.arange(0, BLOCK_K // 32)
    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    xs_ptrs = xs_ptr + offs_m[:, None] * stride_xsm + offs_sk[None, :] * stride_xsk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    ws_ptrs = ws_ptr + offs_n[:, None] * stride_wsn + offs_sk[None, :] * stride_wsk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
        xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
        ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3")
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
        xs_ptrs += (BLOCK_K // 32) * stride_xsk
        ws_ptrs += (BLOCK_K // 32) * stride_wsk

    o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :]
    )


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


def make_kernel_runner(m, n, k, x_dtype, device):
    """Pre-quantize inputs and return a zero-arg closure that launches the GEMM."""
    x = torch.randn(m, k, dtype=x_dtype, device=device)
    w = torch.randn(n, k, dtype=x_dtype, device=device)
    x_q, x_scale = mxfp8_e4m3_quantize(x)
    w_q, w_scale = mxfp8_e4m3_quantize(w)
    w_q, w_scale = w_q.contiguous(), w_scale.contiguous()
    out = torch.empty((m, n), dtype=x_dtype, device=device)

    if m >= 1024:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 256, 8, 2
    else:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 2
    BLOCK_K = 128
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(n, BLOCK_N))

    def run():
        _mxfp8_linear_kernel[grid](
            x_q, x_scale, w_q, w_scale, out, m, n, k,
            x_q.stride(0), x_q.stride(1),
            x_scale.stride(0), x_scale.stride(1),
            w_q.stride(0), w_q.stride(1),
            w_scale.stride(0), w_scale.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=num_warps, num_stages=num_stages,
        )

    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16"], help="activation/weight dtype")
    ap.add_argument("--csv", type=str, default=None,
                    help="write results to this CSV path instead of the stdout table")
    args = ap.parse_args()

    x_dtype = getattr(torch, args.dtype)
    device = "cuda"
    rows = []
    for m, n, k in SHAPES:
        run = make_kernel_runner(m, n, k, x_dtype, device)
        ms = triton.testing.do_bench(run)
        us = ms * 1e3
        tflops = 2 * m * n * k / (ms * 1e-3) / 1e12
        rows.append((f"{m}x{n}x{k}", args.dtype, us, tflops))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["shape_mxnxk", "dtype", "latency_us", "tflops"])
            wr.writerows(rows)
        print(f"wrote {len(rows)} rows to {args.csv}", file=sys.stderr)
    else:
        print(f"{'shape (MxNxK)':>18} {'dtype':>10} {'latency(us)':>12} {'TFLOPS':>9}")
        for shape, dtype, us, tflops in rows:
            print(f"{shape:>18} {dtype:>10} {us:>12.2f} {tflops:>9.2f}")


if __name__ == "__main__":
    main()
