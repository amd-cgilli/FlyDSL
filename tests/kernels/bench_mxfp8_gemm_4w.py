# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch
import triton
import triton.language as tl

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


def triton_mxfp8_reference(x_q, x_scale, w_q, w_scale, M, N, K):
    """Reference output via the Triton ``tl.dot_scaled`` MXFP8 GEMM, fed the same
    quantized fp8 values + E8M0 scales the FlyDSL kernel consumes."""
    x_q, x_scale = x_q.contiguous(), x_scale.contiguous()
    w_q, w_scale = w_q.contiguous(), w_scale.contiguous()
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x_q.device)

    if M >= 1024:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 256, 8, 2
    else:
        BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 2

    BLOCK_K = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _mxfp8_linear_kernel[grid](
        x_q, x_scale, w_q, w_scale, out, M, N, K,
        x_q.stride(0), x_q.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        w_q.stride(0), w_q.stride(1),
        w_scale.stride(0), w_scale.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out


def mxfp8_e4m3_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` [..., K] to FP8 E4M3 values + E8M0 (uint8) block scales.

    Matches HipKittens ``compute_e8m0_scale``: the stored exponent is
    ``clamp(floor(log2(amax)) + 127, 0, 254)`` so each block's max lands in
    ``[1, 2)`` and never reaches the E4M3 NaN cliff at 448.
    """
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    orig_shape = x.shape
    xb = x.reshape(*orig_shape[:-1], orig_shape[-1] // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    amax = xb.abs().amax(dim=-1).to(torch.float32)
    # floor(log2(0)) == -inf; the clamp below maps that to a stored exponent of 0.
    e8m0 = (torch.floor(torch.log2(amax)) + 127.0).clamp(0, 254).to(torch.int32).to(SCALE_DTYPE)
    inv = torch.exp2(127.0 - e8m0.to(torch.float32))  # ldexp(1, 127 - s)
    q = (xb * inv.unsqueeze(-1)).to(FP8_DTYPE)
    return q.reshape(orig_shape), e8m0


def mxfp8_dequantize(q: torch.Tensor, e8m0: torch.Tensor) -> torch.Tensor:
    """Reconstruct an fp32 tensor from FP8 values + E8M0 (uint8) block scales."""
    dim, k = q.shape
    scale = torch.exp2(e8m0.to(torch.float32) - 127.0)  # [dim, k//32]
    qf = q.to(torch.float32).reshape(dim, k // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    return (qf * scale.unsqueeze(-1)).reshape(dim, k)


def make_runner(M, N, K, x_dtype):
    assert K % BLOCK_K == 0
    device = "cuda"

    # HipKittens uses normal(mean=0, std=0.5) for both operands.
    x = torch.randn(M, K, dtype=x_dtype, device=device) * 0.5
    w = torch.randn(N, K, dtype=x_dtype, device=device) * 0.5
    x_q, x_scale = mxfp8_e4m3_quantize(x)  # [M,K] fp8, [M,K//32] u8
    w_q, w_scale = mxfp8_e4m3_quantize(w)  # [N,K] fp8, [N,K//32] u8
    x_q, w_q = x_q.contiguous(), w_q.contiguous()

    # Unpacked E8M0 (uint8) block scales [dim, K//32] consumed directly by the
    # GEMM; the kernel now gathers and packs the 4 MFMA scale bytes on-device, so
    # no pre-pack pass and no row padding is needed.
    sa = x_scale.contiguous()  # [M, K//32]
    sb = w_scale.contiguous()  # [N, K//32]

    c_out = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    launch_gemm = compile_mxfp8_gemm_4w(K=K)
    stream = torch.cuda.current_stream()

    def _gemm_args():
        return (
            x_q.view(-1),
            w_q.view(-1),
            c_out.view(-1),
            sa,
            sb,
            M, N,
            stream,
        )

    gemm = flyc.compile(launch_gemm, *_gemm_args())

    def run():
        gemm(*_gemm_args())

    # Reference from the Triton tl.dot_scaled kernel on the *same* quantized
    # fp8 inputs + E8M0 scales the GEMM consumes.
    c_ref = triton_mxfp8_reference(x_q, x_scale, w_q, w_scale, M, N, K)

    def run_triton():
        triton_mxfp8_reference(x_q, x_scale, w_q, w_scale, M, N, K)

    # Verify accuracy before handing back the benchmark closure.
    run()
    torch.cuda.synchronize()
    assert verify_output(c_out.to(torch.float32), c_ref.to(torch.float32), atol=0.1, rtol=0.1)

    return run, run_triton


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


if __name__ == "__main__":
    header = (
        f"{'shape (MxNxK)':>18} {'dtype':>10} {'latency FlyDSL(us)':>20} {'TFLOPS FlyDSL':>15}"
        f" {'latency Triton(us)':>20} {'TFLOPS Triton':>15}"
    )
    print(header)

    failed = []
    for M, N, K in SHAPES:
        try:
            run, run_triton = make_runner(M, N, K, torch.bfloat16)
        except AssertionError:
            failed.append((M, N, K))
            continue

        flops = 2 * M * N * K

        us = triton.testing.do_bench(run) * 1e3
        tflops = flops / (us * 1e-6) / 1e12

        us_tt = triton.testing.do_bench(run_triton) * 1e3
        tflops_tt = flops / (us_tt * 1e-6) / 1e12

        print(
            f"{f'{M}x{N}x{K}':>18} {'bfloat16':>10} {us:>20.2f} {tflops:>15.2f}"
            f" {us_tt:>20.2f} {tflops_tt:>15.2f}"
        )

    if failed:
        print(f"\n{len(failed)} shape(s) failed the correctness check (skipped above):")
        for M, N, K in failed:
            print(f"  {M}x{N}x{K}")
