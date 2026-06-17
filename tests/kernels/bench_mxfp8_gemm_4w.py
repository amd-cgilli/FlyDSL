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


def preshuffle_scales(scale: torch.Tensor, group: int) -> torch.Tensor:
    """Preshuffle [dim, K//32] uint8 E8M0 scales into the packed uint32 layout the
    GEMM loads with a single int32 read: [ceil(dim/group), 16, K//32] uint32.

    ``group`` is the number of M/N rows one wave-fragment covers (= BLOCK//4 =
    N_TILES*16). The kernel reads a fragment whose first global row is a multiple
    of ``group`` as one uint32 holding the N_TILES sub-row scales
    {frag_base + g*16 + r16 : g in 0..N_TILES-1} little-endian, with byte g
    selected by the scaled-MFMA opsel. Bytes beyond N_TILES (when N_TILES < 4)
    are never read; they and partial-group padding rows hold the E8M0 identity
    0x7F (= 2**0 = 1.0) so every load is a full, in-bounds int32.
    """
    assert group % 16 == 0
    n_tiles = group // 16
    dim, scale_k = scale.shape
    groups = (dim + group - 1) // group
    padded_dim = groups * group
    if padded_dim != dim:
        pad = torch.full(
            (padded_dim - dim, scale_k), 0x7F, dtype=scale.dtype, device=scale.device
        )
        scale = torch.cat([scale, pad], dim=0)
    s = scale.reshape(groups, n_tiles, 16, scale_k)  # [grp, g, r16, col]
    s = s.permute(0, 2, 3, 1).contiguous()  # [grp, r16, col, g]
    if n_tiles < 4:  # pad each word out to a full uint32 with identity bytes
        wpad = torch.full(
            (groups, 16, scale_k, 4 - n_tiles), 0x7F, dtype=s.dtype, device=s.device
        )
        s = torch.cat([s, wpad], dim=-1).contiguous()
    return s.view(torch.int32).reshape(groups, 16, scale_k)  # little-endian uint32


# Block sizes to autotune over for both M and N.
BLOCK_CHOICES = [64, 128, 256]
# Set to see why each (block_m, block_n) config is skipped during the sweep.
VERBOSE_AUTOTUNE = bool(int(os.environ.get("VERBOSE_AUTOTUNE", "0")))


def config_name(block_m, block_n):
    return f"4w_{block_m}x{block_n}x{BLOCK_K}"


def _make_config_runner(block_m, block_n, K, x_q, w_q, x_scale, w_scale, M, N):
    """Build the run closure for one (block_m, block_n) config. Each config needs
    its own group-specific preshuffled scales, compiled kernel, and output
    buffer. Returns (run, c_out)."""
    device = x_q.device
    # Preshuffled E8M0 scales [ceil(dim/group), 16, K//32] uint32, packed host-side
    # so the GEMM loads each scale word with a single padded int32 read. The wave
    # fragment covers BLOCK//4 rows, which is the scale-pack group size.
    sa = preshuffle_scales(x_scale.contiguous(), block_m // 4)  # [ceil(M/(BM//4)), 16, K//32]
    sb = preshuffle_scales(w_scale.contiguous(), block_n // 4)  # [ceil(N/(BN//4)), 16, K//32]

    c_out = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    stream = torch.cuda.current_stream()

    def _gemm_args():
        return (
            x_q.view(-1),
            w_q.view(-1),
            c_out.view(-1),
            sa.reshape(-1),
            sb.reshape(-1),
            M, N,
            stream,
        )

    launch_gemm = compile_mxfp8_gemm_4w(K=K, BLOCK_M=block_m, BLOCK_N=block_n)
    gemm = flyc.compile(launch_gemm, *_gemm_args())

    def run():
        gemm(*_gemm_args())

    return run, c_out


def make_runner(M, N, K, x_dtype):
    """Autotune over all (block_m, block_n) configs: verify each against the
    Triton reference, benchmark the correct ones, and return
    (best_run, run_triton, best_config_name) for the fastest passing config."""
    assert K % BLOCK_K == 0
    device = "cuda"

    # HipKittens uses normal(mean=0, std=0.5) for both operands.
    x = torch.randn(M, K, dtype=x_dtype, device=device) * 0.5
    w = torch.randn(N, K, dtype=x_dtype, device=device) * 0.5
    x_q, x_scale = mxfp8_e4m3_quantize(x)  # [M,K] fp8, [M,K//32] u8
    w_q, w_scale = mxfp8_e4m3_quantize(w)  # [N,K] fp8, [N,K//32] u8
    x_q, w_q = x_q.contiguous(), w_q.contiguous()

    # Reference from the Triton tl.dot_scaled kernel on the *same* quantized
    # fp8 inputs + E8M0 scales the GEMM consumes.
    c_ref = triton_mxfp8_reference(x_q, x_scale, w_q, w_scale, M, N, K).to(torch.float32)

    def run_triton():
        triton_mxfp8_reference(x_q, x_scale, w_q, w_scale, M, N, K)

    best_run = None
    best_name = None
    best_us = float("inf")
    for block_m in BLOCK_CHOICES:
        for block_n in BLOCK_CHOICES:
            name = config_name(block_m, block_n)
            try:
                run, c_out = _make_config_runner(
                    block_m, block_n, K, x_q, w_q, x_scale, w_scale, M, N
                )
                run()
                torch.cuda.synchronize()
            except Exception as e:
                if VERBOSE_AUTOTUNE:
                    print(f"    [{name}] compile/run error: {e}")
                continue
            if not verify_output(c_out.to(torch.float32), c_ref, atol=0.1, rtol=0.1):
                if VERBOSE_AUTOTUNE:
                    print(f"    [{name}] failed correctness check")
                continue
            us = triton.testing.do_bench(run) * 1e3
            if VERBOSE_AUTOTUNE:
                print(f"    [{name}] ok, {us:.2f} us")
            if us < best_us:
                best_us, best_run, best_name = us, run, name

    if best_run is None:
        raise AssertionError("no config passed the correctness check")

    return best_run, run_triton, best_name


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
        f"{'shape (MxNxK)':>18} {'best config':>16} {'latency FlyDSL(us)':>20} {'TFLOPS FlyDSL':>15}"
        f" {'latency Triton(us)':>20} {'TFLOPS Triton':>15} {'Speedup':>10}"
    )
    print(header)

    failed = []
    for M, N, K in SHAPES:
        try:
            run, run_triton, best_config = make_runner(M, N, K, torch.bfloat16)
        except AssertionError:
            failed.append((M, N, K))
            continue

        flops = 2 * M * N * K

        us = triton.testing.do_bench(run) * 1e3
        tflops = flops / (us * 1e-6) / 1e12

        us_tt = triton.testing.do_bench(run_triton) * 1e3
        tflops_tt = flops / (us_tt * 1e-6) / 1e12

        speedup = us_tt / us
        print(
            f"{f'{M}x{N}x{K}':>18} {best_config:>16} {us:>20.2f} {tflops:>15.2f}"
            f" {us_tt:>20.2f} {tflops_tt:>15.2f} {speedup:>10.2f}x"
        )

    if failed:
        print(f"\n{len(failed)} shape(s) failed the correctness check (skipped above):")
        for M, N, K in failed:
            print(f"  {M}x{N}x{K}")
