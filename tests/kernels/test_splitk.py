import os
import sys

import pytest
import torch

import flydsl.compiler as flyc

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from flydsl.runtime.device import get_rocm_arch
from kernels.fp8_gemm_4wave import compile_fp8_gemm_4w
from kernels.fp8_gemm_8wave import compile_fp8_gemm_8w
from tests.test_common import run_perftest, verify_output
from tests.utils import pertoken_quant

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16
ARCH = str(get_rocm_arch())

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)


def _run_torch(a, b, scale_a, scale_b, dtype=torch.float32):
    if scale_a is not None and scale_b is not None:
        a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
        b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    else:
        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)
    c = torch.mm(a_f32, b_f32.T)
    return c.to(dtype)


def test_fp8_gemm_4wave(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    *,
    num_splits: int = 1,
    disable_xcd_remap: bool = False,
    num_warmups: int = 2,
    num_iters: int = 10,
):
    if "gfx95" not in ARCH:
        pytest.skip("FP8 GEMM requires CDNA4")

    size_c = M * N
    size_a = M * K
    size_b = N * K

    device = torch.device("cuda")
    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)
    c_out_raw = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)
    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=FP8_DTYPE)
    b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=FP8_DTYPE)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    scale_a = scale_a.squeeze().contiguous()
    scale_b = scale_b.squeeze().contiguous()

    c_ref = _run_torch(a_q, b_q, scale_a, scale_b)

    IS_SPLIT_K = num_splits > 1
    M_PAD = ((M + tile_m - 1) // tile_m) * tile_m
    if IS_SPLIT_K:
        c_workspace = torch.zeros((num_splits, M_PAD, N), dtype=torch.float32, device=device)
    else:
        c_workspace = torch.empty(0, dtype=torch.float32, device=device)

    launch_gemm_fn, launch_reduce_fn = compile_fp8_gemm_4w(
        M=M, N=N, K=K,
        BLOCK_M=tile_m,
        BLOCK_N=tile_n,
        n_splits=num_splits,
        use_xcd_remap=not disable_xcd_remap)
    print(f"✓ Kernel prepared (M={M} N={N} K={K} BLOCK_M={tile_m} BLOCK_N={tile_n} "
          f"NUM_SPLITS={num_splits} disable_xcd_remap={disable_xcd_remap})")

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    stream = torch.cuda.current_stream()

    def _gemm_args(c, ws, a, b, sa, sb):
        return (
            _as_i8(a).contiguous().view(-1),
            _as_i8(b).contiguous().view(-1),
            c.contiguous().view(-1),
            ws.contiguous().view(-1),
            sa.contiguous().view(-1),
            sb.contiguous().view(-1),
            stream,
        )

    def _reduce_args(ws, c):
        return (
            ws.contiguous().view(-1),
            c.contiguous().view(-1),
            stream,
        )

    compiled_gemm = flyc.compile(launch_gemm_fn, *_gemm_args(c_out_raw, c_workspace, a_q, b_q, scale_a, scale_b))
    if IS_SPLIT_K:
        compiled_reduce = flyc.compile(launch_reduce_fn, *_reduce_args(c_workspace, c_out_raw))

    def _launch(c, ws, a, b, sa, sb):
        compiled_gemm(*_gemm_args(c, ws, a, b, sa, sb))
        if IS_SPLIT_K:
            compiled_reduce(*_reduce_args(ws, c))

    num_iters = max(2, int(num_iters))

    _, us = run_perftest(
        _launch,
        c_out_raw,
        c_workspace,
        a_q,
        b_q,
        scale_a,
        scale_b,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()
    c_out_f32 = c_out_raw.to(torch.float32)
    assert verify_output(c_out_f32, c_ref, rtol=0.1, atol=0.1)

    # A/B fp8, C bf16, scales fp32
    bytes_moved = size_a + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s\n")

    return tflops


def test_fp8_gemm_8wave(
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    *,
    num_splits: int = 1,
    num_warmups: int = 2,
    num_iters: int = 10,
):
    if "gfx95" not in ARCH:
        pytest.skip("FP8 GEMM requires CDNA4")

    size_c = M * N
    size_a = M * K
    size_b = N * K

    device = torch.device("cuda")
    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)
    c_out_raw = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)
    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=FP8_DTYPE)
    b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=FP8_DTYPE)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    scale_a = scale_a.squeeze().contiguous()
    scale_b = scale_b.squeeze().contiguous()

    c_ref = _run_torch(a_q, b_q, scale_a, scale_b)

    IS_SPLIT_K = num_splits > 1
    M_PAD = ((M + tile_m - 1) // tile_m) * tile_m
    if IS_SPLIT_K:
        c_workspace = torch.zeros((num_splits, M_PAD, N), dtype=torch.float32, device=device)
    else:
        c_workspace = torch.empty(0, dtype=torch.float32, device=device)

    launch_gemm_fn, launch_reduce_fn = compile_fp8_gemm_8w(
        M=M, N=N, K=K,
        BLOCK_M=tile_m,
        BLOCK_N=tile_n,
        n_splits=num_splits)
    print(f"✓ 8wave kernel prepared (M={M} N={N} K={K} BLOCK_M={tile_m} BLOCK_N={tile_n} "
          f"NUM_SPLITS={num_splits})")

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    stream = torch.cuda.current_stream()

    def _gemm_args(c, ws, a, b, sa, sb):
        return (
            _as_i8(a).contiguous().view(-1),
            _as_i8(b).contiguous().view(-1),
            c.contiguous().view(-1),
            ws.contiguous().view(-1),
            sa.contiguous().view(-1),
            sb.contiguous().view(-1),
            stream,
        )

    def _reduce_args(ws, c):
        return (
            ws.contiguous().view(-1),
            c.contiguous().view(-1),
            stream,
        )

    compiled_gemm = flyc.compile(launch_gemm_fn, *_gemm_args(c_out_raw, c_workspace, a_q, b_q, scale_a, scale_b))
    if IS_SPLIT_K:
        compiled_reduce = flyc.compile(launch_reduce_fn, *_reduce_args(c_workspace, c_out_raw))

    def _launch(c, ws, a, b, sa, sb):
        compiled_gemm(*_gemm_args(c, ws, a, b, sa, sb))
        if IS_SPLIT_K:
            compiled_reduce(*_reduce_args(ws, c))

    num_iters = max(2, int(num_iters))

    _, us = run_perftest(
        _launch,
        c_out_raw,
        c_workspace,
        a_q,
        b_q,
        scale_a,
        scale_b,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )
    torch.cuda.synchronize()
    c_out_f32 = c_out_raw.to(torch.float32)
    assert verify_output(c_out_f32, c_ref, rtol=0.1, atol=0.1)

    bytes_moved = size_a + size_b + size_c * 2 + (M + N) * 4
    flops = 2 * M * N * K
    tflops = flops / (us / 1e6) / 1e12
    tbps = bytes_moved / 1e12 / (us / 1e6)
    print(f"[flyc] Throughput: {us:.1f} us, {tflops:.2f} TFLOPS, BW: {tbps:.3f} TB/s\n")

    return tflops


DS_SMALL_SHAPES_4W = [
    (1, 256, 7168),
    (1, 2112, 7168),
    (1, 3072, 1536),
    (1, 4096, 512),
    (1, 6144, 1536),
    (1, 7168, 4096),
]

DS_SHAPES_8W = [
    (1, 2048, 7168),
    (1, 3072, 7168),
    (1, 4096, 4096),
    (1, 7168, 4096),
]

BLOCK_K = 128


def _valid_splits(K):
    k_iters = K // BLOCK_K
    return [s for s in range(1, k_iters + 1)
            if k_iters % s == 0 and k_iters // s >= 2]


if __name__ == "__main__":
    torch.set_default_device("cuda")

    print("=== 4-wave split-k ===")
    for m, n, k in DS_SMALL_SHAPES_4W:
        bm = bn = 64
        splits = _valid_splits(k)
        print(f"--- M={m} N={n} K={k} valid splits: {splits} ---")
        for ns in splits:
            test_fp8_gemm_4wave(
                M=m, N=n, K=k,
                tile_m=bm, tile_n=bn,
                num_splits=ns,
                num_iters=100,
                num_warmups=10
            )

    print("\n=== 8-wave split-k ===")
    for m, n, k in DS_SHAPES_8W:
        bm, bn = 128, 256
        splits = _valid_splits(k)
        print(f"--- M={m} N={n} K={k} valid splits: {splits} ---")
        for ns in splits:
            test_fp8_gemm_8wave(
                M=m, N=n, K=k,
                tile_m=bm, tile_n=bn,
                num_splits=ns,
                num_iters=100,
                num_warmups=10
            )
