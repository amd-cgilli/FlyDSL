import os
import sys

import torch
import torch.nn.functional as F
import flydsl.compiler as flyc

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLYDSL_SRC not in sys.path:
    sys.path.insert(0, _PYFLYDSL_SRC)

from kernels.fp8_gemm_4wave import compile_fp8_gemm_256x256x128
from tests.kernels.benchmark_common import bench_kernel_us
from tests.test_common import verify_output
from tests.utils import pertoken_quant

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16


def run_torch(a, b, scale_a, scale_b, dtype=torch.float32):
    if scale_a is not None and scale_b is not None:
        a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
        b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    else:
        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)
    c = torch.mm(a_f32, b_f32.T)
    return c.to(dtype)


def check_gemm(M: int, N: int, K: int):
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

    out_ref = run_torch(a_q, b_q, scale_a, scale_b)

    launch_fn = compile_fp8_gemm_256x256x128(M=M, N=N, K=K)
    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def _args(c):
        return (
            _as_i8(a_q).contiguous().view(-1),
            _as_i8(b_q).contiguous().view(-1),
            c.contiguous().view(-1),
            scale_a.contiguous(),
            scale_b.contiguous(),
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out_raw))

    compiled(*_args(c_out_raw))
    torch.cuda.synchronize()

    # If this fails the entire benchmark fails
    assert verify_output(c_out_raw.to(torch.float32), out_ref, rtol=0.1, atol=0.1)

    median_us = bench_kernel_us(lambda: compiled(*_args(c_out_raw)), warmup=2, iters=101, report_average=True)
    tflops = (2 * M * N * K) * 1e-12 / (median_us * 1e-6)

    return tflops



def bench_gemm(M: int, N: int, K: int) -> tuple[float, tuple[int, int, int]]:
    tile_M, tile_N, tile_K = 256, 256, 128
    # For now only have 256x256x128 as a valid tile size so if M,N,K are not multiples of that we
    # don't benchmark anything

    if M % tile_M != 0 or N % tile_N != 0 or K % tile_K != 0:
        return None

    tflops = check_gemm(M, N, K)

    return tflops, (tile_M, tile_N, tile_K)


def get_torch_scaled_mm_perf(M: int, N: int, K: int) -> float:
    device = torch.device("cuda")
    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)
    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=FP8_DTYPE)
    b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=FP8_DTYPE)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()

    b_qt = b_q.t()
    scale_b_col = scale_b.t()

    def _run():
        F.scaled_mm(
            a_q, b_qt,
            scale_a=scale_a, scale_recipe_a=F.ScalingType.RowWise,
            scale_b=scale_b_col, scale_recipe_b=F.ScalingType.RowWise,
            output_dtype=OUT_DTYPE,
        )

    stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        avg_us = bench_kernel_us(_run, warmup=2, iters=101, report_average=True)
    finally:
        # This is because HIP backend sometimes generates a bunch of warnings like
        # Warning: Latency not found for MI_M=16, MI_N=16, MI_K=128, mi_input_type=BFloat8Float8_fnuz. Returning latency value of 32 (really slow).
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull)
    tflops = (2 * M * N * K) * 1e-12 / (avg_us * 1e-6)
    return tflops
