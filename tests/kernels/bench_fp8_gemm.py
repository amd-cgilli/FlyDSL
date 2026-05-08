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

from kernels.fp8_gemm_4wave import compile_fp8_gemm
from tests.test_common import verify_output, run_perftest
from tests.utils import pertoken_quant

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16


def _run_torch(a, b, scale_a, scale_b, dtype=torch.float32):
    if scale_a is not None and scale_b is not None:
        a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
        b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    else:
        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)
    c = torch.mm(a_f32, b_f32.T)
    return c.to(dtype)


def _check_gemm(M: int, N: int, K: int,
                block_m: int, block_n: int,
                validate_out: bool,
                num_warmups: int = 2,
                num_iters: int = 10):
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

    launch_fn = compile_fp8_gemm(M=M, N=N, K=K, BLOCK_M=block_m, BLOCK_N=block_n)

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def _args(c, a, b, sa, sb):
        return (
            _as_i8(a).contiguous().view(-1),
            _as_i8(b).contiguous().view(-1),
            c.contiguous().view(-1),
            sa.contiguous().view(-1),
            sb.contiguous().view(-1),
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out_raw, a_q, b_q, scale_a, scale_b))

    compiled(*_args(c_out_raw, a_q, b_q, scale_a, scale_b))
    torch.cuda.synchronize()

    # If this fails the entire benchmark fails
    if validate_out:
        out_ref = _run_torch(a_q, b_q, scale_a, scale_b)
        assert verify_output(c_out_raw.to(torch.float32), out_ref, rtol=0.1, atol=0.1)

    def _launch(c, a, b, sa, sb):
        compiled(*_args(c, a, b, sa, sb))

    _, us = run_perftest(
        _launch,
        c_out_raw,
        a_q,
        b_q,
        scale_a,
        scale_b,
        num_iters=num_iters,
        num_warmup=num_warmups,
    )

    tflops = (2 * M * N * K) * 1e-12 / (us * 1e-6)

    return tflops

def _find_best_tile_config(M: int, N: int, K: int, validate_out: bool) -> tuple[int, int, int]:
    valid_sizes = [64, 128, 256]
    tile_K = 128 # this is fixed for now

    tflops_map = {}

    for bm in valid_sizes:
        if M % bm != 0:
            continue
        for bn in valid_sizes:
            if N % bn != 0:
                continue
            tflops_map[(bm, bn, tile_K)] = _check_gemm(M, N, K, bm, bn, validate_out, num_warmups=5, num_iters=20)

    if len(tflops_map) == 0:
        return None

    sorted_items = sorted(tflops_map.items(), key=lambda item: item[1], reverse=True)
    # print(f'Configs for {M}x{N}x{K} -> {dict(sorted_items)}')
    return sorted_items[0][0]

def bench_gemm(M: int, N: int, K: int) -> tuple[float, tuple[int, int, int]]:
    res = _find_best_tile_config(M, N, K, validate_out=True)
    if not res:
        return None

    bm, bn, bk = res
    print(f'Best performing tile size for {M}x{N}x{K} -> {bm}x{bn}x{bk}')

    return _check_gemm(M, N, K, bm, bn, validate_out=False, num_warmups=10, num_iters=100), (bm, bn, bk)

def bench_torch_scaled_mm(M: int, N: int, K: int) -> float:
    device = torch.device("cuda")
    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)
    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=FP8_DTYPE)
    b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=FP8_DTYPE)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()

    b_qt = b_q.t()
    scale_b_t = scale_b.t()

    def _launch(a, b, sa, sb):
        F.scaled_mm(
            a, b,
            scale_a=sa, scale_recipe_a=F.ScalingType.RowWise,
            scale_b=sb, scale_recipe_b=F.ScalingType.RowWise,
            output_dtype=OUT_DTYPE,
        )

    stderr_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        _, us = run_perftest(
            _launch,
            a_q,
            b_qt,
            scale_a,
            scale_b_t,
            num_iters=100,
            num_warmup=10
        )
    finally:
        # This is because HIP backend sometimes generates a bunch of warnings like
        # Warning: Latency not found for MI_M=16, MI_N=16, MI_K=128, mi_input_type=BFloat8Float8_fnuz. Returning latency value of 32 (really slow).
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull)
    tflops = (2 * M * N * K) * 1e-12 / (us * 1e-6)
    return tflops


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = []
    torch_results = []
    fly_results = []

    for s in range(4, 17):
        m = n = k = s * 1024
        torch_tflops = bench_torch_scaled_mm(m, n, k)
        fly_tflops, tile_size = bench_gemm(m, n, k, True)
        print(f'{m}x{n}x{k}: torch={torch_tflops:.2f} TFLOPS, fly={fly_tflops:.2f} TFLOPS (tile={tile_size})')

        sizes.append(m)
        torch_results.append(torch_tflops)
        fly_results.append(fly_tflops)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, torch_results, "o-", label="torch scaled_mm")
    ax.plot(sizes, fly_results, "s-", label="FlyDSL fp8_gemm")
    ax.set_xlabel("M = N = K")
    ax.set_ylabel("TFLOPS")
    ax.set_title("FP8 GEMM Performance")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig("bench_fp8_gemm.png", dpi=150)
    print("Plot saved to bench_fp8_gemm.png")
