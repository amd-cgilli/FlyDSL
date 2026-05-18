import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import flydsl.compiler as flyc
from kernels.fp8_gemm_4wave import compile_fp8_gemm_4w
from kernels.fp8_gemm_8wave import compile_fp8_gemm_8w
from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8
from tests.test_common import run_perftest, verify_output
from tests.utils import pertoken_quant, shuffle_weight

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16


@dataclass
class FlyGemmConfig:
    tile_m: int
    tile_n: int
    num_waves: int
    tflops: float
    num_split: int = 1


def _run_torch(a, b, scale_a, scale_b, dtype=torch.float32):
    a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
    b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    c = torch.mm(a_f32, b_f32.T)
    return c.to(dtype)


def _as_i8(t):
    return t.view(torch.int8) if "float8" in str(t.dtype) else t


def _check_gemm(
    compile_kernel_fn,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    num_splits: int = 1,
    num_warmups: int = 2,
    num_iters: int = 10,
    validate_out: bool = True
):
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

    IS_SPLIT_K = num_splits > 1
    M_PAD = ((M + tile_m - 1) // tile_m) * tile_m
    if IS_SPLIT_K:
        c_workspace = torch.zeros((num_splits, M_PAD, N), dtype=torch.float32, device=device)
    else:
        c_workspace = torch.empty(0, dtype=torch.float32, device=device)

    launch_gemm_fn, launch_reduce_fn = compile_kernel_fn(
        M=M, N=N, K=K,
        BLOCK_M=tile_m,
        BLOCK_N=tile_n,
        n_splits=num_splits)

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

    if validate_out:
        c_ref = _run_torch(a_q, b_q, scale_a, scale_b)
        c_out_f32 = c_out_raw.to(torch.float32)
        assert verify_output(c_out_f32, c_ref, rtol=0.1, atol=0.1)

    flops = 2 * M * N * K
    return flops / (us / 1e6) / 1e12


def _valid_splits(K):
    k_iters = K // 128
    return [s for s in range(1, k_iters + 1)
            if k_iters % s == 0 and k_iters // s >= 2]


_KERNEL_CONFIGS = {
    4: {
        "fn": compile_fp8_gemm_4w,
        "valid_bm": [64, 128, 256],
        "valid_bn": [64, 128, 256],
    },
    8: {
        "fn": compile_fp8_gemm_8w,
        "valid_bm": [128, 256],
        "valid_bn": [256],
    },
}


def _list_available_configs(M: int, N: int, K: int, num_waves: int) -> list[FlyGemmConfig]:
    assert num_waves in [4, 8]

    print(f'Finding best {num_waves}-Wave kernel for M={M}, N={N}, K={K}...')

    splits = _valid_splits(K)
    configs = []

    valid_bm = _KERNEL_CONFIGS[num_waves]["valid_bm"]
    valid_bn = _KERNEL_CONFIGS[num_waves]["valid_bn"]
    compile_fn = _KERNEL_CONFIGS[num_waves]["fn"]

    for bm in valid_bm:
        for bn in valid_bn:
            for s in splits:
                # Heuristic: if the number of thread blocks is a multiple of the number of CUs in the chip skip split-k
                n_blocks = ((M - bm + 1) // bm) * ((N - bn + 1) // bn)
                if n_blocks % 256 == 0 and s != 1:
                    continue
                print(f'[{num_waves}w] testing tile_m={bm} tile_n={bn} splits={s}...', end='')
                try:
                    perf = _check_gemm(
                        compile_fn,
                        M, N, K, bm, bn, num_splits=s,
                        num_warmups=10, num_iters=20,
                        validate_out=False
                    )
                except Exception as e:
                    print(f"    {num_waves}W {bm}x{bn} SK{s}: {type(e).__name__}: {e}")
                    continue
                print(f'{perf:.2f} TFLOPS')
                configs.append(FlyGemmConfig(
                    tile_m=bm, tile_n=bn,
                    num_waves=num_waves, tflops=perf,
                    num_split=s
                ))

    print(f'Found {len(configs)} valid {num_waves}-Wave kernels!')
    return configs


def bench_gemm(M: int, N: int, K: int) -> FlyGemmConfig:
    all_configs = _list_available_configs(M, N, K, 4)
    if M > 1024:
        all_configs.extend(_list_available_configs(M, N, K, 8))

    if not all_configs:
        raise ValueError(f"No valid GEMM config found for M={M}, N={N}, K={K}")

    all_configs.sort(key=lambda c: c.tflops, reverse=True)

    best_config = all_configs[0]

    print(f'Best kernel for M={M} N={N} K={K} is: {best_config.num_waves}-Wave tile_m={best_config.tile_m} tile_n={best_config.tile_n} num_splits={best_config.num_split}')

    bench_tflops = _check_gemm(compile_kernel_fn=_KERNEL_CONFIGS[best_config.num_waves]["fn"],
                               M=M, N=N, K=K,
                               tile_m=best_config.tile_m,
                               tile_n=best_config.tile_n,
                               num_splits=best_config.num_split,
                               num_warmups=10,
                               num_iters=100,
                               validate_out=True)

    return FlyGemmConfig(best_config.tile_m, best_config.tile_n,
                         best_config.num_waves, bench_tflops,
                         best_config.num_split)


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
    return (2 * M * N * K) * 1e-12 / (us * 1e-6)


def bench_preshuffle_gemm(
    M: int, N: int, K: int,
    tile_m: int, tile_n: int, tile_k: int,
    lds_stage: int = 2,
    use_cshuffle: bool = False,
    use_async_copy: bool = False,
    waves_per_eu: int = 0,
    xcd_swizzle: int = 0,
) -> float:
    device = torch.device("cuda")
    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)

    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=FP8_DTYPE)
    b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=FP8_DTYPE)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    b_shuffled = shuffle_weight(b_q, layout=(16, 16))

    scale_a = scale_a.squeeze().contiguous()
    scale_b = scale_b.squeeze().contiguous()

    c_out = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)
    dummy_bias = torch.empty(0, dtype=OUT_DTYPE, device=device)

    wpe = None if waves_per_eu <= 0 else waves_per_eu

    launch_fn = compile_preshuffle_gemm_a8(
        M=M, N=N, K=K,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        in_dtype="fp8", out_dtype="bf16",
        lds_stage=lds_stage,
        use_cshuffle_epilog=use_cshuffle,
        use_async_copy=use_async_copy,
        waves_per_eu=wpe,
        xcd_swizzle=xcd_swizzle,
    )

    def _args(c, a, b, sa, sb):
        return (
            c.contiguous().view(-1),
            _as_i8(a).contiguous().view(-1),
            _as_i8(b).contiguous().view(-1),
            sa.contiguous().view(-1),
            sb.contiguous().view(-1),
            dummy_bias,
            M, N, torch.cuda.current_stream(),
        )

    compiled_fn = flyc.compile(launch_fn, *_args(c_out, a_q, b_shuffled, scale_a, scale_b))

    def _launch(c, a, b, sa, sb):
        compiled_fn(*_args(c, a, b, sa, sb))

    _, us = run_perftest(
        _launch,
        c_out, a_q, b_shuffled, scale_a, scale_b,
        num_iters=100, num_warmup=10,
    )
    torch.cuda.synchronize()

    return (2 * M * N * K) * 1e-12 / (us * 1e-6)
