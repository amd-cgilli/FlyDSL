# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""8-wave FP8 matmul with row-wise scaling for AMD CDNA4.

Algorithm derived from HipKittens FP8_8wave
(https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_8wave/8_wave.cu#L1)
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.fp8_gemm_utils import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    StoreC,
    ceildiv,
    compute_global_swizzle,
    make_fp8_buffer_tensor,
    wait_barrier
)


def compile_fp8_gemm_8w(
        *,
        M: int,
        N: int,
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        b_preshuffled: bool = False,
        n_splits: int = 1):
    BLOCK_K = 128

    assert M >= 1 and N >= 1
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0

    assert n_splits >= 1
    assert K % n_splits == 0, f"K ({K}) must be divisible by n_splits ({n_splits})"
    K_PER_SPLIT = K // n_splits
    assert K_PER_SPLIT % BLOCK_K == 0, f"K_PER_SPLIT ({K_PER_SPLIT}) must be divisible by BLOCK_K ({BLOCK_K})"
    K_ITERS_PER_SPLIT = K_PER_SPLIT // BLOCK_K
    assert K_ITERS_PER_SPLIT >= 2, (
        f"Each split needs >= 2 K iterations for double-buffered prologue, "
        f"got {K_ITERS_PER_SPLIT} (K={K}, n_splits={n_splits}, BLOCK_K={BLOCK_K})"
    )
    _is_split_k = n_splits > 1

    _m_pad = ceildiv(M, BLOCK_M) * BLOCK_M

    N_BLOCKS = ceildiv(N, BLOCK_N)
    K_ITERS = K_PER_SPLIT // BLOCK_K

    N_TILES_A = BLOCK_M // 64
    N_TILES_B = BLOCK_N // 128
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    N_LDS_STEPS_A = LDS_BLOCK_M // 64
    N_LDS_STEPS_B = LDS_BLOCK_N // 64
    N_LDS_ROUNDS = max(N_LDS_STEPS_A, N_LDS_STEPS_B)

    A_lds_cur0_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_0")
    A_lds_cur1_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_1")
    A_lds_next0_alloc = SmemAllocator(None, "gfx950", "A_lds_next_0")
    A_lds_next1_alloc = SmemAllocator(None, "gfx950", "A_lds_next_1")
    B_lds_cur0_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_0")
    B_lds_cur1_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_1")
    B_lds_next0_alloc = SmemAllocator(None, "gfx950", "B_lds_next_0")
    B_lds_next1_alloc = SmemAllocator(None, "gfx950", "B_lds_next_1")

    # half size
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    A_lds_cur0_alloc.ptr = a_lds_size
    A_lds_cur1_alloc.ptr = a_lds_size
    A_lds_next0_alloc.ptr = a_lds_size
    A_lds_next1_alloc.ptr = a_lds_size
    B_lds_cur0_alloc.ptr = b_lds_size
    B_lds_cur1_alloc.ptr = b_lds_size
    B_lds_next0_alloc.ptr = b_lds_size
    B_lds_next1_alloc.ptr = b_lds_size

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type

        a_cur0 = SmemPtr(A_lds_cur0_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_cur1 = SmemPtr(A_lds_cur1_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_next0 = SmemPtr(A_lds_next0_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()
        a_next1 = SmemPtr(A_lds_next1_alloc.get_base(), 0, F8_IR_t, shape=(a_lds_size,)).get()

        b_cur0 = SmemPtr(B_lds_cur0_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_cur1 = SmemPtr(B_lds_cur1_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_next0 = SmemPtr(B_lds_next0_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()
        b_next1 = SmemPtr(B_lds_next1_alloc.get_base(), 0, F8_IR_t, shape=(b_lds_size,)).get()

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64
        wave_m = wave_id // 4
        wave_n = wave_id % 4
        block_m = fx.block_idx.x // N_BLOCKS
        block_n = fx.block_idx.x % N_BLOCKS

        split_k_idx = fx.block_idx.y
        k_base = split_k_idx * K_PER_SPLIT

        A0_gl_offset = (block_m * BLOCK_M) * K + k_base
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K + k_base
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K
        B0_gl_offset = (block_n * BLOCK_N) * K + k_base
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K + k_base

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        a_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        b_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=b_preshuffled)

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

        a_g2s = G2SLoader(a_div, gl_off_a, N_LDS_STEPS_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(b_div, gl_off_b, N_LDS_STEPS_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_m, N_TILES_A)
        b_s2r = S2RLoader(wave_n, N_TILES_B)
        store_c = StoreC(A_scale, B_scale, C, M, N, mfma.idx, N_TILES_A, N_TILES_B, _is_split_k, _m_pad)

        # 2x2 config of 4x2 (instead of 4x4 in 4wave) 16x16 sub-tiles
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        b_g2s.load(b_cur0, B0_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur0, A0_gl_offset + 0 * BLOCK_K)
        b_g2s.load(b_cur1, B1_gl_offset + 0 * B_K_STEP)
        a_g2s.load(a_cur1, A1_gl_offset + 0 * BLOCK_K)

        # if wave_m == 1:
        #     rocdl.s_barrier()

        wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        b_g2s.load(b_next0, B0_gl_offset + 1 * B_K_STEP)
        a_g2s.load(a_next0, A0_gl_offset + 1 * BLOCK_K)
        b_g2s.load(b_next1, B1_gl_offset + 1 * B_K_STEP)

        wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
            a0_frag = a_s2r.load(a_cur0)
            a_g2s.load(a_next1, A1_gl_offset + (k + 1) * BLOCK_K)
            # rocdl.s_barrier()

            rocdl.s_setprio(1)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            # rocdl.s_barrier()

            b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
            b_g2s.load(b_cur0, B0_gl_offset + (k + 2) * B_K_STEP)
            # rocdl.s_barrier()

            rocdl.s_setprio(1)
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            # rocdl.s_barrier()

            a1_frag = a_s2r.load(a_cur1)
            a_g2s.load(a_cur0, A0_gl_offset + (k + 2) * BLOCK_K)
            # rocdl.s_barrier()

            rocdl.s_setprio(1)
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            # rocdl.s_barrier()

            b_g2s.load(b_cur1, B1_gl_offset + (k + 2) * B_K_STEP)
            wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            # rocdl.s_barrier()

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        b0_frag = b_s2r.load(b_cur0, preshuffled=b_preshuffled)
        a0_frag = a_s2r.load(a_cur0)
        # rocdl.s_barrier()

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        # rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        # rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        # rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        # rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        # rocdl.s_barrier()

        b0_frag = b_s2r.load(b_next0, preshuffled=b_preshuffled)
        # rocdl.s_barrier()

        rocdl.s_setprio(1)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        # rocdl.s_barrier()
        # Swap cur and next
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 1
        k = K_ITERS - 1
        a0_frag = a_s2r.load(a_cur0)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        # rocdl.s_barrier()

        b1_frag = b_s2r.load(b_cur1, preshuffled=b_preshuffled)
        # rocdl.s_barrier()

        rocdl.s_setprio(1)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        # rocdl.s_barrier()

        a1_frag = a_s2r.load(a_cur1)
        # rocdl.s_barrier()

        rocdl.s_setprio(1)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        # rocdl.s_barrier()

        # Scale and store back to gmem
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        a_scales_0 = store_c.load_a_scales(base_row)
        a_scales_1 = store_c.load_a_scales(base_row + LDS_BLOCK_M)
        b_scales_0 = store_c.load_b_scales(base_col)
        b_scales_1 = store_c.load_b_scales(base_col + LDS_BLOCK_N)

        store_c.store_with_scales(c00_frag, base_row + 0, base_col + 0, a_scales_0, b_scales_0)
        store_c.store_with_scales(c01_frag, base_row + 0, base_col + LDS_BLOCK_N, a_scales_0, b_scales_1)
        store_c.store_with_scales(c10_frag, base_row + LDS_BLOCK_M, base_col + 0, a_scales_1, b_scales_0)
        store_c.store_with_scales(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N, a_scales_1, b_scales_1)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        stream: fx.Stream,
    ):
        from flydsl._mlir import ir

        A_lds_cur0_alloc.finalized = False
        A_lds_cur1_alloc.finalized = False
        A_lds_next0_alloc.finalized = False
        A_lds_next1_alloc.finalized = False
        B_lds_cur0_alloc.finalized = False
        B_lds_cur1_alloc.finalized = False
        B_lds_next0_alloc.finalized = False
        B_lds_next1_alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            A_lds_cur0_alloc.finalize()
            A_lds_cur1_alloc.finalize()
            A_lds_next0_alloc.finalize()
            A_lds_next1_alloc.finalize()
            B_lds_cur0_alloc.finalize()
            B_lds_cur1_alloc.finalize()
            B_lds_next0_alloc.finalize()
            B_lds_next1_alloc.finalize()
        grid_x = ceildiv(M, BLOCK_M) * N_BLOCKS
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, n_splits, 1), block=(512, 1, 1), stream=stream)

    if not _is_split_k:
        return launch_gemm, None


    REDUCE_BLOCK = 256
    REDUCE_VEC = 4
    REDUCE_ELEMS_PER_BLOCK = REDUCE_BLOCK * REDUCE_VEC

    @flyc.kernel
    def reduce_kernel(C_ws: fx.Tensor, C: fx.Tensor):
        c_ws_rsrc = buffer_ops.create_buffer_resource(
            C_ws, max_size=False, num_records_bytes=n_splits * (_m_pad * N * 4)
        )
        c_rsrc = buffer_ops.create_buffer_resource(C, max_size=False, num_records_bytes=M * N * 2)

        base_idx = fx.block_idx.x * REDUCE_ELEMS_PER_BLOCK + fx.thread_idx.x * REDUCE_VEC
        total_elems = M * N
        if base_idx < total_elems:
            acc = Vec(buffer_ops.buffer_load(c_ws_rsrc, fx.Int32(base_idx), vec_width=4, dtype=fx.Float32))
            for s in range_constexpr(n_splits - 1):
                ws_offset = (s + 1) * _m_pad * N + base_idx
                val = Vec(buffer_ops.buffer_load(c_ws_rsrc, fx.Int32(ws_offset), vec_width=4, dtype=fx.Float32))
                acc = acc + val
            buffer_ops.buffer_store(acc.to(fx.BFloat16), c_rsrc, fx.Int32(base_idx))

    @flyc.jit
    def launch_reduce(C_ws: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        total_elems = M * N
        grid_x = ceildiv(total_elems, REDUCE_ELEMS_PER_BLOCK)
        reduce_kernel(C_ws, C).launch(grid=(grid_x, 1, 1), block=(REDUCE_BLOCK, 1, 1), stream=stream)

    return launch_gemm, launch_reduce
