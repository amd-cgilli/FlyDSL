# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


def compile_fp8_gemm(
        *,
        M: int,
        N: int,
        K: int,
        BLOCK_M: int = 256,
        BLOCK_N: int = 256,
        NUM_SPLITS: int = 1
):
    BLOCK_K = 128

    assert M >= 1 and N >= 1
    assert BLOCK_M >= 128 and BLOCK_N >= 256 and BLOCK_M % 128 == 0 and BLOCK_N % 256 == 0
    assert K % BLOCK_K == 0

    assert NUM_SPLITS >= 1
    assert K % NUM_SPLITS == 0, f"K ({K}) must be divisible by NUM_SPLITS ({NUM_SPLITS})"
    K_PER_SPLIT = K // NUM_SPLITS
    assert K_PER_SPLIT % BLOCK_K == 0, f"K_PER_SPLIT ({K_PER_SPLIT}) must be divisible by BLOCK_K ({BLOCK_K})"
    K_ITERS_PER_SPLIT = K_PER_SPLIT // BLOCK_K
    assert K_ITERS_PER_SPLIT >= 2, (
        f"Each split needs >= 2 K iterations for double-buffered prologue, "
        f"got {K_ITERS_PER_SPLIT} (K={K}, NUM_SPLITS={NUM_SPLITS}, BLOCK_K={BLOCK_K})"
    )
    IS_SPLIT_K = NUM_SPLITS > 1

    N_BLOCKS = N // BLOCK_N
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

    M_PAD = ((M + BLOCK_M - 1) // BLOCK_M) * BLOCK_M

    a_size_bytes = M * K
    b_size_bytes = N * K
    c_size_bytes = M * N * 2
    a_scale_size_bytes = M * 4
    b_scale_size_bytes = N * 4
    workspace_size_bytes = M_PAD * N * 4  # f32 per split slice, padded to BLOCK_M rows

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        C_workspace: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        MfmaAccum_t = Vec.make_type(4, fx.Float32)
        RT_C_i = Vec.filled(4, 0.0, fx.Float32)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        Vec16_t = Vec.make_type(16, fx.Float8E4M3FN)

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
        wave_n_offset = wave_n * (N_TILES_B * 16)
        wave_m_offset = wave_m * (N_TILES_A * 16)
        block_m = fx.block_idx.x // N_BLOCKS
        block_n = fx.block_idx.x % N_BLOCKS

        split_k_idx = fx.block_idx.y
        k_base = split_k_idx * K_PER_SPLIT

        A0_gl_offset = (block_m * BLOCK_M) * K + k_base
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K + k_base
        B0_gl_offset = (block_n * BLOCK_N) * K + k_base
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K + k_base

        A_rsrc = buffer_ops.create_buffer_resource(A, max_size=False,
                                                   num_records_bytes=a_size_bytes)
        B_rsrc = buffer_ops.create_buffer_resource(B_T, max_size=False,
                                                   num_records_bytes=b_size_bytes)
        C_rsrc = buffer_ops.create_buffer_resource(C, max_size=False,
                                                   num_records_bytes=c_size_bytes)
        if const_expr(IS_SPLIT_K):
            C_ws_rsrc = buffer_ops.create_buffer_resource(
                C_workspace, max_size=False, num_records_bytes=NUM_SPLITS * workspace_size_bytes
            )
        A_scale_rsrc = buffer_ops.create_buffer_resource(A_scale, max_size=False,
                                                         num_records_bytes=a_scale_size_bytes)
        B_scale_rsrc = buffer_ops.create_buffer_resource(B_scale, max_size=False,
                                                         num_records_bytes=b_scale_size_bytes)

        def _swizzle_128(row, col):
            offset = row * 128 + col
            swizzle = ((offset % (16 * 128)) >> 8) << 4
            swizzled_offset = offset ^ swizzle
            return swizzled_offset // 128, swizzled_offset % 128

        def _compute_global_swizzle():
            offsets = []
            wave_offset = (wave_id // 2) * 16 * K
            row = ((wave_id % 2) * 64 + lane_id) // 8
            col = (lane_id % 8) * 16
            swz_row, swz_col = _swizzle_128(row, col)
            for round in range_constexpr(N_LDS_ROUNDS):
                offsets.append(wave_offset + (round * 64 + swz_row) * K + swz_col)
            return offsets

        def _load_lds(gl_src, lds_dst, k_offset, gl_offsets, n_steps):
            from flydsl._mlir.dialects import memref as memref_dialect

            def _lds_dst_at(step):
                base_idx = memref_dialect.extract_aligned_pointer_as_index(lds_dst)
                offset_idx = base_idx + fx.Index(wave_id * 1024 + step * 8192)
                return buffer_ops.create_llvm_ptr(offset_idx, address_space=3)

            for step in range_constexpr(n_steps):
                rocdl.raw_ptr_buffer_load_lds(
                    gl_src,
                    _lds_dst_at(step),
                    fx.Int32(16),
                    fx.Int32(gl_offsets[step]),  # voffset
                    fx.Int32(k_offset),  # soffset
                    fx.Int32(0),
                    fx.Int32(1),
                )

        def _pack_i32x4_i32x8(lo, hi):
            # Pack two i32x4 as one i32x8
            return lo.shuffle(hi, list(range(8)))

        def _load_a_rt(lds_src, wave_offset):
            frag = []
            for i in range_constexpr(N_TILES_A):
                halves = []
                for k_i in range_constexpr(2):
                    row = lane_id % 16
                    col = (lane_id // 16) * 16 + k_i * 64
                    row_swz, col_swz = _swizzle_128(row, col)
                    v = Vec.load(Vec16_t, lds_src, [fx.Index(row_swz * 128 + col_swz + wave_offset + i * 2048)])
                    halves.append(v.bitcast(fx.Int32))
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
            return frag

        def _load_b_rt(lds_src, wave_offset):
            frag = []
            for i in range_constexpr(N_TILES_B):
                halves = []
                for k_i in range_constexpr(2):
                    row = lane_id % 16
                    col = (lane_id // 16) * 16 + k_i * 64
                    row_swz, col_swz = _swizzle_128(row, col)
                    v = Vec.load(Vec16_t, lds_src, [fx.Index(row_swz * 128 + col_swz + wave_offset + i * 2048)])
                    halves.append(v.bitcast(fx.Int32))
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
            return frag

        def _store_C_scaled(c_frag, base_row, base_col):
            def _preload_a_scales():
                scales = []
                for i in range_constexpr(N_TILES_A):
                    row = base_row + i * 16 + (lane_id // 16) * 4
                    scales.append(
                        Vec(buffer_ops.buffer_load(A_scale_rsrc, fx.Int32(row), vec_width=4, dtype=fx.Float32))
                    )
                return scales

            def _preload_b_scales():
                scales = []
                for i in range_constexpr(N_TILES_B):
                    col = base_col + i * 16 + lane_id % 16
                    scales.append(buffer_ops.buffer_load(B_scale_rsrc, fx.Int32(col), vec_width=1, dtype=fx.Float32))
                return scales

            a_scales = _preload_a_scales()
            b_scales = _preload_b_scales()
            for ti in range_constexpr(N_TILES_A):
                row = base_row + ti * 16 + (lane_id // 16) * 4
                for tj in range_constexpr(N_TILES_B):
                    col = base_col + tj * 16 + lane_id % 16
                    vec_f32 = Vec(c_frag[_c_idx(ti, tj)])
                    for i in range_constexpr(4):
                        scaled = vec_f32[i] * (a_scales[ti][i] * b_scales[tj])
                        if const_expr(IS_SPLIT_K):
                            ws_offset = split_k_idx * M_PAD * N + (row + i) * N + col
                            buffer_ops.buffer_store(scaled, C_ws_rsrc, fx.Int32(ws_offset))
                        else:
                            buffer_ops.buffer_store(scaled.to(fx.BFloat16), C_rsrc, fx.Int32((row + i) * N + col))

        def _c_idx(i, j):
            return i * N_TILES_B + j

        def _mfma_ABt_all(a, b, c):
            for i in range_constexpr(N_TILES_A):
                for j in range_constexpr(N_TILES_B):
                    c[_c_idx(i, j)] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        MfmaAccum_t, [a[i], b[j], c[_c_idx(i, j)], 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F]
                    )
            return c

        def _wait_barrier(count):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

        # 2x2 config of 4x2 (instead of 4x4 in 4wave) 16x16 sub-tiles
        c00_frag = [RT_C_i] * N_ACCUMS
        c01_frag = [RT_C_i] * N_ACCUMS
        c10_frag = [RT_C_i] * N_ACCUMS
        c11_frag = [RT_C_i] * N_ACCUMS

        global_offsets = _compute_global_swizzle()

        _load_lds(B_rsrc, b_cur0, B0_gl_offset + 0 * BLOCK_K, global_offsets, N_LDS_STEPS_B)
        _load_lds(A_rsrc, a_cur0, A0_gl_offset + 0 * BLOCK_K, global_offsets, N_LDS_STEPS_A)
        _load_lds(B_rsrc, b_cur1, B1_gl_offset + 0 * BLOCK_K, global_offsets, N_LDS_STEPS_B)
        _load_lds(A_rsrc, a_cur1, A1_gl_offset + 0 * BLOCK_K, global_offsets, N_LDS_STEPS_A)

        if wave_m == 1:
            rocdl.s_barrier()

        _wait_barrier(N_LDS_STEPS_A + N_LDS_STEPS_B)

        _load_lds(B_rsrc, b_next0, B0_gl_offset + 1 * BLOCK_K, global_offsets, N_LDS_STEPS_B)
        _load_lds(A_rsrc, a_next0, A0_gl_offset + 1 * BLOCK_K, global_offsets, N_LDS_STEPS_A)
        _load_lds(B_rsrc, b_next1, B1_gl_offset + 1 * BLOCK_K, global_offsets, N_LDS_STEPS_B)

        _wait_barrier(N_LDS_STEPS_A + 2 * N_LDS_STEPS_B)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = _load_b_rt(b_cur0, wave_n_offset * BLOCK_K)
            a0_frag = _load_a_rt(a_cur0, wave_m_offset * BLOCK_K)
            _load_lds(A_rsrc, a_next1, A1_gl_offset + (k + 1) * BLOCK_K, global_offsets, N_LDS_STEPS_A)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = _load_b_rt(b_cur1, wave_n_offset * BLOCK_K)
            _load_lds(B_rsrc, b_cur0, B0_gl_offset + (k + 2) * BLOCK_K, global_offsets, N_LDS_STEPS_B)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = _load_a_rt(a_cur1, wave_m_offset * BLOCK_K)
            _load_lds(A_rsrc, a_cur0, A0_gl_offset + (k + 2) * BLOCK_K, global_offsets, N_LDS_STEPS_A)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            _load_lds(B_rsrc, b_cur1, B1_gl_offset + (k + 2) * BLOCK_K, global_offsets, N_LDS_STEPS_B)
            _wait_barrier(2 * N_LDS_STEPS_A + N_LDS_STEPS_B)

            rocdl.s_setprio(1)
            _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # Step k = K_ITERS - 2
        k = K_ITERS - 2
        b0_frag = _load_b_rt(b_cur0, wave_n_offset * BLOCK_K)
        a0_frag = _load_a_rt(a_cur0, wave_m_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = _load_b_rt(b_cur1, wave_n_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = _load_a_rt(a_cur1, wave_m_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = _load_b_rt(b_next0, wave_n_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()
        # Swap cur and next
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1


        # Step k = K_ITERS - 1
        k = K_ITERS - 1
        a0_frag = _load_a_rt(a_cur0, wave_m_offset * BLOCK_K)
        _wait_barrier(0)

        rocdl.s_setprio(1)
        _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = _load_b_rt(b_cur1, wave_n_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = _load_a_rt(a_cur1, wave_m_offset * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Scale and store back to gmem
        base_row = block_m * BLOCK_M + wave_m_offset
        base_col = block_n * BLOCK_N + wave_n_offset

        _store_C_scaled(c00_frag, base_row + 0, base_col + 0)
        _store_C_scaled(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        _store_C_scaled(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        _store_C_scaled(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        C_workspace: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        stream: fx.Stream,
    ):
        from flydsl._mlir import ir
        from flydsl.compiler.kernel_function import CompilationContext

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
        grid_x = ((M + BLOCK_M - 1) // BLOCK_M) * (N // BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            C_workspace,
            A_scale,
            B_scale,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, NUM_SPLITS, 1), block=(512, 1, 1), stream=stream)

    if not IS_SPLIT_K:
        return launch_gemm, None

    REDUCE_BLOCK = 256
    REDUCE_VEC = 4
    REDUCE_ELEMS_PER_BLOCK = REDUCE_BLOCK * REDUCE_VEC

    @flyc.kernel
    def reduce_kernel(C_workspace: fx.Tensor, C: fx.Tensor):
        C_ws_rsrc = buffer_ops.create_buffer_resource(
            C_workspace, max_size=False, num_records_bytes=NUM_SPLITS * workspace_size_bytes
        )
        C_rsrc = buffer_ops.create_buffer_resource(C, max_size=False, num_records_bytes=c_size_bytes)

        base_idx = fx.block_idx.x * REDUCE_ELEMS_PER_BLOCK + fx.thread_idx.x * REDUCE_VEC
        total_elems = M * N
        if base_idx < total_elems:
            acc = Vec(buffer_ops.buffer_load(C_ws_rsrc, fx.Int32(base_idx), vec_width=4, dtype=fx.Float32))
            for s in range_constexpr(NUM_SPLITS - 1):
                ws_offset = (s + 1) * M_PAD * N + base_idx
                val = Vec(buffer_ops.buffer_load(C_ws_rsrc, fx.Int32(ws_offset), vec_width=4, dtype=fx.Float32))
                acc = acc + val
            buffer_ops.buffer_store(acc.to(fx.BFloat16), C_rsrc, fx.Int32(base_idx))

    @flyc.jit
    def launch_reduce(C_workspace: fx.Tensor, C: fx.Tensor, stream: fx.Stream):
        total_elems = M * N
        grid_x = (total_elems + REDUCE_ELEMS_PER_BLOCK - 1) // REDUCE_ELEMS_PER_BLOCK
        reduce_kernel(
            C_workspace, C
        ).launch(grid=(grid_x, 1, 1), block=(REDUCE_BLOCK, 1, 1), stream=stream)

    return launch_gemm, launch_reduce


