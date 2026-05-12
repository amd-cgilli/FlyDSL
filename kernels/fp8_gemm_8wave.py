# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


def compile_fp8_gemm(
        *,
        M: int,
        N: int,
        K: int
):
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128

    assert M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

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
        block_m = fx.block_idx.x // N_BLOCKS
        block_n = fx.block_idx.x % N_BLOCKS

        A0_gl_offset = (block_m * BLOCK_M) * K
        A1_gl_offset = (block_m * BLOCK_M + LDS_BLOCK_M) * K
        B0_gl_offset = (block_n * BLOCK_N) * K
        B1_gl_offset = (block_n * BLOCK_N + LDS_BLOCK_N) * K

        A_rsrc = buffer_ops.create_buffer_resource(A)
        B_rsrc = buffer_ops.create_buffer_resource(B_T)
        C_rsrc = buffer_ops.create_buffer_resource(C)

        A_scale_rsrc = buffer_ops.create_buffer_resource(A_scale)
        B_scale_rsrc = buffer_ops.create_buffer_resource(B_scale)

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
            for round in range_constexpr(2):
                offsets.append(wave_offset + (round * 64 + swz_row) * K + swz_col)
            return offsets

        def _load_lds(gl_src, lds_dst, k_offset, gl_offsets):
            from flydsl._mlir.dialects import memref as memref_dialect

            lds_base_i = memref_dialect.extract_aligned_pointer_as_index(lds_dst)
            for step in range_constexpr(2):
                lds_ptr = buffer_ops.create_llvm_ptr(
                    fx.Int64(lds_base_i + fx.Index(wave_id * 1024 + step * 8192)), address_space=3
                )
                rocdl.raw_ptr_buffer_load_lds(
                    gl_src,
                    lds_ptr,
                    fx.Int32(16),
                    fx.Int32(gl_offsets[step]),  # voffset
                    fx.Int32(k_offset),  # soffset
                    fx.Int32(0),
                    fx.Int32(0),
                )

        def _pack_i32x4_i32x8(lo, hi):
            # Pack two i32x4 as one i32x8
            return lo.shuffle(hi, list(range(8)))

        def _load_a_rt(lds_src, wave_offset):
            frag = []
            for k_i in range_constexpr(2):
                row = lane_id % 16
                col = (lane_id // 16) * 16 + k_i * 64
                row_swz, col_swz = _swizzle_128(row, col)
                halves = []
                for i in range_constexpr(4):
                    v = Vec.load(Vec16_t, lds_src, [fx.Index(row_swz * 128 + col_swz + wave_offset + i * 2048)])
                    halves.append(v.bitcast(fx.Int32))
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
                frag.append(_pack_i32x4_i32x8(halves[2], halves[3]))
            return frag

        def _load_b_rt(lds_src, wave_offset):
            frag = []
            for k_i in range_constexpr(2):
                row = lane_id % 16
                col = (lane_id // 16) * 16 + k_i * 64
                row_swz, col_swz = _swizzle_128(row, col)
                halves = []
                for i in range_constexpr(2):
                    v = Vec.load(Vec16_t, lds_src, [fx.Index(row_swz * 128 + col_swz + wave_offset + i * 2048)])
                    halves.append(v.bitcast(fx.Int32))
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
            return frag

        def _store_C_scaled(c_frag, base_row, base_col):
            def _preload_a_scales():
                scales = []
                for i in range_constexpr(4):
                    row = base_row + i * 16 + (lane_id // 16) * 4
                    scales.append(
                        Vec(buffer_ops.buffer_load(A_scale_rsrc, fx.Int32(row), vec_width=4, dtype=fx.Float32))
                    )
                return scales

            def _preload_b_scales():
                scales = []
                for i in range_constexpr(2):
                    col = base_col + i * 16 + lane_id % 16
                    scales.append(buffer_ops.buffer_load(B_scale_rsrc, fx.Int32(col), vec_width=1, dtype=fx.Float32))
                return scales

            a_scales = _preload_a_scales()
            b_scales = _preload_b_scales()
            for ti in range_constexpr(4):
                row = base_row + ti * 16 + (lane_id // 16) * 4
                for tj in range_constexpr(2):
                    col = base_col + tj * 16 + lane_id % 16
                    vec_f32 = Vec(c_frag[_c_idx(ti, tj)])
                    for i in range_constexpr(
                        4
                    ):
                        scaled = (vec_f32[i] * (a_scales[ti][i] * b_scales[tj])).to(fx.BFloat16)
                        buffer_ops.buffer_store(scaled, C_rsrc, fx.Int32((row + i) * N + col))

        def _c_idx(i, j):
            return i * 2 + j

        def _mfma_ABt_all(a, b, c):
            for i in range_constexpr(4):
                for j in range_constexpr(2):
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
        c00_frag = [RT_C_i] * 8
        c01_frag = [RT_C_i] * 8
        c10_frag = [RT_C_i] * 8
        c11_frag = [RT_C_i] * 8

        global_offsets = _compute_global_swizzle()

        _load_lds(B_rsrc, b_cur0, B0_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(A_rsrc, a_cur0, A0_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, b_cur1, B1_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(A_rsrc, a_cur1, A1_gl_offset + 0 * BLOCK_K, global_offsets)

        if wave_m == 1:
            rocdl.s_barrier()

        _wait_barrier(4)

        _load_lds(B_rsrc, b_next0, B0_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(A_rsrc, a_next0, A0_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, b_next1, B1_gl_offset + 1 * BLOCK_K, global_offsets)

        _wait_barrier(6)

        for k in range_constexpr(K_ITERS - 2):
            b0_frag = _load_b_rt(b_cur0, wave_n * 32 * BLOCK_K)
            a0_frag = _load_a_rt(a_cur0, wave_m * 64 * BLOCK_K)
            _load_lds(A_rsrc, a_next1, A1_gl_offset + (k + 1) * BLOCK_K, global_offsets)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = _load_b_rt(b_cur1, wave_n * 32 * BLOCK_K)
            _load_lds(B_rsrc, b_cur0, B0_gl_offset + (k + 2) * BLOCK_K, global_offsets)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a1_frag = _load_a_rt(a_cur1, wave_m * 64 * BLOCK_K)
            _load_lds(A_rsrc, a_cur0, A0_gl_offset + (k + 2) * BLOCK_K, global_offsets)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            _load_lds(B_rsrc, b_cur1, B1_gl_offset + (k + 2) * BLOCK_K, global_offsets)
            _wait_barrier(6)

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
        b0_frag = _load_b_rt(b_cur0, wave_n * 32 * BLOCK_K)
        a0_frag = _load_a_rt(a_cur0, wave_m * 64 * BLOCK_K)
        _load_lds(A_rsrc, a_next1, A1_gl_offset + (k + 2) * BLOCK_K, global_offsets)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = _load_b_rt(b_cur1, wave_n * 32 * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = _load_a_rt(a_cur1, wave_m * 64 * BLOCK_K)
        _wait_barrier(4)

        rocdl.s_setprio(1)
        _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b0_frag = _load_b_rt(b_next0, wave_n * 32 * BLOCK_K)
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
        a0_frag = _load_a_rt(a_cur0, wave_m * 64 * BLOCK_K)
        _wait_barrier(0)

        rocdl.s_setprio(1)
        _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = _load_b_rt(b_cur1, wave_n * 32 * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a1_frag = _load_a_rt(a_cur1, wave_m * 64 * BLOCK_K)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        # Scale and store back to gmem
        base_row = block_m * BLOCK_M + wave_m * 64
        base_col = block_n * BLOCK_N + wave_n * 32

        _store_C_scaled(c00_frag, base_row + 0, base_col + 0)
        _store_C_scaled(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        _store_C_scaled(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        _store_C_scaled(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)


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
        grid_x = (M * N) // (BLOCK_M * BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm

