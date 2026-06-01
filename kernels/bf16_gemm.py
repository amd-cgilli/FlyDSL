# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import ceildiv, divmod, wait_barrier


def compile_bf16_gemm(
    *,
    K: int
):
    # 256x256x64
    BLOCK_SIZE = 256
    BLOCK_K = 64

    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K

    BYTES_THREAD = 16
    NUM_WAVES = 8
    BYTES_WAVE = BYTES_THREAD * 64
    LDS_TILE_SIZE = BLOCK_SIZE * BLOCK_K * 2

    N_LDS_STEPS = LDS_TILE_SIZE // (BYTES_THREAD * 512)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # MFMA atom is 32x32x16
        #   - each lane holds 8 BF16 values of A/B
        #   - each lane holds 16 F32 values of C (4 row groups, each row group has 8 rows in it with each lane holding 4 consecutive elements in the column)
        # The accumulator is 128x64 -> I need 8 of these 32x32 fragments (4x2 config)
        Accum_t = Vec.make_type(16, fx.Float32)
        Accum_zero = Vec.filled(16, 0.0, fx.Float32)

        A_rsrc = buffer_ops.create_buffer_resource(A)
        B_rsrc = buffer_ops.create_buffer_resource(B_T)
        C_rsrc = buffer_ops.create_buffer_resource(C)

        c_frag = [
            [Accum_zero for _ in range_constexpr(2)]
            for _ in range_constexpr(4)
        ]

        lds_alloc = fx.SharedAllocator()
        A_lds = [
            lds_alloc.allocate(fx.Array[fx.BFloat16, BLOCK_SIZE * BLOCK_K, 16]).peek().ptr
            for _ in range_constexpr(2)
        ]
        B_lds = [
            lds_alloc.allocate(fx.Array[fx.BFloat16, BLOCK_SIZE * BLOCK_K, 16]).peek().ptr
            for _ in range_constexpr(2)
        ]
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        # This tile (C) row, col
        row, col = divmod(fx.block_idx.x, ceildiv(c_n, BLOCK_SIZE))
        # The row, col of the sub-tile for this wave
        wave_row, wave_col = divmod(wave_id, 4)

        A_gl_offset = row * BLOCK_SIZE * K * 2
        B_gl_offset = col * BLOCK_SIZE * K * 2

        # The 256x64 tile is stored in LDS as 16 contiguous 32x32 subtiles
        # (8 subtile-rows x 2 subtile-cols). The swizzle operates within each
        # 32x32 subtile only.
        SUBTILE_ROWS = 32
        SUBTILE_COLS = 32
        SUBTILES_PER_ROW = BLOCK_K // SUBTILE_COLS  # 64 // 32 = 2

        def _swizzle32_byte_offset(row, col):
            # st_32x32 swizzle from HipKittens: takes (row, col) within a 32x32
            # subtile (both in elements), returns a byte offset within that subtile.
            offset = (row * SUBTILE_COLS + col) * 2
            first_swz = ((offset % 1024) >> 9) << 5
            second_swz = ((offset % 2048) >> 10) << 4
            return offset ^ first_swz ^ second_swz

        SUBTILE_BYTES = SUBTILE_ROWS * SUBTILE_COLS * 2
        SUBTILE_ROW_BYTES = SUBTILE_COLS * 2  # 64

        def _precompute_global_swizzle():
            # Derive the logical tile (row, col) from each thread's linear
            # LDS byte position, using the HipKittens subtile layout:
            #   data[rows*cols] stored as contiguous 32x32 subtiles.
            # Then swizzle within the subtile and map to a global voffset.
            offsets = []
            for step in range_constexpr(N_LDS_STEPS):
                lds_byte = lane_id * BYTES_THREAD + wave_id * BYTES_WAVE + step * BYTES_WAVE * NUM_WAVES

                subtile_id = lds_byte // SUBTILE_BYTES
                st_row_idx = subtile_id // SUBTILES_PER_ROW
                st_col_idx = subtile_id % SUBTILES_PER_ROW
                local_byte = lds_byte % SUBTILE_BYTES
                local_row = local_byte // SUBTILE_ROW_BYTES
                local_col = local_byte % SUBTILE_ROW_BYTES // 2

                swz_byte = _swizzle32_byte_offset(local_row, local_col)
                swz_local_row = swz_byte // SUBTILE_ROW_BYTES
                swz_local_col = swz_byte % SUBTILE_ROW_BYTES // 2

                gl_row = st_row_idx * SUBTILE_ROWS + swz_local_row
                gl_col = st_col_idx * SUBTILE_COLS + swz_local_col
                offsets.append((gl_row * K + gl_col) * 2)
            return offsets

        def _load_lds(gl_src, lds_dst, k_offset, gl_offsets):
            lds_base = fx.Int32(fx.ptrtoint(lds_dst))
            for step in range_constexpr(N_LDS_STEPS):
                lds_ptr = buffer_ops.create_llvm_ptr(
                    lds_base + fx.Int32(wave_id * BYTES_WAVE + step * BYTES_WAVE * NUM_WAVES), address_space=3
                )
                rocdl.raw_ptr_buffer_load_lds(
                    gl_src,
                    lds_ptr,
                    fx.Int32(BYTES_THREAD),
                    fx.Int32(gl_offsets[step]),  # voffset
                    fx.Int32(k_offset),  # soffset
                    fx.Int32(0),
                    fx.Int32(0),
                )

        class BF16x8_shim:
            ir_type = Vec.make_type(8, fx.BFloat16)

        def _load_bf16x8(lds_ptr, byte_offset):
            # lds_ptr is bf16-typed; convert to byte pointer first so
            # the offset is interpreted as bytes, not elements.
            u8_base = fx.recast_iter(fx.Uint8, lds_ptr)
            return fx.ptr_load(u8_base + byte_offset, result_type=BF16x8_shim)

        def _lds_byte_offset(tile_row, tile_col):
            # Convert full-tile (row, col) in elements to an LDS byte offset
            # that accounts for subtile layout and swizzle.
            st_row_idx = tile_row // SUBTILE_ROWS
            st_col_idx = tile_col // SUBTILE_COLS
            local_row = tile_row % SUBTILE_ROWS
            local_col = tile_col % SUBTILE_COLS
            subtile_id = st_row_idx * SUBTILES_PER_ROW + st_col_idx
            subtile_base = subtile_id * SUBTILE_ROWS * SUBTILE_COLS * 2
            return subtile_base + _swizzle32_byte_offset(local_row, local_col)

        def _load_A_rt(lds_src, wave_idx, k_offset):
            frag = []
            for tile_row in range_constexpr(4):
                frag.append([])
                row = wave_idx * 128 + tile_row * 32 + lane_id % 32
                for tile_col in range_constexpr(2):
                    col = tile_col * 16 + (lane_id // 32) * 8 + k_offset
                    frag[tile_row].append(_load_bf16x8(lds_src, _lds_byte_offset(row, col)))
            return frag

        def _load_B_rt(lds_src, wave_idx, k_offset):
            frag = []
            for tile_row in range_constexpr(2):
                frag.append([])
                row = wave_idx * 64 + tile_row * 32 + lane_id % 32
                for tile_col in range_constexpr(2):
                    col = tile_col * 16 + (lane_id // 32) * 8 + k_offset
                    frag[tile_row].append(_load_bf16x8(lds_src, _lds_byte_offset(row, col)))
            return frag

        def _mfma(a, b, c):
            for i in range_constexpr(4):
                for j in range_constexpr(2):
                    for k in range_constexpr(2):
                        c[i][j] = rocdl.mfma_f32_32x32x16_bf16(
                            Accum_t,
                            [a[i][k], b[j][k], c[i][j], 0, 0, 0]
                        )
            return c

        def _store_rt():
            base_row = row * BLOCK_SIZE + wave_row * (BLOCK_SIZE // 2)
            base_col = col * BLOCK_SIZE + wave_col * (BLOCK_SIZE // 4)

            for i in range_constexpr(4):
                row_offset = i * 32
                for j in range_constexpr(2):
                    col_offset = j * 32
                    v_bf16 = c_frag[i][j].to(fx.BFloat16)
                    for group_idx in range_constexpr(4):
                        r = base_row + row_offset + group_idx * 8 + (lane_id // 32) * 4
                        c = base_col + col_offset + lane_id % 32
                        for el_idx in range_constexpr(4):
                            buffer_ops.buffer_store(
                                v_bf16[group_idx * 4 + el_idx], C_rsrc, fx.Int32((r + el_idx) * c_n + c)
                            )

        gl_offsets = _precompute_global_swizzle()
        cur, next = 0, 1

        _load_lds(A_rsrc, A_lds[cur], A_gl_offset + 0 * BLOCK_K * 2, gl_offsets)
        _load_lds(B_rsrc, B_lds[cur], B_gl_offset + 0 * BLOCK_K * 2, gl_offsets)
        wait_barrier(0)

        for k in range_constexpr(K_ITERS - 1):
            a_frag = _load_A_rt(A_lds[cur], wave_row, k_offset=0)
            _load_lds(A_rsrc, A_lds[next], A_gl_offset + (k + 1) * BLOCK_K * 2, gl_offsets)
            b_frag = _load_B_rt(B_lds[cur], wave_col, k_offset=0)
            _load_lds(B_rsrc, B_lds[next], B_gl_offset + (k + 1) * BLOCK_K * 2, gl_offsets)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c_frag = _mfma(a_frag, b_frag, c_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_frag = _load_A_rt(A_lds[cur], wave_row, k_offset=32)
            b_frag = _load_B_rt(B_lds[cur], wave_col, k_offset=32)
            wait_barrier(0)

            rocdl.s_setprio(1)
            c_frag = _mfma(a_frag, b_frag, c_frag)
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            # Swap cur with next
            cur ^= 1
            next ^= 1

        # Epilogue
        a_frag = _load_A_rt(A_lds[cur], wave_row, k_offset=0)
        b_frag = _load_B_rt(B_lds[cur], wave_col, k_offset=0)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c_frag = _mfma(a_frag, b_frag, c_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_frag = _load_A_rt(A_lds[cur], wave_row, k_offset=32)
        b_frag = _load_B_rt(B_lds[cur], wave_col, k_offset=32)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c_frag = _mfma(a_frag, b_frag, c_frag)
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        _store_rt()

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_SIZE) * ceildiv(c_n, BLOCK_SIZE)
        kernel_gemm(
            A,
            B_T,
            C,
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 2},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
