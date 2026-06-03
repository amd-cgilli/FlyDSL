# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import ceildiv, divmod, wait_barrier, xcd_swizzle


def compile_bf16_gemm_16x32(
    *,
    K: int,
    use_xcd_remap: bool = False
):
    # 256x256x64
    BLOCK_SIZE = 256
    BLOCK_K = 64

    assert K % BLOCK_K == 0

    NUM_TILES = K // BLOCK_K
    K_ITERS = (NUM_TILES - 2) // 2

    HALF_BLOCK_SIZE = 256 // 2
    BYTES_THREAD = 16
    NUM_WAVES = 8
    BYTES_WAVE = BYTES_THREAD * 64
    LDS_TILE_SIZE = HALF_BLOCK_SIZE * BLOCK_K * 2

    N_LDS_STEPS = LDS_TILE_SIZE // (BYTES_THREAD * 512)

    @flyc.kernel(known_block_size=[512, 1, 1])
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        # MFMA atom is 16x16x32
        #   - each lane holds 8 BF16 values of A/B
        #   - each lane holds 4 F32 values of C (each lane holding 4 consecutive elements in the column)
        Accum_t = Vec.make_type(4, fx.Float32)
        Accum_zero = Vec.filled(4, 0.0, fx.Float32)

        # Each quadrant is 64x32 (4 row-tiles x 2 col-tiles of 16x16 MFMA atoms)
        # 2x2 quadrants -> 128x64 per wave
        c_frag = [
            [
                [[Accum_zero for _ in range_constexpr(2)] for _ in range_constexpr(4)]
                for _ in range_constexpr(2)
            ]
            for _ in range_constexpr(2)
        ]

        lds_alloc = fx.SharedAllocator()
        A_lds = [
            [lds_alloc.allocate(fx.Array[fx.BFloat16, HALF_BLOCK_SIZE * BLOCK_K, 16]).peek().ptr for _ in range_constexpr(2)]
            for _ in range_constexpr(2)
        ]
        B_lds = [
            [lds_alloc.allocate(fx.Array[fx.BFloat16, HALF_BLOCK_SIZE * BLOCK_K, 16]).peek().ptr for _ in range_constexpr(2)]
            for _ in range_constexpr(2)
        ]
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        n_blocks = ceildiv(c_n, BLOCK_SIZE)
        if const_expr(use_xcd_remap):
            row, col = xcd_swizzle(ceildiv(c_m, BLOCK_SIZE), n_blocks)
        else:
            row, col = divmod(fx.block_idx.x, n_blocks)

        wave_row, wave_col = divmod(wave_id, 4)

        A0_gl_offset = (row * BLOCK_SIZE + 0) * K * 2
        A128_gl_offset = (row * BLOCK_SIZE + HALF_BLOCK_SIZE) * K * 2
        B0_gl_offset = (col * BLOCK_SIZE + 0) * K * 2
        B128_gl_offset = (col * BLOCK_SIZE + HALF_BLOCK_SIZE) * K * 2

        A_rsrc = buffer_ops.create_buffer_resource(A)
        B_rsrc = buffer_ops.create_buffer_resource(B_T)
        C_rsrc = buffer_ops.create_buffer_resource(C)

        SUBTILE_ROWS = 16
        SUBTILE_COLS = 32
        SUBTILES_PER_ROW = BLOCK_K // SUBTILE_COLS

        def _swizzle16_byte_offset(row, col):
            # st_16x32 swizzle from HipKittens: takes (row, col) within a 16x32
            # subtile (both in elements), returns a byte offset within that subtile.
            offset = (row * SUBTILE_COLS + col) * 2
            swz = ((offset % 1024) >> 9) << 5
            return offset ^ swz

        SUBTILE_BYTES = SUBTILE_ROWS * SUBTILE_COLS * 2
        SUBTILE_ROW_BYTES = SUBTILE_COLS * 2

        def _precompute_global_swizzle():
            offsets = []
            for step in range_constexpr(N_LDS_STEPS):
                lds_byte = lane_id * BYTES_THREAD + wave_id * BYTES_WAVE + step * BYTES_WAVE * NUM_WAVES

                subtile_id = lds_byte // SUBTILE_BYTES
                st_row_idx = subtile_id // SUBTILES_PER_ROW
                st_col_idx = subtile_id % SUBTILES_PER_ROW
                local_byte = lds_byte % SUBTILE_BYTES
                local_row = local_byte // SUBTILE_ROW_BYTES
                local_col = local_byte % SUBTILE_ROW_BYTES // 2

                swz_byte = _swizzle16_byte_offset(local_row, local_col)
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
            return subtile_base + _swizzle16_byte_offset(local_row, local_col)

        def _load_A_rt(lds_src, wave_idx):
            frag = []
            for tile_row in range_constexpr(4):
                frag.append([])
                row = wave_idx * 64 + tile_row * 16 + lane_id % 16
                for tile_col in range_constexpr(2):
                    col = tile_col * 32 + (lane_id // 16) * 8
                    frag[tile_row].append(_load_bf16x8(lds_src, _lds_byte_offset(row, col)))
            return frag

        def _load_B_rt(lds_src, wave_idx):
            frag = []
            for tile_row in range_constexpr(2):
                frag.append([])
                row = wave_idx * 32 + tile_row * 16 + lane_id % 16
                for tile_col in range_constexpr(2):
                    col = tile_col * 32 + (lane_id // 16) * 8
                    frag[tile_row].append(_load_bf16x8(lds_src, _lds_byte_offset(row, col)))
            return frag

        def _mfma(a, b, c):
            for i in range_constexpr(4):
                for j in range_constexpr(2):
                    for k in range_constexpr(2):
                        c[i][j] = rocdl.mfma_f32_16x16x32_bf16(
                            Accum_t,
                            [a[i][k], b[j][k], c[i][j], 0, 0, 0]
                        )
            return c

        def _store_rt(row_idx, col_idx):
            base_row = row * BLOCK_SIZE + row_idx * HALF_BLOCK_SIZE + wave_row * 64
            base_col = col * BLOCK_SIZE + col_idx * HALF_BLOCK_SIZE + wave_col * 32

            this_frag = c_frag[row_idx][col_idx]
            for i in range_constexpr(4):
                row_offset = i * 16
                for j in range_constexpr(2):
                    col_offset = j * 16
                    v_bf16 = this_frag[i][j].to(fx.BFloat16)
                    r = base_row + row_offset + (lane_id // 16) * 4
                    c = base_col + col_offset + lane_id % 16
                    for el_idx in range_constexpr(4):
                        buffer_ops.buffer_store(
                            v_bf16[el_idx], C_rsrc, fx.Int32((r + el_idx) * c_n + c)
                        )

        gl_offsets = _precompute_global_swizzle()
        cur, next = 0, 1

        _load_lds(A_rsrc, A_lds[cur][0], A0_gl_offset + 0 * BLOCK_K * 2, gl_offsets)
        _load_lds(B_rsrc, B_lds[cur][0], B0_gl_offset + 0 * BLOCK_K * 2, gl_offsets)
        _load_lds(A_rsrc, A_lds[cur][1], A128_gl_offset + 0 * BLOCK_K * 2, gl_offsets)
        _load_lds(B_rsrc, B_lds[cur][1], B128_gl_offset + 0 * BLOCK_K * 2, gl_offsets)

        if wave_row == 1:
            rocdl.s_barrier()

        # Wait for A/B first half
        wait_barrier(4)

        _load_lds(B_rsrc, B_lds[next][0], B0_gl_offset + 1 * BLOCK_K * 2, gl_offsets)
        _load_lds(A_rsrc, A_lds[next][0], A0_gl_offset + 1 * BLOCK_K * 2, gl_offsets)
        _load_lds(B_rsrc, B_lds[next][1], B128_gl_offset + 1 * BLOCK_K * 2, gl_offsets)

        # Wait for the second half of A/B
        wait_barrier(6)

        for k2 in range_constexpr(K_ITERS):
            # Each step of the main loop handles tile 2k and 2k+1
            k = 2*k2
            b0_frag = _load_B_rt(B_lds[0][0], wave_col)
            a_frag = _load_A_rt(A_lds[0][0], wave_row)

            _load_lds(A_rsrc, A_lds[1][1], A128_gl_offset + (k + 1) * BLOCK_K * 2, gl_offsets)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c_frag[0][0] = _mfma(a_frag, b0_frag, c_frag[0][0])
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = _load_B_rt(B_lds[0][1], wave_col)
            _load_lds(B_rsrc, B_lds[0][0], B0_gl_offset + (k + 2) * BLOCK_K * 2, gl_offsets)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c_frag[0][1] = _mfma(a_frag, b1_frag, c_frag[0][1])
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_frag = _load_A_rt(A_lds[0][1], wave_row)
            _load_lds(A_rsrc, A_lds[0][0], A0_gl_offset + (k + 2) * BLOCK_K * 2, gl_offsets)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c_frag[1][0] = _mfma(a_frag, b0_frag, c_frag[1][0])
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b0_frag = _load_B_rt(B_lds[1][0], wave_col)
            _load_lds(B_rsrc, B_lds[0][1], B128_gl_offset + (k + 2) * BLOCK_K * 2, gl_offsets)
            wait_barrier(6)

            rocdl.s_setprio(1)
            c_frag[1][1] = _mfma(a_frag, b1_frag, c_frag[1][1])
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_frag = _load_A_rt(A_lds[1][0], wave_row)
            _load_lds(A_rsrc, A_lds[0][1], A128_gl_offset + (k + 2) * BLOCK_K * 2, gl_offsets)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c_frag[0][0] = _mfma(a_frag, b0_frag, c_frag[0][0])
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            b1_frag = _load_B_rt(B_lds[1][1], wave_col)
            _load_lds(B_rsrc, B_lds[1][0], B0_gl_offset + (k + 3) * BLOCK_K * 2, gl_offsets)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c_frag[0][1] = _mfma(a_frag, b1_frag, c_frag[0][1])
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            a_frag = _load_A_rt(A_lds[1][1], wave_row)
            _load_lds(A_rsrc, A_lds[1][0], A0_gl_offset + (k + 3) * BLOCK_K * 2, gl_offsets)
            rocdl.s_barrier()

            rocdl.s_setprio(1)
            c_frag[1][0] = _mfma(a_frag, b0_frag, c_frag[1][0])
            rocdl.s_setprio(0)
            rocdl.s_barrier()

            _load_lds(B_rsrc, B_lds[1][1], B128_gl_offset + (k + 3) * BLOCK_K * 2, gl_offsets)
            wait_barrier(6)

            rocdl.s_setprio(1)
            c_frag[1][1] = _mfma(a_frag, b1_frag, c_frag[1][1])
            rocdl.s_setprio(0)
            rocdl.s_barrier()

        # Epilogue
        # k = num_iters - 2
        k = NUM_TILES - 2
        b0_frag = _load_B_rt(B_lds[cur][0], wave_col)
        a_frag = _load_A_rt(A_lds[cur][0], wave_row)
        _load_lds(A_rsrc, A_lds[next][1], A128_gl_offset + (k + 1) * BLOCK_K * 2, gl_offsets)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c_frag[0][0] = _mfma(a_frag, b0_frag, c_frag[0][0])
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = _load_B_rt(B_lds[cur][1], wave_col)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c_frag[0][1] = _mfma(a_frag, b1_frag, c_frag[0][1])
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_frag = _load_A_rt(A_lds[cur][1], wave_row)
        wait_barrier(4)

        rocdl.s_setprio(1)
        c_frag[1][0] = _mfma(a_frag, b0_frag, c_frag[1][0])
        c_frag[1][1] = _mfma(a_frag, b1_frag, c_frag[1][1])
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        cur ^= 1
        next ^= 1

        # k = num_iters - 1
        b0_frag = _load_B_rt(B_lds[cur][0], wave_col)
        a_frag = _load_A_rt(A_lds[cur][0], wave_row)
        wait_barrier(2)

        rocdl.s_setprio(1)
        c_frag[0][0] = _mfma(a_frag, b0_frag, c_frag[0][0])
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        b1_frag = _load_B_rt(B_lds[cur][1], wave_col)
        wait_barrier(0)

        rocdl.s_setprio(1)
        c_frag[0][1] = _mfma(a_frag, b1_frag, c_frag[0][1])
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        a_frag = _load_A_rt(A_lds[cur][1], wave_row)
        rocdl.s_barrier()

        rocdl.s_setprio(1)
        c_frag[1][0] = _mfma(a_frag, b0_frag, c_frag[1][0])
        c_frag[1][1] = _mfma(a_frag, b1_frag, c_frag[1][1])
        rocdl.s_setprio(0)
        rocdl.s_barrier()

        if wave_row == 0:
            rocdl.s_barrier()

        _store_rt(0, 0)
        _store_rt(0, 1)
        _store_rt(1, 0)
        _store_rt(1, 1)

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
            value_attrs={"rocdl.waves_per_eu": 2, "rocdl.flat_work_group_size": "512,512"},
        ).launch(grid=(grid_x, 1, 1), block=(512, 1, 1), stream=stream)

    return launch_gemm
