# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import ceildiv, divmod, wait_barrier, xcd_swizzle
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr.typing import T as _T

def compile_bf16_gemm_16x32_4w(
    *,
    K: int,
    use_xcd_remap: bool = False,
    use_fine_grained_iterleave: bool = True,
    use_sched_group_barrier: bool = False,
):
    # 256x256x64
    BLOCK_SIZE = 256
    BLOCK_K = 64

    assert K % BLOCK_K == 0

    NUM_TILES = K // BLOCK_K

    HALF_BLOCK_SIZE = 256 // 2
    BYTES_THREAD = 16
    NUM_WAVES = 4
    BYTES_WAVE = BYTES_THREAD * 64
    LDS_TILE_SIZE = HALF_BLOCK_SIZE * BLOCK_K * 2

    N_LDS_STEPS = LDS_TILE_SIZE // (BYTES_THREAD * NUM_WAVES * 64)

    @flyc.kernel
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

        # Each quadrant is 64x64 (4 row-tiles x 4 col-tiles of 16x16 MFMA atoms)
        # 2x2 quadrants -> 128x128 per wave
        c_frag = [
            [
                [[Accum_zero for _ in range_constexpr(4)] for _ in range_constexpr(4)]
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

        wave_row, wave_col = divmod(wave_id, 2)

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

        def _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, step):
            lds_base = fx.Int32(fx.ptrtoint(lds_dst))
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
            # Mirror the FP8 4wave kernel's S2R path: build a byte-strided view
            # at the swizzled offset and load through it. Keeping the address
            # math inside the layout op (vs. a raw ptr_load on u8_base+offset)
            # stops the backend from hoisting/spilling the per-read LDS address.
            # lds_ptr is bf16-typed, so recast to bytes before adding the offset.
            u8_base = fx.recast_iter(fx.Uint8, lds_ptr)
            ptr_off = fx.add_offset(u8_base, fx.make_int_tuple(byte_offset))
            view = fx.make_view(ptr_off, fx.make_layout(16, 1))
            return view.load().bitcast(fx.BFloat16)

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

        def _precompute_lds_swz(wave_idx):
            swz_offsets = []
            for tile_row in range_constexpr(4):
                swz_offsets.append([])
                row = wave_idx * 64 + tile_row * 16 + lane_id % 16
                for tile_col in range_constexpr(2):
                    col = tile_col * 32 + (lane_id // 16) * 8
                    swz_offsets[tile_row].append(_lds_byte_offset(row, col))
            return swz_offsets

        def _load_rt(lds_src, wave_idx):
            frag = []
            for tile_row in range_constexpr(4):
                frag.append([])
                row = wave_idx * 64 + tile_row * 16 + lane_id % 16
                for tile_col in range_constexpr(2):
                    col = tile_col * 32 + (lane_id // 16) * 8
                    frag[tile_row].append(_load_bf16x8(lds_src, _lds_byte_offset(row, col)))
            return frag

        def _load_one_rt(lds_src, row_idx, col_idx, wave_idx):
            row = wave_idx * 64 + row_idx * 16 + lane_id % 16
            col = col_idx * 32 + (lane_id // 16) * 8
            return _load_bf16x8(lds_src, _lds_byte_offset(row, col))

        def _mfma(a, b, c):
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    for k in range_constexpr(2):
                        c[i][j] = rocdl.mfma_f32_16x16x32_bf16(
                            Accum_t,
                            [a[i][k], b[j][k], c[i][j], 0, 0, 0]
                        )
            return c

        def _do_mfma(a, b, c):
            a_i32x4 = a.bitcast(fx.Int32)
            b_i32x4 = b.bitcast(fx.Int32)
            res_ty = _T.vec(4, _T.f32)
            res = _llvm.inline_asm(
                res_ty,
                [arith._to_raw(a_i32x4), arith._to_raw(b_i32x4), arith._to_raw(c)],
                "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $0",
                "=a,v,v,0",
                has_side_effects=True,
            )
            return Vec(res)

        def _mfma_one(a, b, c, i, j):
            for k in range_constexpr(2):
                c[i][j] = _do_mfma(a[i][k], b[j][k], c[i][j])
            return c

        def _store_rt():
            for row_idx in range_constexpr(2):
                base_row = row * BLOCK_SIZE + row_idx * HALF_BLOCK_SIZE + wave_row * 64
                for col_idx in range_constexpr(2):
                    base_col = col * BLOCK_SIZE + col_idx * HALF_BLOCK_SIZE + wave_col * 64
                    this_frag = c_frag[row_idx][col_idx]
                    for i in range_constexpr(4):
                        row_offset = i * 16
                        for j in range_constexpr(4):
                            col_offset = j * 16
                            v_bf16 = this_frag[i][j].to(fx.BFloat16)
                            r = base_row + row_offset + (lane_id // 16) * 4
                            c = base_col + col_offset + lane_id % 16
                            for el_idx in range_constexpr(4):
                                buffer_ops.buffer_store(
                                    v_bf16[el_idx], C_rsrc, fx.Int32((r + el_idx) * c_n + c)
                                )

        def _interleaved_block(gl_src, lds_dst, k_offset, gl_offsets, lds_src, wave_idx, a_rt, b_rt, c_rt):
            rt_dst = []

            c_rt = _mfma_one(a_rt, b_rt, c_rt, 0, 0)
            c_rt = _mfma_one(a_rt, b_rt, c_rt, 0, 1)

            # lds_swz = _precompute_lds_swz(wave_idx)
            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 0)
            rt_dst.append([_load_one_rt(lds_src, 0, 0, wave_idx)])
            # rt_dst.append([_load_one_rt(lds_src, lds_swz[0][0])])

            c_rt = _mfma_one(a_rt, b_rt, c_rt, 0, 2)

            rt_dst[0].append(_load_one_rt(lds_src, 0, 1, wave_idx))
            # rt_dst[0].append(_load_one_rt(lds_src, lds_swz[0][1]))

            c_rt = _mfma_one(a_rt, b_rt, c_rt, 0, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 1)

            rt_dst.append([_load_one_rt(lds_src, 1, 0, wave_idx)])
            # rt_dst.append([_load_one_rt(lds_src, lds_swz[1][0])])

            c_rt = _mfma_one(a_rt, b_rt, c_rt, 1, 0)
            c_rt = _mfma_one(a_rt, b_rt, c_rt, 1, 1)

            rt_dst[1].append(_load_one_rt(lds_src, 1, 1, wave_idx))
            # rt_dst[1].append(_load_one_rt(lds_src, lds_swz[1][1]))
            c_rt = _mfma_one(a_rt, b_rt, c_rt, 1, 2)
            c_rt = _mfma_one(a_rt, b_rt, c_rt, 1, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 2)

            rt_dst.append([_load_one_rt(lds_src, 2, 0, wave_idx)])
            # rt_dst.append([_load_one_rt(lds_src, lds_swz[2][0])])

            c_rt = _mfma_one(a_rt, b_rt, c_rt, 2, 0)
            c_rt = _mfma_one(a_rt, b_rt, c_rt, 2, 1)

            rt_dst[2].append(_load_one_rt(lds_src, 2, 1, wave_idx))
            # rt_dst[2].append(_load_one_rt(lds_src, lds_swz[2][1]))

            c_rt = _mfma_one(a_rt, b_rt, c_rt, 2, 2)
            c_rt = _mfma_one(a_rt, b_rt, c_rt, 2, 3)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 3)
            rt_dst.append([_load_one_rt(lds_src, 3, 0, wave_idx)])
            # rt_dst.append([_load_one_rt(lds_src, lds_swz[3][0])])

            c_rt = _mfma_one(a_rt, b_rt, c_rt, 3, 0)
            c_rt = _mfma_one(a_rt, b_rt, c_rt, 3, 1)

            rt_dst[3].append(_load_one_rt(lds_src, 3, 1, wave_idx))
            # rt_dst[3].append(_load_one_rt(lds_src, lds_swz[3][1]))

            c_rt = _mfma_one(a_rt, b_rt, c_rt, 3, 2)
            c_rt = _mfma_one(a_rt, b_rt, c_rt, 3, 3)

            return c_rt, rt_dst

        def _block_scheduler():
            # Describe the desired interleave to the backend scheduler instead of
            # hand-ordering. Counts must match the ops emitted in _ordered_block:
            #   N_LDS_STEPS VMEM loads, 8 DS reads (4 rows x 2 cols), 32 MFMAs.
            for _ in range_constexpr(N_LDS_STEPS):
                rocdl.sched_vmem(1)
            for _ in range_constexpr(8):
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(4)
            rocdl.sched_barrier(0)

        def _ordered_block(gl_src, lds_dst, k_offset, gl_offsets, lds_src, wave_idx, a_rt, b_rt, c_rt):
            # Emit each instruction stream in clean, natural order. The streams are
            # independent (VMEM writes a future tile, DSRD reads a different LDS
            # buffer into next-iter fragments, MFMA consumes already-loaded a/b),
            # so the sched_group_barrier schedule below is free to interleave them.
            rocdl.sched_barrier(0)
            for step in range_constexpr(N_LDS_STEPS):
                _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, step)
            rt_dst = _load_rt(lds_src, wave_idx)
            c_rt = _mfma(a_rt, b_rt, c_rt)
            _block_scheduler()
            return c_rt, rt_dst

        def _compute_block(gl_src, lds_dst, k_offset, gl_offsets, lds_src, wave_idx, a_rt, b_rt, c_rt):
            if const_expr(use_sched_group_barrier):
                c_rt, rt_dst = _ordered_block(
                    gl_src, lds_dst, k_offset, gl_offsets, lds_src, wave_idx, a_rt, b_rt, c_rt
                )
            elif const_expr(use_fine_grained_iterleave):
                c_rt, rt_dst = _interleaved_block(
                    gl_src, lds_dst, k_offset, gl_offsets, lds_src, wave_idx, a_rt, b_rt, c_rt
                )
            else:
                _load_lds(gl_src, lds_dst, k_offset, gl_offsets)
                rt_dst = _load_rt(lds_src, wave_idx)
                c_rt = _mfma(a_rt, b_rt, c_rt)
            return c_rt, rt_dst

        gl_offsets = _precompute_global_swizzle()
        cur, next = 0, 1

        _load_lds(A_rsrc, A_lds[cur][0], A0_gl_offset + 0 * BLOCK_K * 2, gl_offsets)
        _load_lds(B_rsrc, B_lds[cur][0], B0_gl_offset + 0 * BLOCK_K * 2, gl_offsets)
        _load_lds(B_rsrc, B_lds[cur][1], B128_gl_offset + 0 * BLOCK_K * 2, gl_offsets)
        _load_lds(A_rsrc, A_lds[cur][1], A128_gl_offset + 0 * BLOCK_K * 2, gl_offsets)

        _load_lds(A_rsrc, A_lds[next][0], A0_gl_offset + 1 * BLOCK_K * 2, gl_offsets)
        _load_lds(B_rsrc, B_lds[next][0], B0_gl_offset + 1 * BLOCK_K * 2, gl_offsets)
        _load_lds(B_rsrc, B_lds[next][1], B128_gl_offset + 1 * BLOCK_K * 2, gl_offsets)
        _load_lds(A_rsrc, A_lds[next][1], A128_gl_offset + 1 * BLOCK_K * 2, gl_offsets)

        # Each `_load_lds` issues `N_LDS_STEPS` global-to-lds ops
        wait_barrier(3 * N_LDS_STEPS + 3 * N_LDS_STEPS)

        a0_frag = _load_rt(A_lds[cur][0], wave_row)
        b0_frag = _load_rt(B_lds[cur][0], wave_col)

        for k in range_constexpr(NUM_TILES - 2):
            wait_barrier(2 * N_LDS_STEPS + 2 * N_LDS_STEPS)

            c_frag[0][0], b1_frag = _compute_block(
                A_rsrc,
                A_lds[cur][0],
                A0_gl_offset + (k + 2) * BLOCK_K * 2,
                gl_offsets,
                B_lds[cur][1],
                wave_col,
                a0_frag,
                b0_frag,
                c_frag[0][0]
            )
            c_frag[0][1], a1_frag = _compute_block(
                B_rsrc,
                B_lds[cur][0],
                B0_gl_offset + (k + 2) * BLOCK_K * 2,
                gl_offsets,
                A_lds[cur][1],
                wave_row,
                a0_frag,
                b1_frag,
                c_frag[0][1]
            )

            wait_barrier(2 * N_LDS_STEPS + 2 * N_LDS_STEPS)

            c_frag[1][0], a0_frag = _compute_block(
                B_rsrc,
                B_lds[cur][1],
                B128_gl_offset + (k + 2) * BLOCK_K * 2,
                gl_offsets,
                A_lds[next][0],
                wave_row,
                a1_frag,
                b0_frag,
                c_frag[1][0]
            )
            c_frag[1][1], b0_frag = _compute_block(
                A_rsrc,
                A_lds[cur][1],
                A128_gl_offset + (k + 2) * BLOCK_K * 2,
                gl_offsets,
                B_lds[next][0],
                wave_col,
                a1_frag,
                b1_frag,
                c_frag[1][1]
            )

            cur ^= 1
            next ^= 1

        wait_barrier(2 * N_LDS_STEPS + 2 * N_LDS_STEPS)
        b1_frag = _load_rt(B_lds[cur][1], wave_col)
        c_frag[0][0] = _mfma(a0_frag, b0_frag, c_frag[0][0])
        a1_frag = _load_rt(A_lds[cur][1], wave_row)
        c_frag[0][1] = _mfma(a0_frag, b1_frag, c_frag[0][1])

        wait_barrier(1 * N_LDS_STEPS + 1 * N_LDS_STEPS)
        a0_frag = _load_rt(A_lds[next][0], wave_row)
        c_frag[1][0] = _mfma(a1_frag, b0_frag, c_frag[1][0])
        b0_frag = _load_rt(B_lds[next][0], wave_col)
        c_frag[1][1] = _mfma(a1_frag, b1_frag, c_frag[1][1])

        cur ^= 1
        next ^= 1

        wait_barrier(0)
        b1_frag = _load_rt(B_lds[cur][1], wave_col)
        a1_frag = _load_rt(A_lds[cur][1], wave_row)

        c_frag[0][0] = _mfma(a0_frag, b0_frag, c_frag[0][0])
        c_frag[0][1] = _mfma(a0_frag, b1_frag, c_frag[0][1])
        c_frag[1][0] = _mfma(a1_frag, b0_frag, c_frag[1][0])
        c_frag[1][1] = _mfma(a1_frag, b1_frag, c_frag[1][1])

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
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
