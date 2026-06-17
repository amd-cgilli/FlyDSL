import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from kernels.fp8_gemm_utils import ceildiv


def compile_mxfp8_gemm_4w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256
):
    BLOCK_K = 128
    assert BLOCK_M in [64, 128, 256] and BLOCK_N in [64, 128, 256]
    assert K % BLOCK_K == 0

    K_ITERS = K // BLOCK_K
    SCALE_K = K // 32  # E8M0 block-scales per row (one byte per 32 K-elements)

    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16

    # Rows packed into one scale word = the rows a single wave-fragment covers.
    # opsel (0..N_TILES-1) selects the byte, so each wave reads its own word and
    # opsel stays a compile-time immediate for any block size.
    GROUP_A = N_TILES_A * 16
    GROUP_B = N_TILES_B * 16

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    MAX_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

    @flyc.kernel
    def kernel_gemm(
        A: fx.Pointer,
        B_T: fx.Pointer,
        C: fx.Pointer,
        A_scales: fx.Pointer, # preshuffled E8M0 scales, [ceil(m/GROUP_A), 16, K//32] uint32
        B_scales: fx.Pointer, # preshuffled E8M0 scales, [ceil(n/GROUP_B), 16, K//32] uint32
        c_m: fx.Int32,
        c_n: fx.Int32,
    ):
        MfmaAccumType_t = Vec.make_type(4, fx.Float32)
        RT_C_i = Vec.filled(4, 0.0, fx.Float32)

        lds_alloc = fx.SharedAllocator()
        A_lds = [
            [
                lds_alloc.allocate(fx.Array[fx.Float8E4M3FN, LDS_BLOCK_M * 128, 16]).peek().ptr
                for _ in range_constexpr(2)
            ]
            for _ in range_constexpr(2)
        ]

        B_lds = [
            [
                lds_alloc.allocate(fx.Array[fx.Float8E4M3FN,LDS_BLOCK_N * 128, 16]).peek().ptr
                for _ in range_constexpr(2)
            ]
            for _ in range_constexpr(2)
        ]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        n_blocks = ceildiv(c_n, BLOCK_N)
        tile_i = fx.block_idx.x // n_blocks
        tile_j = fx.block_idx.x % n_blocks
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A0_gl_offset = (tile_i * BLOCK_M) * K
        A128_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K
        B0_gl_offset = (tile_j * BLOCK_N) * K
        B128_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K

        a_row_h0 = wave_i * GROUP_A
        a_row_h1 = LDS_BLOCK_M + a_row_h0
        b_row_h0 = wave_j * GROUP_B
        b_row_h1 = LDS_BLOCK_N + b_row_h0

        A_rsrc = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=c_m * K)
        B_rsrc = buffer_ops.create_buffer_resource(B_T, max_size=False, num_records_bytes=c_n * K)
        C_rsrc = buffer_ops.create_buffer_resource(C, max_size=False, num_records_bytes=c_m * c_n * 2)
        sa_groups = ceildiv(c_m, GROUP_A)
        sb_groups = ceildiv(c_n, GROUP_B)
        A_scales_rsrc = buffer_ops.create_buffer_resource(
            A_scales, max_size=False, num_records_bytes=sa_groups * 16 * SCALE_K * 4
        )
        B_scales_rsrc = buffer_ops.create_buffer_resource(
            B_scales, max_size=False, num_records_bytes=sb_groups * 16 * SCALE_K * 4
        )

        def _swizzle_128(row, col):
            offset = row * 128 + col
            swizzle = ((offset % (16 * 128)) >> 8) << 4
            swizzled_offset = offset ^ swizzle
            return swizzled_offset // 128, swizzled_offset % 128

        def _compute_global_swizzle():
            offsets = []
            for round in range_constexpr(MAX_LDS_ROUNDS):
                row = lane_id // 8 + wave_id * 8 + round * 32
                col = (lane_id % 8) * 16
                a, b = _swizzle_128(row, col)
                offsets.append(a * K + b)
            return offsets

        def _load_lds(gl_src, lds_dst, k_offset, gl_offsets, n_steps):
            lds_base = fx.Int32(fx.ptrtoint(lds_dst))
            for step in range_constexpr(n_steps):
                # Each (wave, step) writes 64 lanes * 16 B = 1024 B. The 4 waves of a
                # step occupy 4 * 1024 B, so consecutive steps must stride by 4096 B;
                # using n_steps*1024 overlaps the wave regions for n_steps < 4.
                lds_ptr = buffer_ops.create_llvm_ptr(
                    lds_base + fx.Int32(wave_id * 1024 + step * 4096), address_space=3
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

        def _pack_i32x42_i32x8(lo, hi):
            # Pack 2 i32x4 as i32x8
            return lo.shuffle(hi, list(range(8)))

        # Shim object: `ptr_load` will call `result_type.ir_type`
        # because it expects a FlyDSL object, not an MLIR value.
        # This will probably be fixed in some future version.
        class I32x4:
            ir_type = Vec.make_type(4, fx.Int32)

        def _lds_load_i32x4(lds_ptr, elem_offset):
            i32_ptr = fx.recast_iter(fx.Uint8, lds_ptr + elem_offset)
            return fx.ptr_load(i32_ptr, result_type=I32x4)

        def _load_rt(lds_src, wave_idx, n_steps):
            # Load a 64x128 fragment of A/B from LDS to registers
            # Each 16x128 fragment requires 2 i32x4 (2 ds_read_b128)
            frag = []
            for i in range_constexpr(n_steps):
                row = wave_idx * (n_steps * 16) + i * 16 + lane_id % 16
                halves = []
                for step in range_constexpr(2):
                    col = (lane_id // 16) * 16 + step * 64
                    row_swz, col_swz = _swizzle_128(row, col)
                    halves.append(_lds_load_i32x4(lds_src, row_swz * 128 + col_swz))
                frag.append(_pack_i32x42_i32x8(halves[0], halves[1]))  # i32x8
            return frag

        def _store_rt(c_frag, base_row, base_col):
            for ti in range_constexpr(N_TILES_A):
                row = base_row + ti * 16 + (lane_id // 16) * 4
                for tj in range_constexpr(N_TILES_B):
                    col = base_col + tj * 16 + lane_id % 16
                    vec_bf16 = Vec(c_frag[ti][tj]).to(fx.BFloat16)
                    for i in range_constexpr(4):
                        buffer_ops.buffer_store(
                            vec_bf16[i], C_rsrc, fx.Int32((row + i) * c_n + col)
                        )

        def _mfma_ABt_all(a, b, c, a_scales, b_scales):
            for i in range_constexpr(N_TILES_A):
                for j in range_constexpr(N_TILES_B):
                    c[i][j] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        MfmaAccumType_t,
                        [a[i], b[j], c[i][j], 0, 0, i, a_scales, j, b_scales],
                    )
            return c

        def _wait_barrier(count):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({count})\n\ts_barrier",
                constraints="",
                has_side_effects=True,
            )

        def _load_scales(gl_src, frag_base, group, k):
            # Load the packed scale word the scaled-MFMA consumes via opsel, with a
            # single uint32 load from the host-preshuffled scale tensor
            # [ceil(dim/GROUP), 16, K//32] uint32. The host shuffle gathers the
            # GROUP rows of one wave-fragment {frag_base + g*16 + r16 : g in
            # 0..N_TILES-1} into one little-endian word (byte g selected by
            # opsel=g) and pads partial groups with the E8M0 identity 0x7F (= 1.0),
            # so no per-byte gather or OOB guard is needed.
            #
            # frag_base is the wave-fragment's first global row (a multiple of
            # `group`); the lane's r16 = lane%16 and k_sub = (lane//16) select the
            # row within each 16-row sub-tile and the K-subblock. Flattened uint32
            # offset is grp*(16*SCALE_K) + r16*SCALE_K + (k*4 + k_sub).
            grp = frag_base // group
            r16 = lane_id % 16
            k_sub = lane_id // 16
            col = k * 4 + k_sub
            offset = grp * (16 * SCALE_K) + r16 * SCALE_K + col
            return buffer_ops.buffer_load(gl_src, fx.Int32(offset), vec_width=1, dtype=fx.Uint32)

        c00_frag = [[RT_C_i for _ in range_constexpr(N_TILES_B)] for _ in range_constexpr(N_TILES_A)]
        c01_frag = [[RT_C_i for _ in range_constexpr(N_TILES_B)] for _ in range_constexpr(N_TILES_A)]
        c10_frag = [[RT_C_i for _ in range_constexpr(N_TILES_B)] for _ in range_constexpr(N_TILES_A)]
        c11_frag = [[RT_C_i for _ in range_constexpr(N_TILES_B)] for _ in range_constexpr(N_TILES_A)]

        global_offsets = _compute_global_swizzle()

        # Prologue: pre-load A/B cur
        _load_lds(A_rsrc, A_lds[0][0], A0_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_A)
        _load_lds(B_rsrc, B_lds[0][0], B0_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(B_rsrc, B_lds[0][1], B128_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(A_rsrc, A_lds[0][1], A128_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_A)

        # Issue load for next tile
        _load_lds(A_rsrc, A_lds[1][0], A0_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_A)
        _load_lds(B_rsrc, B_lds[1][0], B0_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(B_rsrc, B_lds[1][1], B128_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(A_rsrc, A_lds[1][1], A128_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_A)

        _wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))

        a0_frag = _load_rt(A_lds[0][0], wave_i, N_TILES_A)

        _wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))

        b0_frag = _load_rt(B_lds[0][0], wave_j, N_TILES_B)

        cur, next = 0, 1
        for k in range_constexpr(K_ITERS - 2):
            sa_h0 = _load_scales(A_scales_rsrc, tile_i * BLOCK_M + a_row_h0, GROUP_A, k)
            sb_h0 = _load_scales(B_scales_rsrc, tile_j * BLOCK_N + b_row_h0, GROUP_B, k)
            sa_h1 = _load_scales(A_scales_rsrc, tile_i * BLOCK_M + a_row_h1, GROUP_A, k)
            sb_h1 = _load_scales(B_scales_rsrc, tile_j * BLOCK_N + b_row_h1, GROUP_B, k)

            _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            b1_frag = _load_rt(B_lds[cur][1], wave_j, N_TILES_B)
            _load_lds(A_rsrc, A_lds[cur][0], A0_gl_offset + (k + 2) * BLOCK_K, global_offsets, N_TILES_A)
            _load_lds(B_rsrc, B_lds[cur][0], B0_gl_offset + (k + 2) * BLOCK_K, global_offsets, N_TILES_B)
            _mfma_ABt_all(a0_frag, b0_frag, c00_frag, sa_h0, sb_h0)

            a1_frag = _load_rt(A_lds[cur][1], wave_i, N_TILES_A)
            _load_lds(B_rsrc, B_lds[cur][1], B128_gl_offset + (k + 2) * BLOCK_K, global_offsets, N_TILES_B)
            _load_lds(A_rsrc, A_lds[cur][1], A128_gl_offset + (k + 2) * BLOCK_K, global_offsets, N_TILES_A)
            _mfma_ABt_all(a0_frag, b1_frag, c01_frag, sa_h0, sb_h1)

            _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            a0_frag = _load_rt(A_lds[next][0], wave_i, N_TILES_A)
            _mfma_ABt_all(a1_frag, b0_frag, c10_frag, sa_h1, sb_h0)

            b0_frag = _load_rt(B_lds[next][0], wave_j, N_TILES_B)
            _mfma_ABt_all(a1_frag, b1_frag, c11_frag, sa_h1, sb_h1)

            # Swap cur and next
            cur ^= 1
            next ^= 1

        # step k = k_iters - 2
        _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))
        k = K_ITERS - 2
        sa_h0 = _load_scales(A_scales_rsrc, tile_i * BLOCK_M + a_row_h0, GROUP_A, k)
        sb_h0 = _load_scales(B_scales_rsrc, tile_j * BLOCK_N + b_row_h0, GROUP_B, k)
        sa_h1 = _load_scales(A_scales_rsrc, tile_i * BLOCK_M + a_row_h1, GROUP_A, k)
        sb_h1 = _load_scales(B_scales_rsrc, tile_j * BLOCK_N + b_row_h1, GROUP_B, k)

        b1_frag = _load_rt(B_lds[cur][1], wave_j, N_TILES_B)
        _mfma_ABt_all(a0_frag, b0_frag, c00_frag, sa_h0, sb_h0)

        a1_frag = _load_rt(A_lds[cur][1], wave_i, N_TILES_A)
        _mfma_ABt_all(a0_frag, b1_frag, c01_frag, sa_h0, sb_h1)

        _wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))

        a0_frag = _load_rt(A_lds[next][0], wave_i, N_TILES_A)
        _mfma_ABt_all(a1_frag, b0_frag, c10_frag, sa_h1, sb_h0)
        b0_frag = _load_rt(B_lds[next][0], wave_j, N_TILES_B)
        _mfma_ABt_all(a1_frag, b1_frag, c11_frag, sa_h1, sb_h1)

        # Swap cur and next
        cur ^= 1
        next ^= 1

        # step k = k_iters - 1
        _wait_barrier(0)
        k = K_ITERS - 1
        sa_h0 = _load_scales(A_scales_rsrc, tile_i * BLOCK_M + a_row_h0, GROUP_A, k)
        sb_h0 = _load_scales(B_scales_rsrc, tile_j * BLOCK_N + b_row_h0, GROUP_B, k)
        sa_h1 = _load_scales(A_scales_rsrc, tile_i * BLOCK_M + a_row_h1, GROUP_A, k)
        sb_h1 = _load_scales(B_scales_rsrc, tile_j * BLOCK_N + b_row_h1, GROUP_B, k)

        b1_frag = _load_rt(B_lds[cur][1], wave_j, N_TILES_B)
        _mfma_ABt_all(a0_frag, b0_frag, c00_frag, sa_h0, sb_h0)

        a1_frag = _load_rt(A_lds[cur][1], wave_i, N_TILES_A)
        _mfma_ABt_all(a0_frag, b1_frag, c01_frag, sa_h0, sb_h1)

        _mfma_ABt_all(a1_frag, b0_frag, c10_frag, sa_h1, sb_h0)
        _mfma_ABt_all(a1_frag, b1_frag, c11_frag, sa_h1, sb_h1)

        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)

        _store_rt(c00_frag, base_row + 0, base_col + 0)
        _store_rt(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        _store_rt(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        _store_rt(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm(A: fx.Pointer, B_T: fx.Pointer, C: fx.Pointer, A_scales: fx.Pointer, B_scales: fx.Pointer, c_m: fx.Int32, c_n: fx.Int32, stream: fx.Stream):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scales,
            B_scales,
            c_m, c_n,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
