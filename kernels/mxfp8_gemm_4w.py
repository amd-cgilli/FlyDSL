import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec


def compile_mxfp8_gemm_4w(
    *,
    M: int,
    N: int,
    K: int,
):
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K

    assert N % BLOCK_N == 0
    assert M % BLOCK_M == 0
    assert K % BLOCK_K == 0

    @flyc.kernel
    def kernel_gemm(
        A: fx.Pointer,
        B_T: fx.Pointer,
        C: fx.Pointer,
        A_scales: fx.Pointer, # [M, K//32]
        B_scales: fx.Pointer, # [N, K//32]
    ):
        MfmaAccumType_t = Vec.make_type(4, fx.Float32)
        RT_C_i = Vec.filled(4, 0.0, fx.Float32)

        lds_alloc = fx.SharedAllocator()
        A_lds = [
            [
                lds_alloc.allocate(fx.Array[fx.Float8E4M3FN, 128 * 128, 16]).peek().ptr
                for _ in range_constexpr(2)
            ]
            for _ in range_constexpr(2)
        ]

        B_lds = [
            [
                lds_alloc.allocate(fx.Array[fx.Float8E4M3FN, 128 * 128, 16]).peek().ptr
                for _ in range_constexpr(2)
            ]
            for _ in range_constexpr(2)
        ]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        tile_i = fx.block_idx.x // N_BLOCKS
        tile_j = fx.block_idx.x % N_BLOCKS
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A0_gl_offset = (tile_i * BLOCK_M) * K
        A128_gl_offset = (tile_i * BLOCK_M + 128) * K
        B0_gl_offset = (tile_j * BLOCK_N) * K
        B128_gl_offset = (tile_j * BLOCK_N + 128) * K

        a_row_h0 = wave_i * 64
        a_row_h1 = 128 + a_row_h0
        b_row_h0 = wave_j * 64
        b_row_h1 = 128 + b_row_h0

        A_rsrc = buffer_ops.create_buffer_resource(A)
        B_rsrc = buffer_ops.create_buffer_resource(B_T)
        C_rsrc = buffer_ops.create_buffer_resource(C)

        def _swizzle_128(row, col):
            offset = row * 128 + col
            swizzle = ((offset % (16 * 128)) >> 8) << 4
            swizzled_offset = offset ^ swizzle
            return swizzled_offset // 128, swizzled_offset % 128

        def _compute_global_swizzle():
            offsets = []
            for round in range_constexpr(4):
                row = lane_id // 8 + wave_id * 8 + round * 32
                col = (lane_id % 8) * 16
                a, b = _swizzle_128(row, col)
                offsets.append(a * K + b)
            return offsets

        def _load_lds(gl_src, lds_dst, k_offset, gl_offsets):
            lds_base = fx.Int32(fx.ptrtoint(lds_dst))
            for step in range_constexpr(4):
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

        def _load_rt(lds_src, wave_idx):
            # Load a 64x128 fragment of A/B from LDS to registers
            # Each 16x128 fragment requires 2 i32x4 (2 ds_read_b128)
            frag = []
            for i in range_constexpr(4):
                row = wave_idx * 64 + i * 16 + lane_id % 16
                halves = []
                for step in range_constexpr(2):
                    col = (lane_id // 16) * 16 + step * 64
                    row_swz, col_swz = _swizzle_128(row, col)
                    halves.append(_lds_load_i32x4(lds_src, row_swz * 128 + col_swz))
                frag.append(_pack_i32x42_i32x8(halves[0], halves[1]))  # i32x8
            return frag

        def _store_rt(c_frag, base_row, base_col):
            for ti in range_constexpr(4):
                row = base_row + ti * 16 + (lane_id // 16) * 4
                for tj in range_constexpr(4):
                    col = base_col + tj * 16 + lane_id % 16
                    vec_bf16 = Vec(c_frag[ti][tj]).to(fx.BFloat16)
                    for i in range_constexpr(4):
                        buffer_ops.buffer_store(
                            vec_bf16[i], C_rsrc, fx.Int32((row + i) * N + col)
                        )

        def _mfma_ABt_all(a, b, c, a_scales, b_scales):
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    c[i][j] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        MfmaAccumType_t,
                        [a[i], b[j], c[i][j], 0, 0, 0, a_scales, 0, b_scales],
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

        c00_frag = [[RT_C_i for _ in range_constexpr(4)] for _ in range_constexpr(4)]
        c01_frag = [[RT_C_i for _ in range_constexpr(4)] for _ in range_constexpr(4)]
        c10_frag = [[RT_C_i for _ in range_constexpr(4)] for _ in range_constexpr(4)]
        c11_frag = [[RT_C_i for _ in range_constexpr(4)] for _ in range_constexpr(4)]

        global_offsets = _compute_global_swizzle()

        # Prologue: pre-load A/B cur
        _load_lds(A_rsrc, A_lds[0][0], A0_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, B_lds[0][0], B0_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, B_lds[0][1], B128_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(A_rsrc, A_lds[0][1], A128_gl_offset + 0 * BLOCK_K, global_offsets)

        # Issue load for next tile
        _load_lds(A_rsrc, A_lds[1][0], A0_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, B_lds[1][0], B0_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, B_lds[1][1], B128_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(A_rsrc, A_lds[1][1], A128_gl_offset + 1 * BLOCK_K, global_offsets)

        _wait_barrier(28)

        a0_frag = _load_rt(A_lds[0][0], wave_i)

        _wait_barrier(24)

        b0_frag = _load_rt(B_lds[0][0], wave_j)

        cur, next = 0, 1
        for k in range_constexpr(K_ITERS - 2):
            sa_h0 = A_scales[k, tile_i + a_row_h0 + lane_id]
            sb_h0 = B_scales[k, tile_j + b_row_h0 + lane_id]
            sa_h1 = A_scales[k, tile_i + a_row_h1 + lane_id]
            sb_h1 = B_scales[k, tile_j + b_row_h1 + lane_id]

            _wait_barrier(16)

            b1_frag = _load_rt(B_lds[cur][1], wave_j)
            _load_lds(A_rsrc, A_lds[cur][0], A0_gl_offset + (k + 2) * BLOCK_K, global_offsets)
            _load_lds(B_rsrc, B_lds[cur][0], B0_gl_offset + (k + 2) * BLOCK_K, global_offsets)
            _mfma_ABt_all(a0_frag, b0_frag, c00_frag, sa_h0, sb_h0)

            a1_frag = _load_rt(A_lds[cur][1], wave_i)
            _load_lds(B_rsrc, B_lds[cur][1], B128_gl_offset + (k + 2) * BLOCK_K, global_offsets)
            _load_lds(A_rsrc, A_lds[cur][1], A128_gl_offset + (k + 2) * BLOCK_K, global_offsets)
            _mfma_ABt_all(a0_frag, b1_frag, c01_frag, sa_h0, sb_h1)

            _wait_barrier(16)

            a0_frag = _load_rt(A_lds[next][0], wave_i)
            _mfma_ABt_all(a1_frag, b0_frag, c10_frag, sa_h1, sb_h0)

            b0_frag = _load_rt(B_lds[next][0], wave_j)
            _mfma_ABt_all(a1_frag, b1_frag, c11_frag, sa_h1, sb_h1)

            # Swap cur and next
            cur ^= 1
            next ^= 1

        # step k = k_iters - 2
        _wait_barrier(16)
        k = K_ITERS - 2
        sa_h0 = A_scales[k, tile_i + a_row_h0 + lane_id]
        sb_h0 = B_scales[k, tile_j + b_row_h0 + lane_id]
        sa_h1 = A_scales[k, tile_i + a_row_h1 + lane_id]
        sb_h1 = B_scales[k, tile_j + b_row_h1 + lane_id]

        b1_frag = _load_rt(B_lds[cur][1], wave_j)
        _mfma_ABt_all(a0_frag, b0_frag, c00_frag, sa_h0, sb_h0)

        a1_frag = _load_rt(A_lds[cur][1], wave_i)
        _mfma_ABt_all(a0_frag, b1_frag, c01_frag, sa_h0, sb_h1)

        _wait_barrier(8)

        a0_frag = _load_rt(A_lds[next][0], wave_i)
        _mfma_ABt_all(a1_frag, b0_frag, c10_frag, sa_h1, sb_h0)
        b0_frag = _load_rt(B_lds[next][0], wave_j)
        _mfma_ABt_all(a1_frag, b1_frag, c11_frag, sa_h1, sb_h1)

        # Swap cur and next
        cur ^= 1
        next ^= 1

        # step k = k_iters - 1
        _wait_barrier(0)
        k = K_ITERS - 1

        base_row = tile_i * BLOCK_M + wave_i * 64
        base_col = tile_j * BLOCK_N + wave_j * 64

        b1_frag = _load_rt(B_lds[cur][1], wave_j)
        _mfma_ABt_all(a0_frag, b0_frag, c00_frag, sa_h0, sb_h0)

        a1_frag = _load_rt(A_lds[cur][1], wave_i)
        _mfma_ABt_all(a0_frag, b1_frag, c01_frag, sa_h0, sb_h1)

        _mfma_ABt_all(a1_frag, b0_frag, c10_frag, sa_h1, sb_h0)
        _mfma_ABt_all(a1_frag, b1_frag, c11_frag, sa_h1, sb_h1)

        _store_rt(c00_frag, base_row + 0, base_col + 0)
        _store_rt(c01_frag, base_row + 0, base_col + 128)
        _store_rt(c10_frag, base_row + 128, base_col + 0)
        _store_rt(c11_frag, base_row + 128, base_col + 128)

    @flyc.jit
    def launch_gemm(A: fx.Pointer, B_T: fx.Pointer, C: fx.Pointer, A_scales: fx.Pointer, B_scales: fx.Pointer, stream: fx.Stream):
        grid_x = (M * N) // (256 * 256)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scales,
            B_scales,
            value_attrs={
                "rocdl.waves_per_eu": 1,
                "rocdl.flat_work_group_size": "256,256",
            },
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    @flyc.kernel
    def repack_scales_kernel(scales: fx.Pointer, scales_repacked: fx.Pointer, dim: fx.Int32, k_steps: fx.Int32):
        idx = fx.block_idx.x * fx.block_dim.x + fx.thread_idx.x
        if idx >= dim * k_steps:
            return

        ki = idx // dim
        row = idx % dim
        r16 = row % 16
        k_sub = (row // 16) % 4
        tile = row // 64

        packed = 0
        for g in range_constexpr(4):
            src_row = tile * 64 + g * 16 + r16
            if src_row < dim:
                src_val = scales[ki, src_row]
                byte_val = (src_val >> (k_sub * 8)) & 0xFF
                packed = packed | (byte_val << (g * 8))

        scales_repacked[ki, row] = packed

    @flyc.jit
    def launch_repack_scales(scales: fx.Pointer, scales_repacked: fx.Pointer, dim: fx.Int32, k_steps: fx.Int32, stream: fx.Stream):
        grid_x = (k_steps * dim + 255) // 256
        repack_scales_kernel(
            scales,
            scales_repacked,
            dim,
            k_steps
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm, launch_repack_scales
