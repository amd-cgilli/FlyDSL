# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""4-wave FP8 matmul with row-wise scaling for AMD CDNA4.

Algorithm derived from HipKittens FP8_4wave
(https://github.com/HazyResearch/HipKittens/blob/7782744ba1fd259a377a99e2ea8f71384cc80e55/kernels/gemm/fp8fp32/FP8_4wave/4_wave.cu#L1).

Global IO, scale loads, and bf16 stores go through the layout API
(``fx.rocdl.make_buffer_tensor`` + ``fx.copy`` with ``BufferCopyLDS128b``
/ ``BufferCopy{16,32,128}b``). MFMAs use ``fly.mma_atom_call_ssa`` so
the chained Vec(4, f32) accumulator stays on AGPR. The XOR swizzle and
the 8-buffer LDS pipeline ping-pong are kept as direct arithmetic to
preserve the original kernel's interleaved-cluster scheduling.

Optional B preshuffle uses the same on-disk layout as
``preshuffle_gemm_v2`` / ``shuffle_weight((16, 16))``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, range_constexpr
from kernels.fp8_gemm_utils import (
    G2SLoader,
    Mfma16x16x128,
    S2RLoader,
    StoreC,
    ceildiv,
    compile_splitk_reduce,
    compute_global_swizzle,
    divmod,
    make_fp8_buffer_tensor,
    pack_i32x4_i32x8,
    swizzle_128,
    wait_barrier,
)


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle(num_pid_m, num_pid_n):
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x

    num_wg = num_pid_m * num_pid_n

    # Simple path: no XCD remapping.
    simple_m, simple_n = divmod(wgid, num_pid_n)

    # XCD-remapped path.
    intra_xcd, xcd = divmod(wgid, NUM_XCDS)
    wgid_remap = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = divmod(wgid_remap, num_wgid_in_group)
    first_pid_m = group_id * WGM
    group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m

    use_simple = (num_wg <= SWIZZLE_THRESHOLD) | (num_wg % NUM_XCDS != 0)
    return (arith.select(use_simple, simple_m, pid_m), arith.select(use_simple, simple_n, pid_n))


def compile_fp8_gemm_4w(
    *,
    K: int,
    BLOCK_M: int = 256,
    BLOCK_N: int = 256,
    use_xcd_remap: bool = True,
    b_preshuffled: bool = False,
    num_splits: int = 1,
    num_lds_stages: int = 2,
):
    # MFMA atom is 16x16x128; 4 waves in a 2x2 config require BLOCK >= 64.
    BLOCK_K = 128
    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    assert BLOCK_M >= 64 and BLOCK_M % 64 == 0 and BLOCK_N >= 64 and BLOCK_N % 64 == 0
    assert K % BLOCK_K == 0

    # Number of 16-row 16x128 tiles per wave per A/B partition.
    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16
    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    N_LDS_ROUNDS = max(N_TILES_A, N_TILES_B)

    _use_interleaved_block = BLOCK_M == 256 and BLOCK_N == 256
    assert num_lds_stages > 1, f"num_lds_stages must be > 1, got {num_lds_stages}"
    assert num_splits >= 1
    assert K % num_splits == 0, f"K ({K}) must be divisible by n_splits ({num_splits})"
    K_PER_SPLIT = K // num_splits
    assert K_PER_SPLIT % BLOCK_K == 0, f"K_PER_SPLIT ({K_PER_SPLIT}) must be divisible by BLOCK_K ({BLOCK_K})"
    K_ITERS_PER_SPLIT = K_PER_SPLIT // BLOCK_K
    assert K_ITERS_PER_SPLIT >= num_lds_stages, (
        f"Each split needs >= {num_lds_stages} K iterations for multi-buffered prologue, "
        f"got {K_ITERS_PER_SPLIT} (K={K}, n_splits={num_splits}, BLOCK_K={BLOCK_K})"
    )

    assert num_lds_stages * (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) <= 160 * 1024, f"LDS too small for {num_lds_stages}-buffering with {BLOCK_M}x{BLOCK_N} tiles"
    LOADS_PER_STAGE = 2 * N_TILES_A + 2 * N_TILES_B

    _is_split_k = num_splits > 1

    _a_lds_size = LDS_BLOCK_M * BLOCK_K
    _b_lds_size = LDS_BLOCK_N * BLOCK_K

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor, B_T: fx.Tensor, C: fx.Tensor, A_scale: fx.Tensor, B_scale: fx.Tensor, c_m: fx.Int32, c_n: fx.Int32
    ):
        F8_IR_t = fx.Float8E4M3FN.ir_type
        lds_alloc = fx.SharedAllocator()

        a_lds = [
            [lds_alloc.allocate(fx.Array[fx.Float8E4M3FN, _a_lds_size, 16]).peek() for _ in range_constexpr(2)]
            for _ in range_constexpr(num_lds_stages)
        ]

        b_lds = [
            [lds_alloc.allocate(fx.Array[fx.Float8E4M3FN, _b_lds_size, 16]).peek() for _ in range_constexpr(2)]
            for _ in range_constexpr(num_lds_stages)
        ]

        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        n_blocks = ceildiv(c_n, BLOCK_N)
        m_pad = ceildiv(c_m, BLOCK_M) * BLOCK_M
        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(ceildiv(c_m, BLOCK_M), n_blocks)
        else:
            tile_i, tile_j = divmod(fx.block_idx.x, n_blocks)

        wave_i = wave_id // 2
        wave_j = wave_id % 2

        split_k_idx = fx.block_idx.y
        k_base = split_k_idx * K_PER_SPLIT

        A0_gl_offset = (tile_i * BLOCK_M) * K + k_base
        A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K + k_base
        A_K_STEP = BLOCK_K
        B0_gl_offset = (tile_j * BLOCK_N) * K + k_base
        B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K + k_base
        B_K_STEP = (2 * 1024) if b_preshuffled else BLOCK_K

        gA = make_fp8_buffer_tensor(A, F8_IR_t)
        gB = make_fp8_buffer_tensor(B_T, F8_IR_t)
        ga_div = fx.logical_divide(gA, fx.make_layout(1, 1))
        gb_div = fx.logical_divide(gB, fx.make_layout(1, 1))

        def _compute_lds_swizzle(s2r, preshuffled=False):
            lds_swz = []
            for row_offset in range_constexpr(s2r.n_tiles):
                row = s2r.wave_idx * (s2r.n_tiles * 16) + row_offset * 16 + lane_id % 16
                swz = []
                for i in range_constexpr(2):
                    col = (lane_id // 16) * 16 + i * 64
                    if const_expr(preshuffled):
                        swz.append((row // 8) * 1024 + (row % 8) * 16 + (col // 16) * 128)
                    else:
                        r, c = swizzle_128(row, col)
                        swz.append(r * BLOCK_K + c)
                lds_swz.append(swz)
            return lds_swz

        mfma = Mfma16x16x128(N_TILES_A, N_TILES_B)

        def _interleaved_cluster(
            lds_dst,
            g2s,
            k_offset,
            s2r,
            lds_src,
            a,
            b,
            c,
            lds_src_preshuffled=False,
        ):
            rt_dst = []

            c[mfma.idx(0, 0)] = mfma.call_one(a, b, c, 0, 0)
            c[mfma.idx(0, 1)] = mfma.call_one(a, b, c, 0, 1)

            lds_swz = _compute_lds_swizzle(s2r, preshuffled=lds_src_preshuffled)
            g2s.load_one(lds_dst, k_offset, 0)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[0][0])

            c[mfma.idx(0, 2)] = mfma.call_one(a, b, c, 0, 2)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[0][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(0, 3)] = mfma.call_one(a, b, c, 0, 3)

            g2s.load_one(lds_dst, k_offset, 1)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[1][0])

            c[mfma.idx(1, 0)] = mfma.call_one(a, b, c, 1, 0)
            c[mfma.idx(1, 1)] = mfma.call_one(a, b, c, 1, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[1][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(1, 2)] = mfma.call_one(a, b, c, 1, 2)
            c[mfma.idx(1, 3)] = mfma.call_one(a, b, c, 1, 3)

            g2s.load_one(lds_dst, k_offset, 2)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[2][0])

            c[mfma.idx(2, 0)] = mfma.call_one(a, b, c, 2, 0)
            c[mfma.idx(2, 1)] = mfma.call_one(a, b, c, 2, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[2][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(2, 2)] = mfma.call_one(a, b, c, 2, 2)
            c[mfma.idx(2, 3)] = mfma.call_one(a, b, c, 2, 3)

            g2s.load_one(lds_dst, k_offset, 3)
            rt_dst_0 = s2r.load_one(lds_src, lds_swz[3][0])

            c[mfma.idx(3, 0)] = mfma.call_one(a, b, c, 3, 0)
            c[mfma.idx(3, 1)] = mfma.call_one(a, b, c, 3, 1)

            rt_dst_1 = s2r.load_one(lds_src, lds_swz[3][1])
            rt_dst.append(pack_i32x4_i32x8(rt_dst_0, rt_dst_1))

            c[mfma.idx(3, 2)] = mfma.call_one(a, b, c, 3, 2)
            c[mfma.idx(3, 3)] = mfma.call_one(a, b, c, 3, 3)

            return c, rt_dst

        def _compute_cluster(
            lds_dst,
            g2s,
            k_offset,
            s2r,
            lds_src,
            a,
            b,
            c,
            lds_src_preshuffled=False,
        ):
            g2s.load(lds_dst, k_offset)
            rt_dst = s2r.load(lds_src, preshuffled=lds_src_preshuffled)
            c = mfma.call(a, b, c)
            return c, rt_dst

        def _compute_block(
            lds_dst,
            g2s,
            k_offset,
            s2r,
            lds_src,
            a,
            b,
            c,
            lds_src_preshuffled=False,
        ):
            if const_expr(_use_interleaved_block):
                return _interleaved_cluster(
                    lds_dst,
                    g2s,
                    k_offset,
                    s2r,
                    lds_src,
                    a,
                    b,
                    c,
                    lds_src_preshuffled=lds_src_preshuffled,
                )
            else:
                return _compute_cluster(
                    lds_dst,
                    g2s,
                    k_offset,
                    s2r,
                    lds_src,
                    a,
                    b,
                    c,
                    lds_src_preshuffled=lds_src_preshuffled,
                )

        # Each wave handles 2x2 64x64 sub-tiles of the output.
        c00_frag = [mfma.zero_value] * N_ACCUMS
        c01_frag = [mfma.zero_value] * N_ACCUMS
        c10_frag = [mfma.zero_value] * N_ACCUMS
        c11_frag = [mfma.zero_value] * N_ACCUMS

        gl_off_a = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=False)
        gl_off_b = compute_global_swizzle(lane_id, wave_id, K, N_LDS_ROUNDS, preshuffled=b_preshuffled)

        a_g2s = G2SLoader(ga_div, gl_off_a, N_TILES_A, F8_IR_t, wave_id)
        b_g2s = G2SLoader(gb_div, gl_off_b, N_TILES_B, F8_IR_t, wave_id)
        a_s2r = S2RLoader(wave_i, N_TILES_A)
        b_s2r = S2RLoader(wave_j, N_TILES_B)
        store_c = StoreC(A_scale, B_scale, C, c_m, c_n, mfma.idx, N_TILES_A, N_TILES_B, _is_split_k, m_pad)

        # Prologue: pre-fill LDS
        for s in range_constexpr(num_lds_stages):
            a_g2s.load(a_lds[s][0], A0_gl_offset + s * A_K_STEP)
            b_g2s.load(b_lds[s][0], B0_gl_offset + s * B_K_STEP)
            b_g2s.load(b_lds[s][1], B1_gl_offset + s * B_K_STEP)
            a_g2s.load(a_lds[s][1], A1_gl_offset + s * A_K_STEP)

        # In total we have in-flight 2 * num_lds_stages * N_TILES loads from global to LDS for A and for B
        wait_barrier(((2 * num_lds_stages - 1) * N_TILES_A) + ((2 * num_lds_stages) * N_TILES_B))
        a0_frag = a_s2r.load(a_lds[0][0])

        wait_barrier(((2 * num_lds_stages - 1) * N_TILES_A) + ((2 * num_lds_stages - 1) * N_TILES_B))

        b0_frag = b_s2r.load(b_lds[0][0], preshuffled=b_preshuffled)

        cs = 0
        for k in range_constexpr(K_ITERS_PER_SPLIT - num_lds_stages):
            ns = (cs + 1) % num_lds_stages
            wait_barrier((num_lds_stages - 1) * LOADS_PER_STAGE)

            c00_frag, b1_frag = _compute_block(
                a_lds[cs][0],
                a_g2s,
                A0_gl_offset + (k + num_lds_stages) * A_K_STEP,
                b_s2r,
                b_lds[cs][1],
                a0_frag,
                b0_frag,
                c00_frag,
                lds_src_preshuffled=b_preshuffled,
            )

            c01_frag, a1_frag = _compute_block(
                b_lds[cs][0],
                b_g2s,
                B0_gl_offset + (k + num_lds_stages) * B_K_STEP,
                a_s2r,
                a_lds[cs][1],
                a0_frag,
                b1_frag,
                c01_frag,
            )

            wait_barrier((num_lds_stages - 1) * LOADS_PER_STAGE)

            c10_frag, a0_frag = _compute_block(
                b_lds[cs][1],
                b_g2s,
                B1_gl_offset + (k + num_lds_stages) * B_K_STEP,
                a_s2r,
                a_lds[ns][0],
                a1_frag,
                b0_frag,
                c10_frag,
            )

            c11_frag, b0_frag = _compute_block(
                a_lds[cs][1],
                a_g2s,
                A1_gl_offset + (k + num_lds_stages) * A_K_STEP,
                b_s2r,
                b_lds[ns][0],
                a1_frag,
                b1_frag,
                c11_frag,
                lds_src_preshuffled=b_preshuffled,
            )

            cs = ns

        # Tail: drain remaining num_lds_stages K-tiles without issuing new loads.
        remaining = num_lds_stages - 1
        for _ in range_constexpr(num_lds_stages - 1):
            ns = (cs + 1) % num_lds_stages
            wait_barrier(remaining * LOADS_PER_STAGE)
            b1_frag = b_s2r.load(b_lds[cs][1], preshuffled=b_preshuffled)
            c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
            a1_frag = a_s2r.load(a_lds[cs][1])
            c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
            wait_barrier((remaining - 1) * LOADS_PER_STAGE + N_TILES_A + N_TILES_B)
            a0_frag = a_s2r.load(a_lds[ns][0])
            c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
            b0_frag = b_s2r.load(b_lds[ns][0], preshuffled=b_preshuffled)
            c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)
            cs = ns
            remaining -= 1

        # Tail step k_iters - 1.
        wait_barrier(0)
        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)
        a_scales_0 = store_c.load_a_scales(base_row)
        a_scales_1 = store_c.load_a_scales(base_row + LDS_BLOCK_M)
        b_scales_0 = store_c.load_b_scales(base_col)
        b_scales_1 = store_c.load_b_scales(base_col + LDS_BLOCK_N)

        b1_frag = b_s2r.load(b_lds[cs][1], preshuffled=b_preshuffled)
        a1_frag = a_s2r.load(a_lds[cs][1])

        c00_frag = mfma.call(a0_frag, b0_frag, c00_frag)
        c01_frag = mfma.call(a0_frag, b1_frag, c01_frag)
        c10_frag = mfma.call(a1_frag, b0_frag, c10_frag)
        c11_frag = mfma.call(a1_frag, b1_frag, c11_frag)

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
        c_m: fx.Int32,
        c_n: fx.Int32,
        stream: fx.Stream,
    ):
        grid_x = ceildiv(c_m, BLOCK_M) * ceildiv(c_n, BLOCK_N)
        kernel_gemm(
            A,
            B_T,
            C,
            A_scale,
            B_scale,
            c_m,
            c_n,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, num_splits, 1), block=(256, 1, 1), stream=stream)

    if not _is_split_k:
        return launch_gemm, None

    launch_reduce = compile_splitk_reduce(BLOCK_M, num_splits)
    return launch_gemm, launch_reduce
