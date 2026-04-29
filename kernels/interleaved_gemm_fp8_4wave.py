# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""4-wave interleaved FP8 GEMM with per-row scaling.

Fixed 256x256x128 tiling.  Both A and B routed through LDS with
per-MFMA interleaving of compute, DMA, and LDS reads.
Uses mfma_scale_f32_16x16x128_f8f6f4 with identity E8M0 scales
(0x7F7F7F7F) and applies per-row A_scale/B_scale at the epilogue.
"""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


_cluster_schedule = [
    ([(0, 0)], None, []),
    ([(0, 1)], 0, [(0, "k0")]),
    ([(0, 2)], None, [(0, "k1")]),
    ([(0, 3)], 1, [(1, "k0")]),
    ([(1, 0), (1, 1)], None, [(1, "k1")]),
    ([(1, 2), (1, 3)], 2, [(2, "k0")]),
    ([(2, 0), (2, 1)], None, [(2, "k1")]),
    ([(2, 2), (2, 3)], 3, [(3, "k0")]),
    ([(3, 0), (3, 1)], None, [(3, "k1")]),
    ([(3, 2), (3, 3)], None, []),
]


def compile_interleaved_gemm_fp8_4wave(
    *,
    M: int = 0,
    N: int = 0,
    K: int,
    out_dtype: str = "bf16",
    waves_per_eu: Optional[int] = None,
):
    tile_m, tile_n, tile_k = 256, 256, 128
    num_waves = 4
    total_threads = num_waves * 64
    gpu_arch = get_hip_arch()
    if str(gpu_arch) != "gfx950":
        raise RuntimeError(f"Interleaved FP8 GEMM requires gfx950, got {gpu_arch}")
    if K % (tile_k * 2) != 0:
        raise ValueError(f"K must be divisible by {tile_k * 2}, got K={K}")

    _out_is_bf16 = out_dtype == "bf16"
    Vec = fx.Vector

    def _fp8_dt():
        return fx.Float8E4M3FN

    def _out_dt():
        return fx.BFloat16 if _out_is_bf16 else fx.Float16

    lds_k_dim = tile_k
    a_lds_bytes = tile_m * lds_k_dim
    b_lds_bytes = tile_n * lds_k_dim
    half_rows = 128
    half_size = half_rows * lds_k_dim
    dma_bytes = 16

    allocator_pong = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_ilvd0")
    allocator_ping = SmemAllocator(None, arch=gpu_arch, global_sym_name="smem_ilvd1")

    a_pong_off = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = a_pong_off + a_lds_bytes
    b_pong_off = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = b_pong_off + b_lds_bytes

    a_ping_off = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = a_ping_off + a_lds_bytes
    b_ping_off = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = b_ping_off + b_lds_bytes

    @flyc.kernel
    def kernel_interleaved(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        from flydsl._mlir.dialects import memref as memref_dialect

        c_n = fx.Index(i32_n)
        mfma_res_ty = Vec.make_type(4, fx.Float32)
        acc_init = Vec.filled(4, 0.0, fx.Float32)
        _vec16_ty = Vec.make_type(16, _fp8_dt())

        base_pong = allocator_pong.get_base()
        base_ping = allocator_ping.get_base()
        _fp8_ir = _fp8_dt().ir_type
        lds_a_pong = SmemPtr(base_pong, a_pong_off, _fp8_ir, shape=(a_lds_bytes,)).get()
        lds_b_pong = SmemPtr(base_pong, b_pong_off, _fp8_ir, shape=(b_lds_bytes,)).get()
        lds_a_ping = SmemPtr(base_ping, a_ping_off, _fp8_ir, shape=(a_lds_bytes,)).get()
        lds_b_ping = SmemPtr(base_ping, b_ping_off, _fp8_ir, shape=(b_lds_bytes,)).get()

        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=False)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=False)

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")
        bx_m = bx * tile_m
        by_n = by * tile_n

        wl_layout = fx.make_layout((num_waves, 64), (64, 1))
        wl_coord = fx.idx2crd(tx, wl_layout)
        wave_id = fx.get(wl_coord, 0)
        lane_id = fx.get(wl_coord, 1)
        wave_i = wave_id // 2
        wave_j = wave_id % 2

        l16_layout = fx.make_layout((4, 16), (16, 1))
        l16_coord = fx.idx2crd(lane_id, l16_layout)
        lane_div_16 = fx.get(l16_coord, 0)
        lane_mod_16 = fx.get(l16_coord, 1)

        c128 = fx.Index(128)
        c16 = fx.Index(16)
        _lds_2048 = fx.Index(16 * 128)
        col_b0 = lane_div_16 * 16
        col_b1 = lane_div_16 * 16 + fx.Index(64)
        _lane_div_8 = lane_id // 8
        _lane_mod_8 = lane_id % 8

        # ---- swizzle_128 (matches HIP) ----
        def swz_flat(row, col):
            offset = row * c128 + col
            swz = ((offset % _lds_2048) // fx.Index(256)) * c16
            return offset ^ swz

        def swz_rc(row, col):
            s = swz_flat(row, col)
            return s // c128, s % c128

        # ---- LDS read ----
        def lds_packs_k64(row, col_bytes, lds_buf):
            idx = swz_flat(row, col_bytes)
            v16 = Vec.load(_vec16_ty, lds_buf, [idx])
            i64x2 = Vec(v16).bitcast(fx.Int64)
            return i64x2[0].ir_value(), i64x2[1].ir_value()

        def pack4(x0, x1, x2, x3):
            return Vec.from_elements([x0, x1, x2, x3], fx.Int64).bitcast(fx.Int32)

        # HIP load_rt(A_lds[buf][half], regs, wave_idx):
        #   row = wave_idx*64 + fi*16 + lane_mod_16   (within 128-row half)
        #   physical row = half*128 + row
        def load_rt(lds_buf, wave_idx, half):
            frags = []
            for fi in range_constexpr(4):
                row = fx.Index(half * 128) + wave_idx * fx.Index(64) + fx.Index(fi * 16) + lane_mod_16
                a0, a1 = lds_packs_k64(row, col_b0, lds_buf)
                a2, a3 = lds_packs_k64(row, col_b1, lds_buf)
                frags.append(pack4(a0, a1, a2, a3))
            return frags

        # ---- DMA (with swizzle_128 on global address) ----
        def _compute_global_offset(base_row, base_k, step, half_idx):
            local_row = _lane_div_8 + wave_id * fx.Index(8) + fx.Index(step * 32)
            col = _lane_mod_8 * c16
            sr, sc = swz_rc(local_row, col)
            global_row = base_row + fx.Index(half_idx * half_rows) + sr
            return fx.Int32(global_row * fx.Index(K) + base_k + sc)

        def dma_half(lds_buf, rsrc, base_row, base_k, half_idx):
            lds_base_i = memref_dialect.extract_aligned_pointer_as_index(lds_buf)
            w_off = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(wave_id * fx.Index(1024)))
            ptr = buffer_ops.create_llvm_ptr(
                fx.Int64(lds_base_i + fx.Index(half_idx * half_size)), address_space=3)
            ptr = buffer_ops.get_element_ptr(ptr, w_off)
            for s in range_constexpr(4):
                g_off = _compute_global_offset(base_row, base_k, s, half_idx)
                if const_expr(s > 0):
                    ptr = buffer_ops.get_element_ptr(ptr, static_byte_offset=4096)
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc, ptr, fx.Int32(dma_bytes), g_off,
                    fx.Int32(0), fx.Int32(0), fx.Int32(1))

        def prepare_dma_steps(lds_buf, rsrc, base_row, base_k, half_idx):
            lds_base_i = memref_dialect.extract_aligned_pointer_as_index(lds_buf)
            w_off = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(wave_id * fx.Index(1024)))
            ptr = buffer_ops.create_llvm_ptr(
                fx.Int64(lds_base_i + fx.Index(half_idx * half_size)), address_space=3)
            ptr = buffer_ops.get_element_ptr(ptr, w_off)
            steps = []
            for s in range_constexpr(4):
                g_off = _compute_global_offset(base_row, base_k, s, half_idx)
                if const_expr(s > 0):
                    ptr = buffer_ops.get_element_ptr(ptr, static_byte_offset=4096)
                steps.append((rsrc, ptr, g_off))
            return steps

        n_accs = 64

        def acc_idx(ci, cj, mi, ni):
            return ((ci * 2 + cj) * 4 + mi) * 4 + ni

        def run_cluster(ci, cj, a_regs, b_regs, accs,
                        dma_step_list, read_row_base, read_buf):
            k0_parts = {}
            next_frags = [None, None, None, None]
            for grp in range_constexpr(10):
                mfmas = _cluster_schedule[grp][0]
                dma_i = _cluster_schedule[grp][1]
                reads = _cluster_schedule[grp][2]
                for mi, ni in mfmas:
                    ai = acc_idx(ci, cj, mi, ni)
                    accs[ai] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [a_regs[mi], b_regs[ni], accs[ai],
                         0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F])
                if dma_i is not None and dma_step_list is not None:
                    r, lp, go = dma_step_list[dma_i]
                    rocdl.raw_ptr_buffer_load_lds(
                        r, lp, fx.Int32(dma_bytes), go,
                        fx.Int32(0), fx.Int32(0), fx.Int32(1))
                if read_buf is not None:
                    for fi, kh in reads:
                        row = read_row_base + fx.Index(fi * 16)
                        if kh == "k0":
                            k0_parts[fi] = lds_packs_k64(row, col_b0, read_buf)
                        else:
                            a0, a1 = k0_parts[fi]
                            a2, a3 = lds_packs_k64(row, col_b1, read_buf)
                            next_frags[fi] = pack4(a0, a1, a2, a3)
            return accs, next_frags

        def run_phase(accs, a_h0, b_h0, lds_a_cur, lds_b_cur, lds_a_next, lds_b_next,
                      dma_k, do_dma):
            # HIP decomposition: read_row = half*128 + wave_idx*64 + fi*16 + lane_mod_16
            # Cluster 1: c[0][0], DMA A_cur half0, read b[1] from B_cur[1]
            dma0 = prepare_dma_steps(lds_a_cur, a_rsrc, bx_m, dma_k, 0) if do_dma else None
            b_h1_row = fx.Index(128) + wave_j * fx.Index(64) + lane_mod_16
            accs, b_h1 = run_cluster(0, 0, a_h0, b_h0, accs, dma0, b_h1_row, lds_b_cur)

            rocdl.sched_barrier(0)

            # Cluster 2: c[0][1], DMA B_cur half0, read a[1] from A_cur[1]
            dma1 = prepare_dma_steps(lds_b_cur, b_rsrc, by_n, dma_k, 0) if do_dma else None
            a_h1_row = fx.Index(128) + wave_i * fx.Index(64) + lane_mod_16
            accs, a_h1 = run_cluster(0, 1, a_h0, b_h1, accs, dma1, a_h1_row, lds_a_cur)

            rocdl.sched_barrier(0)
            if const_expr(do_dma):
                rocdl.s_waitcnt(16)
                gpu.barrier()
            rocdl.s_waitcnt(0)
            rocdl.sched_barrier(0)

            # Cluster 3: c[1][0], DMA B_cur half1, read a[0] from A_next[0]
            dma2 = prepare_dma_steps(lds_b_cur, b_rsrc, by_n, dma_k, 1) if do_dma else None
            a_next_row = wave_i * fx.Index(64) + lane_mod_16
            accs, a_next_h0 = run_cluster(1, 0, a_h1, b_h0, accs, dma2, a_next_row, lds_a_next)

            # Cluster 4: c[1][1], DMA A_cur half1, read b[0] from B_next[0]
            dma3 = prepare_dma_steps(lds_a_cur, a_rsrc, bx_m, dma_k, 1) if do_dma else None
            b_next_row = wave_j * fx.Index(64) + lane_mod_16
            accs, b_next_h0 = run_cluster(1, 1, a_h1, b_h1, accs, dma3, b_next_row, lds_b_next)
            return accs, a_next_h0, b_next_h0

        # ---- Store with per-row scaling (HIP store_rt_scaled decomposition) ----
        # HIP: base_row = tile_i*BLOCK_M + wave_i*64
        #      c[ci][cj] stored at (base_row + ci*128, base_col + cj*128)
        def prefetch_scales():
            s_b = []
            for ni in range_constexpr(4):
                for cj in range_constexpr(2):
                    col = by_n + wave_j * fx.Index(64) + fx.Index(cj * 128 + ni * 16) + lane_mod_16
                    s_b.append(buffer_ops.buffer_load(scale_b_rsrc, col, vec_width=1, dtype=fx.Float32))
            s_a = []
            for mi in range_constexpr(4):
                for ci in range_constexpr(2):
                    row_base = bx_m + wave_i * fx.Index(64) + fx.Index(ci * 128 + mi * 16) + lane_div_16 * 4
                    s_a.append(buffer_ops.buffer_load(scale_a_rsrc, row_base, vec_width=4, dtype=fx.Float32))
            return s_a, s_b

        def store_output_scaled(accs, s_a_list, s_b_list):
            for ci in range_constexpr(2):
                for cj in range_constexpr(2):
                    for mi in range_constexpr(4):
                        s_a_vec = s_a_list[mi * 2 + ci]
                        for ii in range_constexpr(4):
                            row = bx_m + wave_i * fx.Index(64) + fx.Index(ci * 128 + mi * 16) + lane_div_16 * 4 + fx.Index(ii)
                            s_a = Vec(s_a_vec)[ii]
                            for ni in range_constexpr(4):
                                col = by_n + wave_j * fx.Index(64) + fx.Index(cj * 128 + ni * 16) + lane_mod_16
                                s_b = s_b_list[ni * 2 + cj]
                                val = Vec(accs[acc_idx(ci, cj, mi, ni)])[ii]
                                scaled = val * (s_a * s_b)
                                out_val = _out_dt()(scaled)
                                buffer_ops.buffer_store(out_val, c_rsrc, row * c_n + col)

        # ==== Prologue: 8 half-tile DMAs in HIP order, two staggered barriers ====
        k0 = fx.Index(0)
        k1 = fx.Index(tile_k)

        dma_half(lds_a_pong, a_rsrc, bx_m, k0, 0)    # A_lds[cur][0]  → wait 28
        dma_half(lds_b_pong, b_rsrc, by_n, k0, 0)     # B_lds[cur][0]  → wait 24
        dma_half(lds_b_pong, b_rsrc, by_n, k0, 1)     # B_lds[cur][1]  → wait 20
        dma_half(lds_a_pong, a_rsrc, bx_m, k0, 1)    # A_lds[cur][1]  → wait 16
        dma_half(lds_a_ping, a_rsrc, bx_m, k1, 0)    # A_lds[next][0] → wait 12
        dma_half(lds_b_ping, b_rsrc, by_n, k1, 0)     # B_lds[next][0] → wait 8
        dma_half(lds_b_ping, b_rsrc, by_n, k1, 1)     # B_lds[next][1] → wait 4
        dma_half(lds_a_ping, a_rsrc, bx_m, k1, 1)    # A_lds[next][1] → wait 0

        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(28)
        gpu.barrier()
        rocdl.sched_barrier(0)

        a_h0 = load_rt(lds_a_pong, wave_i, 0)   # a[0] from A_lds[cur][0]

        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(24)
        gpu.barrier()
        rocdl.sched_barrier(0)

        b_h0 = load_rt(lds_b_pong, wave_j, 0)   # b[0] from B_lds[cur][0]

        accs = [acc_init] * n_accs

        num_groups = K // (tile_k * 2)
        main_iters = num_groups - 1

        # ==== Main loop ====
        if const_expr(main_iters > 0):
            init_state = list(accs)
            results = init_state

            SmemPtr._view_cache = None

            for k_iv, inner in range(0, main_iters * tile_k * 2, tile_k * 2, init=init_state):
                accs_in = list(inner)

                rocdl.s_waitcnt(0)
                gpu.barrier()
                ah0_in = load_rt(lds_a_pong, wave_i, 0)
                bh0_in = load_rt(lds_b_pong, wave_j, 0)

                dma_k_pong = k_iv + fx.Index(tile_k * 2)
                accs_in, ah1_in, bh1_in = run_phase(
                    accs_in, ah0_in, bh0_in,
                    lds_a_pong, lds_b_pong, lds_a_ping, lds_b_ping,
                    dma_k_pong, True)
                rocdl.s_waitcnt(0)
                gpu.barrier()

                dma_k_ping = k_iv + fx.Index(tile_k * 3)
                accs_in, _, _ = run_phase(
                    accs_in, ah1_in, bh1_in,
                    lds_a_ping, lds_b_ping, lds_a_pong, lds_b_pong,
                    dma_k_ping, True)

                results = yield list(accs_in)

            SmemPtr._view_cache = None
            accs = list(results)

        # ==== Helpers for epilogue ====
        def mfma_all_quad(accs, ci, cj, a_frags, b_frags):
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    ai = acc_idx(ci, cj, i, j)
                    accs[ai] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [a_frags[i], b_frags[j], accs[ai],
                         0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F])

        def store_quad_scaled(accs, ci, cj, s_a_list, s_b_list):
            base_row = bx_m + wave_i * fx.Index(64)
            base_col = by_n + wave_j * fx.Index(64)
            for mi in range_constexpr(4):
                s_a_vec = s_a_list[mi * 2 + ci]
                row_base = base_row + fx.Index(ci * 128 + mi * 16) + lane_div_16 * 4
                for ii in range_constexpr(4):
                    row = row_base + fx.Index(ii)
                    s_a = Vec(s_a_vec)[ii]
                    idx_base = row * c_n + base_col + fx.Index(cj * 128) + lane_mod_16
                    for ni in range_constexpr(4):
                        s_b = s_b_list[ni * 2 + cj]
                        val = Vec(accs[acc_idx(ci, cj, mi, ni)])[ii]
                        out_val = _out_dt()(val * (s_a * s_b))
                        buffer_ops.buffer_store(out_val, c_rsrc, idx_base + (ni * 16))

        # ==== Epilogue k_iters-2: compute without DMA (matching HIP) ====
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(16)
        gpu.barrier()
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)

        b1 = load_rt(lds_b_pong, wave_j, 1)       # b[1] from B_cur[1]

        rocdl.sched_barrier(0)
        mfma_all_quad(accs, 0, 0, a_h0, b_h0)
        rocdl.sched_barrier(0)

        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)

        a1 = load_rt(lds_a_pong, wave_i, 1)        # a[1] from A_cur[1]

        rocdl.sched_barrier(0)
        mfma_all_quad(accs, 0, 1, a_h0, b1)
        rocdl.sched_barrier(0)

        rocdl.s_waitcnt(8)
        gpu.barrier()
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)

        a_h0 = load_rt(lds_a_ping, wave_i, 0)      # a[0] from A_next[0]

        rocdl.sched_barrier(0)
        mfma_all_quad(accs, 1, 0, a1, b_h0)
        rocdl.sched_barrier(0)

        b_h0 = load_rt(lds_b_ping, wave_j, 0)      # b[0] from B_next[0]

        rocdl.sched_barrier(0)
        mfma_all_quad(accs, 1, 1, a1, b1)
        rocdl.sched_barrier(0)

        # ==== Epilogue k_iters-1: compute + interleaved store (matching HIP) ====
        # Prefetch scales — hidden behind the 64 MFMAs below
        s_a_list, s_b_list = prefetch_scales()

        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(0)
        gpu.barrier()
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)

        b1 = load_rt(lds_b_ping, wave_j, 1)        # b[1] from B_cur[1]

        rocdl.sched_barrier(0)
        mfma_all_quad(accs, 0, 0, a_h0, b_h0)
        rocdl.sched_barrier(0)

        store_quad_scaled(accs, 0, 0, s_a_list, s_b_list)

        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)

        a1 = load_rt(lds_a_ping, wave_i, 1)        # a[1] from A_cur[1]

        rocdl.sched_barrier(0)
        mfma_all_quad(accs, 0, 1, a_h0, b1)
        rocdl.sched_barrier(0)

        store_quad_scaled(accs, 0, 1, s_a_list, s_b_list)

        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(0)
        rocdl.sched_barrier(0)

        rocdl.sched_barrier(0)
        mfma_all_quad(accs, 1, 0, a1, b_h0)
        rocdl.sched_barrier(0)

        store_quad_scaled(accs, 1, 0, s_a_list, s_b_list)

        rocdl.sched_barrier(0)
        mfma_all_quad(accs, 1, 1, a1, b1)
        rocdl.sched_barrier(0)

        store_quad_scaled(accs, 1, 1, s_a_list, s_b_list)

    @flyc.jit
    def launch_interleaved(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        allocator_pong.finalized = False
        allocator_ping.finalized = False
        ctx = CompilationContext.get_current()
        from flydsl._mlir import ir
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        launcher = kernel_interleaved(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n)
        if const_expr(waves_per_eu is not None):
            _wpe = int(waves_per_eu)
            if const_expr(_wpe >= 1):
                for op in ctx.gpu_module_body.operations:
                    if const_expr(hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func"):
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            fx.Int32.ir_type, _wpe)
        launcher.launch(grid=(gx, gy, 1), block=(total_threads, 1, 1), stream=stream)

    return launch_interleaved


__all__ = ["compile_interleaved_gemm_fp8_4wave"]
