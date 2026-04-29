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

        # ---- LDS read with swizzle_xor16 (same as preshuffle, simpler codegen) ----
        _lds_k = fx.Index(lds_k_dim)
        _kb16 = fx.Index(lds_k_dim // 16)
        _kb16_mask = _kb16 - fx.Index(1)

        def lds_packs_k64(row, col_bytes, lds_buf):
            rem = fx.arith.andi(row, _kb16_mask)
            col_swz = col_bytes ^ (rem * c16)
            idx = row * _lds_k + col_swz
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

        # ---- Pre-compute DMA offsets ONCE (swizzle_xor16 on global addr) ----
        _K_idx = fx.Index(K)
        _global_dma_offsets = []
        for step in range_constexpr(4):
            local_row = _lane_div_8 + wave_id * fx.Index(8) + fx.Index(step * 32)
            col = _lane_mod_8 * c16
            rem = fx.arith.andi(local_row, _kb16_mask)
            col_swz = col ^ (rem * c16)
            _global_dma_offsets.append(local_row * _K_idx + col_swz)

        _a_base = [bx_m * _K_idx, (bx_m + fx.Index(half_rows)) * _K_idx]
        _b_base = [by_n * _K_idx, (by_n + fx.Index(half_rows)) * _K_idx]

        w_off = rocdl.readfirstlane(fx.Int64.ir_type, fx.Int64(wave_id * fx.Index(1024)))

        def _dma_step(rsrc, base_byte_off, base_k, step, lds_buf, half_idx, lds_ptr_state):
            """Issue one DMA step. Builds LDS pointer incrementally."""
            if const_expr(step == 0):
                lds_base_i = memref_dialect.extract_aligned_pointer_as_index(lds_buf)
                ptr = buffer_ops.create_llvm_ptr(
                    fx.Int64(lds_base_i + fx.Index(half_idx * half_size)), address_space=3)
                ptr = buffer_ops.get_element_ptr(ptr, w_off)
                lds_ptr_state[0] = ptr
            else:
                lds_ptr_state[0] = buffer_ops.get_element_ptr(lds_ptr_state[0], static_byte_offset=4096)
            g_off = fx.Int32(base_byte_off + base_k + _global_dma_offsets[step])
            rocdl.raw_ptr_buffer_load_lds(
                rsrc, lds_ptr_state[0], fx.Int32(dma_bytes), g_off,
                fx.Int32(0), fx.Int32(0), fx.Int32(1))

        def dma_half(rsrc, base_byte_off, base_k, lds_buf, half_idx):
            st = [None]
            for s in range_constexpr(4):
                _dma_step(rsrc, base_byte_off, base_k, s, lds_buf, half_idx, st)

        n_accs = 64

        def acc_idx(ci, cj, mi, ni):
            return ((ci * 2 + cj) * 4 + mi) * 4 + ni

        def run_cluster(ci, cj, a_regs, b_regs, accs,
                        dma_rsrc, dma_base_byte_off, dma_k,
                        dma_lds_buf, dma_half_idx, do_dma,
                        read_row_base, read_buf):
            k0_parts = {}
            next_frags = [None, None, None, None]
            dma_ptr_st = [None]
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
                if dma_i is not None and do_dma:
                    _dma_step(dma_rsrc, dma_base_byte_off, dma_k,
                              dma_i, dma_lds_buf, dma_half_idx, dma_ptr_st)
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
            b_h1_row = fx.Index(128) + wave_j * fx.Index(64) + lane_mod_16
            accs, b_h1 = run_cluster(0, 0, a_h0, b_h0, accs,
                                     a_rsrc, _a_base[0], dma_k,
                                     lds_a_cur, 0, do_dma,
                                     b_h1_row, lds_b_cur)

            a_h1_row = fx.Index(128) + wave_i * fx.Index(64) + lane_mod_16
            accs, a_h1 = run_cluster(0, 1, a_h0, b_h1, accs,
                                     b_rsrc, _b_base[0], dma_k,
                                     lds_b_cur, 0, do_dma,
                                     a_h1_row, lds_a_cur)

            if const_expr(do_dma):
                rocdl.s_waitcnt(16)
                gpu.barrier()
            rocdl.s_waitcnt(0)

            a_next_row = wave_i * fx.Index(64) + lane_mod_16
            accs, a_next_h0 = run_cluster(1, 0, a_h1, b_h0, accs,
                                          b_rsrc, _b_base[1], dma_k,
                                          lds_b_cur, 1, do_dma,
                                          a_next_row, lds_a_next)

            b_next_row = wave_j * fx.Index(64) + lane_mod_16
            accs, b_next_h0 = run_cluster(1, 1, a_h1, b_h1, accs,
                                          a_rsrc, _a_base[1], dma_k,
                                          lds_a_cur, 1, do_dma,
                                          b_next_row, lds_b_next)
            return accs, a_next_h0, b_next_h0

        # ---- Store with per-row scaling (HIP store_rt_scaled decomposition) ----
        # HIP: base_row = tile_i*BLOCK_M + wave_i*64
        # ---- hot_loop_scheduler: control MFMA/VMEM/DSRD interleaving ----
        # Per run_phase: 64 MFMAs, 16 DMA (vmem), 32 LDS reads (dsrd)
        _mfma_total = 64
        _num_vmem = 16
        _num_dsrd = 32

        def _build_schedule(numer, denom):
            if denom <= 0:
                return [0] * max(denom, 0)
            out = []
            prev = 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + (denom - 1)) // denom
                out.append(cur - prev)
                prev = cur
            return out

        _vmem_schedule = _build_schedule(_num_vmem, _mfma_total)
        _dsrd_schedule = _build_schedule(_num_dsrd, _mfma_total)

        def hot_loop_scheduler():
            for mfma_idx in range_constexpr(_mfma_total):
                rocdl.sched_mfma(1)
                n_dsrd = _dsrd_schedule[mfma_idx]
                if const_expr(n_dsrd > 0):
                    rocdl.sched_dsrd(n_dsrd)
                n_vmem = _vmem_schedule[mfma_idx]
                if const_expr(n_vmem > 0):
                    rocdl.sched_vmem(n_vmem)
            rocdl.sched_barrier(0)

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

        dma_half(a_rsrc, _a_base[0], k0, lds_a_pong, 0)
        dma_half(b_rsrc, _b_base[0], k0, lds_b_pong, 0)
        dma_half(b_rsrc, _b_base[1], k0, lds_b_pong, 1)
        dma_half(a_rsrc, _a_base[1], k0, lds_a_pong, 1)
        dma_half(a_rsrc, _a_base[0], k1, lds_a_ping, 0)
        dma_half(b_rsrc, _b_base[0], k1, lds_b_ping, 0)
        dma_half(b_rsrc, _b_base[1], k1, lds_b_ping, 1)
        dma_half(a_rsrc, _a_base[1], k1, lds_a_ping, 1)

        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(28)
        gpu.barrier()
        rocdl.sched_barrier(0)

        accs = [acc_init] * n_accs

        num_groups = K // (tile_k * 2)
        main_iters = num_groups - 1

        # ==== Flat compute: all MFMAs for one buffer, then DMA ====
        def compute_flat(accs, lds_a, lds_b):
            """Load all register tiles from LDS, then compute all 64 MFMAs."""
            a0 = load_rt(lds_a, wave_i, 0)
            a1 = load_rt(lds_a, wave_i, 1)
            b0 = load_rt(lds_b, wave_j, 0)
            b1 = load_rt(lds_b, wave_j, 1)
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    accs[acc_idx(0, 0, i, j)] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty, [a0[i], b0[j], accs[acc_idx(0, 0, i, j)],
                         0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F])
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    accs[acc_idx(0, 1, i, j)] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty, [a0[i], b1[j], accs[acc_idx(0, 1, i, j)],
                         0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F])
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    accs[acc_idx(1, 0, i, j)] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty, [a1[i], b0[j], accs[acc_idx(1, 0, i, j)],
                         0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F])
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    accs[acc_idx(1, 1, i, j)] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty, [a1[i], b1[j], accs[acc_idx(1, 1, i, j)],
                         0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F])

        def dma_tile(lds_a, lds_b, k_off):
            """DMA one full A+B tile to LDS."""
            dma_half(a_rsrc, _a_base[0], k_off, lds_a, 0)
            dma_half(a_rsrc, _a_base[1], k_off, lds_a, 1)
            dma_half(b_rsrc, _b_base[0], k_off, lds_b, 0)
            dma_half(b_rsrc, _b_base[1], k_off, lds_b, 1)

        # ==== Main loop (preshuffle-style: flat compute + DMA) ====
        if const_expr(main_iters > 0):
            init_state = list(accs)
            results = init_state

            SmemPtr._view_cache = None

            for k_iv, inner in range(0, main_iters * tile_k * 2, tile_k * 2, init=init_state):
                accs_in = list(inner)

                # Phase 1: compute on pong
                rocdl.s_waitcnt(0)
                gpu.barrier()
                compute_flat(accs_in, lds_a_pong, lds_b_pong)
                dma_tile(lds_a_pong, lds_b_pong, k_iv + fx.Index(tile_k * 2))
                hot_loop_scheduler()
                rocdl.s_waitcnt(0)
                gpu.barrier()

                # Phase 2: compute on ping
                compute_flat(accs_in, lds_a_ping, lds_b_ping)
                dma_tile(lds_a_ping, lds_b_ping, k_iv + fx.Index(tile_k * 3))
                hot_loop_scheduler()

                results = yield list(accs_in)

            SmemPtr._view_cache = None
            accs = list(results)

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

        # ==== Epilogue k_iters-2: flat compute on pong ====
        rocdl.s_waitcnt(0)
        gpu.barrier()
        compute_flat(accs, lds_a_pong, lds_b_pong)

        # ==== Epilogue k_iters-1: flat compute on ping + interleaved store ====
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=False)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=False)
        s_a_list, s_b_list = prefetch_scales()

        rocdl.s_waitcnt(0)
        gpu.barrier()
        compute_flat(accs, lds_a_ping, lds_b_ping)

        store_quad_scaled(accs, 0, 0, s_a_list, s_b_list)
        store_quad_scaled(accs, 0, 1, s_a_list, s_b_list)
        store_quad_scaled(accs, 1, 0, s_a_list, s_b_list)
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
