# SPDX-License-Identifier: Apache-2.0

"""4-wave FP8 blockscale preshuffle GEMM with DMA A loads and interleaved schedule.

C[M,N] = A[M,K] @ B[N,K]^T   (FP8e4m3 -> BF16, f32 accumulator)

Configurable tile_m/tile_n/tile_k. DMA for A (raw_ptr_buffer_load_lds),
preshuffle B from global to registers, ping-pong LDS, block-scaling.

Scale layouts:
  scale_a: [scale_k, M]  transposed, f32
  scale_b: [scale_n, scale_k] row-major, f32
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr, vector
from flydsl.expr import buffer_ops, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir

from kernels.mfma_preshuffle_pipeline import (
    swizzle_xor16,
    tile_chunk_coord_i32,
    crd2idx,
    _buffer_load_vec,
)
from kernels.mfma_epilogues import mfma_epilog

SCALE_BLOCK_K = 128
SCALE_BLOCK_N = 128


def compile_fp8_gemm_4wave(
    *,
    M: int,
    N: int,
    K: int,
    tile_m: int = 64,
    tile_n: int = 128,
    tile_k: int = 128,
    scale_block_k: int = SCALE_BLOCK_K,
):
    """Compile FP8 blockscale GEMM with DMA A loads and interleaved schedule."""
    assert K % tile_k == 0
    assert N % tile_n == 0
    assert tile_k % scale_block_k == 0
    assert tile_k % 64 == 0

    k_iters = K // tile_k
    scale_k = K // scale_block_k
    sb_per_tile = tile_k // scale_block_k
    ku_per_sb = scale_block_k // 64

    elem_bytes = 1
    total_threads = 256
    wave_size = 64
    num_waves = 4
    tile_k_bytes = tile_k * elem_bytes

    # DMA config
    dma_bytes = 16  # gfx950: 128-bit DMA
    dma_dwords = dma_bytes // 4
    bytes_a_per_tile = tile_m * tile_k * elem_bytes
    num_dma_ops = bytes_a_per_tile // (total_threads * dma_bytes)

    # Layout constants
    kpack_elems = 16
    m_repeat = tile_m // 16
    k_unroll = tile_k_bytes // 64
    n_per_wave = tile_n // num_waves
    num_acc_n = n_per_wave // 16
    n_accs = m_repeat * num_acc_n

    k_bytes_factor = K * elem_bytes
    tile_k_dwords = tile_k // 4

    # B preshuffle strides
    n0_val = N // 16
    k0_val = K // 64
    _stride_nlane = kpack_elems
    _stride_klane = 16 * _stride_nlane
    _stride_k0 = 4 * _stride_klane
    _stride_n0 = k0_val * _stride_k0

    # ---- Ping-pong LDS ----
    lds_tile_bytes = tile_m * tile_k_bytes
    allocator_pong = SmemAllocator(None, arch="gfx950", global_sym_name="smem0")
    allocator_ping = SmemAllocator(None, arch="gfx950", global_sym_name="smem1")

    lds_pong_offset = allocator_pong._align(allocator_pong.ptr, 16)
    allocator_pong.ptr = lds_pong_offset + lds_tile_bytes

    lds_ping_offset = allocator_ping._align(allocator_ping.ptr, 16)
    allocator_ping.ptr = lds_ping_offset + lds_tile_bytes

    def _v16():
        return T.f8x16

    # ------------------------------------------------------------------
    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_b: fx.Tensor,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        from flydsl._mlir.dialects import llvm, memref as memref_dialect
        from flydsl._mlir.dialects import math as math_dialect

        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        by = gpu.block_id("y")

        acc_init = arith.constant_vector(0.0, T.f32x4)

        # ---- B layout ----
        layout_b = fx.make_layout(
            (n0_val, k0_val, 4, 16, kpack_elems),
            (_stride_n0, _stride_k0, _stride_klane, _stride_nlane, 1),
        )

        k_blocks16 = arith.index(tile_k_bytes // 16)
        _lds_k_dim_c = fx.Index(tile_k)

        # ---- LDS ----
        base_ptr_pong = allocator_pong.get_base()
        base_ptr_ping = allocator_ping.get_base()
        lds_a_pong = SmemPtr(base_ptr_pong, lds_pong_offset, T.f8, shape=(lds_tile_bytes,)).get()
        lds_a_ping = SmemPtr(base_ptr_ping, lds_ping_offset, T.f8, shape=(lds_tile_bytes,)).get()

        # ---- Buffer resources ----
        a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=True)
        b_rsrc = buffer_ops.create_buffer_resource(arg_b, max_size=True)
        c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=True)
        scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=True)
        scale_b_rsrc = buffer_ops.create_buffer_resource(arg_scale_b, max_size=True)

        c_n = arith.index_cast(T.index, i32_n)

        bx_m = bx * tile_m
        by_n = by * tile_n

        # ---- Wave / lane ----
        layout_wave_lane = fx.make_layout((4, wave_size), (64, 1))
        coord_wave_lane = fx.idx2crd(tx, layout_wave_lane)
        wave_id = fx.get(coord_wave_lane, 0)
        lane_id = fx.get(coord_wave_lane, 1)

        layout_lane16 = fx.make_layout((4, 16), (16, 1))
        coord_lane16 = fx.idx2crd(lane_id, layout_lane16)
        lane_div_16 = fx.get(coord_lane16, 0)
        lane_mod_16 = fx.get(coord_lane16, 1)

        row_a_lds = lane_mod_16
        col_offset_base_bytes = lane_div_16 * kpack_elems
        n_tile_base = wave_id * n_per_wave

        # ---- B load: per-wave N indices ----
        n_intra_list = []
        n_blk_list = []
        for i in range_constexpr(num_acc_n):
            global_n = by_n + n_tile_base + (i * 16) + lane_mod_16
            n_blk_list.append(global_n // 16)
            n_intra_list.append(global_n % 16)

        c64_b = 64

        def load_b_packs_k64(base_k, ku, ni):
            base_k_bytes = base_k
            k0_base = base_k_bytes // c64_b
            k0 = k0_base + ku
            k1 = lane_div_16
            coord_pack = (n_blk_list[ni], k0, k1, n_intra_list[ni], fx.Index(0))
            idx_pack = crd2idx(coord_pack, layout_b)
            b16 = _buffer_load_vec(
                buffer_ops, vector, b_rsrc, idx_pack,
                elem_type=T.f8, vec_elems=16, elem_bytes=elem_bytes,
                offset_in_bytes=True,
            )
            b_i64x2 = vector.bitcast(T.i64x2, b16)
            b0 = vector.extract(b_i64x2, static_position=[0], dynamic_position=[])
            b1 = vector.extract(b_i64x2, static_position=[1], dynamic_position=[])
            return b0, b1

        def load_b_tile(base_k):
            b_tile = []
            for ku in range_constexpr(k_unroll):
                packs0 = []
                packs1 = []
                for ni in range_constexpr(num_acc_n):
                    b0, b1 = load_b_packs_k64(base_k, ku, ni)
                    packs0.append(b0)
                    packs1.append(b1)
                b_tile.append((packs0, packs1))
            return b_tile

        # ---- DMA A: global -> LDS ----
        la_tile_dma = fx.make_layout((tile_m, tile_k_dwords), (tile_k_dwords, 1))
        tx_dma_base = tx * fx.Index(dma_dwords)
        c4 = fx.Index(4)

        def dma_coord(i):
            return tile_chunk_coord_i32(
                arith, tx_i32_base=tx_dma_base, i=i,
                total_threads=total_threads,
                layout_tile_div4=la_tile_dma,
                chunk_i32=dma_dwords,
            )

        def dma_a_to_lds(base_k, lds_buf):
            dma_size = arith.constant(dma_bytes, type=T.i32)
            soffset_c = arith.constant(0, type=T.i32)
            offset_c = arith.constant(0, type=T.i32)
            aux_c = arith.constant(1, type=T.i32)

            for i in range_constexpr(num_dma_ops):
                row_local, col_dword = dma_coord(i)
                col_swz = swizzle_xor16(row_local, col_dword * c4, k_blocks16)
                row_global = bx_m + row_local
                global_byte = row_global * fx.Index(k_bytes_factor) + (base_k + col_swz)
                voffset = arith.index_cast(T.i32, global_byte)

                if i == 0:
                    lds_addr = (memref_dialect.extract_aligned_pointer_as_index(lds_buf)
                                + wave_id * fx.Index(wave_size * dma_bytes))
                    lds_ptr_i64 = rocdl.readfirstlane(T.i64, arith.index_cast(T.i64, lds_addr))
                else:
                    lds_ptr_i64 = lds_ptr_i64 + arith.constant(total_threads * dma_bytes, type=T.i64)
                lds_ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), lds_ptr_i64)

                rocdl.raw_ptr_buffer_load_lds(
                    a_rsrc, lds_ptr, dma_size, voffset, soffset_c, offset_c, aux_c)

        # ---- LDS -> register ----
        def lds_load_packs_k64(curr_row, col_base, lds_buf):
            col_swz = swizzle_xor16(curr_row, col_base, k_blocks16)
            idx = curr_row * _lds_k_dim_c + col_swz
            loaded = vector.load_op(_v16(), lds_buf, [idx])
            v2 = vector.bitcast(T.i64x2, loaded)
            return (vector.extract(v2, static_position=[0], dynamic_position=[]),
                    vector.extract(v2, static_position=[1], dynamic_position=[]))

        # ---- MFMA ----
        mfma_res_ty = T.f32x4

        def pack_i64x4(x0, x1, x2, x3):
            v4 = vector.from_elements(T.vec(4, T.i64), [x0, x1, x2, x3])
            return vector.bitcast(T.vec(8, T.i32), v4)

        # ---- Block-scale ----
        c_scale_block_k = fx.Index(scale_block_k)
        c_scale_k = fx.Index(scale_k)
        c_128 = fx.Index(128)
        c_M = fx.Index(M)
        row_off_base = lane_div_16 * 4

        def load_scales_for_tile(k_base):
            all_combined = []
            for sb in range_constexpr(sb_per_tile):
                kb = k_base // c_scale_block_k + fx.Index(sb)
                sa_base = kb * c_M
                s_a_vecs = []
                for mi in range_constexpr(m_repeat):
                    row_g = bx_m + arith.index(mi * 16) + row_off_base
                    s_a_vecs.append(vector.bitcast(T.f32x4,
                        buffer_ops.buffer_load(scale_a_rsrc, sa_base + row_g,
                                               vec_width=4, dtype=T.f32)))
                s_b_vals = []
                for ni in range_constexpr(num_acc_n):
                    col_ni = by_n + n_tile_base + arith.index(ni * 16)
                    n_block = col_ni // c_128
                    s_b_vals.append(
                        buffer_ops.buffer_load(scale_b_rsrc, n_block * c_scale_k + kb,
                                               vec_width=1, dtype=T.f32))
                combined = []
                for mi in range_constexpr(m_repeat):
                    mi_c = []
                    for ni in range_constexpr(num_acc_n):
                        sb_v4 = vector.broadcast(T.f32x4, s_b_vals[ni])
                        mi_c.append(ArithValue(s_a_vecs[mi]) * ArithValue(sb_v4))
                    combined.append(mi_c)
                all_combined.append(combined)
            return all_combined

        # ---- Compute tile ----
        def compute_tile_blockscale(global_accs, b_tile_in, lds_buffer, pre_scales):
            current = list(global_accs)
            for sb in range_constexpr(sb_per_tile):
                combined = pre_scales[sb]
                block_accs = [acc_init] * n_accs

                ku0 = sb * ku_per_sb
                ku1 = ku0 + 1
                b0_p0, b0_p1 = b_tile_in[ku0]
                b1_p0, b1_p1 = b_tile_in[ku1]
                col0 = col_offset_base_bytes + (ku0 * 64)
                col1 = col_offset_base_bytes + (ku1 * 64)

                for mi in range_constexpr(m_repeat):
                    row = row_a_lds + (mi * 16)
                    a0, a1 = lds_load_packs_k64(row, col0, lds_buffer)
                    a2, a3 = lds_load_packs_k64(row, col1, lds_buffer)
                    a128 = pack_i64x4(a0, a1, a2, a3)
                    for ni in range_constexpr(num_acc_n):
                        b128 = pack_i64x4(b0_p0[ni], b0_p1[ni], b1_p0[ni], b1_p1[ni])
                        acc_idx = mi * num_acc_n + ni
                        block_accs[acc_idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [a128, b128, block_accs[acc_idx],
                             0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F],
                        )

                for mi in range_constexpr(m_repeat):
                    for ni in range_constexpr(num_acc_n):
                        acc_idx = mi * num_acc_n + ni
                        current[acc_idx] = math_dialect.fma(
                            block_accs[acc_idx], combined[mi][ni], current[acc_idx])
            return current

        # ---- Epilogue ----
        def store_output(final_accs):
            def body_row(*, mi, ii, row_in_tile, row):
                col_base = by_n + n_tile_base + lane_mod_16
                idx_base = row * c_n + col_base
                for ni in range_constexpr(num_acc_n):
                    acc = final_accs[mi * num_acc_n + ni]
                    val = vector.extract(acc, static_position=[ii], dynamic_position=[])
                    val_out = arith.trunc_f(T.bf16, val)
                    buffer_ops.buffer_store(val_out, c_rsrc, idx_base + (ni * 16))

            mfma_epilog(
                use_cshuffle=False, arith=arith, range_constexpr=range_constexpr,
                m_repeat=m_repeat, lane_div_16=lane_div_16, bx_m=bx_m, body_row=body_row,
            )

        # ================================================================
        # MAIN PIPELINE
        # ================================================================
        rocdl.sched_barrier(0)

        # Prologue: DMA A[k=0]->pong, A[k=1]->ping
        dma_a_to_lds(fx.Index(0), lds_a_pong)
        dma_a_to_lds(fx.Index(tile_k), lds_a_ping)

        # Load B[k=0] and scales[k=0]
        b_tile_cur = load_b_tile(fx.Index(0))
        scales_cur = load_scales_for_tile(fx.Index(0))

        # Wait for pong DMA, k=1 DMA can still be in flight
        rocdl.s_waitcnt(num_dma_ops)
        gpu.barrier()

        global_accs = [acc_init] * n_accs

        # Main loop: k=0..k_iters-2
        for k_it in range_constexpr(k_iters - 1):
            is_even = (k_it % 2) == 0
            cur_lds = lds_a_pong if is_even else lds_a_ping
            nxt_lds = lds_a_ping if is_even else lds_a_pong

            # Compute current tile
            global_accs = compute_tile_blockscale(global_accs, b_tile_cur, cur_lds, scales_cur)

            # Interleave: DMA A[k+2] to nxt_lds (overlaps with tail of compute)
            next_k = k_it + 1
            if next_k < k_iters - 1:
                dma_a_to_lds(fx.Index((next_k + 1) * tile_k), nxt_lds)

            # Load B[k+1] + scales[k+1]
            next_bk = next_k * tile_k
            b_tile_nxt = load_b_tile(fx.Index(next_bk))
            scales_nxt = load_scales_for_tile(fx.Index(next_bk))

            rocdl.s_waitcnt(0)
            gpu.barrier()

            b_tile_cur = b_tile_nxt
            scales_cur = scales_nxt

        # Epilogue: last K tile
        is_even_last = ((k_iters - 1) % 2) == 0
        last_lds = lds_a_pong if is_even_last else lds_a_ping
        global_accs = compute_tile_blockscale(global_accs, b_tile_cur, last_lds, scales_cur)

        store_output(global_accs)

    # ------------------------------------------------------------------
    @flyc.jit
    def launch_gemm(
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
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator_pong.finalize()
            allocator_ping.finalize()

        gx = (i32_m + (tile_m - 1)) // tile_m
        gy = i32_n // tile_n

        launcher = kernel_gemm(arg_c, arg_a, arg_b, arg_scale_a, arg_scale_b, i32_m, i32_n)
        for op in ctx.gpu_module_body.operations:
            if hasattr(op, 'attributes') and op.OPERATION_NAME == "gpu.func":
                op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(T.i32, 1)
        launcher.launch(
            grid=(gx, gy, 1),
            block=(total_threads, 1, 1),
            stream=stream,
        )

    return launch_gemm
