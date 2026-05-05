import os
import sys
import time

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLYDSL_SRC not in sys.path:
    sys.path.insert(0, _PYFLYDSL_SRC)

from tests.test_common import verify_output
from tests.utils import pertoken_quant

FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16


def compile_fp8_gemm_4wave(
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

    A_lds_cur0_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_0")
    A_lds_cur1_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_1")
    A_lds_next0_alloc = SmemAllocator(None, "gfx950", "A_lds_next_0")
    A_lds_next1_alloc = SmemAllocator(None, "gfx950", "A_lds_next_1")
    B_lds_cur0_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_0")
    B_lds_cur1_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_1")
    B_lds_next0_alloc = SmemAllocator(None, "gfx950", "B_lds_next_0")
    B_lds_next1_alloc = SmemAllocator(None, "gfx950", "B_lds_next_1")

    # half size
    a_lds_size = (BLOCK_M // 2) * BLOCK_K
    b_lds_size = (BLOCK_N // 2) * BLOCK_K

    A_lds_cur0_alloc.ptr = a_lds_size
    A_lds_cur1_alloc.ptr = a_lds_size
    A_lds_next0_alloc.ptr = a_lds_size
    A_lds_next1_alloc.ptr = a_lds_size
    B_lds_cur0_alloc.ptr = b_lds_size
    B_lds_cur1_alloc.ptr = b_lds_size
    B_lds_next0_alloc.ptr = b_lds_size
    B_lds_next1_alloc.ptr = b_lds_size

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
    ):
        # === Type declarations ===
        MfmaAccumType_t = Vec.make_type(4, fx.Float32)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        Vec16_t = Vec.make_type(16, fx.Float8E4M3FN)
        # Initial value for the C register tile
        RT_C_i = Vec.filled(4, 0.0, fx.Float32)

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

        tile_i = fx.block_idx.x // N_BLOCKS
        tile_j = fx.block_idx.x % N_BLOCKS
        wave_i = wave_id // 2
        wave_j = wave_id % 2
        A0_gl_offset = (tile_i * BLOCK_M) * K
        A128_gl_offset = (tile_i * BLOCK_M + 128) * K
        B0_gl_offset = (tile_j * BLOCK_N) * K
        B128_gl_offset = (tile_j * BLOCK_N + 128) * K

        A_rsrc = buffer_ops.create_buffer_resource(A)
        B_rsrc = buffer_ops.create_buffer_resource(B_T)
        C_rsrc = buffer_ops.create_buffer_resource(C)

        A_scale_rsrc = buffer_ops.create_buffer_resource(A_scale)
        B_scale_rsrc = buffer_ops.create_buffer_resource(B_scale)

        def _c_idx(i, j):
            return i * 4 + j

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

        def _compute_lds_swizzle(wave_idx):
            lds_swz = []
            for row_offset in range_constexpr(4):
                row = wave_idx * 64 + row_offset * 16 + lane_id % 16
                swz = []
                for i in range_constexpr(2):
                    col = (lane_id // 16) * 16 + i * 64
                    swz_row, swz_col = _swizzle_128(row, col)
                    swz.append(swz_row * 128 + swz_col)
                lds_swz.append(swz)
            return lds_swz

        def _load_lds(gl_src, lds_dst, k_offset, gl_offsets):
            from flydsl._mlir.dialects import memref as memref_dialect
            lds_base_i = memref_dialect.extract_aligned_pointer_as_index(lds_dst)
            for step in range_constexpr(4):
                lds_ptr = buffer_ops.create_llvm_ptr(
                    fx.Int64(lds_base_i + fx.Index(wave_id * 1024 + step * 4096)),
                    address_space=3
                )
                rocdl.raw_ptr_buffer_load_lds(
                    gl_src, lds_ptr,
                    fx.Int32(16),
                    fx.Int32(gl_offsets[step]), # voffset
                    fx.Int32(k_offset), # soffset
                    fx.Int32(0),
                    fx.Int32(0)
                )

        def _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, step):
            from flydsl._mlir.dialects import memref as memref_dialect
            lds_base_i = memref_dialect.extract_aligned_pointer_as_index(lds_dst)
            lds_ptr = buffer_ops.create_llvm_ptr(
                fx.Int64(lds_base_i + fx.Index(wave_id * 1024 + step * 4096)),
                address_space=3
            )
            rocdl.raw_ptr_buffer_load_lds(
                gl_src, lds_ptr,
                fx.Int32(16),
                fx.Int32(gl_offsets[step]), # voffset
                fx.Int32(k_offset), # soffset
                fx.Int32(0),
                fx.Int32(0)
            )

        def _pack_i32x42_i32x8(lo, hi):
            # Pack 2 i32x4 as i32x8
            return lo.shuffle(hi, list(range(8)))

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
                    v = Vec.load(Vec16_t, lds_src, [fx.Index(row_swz * 128 + col_swz)])
                    halves.append(v.bitcast(fx.Int32)) # i32x4
                frag.append(_pack_i32x42_i32x8(halves[0], halves[1])) # i32x8
            return frag

        def _load_one_rt(lds_src, lds_swz, row, k):
            # Load half of a 16x128 tile from LDS to registers
            v = Vec.load(Vec16_t, lds_src, [fx.Index(lds_swz[row][k])])
            return v.bitcast(fx.Int32) # return a i32x4

        def _store_rt(c_frag, base_row, base_col):
            for ti in range_constexpr(4):
                row = base_row + ti * 16 + (lane_id // 16) * 4
                a_scale_v = Vec(buffer_ops.buffer_load(A_scale_rsrc, fx.Int32(row), vec_width=4, dtype=fx.Float32))
                for tj in range_constexpr(4):
                    col = base_col + tj * 16 + lane_id % 16
                    b_scale = buffer_ops.buffer_load(B_scale_rsrc, fx.Int32(col), vec_width=1, dtype=fx.Float32)
                    vec_f32 = Vec(c_frag[_c_idx(ti, tj)])
                    for i in range_constexpr(4):
                        scaled = (vec_f32[i] * (a_scale_v[i] * b_scale)).to(fx.BFloat16)
                        buffer_ops.buffer_store(scaled, C_rsrc, fx.Int32((row + i) * N + col))

        def _mfma_ABt(a, b, c, m, n):
            c[_c_idx(m, n)] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(MfmaAccumType_t, [a[m], b[n], c[_c_idx(m, n)], 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F])
            return c

        def _mfma_ABt_all(a, b, c):
            for i in range_constexpr(4):
                for j in range_constexpr(4):
                    c[_c_idx(i, j)] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(MfmaAccumType_t, [a[i], b[j], c[_c_idx(i, j)], 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F])
            return c

        def _wait_barrier(count):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
                constraints="",
                has_side_effects=True
            )

        def _interleaved_cluster(lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, a, b, c):
            # Compute a 64x64 output tile using 4x4 MFMA instructions
            # returns the updated accumulator and the next fragment loaded from lds_src
            rt_dst = []

            rocdl.sched_barrier(0)
            c = _mfma_ABt(a, b, c, 0, 0)
            c = _mfma_ABt(a, b, c, 0, 1)
            rocdl.sched_barrier(0)

            lds_swz = _compute_lds_swizzle(wave_idx)
            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 0)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 0, 0)

            rocdl.sched_barrier(0)
            c = _mfma_ABt(a, b, c, 0, 2)
            rocdl.sched_barrier(0)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 0, 1)
            rt_dst.append(_pack_i32x42_i32x8(rt_dst_0, rt_dst_1))

            rocdl.sched_barrier(0)
            c = _mfma_ABt(a, b, c, 0, 3)
            rocdl.sched_barrier(0)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 1)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 1, 0)

            rocdl.sched_barrier(0)
            c = _mfma_ABt(a, b, c, 1, 0)
            c = _mfma_ABt(a, b, c, 1, 1)
            rocdl.sched_barrier(0)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 1, 1)
            rt_dst.append(_pack_i32x42_i32x8(rt_dst_0, rt_dst_1))

            rocdl.sched_barrier(0)
            c = _mfma_ABt(a, b, c, 1, 2)
            c = _mfma_ABt(a, b, c, 1, 3)
            rocdl.sched_barrier(0)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 2)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 2, 0)

            rocdl.sched_barrier(0)
            c = _mfma_ABt(a, b, c, 2, 0)
            c = _mfma_ABt(a, b, c, 2, 1)
            rocdl.sched_barrier(0)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 2, 1)
            rt_dst.append(_pack_i32x42_i32x8(rt_dst_0, rt_dst_1))

            rocdl.sched_barrier(0)
            c = _mfma_ABt(a, b, c, 2, 2)
            c = _mfma_ABt(a, b, c, 2, 3)
            rocdl.sched_barrier(0)

            _load_one_lds(gl_src, lds_dst, k_offset, gl_offsets, 3)
            rt_dst_0 = _load_one_rt(lds_src, lds_swz, 3, 0)

            rocdl.sched_barrier(0)
            c = _mfma_ABt(a, b, c, 3, 0)
            c = _mfma_ABt(a, b, c, 3, 1)
            rocdl.sched_barrier(0)

            rt_dst_1 = _load_one_rt(lds_src, lds_swz, 3, 1)
            rt_dst.append(_pack_i32x42_i32x8(rt_dst_0, rt_dst_1))

            rocdl.sched_barrier(0)
            c = _mfma_ABt(a, b, c, 3, 2)
            c = _mfma_ABt(a, b, c, 3, 3)
            rocdl.sched_barrier(0)

            return c, rt_dst


        # Each wave handles 2x2 64x64 sub-tiles of the output
        c00_frag = [RT_C_i] * 16
        c01_frag = [RT_C_i] * 16
        c10_frag = [RT_C_i] * 16
        c11_frag = [RT_C_i] * 16

        global_offsets = _compute_global_swizzle()

        # Prologue: pre-load A/B cur
        _load_lds(A_rsrc, a_cur0, A0_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, b_cur0, B0_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, b_cur1, B128_gl_offset + 0 * BLOCK_K, global_offsets)
        _load_lds(A_rsrc, a_cur1, A128_gl_offset + 0 * BLOCK_K, global_offsets)

        # Issue load for next tile
        _load_lds(A_rsrc, a_next0, A0_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, b_next0, B0_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(B_rsrc, b_next1, B128_gl_offset + 1 * BLOCK_K, global_offsets)
        _load_lds(A_rsrc, a_next1, A128_gl_offset + 1 * BLOCK_K, global_offsets)

        rocdl.sched_barrier(0)
        _wait_barrier(28)
        rocdl.sched_barrier(0)

        a0_frag = _load_rt(a_cur0, wave_i)

        rocdl.sched_barrier(0)
        _wait_barrier(24)
        rocdl.sched_barrier(0)

        b0_frag = _load_rt(b_cur0, wave_j)

        for k in range_constexpr(K_ITERS - 2):
            rocdl.sched_barrier(0)
            _wait_barrier(16)
            rocdl.sched_barrier(0)

            c00_frag, b1_frag = _interleaved_cluster(
                a_cur0, A_rsrc, A0_gl_offset + (k + 2) * BLOCK_K, global_offsets,
                wave_j, b_cur1, a0_frag, b0_frag, c00_frag
            )

            c01_frag, a1_frag = _interleaved_cluster(
                b_cur0, B_rsrc, B0_gl_offset + (k + 2) * BLOCK_K, global_offsets,
                wave_i, a_cur1, a0_frag, b1_frag, c01_frag
            )

            rocdl.sched_barrier(0)
            _wait_barrier(16)
            rocdl.sched_barrier(0)

            c10_frag, a0_frag = _interleaved_cluster(
                b_cur1, B_rsrc, B128_gl_offset + (k + 2) * BLOCK_K, global_offsets,
                wave_i, a_next0, a1_frag, b0_frag, c10_frag
            )

            c11_frag, b0_frag = _interleaved_cluster(
                a_cur1, A_rsrc, A128_gl_offset + (k + 2) * BLOCK_K, global_offsets,
                wave_j, b_next0, a1_frag, b1_frag, c11_frag
            )

            # Swap cur and next
            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # step k = k_iters - 2
        rocdl.sched_barrier(0)
        _wait_barrier(16)
        rocdl.sched_barrier(0)

        b1_frag = _load_rt(b_cur1, wave_j)

        rocdl.sched_barrier(0)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        rocdl.sched_barrier(0)

        a1_frag = _load_rt(a_cur1, wave_i)

        rocdl.sched_barrier(0)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        rocdl.sched_barrier(0)

        rocdl.sched_barrier(0)
        _wait_barrier(8)
        rocdl.sched_barrier(0)

        a0_frag = _load_rt(a_next0, wave_i)

        rocdl.sched_barrier(0)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        rocdl.sched_barrier(0)

        b0_frag = _load_rt(b_next0, wave_j)

        rocdl.sched_barrier(0)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
        rocdl.sched_barrier(0)

        # Swap cur and next
        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # step k = k_iters - 1
        base_row = tile_i * BLOCK_M + wave_i * 64
        base_col = tile_j * BLOCK_N + wave_j * 64

        rocdl.sched_barrier(0)
        _wait_barrier(0)
        rocdl.sched_barrier(0)

        b1_frag = _load_rt(b_cur1, wave_j)
        a1_frag = _load_rt(a_cur1, wave_i)

        rocdl.sched_barrier(0)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        rocdl.sched_barrier(0)

        rocdl.sched_barrier(0)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        rocdl.sched_barrier(0)

        rocdl.sched_barrier(0)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        rocdl.sched_barrier(0)

        rocdl.sched_barrier(0)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)
        rocdl.sched_barrier(0)

        _store_rt(c00_frag, base_row + 0, base_col + 0)
        _store_rt(c01_frag, base_row + 0, base_col + 128)
        _store_rt(c10_frag, base_row + 128, base_col + 0)
        _store_rt(c11_frag, base_row + 128, base_col + 128)



    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        stream: fx.Stream
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
        grid_x = (M * N) // (256 * 256)
        kernel_gemm(A, B_T, C, A_scale, B_scale).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm


def run_torch(a, b, scale_a, scale_b, dtype=torch.float32):
    if scale_a is not None and scale_b is not None:
        a_f32 = a.to(torch.float32) * scale_a.view(-1, 1)
        b_f32 = b.to(torch.float32) * scale_b.view(-1, 1)
    else:
        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)
    c = torch.mm(a_f32, b_f32.T)
    return c.to(dtype)

def check_gemm(size: int):
    M = N = K = size
    device = torch.device("cuda")
    a_fp32 = torch.rand(M, K, device=device, dtype=torch.float32)
    b_fp32_t = torch.rand(N, K, device=device, dtype=torch.float32)
    c_out_raw = torch.zeros((M, N), dtype=OUT_DTYPE, device=device)
    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=FP8_DTYPE)
    b_q, scale_b = pertoken_quant(b_fp32_t, quant_dtype=FP8_DTYPE)

    a_q = a_q.contiguous()
    b_q = b_q.contiguous()
    scale_a = scale_a.squeeze().contiguous()
    scale_b = scale_b.squeeze().contiguous()

    out_ref = run_torch(a_q, b_q, scale_a, scale_b)

    launch_fn = compile_fp8_gemm_4wave(M=M, N=N, K=K)
    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def _args(c):
        return (
            _as_i8(a_q).contiguous().view(-1),
            _as_i8(b_q).contiguous().view(-1),
            c.contiguous().view(-1),
            scale_a.contiguous(),
            scale_b.contiguous(),
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(c_out_raw))

    compiled(*_args(c_out_raw))
    torch.cuda.synchronize()

    assert verify_output(c_out_raw.to(torch.float32), out_ref, rtol=0.1, atol=0.1)

    if True:
        best = float("+inf")
        for _ in range(1000):
            s = time.perf_counter()
            compiled(*_args(c_out_raw))
            torch.cuda.synchronize()

            t = time.perf_counter() - s
            best = best if t > best else t

        tflops = (2 * M * N * K) * 1e-12 / best

        print(f'FP8 GEMM SIZE={size} TFLOPS={round(tflops, 1)}')


if __name__ == "__main__":
    check_gemm(1024*12)
