import torch
import torch.nn.functional as F

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, range_constexpr, rocdl, gpu
from flydsl.expr.typing import BFloat16
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16


# Simple GEMM with fp8 input bf16 output 16x16 tiles, 1 wave
def compile_simple_gemm(
        *,
        M: int,
        N: int,
        K: int,
):
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = 128

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K

    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor
    ):
        lane_id = fx.thread_idx.x
        block_i = fx.block_idx.x // N_BLOCKS
        block_j = fx.block_idx.x % N_BLOCKS

        a_buffer_desc = buffer_ops.create_buffer_resource(A)
        b_buffer_desc = buffer_ops.create_buffer_resource(B_T)
        c_buffer_desc = buffer_ops.create_buffer_resource(C)

        def _load_operand(buff_desc, tile_offset, k_offset):
            row = lane_id % 16
            col = lane_id // 16
            base_offset = (row * K + k_offset + tile_offset) // 4
            elements = []
            for step in range_constexpr(2):
                v = Vec(buffer_ops.buffer_load(buff_desc, fx.Int32(base_offset + col * 4 + step * 16), vec_width=4, dtype=fx.Int32))
                for el_idx in range_constexpr(4):
                    elements.append(v[el_idx])
            return Vec.from_elements(elements, fx.Int32)

        def _store(C_frag):
            group = lane_id // 16
            row = block_i * BLOCK_M + group * 4
            col = lane_id % 16 + block_j * BLOCK_N
            bf16_vec = Vec(C_frag).to(BFloat16)
            buffer_ops.buffer_store(bf16_vec[0], c_buffer_desc, fx.Int32((row + 0) * N + col))
            buffer_ops.buffer_store(bf16_vec[1], c_buffer_desc, fx.Int32((row + 1) * N + col))
            buffer_ops.buffer_store(bf16_vec[2], c_buffer_desc, fx.Int32((row + 2) * N + col))
            buffer_ops.buffer_store(bf16_vec[3], c_buffer_desc, fx.Int32((row + 3) * N + col))

        mfma_accum_type = Vec.make_type(4, fx.Float32)
        accum = Vec.filled(4, 0.0, fx.Float32)
        A_offset = block_i * BLOCK_M * K
        B_offset = block_j * BLOCK_N * K
        for k_iter in range_constexpr(K_ITERS):
            k_offset = k_iter * BLOCK_K

            a_frag = _load_operand(a_buffer_desc, A_offset, k_offset)
            b_frag = _load_operand(b_buffer_desc, B_offset, k_offset)

            accum = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                mfma_accum_type, [a_frag, b_frag, accum, 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F]
            )

        _store(accum)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream
    ):
        grid_x = (M * N) // (16 * 16)
        kernel_gemm(A, B_T, C).launch(grid=(grid_x, 1, 1), block=(64, 1, 1), stream=stream)

    return launch_gemm


def compile_4wave_gemm(
        *,
        M: int,
        N: int,
        K: int
):
    BLOCK_M = 256
    BLOCK_N = 256
    BLOCK_K = 128

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K // BLOCK_K

    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0

    A_lds_alloc = SmemAllocator(None, "gfx950", "smem0")
    B_lds_alloc = SmemAllocator(None, "gfx950", "smem1")
    bytes_buff = 128 * 128 #half size
    A_lds_off_1 = A_lds_alloc._align(A_lds_alloc.ptr, 16)
    A_lds_off_2 = A_lds_off_1 + bytes_buff
    A_lds_alloc.ptr = A_lds_off_2 + bytes_buff

    B_lds_off_1 = B_lds_alloc._align(B_lds_alloc.ptr, 16)
    B_lds_off_2 = B_lds_off_1 + bytes_buff
    B_lds_alloc.ptr = B_lds_off_2 + bytes_buff

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor
    ):
        lane_id = fx.thread_idx.x % 64
        wave_id = fx.thread_idx.x // 64

        tile_i = fx.block_idx.x // N_BLOCKS
        tile_j = fx.block_idx.x % N_BLOCKS
        wave_i = wave_id // 2
        wave_j = wave_id % 2

        A_rsrc = buffer_ops.create_buffer_resource(A)
        B_rsrc = buffer_ops.create_buffer_resource(B_T)
        C_rsrc = buffer_ops.create_buffer_resource(C)

        fp8_ir = fx.Float8E4M3FN.ir_type
        vec_16_t = Vec.make_type(16, fx.Float8E4M3FN)
        accum_i = Vec.filled(4, 0.0, fx.Float32)
        mfma_accum_t = Vec.make_type(4, fx.Float32)

        A_lds_1st_half = SmemPtr(A_lds_alloc.get_base(), A_lds_off_1, fp8_ir, shape=(128*128,)).get()
        A_lds_2nd_half = SmemPtr(A_lds_alloc.get_base(), A_lds_off_2, fp8_ir, shape=(128*128,)).get()
        B_lds_1st_half = SmemPtr(B_lds_alloc.get_base(), B_lds_off_1, fp8_ir, shape=(128*128,)).get()
        B_lds_2nd_half = SmemPtr(B_lds_alloc.get_base(), B_lds_off_2, fp8_ir, shape=(128*128,)).get()

        from flydsl._mlir.dialects import memref as _memref_dialect

        def _memref_to_lds_ptr(m):
            base_idx = _memref_dialect.extract_aligned_pointer_as_index(m)
            return buffer_ops.create_llvm_ptr(fx.Int64(base_idx), address_space=3)

        A_lds_1st_ptr = _memref_to_lds_ptr(A_lds_1st_half)
        A_lds_2nd_ptr = _memref_to_lds_ptr(A_lds_2nd_half)
        B_lds_1st_ptr = _memref_to_lds_ptr(B_lds_1st_half)
        B_lds_2nd_ptr = _memref_to_lds_ptr(B_lds_2nd_half)

        def _cooperative_load(gl_src, gl_offset, lds_dst):
            for round in range_constexpr(4):
                row = lane_id // 8 + wave_id * 8 + round * 32
                col = (lane_id % 8) * 16
                voff = row * K + col
                lds_ptr_offset = buffer_ops.get_element_ptr(
                    lds_dst,
                    byte_offset=fx.Int32(wave_id * 1024 + round * 4096)
                )
                rocdl.raw_ptr_buffer_load_lds(
                    gl_src, lds_ptr_offset,
                    fx.Int32(16),
                    fx.Int32(voff),
                    fx.Int32(gl_offset), fx.Int32(0), fx.Int32(1)
                )

        def _acc_idx(i, j):
            return i * 4 + j

        def _load_rt(lds_src, wave_idx, row_offset):
            halves = []
            row = wave_idx * 64 + row_offset * 16 + lane_id % 16
            for step in range_constexpr(2):
                col = (lane_id // 16) * 16 + step * 64
                v = Vec.load(vec_16_t, lds_src, [fx.Index(row * 128 + col)])
                halves.append(v.bitcast(fx.Int32)) # i32x4
            return halves[0].shuffle(halves[1], list(range(8))) # i32x8

        def _store_rt(rt_src, base_row, base_col):
            for ti in range_constexpr(4):
                for tj in range_constexpr(4):
                    bf16_vec = Vec(rt_src[_acc_idx(ti, tj)]).to(BFloat16)
                    row = base_row + ti * 16 + (lane_id // 16) * 4
                    col = base_col + tj * 16 + lane_id % 16
                    buffer_ops.buffer_store(bf16_vec[0], C_rsrc, fx.Int32((row + 0) * N + col))
                    buffer_ops.buffer_store(bf16_vec[1], C_rsrc, fx.Int32((row + 1) * N + col))
                    buffer_ops.buffer_store(bf16_vec[2], C_rsrc, fx.Int32((row + 2) * N + col))
                    buffer_ops.buffer_store(bf16_vec[3], C_rsrc, fx.Int32((row + 3) * N + col))

        def _compute_cluster(acc, lds_A_src, lds_B_src):
            for ti in range_constexpr(4):
                a_frag = _load_rt(lds_A_src, wave_i, ti)
                for tj in range_constexpr(4):
                    b_frag = _load_rt(lds_B_src, wave_j, tj)
                    idx = _acc_idx(ti, tj)
                    acc[idx] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(mfma_accum_t, [a_frag, b_frag, acc[idx], 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F])
            return acc


        accum_00 = [accum_i] * 16
        accum_01 = [accum_i] * 16
        accum_10 = [accum_i] * 16
        accum_11 = [accum_i] * 16

        for k_iter in range_constexpr(K_ITERS):
            k_step = k_iter * BLOCK_K
            _cooperative_load(A_rsrc, (tile_i * BLOCK_M) * K + k_step, A_lds_1st_ptr) # 12
            _cooperative_load(B_rsrc, (tile_j * BLOCK_N) * K + k_step, B_lds_1st_ptr) # 8
            _cooperative_load(B_rsrc, (tile_j * BLOCK_N + 128) * K + k_step, B_lds_2nd_ptr) # 4
            _cooperative_load(A_rsrc, (tile_i * BLOCK_M + 128) * K + k_step, A_lds_2nd_ptr) # 0

            rocdl.s_waitcnt(0)
            gpu.barrier()

            _compute_cluster(accum_00, A_lds_1st_half, B_lds_1st_half)
            _compute_cluster(accum_01, A_lds_1st_half, B_lds_2nd_half)
            _compute_cluster(accum_10, A_lds_2nd_half, B_lds_1st_half)
            _compute_cluster(accum_11, A_lds_2nd_half, B_lds_2nd_half)

            gpu.barrier()


        base_row = tile_i * BLOCK_M + wave_i * 64
        base_col = tile_j * BLOCK_N + wave_j * 64
        _store_rt(accum_00, base_row+0, base_col+0)
        _store_rt(accum_01, base_row+0, base_col+128)
        _store_rt(accum_10, base_row+128, base_col+0)
        _store_rt(accum_11, base_row+128, base_col+128)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream
    ):
        from flydsl._mlir import ir
        from flydsl.compiler.kernel_function import CompilationContext
        A_lds_alloc.finalized = False
        B_lds_alloc.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            A_lds_alloc.finalize()
            B_lds_alloc.finalize()
        grid_x = (M * N) // (256 * 256)
        kernel_gemm(A, B_T, C).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm

def check_numbers(x: torch.Tensor, y: torch.Tensor, rtol: float, atol: float):
    diff = (x - y).abs()
    max_diff = diff.max().item()
    avg_diff = diff.mean().item()
    if not torch.allclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
        raise Exception(f"Kernel doesn't match PyTorch (max_diff={max_diff} avg_diff={avg_diff})")
    print(f'=== SUCCESS ===\n--> Kernel matches PyTorch within numerical precision (max_diff={max_diff} avg_diff={avg_diff})')


def check_gemm(size: int):
    M = N = K = size
    x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(FP8_DTYPE)
    weight = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(FP8_DTYPE)
    out = torch.zeros((M, N), dtype=OUT_DTYPE, device="cuda")

    out_ref = F.linear(x.to(torch.float32), weight.to(torch.float32)).to(OUT_DTYPE)

    launch_fn = compile_4wave_gemm(M=M, N=N, K=K)
    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def _args(c):
        return (
            _as_i8(x).contiguous().view(-1),
            _as_i8(weight).contiguous().view(-1),
            c.contiguous().view(-1),
            torch.cuda.current_stream(),
        )

    compiled = flyc.compile(launch_fn, *_args(out))

    compiled(*_args(out))
    torch.cuda.synchronize()

    check_numbers(out, out_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    check_gemm(1024)
