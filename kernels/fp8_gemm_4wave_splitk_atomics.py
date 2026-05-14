import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly as _fly
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects import scf as _scf
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr


def _divmod(a, b):
    return (a // b, a % b)


def _min(a, b):
    return arith.select(a < b, a, b)


def _xcd_swizzle(num_pid_m, num_pid_n):
    NUM_XCDS = 8
    WGM = 4
    NUM_CUS = 32 * NUM_XCDS
    SWIZZLE_THRESHOLD = 4 * NUM_CUS

    wgid = fx.block_idx.x

    num_wg = num_pid_m * num_pid_n

    if num_wg <= SWIZZLE_THRESHOLD or num_wg % NUM_XCDS != 0:
        # naive mapping
        return _divmod(wgid, num_pid_n)

    intra_xcd, xcd = _divmod(wgid, NUM_XCDS)
    wgid = xcd * (num_wg // NUM_XCDS) + intra_xcd
    num_wgid_in_group = WGM * num_pid_n
    group_id, intra_group = _divmod(wgid, num_wgid_in_group)
    first_pid_m = group_id * WGM
    group_size_m = _min(num_pid_m - first_pid_m, WGM)
    pid_n, intra_group_m = _divmod(intra_group, group_size_m)
    pid_m = first_pid_m + intra_group_m
    return (pid_m, pid_n)


def compile_fp8_gemm(
    *, M: int, N: int, K: int, BLOCK_M: int = 64, BLOCK_N: int = 64, NUM_SPLITS: int = 1, use_xcd_remap: bool = True
):
    # fixed for MFMA 16x16x128
    BLOCK_K = 128

    LDS_BLOCK_M = BLOCK_M // 2
    LDS_BLOCK_N = BLOCK_N // 2

    # The base mfma atom is 16x16, we use 4 waves in a 2x2 config so the block size must be at least 64 to keep this config
    assert BLOCK_M >= 64 and BLOCK_M % 64 == 0
    assert BLOCK_N >= 64 and BLOCK_N % 64 == 0

    assert N % BLOCK_N == 0 and K % BLOCK_K == 0
    assert M >= 1

    assert NUM_SPLITS >= 1
    assert K % NUM_SPLITS == 0, f"K ({K}) must be divisible by NUM_SPLITS ({NUM_SPLITS})"
    K_PER_SPLIT = K // NUM_SPLITS
    assert K_PER_SPLIT % BLOCK_K == 0, f"K_PER_SPLIT ({K_PER_SPLIT}) must be divisible by BLOCK_K ({BLOCK_K})"
    K_ITERS_PER_SPLIT = K_PER_SPLIT // BLOCK_K
    assert K_ITERS_PER_SPLIT >= 2, (
        f"Each split needs >= 2 K iterations for double-buffered prologue, "
        f"got {K_ITERS_PER_SPLIT} (K={K}, NUM_SPLITS={NUM_SPLITS}, BLOCK_K={BLOCK_K})"
    )
    IS_SPLIT_K = NUM_SPLITS > 1

    N_BLOCKS = N // BLOCK_N
    K_ITERS = K_PER_SPLIT // BLOCK_K

    N_TILES_A = BLOCK_M // 4 // 16
    N_TILES_B = BLOCK_N // 4 // 16

    N_ACCUMS = N_TILES_A * N_TILES_B
    assert N_ACCUMS > 0

    A_lds_cur0_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_0")
    A_lds_cur1_alloc = SmemAllocator(None, "gfx950", "A_lds_cur_1")
    A_lds_next0_alloc = SmemAllocator(None, "gfx950", "A_lds_next_0")
    A_lds_next1_alloc = SmemAllocator(None, "gfx950", "A_lds_next_1")
    B_lds_cur0_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_0")
    B_lds_cur1_alloc = SmemAllocator(None, "gfx950", "B_lds_cur_1")
    B_lds_next0_alloc = SmemAllocator(None, "gfx950", "B_lds_next_0")
    B_lds_next1_alloc = SmemAllocator(None, "gfx950", "B_lds_next_1")

    if IS_SPLIT_K:
        bc_lds_alloc = SmemAllocator(None, "gfx950", "bc_lds")
        bc_lds_alloc.ptr = 4  # 1 x i32

    # half size
    a_lds_size = LDS_BLOCK_M * BLOCK_K
    b_lds_size = LDS_BLOCK_N * BLOCK_K

    A_lds_cur0_alloc.ptr = a_lds_size
    A_lds_cur1_alloc.ptr = a_lds_size
    A_lds_next0_alloc.ptr = a_lds_size
    A_lds_next1_alloc.ptr = a_lds_size
    B_lds_cur0_alloc.ptr = b_lds_size
    B_lds_cur1_alloc.ptr = b_lds_size
    B_lds_next0_alloc.ptr = b_lds_size
    B_lds_next1_alloc.ptr = b_lds_size

    a_size_bytes = M * K
    b_size_bytes = N * K
    c_size_bytes = M * N * 2
    a_scale_size_bytes = M * 4
    b_scale_size_bytes = N * 4

    @flyc.kernel
    def kernel_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        semaphore: fx.Tensor,
        signal: fx.Tensor,
    ):
        _ptr_type = ir.Type.parse("!llvm.ptr<1>")
        _i64_type = T.i64

        MfmaAccum_t = Vec.make_type(4, fx.Float32)
        RT_C_i = Vec.filled(4, 0.0, fx.Float32)
        F8_IR_t = fx.Float8E4M3FN.ir_type
        Vec16_t = Vec.make_type(16, fx.Float8E4M3FN)

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
        tid = fx.thread_idx.x

        if const_expr(use_xcd_remap):
            tile_i, tile_j = _xcd_swizzle(M // BLOCK_M, N // BLOCK_N)
        else:
            tile_i, tile_j = _divmod(fx.block_idx.x, N_BLOCKS)

        wave_i = wave_id // 2
        wave_j = wave_id % 2

        split_k_idx = fx.block_idx.y
        k_base = split_k_idx * K_PER_SPLIT

        A0_gl_offset = (tile_i * BLOCK_M) * K + k_base
        A1_gl_offset = (tile_i * BLOCK_M + LDS_BLOCK_M) * K + k_base
        B0_gl_offset = (tile_j * BLOCK_N) * K + k_base
        B1_gl_offset = (tile_j * BLOCK_N + LDS_BLOCK_N) * K + k_base

        A_rsrc = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=a_size_bytes)
        B_rsrc = buffer_ops.create_buffer_resource(B_T, max_size=False, num_records_bytes=b_size_bytes)
        C_rsrc = buffer_ops.create_buffer_resource(C, max_size=False, num_records_bytes=c_size_bytes)

        A_scale_rsrc = buffer_ops.create_buffer_resource(A_scale, max_size=False, num_records_bytes=a_scale_size_bytes)
        B_scale_rsrc = buffer_ops.create_buffer_resource(B_scale, max_size=False, num_records_bytes=b_scale_size_bytes)

        def _get_llvm_ptr(tensor, offset, dtype_bytes):
            base_ptr = _fly.extract_aligned_pointer_as_index(_ptr_type, tensor)
            base_int = _llvm.PtrToIntOp(_i64_type, base_ptr).result
            byte_offset = arith.index_cast(T.i64, fx.Index(offset) * fx.Index(dtype_bytes))
            ptr_int = _llvm.AddOp(base_int, byte_offset, _llvm.IntegerOverflowFlags(0)).result
            ptr = _llvm.IntToPtrOp(_ptr_type, ptr_int).result
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def _wait_barrier(count):
            _llvm.inline_asm(
                res=None,
                operands_=[],
                asm_string=f"s_waitcnt vmcnt({count})\ns_barrier",
                constraints="",
                has_side_effects=True,
            )

        if const_expr(IS_SPLIT_K):
            bc_lds = SmemPtr(bc_lds_alloc.get_base(), 0, ir.IntegerType.get_signless(32), shape=(1,)).get()
            signal_idx = fx.Int32(fx.block_idx.x)

        def _zero_c():
            is_t0 = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            is_t0_if = _scf.IfOp(is_t0, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_if.then_block):
                sem_ptr = _get_llvm_ptr(semaphore, signal_idx, 4)
                prev = _llvm.AtomicRMWOp(
                    _llvm.AtomicBinOp.add, sem_ptr,
                    arith.constant(1, type=T.i32),
                    _llvm.AtomicOrdering.monotonic,
                    syncscope="agent", alignment=4,
                ).result
                vec_v = vector.from_elements(T.vec(1, T.i32), [prev])
                vector.store(vec_v, bc_lds, [fx.Index(0)], alignment=4)
                _scf.YieldOp([])
            _wait_barrier(0)
            loaded = vector.load_op(T.vec(1, T.i32), bc_lds, [fx.Index(0)])
            arrive_idx = fx.Index(vector.extract(loaded, static_position=[0], dynamic_position=[]))
            cond_first = arith.cmpi(arith.CmpIPredicate.eq, arrive_idx, fx.Index(0))
            cond_first_if = _scf.IfOp(cond_first, results_=[], has_else=False)
            with ir.InsertionPoint(cond_first_if.then_block):
                TILE_PAIRS = (BLOCK_M * BLOCK_N) // 2
                PAIRS_PER_THREAD = TILE_PAIRS // 256
                c_base_ptr_z = _fly.extract_aligned_pointer_as_index(_ptr_type, C)
                c_base_int_z = _llvm.PtrToIntOp(_i64_type, c_base_ptr_z).result
                zero_i32 = arith.constant(0, type=T.i32)
                for ei in range_constexpr(PAIRS_PER_THREAD):
                    flat_pair = tid + ei * 256
                    row = flat_pair // (BLOCK_N // 2)
                    col_pair = flat_pair % (BLOCK_N // 2)
                    row_abs = tile_i * BLOCK_M + row
                    in_bounds_z = arith.cmpi(arith.CmpIPredicate.ult, fx.Index(row_abs), fx.Index(M))
                    bounds_z_if = _scf.IfOp(in_bounds_z, results_=[], has_else=False)
                    with ir.InsertionPoint(bounds_z_if.then_block):
                        elem_offset = row_abs * N + tile_j * BLOCK_N + col_pair * 2
                        byte_offset = arith.index_cast(T.i64, fx.Index(elem_offset) * fx.Index(2))
                        addr_i64 = _llvm.AddOp(c_base_int_z, byte_offset, _llvm.IntegerOverflowFlags(0)).result
                        ptr = _llvm.IntToPtrOp(_ptr_type, addr_i64).result
                        ptr_v = ptr._value if const_expr(hasattr(ptr, "_value")) else ptr
                        _llvm.InlineAsmOp(
                            None, [ptr_v, zero_i32],
                            "global_store_dword $0, $1, off sc0 sc1", "v,v",
                            has_side_effects=True,
                        )
                        _scf.YieldOp([])
                _wait_barrier(0)
                is_t0_if2 = _scf.IfOp(is_t0, results_=[], has_else=False)
                with ir.InsertionPoint(is_t0_if2.then_block):
                    sig_ptr = _get_llvm_ptr(signal, signal_idx, 4)
                    _llvm.InlineAsmOp(
                        None, [sig_ptr, arith.constant(1, type=T.i32)],
                        "global_store_dword $0, $1, off sc0 sc1", "v,v",
                        has_side_effects=True,
                    )
                    _scf.YieldOp([])
                _wait_barrier(0)
                _scf.YieldOp([])

        def _split_k_barrier():
            is_t0 = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(tid), fx.Index(0))
            is_t0_if = _scf.IfOp(is_t0, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_if.then_block):
                init_cur = arith.constant(0, type=T.i32)
                w = _scf.WhileOp([T.i32], [init_cur])
                before = ir.Block.create_at_start(w.before, [T.i32])
                after = ir.Block.create_at_start(w.after, [T.i32])
                with ir.InsertionPoint(before):
                    cur = before.arguments[0]
                    need_wait = arith.CmpIOp(arith.CmpIPredicate.eq, cur, arith.constant(0, type=T.i32)).result
                    _scf.ConditionOp(need_wait, [cur])
                with ir.InsertionPoint(after):
                    sig_ptr = _get_llvm_ptr(signal, signal_idx, 4)
                    data = _llvm.InlineAsmOp(
                        T.i32, [sig_ptr],
                        "global_load_dword $0, $1, off sc1", "=v,v",
                        has_side_effects=True,
                    ).result
                    rocdl.s_waitcnt(0)
                    _scf.YieldOp([data])
                _scf.YieldOp([])
            rocdl.sched_barrier(0)
            _wait_barrier(0)
            is_t0_if2 = _scf.IfOp(is_t0, results_=[], has_else=False)
            with ir.InsertionPoint(is_t0_if2.then_block):
                sem_ptr = _get_llvm_ptr(semaphore, signal_idx, 4)
                arrive_idx = _llvm.AtomicRMWOp(
                    _llvm.AtomicBinOp.add, sem_ptr,
                    arith.constant(1, type=T.i32),
                    _llvm.AtomicOrdering.monotonic,
                    syncscope="agent", alignment=4,
                ).result
                cond_last = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(arrive_idx), fx.Index(2 * NUM_SPLITS - 1))
                cond_last_if = _scf.IfOp(cond_last, results_=[], has_else=False)
                with ir.InsertionPoint(cond_last_if.then_block):
                    sem_ptr_reset = _get_llvm_ptr(semaphore, signal_idx, 4)
                    _llvm.InlineAsmOp(
                        None, [sem_ptr_reset, arith.constant(0, type=T.i32)],
                        "global_store_dword $0, $1, off sc0 sc1", "v,v",
                        has_side_effects=True,
                    )
                    sig_ptr_reset = _get_llvm_ptr(signal, signal_idx, 4)
                    _llvm.InlineAsmOp(
                        None, [sig_ptr_reset, arith.constant(0, type=T.i32)],
                        "global_store_dword $0, $1, off sc0 sc1", "v,v",
                        has_side_effects=True,
                    )
                    _scf.YieldOp([])
                _scf.YieldOp([])
            _wait_barrier(0)

        def _swizzle_128(row, col):
            offset = row * 128 + col
            swizzle = ((offset % (16 * 128)) >> 8) << 4
            swizzled_offset = offset ^ swizzle
            return swizzled_offset // 128, swizzled_offset % 128

        def _compute_global_swizzle():
            offsets = []
            for round in range_constexpr(max(N_TILES_A, N_TILES_B)):
                row = lane_id // 8 + wave_id * 8 + round * 32
                col = (lane_id % 8) * 16
                a, b = _swizzle_128(row, col)
                offsets.append(a * K + b)
            return offsets

        def _load_lds(gl_src, lds_dst, k_offset, gl_offsets, n_tiles):
            assert len(gl_offsets) >= n_tiles

            lds_base_i = memref_dialect.extract_aligned_pointer_as_index(lds_dst)
            for step in range_constexpr(n_tiles):
                lds_ptr = buffer_ops.create_llvm_ptr(
                    fx.Int64(lds_base_i + fx.Index(wave_id * 1024 + step * 4096)), address_space=3
                )
                rocdl.raw_ptr_buffer_load_lds(
                    gl_src,
                    lds_ptr,
                    fx.Int32(16),
                    fx.Int32(gl_offsets[step]),  # voffset
                    fx.Int32(k_offset),  # soffset
                    fx.Int32(0),
                    fx.Int32(1),
                )

        def _pack_i32x4_i32x8(lo, hi):
            return lo.shuffle(hi, list(range(8)))

        def _load_rt(lds_src, wave_idx, n_tiles):
            frag = []
            for i in range_constexpr(n_tiles):
                row = wave_idx * (n_tiles * 16) + i * 16 + lane_id % 16
                halves = []
                for step in range_constexpr(2):
                    col = (lane_id // 16) * 16 + step * 64
                    row_swz, col_swz = _swizzle_128(row, col)
                    v = Vec.load(Vec16_t, lds_src, [fx.Index(row_swz * 128 + col_swz)])
                    halves.append(v.bitcast(fx.Int32))
                frag.append(_pack_i32x4_i32x8(halves[0], halves[1]))
            return frag

        def _store_C_scaled(c_frag, base_row, base_col):
            def _preload_a_scales():
                scales = []
                for i in range_constexpr(N_TILES_A):
                    row = base_row + i * 16 + (lane_id // 16) * 4
                    scales.append(
                        Vec(buffer_ops.buffer_load(A_scale_rsrc, fx.Int32(row), vec_width=4, dtype=fx.Float32))
                    )
                return scales

            def _preload_b_scales():
                scales = []
                for i in range_constexpr(N_TILES_B):
                    col = base_col + i * 16 + lane_id % 16
                    scales.append(buffer_ops.buffer_load(B_scale_rsrc, fx.Int32(col), vec_width=1, dtype=fx.Float32))
                return scales

            a_scales = _preload_a_scales()
            b_scales = _preload_b_scales()

            if const_expr(IS_SPLIT_K):
                c_base_ptr = _fly.extract_aligned_pointer_as_index(_ptr_type, C)
                c_base_int = _llvm.PtrToIntOp(_i64_type, c_base_ptr).result

            for ti in range_constexpr(N_TILES_A):
                row = base_row + ti * 16 + (lane_id // 16) * 4
                for tj in range_constexpr(N_TILES_B):
                    col = base_col + tj * 16 + lane_id % 16
                    vec_f32 = Vec(c_frag[_c_idx(ti, tj)])
                    for i in range_constexpr(4):
                        scaled = vec_f32[i] * (a_scales[ti][i] * b_scales[tj])
                        if const_expr(IS_SPLIT_K):
                            in_bounds = arith.cmpi(arith.CmpIPredicate.ult, fx.Index(row + i), fx.Index(M))
                            bounds_if = _scf.IfOp(in_bounds, results_=[], has_else=False)
                            with ir.InsertionPoint(bounds_if.then_block):
                                scaled_bf16 = scaled.to(fx.BFloat16)
                                elem_offset = (row + i) * N + col
                                byte_offset = arith.index_cast(T.i64, fx.Index(elem_offset) * fx.Index(2))
                                addr_i64 = _llvm.AddOp(c_base_int, byte_offset, _llvm.IntegerOverflowFlags(0)).result
                                ptr = _llvm.IntToPtrOp(_ptr_type, addr_i64).result
                                ptr_v = ptr._value if const_expr(hasattr(ptr, "_value")) else ptr
                                val_v = scaled_bf16.ir_value()
                                _llvm.AtomicRMWOp(
                                    _llvm.AtomicBinOp.fadd, ptr_v, val_v,
                                    _llvm.AtomicOrdering.monotonic,
                                    syncscope="agent", alignment=2,
                                )
                                _scf.YieldOp([])
                        else:
                            buffer_ops.buffer_store(scaled.to(fx.BFloat16), C_rsrc, fx.Int32((row + i) * N + col))

        def _c_idx(i, j):
            return i * N_TILES_B + j

        def _mfma_ABt_all(a, b, c):
            assert len(a) == N_TILES_A
            assert len(b) == N_TILES_B
            assert len(c) == N_TILES_A * N_TILES_B

            for i in range_constexpr(N_TILES_A):
                for j in range_constexpr(N_TILES_B):
                    c[_c_idx(i, j)] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        MfmaAccum_t, [a[i], b[j], c[_c_idx(i, j)], 0, 0, 0, 0x7F7F7F7F, 0, 0x7F7F7F7F]
                    )
            return c

        def _compute_block(lds_dst, gl_src, k_offset, gl_offsets, wave_idx, lds_src, n_tiles_lds, n_tiles_rt, a, b, c):
            _load_lds(gl_src, lds_dst, k_offset, gl_offsets, n_tiles_lds)
            rt_dst = _load_rt(lds_src, wave_idx, n_tiles_rt)
            c = _mfma_ABt_all(a, b, c)
            return c, rt_dst

        # Split-K: zero C and synchronize before compute
        if const_expr(IS_SPLIT_K):
            _zero_c()

        # Each wave handles 2x2 64x64 sub-tiles of the output
        c00_frag = [RT_C_i] * N_ACCUMS
        c01_frag = [RT_C_i] * N_ACCUMS
        c10_frag = [RT_C_i] * N_ACCUMS
        c11_frag = [RT_C_i] * N_ACCUMS

        global_offsets = _compute_global_swizzle()

        # Prologue: pre-load A/B cur
        _load_lds(A_rsrc, a_cur0, A0_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_A)
        _load_lds(B_rsrc, b_cur0, B0_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(B_rsrc, b_cur1, B1_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(A_rsrc, a_cur1, A1_gl_offset + 0 * BLOCK_K, global_offsets, N_TILES_A)

        # Issue load for next tile
        _load_lds(A_rsrc, a_next0, A0_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_A)
        _load_lds(B_rsrc, b_next0, B0_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(B_rsrc, b_next1, B1_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_B)
        _load_lds(A_rsrc, a_next1, A1_gl_offset + 1 * BLOCK_K, global_offsets, N_TILES_A)

        _wait_barrier((3 * N_TILES_A) + (4 * N_TILES_B))  # wait for a_cur0

        a0_frag = _load_rt(a_cur0, wave_i, N_TILES_A)

        _wait_barrier((3 * N_TILES_A) + (3 * N_TILES_B))  # wait for b_cur0

        b0_frag = _load_rt(b_cur0, wave_j, N_TILES_B)

        for k in range_constexpr(K_ITERS - 2):
            _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c00_frag, b1_frag = _compute_block(
                a_cur0, A_rsrc, A0_gl_offset + (k + 2) * BLOCK_K,
                global_offsets, wave_j, b_cur1,
                N_TILES_A, N_TILES_B, a0_frag, b0_frag, c00_frag,
            )

            c01_frag, a1_frag = _compute_block(
                b_cur0, B_rsrc, B0_gl_offset + (k + 2) * BLOCK_K,
                global_offsets, wave_i, a_cur1,
                N_TILES_B, N_TILES_A, a0_frag, b1_frag, c01_frag,
            )

            _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

            c10_frag, a0_frag = _compute_block(
                b_cur1, B_rsrc, B1_gl_offset + (k + 2) * BLOCK_K,
                global_offsets, wave_i, a_next0,
                N_TILES_B, N_TILES_A, a1_frag, b0_frag, c10_frag,
            )

            c11_frag, b0_frag = _compute_block(
                a_cur1, A_rsrc, A1_gl_offset + (k + 2) * BLOCK_K,
                global_offsets, wave_j, b_next0,
                N_TILES_A, N_TILES_B, a1_frag, b1_frag, c11_frag,
            )

            a_cur0, a_next0 = a_next0, a_cur0
            a_cur1, a_next1 = a_next1, a_cur1
            b_cur0, b_next0 = b_next0, b_cur0
            b_cur1, b_next1 = b_next1, b_cur1

        # step k = k_iters - 2
        _wait_barrier((2 * N_TILES_A) + (2 * N_TILES_B))

        b1_frag = _load_rt(b_cur1, wave_j, N_TILES_B)
        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        a1_frag = _load_rt(a_cur1, wave_i, N_TILES_A)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)

        _wait_barrier((1 * N_TILES_A) + (1 * N_TILES_B))

        a0_frag = _load_rt(a_next0, wave_i, N_TILES_A)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        b0_frag = _load_rt(b_next0, wave_j, N_TILES_B)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)

        a_cur0, a_next0 = a_next0, a_cur0
        a_cur1, a_next1 = a_next1, a_cur1
        b_cur0, b_next0 = b_next0, b_cur0
        b_cur1, b_next1 = b_next1, b_cur1

        # step k = k_iters - 1
        base_row = tile_i * BLOCK_M + wave_i * (N_TILES_A * 16)
        base_col = tile_j * BLOCK_N + wave_j * (N_TILES_B * 16)

        _wait_barrier(0)

        b1_frag = _load_rt(b_cur1, wave_j, N_TILES_B)
        a1_frag = _load_rt(a_cur1, wave_i, N_TILES_A)

        c00_frag = _mfma_ABt_all(a0_frag, b0_frag, c00_frag)
        c01_frag = _mfma_ABt_all(a0_frag, b1_frag, c01_frag)
        c10_frag = _mfma_ABt_all(a1_frag, b0_frag, c10_frag)
        c11_frag = _mfma_ABt_all(a1_frag, b1_frag, c11_frag)

        # Split-K: wait for all splits to finish zeroing before atomic stores
        if const_expr(IS_SPLIT_K):
            _split_k_barrier()

        _store_C_scaled(c00_frag, base_row + 0, base_col + 0)
        _store_C_scaled(c01_frag, base_row + 0, base_col + LDS_BLOCK_N)
        _store_C_scaled(c10_frag, base_row + LDS_BLOCK_M, base_col + 0)
        _store_C_scaled(c11_frag, base_row + LDS_BLOCK_M, base_col + LDS_BLOCK_N)

    @flyc.jit
    def launch_gemm(
        A: fx.Tensor,
        B_T: fx.Tensor,
        C: fx.Tensor,
        A_scale: fx.Tensor,
        B_scale: fx.Tensor,
        semaphore: fx.Tensor,
        signal: fx.Tensor,
        stream: fx.Stream,
    ):
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
            if const_expr(IS_SPLIT_K):
                bc_lds_alloc.finalized = False
                bc_lds_alloc.finalize()
        grid_x = ((M + BLOCK_M - 1) // BLOCK_M) * (N // BLOCK_N)
        kernel_gemm(
            A, B_T, C, A_scale, B_scale, semaphore, signal,
            value_attrs={"rocdl.waves_per_eu": 1, "rocdl.flat_work_group_size": "256,256"},
        ).launch(grid=(grid_x, NUM_SPLITS, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm
