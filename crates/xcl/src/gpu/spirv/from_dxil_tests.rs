use super::from_dxil::translate_compute;
use crate::gpu::dxil::container::Stage;
use crate::gpu::dxil::module::{BinOp, Pred};
use crate::gpu::dxil::shader::Shader;

/// Writes the module to `$XCL_SPIRV_DUMP/<name>.spv` when set (checked with `spirv-val`
/// outside the test, e.g. by `scripts/linux/check-spirv-dumps.sh`).
fn dump(name: &str, words: &[u32]) {
    if let Ok(dir) = std::env::var("XCL_SPIRV_DUMP") {
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(std::path::Path::new(&dir).join(format!("{name}.spv")), words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    }
}

fn finish(mut s: Shader, name: &str) -> Vec<u32> {
    if !s.m.terminated() {
        s.m.ret();
    }
    s.m.optimize();
    let words = translate_compute(&s.m, [64, 1, 1]).unwrap_or_else(|e| panic!("{name}: {e}"));
    dump(name, &words);
    assert_eq!(words[0], super::writer::MAGIC);
    words
}

/// out[tid] = in[tid] * 3 + tid
#[test]
fn straight_line_kernel() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let input = s.raw_uav(0, 0);
    let output = s.raw_uav(1, 0);
    let tid = s.thread_id(0);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    let x = s.raw_load(input, off, 1)[0];
    let three = s.m.i32c(3);
    let y = s.m.bin(BinOp::Mul, x, three);
    let z = s.m.bin(BinOp::Add, y, tid);
    s.raw_store(output, off, &[z]);
    finish(s, "straight_line");
}

/// if (tid < 32) v = in[tid] else v = tid ^ 7; out[tid] = v   (a diamond with a phi)
#[test]
fn diamond_with_phi() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let input = s.raw_uav(0, 0);
    let output = s.raw_uav(1, 0);
    let tid = s.thread_id(0);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    let lim = s.m.i32c(32);
    let c = s.m.cmp(Pred::ULt, tid, lim);
    let (bt, bf, bm) = (s.m.new_block(), s.m.new_block(), s.m.new_block());
    s.m.cond_br(c, bt, bf);
    s.m.set_block(bt);
    let a = s.raw_load(input, off, 1)[0];
    s.m.br(bm);
    s.m.set_block(bf);
    let seven = s.m.i32c(7);
    let b = s.m.bin(BinOp::Xor, tid, seven);
    s.m.br(bm);
    s.m.set_block(bm);
    let i32 = s.m.i32();
    let phi = s.m.phi(i32);
    s.m.add_incoming(phi, a, bt);
    s.m.add_incoming(phi, b, bf);
    s.raw_store(output, off, &[phi]);
    finish(s, "diamond");
}

#[test]
fn wave_ops_and_barrier() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let output = s.raw_uav(0, 0);
    let lane = s.wave_lane_index();
    let one = s.m.i32c(1);
    let other = s.m.bin(BinOp::Xor, lane, one);
    let v = s.wave_read_lane_at(lane, other);
    let first = s.wave_read_lane_first(v);
    let zero = s.m.i32c(0);
    let odd = s.m.bin(BinOp::And, lane, one);
    let c = s.m.cmp(Pred::Ne, odd, zero);
    let ballot = s.wave_ballot(c);
    s.barrier(crate::gpu::dxil::ops::barrier::GROUP_SYNC);
    let tid = s.thread_id(0);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    let sum = s.m.bin(BinOp::Add, first, ballot[0]);
    s.raw_store(output, off, &[sum]);
    finish(s, "wave_ops");
}

/// sum = 0; i = 0; do { sum += i * 3; i += 1 } while (i <= tid); out[tid] = sum   (a do-while
/// loop: header = latch, one exit)
#[test]
fn counting_loop() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let output = s.raw_uav(0, 0);
    let tid = s.thread_id(0);
    let entry = s.m.current_block();
    let head = s.m.new_block();
    let exit = s.m.new_block();
    s.m.br(head);
    s.m.set_block(head);
    let i32 = s.m.i32();
    let i = s.m.phi(i32);
    let sum = s.m.phi(i32);
    let three = s.m.i32c(3);
    let one = s.m.i32c(1);
    let zero = s.m.i32c(0);
    let t = s.m.bin(BinOp::Mul, i, three);
    let sum2 = s.m.bin(BinOp::Add, sum, t);
    let i2 = s.m.bin(BinOp::Add, i, one);
    let more = s.m.cmp(Pred::ULe, i2, tid);
    s.m.cond_br(more, head, exit);
    s.m.add_incoming(i, zero, entry);
    s.m.add_incoming(i, i2, head);
    s.m.add_incoming(sum, zero, entry);
    s.m.add_incoming(sum, sum2, head);
    s.m.set_block(exit);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    s.raw_store(output, off, &[sum2]);
    finish(s, "counting_loop");
}

/// A loop with two exits: i = 0; loop { if (i == tid) { out = i * 2; break A } i++; if (i >= 40) break B }
/// A: out[tid] = v (v = i*2 from the loop), B: out[tid] = 7
#[test]
fn loop_with_two_exits() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let output = s.raw_uav(0, 0);
    let tid = s.thread_id(0);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    let entry = s.m.current_block();
    let head = s.m.new_block();
    let latch = s.m.new_block();
    let (exit_a, exit_b) = (s.m.new_block(), s.m.new_block());
    s.m.br(head);
    s.m.set_block(head);
    let i32 = s.m.i32();
    let i = s.m.phi(i32);
    let two = s.m.i32c(2);
    let dbl = s.m.bin(BinOp::Mul, i, two);
    let hit = s.m.cmp(Pred::Eq, i, tid);
    s.m.cond_br(hit, exit_a, latch);
    s.m.set_block(latch);
    let one = s.m.i32c(1);
    let i2 = s.m.bin(BinOp::Add, i, one);
    let forty = s.m.i32c(40);
    let more = s.m.cmp(Pred::ULt, i2, forty);
    s.m.cond_br(more, head, exit_b);
    let zero = s.m.i32c(0);
    s.m.add_incoming(i, zero, entry);
    s.m.add_incoming(i, i2, latch);
    s.m.set_block(exit_a);
    let v = s.m.phi(i32);
    s.m.add_incoming(v, dbl, head);
    s.raw_store(output, off, &[v]);
    s.m.ret();
    s.m.set_block(exit_b);
    let seven = s.m.i32c(7);
    s.raw_store(output, off, &[seven]);
    s.m.ret();
    finish(s, "loop_two_exits");
}

/// The compact form's shape: a loop whose header dispatches with a `switch`
///   i = 0; acc = 0; loop { switch (i & 3) { 0: acc += 1; 1: acc += 10; default: acc += 100 } i++ } while (i < tid)
#[test]
fn loop_header_with_switch() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let output = s.raw_uav(0, 0);
    let tid = s.thread_id(0);
    let entry = s.m.current_block();
    let head = s.m.new_block();
    let (c0, c1, cd) = (s.m.new_block(), s.m.new_block(), s.m.new_block());
    let latch = s.m.new_block();
    let exit = s.m.new_block();
    s.m.br(head);
    s.m.set_block(head);
    let i32 = s.m.i32();
    let i = s.m.phi(i32);
    let acc = s.m.phi(i32);
    let three = s.m.i32c(3);
    let sel = s.m.bin(BinOp::And, i, three);
    let (k0, k1) = (s.m.i32c(0), s.m.i32c(1));
    s.m.switch(sel, cd, &[(k0, c0), (k1, c1)]);
    let mut adds = Vec::new();
    for (b, v) in [(c0, 1), (c1, 10), (cd, 100)] {
        s.m.set_block(b);
        let k = s.m.i32c(v);
        let r = s.m.bin(BinOp::Add, acc, k);
        adds.push((r, b));
        s.m.br(latch);
    }
    s.m.set_block(latch);
    let acc2 = s.m.phi(i32);
    for (r, b) in adds {
        s.m.add_incoming(acc2, r, b);
    }
    let one = s.m.i32c(1);
    let i2 = s.m.bin(BinOp::Add, i, one);
    let more = s.m.cmp(Pred::ULt, i2, tid);
    s.m.cond_br(more, head, exit);
    let zero = s.m.i32c(0);
    s.m.add_incoming(i, zero, entry);
    s.m.add_incoming(i, i2, latch);
    s.m.add_incoming(acc, zero, entry);
    s.m.add_incoming(acc, acc2, latch);
    s.m.set_block(exit);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    s.raw_store(output, off, &[acc2]);
    finish(s, "loop_switch");
}

/// A join of both arms that one arm can skip (LLVM tail merging):
///   if (tid < 32) { v = in[tid] * 3 } else { if (tid & 1) { r = 9; goto end } v = tid + 100 }
///   r = v ^ 5; end: out[tid] = r
#[test]
fn partial_join() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let input = s.raw_uav(0, 0);
    let output = s.raw_uav(1, 0);
    let tid = s.thread_id(0);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    let (arm_a, arm_b, arm_c, join, end) = (s.m.new_block(), s.m.new_block(), s.m.new_block(), s.m.new_block(), s.m.new_block());
    let lim = s.m.i32c(32);
    let c = s.m.cmp(Pred::ULt, tid, lim);
    s.m.cond_br(c, arm_a, arm_b);
    s.m.set_block(arm_a);
    let x = s.raw_load(input, off, 1)[0];
    let three = s.m.i32c(3);
    let a = s.m.bin(BinOp::Mul, x, three);
    s.m.br(join);
    s.m.set_block(arm_b);
    let one = s.m.i32c(1);
    let zero = s.m.i32c(0);
    let odd = s.m.bin(BinOp::And, tid, one);
    let skip = s.m.cmp(Pred::Ne, odd, zero);
    s.m.cond_br(skip, end, arm_c);
    s.m.set_block(arm_c);
    let hundred = s.m.i32c(100);
    let b = s.m.bin(BinOp::Add, tid, hundred);
    s.m.br(join);
    s.m.set_block(join);
    let i32 = s.m.i32();
    let v = s.m.phi(i32);
    s.m.add_incoming(v, a, arm_a);
    s.m.add_incoming(v, b, arm_c);
    let five = s.m.i32c(5);
    let x5 = s.m.bin(BinOp::Xor, v, five);
    s.m.br(end);
    s.m.set_block(end);
    let r = s.m.phi(i32);
    let nine = s.m.i32c(9);
    s.m.add_incoming(r, x5, join);
    s.m.add_incoming(r, nine, arm_b);
    s.raw_store(output, off, &[r]);
    finish(s, "partial_join");
}

/// Raw buffers from the descriptor heap: out = ResourceDescriptorHeap[tid & 3] (UAV,
/// non-uniform), src = ResourceDescriptorHeap[5] (SRV); out[tid] = src[tid] + 1
#[test]
fn heap_buffers() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let tid = s.thread_id(0);
    let three = s.m.i32c(3);
    let which = s.m.bin(BinOp::And, tid, three);
    let out = s.heap_raw_buffer(which, true, true);
    let five = s.m.i32c(5);
    let src = s.heap_raw_buffer(five, false, false);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    let x = s.raw_load(src, off, 1)[0];
    let one = s.m.i32c(1);
    let y = s.m.bin(BinOp::Add, x, one);
    s.raw_store(out, off, &[y]);
    let words = finish(s, "heap_buffers");
    // both heap arrays sit in vkd3d's virtual heap set 100, binding 0
    let decorations = words.windows(4).filter(|w| w[0] == (4 << 16 | 71) && w[2] == 34 && w[3] == 100).count();
    assert_eq!(decorations, 2);
}
