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

/// A loop with two exits is refused with a message (until it is restructured).
#[test]
fn loop_with_two_exits_is_refused() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let head = s.m.new_block();
    let body = s.m.new_block();
    let (exit1, exit2) = (s.m.new_block(), s.m.new_block());
    s.m.br(head);
    s.m.set_block(head);
    let t = s.m.i1c(true);
    s.m.cond_br(t, body, exit1);
    s.m.set_block(body);
    s.m.cond_br(t, head, exit2);
    s.m.set_block(exit1);
    s.m.ret();
    s.m.set_block(exit2);
    s.m.ret();
    let e = translate_compute(&s.m, [1, 1, 1]).unwrap_err();
    assert!(e.contains("exits"), "{e}");
}
