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

/// A loop in one arm whose exit joins the other arm (GTA VI shape):
///   if (tid < 32) v = tid * 3 else { for (i = 0;; i++) { if (i == tid) { v = i * 2; break } if (i + 1 >= 40) { r = 7; goto end } } }
///   r = v ^ 5; end: out[tid] = r
#[test]
fn loop_exit_into_other_arm() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let output = s.raw_uav(1, 0);
    let tid = s.thread_id(0);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    let entry = s.m.current_block();
    let (head, latch, other, join, end) = (s.m.new_block(), s.m.new_block(), s.m.new_block(), s.m.new_block(), s.m.new_block());
    let lim = s.m.i32c(32);
    let c = s.m.cmp(Pred::ULt, tid, lim);
    let three = s.m.i32c(3);
    let t3 = s.m.bin(BinOp::Mul, tid, three);
    s.m.cond_br(c, join, head);
    s.m.set_block(head);
    let i32 = s.m.i32();
    let i = s.m.phi(i32);
    let two = s.m.i32c(2);
    let dbl = s.m.bin(BinOp::Mul, i, two);
    let hit = s.m.cmp(Pred::Eq, i, tid);
    s.m.cond_br(hit, join, latch);
    s.m.set_block(latch);
    let one = s.m.i32c(1);
    let i2 = s.m.bin(BinOp::Add, i, one);
    let forty = s.m.i32c(40);
    let more = s.m.cmp(Pred::ULt, i2, forty);
    s.m.cond_br(more, head, other);
    let zero = s.m.i32c(0);
    s.m.add_incoming(i, zero, entry);
    s.m.add_incoming(i, i2, latch);
    s.m.set_block(other);
    s.m.br(end);
    s.m.set_block(join);
    let v = s.m.phi(i32);
    s.m.add_incoming(v, t3, entry);
    s.m.add_incoming(v, dbl, head);
    let five = s.m.i32c(5);
    let x = s.m.bin(BinOp::Xor, v, five);
    s.m.br(end);
    s.m.set_block(end);
    let r = s.m.phi(i32);
    let seven = s.m.i32c(7);
    s.m.add_incoming(r, x, join);
    s.m.add_incoming(r, seven, other);
    s.raw_store(output, off, &[r]);
    finish(s, "loop_exit_other_arm");
}

/// Textures and samplers from the heaps and a static sampler:
///   t = Texture2D<float4> heap[3], u = RWTexture2D<float4> heap[4], b = Texture1D<uint4> heap[5],
///   s = SamplerState heap[0], ss = static sampler s0 space1;
///   u[id] = t.Load(id) + t.SampleLevel(s, uv, 0) + t.GatherRed(ss, uv) + float(b.Load(x).x + dims(t).x)
#[test]
fn heap_textures() {
    use crate::gpu::dxil::ops::{CompType, TexKind};
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let x = s.thread_id(0);
    let y = s.thread_id(1);
    let (i3, i4, i5, i0) = (s.m.i32c(3), s.m.i32c(4), s.m.i32c(5), s.m.i32c(0));
    let t = s.heap_texture(i3, TexKind::D2, false, CompType::F32, 4, false);
    let u = s.heap_texture(i4, TexKind::D2, true, CompType::F32, 4, false);
    let b = s.heap_texture(i5, TexKind::D1, false, CompType::U32, 4, true);
    let smp = s.heap_sampler(i0, false, false);
    let ss = s.sampler(0, 1, false);
    let l = s.texture_load(t, Some(i0), &[x, y], &[]);
    let f32 = s.m.f32();
    let fx = s.m.cast(crate::gpu::dxil::module::CastOp::UIToFP, x, f32);
    let fy = s.m.cast(crate::gpu::dxil::module::CastOp::UIToFP, y, f32);
    let zero = s.m.f32c(0);
    let sl = s.sample_level(t, smp, &[fx, fy], &[], zero);
    let g = s.gather(t, ss, &[fx, fy], &[], 0);
    let bl = s.texture_load(b, Some(i0), &[x], &[]);
    let dims = s.get_dimensions(t.handle, Some(i0));
    let w = s.m.extract(dims, 0);
    let sum = s.m.bin(BinOp::Add, bl[0], w);
    let fs = s.m.cast(crate::gpu::dxil::module::CastOp::UIToFP, sum, f32);
    let mut out = [l[0]; 4];
    for k in 0..4 {
        let a = s.m.bin(BinOp::Add, l[k], sl[k]);
        let a = s.m.bin(BinOp::Add, a, g[k]);
        out[k] = s.m.bin(BinOp::Add, a, fs);
    }
    s.texture_store(u, &[x, y], out);
    finish(s, "heap_textures");
}

/// out[tid] = dot3(in[tid*3..], in[tid*3..]) as uint: dx.op.dot3 as multiplies and adds
#[test]
fn dot3() {
    use crate::gpu::dxil::module::{CastOp, FnAttrs};
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let input = s.raw_uav(0, 0);
    let output = s.raw_uav(1, 0);
    let tid = s.thread_id(0);
    let twelve = s.m.i32c(12);
    let off = s.m.bin(BinOp::Mul, tid, twelve);
    let v = s.raw_load(input, off, 7);
    let f32 = s.m.f32();
    let f: Vec<_> = v.iter().map(|&x| s.m.cast(CastOp::UIToFP, x, f32)).collect();
    let d = s.op(55, "dot3", Some(f32), f32, &[f[0], f[1], f[2], f[0], f[1], f[2]], FnAttrs::ReadNone);
    let i32 = s.m.i32();
    let u = s.m.cast(CastOp::FPToUI, d, i32);
    let four = s.m.i32c(4);
    let o = s.m.bin(BinOp::Mul, tid, four);
    s.raw_store(output, o, &[u]);
    let words = finish(s, "dot3");
    let mut fmul = 0;
    let mut i = 5;
    while i < words.len() {
        if words[i] & 0xffff == 133 {
            fmul += 1;
        }
        i += (words[i] >> 16).max(1) as usize;
    }
    assert_eq!(fmul, 3, "three products");
}

/// One block only returns and is reached from the entry, from a nested selection and after
/// both: every branch to it gets a return of its own (valid structure for spirv-val).
#[test]
fn shared_return_block() {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let output = s.raw_uav(1, 0);
    let tid = s.thread_id(0);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    let (a, b, c, r) = (s.m.new_block(), s.m.new_block(), s.m.new_block(), s.m.new_block());
    let k32 = s.m.i32c(32);
    let c1 = s.m.cmp(Pred::ULt, tid, k32);
    s.m.cond_br(c1, a, r);
    s.m.set_block(a);
    let k16 = s.m.i32c(16);
    let c2 = s.m.cmp(Pred::ULt, tid, k16);
    s.m.cond_br(c2, b, c);
    s.m.set_block(b);
    let k8 = s.m.i32c(8);
    let c3 = s.m.cmp(Pred::ULt, tid, k8);
    s.m.cond_br(c3, r, c);
    s.m.set_block(c);
    s.raw_store(output, off, &[tid]);
    s.m.br(r);
    s.m.set_block(r);
    s.m.ret();
    let words = finish(s, "shared_return");
    let returns = {
        let (mut n, mut i) = (0, 5);
        while i < words.len() {
            if words[i] & 0xffff == 253 {
                n += 1;
            }
            i += (words[i] >> 16).max(1) as usize;
        }
        n
    };
    assert!(returns >= 3, "a return per branch to the shared block, got {returns}");
}

/// Control flow reduced from a GTA VI compute program: the loop 3-4-5 in one arm of the
/// selection at 1 has its latch 5 only loop back, and leaves to 6 or into the other arm (2).
/// The latch must count as going on to the loop's exit when merges are chosen; as a dead end
/// it made the selection seem to have no merge (spirv-val: "exits the selection").
#[test]
fn join_after_a_loop_in_one_arm() {
    let succ = [vec![1], vec![2, 3], vec![6], vec![6, 4], vec![5, 2], vec![3], vec![]];
    finish(cfg_module(&succ), "join_after_loop");
}

/// A module with the control flow `succ` (block b branches to `succ[b]`: return, branch,
/// conditional branch on a bit of the thread id, or a switch on it); every block stores.
fn cfg_module(succ: &[Vec<usize>]) -> Shader {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let output = s.raw_uav(0, 0);
    let tid = s.thread_id(0);
    let four = s.m.i32c(4);
    let off = s.m.bin(BinOp::Mul, tid, four);
    let mut blocks = vec![s.m.current_block()];
    for _ in 1..succ.len() {
        blocks.push(s.m.new_block());
    }
    for (b, ss) in succ.iter().enumerate() {
        s.m.set_block(blocks[b]);
        let k = s.m.i32c(b as u32);
        s.raw_store(output, off, &[k]);
        match ss[..] {
            [] => s.m.ret(),
            [t] => s.m.br(blocks[t]),
            [t, f] => {
                let bit = s.m.i32c(1 << (b % 6));
                let x = s.m.bin(BinOp::And, tid, bit);
                let zero = s.m.i32c(0);
                let c = s.m.cmp(Pred::Eq, x, zero);
                s.m.cond_br(c, blocks[t], blocks[f]);
            }
            _ => {
                let cases: Vec<_> = ss[1..].iter().enumerate().map(|(i, &t)| (s.m.i32c(i as u32), blocks[t])).collect();
                s.m.switch(tid, blocks[ss[0]], &cases);
            }
        }
    }
    s
}

/// `XCL_SPIRV_CFG=<file>` (lines of successors, as `XCL_SPIRV_CFG_DUMP` writes them): that
/// control flow translated and dumped as `cfg_file.spv` (for reducing a failing shape).
#[test]
fn cfg_from_file() {
    let Ok(path) = std::env::var("XCL_SPIRV_CFG") else { return };
    let succ: Vec<Vec<usize>> = std::fs::read_to_string(path).unwrap().lines().map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect()).collect();
    let s = cfg_module(&succ);
    let words = translate_compute(&s.m, [64, 1, 1]).unwrap_or_else(|e| panic!("cfg: {e}"));
    dump("cfg_file", &words);
}
