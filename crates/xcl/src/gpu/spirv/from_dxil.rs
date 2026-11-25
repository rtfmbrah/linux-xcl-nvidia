//! The translator's in-memory DXIL module ([`Module`]) → a SPIR-V compute module (Vulkan B1).
//!
//! Scope of this first step: compute entries of the recompiler's test ABI — integer / float
//! ALU, compares, selects, phis, casts, `dx.op` system values, raw UAVs from static bindings
//! (`u<lower>, space<s>` → `StorageBuffer` at set `s`, binding `lower`), group-shared arrays,
//! barriers, wave lane index / read-lane / ballot, and structured selections (`if`, `switch`)
//! whose merge is the branch's immediate post-dominator. Loops, heap handles, images and
//! atomics are refused with a message, so callers fall back to the DXIL path.

use std::collections::HashMap;

use super::writer::{op, Section, Writer};
use crate::gpu::dxil::module::{BinOp, Block, CastOp, ConstDef, Module, Op, Pred, Ty, TypeDef, Value};

// enumerants (SPIR-V specification §3)
const CAP_SHADER: u32 = 1;
const CAP_FLOAT16: u32 = 9;
const CAP_FLOAT64: u32 = 10;
const CAP_INT64: u32 = 11;
const CAP_INT16: u32 = 22;
const CAP_INT8: u32 = 39;
const CAP_GROUP_NON_UNIFORM: u32 = 61;
const CAP_GROUP_NON_UNIFORM_BALLOT: u32 = 64;
const CAP_GROUP_NON_UNIFORM_SHUFFLE: u32 = 65;
const SC_INPUT: u32 = 1;
const SC_WORKGROUP: u32 = 4;
const SC_FUNCTION: u32 = 7;
const SC_STORAGE_BUFFER: u32 = 12;
const DEC_BLOCK: u32 = 2;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_BUILTIN: u32 = 11;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;
const BUILTIN_WORKGROUP_ID: u32 = 26;
const BUILTIN_LOCAL_INVOCATION_ID: u32 = 27;
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
const BUILTIN_LOCAL_INVOCATION_INDEX: u32 = 29;
const BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID: u32 = 41;
const SCOPE_WORKGROUP: u32 = 2;
const SCOPE_SUBGROUP: u32 = 3;
const SEM_ACQUIRE_RELEASE: u32 = 0x8;
const SEM_UNIFORM_MEMORY: u32 = 0x40;
const SEM_WORKGROUP_MEMORY: u32 = 0x100;

/// How a value of the DXIL module exists in the SPIR-V module.
#[derive(Clone, Copy)]
enum Val {
    /// an ordinary SSA id
    Id(u32),
    /// a raw buffer handle (`createHandleFromBinding` / `annotateHandle`): the buffer variable
    Buffer(u32),
    /// a `rawBufferLoad`'s result struct: buffer variable, word index id, component mask
    RawLoad(u32, u32),
    /// a ballot's `{i32 x4}` result: the `uvec4` id
    Ballot(u32),
    /// a value with no use in SPIR-V (dx.types constants, `void` results)
    None,
}

struct Tx<'a> {
    m: &'a Module,
    w: Writer,
    vals: HashMap<Value, Val>,
    types: HashMap<u32, u32>,
    labels: Vec<u32>,
    buffers: HashMap<(u32, u32), u32>,
    builtins: HashMap<u32, u32>,
    interface: Vec<u32>,
    /// Function-scope variables (allocas): must open the first block.
    locals: Vec<(u32, u32)>,
    body: Vec<u32>,
}

fn err<T>(msg: impl Into<String>) -> Result<T, String> {
    Err(msg.into())
}

impl<'a> Tx<'a> {
    fn emit(&mut self, opcode: u16, ops: &[u32]) {
        self.body.push(((ops.len() as u32 + 1) << 16) | opcode as u32);
        self.body.extend_from_slice(ops);
    }

    fn uint(&mut self) -> u32 {
        self.w.int(32, false)
    }

    fn ty(&mut self, t: Ty) -> Result<u32, String> {
        if let Some(&id) = self.types.get(&t.index()) {
            return Ok(id);
        }
        let id = match self.m.type_def(t).clone() {
            TypeDef::Void => self.w.void(),
            TypeDef::Int(1) => self.w.bool(),
            TypeDef::Int(bits) => {
                match bits {
                    8 => self.w.capability(CAP_INT8),
                    16 => self.w.capability(CAP_INT16),
                    64 => self.w.capability(CAP_INT64),
                    _ => {}
                }
                self.w.int(bits, false)
            }
            TypeDef::Half => {
                self.w.capability(CAP_FLOAT16);
                self.w.float(16)
            }
            TypeDef::Float => self.w.float(32),
            TypeDef::Double => {
                self.w.capability(CAP_FLOAT64);
                self.w.float(64)
            }
            TypeDef::Array(n, e) => {
                let e = self.ty(e)?;
                let len = self.w.constant_u32(n as u32);
                self.w.ty(op::TYPE_ARRAY, &[e, len])
            }
            TypeDef::Ptr(p, space) => {
                let p = self.ty(p)?;
                let sc = match space {
                    3 => SC_WORKGROUP,
                    0 => SC_FUNCTION,
                    s => return err(format!("pointer in address space {s}")),
                };
                self.w.pointer(sc, p)
            }
            TypeDef::Struct(None, elems) => {
                let ids = elems.into_iter().map(|e| self.ty(e)).collect::<Result<Vec<_>, _>>()?;
                self.w.ty(op::TYPE_STRUCT, &ids)
            }
            other => return err(format!("type {other:?}")),
        };
        self.types.insert(t.index(), id);
        Ok(id)
    }

    fn is_float(&self, t: Ty) -> bool {
        matches!(self.m.type_def(t), TypeDef::Half | TypeDef::Float | TypeDef::Double)
    }
    fn is_bool(&self, t: Ty) -> bool {
        matches!(self.m.type_def(t), TypeDef::Int(1))
    }
    fn value_ty(&self, v: Value) -> Option<Ty> {
        match v {
            Value::Inst(i) => Some(self.m.inst_ty(i)),
            Value::Const(c) => Some(match self.m.const_def(c) {
                ConstDef::Int(t, _) | ConstDef::Float(t, _) | ConstDef::Undef(t) | ConstDef::Null(t) | ConstDef::Aggregate(t, _) => *t,
            }),
            _ => None,
        }
    }

    /// The SPIR-V id of an operand (constants are created on first use).
    fn id(&mut self, v: Value) -> Result<u32, String> {
        if let Some(Val::Id(id)) = self.vals.get(&v) {
            return Ok(*id);
        }
        match v {
            Value::Const(c) => {
                let def = self.m.const_def(c).clone();
                let id = match def {
                    ConstDef::Int(t, x) => {
                        if self.is_bool(t) {
                            self.w.constant_bool(x & 1 != 0)
                        } else {
                            let ty = self.ty(t)?;
                            let bits = match self.m.type_def(t) { TypeDef::Int(b) => *b, _ => 32 };
                            if bits == 64 { self.w.constant(ty, &[x as u32, (x as u64 >> 32) as u32]) } else { self.w.constant(ty, &[(x as u32) & mask(bits)]) }
                        }
                    }
                    ConstDef::Float(t, b) => {
                        let ty = self.ty(t)?;
                        if matches!(self.m.type_def(t), TypeDef::Double) { self.w.constant(ty, &[b as u32, (b >> 32) as u32]) } else { self.w.constant(ty, &[b as u32]) }
                    }
                    ConstDef::Undef(t) => {
                        let ty = self.ty(t)?;
                        self.w.undef(ty)
                    }
                    ConstDef::Null(t) => {
                        let ty = self.ty(t)?;
                        self.w.constant_null(ty)
                    }
                    ConstDef::Aggregate(t, parts) => {
                        let ty = self.ty(t)?;
                        let ids = parts.iter().map(|&p| self.id(p)).collect::<Result<Vec<_>, _>>()?;
                        self.w.constant_composite(ty, &ids)
                    }
                };
                self.vals.insert(v, Val::Id(id));
                Ok(id)
            }
            _ => err(format!("operand {v:?} has no SPIR-V value")),
        }
    }

    fn builtin(&mut self, which: u32, vector: bool) -> u32 {
        if let Some(&v) = self.builtins.get(&which) {
            return v;
        }
        let u = self.uint();
        let t = if vector { self.w.vector(u, 3) } else { u };
        let p = self.w.pointer(SC_INPUT, t);
        let var = self.w.global_variable(p, SC_INPUT);
        self.w.decorate(var, DEC_BUILTIN, &[which]);
        self.builtins.insert(which, var);
        self.interface.push(var);
        var
    }

    /// Loads a builtin; `comp` picks a component of the vector ones.
    fn read_builtin(&mut self, which: u32, comp: Option<u32>) -> u32 {
        let var = self.builtin(which, comp.is_some());
        let u = self.uint();
        let t = if comp.is_some() { self.w.vector(u, 3) } else { u };
        let r = self.w.id();
        self.emit(op::LOAD, &[t, r, var]);
        match comp {
            Some(c) => {
                let e = self.w.id();
                self.emit(op::COMPOSITE_EXTRACT, &[u, e, r, c]);
                e
            }
            None => r,
        }
    }

    /// The `StorageBuffer` variable of raw UAV `u<lower>, space<space>` (`uint data[]`).
    fn buffer(&mut self, space: u32, lower: u32) -> u32 {
        if let Some(&v) = self.buffers.get(&(space, lower)) {
            return v;
        }
        let u = self.uint();
        let arr = self.w.ty(op::TYPE_RUNTIME_ARRAY, &[u]);
        if !self.w_has_stride(arr) {
            self.w.decorate(arr, DEC_ARRAY_STRIDE, &[4]);
        }
        let st = self.w.distinct_type(op::TYPE_STRUCT, &[arr]);
        self.w.decorate(st, DEC_BLOCK, &[]);
        self.w.member_decorate(st, 0, DEC_OFFSET, &[0]);
        let p = self.w.pointer(SC_STORAGE_BUFFER, st);
        let var = self.w.global_variable(p, SC_STORAGE_BUFFER);
        self.w.decorate(var, DEC_DESCRIPTOR_SET, &[space]);
        self.w.decorate(var, DEC_BINDING, &[lower]);
        self.buffers.insert((space, lower), var);
        self.interface.push(var);
        var
    }

    fn w_has_stride(&mut self, arr: u32) -> bool {
        // the runtime array type is shared: decorate it once
        let key = u32::MAX - arr;
        if self.types.contains_key(&key) {
            return true;
        }
        self.types.insert(key, arr);
        false
    }

    /// Pointer to word `index` of a buffer.
    fn buffer_word(&mut self, buf: u32, index: u32) -> u32 {
        let u = self.uint();
        let p = self.w.pointer(SC_STORAGE_BUFFER, u);
        let zero = self.w.constant_u32(0);
        let r = self.w.id();
        self.emit(op::ACCESS_CHAIN, &[p, r, buf, zero, index]);
        r
    }

    /// Byte offset → word index (`offset >> 2`).
    fn word_index(&mut self, offset: u32, add: u32) -> u32 {
        let u = self.uint();
        let two = self.w.constant_u32(2);
        let r = self.w.id();
        self.emit(194, &[u, r, offset, two]); // OpShiftRightLogical
        if add == 0 {
            return r;
        }
        let a = self.w.constant_u32(add);
        let s = self.w.id();
        self.emit(128, &[u, s, r, a]); // OpIAdd
        s
    }

    fn call(&mut self, inst: u32, name: &str, args: &[Value]) -> Result<Val, String> {
        let opcode = self.m.const_u64(args[0]).ok_or("dx.op without a constant opcode")? as u32;
        let a = &args[1..];
        let c = |tx: &Self, i: usize| tx.m.const_u64(a[i]).map(|v| v as u32).ok_or_else(|| format!("{name}: operand {i} not constant"));
        Ok(match opcode {
            93 => Val::Id(self.read_builtin(BUILTIN_GLOBAL_INVOCATION_ID, Some(c(self, 0)?))),
            94 => Val::Id(self.read_builtin(BUILTIN_WORKGROUP_ID, Some(c(self, 0)?))),
            95 => Val::Id(self.read_builtin(BUILTIN_LOCAL_INVOCATION_ID, Some(c(self, 0)?))),
            96 => Val::Id(self.read_builtin(BUILTIN_LOCAL_INVOCATION_INDEX, None)),
            217 => {
                // createHandleFromBinding(ResBind{lower, upper, space, class}, index, nonuniform)
                let Value::Const(b) = a[0] else { return err("createHandleFromBinding: binding not constant") };
                let ConstDef::Aggregate(_, parts) = self.m.const_def(b).clone() else { return err("createHandleFromBinding: binding not a struct") };
                let class = self.m.const_u64(parts[3]).unwrap_or(0);
                if class != 1 {
                    return err(format!("createHandleFromBinding: resource class {class} (only UAVs)"));
                }
                let (lower, space) = (self.m.const_u64(parts[0]).unwrap_or(0) as u32, self.m.const_u64(parts[2]).unwrap_or(0) as u32);
                let index = c(self, 1)?;
                Val::Buffer(self.buffer(space, index.max(lower)))
            }
            216 => match self.vals.get(&a[0]) {
                // annotateHandle: the handle itself
                Some(&v @ Val::Buffer(_)) => v,
                _ => return err("annotateHandle of an unsupported handle"),
            },
            139 => {
                // rawBufferLoad(handle, offset, undef, mask, align): components read on extract
                let Some(&Val::Buffer(buf)) = self.vals.get(&a[0]) else { return err("rawBufferLoad from an unsupported handle") };
                let off = self.id(a[1])?;
                let idx = self.word_index(off, 0);
                Val::RawLoad(buf, idx)
            }
            140 => {
                // rawBufferStore(handle, offset, undef, v0..v3, mask, align)
                let Some(&Val::Buffer(buf)) = self.vals.get(&a[0]) else { return err("rawBufferStore to an unsupported handle") };
                let mask = c(self, 7)?;
                let off = self.id(a[1])?;
                for k in 0..4 {
                    if mask & (1 << k) == 0 {
                        continue;
                    }
                    let v = self.id(a[3 + k])?;
                    let idx = self.word_index(off, k as u32);
                    let p = self.buffer_word(buf, idx);
                    self.emit(op::STORE, &[p, v]);
                }
                Val::None
            }
            80 => {
                // barrier(mode)
                let mode = c(self, 0)?;
                let mut sem = SEM_ACQUIRE_RELEASE;
                if mode & 8 != 0 {
                    sem |= SEM_WORKGROUP_MEMORY;
                }
                if mode & 6 != 0 {
                    sem |= SEM_UNIFORM_MEMORY;
                }
                let (scope, s) = (self.w.constant_u32(SCOPE_WORKGROUP), self.w.constant_u32(sem));
                if mode & 1 != 0 {
                    self.emit(224, &[scope, scope, s]); // OpControlBarrier
                } else {
                    self.emit(225, &[scope, s]); // OpMemoryBarrier
                }
                Val::None
            }
            111 => {
                self.w.capability(CAP_GROUP_NON_UNIFORM);
                Val::Id(self.read_builtin(BUILTIN_SUBGROUP_LOCAL_INVOCATION_ID, None))
            }
            117 | 118 => {
                // waveReadLaneAt / waveReadLaneFirst
                let t = self.ty(self.m.inst_ty(inst))?;
                let v = self.id(a[0])?;
                let scope = self.w.constant_u32(SCOPE_SUBGROUP);
                let r = self.w.id();
                if opcode == 117 {
                    self.w.capability(CAP_GROUP_NON_UNIFORM_SHUFFLE);
                    let lane = self.id(a[1])?;
                    self.emit(345, &[t, r, scope, v, lane]); // OpGroupNonUniformShuffle
                } else {
                    self.w.capability(CAP_GROUP_NON_UNIFORM_BALLOT);
                    self.emit(338, &[t, r, scope, v]); // OpGroupNonUniformBroadcastFirst
                }
                Val::Id(r)
            }
            116 => {
                self.w.capability(CAP_GROUP_NON_UNIFORM_BALLOT);
                let u = self.uint();
                let v4 = self.w.vector(u, 4);
                let cond = self.id(a[0])?;
                let scope = self.w.constant_u32(SCOPE_SUBGROUP);
                let r = self.w.id();
                self.emit(339, &[v4, r, scope, cond]); // OpGroupNonUniformBallot
                Val::Ballot(r)
            }
            _ => return err(format!("dx.op {opcode} ({name}) not translated yet")),
        })
    }

    fn inst(&mut self, i: u32) -> Result<(), String> {
        let ty = self.m.inst_ty(i);
        let opx = self.m.inst_op(i).clone();
        let v = Value::Inst(i);
        let out = match opx {
            Op::Bin(bop, a, b, _) => {
                let t = self.ty(ty)?;
                let (x, y) = (self.id(a)?, self.id(b)?);
                let opcode = if self.is_bool(ty) {
                    match bop {
                        BinOp::And => 167,
                        BinOp::Or => 166,
                        BinOp::Xor => 165,
                        _ => return err(format!("{bop:?} on i1")),
                    }
                } else if self.is_float(ty) {
                    match bop {
                        BinOp::Add => 129,
                        BinOp::Sub => 131,
                        BinOp::Mul => 133,
                        BinOp::UDiv | BinOp::SDiv => 136,
                        BinOp::URem | BinOp::SRem => 140,
                        _ => return err(format!("float {bop:?}")),
                    }
                } else {
                    match bop {
                        BinOp::Add => 128,
                        BinOp::Sub => 130,
                        BinOp::Mul => 132,
                        BinOp::UDiv => 134,
                        BinOp::SDiv => 135,
                        BinOp::URem => 137,
                        BinOp::SRem => 138,
                        BinOp::Shl => 196,
                        BinOp::LShr => 194,
                        BinOp::AShr => 195,
                        BinOp::And => 199,
                        BinOp::Or => 197,
                        BinOp::Xor => 198,
                    }
                };
                let r = self.w.id();
                self.emit(opcode, &[t, r, x, y]);
                Val::Id(r)
            }
            Op::Cast(cop, a, to) => Val::Id(self.cast(cop, a, to)?),
            Op::Cmp(p, a, b) => {
                let bt = self.w.bool();
                let (x, y) = (self.id(a)?, self.id(b)?);
                let at = self.value_ty(a).ok_or("compare of a non-value")?;
                Val::Id(self.cmp(p, bt, x, y, self.is_bool(at))?)
            }
            Op::Select(c, a, b) => {
                let t = self.ty(ty)?;
                let (c, x, y) = (self.id(c)?, self.id(a)?, self.id(b)?);
                let r = self.w.id();
                self.emit(169, &[t, r, c, x, y]);
                Val::Id(r)
            }
            Op::Extract(agg, idx) => match self.vals.get(&agg).copied() {
                Some(Val::RawLoad(buf, word)) => {
                    let u = self.uint();
                    let index = if idx == 0 {
                        word
                    } else {
                        let k = self.w.constant_u32(idx);
                        let s = self.w.id();
                        self.emit(128, &[u, s, word, k]);
                        s
                    };
                    let p = self.buffer_word(buf, index);
                    let r = self.w.id();
                    self.emit(op::LOAD, &[u, r, p]);
                    Val::Id(r)
                }
                Some(Val::Ballot(b)) => {
                    let u = self.uint();
                    let r = self.w.id();
                    self.emit(op::COMPOSITE_EXTRACT, &[u, r, b, idx]);
                    Val::Id(r)
                }
                _ => {
                    let t = self.ty(ty)?;
                    let a = self.id(agg)?;
                    let r = self.w.id();
                    self.emit(op::COMPOSITE_EXTRACT, &[t, r, a, idx]);
                    Val::Id(r)
                }
            },
            Op::Call(f, args) => {
                let name = self.m.func_decl(f).0.to_string();
                if !name.starts_with("dx.op.") {
                    return err(format!("call to {name}"));
                }
                self.call(i, &name, &args)?
            }
            Op::Phi(list) => {
                let t = self.ty(ty)?;
                let r = match self.vals.get(&v) { Some(Val::Id(r)) => *r, _ => unreachable!("phi ids are made up front") };
                let mut ops = vec![t, r];
                for (val, from) in list {
                    ops.push(self.id(val)?);
                    ops.push(self.labels[from.0 as usize]);
                }
                self.emit(op::PHI, &ops);
                return Ok(());
            }
            Op::Load(p, _) => {
                let t = self.ty(ty)?;
                let p = self.id(p)?;
                let r = self.w.id();
                self.emit(op::LOAD, &[t, r, p]);
                Val::Id(r)
            }
            Op::Store(p, x, _) => {
                let (p, x) = (self.id(p)?, self.id(x)?);
                self.emit(op::STORE, &[p, x]);
                Val::None
            }
            Op::Gep(p, idx) => {
                // LLVM's first index steps over the pointer itself; DXIL always passes 0 there
                let t = self.ty(ty)?;
                let base = self.id(p)?;
                let mut ops = vec![t, 0, base];
                for &k in idx.iter().skip(1) {
                    ops.push(self.id(k)?);
                }
                let r = self.w.id();
                ops[1] = r;
                self.emit(op::ACCESS_CHAIN, &ops);
                Val::Id(r)
            }
            Op::Alloca(at, _, _) => {
                let t = self.ty(at)?;
                let p = self.w.pointer(SC_FUNCTION, t);
                let r = self.w.id();
                self.locals.push((p, r));
                Val::Id(r)
            }
            Op::AtomicRmw(..) | Op::CmpXchg(..) => return err("atomics not translated yet"),
            Op::Br(_) | Op::CondBr(..) | Op::Switch(..) | Op::Ret | Op::Unreachable => unreachable!("terminators are emitted per block"),
        };
        self.vals.insert(v, out);
        Ok(())
    }

    fn cast(&mut self, cop: CastOp, a: Value, to: Ty) -> Result<u32, String> {
        let t = self.ty(to)?;
        let x = self.id(a)?;
        let from = self.value_ty(a).ok_or("cast of a non-value")?;
        let r = self.w.id();
        match cop {
            CastOp::ZExt | CastOp::SExt if self.is_bool(from) => {
                let one = if cop == CastOp::ZExt { 1u64 } else { u64::MAX };
                let bits = match self.m.type_def(to) { TypeDef::Int(b) => *b, _ => 32 };
                let (on, off) = if bits == 64 {
                    (self.w.constant(t, &[one as u32, (one >> 32) as u32]), self.w.constant(t, &[0, 0]))
                } else {
                    (self.w.constant(t, &[(one as u32) & mask(bits)]), self.w.constant(t, &[0]))
                };
                self.emit(169, &[t, r, x, on, off]);
            }
            CastOp::Trunc if self.is_bool(to) => {
                // the low bit
                let ft = self.ty(from)?;
                let bits = match self.m.type_def(from) { TypeDef::Int(b) => *b, _ => 32 };
                let (one, zero) = if bits == 64 { (self.w.constant(ft, &[1, 0]), self.w.constant(ft, &[0, 0])) } else { (self.w.constant(ft, &[1]), self.w.constant(ft, &[0])) };
                let and = self.w.id();
                self.emit(199, &[ft, and, x, one]);
                self.emit(171, &[t, r, and, zero]);
            }
            _ => {
                let opcode = match cop {
                    CastOp::Trunc | CastOp::ZExt => 113,
                    CastOp::SExt => 114,
                    CastOp::FPToUI => 109,
                    CastOp::FPToSI => 110,
                    CastOp::UIToFP => 112,
                    CastOp::SIToFP => 111,
                    CastOp::FPTrunc | CastOp::FPExt => 115,
                    CastOp::BitCast => 124,
                };
                self.emit(opcode, &[t, r, x]);
            }
        }
        Ok(r)
    }

    fn cmp(&mut self, p: Pred, bt: u32, x: u32, y: u32, bools: bool) -> Result<u32, String> {
        let r = self.w.id();
        let opcode = match p {
            Pred::Eq if bools => 164,
            Pred::Ne if bools => 165,
            Pred::Eq => 170,
            Pred::Ne => 171,
            Pred::UGt => 172,
            Pred::SGt => 173,
            Pred::UGe => 174,
            Pred::SGe => 175,
            Pred::ULt => 176,
            Pred::SLt => 177,
            Pred::ULe => 178,
            Pred::SLe => 179,
            Pred::FOEq => 180,
            Pred::FUEq => 181,
            Pred::FONe => 182,
            Pred::FUNe => 183,
            Pred::FOLt => 184,
            Pred::FULt => 185,
            Pred::FOGt => 186,
            Pred::FUGt => 187,
            Pred::FOLe => 188,
            Pred::FULe => 189,
            Pred::FOGe => 190,
            Pred::FUGe => 191,
            Pred::FTrue | Pred::FFalse => return Ok(self.w.constant_bool(p == Pred::FTrue)),
            Pred::FOrd | Pred::FUno => {
                // unordered: either is NaN
                let (nx, ny) = (self.w.id(), self.w.id());
                self.emit(156, &[bt, nx, x]); // OpIsNan
                self.emit(156, &[bt, ny, y]);
                let uno = if p == Pred::FUno { r } else { self.w.id() };
                self.emit(166, &[bt, uno, nx, ny]); // OpLogicalOr
                if p == Pred::FOrd {
                    self.emit(168, &[bt, r, uno]); // OpLogicalNot
                }
                return Ok(r);
            }
        };
        self.emit(opcode, &[bt, r, x, y]);
        Ok(r)
    }
}

fn mask(bits: u32) -> u32 {
    if bits >= 32 { u32::MAX } else { (1 << bits) - 1 }
}

fn successors(m: &Module, b: Block) -> Vec<Block> {
    match m.block_insts(b).last().map(|&i| m.inst_op(i)) {
        Some(Op::Br(t)) => vec![*t],
        Some(Op::CondBr(_, t, f)) => vec![*t, *f],
        Some(Op::Switch(_, d, cases)) => std::iter::once(*d).chain(cases.iter().map(|c| c.1)).collect(),
        _ => Vec::new(),
    }
}

/// Immediate post-dominators (index `n` = the virtual exit), by the iterative
/// Cooper–Harvey–Kennedy algorithm over the reverse CFG.
fn post_dominators(m: &Module) -> Vec<usize> {
    let n = m.block_count();
    let succ: Vec<Vec<usize>> = (0..n).map(|b| {
        let s: Vec<usize> = successors(m, Block(b as u32)).iter().map(|x| x.0 as usize).collect();
        if s.is_empty() { vec![n] } else { s }
    }).collect();
    // reverse-CFG post-order from the exit = blocks in decreasing order of a DFS finish; a
    // simple order that works for the iteration: blocks from last to first, exit first
    let mut order: Vec<usize> = (0..n).rev().collect();
    order.insert(0, n);
    let rank: HashMap<usize, usize> = order.iter().enumerate().map(|(i, &b)| (b, i)).collect();
    let mut ipdom = vec![usize::MAX; n + 1];
    ipdom[n] = n;
    let intersect = |ip: &Vec<usize>, mut a: usize, mut b: usize| {
        while a != b {
            while rank[&a] > rank[&b] {
                a = ip[a];
            }
            while rank[&b] > rank[&a] {
                b = ip[b];
            }
        }
        a
    };
    let mut changed = true;
    while changed {
        changed = false;
        for &b in order.iter().skip(1) {
            let mut new = usize::MAX;
            for &s in &succ[b] {
                if ipdom[s] == usize::MAX {
                    continue;
                }
                new = if new == usize::MAX { s } else { intersect(&ipdom, new, s) };
            }
            if new != usize::MAX && ipdom[b] != new {
                ipdom[b] = new;
                changed = true;
            }
        }
    }
    ipdom
}

/// Translates the module of a finished compute entry (after `Module::optimize`) to SPIR-V words.
pub fn translate_compute(m: &Module, numthreads: [u32; 3]) -> Result<Vec<u32>, String> {
    let n = m.block_count();
    // structure: no back edges (loops) yet; selection merges at immediate post-dominators
    for b in 0..n {
        for s in successors(m, Block(b as u32)) {
            if s.0 as usize <= b {
                return err(format!("loop (back edge {b} -> {}) not translated yet", s.0));
            }
        }
    }
    let ipdom = post_dominators(m);
    let mut merged: HashMap<usize, usize> = HashMap::new();
    for b in 0..n {
        if successors(m, Block(b as u32)).len() > 1 {
            let mrg = ipdom[b];
            if mrg >= n {
                return err(format!("selection at block {b} without a merge block"));
            }
            if let Some(other) = merged.insert(mrg, b) {
                return err(format!("blocks {other} and {b} share merge block {mrg} (not structured yet)"));
            }
        }
    }

    let mut tx = Tx {
        m,
        w: Writer::new(),
        vals: HashMap::new(),
        types: HashMap::new(),
        labels: Vec::new(),
        buffers: HashMap::new(),
        builtins: HashMap::new(),
        interface: Vec::new(),
        locals: Vec::new(),
        body: Vec::new(),
    };
    tx.w.capability(CAP_SHADER);
    tx.labels = (0..n).map(|_| tx.w.id()).collect();
    // phis may be used before they are defined (loops later, and in any order now)
    for b in 0..n {
        for &i in m.block_insts(Block(b as u32)) {
            if matches!(m.inst_op(i), Op::Phi(_)) {
                let id = tx.w.id();
                tx.vals.insert(Value::Inst(i), Val::Id(id));
            }
        }
    }
    // group-shared globals referenced by the code
    for b in 0..n {
        for &i in m.block_insts(Block(b as u32)) {
            let ops: Vec<Value> = match m.inst_op(i) {
                Op::Load(p, _) | Op::Gep(p, _) => vec![*p],
                Op::Store(p, _, _) => vec![*p],
                _ => vec![],
            };
            for p in ops {
                if let Value::Global(g) = p {
                    if tx.vals.contains_key(&p) {
                        continue;
                    }
                    let gd = m.global_def(g).clone();
                    if gd.addrspace != 3 {
                        return err(format!("global {} in address space {}", gd.name, gd.addrspace));
                    }
                    let t = tx.ty(gd.ty)?;
                    let pt = tx.w.pointer(SC_WORKGROUP, t);
                    let var = tx.w.global_variable(pt, SC_WORKGROUP);
                    tx.interface.push(var);
                    tx.vals.insert(p, Val::Id(var));
                }
            }
        }
    }

    for b in 0..n {
        let label = tx.labels[b];
        tx.emit(op::LABEL, &[label]);
        let insts = m.block_insts(Block(b as u32)).to_vec();
        let (term, rest) = insts.split_last().ok_or_else(|| format!("block {b} is empty"))?;
        for &i in rest {
            tx.inst(i)?;
        }
        match m.inst_op(*term).clone() {
            Op::Br(t) => tx.emit(op::BRANCH, &[tx.labels[t.0 as usize]]),
            Op::CondBr(c, t, f) => {
                let c = tx.id(c)?;
                let mrg = tx.labels[ipdom[b]];
                tx.emit(op::SELECTION_MERGE, &[mrg, 0]);
                tx.emit(op::BRANCH_CONDITIONAL, &[c, tx.labels[t.0 as usize], tx.labels[f.0 as usize]]);
            }
            Op::Switch(v, d, cases) => {
                let x = tx.id(v)?;
                let mrg = tx.labels[ipdom[b]];
                tx.emit(op::SELECTION_MERGE, &[mrg, 0]);
                let mut ops = vec![x, tx.labels[d.0 as usize]];
                for (cv, cb) in cases {
                    ops.push(m.const_u64(cv).ok_or("switch case not constant")? as u32);
                    ops.push(tx.labels[cb.0 as usize]);
                }
                tx.emit(op::SWITCH, &ops);
            }
            Op::Ret => tx.emit(op::RETURN, &[]),
            Op::Unreachable => tx.emit(op::UNREACHABLE, &[]),
            other => return err(format!("block {b} ends in {other:?}")),
        }
    }

    // the function: header, the locals at the top of the first block, the body
    let void = tx.w.void();
    let fn_ty = tx.w.function_type(void, &[]);
    let main = tx.w.id();
    tx.w.emit(Section::Functions, op::FUNCTION, &[void, main, 0, fn_ty]);
    let body = std::mem::take(&mut tx.body);
    let first_label_len = 2;
    tx.w.emit(Section::Functions, op::LABEL, &[body[1]]);
    for (p, r) in std::mem::take(&mut tx.locals) {
        tx.w.emit(Section::Functions, op::VARIABLE, &[p, r, SC_FUNCTION]);
    }
    let mut i = first_label_len;
    while i < body.len() {
        let len = (body[i] >> 16) as usize;
        tx.w.emit(Section::Functions, (body[i] & 0xffff) as u16, &body[i + 1..i + len]);
        i += len;
    }
    tx.w.emit(Section::Functions, op::FUNCTION_END, &[]);

    tx.w.emit(Section::MemoryModel, op::MEMORY_MODEL, &[0, 1]); // Logical GLSL450
    let mut ep = vec![5, main]; // GLCompute
    ep.extend(Writer::string("main"));
    ep.extend(tx.interface.iter().copied());
    tx.w.emit(Section::EntryPoints, op::ENTRY_POINT, &ep);
    tx.w.emit(Section::ExecutionModes, op::EXECUTION_MODE, &[main, 17, numthreads[0], numthreads[1], numthreads[2]]);
    Ok(tx.w.finish())
}
