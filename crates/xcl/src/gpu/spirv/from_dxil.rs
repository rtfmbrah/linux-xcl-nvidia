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
const CAP_SHADER_NON_UNIFORM: u32 = 5301;
const CAP_RUNTIME_DESCRIPTOR_ARRAY: u32 = 5302;
const CAP_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING: u32 = 5308;
const SC_INPUT: u32 = 1;
const SC_WORKGROUP: u32 = 4;
const SC_PRIVATE: u32 = 6;
const SC_FUNCTION: u32 = 7;
const SC_STORAGE_BUFFER: u32 = 12;
const SC_PUSH_CONSTANT: u32 = 9;
const SC_PHYSICAL_STORAGE_BUFFER: u32 = 5349;
const DEC_BLOCK: u32 = 2;
const DEC_ARRAY_STRIDE: u32 = 6;
const DEC_BUILTIN: u32 = 11;
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;
const DEC_NON_UNIFORM: u32 = 5300;
/// vkd3d-proton's virtual descriptor set of the shader-visible heap (`VKD3D_SHADER_GLOBAL_HEAP_VIRTUAL_DESCRIPTOR_SET`)
const VKD3D_HEAP_SET: u32 = 100;
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
    /// a raw buffer handle (`createHandleFromBinding` / `annotateHandle`): the buffer
    /// variable, or a pointer to one element of the heap's buffer array
    Buffer(u32),
    /// a `rawBufferLoad`'s result struct: buffer variable, word index id, component mask
    RawLoad(u32, u32),
    /// a ballot's `{i32 x4}` result: the `uvec4` id
    Ballot(u32),
    /// a two-result op (`dx.types.twoi32`, `i32c`, `splitDouble`): the two values
    Pair(u32, u32),
    /// `createHandleFromHeap(index, sampler, nonuniform)`: the index id; not usable before
    /// `annotateHandle` says its kind
    HeapHandle(bool, u32, bool),
    /// a texture handle: the loaded image id and its entry in `Tx::images`
    Image(u32, u32),
    /// a sampler handle: the loaded sampler id
    Sampler(u32),
    /// a texture op's `ResRet` / `getDimensions` struct: the `vec4` id and its component type
    Vec4(u32, u32),
    /// a value with no use in SPIR-V (dx.types constants, `void` results)
    None,
}

/// A texture handle's image type and where it was loaded from.
#[derive(Clone, Copy)]
struct ImageInfo {
    /// SPIR-V `Dim`: 0 1D, 1 2D, 2 3D, 3 Cube
    dim: u32,
    arrayed: bool,
    /// a UAV (storage image)
    storage: bool,
    /// sampled type: 0 float, 1 int, 2 uint
    kind: u8,
    /// heap index id and whether it is non-uniform
    index: u32,
    nonuniform: bool,
}

#[path = "image.rs"]
mod image;

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
    /// phis whose incoming operands are filled in at the end: word index of their operands, edges
    pending_phis: Vec<(usize, usize, u32)>,
    /// storage class of pointer values whose class is not their DXIL address space's
    ptr_sc: HashMap<Value, u32>,
    /// buffer variable → accesses seen (1 read, 2 write)
    buffer_access: HashMap<u32, u8>,
    /// heap buffer handle (element pointer) → the heap variable it points into, and whether
    /// its index is non-uniform
    buffer_base: HashMap<u32, (u32, bool)>,
    /// heap buffer variable → its element (block) type
    heap_blocks: HashMap<u32, u32>,
    /// texture handles (`Val::Image`)
    images: Vec<ImageInfo>,
    /// the push-constant block of root descriptors (its id, declared at the end)
    push_var: Option<u32>,
    /// root UAVs read (bit k: `u<k>`)
    root_uavs: u32,
    /// blocks read through buffer device addresses
    psb_blocks: std::collections::HashSet<u32>,
    /// node being emitted (for `subst`)
    cur: usize,
    subst: Vec<(Vec<bool>, usize, u32)>,
    synth_ids: HashMap<(usize, u32), u32>,
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
                    2 => SC_PRIVATE,
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
        if let Value::Inst(i) = v {
            let cur = self.cur;
            if let Some(&(_, mrg, e)) = self.subst.iter().find(|(body, _, e)| *e == i && !body.get(cur).copied().unwrap_or(true)) {
                return Ok(self.synth_ids[&(mrg, e)]);
            }
        }
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

    /// Raw UAV `u<lower>, space 0` (a root UAV of XCL's root signature): read through its
    /// buffer device address, which vkd3d-proton keeps in push-constant member `lower` (root
    /// descriptor `lower`) when it does not lower root descriptors to SSBOs. A pointer to a
    /// `uint data[]` block of its own (`NonWritable` / `NonReadable` go on its member).
    fn root_buffer(&mut self, space: u32, lower: u32) -> Result<u32, String> {
        self.root_buffer_at(space, lower, lower, None)
    }

    /// Root UAV `u<lower>` (or, with `index`, the one of `u<lower>..=u<upper>` it names): a
    /// pointer from the buffer address in the push constants.
    fn root_buffer_at(&mut self, space: u32, lower: u32, upper: u32, index: Option<u32>) -> Result<u32, String> {
        if space != 0 || upper >= 16 || lower > upper {
            return err(format!("root UAV u{lower}..u{upper} in space {space}"));
        }
        for k in lower..=upper {
            self.root_uavs |= 1 << k;
        }
        let u = self.uint();
        let uv2 = self.w.vector(u, 2);
        let push = match self.push_var {
            Some(v) => v,
            None => {
                let v = self.w.id();
                self.push_var = Some(v);
                v
            }
        };
        // one block type per root UAV, and one per indexed range (their access decorations differ)
        let key = if index.is_some() { 0x7100_0000 | upper << 8 | lower } else { lower };
        let st = match self.buffers.get(&(0x7000_0000, key)) {
            Some(&st) => st,
            None => {
                let arr = self.w.ty(op::TYPE_RUNTIME_ARRAY, &[u]);
                if !self.w_has_stride(arr) {
                    self.w.decorate(arr, DEC_ARRAY_STRIDE, &[4]);
                }
                let st = self.w.distinct_type(op::TYPE_STRUCT, &[arr]);
                self.w.decorate(st, DEC_BLOCK, &[]);
                self.w.member_decorate(st, 0, DEC_OFFSET, &[0]);
                self.buffers.insert((0x7000_0000, key), st);
                self.psb_blocks.insert(st);
                st
            }
        };
        // the push block is `uvec2[n]` (member 0): a computed index needs an array
        let ppc = self.w.pointer(SC_PUSH_CONSTANT, uv2);
        let zero = self.w.constant_u32(0);
        let k = match index {
            Some(i) => i,
            None => self.w.constant_u32(lower),
        };
        let p = self.w.id();
        self.emit(op::ACCESS_CHAIN, &[ppc, p, push, zero, k]);
        let addr = self.w.id();
        self.emit(op::LOAD, &[uv2, addr, p]);
        let pst = self.w.pointer(SC_PHYSICAL_STORAGE_BUFFER, st);
        let r = self.w.id();
        self.emit(124, &[pst, r, addr]); // OpBitcast
        self.buffer_base.insert(r, (st, false));
        Ok(r)
    }

    /// Memory operands of a load / store through a word pointer of `buf` (`Aligned 4` for
    /// buffer device addresses).
    fn word_access(&self, buf: u32) -> &'static [u32] {
        match self.buffer_base.get(&buf) {
            Some((b, _)) if self.psb_blocks.contains(b) => &[2, 4],
            _ => &[],
        }
    }

    /// The heap's raw buffers (`ResourceDescriptorHeap[i]` as `(RW)ByteAddressBuffer`, SRVs
    /// and UAVs apart): `uint data[]` blocks as a runtime array in
    /// vkd3d's virtual descriptor-heap set (100, binding 0), as dxil-spirv declares them.
    fn heap_buffers(&mut self, uav: bool) -> u32 {
        let key = (u32::MAX, uav as u32);
        if let Some(&v) = self.buffers.get(&key) {
            return v;
        }
        self.w.capability(CAP_RUNTIME_DESCRIPTOR_ARRAY);
        let u = self.uint();
        let arr = self.w.ty(op::TYPE_RUNTIME_ARRAY, &[u]);
        if !self.w_has_stride(arr) {
            self.w.decorate(arr, DEC_ARRAY_STRIDE, &[4]);
        }
        let st = self.w.distinct_type(op::TYPE_STRUCT, &[arr]);
        self.w.decorate(st, DEC_BLOCK, &[]);
        self.w.member_decorate(st, 0, DEC_OFFSET, &[0]);
        let heap = self.w.ty(op::TYPE_RUNTIME_ARRAY, &[st]);
        let p = self.w.pointer(SC_STORAGE_BUFFER, heap);
        let var = self.w.global_variable(p, SC_STORAGE_BUFFER);
        self.w.decorate(var, DEC_DESCRIPTOR_SET, &[VKD3D_HEAP_SET]);
        self.w.decorate(var, DEC_BINDING, &[if uav { 3 } else { 2 }]);
        self.buffers.insert(key, var);
        self.heap_blocks.insert(var, st);
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
    /// `access`: 1 read, 2 write, 3 both (atomics); recorded for `NonWritable` / `NonReadable`.
    fn buffer_word(&mut self, buf: u32, index: u32, access: u8) -> u32 {
        let (var, nonuniform) = self.buffer_base.get(&buf).copied().unwrap_or((buf, false));
        *self.buffer_access.entry(var).or_default() |= access;
        let u = self.uint();
        let sc = if self.psb_blocks.contains(&var) { SC_PHYSICAL_STORAGE_BUFFER } else { SC_STORAGE_BUFFER };
        let p = self.w.pointer(sc, u);
        let zero = self.w.constant_u32(0);
        let r = self.w.id();
        self.emit(op::ACCESS_CHAIN, &[p, r, buf, zero, index]);
        if nonuniform {
            self.w.decorate(r, DEC_NON_UNIFORM, &[]);
        }
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
                if class == 3 {
                    // a static sampler
                    let space = self.m.const_u64(parts[2]).unwrap_or(0) as u32;
                    return Ok(self.sampler(None, (space, c(self, 1)?)));
                }
                if class != 1 {
                    return err(format!("createHandleFromBinding: resource class {class} (only UAVs)"));
                }
                let (lower, space) = (self.m.const_u64(parts[0]).unwrap_or(0) as u32, self.m.const_u64(parts[2]).unwrap_or(0) as u32);
                if self.m.const_u64(a[1]).is_none() {
                    // a computed index into a range of root UAVs (`Shader::raw_uav_indexed`)
                    let upper = self.m.const_u64(parts[1]).unwrap_or(0) as u32;
                    let index = self.id(a[1])?;
                    return Ok(Val::Buffer(self.root_buffer_at(space, lower, upper, Some(index))?));
                }
                let index = c(self, 1)?;
                Val::Buffer(self.root_buffer(space, index.max(lower))?)
            }
            216 => match self.vals.get(&a[0]) {
                // annotateHandle: the handle itself
                Some(&v @ (Val::Buffer(_) | Val::Sampler(_))) => v,
                Some(&Val::HeapHandle(sampler, index, nonuniform)) => {
                    // resource properties: kind (bits 0-7), UAV (bit 12); dword 1: format / stride
                    let props = match a[1] {
                        Value::Const(c) => match self.m.const_def(c).clone() {
                            ConstDef::Aggregate(_, parts) => parts.iter().map(|&p| self.m.const_u64(p).unwrap_or(0) as u32).collect::<Vec<_>>(),
                            _ => vec![],
                        },
                        _ => vec![],
                    };
                    let p0 = props.first().copied().unwrap_or(0);
                    let kind = p0 & 0xff;
                    if sampler && kind == 14 {
                        return Ok(self.sampler(Some((index, nonuniform)), (0, 0)));
                    }
                    if !sampler && kind != 11 && kind != 12 && kind != 13 {
                        return self.heap_texture(kind, p0 & 0x1000 != 0, props.get(1).copied().unwrap_or(0) & 0xff, index, nonuniform);
                    }
                    if !sampler && kind == 11 {
                        let var = self.heap_buffers(p0 & 0x1000 != 0);
                        let st = self.heap_blocks[&var];
                        let pt = self.w.pointer(SC_STORAGE_BUFFER, st);
                        let r = self.w.id();
                        self.emit(op::ACCESS_CHAIN, &[pt, r, var, index]);
                        if nonuniform {
                            self.w.capability(CAP_SHADER_NON_UNIFORM);
                            self.w.capability(CAP_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);
                            self.w.decorate(r, DEC_NON_UNIFORM, &[]);
                        }
                        self.buffer_base.insert(r, (var, nonuniform));
                        return Ok(Val::Buffer(r));
                    }
                    return err(format!(
                        "heap {} kind {} {}",
                        if sampler { "sampler" } else if p0 & 0x1000 != 0 { "UAV" } else { "SRV" },
                        p0 & 0xff,
                        if p0 & 0xff == 11 { "(raw buffer)" } else { "" }
                    ));
                }
                _ => return err("annotateHandle of an unsupported handle"),
            },
            218 => {
                // createHandleFromHeap(index, sampler, nonuniform)
                let sampler = c(self, 1)? != 0;
                let index = self.id(a[0])?;
                let nonuniform = self.m.const_u64(a[2]).is_some_and(|v| v != 0);
                Val::HeapHandle(sampler, index, nonuniform)
            }
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
                let base = self.word_index(off, 0);
                let u = self.uint();
                for k in 0..4 {
                    if mask & (1 << k) == 0 {
                        continue;
                    }
                    let v = self.id(a[3 + k])?;
                    let idx = if k == 0 {
                        base
                    } else {
                        let kk = self.w.constant_u32(k as u32);
                        self.op2(128, u, base, kk)
                    };
                    let p = self.buffer_word(buf, idx, 2);
                    let mut ops = vec![p, v];
                    ops.extend_from_slice(self.word_access(buf));
                    self.emit(op::STORE, &ops);
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
            78 | 79 => {
                // atomicBinOp(handle, op, c0, c1, c2, v) / atomicCompareExchange(handle, c0, c1, c2, cmp, new)
                // on a raw buffer: c0 is the byte offset
                let Some(&Val::Buffer(buf)) = self.vals.get(&a[0]) else { return err(format!("{name} on an unsupported handle")) };
                if !matches!(self.m.type_def(self.m.inst_ty(inst)), TypeDef::Int(32)) {
                    // the buffer is declared as uint[]: 64-bit atomics need a view of their own
                    return err(format!("{name} on a non-32-bit value"));
                }
                let t = self.ty(self.m.inst_ty(inst))?;
                let off_at = if opcode == 78 { 2 } else { 1 };
                let off = self.id(a[off_at])?;
                let idx = self.word_index(off, 0);
                let p = self.buffer_word(buf, idx, 3);
                let sem = SEM_ACQUIRE_RELEASE | SEM_UNIFORM_MEMORY;
                if opcode == 78 {
                    let x = self.id(a[5])?;
                    let opn = match c(self, 1)? {
                        0 => 234, // IAdd
                        1 => 240, // And
                        2 => 241, // Or
                        3 => 242, // Xor
                        4 => 236, // SMin
                        5 => 238, // SMax
                        6 => 237, // UMin
                        7 => 239, // UMax
                        8 => 229, // Exchange
                        o => return err(format!("atomicBinOp {o}")),
                    };
                    Val::Id(self.atomic(t, opn, p, x, 1, sem))
                } else {
                    let (cmp, new) = (self.id(a[4])?, self.id(a[5])?);
                    Val::Id(self.cmpxchg(t, p, new, cmp, 1, sem))
                }
            }
            62 | 63 | 65 | 66 | 67 | 72 | 73 => return self.texture_op(opcode, a),
            71 => Val::Id(self.w.constant_bool(true)), // checkAccessFullyMapped
            _ => return self.math_op(inst, opcode, name, a),
        })
    }

    fn atomic(&mut self, t: u32, opn: u16, p: u32, x: u32, scope: u32, sem: u32) -> u32 {
        if scope == 1 {
            self.w.capability(5346); // VulkanMemoryModelDeviceScope
        }
        let (sc, se) = (self.w.constant_u32(scope), self.w.constant_u32(sem));
        let r = self.w.id();
        self.emit(opn, &[t, r, p, sc, se, x]);
        r
    }
    fn cmpxchg(&mut self, t: u32, p: u32, new: u32, cmp: u32, scope: u32, sem: u32) -> u32 {
        if scope == 1 {
            self.w.capability(5346); // VulkanMemoryModelDeviceScope
        }
        let (sc, se, su) = (self.w.constant_u32(scope), self.w.constant_u32(sem), self.w.constant_u32(sem & !SEM_ACQUIRE_RELEASE | 0x2));
        let r = self.w.id();
        // the unequal case only reads: Acquire instead of AcquireRelease
        self.emit(230, &[t, r, p, sc, se, su, new, cmp]); // OpAtomicCompareExchange
        r
    }

    fn glsl(&mut self) -> u32 {
        self.w.ext_inst_import("GLSL.std.450")
    }
    /// `OpExtInst` of GLSL.std.450 instruction `n`.
    fn ext(&mut self, t: u32, n: u32, args: &[u32]) -> u32 {
        let set = self.glsl();
        let r = self.w.id();
        let mut ops = vec![t, r, set, n];
        ops.extend_from_slice(args);
        self.emit(op::EXT_INST, &ops);
        r
    }
    fn op2(&mut self, opcode: u16, t: u32, x: u32, y: u32) -> u32 {
        let r = self.w.id();
        self.emit(opcode, &[t, r, x, y]);
        r
    }
    fn op1(&mut self, opcode: u16, t: u32, x: u32) -> u32 {
        let r = self.w.id();
        self.emit(opcode, &[t, r, x]);
        r
    }
    fn select(&mut self, t: u32, c: u32, x: u32, y: u32) -> u32 {
        let r = self.w.id();
        self.emit(169, &[t, r, c, x, y]);
        r
    }
    /// A group operation (`OpGroupNonUniform*` with a GroupOperation operand).
    fn group_op(&mut self, opcode: u16, t: u32, group_op: u32, v: u32) -> u32 {
        self.w.capability(CAP_GROUP_NON_UNIFORM);
        self.w.capability(63); // GroupNonUniformArithmetic
        let scope = self.w.constant_u32(SCOPE_SUBGROUP);
        let r = self.w.id();
        self.emit(opcode, &[t, r, scope, group_op, v]);
        r
    }

    /// The math, bit and remaining wave `dx.op`s.
    fn math_op(&mut self, inst: u32, opcode: u32, name: &str, a: &[Value]) -> Result<Val, String> {
        let rty = self.m.inst_ty(inst);
        let argt = a.first().and_then(|&v| self.value_ty(v));
        let c = |tx: &Self, i: usize| tx.m.const_u64(a[i]).map(|v| v as u32).ok_or_else(|| format!("{name}: operand {i} not constant"));
        let u = self.uint();
        let bt = self.w.bool();
        Ok(match opcode {
            // float unary (GLSL.std.450)
            6 | 7 | 12..=29 => {
                let t = self.ty(rty)?;
                let x = self.id(a[0])?;
                if opcode == 7 {
                    // saturate: NaN → 0
                    let ft = argt.ok_or("saturate of a non-value")?;
                    let (z, o) = (self.float_const(ft, 0.0)?, self.float_const(ft, 1.0)?);
                    return Ok(Val::Id(self.ext(t, 81, &[x, z, o])));
                }
                let n = match opcode {
                    6 => 4,   // FAbs
                    12 => 14, // Cos
                    13 => 13, // Sin
                    14 => 15, // Tan
                    15 => 17, // Acos
                    16 => 16, // Asin
                    17 => 18, // Atan
                    18 => 20, // Cosh
                    19 => 19, // Sinh
                    20 => 21, // Tanh
                    21 => 29, // Exp2
                    22 => 10, // Fract
                    23 => 30, // Log2
                    24 => 31, // Sqrt
                    25 => 32, // InverseSqrt
                    26 => 2,  // RoundEven
                    27 => 8,  // Floor
                    28 => 9,  // Ceil
                    _ => 3,   // Trunc
                };
                Val::Id(self.ext(t, n, &[x]))
            }
            // isSpecialFloat
            8..=11 => {
                let x = self.id(a[0])?;
                let ft = argt.ok_or("isSpecialFloat of a non-value")?;
                let nan = self.op1(156, bt, x);
                let inf = self.op1(157, bt, x);
                Val::Id(match opcode {
                    8 => nan,
                    9 => inf,
                    _ => {
                        let either = self.op2(166, bt, nan, inf);
                        let finite = self.op1(168, bt, either);
                        if opcode == 10 {
                            finite
                        } else {
                            let t = self.ty(ft)?;
                            let ax = self.ext(t, 4, &[x]);
                            let min_normal = match self.m.type_def(ft) {
                                TypeDef::Half => self.w.constant(t, &[0x0400]),
                                TypeDef::Double => self.w.constant(t, &[0, 0x0010_0000]),
                                _ => self.w.constant(t, &[0x0080_0000]),
                            };
                            let big = self.op2(190, bt, ax, min_normal); // FOrdGreaterThanEqual
                            self.op2(167, bt, big, finite)
                        }
                    }
                })
            }
            30 => {
                let t = self.ty(rty)?;
                let x = self.id(a[0])?;
                Val::Id(self.op1(204, t, x)) // OpBitReverse
            }
            31..=34 => {
                let x = self.id(a[0])?;
                if !matches!(argt.map(|t| self.m.type_def(t).clone()), Some(TypeDef::Int(32))) {
                    return err(format!("{name} on a non-32-bit value"));
                }
                Val::Id(match opcode {
                    31 => self.op1(205, u, x), // OpBitCount
                    32 => self.ext(u, 73, &[x]), // FindILsb (-1 for 0)
                    _ => {
                        // DXIL counts from the MSB: 31 - FindUMsb / FindSMsb, -1 stays -1
                        let msb = self.ext(u, if opcode == 33 { 75 } else { 74 }, &[x]);
                        let (m1, k31) = (self.w.constant_u32(u32::MAX), self.w.constant_u32(31));
                        let none = self.op2(170, bt, msb, m1);
                        let from_top = self.op2(130, u, k31, msb);
                        self.select(u, none, m1, from_top)
                    }
                })
            }
            35..=40 => {
                let t = self.ty(rty)?;
                let (x, y) = (self.id(a[0])?, self.id(a[1])?);
                let n = match opcode {
                    35 => 80, // NMax
                    36 => 79, // NMin
                    37 => 42, // SMax
                    38 => 39, // SMin
                    39 => 41, // UMax
                    _ => 38,  // UMin
                };
                Val::Id(self.ext(t, n, &[x, y]))
            }
            41 | 42 => {
                let (x, y) = (self.id(a[0])?, self.id(a[1])?);
                let st = self.w.ty(op::TYPE_STRUCT, &[u, u]);
                let r = self.op2(if opcode == 41 { 152 } else { 151 }, st, x, y); // S/UMulExtended {lo, hi}
                let lo = self.extract(u, r, 0);
                let hi = self.extract(u, r, 1);
                Val::Pair(hi, lo)
            }
            43 => {
                let (x, y) = (self.id(a[0])?, self.id(a[1])?);
                let (zero, ones) = (self.w.constant_u32(0), self.w.constant_u32(u32::MAX));
                let by_zero = self.op2(170, bt, y, zero);
                let q = self.op2(134, u, x, y);
                let r = self.op2(137, u, x, y);
                let q = self.select(u, by_zero, ones, q);
                let r = self.select(u, by_zero, ones, r);
                Val::Pair(q, r)
            }
            44 | 45 => {
                let (x, y) = (self.id(a[0])?, self.id(a[1])?);
                let st = self.w.ty(op::TYPE_STRUCT, &[u, u]);
                let r = self.op2(if opcode == 44 { 149 } else { 150 }, st, x, y); // IAddCarry / ISubBorrow
                let v = self.extract(u, r, 0);
                let cy = self.extract(u, r, 1);
                let zero = self.w.constant_u32(0);
                let flag = self.op2(171, bt, cy, zero);
                Val::Pair(v, flag)
            }
            46..=49 => {
                let t = self.ty(rty)?;
                let (x, y, z) = (self.id(a[0])?, self.id(a[1])?, self.id(a[2])?);
                Val::Id(match opcode {
                    47 => self.ext(t, 50, &[x, y, z]), // Fma
                    // FMad fused, as NVIDIA's hardware does it (and dxil-spirv): DXIL allows either
                    46 => self.ext(t, 50, &[x, y, z]),
                    _ => {
                        let p = self.op2(132, t, x, y);
                        self.op2(128, t, p, z)
                    }
                })
            }
            51 | 52 => {
                // Ibfe / Ubfe(width, offset, value) with D3D's rules for width/offset
                let (w0, o0, v) = (self.id(a[0])?, self.id(a[1])?, self.id(a[2])?);
                let (k31, k32, zero) = (self.w.constant_u32(31), self.w.constant_u32(32), self.w.constant_u32(0));
                let w = self.op2(199, u, w0, k31);
                let o = self.op2(199, u, o0, k31);
                let wo = self.op2(128, u, w, o);
                let fits = self.op2(176, bt, wo, k32);
                let l = self.op2(130, u, k32, wo);
                let shl = self.op2(196, u, v, l);
                let rsh = self.op2(130, u, k32, w);
                let shr = if opcode == 51 { 195 } else { 194 };
                let narrow = self.op2(shr, u, shl, rsh);
                let wide = self.op2(shr, u, v, o);
                let x = self.select(u, fits, narrow, wide);
                let none = self.op2(170, bt, w, zero);
                Val::Id(self.select(u, none, zero, x))
            }
            53 => {
                // Bfi(width, offset, value, replaced)
                let (w0, o0, v, d) = (self.id(a[0])?, self.id(a[1])?, self.id(a[2])?, self.id(a[3])?);
                let (k31, one) = (self.w.constant_u32(31), self.w.constant_u32(1));
                let w = self.op2(199, u, w0, k31);
                let o = self.op2(199, u, o0, k31);
                let bit = self.op2(196, u, one, w);
                let low = self.op2(130, u, bit, one);
                let mask = self.op2(196, u, low, o);
                let vs = self.op2(196, u, v, o);
                let ins = self.op2(199, u, vs, mask);
                let notm = self.op1(200, u, mask); // OpNot
                let keep = self.op2(199, u, d, notm);
                Val::Id(self.op2(197, u, ins, keep))
            }
            101 => {
                let t = self.ty(rty)?;
                let (lo, hi) = (self.id(a[0])?, self.id(a[1])?);
                let v2 = self.w.vector(u, 2);
                let r = self.w.id();
                self.emit(80, &[v2, r, lo, hi]); // OpCompositeConstruct
                Val::Id(self.op1(124, t, r))
            }
            102 => {
                let x = self.id(a[0])?;
                let v2 = self.w.vector(u, 2);
                let b = self.op1(124, v2, x);
                let lo = self.extract(u, b, 0);
                let hi = self.extract(u, b, 1);
                Val::Pair(lo, hi)
            }
            130 => {
                // legacyF32ToF16: the half's bits in the low 16 bits (PackHalf2x16 of (x, 0))
                let f = self.id(a[0])?;
                let f32t = self.w.float(32);
                let v2 = self.w.vector(f32t, 2);
                let zero = self.w.constant(f32t, &[0]);
                let r = self.w.id();
                self.emit(80, &[v2, r, f, zero]);
                Val::Id(self.ext(u, 58, &[r]))
            }
            131 => {
                // legacyF16ToF32: the low 16 bits as a half
                let h = self.id(a[0])?;
                let f32t = self.w.float(32);
                let v2 = self.w.vector(f32t, 2);
                let pair = self.ext(v2, 62, &[h]);
                Val::Id(self.extract(f32t, pair, 0))
            }
            110 => {
                self.w.capability(CAP_GROUP_NON_UNIFORM);
                let scope = self.w.constant_u32(SCOPE_SUBGROUP);
                let r = self.w.id();
                self.emit(333, &[bt, r, scope]); // OpGroupNonUniformElect
                Val::Id(r)
            }
            112 => {
                self.w.capability(CAP_GROUP_NON_UNIFORM);
                Val::Id(self.read_builtin(36, None)) // SubgroupSize
            }
            113..=115 => {
                self.w.capability(CAP_GROUP_NON_UNIFORM);
                self.w.capability(62); // GroupNonUniformVote
                let x = self.id(a[0])?;
                let scope = self.w.constant_u32(SCOPE_SUBGROUP);
                let r = self.w.id();
                let opn = match opcode {
                    113 => 335,
                    114 => 334,
                    _ => 336,
                };
                self.emit(opn, &[bt, r, scope, x]);
                Val::Id(r)
            }
            119 | 121 => {
                let t = self.ty(rty)?;
                let x = self.id(a[0])?;
                let (wop, unsigned) = (c(self, 1)?, c(self, 2)? != 0);
                let float = self.is_float(rty);
                let opn = match (wop, float) {
                    (0, false) => 349,
                    (0, true) => 350,
                    (1, false) => 351,
                    (1, true) => 352,
                    (2, true) => 355,
                    (2, false) => if unsigned { 354 } else { 353 },
                    (3, true) => 358,
                    _ => if unsigned { 357 } else { 356 },
                };
                Val::Id(self.group_op(opn, t, if opcode == 119 { 0 } else { 2 }, x))
            }
            120 => {
                let t = self.ty(rty)?;
                let x = self.id(a[0])?;
                let opn = 359 + c(self, 1)? as u16;
                Val::Id(self.group_op(opn, t, 0, x))
            }
            122 | 123 => {
                self.w.capability(CAP_GROUP_NON_UNIFORM);
                self.w.capability(68); // GroupNonUniformQuad
                let t = self.ty(rty)?;
                let x = self.id(a[0])?;
                let k = if opcode == 122 { self.id(a[1])? } else { self.w.constant_u32(c(self, 1)?) };
                let scope = self.w.constant_u32(SCOPE_SUBGROUP);
                let r = self.w.id();
                self.emit(if opcode == 122 { 365 } else { 366 }, &[t, r, scope, x, k]);
                Val::Id(r)
            }
            135 | 136 => {
                self.w.capability(CAP_GROUP_NON_UNIFORM_BALLOT);
                let cond = self.id(a[0])?;
                let v4 = self.w.vector(u, 4);
                let scope = self.w.constant_u32(SCOPE_SUBGROUP);
                let b = self.w.id();
                self.emit(339, &[v4, b, scope, cond]);
                let r = self.w.id();
                self.emit(342, &[u, r, scope, if opcode == 135 { 0 } else { 2 }, b]); // BallotBitCount
                Val::Id(r)
            }
            _ => return err(format!("dx.op {opcode} ({name}) not translated yet")),
        })
    }

    fn extract(&mut self, t: u32, comp: u32, idx: u32) -> u32 {
        let r = self.w.id();
        self.emit(op::COMPOSITE_EXTRACT, &[t, r, comp, idx]);
        r
    }

    fn float_const(&mut self, t: Ty, v: f64) -> Result<u32, String> {
        let ty = self.ty(t)?;
        Ok(match self.m.type_def(t) {
            TypeDef::Half => self.w.constant(ty, &[half_bits(v as f32) as u32]),
            TypeDef::Double => {
                let b = v.to_bits();
                self.w.constant(ty, &[b as u32, (b >> 32) as u32])
            }
            _ => self.w.constant(ty, &[(v as f32).to_bits()]),
        })
    }

    fn inst(&mut self, i: u32) -> Result<(), String> {
        let ty = self.m.inst_ty(i);
        let opx = self.m.inst_op(i).clone();
        let v = Value::Inst(i);
        let out = match opx {
            Op::Bin(bop, a, b, flags) => {
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
                if flags == 0 && self.is_float(ty) {
                    // no fast-math flags: exact, not to be contracted into an FMA
                    self.w.decorate(r, 42, &[]); // NoContraction
                }
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
                    let p = self.buffer_word(buf, index, 1);
                    let r = self.w.id();
                    let mut ops = vec![u, r, p];
                    ops.extend_from_slice(self.word_access(buf));
                    self.emit(op::LOAD, &ops);
                    Val::Id(r)
                }
                Some(Val::Pair(x, y)) => Val::Id(if idx == 0 { x } else { y }),
                Some(Val::Vec4(v, comp)) => {
                    if idx >= 4 {
                        Val::Id(self.w.constant_u32(0)) // the status word of a ResRet
                    } else {
                        let r = self.w.id();
                        self.emit(op::COMPOSITE_EXTRACT, &[comp, r, v, idx]);
                        Val::Id(r)
                    }
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
            Op::Phi(_) => unreachable!("phis are emitted per block"),
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
                // LLVM's first index steps over the pointer itself; DXIL always passes 0 there.
                // The result lives where the base does (a constant table is Private).
                let t = match self.ptr_sc.get(&p).copied() {
                    Some(sc) => {
                        let TypeDef::Ptr(pointee, _) = self.m.type_def(ty).clone() else { return err("gep result is not a pointer") };
                        let pt = self.ty(pointee)?;
                        self.ptr_sc.insert(v, sc);
                        self.w.pointer(sc, pt)
                    }
                    None => self.ty(ty)?,
                };
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
            Op::AtomicRmw(rop, p, x) => {
                let t = self.ty(ty)?;
                let (p, x) = (self.id(p)?, self.id(x)?);
                Val::Id(self.atomic(t, rmw_opcode(rop)?, p, x, SCOPE_WORKGROUP, SEM_ACQUIRE_RELEASE | SEM_WORKGROUP_MEMORY))
            }
            Op::CmpXchg(p, cmp, new) => {
                // LLVM's {old, success}
                let vt = self.value_ty(cmp).ok_or("cmpxchg of a non-value")?;
                let t = self.ty(vt)?;
                let (p, cmp, new) = (self.id(p)?, self.id(cmp)?, self.id(new)?);
                let old = self.cmpxchg(t, p, new, cmp, SCOPE_WORKGROUP, SEM_ACQUIRE_RELEASE | SEM_WORKGROUP_MEMORY);
                let bt = self.w.bool();
                let ok = self.op2(170, bt, old, cmp);
                Val::Pair(old, ok)
            }
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

fn rmw_opcode(op: crate::gpu::dxil::module::RmwOp) -> Result<u16, String> {
    use crate::gpu::dxil::module::RmwOp as R;
    Ok(match op {
        R::Xchg => 229,
        R::Add => 234,
        R::Sub => 235,
        R::Min => 236,
        R::UMin => 237,
        R::Max => 238,
        R::UMax => 239,
        R::And => 240,
        R::Or => 241,
        R::Xor => 242,
        R::Nand => return err("atomic nand"),
    })
}

/// 0.0 and 1.0 (the only half constants needed so far) as IEEE half bits.
fn half_bits(v: f32) -> u16 {
    if v == 1.0 { 0x3c00 } else { 0 }
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

/// An incoming value of a phi after merge blocks were inserted.
#[derive(Clone, Copy, Debug)]
enum PhiIn {
    Val(Value),
    /// the phi a synthetic block (node) holds for the original phi `inst`
    Synth(usize, u32),
    /// a constant (the exit selector of a loop's merged exits)
    Lit(u32),
    /// no value on this edge
    Undef,
}

/// Phi keys from here on are exit selectors (`uint`), not module instructions.
const SELECTOR: u32 = 0xF000_0000;

/// How the CFG maps to SPIR-V's structured control flow. Nodes `0..n` are the module's
/// blocks, nodes from `n` are synthetic merge blocks the structuring inserts.
struct Structure {
    /// nodes in emission order
    order: Vec<usize>,
    /// successors of every node, in terminator order (edges already redirected)
    succ: Vec<Vec<usize>>,
    /// per node: phis as (original phi instruction, incoming (value, predecessor node))
    phis: Vec<Vec<(u32, Vec<(PhiIn, usize)>)>>,
    /// loop header → (merge node, continue target)
    loops: HashMap<usize, (usize, usize)>,
    /// selection header → merge node
    selections: HashMap<usize, usize>,
    /// loop header that also selects → the synthetic node that holds its branch
    split: HashMap<usize, usize>,
    /// synthetic node that switches on an exit selector phi (key) to its successors
    switches: HashMap<usize, u32>,
    /// loop values used after a merged multi-exit loop: (loop body, merge node, instruction);
    /// outside the body the instruction reads as the merge node's phi for it
    subst: Vec<(Vec<bool>, usize, u32)>,
}

/// Finds loops (back edges of a DFS; one latch and one exit each) and selection merges (the
/// immediate post-dominator in the CFG without back edges). Branches that leave a loop to its
/// merge or continue target (break / continue) need no selection merge. Constructs that
/// would share a merge block get synthetic merge blocks of their own (the outermost keeps
/// the original), with the phis of the original split over them.
fn structure(m: &Module) -> Result<Structure, String> {
    let n0 = m.block_count();
    let mut succ: Vec<Vec<usize>> = (0..n0).map(|b| successors(m, Block(b as u32)).iter().map(|x| x.0 as usize).collect()).collect();
    let mut phis: Vec<Vec<(u32, Vec<(PhiIn, usize)>)>> = (0..n0)
        .map(|b| {
            m.block_insts(Block(b as u32))
                .iter()
                .filter_map(|&i| match m.inst_op(i) {
                    Op::Phi(list) => Some((i, list.iter().map(|&(v, p)| (PhiIn::Val(v), p.0 as usize)).collect())),
                    _ => None,
                })
                .collect()
        })
        .collect();
    // a loop header whose branch also selects between blocks of the loop (the compact form's
    // `switch` dispatcher): its branch moves to a node of its own, so the header can carry the
    // loop merge and that node the selection merge
    let mut split = HashMap::new();
    {
        let (back, _) = dfs(&succ);
        let mut headers: Vec<usize> = back.iter().map(|e| e.1).collect();
        headers.sort_unstable();
        headers.dedup();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); succ.len()];
        for (b, ss) in succ.iter().enumerate() {
            for &t in ss {
                preds[t].push(b);
            }
        }
        for h in headers {
            let mut body = vec![false; succ.len()];
            body[h] = true;
            let mut work: Vec<usize> = back.iter().filter(|e| e.1 == h).map(|e| e.0).collect();
            while let Some(x) = work.pop() {
                if !body[x] {
                    body[x] = true;
                    work.extend(preds[x].iter().copied());
                }
            }
            let inner = succ[h].iter().filter(|&&t| t != h && body[t]).count();
            if inner > 1 {
                let hs = succ.len();
                let moved = std::mem::replace(&mut succ[h], vec![hs]);
                succ.push(moved);
                phis.push(Vec::new());
                for &t in &succ[hs] {
                    for (_, list) in phis[t].iter_mut() {
                        for e in list.iter_mut() {
                            if e.1 == h {
                                e.1 = hs;
                            }
                        }
                    }
                }
                split.insert(h, hs);
            }
        }
    }
    // a loop that leaves to several blocks: every exit edge goes to one new merge node instead,
    // which switches on a selector phi to where the edge went (and carries the exits' phis)
    let mut switches: HashMap<usize, u32> = HashMap::new();
    let mut subst: Vec<(Vec<bool>, usize, u32)> = Vec::new();
    loop {
        let (back, _) = dfs(&succ);
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); succ.len()];
        for (b, ss) in succ.iter().enumerate() {
            for &t in ss {
                preds[t].push(b);
            }
        }
        let mut headers: Vec<usize> = back.iter().map(|e| e.1).collect();
        headers.sort_unstable();
        headers.dedup();
        // innermost loops first: the fewest blocks
        let mut bodies: Vec<(usize, Vec<bool>)> = headers
            .iter()
            .map(|&h| {
                let mut body = vec![false; succ.len()];
                body[h] = true;
                let mut work: Vec<usize> = back.iter().filter(|e| e.1 == h).map(|e| e.0).collect();
                while let Some(x) = work.pop() {
                    if !body[x] {
                        body[x] = true;
                        work.extend(preds[x].iter().copied());
                    }
                }
                (h, body)
            })
            .collect();
        bodies.sort_by_key(|(_, b)| b.iter().filter(|&&x| x).count());
        let mut changed = false;
        for (_, body) in bodies {
            let edges: Vec<(usize, usize)> = (0..succ.len()).filter(|&x| body[x]).flat_map(|x| succ[x].iter().map(move |&t| (x, t))).filter(|&(_, t)| !body[t]).collect();
            let mut targets: Vec<usize> = edges.iter().map(|e| e.1).collect();
            targets.sort_unstable();
            targets.dedup();
            if targets.len() < 2 {
                continue;
            }
            let mut from: Vec<usize> = edges.iter().map(|e| e.0).collect();
            from.sort_unstable();
            from.dedup();
            if from.iter().any(|&p| edges.iter().filter(|e| e.0 == p).map(|e| e.1).collect::<std::collections::BTreeSet<_>>().len() > 1) {
                return err("loop exit block that leaves to two places");
            }
            // values of the loop used after it other than through the exits' phis: after the
            // merge they reach their uses through a phi of the merge node (undefined on edges
            // from blocks they do not dominate), and every use outside the loop reads that phi
            let defined: std::collections::HashSet<u32> =
                (0..n0).filter(|&b| body[b]).flat_map(|b| m.block_insts(Block(b as u32)).iter().copied()).collect();
            let mut escaping: Vec<u32> = Vec::new();
            for b in (0..n0).filter(|&b| !body[b]) {
                for &i in m.block_insts(Block(b as u32)) {
                    let uses: Vec<Value> = match m.inst_info(i) {
                        crate::gpu::dxil::module::InstInfo::Call(_, args) => args.to_vec(),
                        crate::gpu::dxil::module::InstInfo::Phi(list) => {
                            list.iter().filter(|(_, from_b)| !body[from_b.0 as usize]).map(|(v, _)| *v).collect()
                        }
                        crate::gpu::dxil::module::InstInfo::Load(p) => vec![p],
                        crate::gpu::dxil::module::InstInfo::Store(p, v) => vec![p, v],
                        crate::gpu::dxil::module::InstInfo::Other(v) => v,
                        crate::gpu::dxil::module::InstInfo::Term(c, _) => c.into_iter().collect(),
                    };
                    for u in uses {
                        if let Value::Inst(d) = u {
                            if defined.contains(&d) && !escaping.contains(&d) && !matches!(m.inst_op(d), Op::Alloca(..)) {
                                escaping.push(d);
                            }
                        }
                    }
                }
            }
            let idom = dominator_tree(&succ);
            let dominates_node = |a: usize, mut x: usize| loop {
                if x == a {
                    return true;
                }
                if x == 0 || idom[x] == usize::MAX || idom[x] == x {
                    return false;
                }
                x = idom[x];
            };
            let mrg = succ.len();
            let key = SELECTOR + switches.len() as u32;
            succ.push(targets.clone());
            phis.push(Vec::new());
            let target_of = |p: usize| edges.iter().find(|e| e.0 == p).map(|e| e.1).unwrap();
            phis[mrg].push((key, from.iter().map(|&p| (PhiIn::Lit(targets.iter().position(|&t| t == target_of(p)).unwrap() as u32), p)).collect()));
            for &t in &targets {
                let mut moved = Vec::new();
                for (orig, list) in phis[t].iter_mut() {
                    let (take, keep): (Vec<_>, Vec<_>) = list.drain(..).partition(|e| from.contains(&e.1));
                    *list = keep;
                    list.push((PhiIn::Synth(mrg, *orig), mrg));
                    moved.push((*orig, take));
                }
                for (orig, take) in moved {
                    let entries = from.iter().map(|&p| (take.iter().find(|e| e.1 == p).map_or(PhiIn::Undef, |e| e.0), p)).collect();
                    phis[mrg].push((orig, entries));
                }
            }
            for &p in &from {
                for t in succ[p].iter_mut() {
                    if targets.contains(t) && !body[*t] {
                        *t = mrg;
                    }
                }
            }
            for &e in &escaping {
                let def = m.def_block(Value::Inst(e)).map_or(usize::MAX, |b| b.0 as usize);
                let entries = from.iter().map(|&p| (if def != usize::MAX && dominates_node(def, p) { PhiIn::Val(Value::Inst(e)) } else { PhiIn::Undef }, p)).collect();
                phis[mrg].push((e, entries));
                subst.push((body.clone(), mrg, e));
            }
            switches.insert(mrg, key);
            changed = true;
            break; // the graph changed: look again
        }
        if !changed {
            break;
        }
    }
    ladder_partial_joins(&mut succ, &mut phis, &mut switches);
    let n = succ.len();
    let (back, post) = dfs(&succ);
    let is_back = |b: usize, s: usize| back.contains(&(b, s));
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for b in 0..n {
        for &s in &succ[b] {
            preds[s].push(b);
        }
    }
    let topo: Vec<usize> = post.iter().rev().copied().collect();
    let mut rank = vec![usize::MAX; n + 1];
    for (i, &b) in topo.iter().enumerate() {
        rank[b] = i;
    }
    rank[n] = n + 1;
    // loops
    let mut loops: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut loop_of: Vec<Option<usize>> = vec![None; n];
    let mut headers: Vec<usize> = back.iter().map(|e| e.1).collect();
    headers.sort_by_key(|&h| rank[h]);
    headers.dedup();
    for &h in &headers {
        let latches: Vec<usize> = back.iter().filter(|e| e.1 == h).map(|e| e.0).collect();
        if latches.len() != 1 {
            return err(format!("loop with {} latches", latches.len()));
        }
        let latch = latches[0];
        let mut body = vec![false; n];
        body[h] = true;
        let mut work = vec![latch];
        while let Some(x) = work.pop() {
            if !body[x] {
                body[x] = true;
                work.extend(preds[x].iter().copied());
            }
        }
        let mut exits: Vec<usize> = (0..n).filter(|&x| body[x]).flat_map(|x| succ[x].iter().copied()).filter(|&s| !body[s]).collect();
        exits.sort_unstable();
        exits.dedup();
        if exits.len() != 1 {
            return err(format!("loop with {} exits", exits.len()));
        }
        loops.insert(h, (exits[0], latch));
        for x in 0..n {
            if body[x] {
                loop_of[x] = Some(h);
            }
        }
    }
    let fsucc = |b: usize| -> Vec<usize> { succ[b].iter().copied().filter(|&s| !is_back(b, s)).collect() };
    // immediate post-dominators over the forward graph (`n` = virtual exit) and dominators
    let meet = |ip: &Vec<usize>, mut x: usize, mut y: usize, up: bool| {
        // `up`: walking towards higher rank (post-dominators); else towards lower (dominators)
        while x != y {
            if up {
                while rank[x] < rank[y] {
                    x = ip[x];
                }
                while rank[y] < rank[x] {
                    y = ip[y];
                }
            } else {
                while rank[x] > rank[y] {
                    x = ip[x];
                }
                while rank[y] > rank[x] {
                    y = ip[y];
                }
            }
        }
        x
    };
    let mut ipdom = vec![usize::MAX; n + 1];
    ipdom[n] = n;
    for &b in topo.iter().rev() {
        let mut new = usize::MAX;
        let mut ss = fsucc(b);
        if ss.is_empty() {
            ss.push(n);
        }
        for s in ss {
            if ipdom[s] != usize::MAX {
                new = if new == usize::MAX { s } else { meet(&ipdom, new, s, true) };
            }
        }
        ipdom[b] = new;
    }
    let mut idom = vec![usize::MAX; n];
    idom[0] = 0;
    for &b in topo.iter().skip(1) {
        let mut new = usize::MAX;
        for &p in &preds[b] {
            if is_back(p, b) || idom[p] == usize::MAX {
                continue;
            }
            new = if new == usize::MAX { p } else { meet(&idom, new, p, false) };
        }
        idom[b] = new;
    }
    let dominates = |a: usize, mut x: usize| loop {
        if x == a {
            return true;
        }
        if x == 0 || x >= n || idom[x] == usize::MAX {
            return false;
        }
        x = idom[x];
    };
    // the merge each construct wants
    let mut wants: Vec<(usize, usize)> = loops.iter().map(|(&h, &(mrg, _))| (h, mrg)).collect();
    let mut sel_headers: Vec<usize> = Vec::new();
    for &b in &topo {
        if succ[b].len() < 2 || loops.contains_key(&b) {
            if let Some(&(mrg, _)) = loops.get(&b) {
                let inner = succ[b].iter().filter(|&&s| s != mrg && s != b).count();
                if inner > 1 {
                    return err("loop header that also selects (not split yet)");
                }
            }
            continue;
        }
        if let Some(h) = loop_of[b] {
            let (mrg, cont) = loops[&h];
            let other = succ[b].iter().filter(|&&s| s != mrg && s != cont && s != h).count();
            if other < succ[b].len() && other <= 1 {
                continue; // break / continue
            }
        }
        sel_headers.push(b);
        wants.push((b, ipdom[b]));
    }
    // nodes, edges and phis that the merges below rewrite
    let mut vsucc: Vec<Vec<usize>> = succ.clone();
    let mut owner: Vec<usize> = (0..n).collect(); // synthetic node → construct header it closes
    let mut order: Vec<usize> = topo.clone();
    let new_node = |vsucc: &mut Vec<Vec<usize>>, owner: &mut Vec<usize>, phis: &mut Vec<Vec<(u32, Vec<(PhiIn, usize)>)>>, own: usize| {
        vsucc.push(Vec::new());
        owner.push(own);
        phis.push(Vec::new());
        vsucc.len() - 1
    };
    let mut merge_of: HashMap<usize, usize> = HashMap::new();
    // selections whose branches never meet again: an unreachable merge block of their own
    for &(h, mrg) in &wants {
        if mrg == n {
            let x = new_node(&mut vsucc, &mut owner, &mut phis, h);
            order.push(x);
            merge_of.insert(h, x);
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(h, mrg) in &wants {
        if mrg < n {
            groups.entry(mrg).or_default().push(h);
        }
    }
    let mut group_list: Vec<(usize, Vec<usize>)> = groups.into_iter().collect();
    group_list.sort_by_key(|g| rank[g.0]);
    for (mrg, mut hs) in group_list {
        // the outermost (dominating every other) keeps the block
        hs.sort_by_key(|&h| rank[h]);
        let keeper = hs[0];
        // siblings (none enclosing the others): every one gets a merge block of its own
        let siblings = !hs.iter().all(|&h| dominates(keeper, h));
        // the others, innermost first, get a block that branches on to `mrg`; so does the
        // keeper when `mrg` is also reached from outside it (a merge must be dominated)
        let is_continue = loops.values().any(|&(_, l)| l == mrg);
        let keeper_too = siblings || !dominates(keeper, mrg) || (is_continue && !loops.contains_key(&keeper));
        if !keeper_too {
            merge_of.insert(keeper, mrg);
        }
        let mut redo: Vec<usize> = hs.iter().skip(1).rev().copied().collect();
        if keeper_too {
            redo.push(keeper);
        }
        for h in redo {
            let x = new_node(&mut vsucc, &mut owner, &mut phis, h);
            let inside = |node: usize| {
                let o = if node < n { node } else { owner[node] };
                dominates(h, o)
            };
            let from: Vec<usize> = (0..vsucc.len()).filter(|&p| p != x && vsucc[p].contains(&mrg) && inside(p)).collect();
            if from.is_empty() {
                return err(format!("construct at {h} has no edge into merge {mrg}"));
            }
            for &p in &from {
                for t in vsucc[p].iter_mut() {
                    if *t == mrg {
                        *t = x;
                    }
                }
            }
            vsucc[x] = vec![mrg];
            // split the phis of `mrg`: the incoming edges that now come through `x`
            let mut moved = Vec::new();
            for (k, (orig, list)) in phis[mrg].iter_mut().enumerate() {
                let (take, keep): (Vec<_>, Vec<_>) = list.drain(..).partition(|e| from.contains(&e.1));
                *list = keep;
                if !take.is_empty() {
                    list.push((PhiIn::Synth(x, *orig), x));
                    moved.push((k, *orig, take));
                }
            }
            for (_, orig, take) in moved {
                phis[x].push((orig, take));
            }
            let at = order.iter().position(|&o| o == mrg).unwrap_or(order.len());
            order.insert(at, x);
            merge_of.insert(h, x);
        }
    }
    let mut loops_out = HashMap::new();
    for (h, (_, cont)) in loops {
        loops_out.insert(h, (merge_of[&h], cont));
    }
    let selections = sel_headers.iter().map(|&h| (h, merge_of[&h])).collect();
    Ok(Structure { order, succ: vsucc, phis, loops: loops_out, selections, split, switches, subst })
}

/// A join `t` of a selection's arms that is not the selection's merge `M`: some path from the
/// header `A = idom(t)` reaches `M` without passing `t`, as in
/// `if (a) { X } else { Y; if (b) goto end; Z } T; end:` (LLVM's tail merging makes these).
/// SPIR-V constructs cannot branch into another arm, so the edges into `t` and the edges into
/// `M` from outside `t`'s region all go to one new node instead, which switches on a selector
/// phi to `t` or `M` (and carries their phis); it becomes the merge of the selection.
/// Only joins whose region `t` is entered through `t` alone and that sit with `A` and `M` in
/// the same loop are rewritten; the others stay for `spirv-val` to refuse.
fn ladder_partial_joins(succ: &mut Vec<Vec<usize>>, phis: &mut Vec<Vec<(u32, Vec<(PhiIn, usize)>)>>, switches: &mut HashMap<usize, u32>) {
    for _ in 0..1024 {
        let n = succ.len();
        let (back, _) = dfs(succ);
        let back: std::collections::HashSet<(usize, usize)> = back.into_iter().collect();
        let fwd: Vec<Vec<usize>> = (0..n).map(|b| succ[b].iter().copied().filter(|&s| !back.contains(&(b, s))).collect()).collect();
        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in 0..n {
            for &s in &fwd[b] {
                if !preds[s].contains(&b) {
                    preds[s].push(b);
                }
            }
        }
        // innermost loop of every node; loop headers, latches and exit targets are left alone
        let mut loop_of = vec![usize::MAX; n];
        let mut size = vec![usize::MAX; n];
        let mut special = vec![false; n];
        let mut all_preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in 0..n {
            for &s in &succ[b] {
                all_preds[s].push(b);
            }
        }
        let mut headers: Vec<usize> = back.iter().map(|e| e.1).collect();
        headers.sort_unstable();
        headers.dedup();
        for &h in &headers {
            let mut body = vec![false; n];
            body[h] = true;
            let mut work: Vec<usize> = back.iter().filter(|e| e.1 == h).map(|e| e.0).collect();
            while let Some(x) = work.pop() {
                if !body[x] {
                    body[x] = true;
                    work.extend(all_preds[x].iter().copied());
                }
            }
            let count = body.iter().filter(|&&x| x).count();
            for x in 0..n {
                if body[x] {
                    if count < size[x] {
                        size[x] = count;
                        loop_of[x] = h;
                    }
                    for &s in &succ[x] {
                        if !body[s] {
                            special[s] = true;
                        }
                    }
                }
            }
            special[h] = true;
        }
        for e in &back {
            special[e.0] = true;
        }
        let idom = dominator_tree(succ);
        let ipdom = post_dominator_tree(&fwd);
        let dominates = |a: usize, mut x: usize| loop {
            if x == a {
                return true;
            }
            if x == 0 || idom[x] == usize::MAX || idom[x] == x {
                return false;
            }
            x = idom[x];
        };
        let mut done = false;
        for t in 0..n {
            if preds[t].len() < 2 || special[t] || idom[t] == usize::MAX {
                continue;
            }
            let a = idom[t];
            let mm = ipdom[a];
            if mm >= n || mm == t || special[a] || loop_of[a] != loop_of[t] || loop_of[mm] != loop_of[t] {
                continue;
            }
            // t's region: everything reached from t before M; entered through t only
            let mut cut = vec![false; n];
            let mut work = vec![t];
            while let Some(x) = work.pop() {
                if x != mm && !cut[x] {
                    cut[x] = true;
                    work.extend(fwd[x].iter().copied());
                }
            }
            if (0..n).any(|c| cut[c] && c != t && preds[c].iter().any(|&p| !cut[p])) {
                continue;
            }
            let from_t = preds[t].clone();
            let from_m: Vec<usize> = preds[mm].iter().copied().filter(|&p| !cut[p] && dominates(a, p)).collect();
            if from_m.is_empty() {
                continue;
            }
            // a block that branches to both (a merged loop exit whose empty block LLVM removed):
            // its edge to M gets a block of its own first, so the selector has one value per edge
            if let Some(&p) = from_t.iter().find(|p| from_m.contains(p)) {
                let e = succ.len();
                succ.push(vec![mm]);
                phis.push(Vec::new());
                for s in succ[p].iter_mut() {
                    if *s == mm {
                        *s = e;
                    }
                }
                let mut moved = Vec::new();
                for (orig, list) in phis[mm].iter_mut() {
                    let (take, keep): (Vec<_>, Vec<_>) = list.drain(..).partition(|x| x.1 == p);
                    *list = keep;
                    list.push((PhiIn::Synth(e, *orig), e));
                    moved.push((*orig, take));
                }
                phis[e] = moved;
                done = true;
                break;
            }
            if phis[t].iter().any(|(o, _)| phis[mm].iter().any(|(o2, _)| o == o2)) {
                continue;
            }
            let j = succ.len();
            let key = SELECTOR + switches.len() as u32;
            succ.push(vec![t, mm]);
            phis.push(Vec::new());
            let from: Vec<(usize, usize)> = from_t.iter().map(|&p| (p, 0)).chain(from_m.iter().map(|&p| (p, 1))).collect();
            phis[j].push((key, from.iter().map(|&(p, k)| (PhiIn::Lit(k as u32), p)).collect()));
            for (k, target) in [t, mm].into_iter().enumerate() {
                let mut moved = Vec::new();
                for (orig, list) in phis[target].iter_mut() {
                    let (take, keep): (Vec<_>, Vec<_>) = list.drain(..).partition(|e| from.iter().any(|f| f.0 == e.1 && f.1 == k));
                    *list = keep;
                    list.push((PhiIn::Synth(j, *orig), j));
                    moved.push((*orig, take));
                }
                for (orig, take) in moved {
                    let entries = from.iter().map(|&(p, _)| (take.iter().find(|e| e.1 == p).map_or(PhiIn::Undef, |e| e.0), p)).collect();
                    phis[j].push((orig, entries));
                }
            }
            for &(p, k) in &from {
                let target = if k == 0 { t } else { mm };
                for s in succ[p].iter_mut() {
                    if *s == target {
                        *s = j;
                    }
                }
            }
            switches.insert(j, key);
            done = true;
            break;
        }
        if !done {
            return;
        }
    }
}

/// Immediate post-dominators over a forward (acyclic) graph; `succ.len()` is the virtual exit
/// after the blocks without successors (`usize::MAX`: no path to an exit).
fn post_dominator_tree(fwd: &[Vec<usize>]) -> Vec<usize> {
    let n = fwd.len();
    // reversed graph, node 0 = the virtual exit, node b + 1 = block b
    let mut r: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for (b, ss) in fwd.iter().enumerate() {
        if ss.is_empty() {
            r[0].push(b + 1);
        }
        for &s in ss {
            r[s + 1].push(b + 1);
        }
    }
    let id = dominator_tree(&r);
    (0..n)
        .map(|b| match id[b + 1] {
            usize::MAX => usize::MAX,
            0 => n,
            x => x - 1,
        })
        .collect()
}

/// Immediate dominators over the graph without back edges (`usize::MAX`: unreachable).
fn dominator_tree(succ: &[Vec<usize>]) -> Vec<usize> {
    let n = succ.len();
    let (back, post) = dfs(succ);
    let topo: Vec<usize> = post.iter().rev().copied().collect();
    let mut rank = vec![usize::MAX; n];
    for (i, &b) in topo.iter().enumerate() {
        rank[b] = i;
    }
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (b, ss) in succ.iter().enumerate() {
        for &t in ss {
            if !back.contains(&(b, t)) {
                preds[t].push(b);
            }
        }
    }
    let mut idom = vec![usize::MAX; n];
    idom[0] = 0;
    for &b in topo.iter().skip(1) {
        let mut new = usize::MAX;
        for &p in &preds[b] {
            if idom[p] == usize::MAX {
                continue;
            }
            new = if new == usize::MAX {
                p
            } else {
                let (mut x, mut y) = (new, p);
                while x != y {
                    while rank[x] > rank[y] {
                        x = idom[x];
                    }
                    while rank[y] > rank[x] {
                        y = idom[y];
                    }
                }
                x
            };
        }
        idom[b] = new;
    }
    idom
}

/// Back edges of a DFS from node 0, and its post-order.
fn dfs(succ: &[Vec<usize>]) -> (Vec<(usize, usize)>, Vec<usize>) {
    let n = succ.len();
    let mut back: Vec<(usize, usize)> = Vec::new();
    let mut state = vec![0u8; n];
    let mut post: Vec<usize> = Vec::new();
    let mut stack: Vec<(usize, usize)> = vec![(0, 0)];
    state[0] = 1;
    while let Some(&mut (b, ref mut k)) = stack.last_mut() {
        if *k < succ[b].len() {
            let s = succ[b][*k];
            *k += 1;
            match state[s] {
                0 => {
                    state[s] = 1;
                    stack.push((s, 0));
                }
                1 => back.push((b, s)),
                _ => {}
            }
        } else {
            state[b] = 2;
            post.push(b);
            stack.pop();
        }
    }
    (back, post)
}

/// The terminator `term` (of a module block) as the end of node `b`, to `b`'s successors.
fn emit_terminator(tx: &mut Tx, st: &Structure, b: usize, term: u32) -> Result<(), String> {
    let m = tx.m;
    let tl: Vec<u32> = st.succ[b].iter().map(|&x| tx.labels[x]).collect();
    match m.inst_op(term).clone() {
        Op::Br(_) => tx.emit(op::BRANCH, &[tl[0]]),
        Op::CondBr(c, _, _) => {
            let c = tx.id(c)?;
            tx.emit(op::BRANCH_CONDITIONAL, &[c, tl[0], tl[1]]);
        }
        Op::Switch(v, _, cases) => {
            let x = tx.id(v)?;
            let mut ops = vec![x, tl[0]];
            for (k, (cv, _)) in cases.iter().enumerate() {
                ops.push(m.const_u64(*cv).ok_or("switch case not constant")? as u32);
                ops.push(tl[k + 1]);
            }
            tx.emit(op::SWITCH, &ops);
        }
        Op::Ret => tx.emit(op::RETURN, &[]),
        Op::Unreachable => tx.emit(op::UNREACHABLE, &[]),
        other => return err(format!("block {b} ends in {other:?}")),
    }
    Ok(())
}

pub fn translate_compute(m: &Module, numthreads: [u32; 3]) -> Result<Vec<u32>, String> {
    let n = m.block_count();
    let st = structure(m)?;

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
        pending_phis: Vec::new(),
        ptr_sc: HashMap::new(),
        buffer_access: HashMap::new(),
        buffer_base: HashMap::new(),
        heap_blocks: HashMap::new(),
        images: Vec::new(),
        push_var: None,
        root_uavs: 0,
        psb_blocks: Default::default(),
        cur: 0,
        subst: Vec::new(),
        synth_ids: HashMap::new(),
    };
    tx.w.capability(CAP_SHADER);
    tx.labels = (0..st.succ.len()).map(|_| tx.w.id()).collect();
    // phis may be used before they are defined: every phi (and synthetic one) gets its id now
    let mut synth_ids: HashMap<(usize, u32), u32> = HashMap::new();
    for (node, list) in st.phis.iter().enumerate() {
        for (orig, _) in list {
            let id = tx.w.id();
            if node < n {
                tx.vals.insert(Value::Inst(*orig), Val::Id(id));
            } else {
                synth_ids.insert((node, *orig), id);
            }
        }
    }
    tx.synth_ids = synth_ids.clone();
    tx.subst = st.subst.clone();
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
                    let t = tx.ty(gd.ty)?;
                    let var = match (gd.addrspace, gd.init) {
                        (3, _) => {
                            let pt = tx.w.pointer(SC_WORKGROUP, t);
                            tx.w.global_variable(pt, SC_WORKGROUP)
                        }
                        // a constant table (`static const` in the program)
                        (2, Some(init)) | (0, Some(init)) => {
                            let pt = tx.w.pointer(SC_PRIVATE, t);
                            let init = tx.id(init)?;
                            tx.w.global_variable_init(pt, SC_PRIVATE, init)
                        }
                        (space, _) => return err(format!("global {} in address space {space}", gd.name)),
                    };
                    tx.interface.push(var);
                    tx.vals.insert(p, Val::Id(var));
                    if gd.addrspace != 3 {
                        tx.ptr_sc.insert(p, SC_PRIVATE);
                    }
                }
            }
        }
    }

    for &b in &st.order {
        tx.cur = b;
        let label = tx.labels[b];
        tx.emit(op::LABEL, &[label]);
        if b >= n {
            // a synthetic merge block: its phis, then on to the next merge
            for (orig, _) in &st.phis[b] {
                let t = if *orig >= SELECTOR { tx.uint() } else { tx.ty(m.inst_ty(*orig))? };
                let r = synth_ids[&(b, *orig)];
                let mut ops = vec![t, r];
                ops.resize(2 + 2 * st.phis[b].iter().find(|p| p.0 == *orig).map_or(0, |p| p.1.len()), 0);
                let at = tx.body.len() + 1;
                tx.emit(op::PHI, &ops);
                tx.pending_phis.push((at, b, *orig));
            }
            if let Some(&key) = st.switches.get(&b) {
                // merged loop exits: on to where the loop was left
                if let Some(&mrg) = st.selections.get(&b) {
                    tx.emit(op::SELECTION_MERGE, &[tx.labels[mrg], 0]);
                }
                let sel = synth_ids[&(b, key)];
                let tl: Vec<u32> = st.succ[b].iter().map(|&x| tx.labels[x]).collect();
                let mut ops = vec![sel, *tl.last().unwrap()];
                for (k, &l) in tl.iter().enumerate().take(tl.len() - 1) {
                    ops.push(k as u32);
                    ops.push(l);
                }
                tx.emit(op::SWITCH, &ops);
                continue;
            }
            if let Some((&h, _)) = st.split.iter().find(|(_, &x)| x == b) {
                // the branch of split loop header `h`
                if let Some(&mrg) = st.selections.get(&b) {
                    tx.emit(op::SELECTION_MERGE, &[tx.labels[mrg], 0]);
                }
                let term = *m.block_insts(Block(h as u32)).last().ok_or("empty loop header")?;
                emit_terminator(&mut tx, &st, b, term)?;
                continue;
            }
            match st.succ[b].first() {
                Some(&t) => tx.emit(op::BRANCH, &[tx.labels[t]]),
                None => tx.emit(op::UNREACHABLE, &[]),
            }
            continue;
        }
        let insts = m.block_insts(Block(b as u32)).to_vec();
        let (term, rest) = insts.split_last().ok_or_else(|| format!("block {b} is empty"))?;
        for &i in rest {
            if let Op::Phi(_) = m.inst_op(i) {
                let t = tx.ty(m.inst_ty(i))?;
                let Some(Val::Id(r)) = tx.vals.get(&Value::Inst(i)).copied() else { unreachable!() };
                let k = st.phis[b].iter().find(|p| p.0 == i).map_or(0, |p| p.1.len());
                let mut ops = vec![t, r];
                ops.resize(2 + 2 * k, 0);
                let at = tx.body.len() + 1;
                tx.emit(op::PHI, &ops);
                tx.pending_phis.push((at, b, i));
                continue;
            }
            tx.inst(i)?;
        }
        if let Some(&(mrg, cont)) = st.loops.get(&b) {
            tx.emit(op::LOOP_MERGE, &[tx.labels[mrg], tx.labels[cont], 0]);
        } else if let Some(&mrg) = st.selections.get(&b) {
            tx.emit(op::SELECTION_MERGE, &[tx.labels[mrg], 0]);
        }
        if let Some(&hs) = st.split.get(&b) {
            tx.emit(op::BRANCH, &[tx.labels[hs]]);
            continue;
        }
        emit_terminator(&mut tx, &st, b, *term)?;
    }

    for (at, node, orig) in std::mem::take(&mut tx.pending_phis) {
        let list = st.phis[node].iter().find(|p| p.0 == orig).map(|p| p.1.clone()).unwrap_or_default();
        for (k, (val, from)) in list.into_iter().enumerate() {
            // the value as it is at the end of the predecessor
            tx.cur = from;
            let id = match val {
                PhiIn::Val(v) => tx.id(v)?,
                PhiIn::Synth(x, o) => synth_ids[&(x, o)],
                PhiIn::Lit(k) => tx.w.constant_u32(k),
                PhiIn::Undef => {
                    let t = if orig >= SELECTOR { tx.uint() } else { tx.ty(m.inst_ty(orig))? };
                    tx.w.undef(t)
                }
            };
            tx.body[at + 2 + 2 * k] = id;
            tx.body[at + 3 + 2 * k] = tx.labels[from];
        }
    }

    // buffers only read / only written: drivers need not order the others' accesses against
    // them (dxil-spirv does the same; NVIDIA's compiler builds such programs much faster)
    let mut acc: Vec<(u32, u8)> = tx.buffer_access.iter().map(|(&b, &a)| (b, a)).collect();
    acc.sort_unstable();
    for (var, a) in acc {
        let d = match a {
            1 => 24, // NonWritable
            2 => 25, // NonReadable
            _ => continue,
        };
        if tx.psb_blocks.contains(&var) {
            tx.w.member_decorate(var, 0, d, &[]);
        } else {
            tx.w.decorate(var, d, &[]);
        }
    }
    // the root descriptors' push constants: one buffer address (uvec2) per root UAV read
    if let Some(push) = tx.push_var {
        let n = 32 - tx.root_uavs.leading_zeros();
        let u = tx.uint();
        let uv2 = tx.w.vector(u, 2);
        let len = tx.w.constant_u32(n);
        let arr = tx.w.distinct_type(op::TYPE_ARRAY, &[uv2, len]);
        tx.w.decorate(arr, DEC_ARRAY_STRIDE, &[8]);
        let st = tx.w.distinct_type(op::TYPE_STRUCT, &[arr]);
        tx.w.decorate(st, DEC_BLOCK, &[]);
        tx.w.member_decorate(st, 0, DEC_OFFSET, &[0]);
        let pt = tx.w.pointer(SC_PUSH_CONSTANT, st);
        tx.w.emit(Section::Globals, op::VARIABLE, &[pt, push, SC_PUSH_CONSTANT]);
        tx.interface.push(push);
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

    // the Vulkan memory model (as dxil-spirv): NVIDIA's compiler builds the same program
    // markedly faster than under GLSL450 (measured on GTA VI compute programs)
    tx.w.capability(5345); // VulkanMemoryModel
    if tx.psb_blocks.is_empty() {
        tx.w.emit(Section::MemoryModel, op::MEMORY_MODEL, &[0, 3]); // Logical Vulkan
    } else {
        tx.w.capability(5347); // PhysicalStorageBufferAddresses
        tx.w.emit(Section::MemoryModel, op::MEMORY_MODEL, &[5348, 3]); // PhysicalStorageBuffer64 Vulkan
    }
    let mut ep = vec![5, main]; // GLCompute
    ep.extend(Writer::string("main"));
    ep.extend(tx.interface.iter().copied());
    tx.w.emit(Section::EntryPoints, op::ENTRY_POINT, &ep);
    tx.w.emit(Section::ExecutionModes, op::EXECUTION_MODE, &[main, 17, numthreads[0], numthreads[1], numthreads[2]]);
    let root_uavs = tx.root_uavs;
    let mut words = tx.w.finish();
    // the generator word carries the root UAVs read (for the `XSPV` header)
    words[2] = 0x5843_0000 | root_uavs;
    Ok(words)
}
