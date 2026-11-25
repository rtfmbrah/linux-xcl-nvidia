//! Textures and samplers of compute programs: heap textures and samplers as runtime arrays in
//! vkd3d-proton's virtual descriptor-heap set (100, binding 0), one array per image type, as
//! dxil-spirv declares them; static samplers at (set = register space, binding = register),
//! which the fork's vkd3d hook remaps like root UAVs.

use super::op;
use super::{err, ImageInfo, Tx, Val, DEC_BINDING, DEC_DESCRIPTOR_SET, DEC_NON_UNIFORM, VKD3D_HEAP_SET};
use crate::gpu::dxil::module::{ConstDef, Value};

const TYPE_IMAGE: u16 = 25;
const TYPE_SAMPLER: u16 = 26;
const TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_COMPOSITE_CONSTRUCT: u16 = 80;
const OP_SAMPLED_IMAGE: u16 = 86;
const OP_IMAGE_SAMPLE_EXPLICIT_LOD: u16 = 88;
const OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD: u16 = 90;
const OP_IMAGE_FETCH: u16 = 95;
const OP_IMAGE_GATHER: u16 = 96;
const OP_IMAGE_READ: u16 = 98;
const OP_IMAGE_WRITE: u16 = 99;
const OP_IMAGE_QUERY_SIZE_LOD: u16 = 103;
const OP_IMAGE_QUERY_SIZE: u16 = 104;
const OP_IMAGE_QUERY_LEVELS: u16 = 106;
const OP_BITCAST: u16 = 124;
const SC_UNIFORM_CONSTANT: u32 = 0;

const CAP_IMAGE_CUBE_ARRAY: u32 = 34;
const CAP_SAMPLED_1D: u32 = 43;
const CAP_IMAGE_1D: u32 = 44;
const CAP_SAMPLED_CUBE_ARRAY: u32 = 45;
const CAP_IMAGE_QUERY: u32 = 50;
const CAP_STORAGE_IMAGE_READ_WITHOUT_FORMAT: u32 = 55;
const CAP_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT: u32 = 56;
const CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING: u32 = 5307;
const CAP_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING: u32 = 5309;

const IMAGE_OPERANDS_LOD: u32 = 0x2;
const IMAGE_OPERANDS_GRAD: u32 = 0x4;

/// Sampled type of an image: 0 float, 1 signed int, 2 unsigned int.
fn sampled_kind(comp: u32) -> Option<u8> {
    match comp {
        9 | 13 | 14 => Some(0), // F32, SNormF32, UNormF32
        4 => Some(1),           // I32
        5 => Some(2),           // U32
        _ => None,
    }
}

impl Tx<'_> {
    fn sampled_type(&mut self, kind: u8) -> u32 {
        match kind {
            0 => self.w.float(32),
            1 => self.w.int(32, true),
            _ => self.w.int(32, false),
        }
    }

    /// The image type of `info` (`depth`: 1 for comparison sampling).
    fn image_type(&mut self, info: ImageInfo, depth: u32) -> u32 {
        let st = self.sampled_type(info.kind);
        self.w.ty(TYPE_IMAGE, &[st, info.dim, depth, info.arrayed as u32, 0, if info.storage { 2 } else { 1 }, 0])
    }

    /// The heap array of images of one type (`UniformConstant`, set 100 binding 0).
    fn heap_images(&mut self, img_ty: u32, storage: bool) -> u32 {
        let key = (u32::MAX - 1, img_ty);
        if let Some(&v) = self.buffers.get(&key) {
            return v;
        }
        self.w.capability(super::CAP_RUNTIME_DESCRIPTOR_ARRAY);
        let arr = self.w.ty(op::TYPE_RUNTIME_ARRAY, &[img_ty]);
        let p = self.w.pointer(SC_UNIFORM_CONSTANT, arr);
        let var = self.w.global_variable(p, SC_UNIFORM_CONSTANT);
        self.w.decorate(var, DEC_DESCRIPTOR_SET, &[VKD3D_HEAP_SET]);
        self.w.decorate(var, DEC_BINDING, &[storage as u32]); // placeholder: 0 sampled, 1 storage
        self.buffers.insert(key, var);
        self.interface.push(var);
        var
    }

    /// Loads image `index` of the heap as type `img_ty`.
    fn load_heap_image(&mut self, img_ty: u32, index: u32, nonuniform: bool, storage: bool) -> u32 {
        let var = self.heap_images(img_ty, storage);
        let pt = self.w.pointer(SC_UNIFORM_CONSTANT, img_ty);
        let p = self.w.id();
        self.emit(op::ACCESS_CHAIN, &[pt, p, var, index]);
        let r = self.w.id();
        self.emit(op::LOAD, &[img_ty, r, p]);
        if nonuniform {
            self.w.capability(super::CAP_SHADER_NON_UNIFORM);
            self.w.capability(if storage { CAP_STORAGE_IMAGE_ARRAY_NON_UNIFORM_INDEXING } else { CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING });
            self.w.decorate(p, DEC_NON_UNIFORM, &[]);
            self.w.decorate(r, DEC_NON_UNIFORM, &[]);
        }
        r
    }

    /// `annotateHandle` of a heap texture: kind (DXIL `ResourceKind`), UAV, component type.
    pub(super) fn heap_texture(&mut self, kind: u32, uav: bool, comp: u32, index: u32, nonuniform: bool) -> Result<Val, String> {
        let (dim, arrayed) = match kind {
            1 => (0, false), // Texture1D
            2 => (1, false), // Texture2D
            4 => (2, false), // Texture3D
            5 => (3, false), // TextureCube
            6 => (0, true),  // Texture1DArray
            7 => (1, true),  // Texture2DArray
            9 => (3, true),  // TextureCubeArray
            k => return err(format!("heap texture kind {k}")),
        };
        let Some(sk) = sampled_kind(comp) else { return err(format!("heap texture component type {comp}")) };
        let info = ImageInfo { dim, arrayed, storage: uav, kind: sk, index, nonuniform };
        match (dim, arrayed, uav) {
            (0, _, false) => self.w.capability(CAP_SAMPLED_1D),
            (0, _, true) => self.w.capability(CAP_IMAGE_1D),
            (3, true, false) => self.w.capability(CAP_SAMPLED_CUBE_ARRAY),
            (3, true, true) => self.w.capability(CAP_IMAGE_CUBE_ARRAY),
            _ => {}
        }
        if uav {
            self.w.capability(CAP_STORAGE_IMAGE_READ_WITHOUT_FORMAT);
            self.w.capability(CAP_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT);
        }
        let ty = self.image_type(info, 0);
        let id = self.load_heap_image(ty, index, nonuniform, uav);
        self.images.push(info);
        Ok(Val::Image(id, self.images.len() as u32 - 1))
    }

    /// A sampler: from the sampler heap (`index`), or a static one (`space`, `register`).
    pub(super) fn sampler(&mut self, heap: Option<(u32, bool)>, binding: (u32, u32)) -> Val {
        let st = self.w.ty(TYPE_SAMPLER, &[]);
        let pt = self.w.pointer(SC_UNIFORM_CONSTANT, st);
        let r = self.w.id();
        match heap {
            Some((index, nonuniform)) => {
                let key = (u32::MAX - 2, 0);
                let var = match self.buffers.get(&key) {
                    Some(&v) => v,
                    None => {
                        self.w.capability(super::CAP_RUNTIME_DESCRIPTOR_ARRAY);
                        let arr = self.w.ty(op::TYPE_RUNTIME_ARRAY, &[st]);
                        let ap = self.w.pointer(SC_UNIFORM_CONSTANT, arr);
                        let v = self.w.global_variable(ap, SC_UNIFORM_CONSTANT);
                        self.w.decorate(v, DEC_DESCRIPTOR_SET, &[VKD3D_HEAP_SET]);
                        self.w.decorate(v, DEC_BINDING, &[4]); // placeholder: the sampler heap
                        self.buffers.insert(key, v);
                        self.interface.push(v);
                        v
                    }
                };
                let p = self.w.id();
                self.emit(op::ACCESS_CHAIN, &[pt, p, var, index]);
                self.emit(op::LOAD, &[st, r, p]);
                if nonuniform {
                    self.w.capability(super::CAP_SHADER_NON_UNIFORM);
                    self.w.capability(CAP_SAMPLED_IMAGE_ARRAY_NON_UNIFORM_INDEXING);
                    self.w.decorate(p, DEC_NON_UNIFORM, &[]);
                    self.w.decorate(r, DEC_NON_UNIFORM, &[]);
                }
            }
            None => {
                let key = (binding.0 | 0x4000_0000, binding.1);
                let var = match self.buffers.get(&key) {
                    Some(&v) => v,
                    None => {
                        let v = self.w.global_variable(pt, SC_UNIFORM_CONSTANT);
                        self.w.decorate(v, DEC_DESCRIPTOR_SET, &[binding.0]);
                        self.w.decorate(v, DEC_BINDING, &[binding.1]);
                        self.buffers.insert(key, v);
                        self.interface.push(v);
                        v
                    }
                };
                self.emit(op::LOAD, &[st, r, var]);
            }
        }
        Val::Sampler(r)
    }

    fn image_of(&self, v: Value) -> Result<(u32, ImageInfo), String> {
        match self.vals.get(&v) {
            Some(&Val::Image(id, k)) => Ok((id, self.images[k as usize])),
            _ => err("texture op on an unsupported handle"),
        }
    }

    fn is_undef(&self, v: Value) -> bool {
        matches!(v, Value::Const(c) if matches!(self.m.const_def(c), ConstDef::Undef(_)))
    }

    /// No texel offsets (undef or 0): offsets are not supported yet.
    fn no_offsets(&self, offs: &[Value]) -> Result<(), String> {
        for &o in offs {
            if !self.is_undef(o) && self.m.const_u64(o) != Some(0) {
                return err("texture op with texel offsets");
            }
        }
        Ok(())
    }

    /// A scalar or vector of the first `n` operands (`float`: f32 components, else u32).
    fn vector_of(&mut self, vals: &[Value], float: bool) -> Result<u32, String> {
        let ids = vals.iter().map(|&v| self.id(v)).collect::<Result<Vec<_>, _>>()?;
        if ids.len() == 1 {
            return Ok(ids[0]);
        }
        let c = if float { self.w.float(32) } else { self.uint() };
        let vt = self.w.vector(c, ids.len() as u32);
        let r = self.w.id();
        let mut ops = vec![vt, r];
        ops.extend(ids);
        self.emit(OP_COMPOSITE_CONSTRUCT, &ops);
        Ok(r)
    }

    /// Coordinate count of an image: its dimensions plus the array slice.
    fn coord_count(info: ImageInfo) -> usize {
        let d = match info.dim {
            0 => 1,
            1 => 2,
            _ => 3,
        };
        d + info.arrayed as usize
    }

    /// A `vec4` image result as DXIL sees it (`ResRet`): signed ints as u32.
    fn texel_result(&mut self, info: ImageInfo, opcode: u16, operands: &[u32]) -> Val {
        let st = self.sampled_type(info.kind);
        let v4 = self.w.vector(st, 4);
        let r = self.w.id();
        let mut ops = vec![v4, r];
        ops.extend_from_slice(operands);
        self.emit(opcode, &ops);
        let comp = if info.kind == 0 { self.w.float(32) } else { self.uint() };
        if info.kind == 1 {
            let u = self.uint();
            let uv4 = self.w.vector(u, 4);
            let b = self.w.id();
            self.emit(OP_BITCAST, &[uv4, b, r]);
            return Val::Vec4(b, comp);
        }
        Val::Vec4(r, comp)
    }

    fn sampled_image(&mut self, img: u32, info: ImageInfo, sampler: Value, depth: u32) -> Result<u32, String> {
        let Some(&Val::Sampler(s)) = self.vals.get(&sampler) else { return err("sample with an unsupported sampler") };
        let (img, ty) = if depth == 0 {
            (img, self.image_type(info, 0))
        } else {
            let ty = self.image_type(info, depth);
            (self.load_heap_image(ty, info.index, info.nonuniform, false), ty)
        };
        let sit = self.w.ty(TYPE_SAMPLED_IMAGE, &[ty]);
        let r = self.w.id();
        self.emit(OP_SAMPLED_IMAGE, &[sit, r, img, s]);
        if info.nonuniform {
            self.w.decorate(r, DEC_NON_UNIFORM, &[]);
        }
        Ok(r)
    }

    /// The texture ops (`a`: the operands after the opcode).
    pub(super) fn texture_op(&mut self, opcode: u32, a: &[Value]) -> Result<Val, String> {
        Ok(match opcode {
            66 => {
                // textureLoad(handle, mip, c0, c1, c2, o0, o1, o2)
                let (img, info) = self.image_of(a[0])?;
                self.no_offsets(&a[5..8])?;
                let n = Self::coord_count(info);
                let c = self.vector_of(&a[2..2 + n], false)?;
                if info.storage {
                    self.texel_result(info, OP_IMAGE_READ, &[img, c])
                } else if self.is_undef(a[1]) {
                    self.texel_result(info, OP_IMAGE_FETCH, &[img, c])
                } else {
                    let lod = self.id(a[1])?;
                    self.texel_result(info, OP_IMAGE_FETCH, &[img, c, IMAGE_OPERANDS_LOD, lod])
                }
            }
            67 => {
                // textureStore(handle, c0, c1, c2, v0, v1, v2, v3, mask)
                let (img, info) = self.image_of(a[0])?;
                if !info.storage {
                    return err("textureStore to an SRV");
                }
                let n = Self::coord_count(info);
                let c = self.vector_of(&a[1..1 + n], false)?;
                let vals: Vec<Value> = a[4..8].iter().map(|&v| if self.is_undef(v) { a[4] } else { v }).collect();
                let mut t = self.vector_of(&vals, info.kind == 0)?;
                if info.kind == 1 {
                    let i = self.w.int(32, true);
                    let iv4 = self.w.vector(i, 4);
                    let b = self.w.id();
                    self.emit(OP_BITCAST, &[iv4, b, t]);
                    t = b;
                }
                self.emit(OP_IMAGE_WRITE, &[img, c, t]);
                Val::None
            }
            62 | 63 => {
                // sampleLevel(handle, sampler, c0..c3, o0..o2, lod)
                // sampleGrad(handle, sampler, c0..c3, o0..o2, ddx0..2, ddy0..2, clamp)
                let (img, info) = self.image_of(a[0])?;
                self.no_offsets(&a[6..9])?;
                let n = Self::coord_count(info);
                let c = self.vector_of(&a[2..2 + n], true)?;
                let si = self.sampled_image(img, info, a[1], 0)?;
                if opcode == 62 {
                    let lod = self.id(a[9])?;
                    self.texel_result(info, OP_IMAGE_SAMPLE_EXPLICIT_LOD, &[si, c, IMAGE_OPERANDS_LOD, lod])
                } else {
                    if !self.is_undef(a[15]) {
                        return err("sampleGrad with a LOD clamp");
                    }
                    let g = n - info.arrayed as usize;
                    let dx = self.vector_of(&a[9..9 + g], true)?;
                    let dy = self.vector_of(&a[12..12 + g], true)?;
                    self.texel_result(info, OP_IMAGE_SAMPLE_EXPLICIT_LOD, &[si, c, IMAGE_OPERANDS_GRAD, dx, dy])
                }
            }
            65 => {
                // sampleCmpLevelZero(handle, sampler, c0..c3, o0..o2, cmp)
                let (img, info) = self.image_of(a[0])?;
                if info.kind != 0 || info.storage {
                    return err("sampleCmpLevelZero on a non-float texture");
                }
                self.no_offsets(&a[6..9])?;
                let n = Self::coord_count(info);
                let c = self.vector_of(&a[2..2 + n], true)?;
                let si = self.sampled_image(img, info, a[1], 1)?;
                let dref = self.id(a[9])?;
                let f = self.w.float(32);
                let zero = self.w.constant(f, &[0]);
                let r = self.w.id();
                self.emit(OP_IMAGE_SAMPLE_DREF_EXPLICIT_LOD, &[f, r, si, c, dref, IMAGE_OPERANDS_LOD, zero]);
                let v4 = self.w.vector(f, 4);
                let v = self.w.id();
                self.emit(OP_COMPOSITE_CONSTRUCT, &[v4, v, r, r, r, r]);
                Val::Vec4(v, f)
            }
            73 => {
                // textureGather(handle, sampler, c0..c3, o0, o1, channel)
                let (img, info) = self.image_of(a[0])?;
                self.no_offsets(&a[6..8])?;
                let n = Self::coord_count(info);
                let c = self.vector_of(&a[2..2 + n], true)?;
                let si = self.sampled_image(img, info, a[1], 0)?;
                let ch = self.m.const_u64(a[8]).ok_or("textureGather: channel not constant")? as u32;
                let chid = self.w.constant_u32(ch);
                self.texel_result(info, OP_IMAGE_GATHER, &[si, c, chid])
            }
            72 => {
                // getDimensions(handle, mip) → {width, height, depth / slices, mip levels}
                let (img, info) = self.image_of(a[0])?;
                self.w.capability(CAP_IMAGE_QUERY);
                let u = self.uint();
                let n = Self::coord_count(info) - if info.dim == 3 { 1 } else { 0 };
                let size_ty = if n == 1 { u } else { self.w.vector(u, n as u32) };
                let size = self.w.id();
                if info.storage {
                    self.emit(OP_IMAGE_QUERY_SIZE, &[size_ty, size, img]);
                } else {
                    let mip = if self.is_undef(a[1]) { self.w.constant_u32(0) } else { self.id(a[1])? };
                    self.emit(OP_IMAGE_QUERY_SIZE_LOD, &[size_ty, size, img, mip]);
                }
                let mut parts = Vec::new();
                for k in 0..3 {
                    if k < n {
                        if n == 1 {
                            parts.push(size);
                        } else {
                            let e = self.w.id();
                            self.emit(op::COMPOSITE_EXTRACT, &[u, e, size, k as u32]);
                            parts.push(e);
                        }
                    } else {
                        parts.push(self.w.constant_u32(0));
                    }
                }
                let levels = if info.storage {
                    self.w.constant_u32(1)
                } else {
                    let l = self.w.id();
                    self.emit(OP_IMAGE_QUERY_LEVELS, &[u, l, img]);
                    l
                };
                parts.push(levels);
                let v4 = self.w.vector(u, 4);
                let v = self.w.id();
                let mut ops = vec![v4, v];
                ops.extend(parts);
                self.emit(OP_COMPOSITE_CONSTRUCT, &ops);
                Val::Vec4(v, u)
            }
            _ => return err(format!("texture op {opcode}")),
        })
    }
}
