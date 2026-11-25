//! A SPIR-V module as word streams per logical-layout section (SPIR-V 1.6, §2.4), written
//! out in that order by [`Writer::finish`]. Types and constants are deduplicated, so a
//! caller asks for `uint` or `5u` wherever it needs one and gets the same id.

use std::collections::HashMap;

/// SPIR-V version written in the header (1.6, Vulkan 1.3).
pub const VERSION_1_6: u32 = 0x0001_0600;
pub const MAGIC: u32 = 0x0723_0203;

/// Opcodes used by the writer and its callers (SPIR-V specification, §3.52).
pub mod op {
    pub const UNDEF: u16 = 1;
    pub const NAME: u16 = 5;
    pub const EXTENSION: u16 = 10;
    pub const EXT_INST_IMPORT: u16 = 11;
    pub const EXT_INST: u16 = 12;
    pub const MEMORY_MODEL: u16 = 14;
    pub const ENTRY_POINT: u16 = 15;
    pub const EXECUTION_MODE: u16 = 16;
    pub const CAPABILITY: u16 = 17;
    pub const TYPE_VOID: u16 = 19;
    pub const TYPE_BOOL: u16 = 20;
    pub const TYPE_INT: u16 = 21;
    pub const TYPE_FLOAT: u16 = 22;
    pub const TYPE_VECTOR: u16 = 23;
    pub const TYPE_ARRAY: u16 = 28;
    pub const TYPE_RUNTIME_ARRAY: u16 = 29;
    pub const TYPE_STRUCT: u16 = 30;
    pub const TYPE_POINTER: u16 = 32;
    pub const TYPE_FUNCTION: u16 = 33;
    pub const CONSTANT_TRUE: u16 = 41;
    pub const CONSTANT_FALSE: u16 = 42;
    pub const CONSTANT: u16 = 43;
    pub const CONSTANT_COMPOSITE: u16 = 44;
    pub const CONSTANT_NULL: u16 = 46;
    pub const FUNCTION: u16 = 54;
    pub const FUNCTION_END: u16 = 56;
    pub const VARIABLE: u16 = 59;
    pub const LOAD: u16 = 61;
    pub const STORE: u16 = 62;
    pub const ACCESS_CHAIN: u16 = 65;
    pub const DECORATE: u16 = 71;
    pub const MEMBER_DECORATE: u16 = 72;
    pub const COMPOSITE_EXTRACT: u16 = 81;
    pub const PHI: u16 = 245;
    pub const LOOP_MERGE: u16 = 246;
    pub const SELECTION_MERGE: u16 = 247;
    pub const LABEL: u16 = 248;
    pub const BRANCH: u16 = 249;
    pub const BRANCH_CONDITIONAL: u16 = 250;
    pub const SWITCH: u16 = 251;
    pub const RETURN: u16 = 253;
    pub const UNREACHABLE: u16 = 255;
}

/// The logical-layout sections, in the order they are written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Capabilities,
    Extensions,
    ExtInstImports,
    MemoryModel,
    EntryPoints,
    ExecutionModes,
    Debug,
    Annotations,
    /// types, constants and module-scope variables (interleaved: a type must precede its use)
    Globals,
    Functions,
}

const SECTIONS: usize = 10;

pub struct Writer {
    sections: [Vec<u32>; SECTIONS],
    next_id: u32,
    /// `[opcode, operands…]` of a type or constant → its id
    unique: HashMap<Vec<u32>, u32>,
    capabilities: Vec<u32>,
    extensions: Vec<String>,
}

impl Default for Writer {
    fn default() -> Self {
        Self::new()
    }
}

impl Writer {
    pub fn new() -> Self {
        Writer { sections: Default::default(), next_id: 1, unique: HashMap::new(), capabilities: Vec::new(), extensions: Vec::new() }
    }

    /// A fresh result id.
    pub fn id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Appends one instruction to a section.
    pub fn emit(&mut self, s: Section, opcode: u16, operands: &[u32]) {
        let n = operands.len() + 1;
        assert!(n <= 0xffff, "instruction too long ({n} words)");
        let words = &mut self.sections[s as usize];
        words.push(((n as u32) << 16) | opcode as u32);
        words.extend_from_slice(operands);
    }

    /// A literal string: UTF-8, nul-terminated, padded to whole words (§2.2.1).
    pub fn string(s: &str) -> Vec<u32> {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        while bytes.len() % 4 != 0 {
            bytes.push(0);
        }
        bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    pub fn capability(&mut self, cap: u32) {
        if !self.capabilities.contains(&cap) {
            self.capabilities.push(cap);
            self.emit(Section::Capabilities, op::CAPABILITY, &[cap]);
        }
    }

    pub fn extension(&mut self, name: &str) {
        if !self.extensions.iter().any(|e| e == name) {
            self.extensions.push(name.to_string());
            self.emit(Section::Extensions, op::EXTENSION, &Self::string(name));
        }
    }

    /// `OpExtInstImport`, once per set name.
    pub fn ext_inst_import(&mut self, name: &str) -> u32 {
        let mut key = vec![op::EXT_INST_IMPORT as u32];
        key.extend(Self::string(name));
        if let Some(&id) = self.unique.get(&key) {
            return id;
        }
        let id = self.id();
        let mut ops = vec![id];
        ops.extend(Self::string(name));
        self.emit(Section::ExtInstImports, op::EXT_INST_IMPORT, &ops);
        self.unique.insert(key, id);
        id
    }

    pub fn name(&mut self, target: u32, name: &str) {
        let mut ops = vec![target];
        ops.extend(Self::string(name));
        self.emit(Section::Debug, op::NAME, &ops);
    }

    pub fn decorate(&mut self, target: u32, decoration: u32, literals: &[u32]) {
        let mut ops = vec![target, decoration];
        ops.extend_from_slice(literals);
        self.emit(Section::Annotations, op::DECORATE, &ops);
    }

    pub fn member_decorate(&mut self, ty: u32, member: u32, decoration: u32, literals: &[u32]) {
        let mut ops = vec![ty, member, decoration];
        ops.extend_from_slice(literals);
        self.emit(Section::Annotations, op::MEMBER_DECORATE, &ops);
    }

    /// A type or constant instruction whose result id comes second (constants: after the type)
    /// or first (types), deduplicated on its opcode and the other operands.
    fn unique_global(&mut self, opcode: u16, before_id: &[u32], after_id: &[u32]) -> u32 {
        let mut key = vec![opcode as u32];
        key.extend_from_slice(before_id);
        key.push(u32::MAX);
        key.extend_from_slice(after_id);
        if let Some(&id) = self.unique.get(&key) {
            return id;
        }
        let id = self.id();
        let mut ops = before_id.to_vec();
        ops.push(id);
        ops.extend_from_slice(after_id);
        self.emit(Section::Globals, opcode, &ops);
        self.unique.insert(key, id);
        id
    }

    /// A type: `OpTypeX %result operands…`. Structs that need distinct decorations
    /// (`Block`, offsets) must use [`Writer::distinct_type`].
    pub fn ty(&mut self, opcode: u16, operands: &[u32]) -> u32 {
        self.unique_global(opcode, &[], operands)
    }

    /// A type that is never shared (a struct decorated on its own).
    pub fn distinct_type(&mut self, opcode: u16, operands: &[u32]) -> u32 {
        let id = self.id();
        let mut ops = vec![id];
        ops.extend_from_slice(operands);
        self.emit(Section::Globals, opcode, &ops);
        id
    }

    pub fn void(&mut self) -> u32 {
        self.ty(op::TYPE_VOID, &[])
    }
    pub fn bool(&mut self) -> u32 {
        self.ty(op::TYPE_BOOL, &[])
    }
    pub fn int(&mut self, bits: u32, signed: bool) -> u32 {
        self.ty(op::TYPE_INT, &[bits, signed as u32])
    }
    pub fn float(&mut self, bits: u32) -> u32 {
        self.ty(op::TYPE_FLOAT, &[bits])
    }
    pub fn vector(&mut self, elem: u32, n: u32) -> u32 {
        self.ty(op::TYPE_VECTOR, &[elem, n])
    }
    pub fn pointer(&mut self, storage_class: u32, pointee: u32) -> u32 {
        self.ty(op::TYPE_POINTER, &[storage_class, pointee])
    }
    pub fn function_type(&mut self, ret: u32, params: &[u32]) -> u32 {
        let mut ops = vec![ret];
        ops.extend_from_slice(params);
        self.ty(op::TYPE_FUNCTION, &ops)
    }

    /// `OpConstant` of a scalar type; `words` are the value's little-endian words (one for
    /// 32 bits or less, two for 64).
    pub fn constant(&mut self, ty: u32, words: &[u32]) -> u32 {
        self.unique_global(op::CONSTANT, &[ty], words)
    }
    pub fn constant_u32(&mut self, v: u32) -> u32 {
        let t = self.int(32, false);
        self.constant(t, &[v])
    }
    pub fn constant_bool(&mut self, v: bool) -> u32 {
        let t = self.bool();
        self.unique_global(if v { op::CONSTANT_TRUE } else { op::CONSTANT_FALSE }, &[t], &[])
    }
    pub fn constant_composite(&mut self, ty: u32, parts: &[u32]) -> u32 {
        self.unique_global(op::CONSTANT_COMPOSITE, &[ty], parts)
    }
    pub fn constant_null(&mut self, ty: u32) -> u32 {
        self.unique_global(op::CONSTANT_NULL, &[ty], &[])
    }
    pub fn undef(&mut self, ty: u32) -> u32 {
        self.unique_global(op::UNDEF, &[ty], &[])
    }

    /// A module-scope `OpVariable` (never shared).
    pub fn global_variable(&mut self, ptr_ty: u32, storage_class: u32) -> u32 {
        let id = self.id();
        self.emit(Section::Globals, op::VARIABLE, &[ptr_ty, id, storage_class]);
        id
    }

    /// A module-scope `OpVariable` with an initializer (Private constant tables).
    pub fn global_variable_init(&mut self, ptr_ty: u32, storage_class: u32, init: u32) -> u32 {
        let id = self.id();
        self.emit(Section::Globals, op::VARIABLE, &[ptr_ty, id, storage_class, init]);
        id
    }

    /// The finished module: header, then the sections in logical-layout order.
    pub fn finish(self) -> Vec<u32> {
        let mut out = vec![MAGIC, VERSION_1_6, 0, self.next_id, 0];
        for s in self.sections {
            out.extend(s);
        }
        out
    }
}
