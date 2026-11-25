use super::writer::{op, Section, Writer, MAGIC, VERSION_1_6};

/// The smallest valid compute module: `void main() {}` with a 64x1x1 workgroup.
pub(super) fn empty_compute() -> Vec<u32> {
    let mut w = Writer::new();
    w.capability(1); // Shader
    w.emit(Section::MemoryModel, op::MEMORY_MODEL, &[0, 1]); // Logical GLSL450
    let void = w.void();
    let fn_ty = w.function_type(void, &[]);
    let main = w.id();
    let mut ep = vec![5, main]; // GLCompute
    ep.extend(Writer::string("main"));
    w.emit(Section::EntryPoints, op::ENTRY_POINT, &ep);
    w.emit(Section::ExecutionModes, op::EXECUTION_MODE, &[main, 17, 64, 1, 1]); // LocalSize
    w.name(main, "main");
    w.emit(Section::Functions, op::FUNCTION, &[void, main, 0, fn_ty]);
    let label = w.id();
    w.emit(Section::Functions, op::LABEL, &[label]);
    w.emit(Section::Functions, op::RETURN, &[]);
    w.emit(Section::Functions, op::FUNCTION_END, &[]);
    w.finish()
}

#[test]
fn header_and_layout() {
    let m = empty_compute();
    assert_eq!(&m[..3], &[MAGIC, VERSION_1_6, 0]);
    assert_eq!(m[4], 0, "schema");
    // the first instruction is OpCapability Shader, the last OpFunctionEnd
    assert_eq!(m[5], (2 << 16) | op::CAPABILITY as u32);
    assert_eq!(*m.last().unwrap(), (1 << 16) | op::FUNCTION_END as u32);
    // every instruction's word count adds up to the module's length
    let mut i = 5;
    while i < m.len() {
        let n = (m[i] >> 16) as usize;
        assert!(n > 0);
        i += n;
    }
    assert_eq!(i, m.len());
}

#[test]
fn types_and_constants_are_shared() {
    let mut w = Writer::new();
    let a = w.int(32, false);
    let b = w.int(32, false);
    let s = w.int(32, true);
    assert_eq!(a, b);
    assert_ne!(a, s);
    let c1 = w.constant_u32(5);
    let c2 = w.constant_u32(5);
    let c3 = w.constant_u32(6);
    assert_eq!(c1, c2);
    assert_ne!(c1, c3);
    // same literal, other type: another constant
    let c4 = w.constant(s, &[5]);
    assert_ne!(c1, c4);
    assert_eq!(w.constant_bool(true), w.constant_bool(true));
    assert_ne!(w.constant_bool(true), w.constant_bool(false));
    assert_eq!(w.ext_inst_import("GLSL.std.450"), w.ext_inst_import("GLSL.std.450"));
}

#[test]
fn strings_are_nul_terminated_and_padded() {
    assert_eq!(Writer::string(""), vec![0]);
    assert_eq!(Writer::string("abc"), vec![u32::from_le_bytes(*b"abc\0")]);
    assert_eq!(Writer::string("main"), vec![u32::from_le_bytes(*b"main"), 0]);
}

/// With `XCL_SPIRV_VAL` set to a `spirv-val` executable, the module also passes the
/// Khronos validator (Vulkan 1.3 environment); without it the test checks nothing more.
#[test]
fn empty_compute_validates() {
    let Ok(val) = std::env::var("XCL_SPIRV_VAL") else { return };
    let words = empty_compute();
    let path = std::env::temp_dir().join(format!("xcl-spirv-test-{}.spv", std::process::id()));
    std::fs::write(&path, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    let out = std::process::Command::new(val).arg("--target-env").arg("vulkan1.3").arg(&path).output().expect("spirv-val");
    let _ = std::fs::remove_file(&path);
    assert!(out.status.success(), "spirv-val: {}", String::from_utf8_lossy(&out.stderr));
}
