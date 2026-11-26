//! SPIR-V emission for the Vulkan backend (fork `linux-xcl-nvidia`). Design and roadmap:
//! `docs/specs/2026-11-25-vulkan-backend.md`.
//!
//! * [`writer`] — the SPIR-V binary format: ids, logical-layout sections, deduplicated
//!   types and constants. Own code: the Apache-2.0-only `rspirv` / `spirv` crates are not
//!   GPL-2.0-compatible. Opcode and enumerant numbers are from the Khronos SPIR-V
//!   specification (unified1), whose grammar is under the MIT-style Khronos licence.

pub mod from_dxil;

static WANTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the direct translator also makes SPIR-V of compute entries (`xcl shaders --spirv`,
/// or `XCL_SPIRV=1` at run time).
pub fn wanted() -> bool {
    WANTED.load(std::sync::atomic::Ordering::Relaxed) || live()
}

/// `XCL_SPIRV=1`: compute programs carry their SPIR-V in an `XSPV` part of the DXIL container,
/// which the fork's vkd3d-proton build takes instead of running dxil-spirv (Vulkan stage C).
pub fn live() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SPIRV").is_ok_and(|v| v.trim() == "1"))
}

/// `XCL_SPIRV_DIRECT_ALL=1`: with `XCL_SPIRV=1`, programs without SPIR-V take the direct DXIL
/// path too (else they go the HLSL path).
pub fn direct_all() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("XCL_SPIRV_DIRECT_ALL").is_ok_and(|v| v.trim() == "1"))
}

/// `XCL_SPIRV_ALLOW=<file>`: only programs whose DXIL container (FNV-1a 64 of its bytes, hex,
/// one per line) is listed carry SPIR-V – those whose SPIR-V passed `spirv-val` offline
/// (`xcl shaders --direct --spirv --dump`). NVIDIA's compiler crashes on invalid SPIR-V instead
/// of refusing it. Unset: every translated program carries it.
pub fn allowed(dxil: &[u8]) -> bool {
    static LIST: std::sync::OnceLock<Option<std::collections::HashSet<u64>>> = std::sync::OnceLock::new();
    let list = LIST.get_or_init(|| {
        let path = std::env::var("XCL_SPIRV_ALLOW").ok()?;
        let text = std::fs::read_to_string(&path).map_err(|e| crate::xlog_warn!("XCL_SPIRV_ALLOW {path}: {e}")).ok()?;
        let set: std::collections::HashSet<u64> = text.lines().filter_map(|l| u64::from_str_radix(l.trim(), 16).ok()).collect();
        crate::xlog_info!("SPIR-V: {} programs allowed ({path})", set.len());
        Some(set)
    });
    list.as_ref().is_none_or(|l| l.contains(&crate::gpu::shader::fnv64(dxil)))
}

/// Logs what became of a compute program's SPIR-V at run time (`attached`, or why there is
/// none): each outcome the first time, then every 100th.
pub fn note_outcome(reason: &str) {
    static SEEN: std::sync::Mutex<Option<std::collections::HashMap<String, u32>>> = std::sync::Mutex::new(None);
    let mut g = SEEN.lock().unwrap();
    let n = g.get_or_insert_with(Default::default).entry(reason.to_string()).or_insert(0);
    *n += 1;
    if *n == 1 || *n % 100 == 0 {
        crate::xlog_info!("SPIR-V: compute programs: {reason} ({n}x)");
        if std::env::var("XCL_SPIRV_DEBUG").is_ok() {
            eprintln!("SPIR-V: compute programs: {reason} ({n}x)");
        }
    }
}

/// The `XSPV` part (version 2): the workgroup size, the wave size, the root UAVs the program
/// reads through buffer device addresses (bit k: `u<k>`, from the SPIR-V generator word), then
/// the SPIR-V words. The fork's vkd3d-proton hook checks the bindings against the root
/// signature and refuses (dxil-spirv compiles the DXIL) where they differ.
pub fn xspv_part(numthreads: [u32; 3], wave: u32, words: &[u32]) -> Vec<u8> {
    let root_uavs = words.get(2).map_or(0, |g| g & 0xffff);
    [2, numthreads[0], numthreads[1], numthreads[2], wave, root_uavs].iter().chain(words).flat_map(|w| w.to_le_bytes()).collect()
}
pub fn set_wanted(on: bool) {
    WANTED.store(on, std::sync::atomic::Ordering::Relaxed);
}
pub mod writer;

#[cfg(test)]
mod writer_tests;
#[cfg(test)]
mod from_dxil_tests;
