//! SPIR-V emission for the Vulkan backend (fork `linux-xcl-nvidia`). Design and roadmap:
//! `docs/specs/2026-11-25-vulkan-backend.md`.
//!
//! * [`writer`] — the SPIR-V binary format: ids, logical-layout sections, deduplicated
//!   types and constants. Own code: the Apache-2.0-only `rspirv` / `spirv` crates are not
//!   GPL-2.0-compatible. Opcode and enumerant numbers are from the Khronos SPIR-V
//!   specification (unified1), whose grammar is under the MIT-style Khronos licence.

pub mod from_dxil;

static WANTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the direct translator also makes SPIR-V of compute entries (`xcl shaders --spirv`).
pub fn wanted() -> bool {
    WANTED.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_wanted(on: bool) {
    WANTED.store(on, std::sync::atomic::Ordering::Relaxed);
}
pub mod writer;

#[cfg(test)]
mod writer_tests;
#[cfg(test)]
mod from_dxil_tests;
