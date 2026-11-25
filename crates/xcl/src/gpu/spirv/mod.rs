//! SPIR-V emission for the Vulkan backend (fork `linux-xcl-nvidia`). Design and roadmap:
//! `docs/specs/2026-11-25-vulkan-backend.md`.
//!
//! * [`writer`] — the SPIR-V binary format: ids, logical-layout sections, deduplicated
//!   types and constants. Own code: the Apache-2.0-only `rspirv` / `spirv` crates are not
//!   GPL-2.0-compatible. Opcode and enumerant numbers are from the Khronos SPIR-V
//!   specification (unified1), whose grammar is under the MIT-style Khronos licence.

pub mod writer;

#[cfg(test)]
mod writer_tests;
