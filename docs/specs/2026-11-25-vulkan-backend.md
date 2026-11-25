# Vulkan backend — design and roadmap (2026-11-25, fork `linux-xcl-nvidia`)

Status: proposal. Fork of xcl-dev/xcl; everything here is GPL-2.0-only like XCL.

## 1. Goal

Run console titles through XCL on **any** x86-64 PC with a Vulkan driver — Linux and Windows,
NVIDIA, AMD and Intel — by translating console GPU work to **Vulkan and SPIR-V directly**, instead
of D3D12 + DXIL (which on Linux then goes through vkd3d-proton and dxil-spirv to reach Vulkan anyway).

Non-goals (for now): ARM hosts (the title's x64 code runs natively, so x86-64 stays required);
replacing Wine for the title's Win32 imports (stage E, optional); upstreaming the whole backend in
one piece (only small, generic parts go upstream as separate PRs).

## 2. Why upstream chose D3D12, and why that hurts on Linux

D3D12 was the right first host API for upstream: D3D12.X is a superset of D3D12 (root signatures,
descriptor heaps, PSOs, resource states map almost 1:1), Windows was the target, and DXC/DXIL are
mature. The original design (`2026-11-20-xcl-design.md` §5.7) already names vkd3d-proton as the
route to Linux.

On Linux every pipeline then takes this path:

```
RDNA2 code ─XCL─▶ DXIL ─dxil-spirv (CPU)─▶ SPIR-V ─NVIDIA/Mesa compiler (CPU)─▶ GPU code
                 (HLSL+DXC or direct)      inside vkd3d-proton's d3d12core.dll
```

Measured on a retail title (GTA VI 1.0.2.1) with an NVIDIA desktop GPU under Proton Experimental,
vkd3d-proton 3.1 with dxil-spirv 53aac41 (2026-11-24/25):

- Menus at 30 fps; world load starts ≈85 s; from ≈120 s no present: the title waits for compute
  pipelines. 42 large compute pipelines took 1 168 s of driver time in one run (one CPU core each,
  20–190 s per pipeline); the run ended out of memory after 9 min.
- dxil-spirv recursion on large DXIL: stack overflow in all prebuild threads, a crash in
  `cfg_structurizer`; DXIL programs up to 30 MB from the HLSL path.
- vkd3d-proton on NVIDIA keeps only shader-module identifiers in its cache
  (`VK_EXT_shader_module_identifier`), so a prebuilt pipeline is reused only while the NVIDIA disk
  cache still holds it (default ≈1 GB, then eviction); the D3D12 PSO blobs XCL stores are dummies
  under vkd3d ("0 KB").

Translating DXIL to SPIR-V means rebuilding *structured* control flow from unstructured LLVM IR —
dxil-spirv's structurizer, the expensive and fragile part. XCL itself knows the structure: its
direct translator builds a structured block CFG from the console program (direct-DXIL spec,
"Translator architecture" 4) and then throws that structure away when it writes DXIL.

## 3. Target architecture

Two independent halves, done in that order:

1. **Shader backend: RDNA2 → SPIR-V.** A second emitter behind the direct translator's SSA/lane
   model (`gpu/dxil/ssa.rs`, `lanes.rs`, `lower*.rs`), next to the DXIL writer. The emitter receives
   structured control flow and writes `OpSelectionMerge` / `OpLoopMerge` itself — no generic
   structurizer.
2. **API backend: D3D12.X → Vulkan.** The host side of `hle/d3d12x` (about 36 000 lines today:
   `real.rs`, `list_x.rs`, `vram.rs`, `sparse.rs`, …) talks to Vulkan through `ash` instead of
   D3D12.

Until (2) exists, (1) goes live through **SPIR-V passthrough in our own vkd3d-proton build**
(stage C), so the shader work pays off without rewriting the API side first.

### 3.1 Shader backend

- Interface: the lowering code calls ≈155 distinct builder functions of `Shader`/`Module`
  (`set_block`, `phi`, `add_incoming`, `load`/`store`, `cast`, `extract`, constants, `dx.op`
  wrappers in `ops.rs`). Step one is a refactor with no behaviour change: put those behind a trait
  (`ShaderSink`), implemented by the existing DXIL module. The SPIR-V module implements it second.
- SPIR-V writer: own code (the binary format is a flat word stream; XCL already writes LLVM
  bitstreams by hand). Dependencies must be GPL-2.0-compatible: `rspirv`/`spirv` are Apache-2.0
  only and **cannot** be used; the Khronos SPIR-V grammar/headers (MIT-style) can, with a
  provenance note.
- Mapping: SSA values and phis map directly; `dx.op` intrinsics map to GLSL.std.450 / core ops
  (wave ops → `GroupNonUniform*`, raw buffers → SSBO/`PhysicalStorageBuffer`, groupshared →
  `Workgroup`, 64-bit/float atomics per available extensions); exactness rules stay those of the
  direct-DXIL spec (no fast-math; explicit NaN/denorm handling via `FPFastMathMode`/execution modes).
- Control flow: the structured CFG of the direct translator becomes merge/continue blocks. The
  compact form (`XCL_DIRECT_COMPACT`, a loop around a `switch(class)`) is structured by
  construction and is the first target: it covers the giant compute programs.
- Subgroup size: request 64 where supported (`VK_EXT_subgroup_size_control`, AMD), 32 on NVIDIA
  with the existing wave64-on-wave32 handling (`split_wave64`, whole-wave regions).

### 3.2 Resource and binding model

XCL's ABI today is SM 6.6 heap indexing plus a fixed root signature (`abi_rs_for`, `pso.rs`).

- Stage C (passthrough): our SPIR-V must use **vkd3d-proton's** descriptor layout for that root
  signature (the layout dxil-spirv's remapping produces: descriptor-heap/descriptor-buffer sets,
  push constants for root constants). This couples the emitter to vkd3d internals — accepted for
  the transition, isolated in one module, covered by tests that compare against dxil-spirv's
  output for the same root signature.
- Stage D (native): XCL owns the layout: one descriptor buffer or `VK_EXT_descriptor_heap`
  (where available) mirroring the console heap, push constants for user data, buffer device
  addresses for the unified-memory windows (`gpu/uma.rs`).

### 3.3 Caching

Own pipeline cache keyed by (translator source hash, SPIR-V hash, state, driver UUID), storing
`VkPipelineCache` data per pipeline or `VK_KHR_pipeline_binary` blobs — so a prebuild
(`xcl pipelines`) is reusable regardless of the NVIDIA disk-cache size. Stage C keeps vkd3d's cache.

## 4. Roadmap

| Stage | Content | Exit criterion |
|---|---|---|
| A | **Measure.** dxil-spirv time vs. driver time per large pipeline (offline: `xcl shaders --direct --dump` → time dxil-spirv and a Vulkan compile of its output); vkd3d cache hit rate in a run (`VKD3D_CONFIG=pipeline_library_log`); newer vkd3d-proton/dxil-spirv as baseline | Numbers that say how much stage B/C can gain |
| B0 | `ShaderSink` trait over the DXIL builder, no behaviour change | All direct-DXIL tests unchanged |
| B1 | SPIR-V writer + compute kernels of the test ABI; `spirv-val` as oracle; differential GPU tests through a small Vulkan compute runner (`ash`) against the existing RDNA instruction tests | ALU/memory/cross-lane tests pass on NVIDIA and AMD |
| B2 | Compact form (giant compute programs) | ScreenProbes-class programs translate, validate, compile in seconds |
| C | SPIR-V passthrough in our vkd3d-proton build (container part or private interface), compute first, DXIL fallback | GTA world load with SPIR-V compute pipelines; load time and RAM measured against stage A |
| D | Vulkan API backend for `hle/d3d12x` behind a host-API switch; graphics shaders (VS/PS/mesh) | Testtitle e2e and a retail title's menus on Vulkan |
| E | Optional: platform layer (memory, threads, files, sync) in XCL; native Linux start without Wine | Testtitle runs as a Linux process |

## 5. Testing

- Every emitted SPIR-V module through `spirv-val` in tests.
- Differential tests: the same RDNA program through DXIL (on D3D12 or vkd3d) and SPIR-V (on
  Vulkan) must give the same registers and memory — the method the direct-DXIL backend already uses.
- Corpus sweep over a title's shader pack (`xcl shaders <pack> --direct`) extended with a SPIR-V
  column (translated / validated / compiled, sizes, times).
- Live: behind a switch (`XCL_SPIRV=1`), per-pipeline fallback to the DXIL path, logged.

## 6. Risks

- Coupling to vkd3d-proton internals in stage C; mitigated by keeping it small and replacing it in D.
- Exactness differences between DXIL and SPIR-V semantics (float modes, wave ops on NVIDIA 32-lane
  subgroups); mitigated by differential tests.
- Size of stage D; it is only started if A–C confirm the gain.
- Upstream moves fast (≈150 merges a day); the fork rebases on `main` and keeps the backend in new
  files where possible to keep conflicts small.

## 7. Provenance and licensing

GPL-2.0-only like XCL. Usable: Khronos SPIR-V specification and grammar (MIT-style), Mesa
(MIT, with provenance note), dxil-spirv (MIT) as reference, `ash` (MIT OR Apache-2.0, used under
MIT). Patches to vkd3d-proton stay LGPL-2.1 in that project. Not usable: Apache-2.0-only crates
(`rspirv`, `spirv`), GPL-3.0 code (VIctory), anything from the console SDK.
