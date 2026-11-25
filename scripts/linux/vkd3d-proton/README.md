# vkd3d-proton patch for linux-xcl-nvidia

`xcl-vkd3d-proton.patch` applies to vkd3d-proton master (tested at 71cae091, dxil-spirv b540c8a2).
vkd3d-proton is LGPL-2.1; this patch is under the same licence.

- `libs/vkd3d/command.c`: accept XCL's command lists and allocators that carry an extended vtable.
- `libs/vkd3d-shader/dxil.c`: a compute shader whose DXBC container has an `XSPV` part (written by
  XCL with `XCL_SPIRV=1`) is taken as SPIR-V instead of being compiled by dxil-spirv; its root-UAV
  descriptor decorations (DescriptorSet = register space, Binding = register) are rewritten to the
  root signature's bindings. Anything it cannot map falls back to dxil-spirv.

Build: cross-compile as in vkd3d-proton's README ("Compiling manually", `build-win64.txt`) and copy
`d3d12.dll` / `d3d12core.dll` into the Wine prefix's `system32`.
