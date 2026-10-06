---
name: Bug report
about: A model's output differs from the reference implementation, a checkpoint fails to load, a crash, or a test fails
labels: bug
---

**The model and input:** the model (SAM, Depth-Anything, DSINE, ...), the
checkpoint, the image (attach it if you can) and any prompts or options.

**What the reference produces** (the upstream PyTorch implementation or
`controlnet_aux` with the same weights and input):

**What brovisionml produced instead** (the output image or numbers, the
error, a crash, or the failing `ctest --output-on-failure` output — paste
it):

```
```

**Environment:**
- OS:
- `--device` / brotensor backend (CPU / CUDA / Metal / Vulkan), GPU and driver version:
- Compiler / toolchain (MSVC / GCC / Clang):
- Called from C++, a CLI tool, or JavaScript (`bro.vision`):
- brovisionml commit, and brotensor commit if built from siblings:
