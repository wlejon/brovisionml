# StyleGAN3 (image generation)

The repo's *generative* model: NVlabs StyleGAN3 ("Alias-Free GAN") in both
its rotation-equivariant **config-R** and translation-equivariant
**config-T** forms. It runs the data flow every other model here reverses —
a latent **z → w+ → RGB image** — and composes the StyleGAN3 op primitives
that live in `brotensor` (`modulated_conv2d`, `upfirdn2d`, `bias_act`,
`filtered_lrelu`, the sin/cos Fourier elementwise, `pixel_norm`) rather than
shipping its own kernels. Because those ops carry backward passes, the same
network also supports **image → W+ inversion** (see below) — the basis for
latent-space editing.

Two sub-networks, mirroring `networks_stylegan3.py`:

- **`MappingNetwork`** — `z → w+`. Normalizes `z` to unit second moment
  (`pixel_norm`), runs two leaky-ReLU fully-connected layers (with the 0.01
  learning-rate-multiplier gains baked in at load), broadcasts to
  `num_ws = num_layers + 2` rows, and applies truncation toward the stored
  `w_avg`. Runs in FP32 on every device, as the reference does.
- **`SynthesisNetwork`** — `w+ → image`. A `SynthesisInput` Fourier-feature
  map (a learned affine turns `w[0]` into a per-sample rotation+translation
  applied to fixed random frequencies, sampled on a grid as `sin` features
  and mixed by a learned matrix), followed by `num_layers + 1`
  `SynthesisLayer`s (the last being ToRGB). Each layer is `affine →
  modulated conv → filtered leaky-ReLU`, with per-layer up/down factors,
  padding, and **designed** low-pass filters derived from the band-limit
  schedule — the filters are computed here, not read from the checkpoint.

## Configs

| Preset | Family | conv kernel | channel max | down filters |
|---|---|---|---|---|
| `Config::r256()` / `r512()` / `r1024()` | config-R | 1×1 | 1024 | radially-symmetric jinc (non-critical layers) |
| `Config::t256()` / `t512()` / `t1024()` | config-T | 3×3 | 512 | separable Kaiser |

All presets: `z_dim = w_dim = 512`, `num_ws = 16`. `channel_base` doubles at
512²/1024² (65536 for config-R, 32768 for config-T), matching the released
checkpoints.

Filter design is in-library: separable Kaiser/firwin lowpass for upsampling,
windowed radial jinc for config-R's non-critical downsampling, with portable
Bessel I0/J1 (a series-expansion I0 and libm `j1` fallback when C++17
special math is unavailable, e.g. libc++/Apple clang) matching
`scipy.special` to ~1e-12.

## API

```cpp
brovisionml::stylegan3::Generator gen(brovisionml::stylegan3::Config::r256());
gen.load("weights/stylegan3-r-ffhqu-256");   // dir holding model.safetensors
gen.to(brotensor::Device::CUDA);              // optional GPU migration

auto img = gen.generate(z, /*truncation_psi=*/0.7f);  // one-shot z → 8-bit RGB
auto ws  = gen.map(z, 0.7f);                          // (num_ws, w_dim) W+
auto raw = gen.synthesize(ws);                        // raw FP32 NCHW image
auto png = gen.render(ws);                            // W+ → 8-bit RGB
```

`map()` applies truncation toward the stored `w_avg` (`truncation_psi = 1`
disables; `truncation_cutoff < 0` truncates all rows).

The `stylegan3_generate` CLI driver:

```bash
stylegan3_generate weights/stylegan3-r-ffhqu-256 --res 256 --seed 42 \
    --trunc 0.7 --out out.png --cuda
# flags: --res 256|512|1024 (must match checkpoint)  --seed N  --trunc PSI
#        --out PATH  --cuda
```

## Inversion (image → W+)

`Generator::invert()` projects an image into W+ with an Adam optimizer over
the synthesis backward (cosine LR schedule with warm-up, per-sample
image-space MSE, optional L2 pull toward `w_avg`):

```cpp
brovisionml::stylegan3::Generator::InvertOptions opt;
opt.num_steps = 350;        // Adam iterations (default)
opt.lr        = 0.05f;
opt.reg_w     = 0.0f;       // L2 regularization toward w_avg (0 = off)
opt.init_w    = prev.w;     // optional: resume/refine from a prior result
opt.on_step   = [](int step, float mse) { /* progress */ };

auto res = gen.invert(target_image, opt);   // target: a stylegan3::Image
// res.w    — recovered W+ (num_ws, w_dim)
// res.loss — final image-space MSE
```

`init_w` resumes from a previous latent instead of `w_avg`, enabling
progressive refinement. The inversion path (forward_cached + backward) runs
FP32 regardless of the FP16 synthesis setting, to keep the gradient chain
exact.

## GPU path

On GPU the top `Config::num_fp16_res` (default 4) resolution bands of the
forward `synthesize()` run **FP16** — the modulated conv routes through
brotensor's WMMA implicit-GEMM, the filtered leaky-ReLU takes FP16 I/O with
FP32 FIR math, and frozen weights/filters pre-cast at `to()` time.
`Config::force_fp32 = true` disables the fast path for parity debugging.
Mapping and inversion stay FP32. See [performance.md](performance.md).

## Weights

StyleGAN3 ships as NVlabs Python *pickles*, so there is a one-time
conversion step — the only model with an in-repo converter, because
unpickling needs the NVlabs repo on `PYTHONPATH`:

```bash
scripts/download-stylegan3.sh stylegan3-r-ffhqu-256   # download + convert
# or by hand:
python scripts/convert-stylegan3.py stylegan3-r-afhqv2-512x512.pkl \
    weights/stylegan3-r-afhqv2-512/model.safetensors \
    --repo /path/to/NVlabs/stylegan3
```

The converter flattens the EMA generator's `state_dict` into FP32
safetensors with the exact tensor names the loader reads (`mapping.fc{i}.*`,
`mapping.w_avg`, `synthesis.input.*`,
`synthesis.L{idx}_{size}_{channels}.*`); the per-layer
`up_filter`/`down_filter` buffers are dropped since brovisionml designs
them. Available variants: `stylegan3-{r,t}-ffhqu-256`, `-afhqv2-512`,
`-ffhq-1024`, `-ffhqu-1024`, `-metfaces-1024`, `-metfacesu-1024`.

## Parity & tests

`tests/test_stylegan3_parity.cpp` always validates the config-R / config-T
channel schedules structurally (layer names and channel tapers match the
released presets). With a converted checkpoint plus a golden
(`scripts/dump-stylegan3-golden.py`, which replays a fixed latent through
the NVlabs generator in FP32) it gates numerically: mapping mean-abs <
5e-4, synthesis mean < 5e-3 / max < 5e-2 — tolerances that hold for the
CUDA FP16 fast path too (measured max ~2.4e-2). GPU gates run on CUDA or
Metal; the CPU gate runs at ≤256² only.

`test_stylegan3_invert.cpp` certifies the gradient chain
(`forward_cached(ws) == synthesize(ws)` exactly, finite-difference checks
per layer) and runs a generate→invert round trip, requiring the final MSE
and the rendered uint8 difference to drive tight. `test_stylegan3_generate`
runs structural checks always and, with weights present, the full pipeline
with CPU/GPU parity. All weight-gated tests skip cleanly when `weights/` is
empty.
