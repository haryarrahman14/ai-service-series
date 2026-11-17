# Fairness audit: Python vs Go vs Rust

What was checked to make sure `python/main.py`, `go/main.go` and `rust/src/main.rs` do the same work, and where they still differ. Measured on an Apple M1 Pro (8 performance + 2 efficiency cores), macOS, on AC power, under `caffeinate -i`. Raw output of the five runs quoted below: `results/run1.txt` to `run5.txt`.

## The four cases

Same list, same order, same names in all three programs.

| Case | Image | Model |
|---|---|---|
| `small PNG` | `shared/cat.png`, 451x300, 8-bit RGB PNG (0.135 MP) | MobileNetV2 |
| `small JPEG` | `shared/cat.jpg`, the same picture as baseline JPEG, quality 90, 4:2:0 (35 KB) | MobileNetV2 |
| `large JPEG` | `shared/cat-large.jpg`, the same picture **enlarged** to 1920x1277 with Lanczos, then JPEG quality 90, 4:2:0 (2.45 MP, 271 KB). It is an upscaled small image, not a native large photo: it is smoother than a real camera photo, which makes it easier to compress and probably quicker to decode | MobileNetV2 |
| `large model` | `shared/cat.png` | ResNet50 (torchvision `IMAGENET1K_V2`, 102 MB ONNX) |

`shared/export_model.py` writes both models and both JPEGs. The JPEGs are deterministic: re-encoding them gives byte-identical files (same SHA-1), and they contain only a JFIF header, with no EXIF and no ICC profile. Both models are exported the same way: input `input` [1,3,224,224], output `logits`, the same preprocessing, and one shared `labels.txt` (the two weight sets use identical category lists, which the export script asserts).

## Identical in all three

| | Python | Go | Rust |
|---|---|---|---|
| Models, images | the files above | same files | same files |
| ONNX Runtime | wheel `onnxruntime==1.32.0`; ORT is linked statically into `onnxruntime_pybind11_state.so`, which is the only ORT image mapped in the process (checked with `vmmap`) | `shared/onnxruntime/lib/libonnxruntime.dylib` (official 1.32.0 release, loaded at runtime, checked with `DYLD_PRINT_LIBRARIES`) | `shared/onnxruntime/lib/libonnxruntime.1.32.0.dylib`, the same binary as Go's (same Mach-O UUID) |
| Sessions | one per model, created once before the cases run | same | same |
| Execution provider | `CPUExecutionProvider` (`get_providers()` returns only CPU, for both models) | default session options, CPU | `AutoDevicePolicy::PreferCPU`, which registers no extra EP (ort's own default on macOS would add CoreML) |
| Threads | `intra_op_num_threads=0` (ORT default) | `nil` options (ORT default) | ort builder default: no thread setting, no global pool |
| Graph optimisation | ORT default, `ORT_ENABLE_ALL` (read back from both sessions) | ORT default | ORT default (ort does not change it) |
| Build | CPython 3.12.13; the work runs in Pillow, NumPy and ORT C/C++ code | `go build` (default, optimised, no race detector) | `cargo run --release` (opt-level 3, no profile overrides) |
| Iterations | 10 warm-ups and 200 timed requests per case | same | same |
| Clock | `time.perf_counter()` | `time.Now()` (monotonic reading) | `std::time::Instant` |
| Statistics | median, and p95 with linear interpolation (`np.percentile`) | same formula, hand-written | same formula, hand-written |

The Python wheel and the shared dylib are **separate builds of the same version**: both report ORT 1.32.0 at git commit `fab30a4`, but they are different binaries. Go and Rust load the exact same file.

## Which decoders actually run

| | PNG | JPEG |
|---|---|---|
| Python (Pillow 13.0.0) | Pillow's own PNG decoder with its bundled zlib-ng (`.dylibs/libz.1.3.1.zlib-ng.dylib`) | bundled libjpeg-turbo 3.2.0 (`features.version("libjpeg_turbo")`). It is SIMD-accelerated on arm64: the bundled `libjpeg.62.4.0.dylib` contains the NEON routines (`jsimd_*_neon`: IDCT, fancy upsampling, YCbCr to RGB) |
| Go 1.27.4 | standard `image/png` with `compress/zlib`, pure Go | standard `image/jpeg`, pure Go: no assembly in `src/image/jpeg`, and the Go compiler does not auto-vectorise |
| Rust (`image` 0.25.10) | `png` 0.18.1 with `fdeflate` | `zune-jpeg` 0.5.15 (pulled in by `image`'s `jpeg` feature, with default features). Its `neon` feature is on, and `zune-core`'s `use_neon()` returns true on aarch64, so it uses NEON for IDCT, upsampling and colour conversion |

All three decode on a single thread.

## What each timed segment contains

| Segment | Python | Go | Rust |
|---|---|---|---|
| decode | open, read, full decode to RGB. `convert("RGB")` forces Pillow's lazy `load()` (`Image.open` alone takes 0.03 ms) | open, `image.Decode`. PNG becomes `*image.RGBA`; JPEG becomes `*image.YCbCr` 4:2:0, **not yet converted to RGB** | `image::open` (reads the file into memory, then decodes to RGB8) + `to_rgb8()` copy |
| resize + crop | Pillow `BILINEAR` to 385x256, `crop` at (80, 16) | `draw.BiLinear.Scale` to 385x256, `SubImage` at (80, 16). For JPEG this step also does the YCbCr to RGB conversion, in `x/image/draw`'s specialised `YCbCr420` path | `imageops::resize` `Triangle` to 385x256, `crop_imm(...).to_image()` at (80, 16) |
| normalize | `/255`, `-mean`, `/std`, HWC to NCHW, made contiguous | same maths, written directly as NCHW | same maths, written directly as NCHW |
| infer | `session.run`; logits come back as a NumPy array | `CreateTensor` (zero-copy, pinned Go memory), `Run`, copy of 1000 logits | `Tensor::from_array` (takes ownership of the `Vec`), `run`, copy of 1000 logits |
| postprocess | softmax + argmax | same | same |

Intermediate sizes are identical in every case: 451x300 and 1920x1277 both resize to 385x256 and are cropped at left 80, top 16. All three use round-half-to-even for the size and the offsets.

**No shortcut in Pillow.** Pillow can decode a JPEG at 1/2, 1/4 or 1/8 scale in the DCT domain (`draft`), and `resize` can pre-shrink with `reducing_gap`. Neither is active here. After `Image.open(...).convert("RGB")`, `decoderconfig` is `()` (it is only set by `draft()`), the decoded size is the full 1920x1277, and the tile is a plain full-size `jpeg` tile. `resize` is called without `reducing_gap`, which defaults to `None`. Pillow's decoded pixels also match Rust's full-size decode to within 4 levels (below), which a 1/8-scale decode could not do. Go and Rust have no such mode on these code paths.

## Same input, measured

Each language's decoded image, its resized image, its input tensor and its logits were dumped once (from throwaway copies of the programs, not in this repo) and compared. Pixel values are out of 255. For Go's JPEG cases the "decoded" image is the `*image.YCbCr` converted with the standard library; Go's real pipeline never stores an RGB copy at full size.

| | Decoded pixels, max / mean abs diff | Resized 385x256, max / mean | Input tensor, max / mean | Logits, max |
|---|---|---|---|---|
| small PNG, Py vs Go | identical | 1 / 0.22 | 0.0175 / 0.0041 | 0.044 |
| small PNG, Py vs Rust | identical | 1 / 0.18 | 0.0175 / 0.0032 | 0.045 |
| small PNG, Go vs Rust | identical | 1 / 0.14 | 0.0175 / 0.0026 | 0.033 |
| small JPEG, Py vs Go | 14 / 0.40 | 8 / 0.39 | 0.139 / 0.0077 | 0.076 |
| small JPEG, Py vs Rust | 4 / 0.18 | 3 / 0.29 | 0.052 / 0.0059 | 0.096 |
| small JPEG, Go vs Rust | 14 / 0.40 | 8 / 0.38 | 0.139 / 0.0077 | 0.077 |
| large JPEG, Py vs Go | 8 / 0.21 | 2 / 0.20 | 0.034 / 0.0032 | 0.037 |
| large JPEG, Py vs Rust | 4 / 0.08 | 2 / 0.12 | 0.035 / 0.0025 | 0.042 |
| large JPEG, Go vs Rust | 8 / 0.20 | 2 / 0.20 | 0.035 / 0.0035 | 0.068 |
| large model (ResNet50), Py vs Go | identical | (as small PNG) | 0.0175 / 0.0041 | 0.031 |
| large model, Py vs Rust | identical | | 0.0175 / 0.0032 | 0.021 |
| large model, Go vs Rust | identical | | 0.0175 / 0.0026 | 0.013 |

Top-1 is class 285 (Egyptian cat) for every language in every case, and the cat self-check passes everywhere, ResNet50 included. `normalize` itself is exact: each language's own crop run through a NumPy reference gives a difference of 0.0.

Why JPEG pixels differ:
- libjpeg-turbo and zune-jpeg both use "fancy" (triangle) chroma upsampling, and they agree to within 4 levels.
- Go's `image/jpeg` keeps the chroma planes at half resolution, and the conversion to RGB picks the nearest chroma sample, so colour edges differ by up to 14 levels at full size.
- After downscaling to 385x256 that shrinks to 8 levels for the small image and 2 for the large one.

These are legitimate decoder differences. They change the probabilities in the first or second decimal; they do not change the work.

Each program prints the tensor's mean and population std per case (`input tensor: mean…, std…`), so readers can check this themselves. Go's PNG tensor sits slightly lower (mean -0.1072 against -0.1059) because `x/image/draw` rounds to 16 bits and then drops the low byte.

## Differences that remain

- **Three different resize implementations.** Pillow `BILINEAR` (fixed-point integers), `x/image/draw` `BiLinear` (float64 kernel), `image` `Triangle` (f32). All three are the same algorithm, a separable triangle filter whose support widens when downscaling; on the same input they agree to within one level. *Judgement: fair; each is its ecosystem's standard bilinear resize. Pillow's integer arithmetic favours Python; Go's float64 scratch buffer (4 MB per call for the small images, 16 MB for the large one) costs Go.*
- **Go converts JPEG colour inside `resize`, not `decode`.** libjpeg-turbo and zune-jpeg hand back RGB, so their colour conversion counts as decode. Go's decoder hands back YCbCr, and `x/image/draw` converts while it scales. Go's JPEG `decode` therefore looks cheaper and its `resize` more expensive than the same steps elsewhere; for the small JPEG, Go's resize is 3.07 ms against 2.67 ms for the PNG. Converting to RGBA first does not help: `go test -bench` on the large JPEG gives 28.1 ms for `resize` straight from YCbCr, 7.9 ms for `draw.Draw` to RGBA alone, and 28.1 to 28.4 ms for convert-then-resize. *Judgement: compare `prep` (decode + resize + normalize), not the two segments separately, for JPEG in Go. The total is fair; this is the idiomatic Go path.*
- **Go's nearest-neighbour chroma.** Skipping fancy upsampling is cheaper per pixel and lower in quality. *Judgement: a small speed advantage to Go's decoder, inherent to the standard library.*
- **Pixel format.** Go decodes an RGB PNG to `*image.RGBA` and resizes 4 channels where Python and Rust handle 3. *Judgement: biases against Go by up to a third more pixel work, but it is what idiomatic Go code gets.*
- **SIMD.** Pillow's libjpeg-turbo and Rust's zune-jpeg use NEON; Go's `image/jpeg` is scalar pure Go. *Judgement: the largest single reason Go's JPEG decode is slower. A property of the ecosystems, not a benchmark artifact. There is no standard-library alternative, only third-party (often cgo) decoders, which this brief excludes.*
- **PNG inflate.** The PNG's pixel data is 223,692 bytes of zlib, with the Paeth filter on 299 of 300 rows. Inflating it alone takes 3.39 ms with Go's `compress/zlib` and 0.86 ms with Python's `zlib` module (system zlib 1.2.12). Pillow itself uses its bundled zlib-ng, which is at least as fast. *Judgement: real; more than half of Go's PNG decode.*
- **Go's `infer` copies the logits** (`slices.Clone`, 1000 floats); Rust's `to_vec()` does the same. *Judgement: negligible (4 KB).*
- **Rust segment boundaries.** `decode` includes a `to_rgb8()` copy and `resize` includes a `to_image()` copy of the crop. Timed alone, `open` + `into_rgb8()` (no copy) costs the same 1.68 ms on the PNG. *Judgement: negligible.*
- **Python's lazy loading.** `Image.open` only reads the header; the full decode happens in `convert`, inside the timed `decode`. *Judgement: no bias.*
- **Model time is not perfectly constant across cases.** In `large JPEG`, Go's and Rust's `infer` median (5.39 and 5.32 ms) is about 0.3 to 0.4 ms below their small-image cases and below Python's 5.69 ms. It is not stable: per run, Go's value goes 5.31 / 5.23 / 5.39 / 5.59 / 5.72 ms and Rust's 5.27 / 5.30 / 5.32 / 5.54 / 5.68 ms. Same model, same input size, so the likely cause is machine state (clock and cache) after the heavy preprocessing, not the languages. *Judgement: a run-condition effect of about 7%, small next to the prep differences.*

## Fixed during the first audit

Python's `normalize` used to return a transposed NumPy *view* (HWC data with NCHW strides). The copy into a real NCHW buffer then happened inside ORT's Python binding, i.e. inside the timed `infer`, while Go and Rust pay for that layout in `normalize`. `normalize` now ends with `np.ascontiguousarray(...)`. The same work just moved segment, about 0.06 to 0.10 ms from `infer` to `normalize`. No unfairness was found that needed a code change for the new cases.

## Is any language handicapped?

- **Python**: Pillow + NumPy + the official ORT wheel is the default stack. NumPy's `normalize` (0.9 ms) makes four passes over the array with temporary arrays: ordinary NumPy code.
- **Go**: `image.Decode` with `image/png` and `image/jpeg`, plus `golang.org/x/image/draw`, is what a Go developer reaches for first. Alternatives checked with `go test -bench` on the small PNG:

  | Go resize option | time | output vs `BiLinear` |
  |---|---|---|
  | `draw.BiLinear.Scale` (used) | 2.55 ms | |
  | `draw.ApproxBiLinear.Scale` | 1.37 ms | up to 16 levels off; no anti-aliasing when downscaling, so a different, cheaper algorithm |
  | `draw.CatmullRom.Scale` | 4.00 ms | up to 21 levels off; bicubic, more expensive |
  | `draw.BiLinear.NewScaler(...)` created once, reused | 1.99 ms | identical |

  `BiLinear` is the fair equivalent. A reused `NewScaler` caches weights and the scratch buffer across requests, which Pillow and `image` do not do per call; it is an optimisation, not the default, so it is not applied. The same goes for reading the file into memory before `png.Decode`, which saves about 0.1 ms.
- **Rust**: the `image` crate and `ort` are the usual choices; the build is `--release`.

## Plausibility

Median `decode` and `resize` across the five runs, as throughput in source megapixels per second:

| | small PNG decode | small JPEG decode | large JPEG decode | large JPEG resize |
|---|---|---|---|---|
| Python | 3.40 ms = 40 MP/s | 0.75 ms = 180 MP/s | 6.13 ms = 400 MP/s | 7.04 ms = 348 MP/s |
| Go | 6.19 ms = 22 MP/s | 1.68 ms = 80 MP/s | 19.44 ms = 126 MP/s | 28.90 ms = 85 MP/s |
| Rust | 1.79 ms = 75 MP/s | 0.87 ms = 155 MP/s | 8.89 ms = 276 MP/s | 10.47 ms = 234 MP/s |

Notes on the table:
- Hundreds of MP/s for libjpeg-turbo with NEON, roughly half that for zune-jpeg, and around 100 MP/s for scalar pure-Go JPEG are believable figures for one M1 Pro core.
- The small JPEG has a lower MP/s than the large one in every language because fixed per-file costs (open, header parsing, Huffman tables) dominate at 0.135 MP, and the enlarged image compresses unusually well.

Independent checks outside the programs:
- Python `timeit` (best of 5): `decode` 3.16 / 0.55 / 5.93 ms for PNG / JPEG / large JPEG; `resize` 0.86 / 0.86 / 7.00 ms.
- Rust loop (best of 5): `decode` 1.68 / 0.75 / 8.71 ms; `resize` 1.85 / 1.85 / 10.32 ms.
- `go test -bench`: `Decode` 6.26 ms, `DecodeJPEG` 1.64 ms, `DecodeLargeJPEG` 19.96 ms; resize of the large JPEG 28.1 ms.

Each check is at or a little below its in-program median, as expected for a best-of measurement. Python's small-JPEG decode is 0.55 ms in a tight loop against 0.75 ms inside the request loop, probably because the caches are cold after each `infer`.

**Model.** `infer` median for ResNet50 against MobileNetV2 on the same image:
- Python: 37.55 / 5.67 = 6.6x
- Go: 37.75 / 5.81 = 6.5x
- Rust: 37.51 / 5.63 = 6.7x

ResNet50 is about 4.1 GMACs per image against about 0.3 GMACs for MobileNetV2, i.e. roughly 220 and 110 GFLOP/s across the CPU cores. That is plausible for ORT's MLAS kernels on an M1 Pro, and depthwise convolutions are known to use the hardware less efficiently. The three languages land within 0.6% of each other on ResNet50.

**Threads**, sampled with `ps -M` mid-benchmark (small PNG): 7 OS threads for Python and 7 for Rust (main thread plus ORT's pool); 17 for Go, the extra ones belonging to the Go runtime.

## Does "the model step costs the same; the differences come from image handling" hold?

Medians of the five runs, in ms (prep = decode + resize + normalize, per request):

| Case | model Py / Go / Rust | prep Py / Go / Rust | total Py / Go / Rust |
|---|---|---|---|
| small PNG | 5.67 / 5.81 / 5.63 | 5.19 / 9.11 / 3.81 | 10.88 / 14.98 / 9.47 |
| small JPEG | 5.67 / 5.76 / 5.64 | 2.50 / 4.98 / 2.90 | 8.22 / 10.76 / 8.53 |
| large JPEG | 5.69 / 5.39 / 5.32 | 14.07 / 48.54 / 19.48 | 19.81 / 53.89 / 24.82 |
| large model | 37.55 / 37.75 / 37.51 | 5.26 / 9.13 / 3.87 | 42.86 / 46.97 / 41.40 |

- **small PNG, small JPEG, large model: yes.** The model step is within 0.18 ms (3%) across languages for MobileNetV2 and within 0.24 ms (0.6%) for ResNet50. The total gaps (up to 5.5 ms) are almost entirely prep.
- **large JPEG: yes for the conclusion, not exactly for the premise.** The model step varies by 0.37 ms (7%, with Python slowest), but that is about 1% of the 34 ms gap between Go and Python, which is prep.
- **Which language is fastest depends on the case.** Rust is fastest overall for the PNG and ResNet50 cases. Python is fastest for both JPEG cases, because Pillow's libjpeg-turbo decode and its integer resize beat Rust's `image` crate there. Go is slowest in every case, by 2.5 ms (small JPEG) to 34 ms (large JPEG), all of it image handling.
- **Image handling's share of the request:**
  - With MobileNetV2 it is 30 to 90% depending on format and size.
  - With ResNet50 it drops to 9 to 19%, so a heavy model shrinks the language difference in relative terms. In absolute terms the difference stays the same: Go's prep costs about 4 to 5 ms more than Rust's in both PNG cases.
