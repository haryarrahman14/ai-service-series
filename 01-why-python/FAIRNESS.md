# Fairness audit: Python vs Go vs Rust

> Note: the article's tables use the median of ten runs (`results/run1.txt` to `run10.txt`). The tables in this audit were computed from the first five (`run1` to `run5`). Across all ten, Go's model step in the two small cases alternates between about 5.7 ms (level with Python and Rust) and about 6.3 ms; the cause is not established. A single test with Go's garbage collector off brought it level with Rust, which suggests the collector but does not prove it.

What was checked to make sure `python/main.py`, `go/main.go` and `rust/src/main.rs` do the same work, and where they still differ. Measured on an Apple M1 Pro (8 performance + 2 efficiency cores), macOS, on AC power, under `caffeinate -i`. Versions: CPython 3.14.6 (uv's build), Go 1.27.4, Rust 1.100.0 (pinned for the crate by `rust/rust-toolchain.toml`); onnxruntime 1.32.0, numpy 2.5.5, Pillow 13.0.0, ort 2.0.0-rc.15, image 0.25.10, golang.org/x/image 0.48.0, ONNX Runtime Go binding at the v1.32.0 release commit (`fab30a4`). Raw output of the five runs quoted below: `results/run1.txt` to `run5.txt` (measured 2026-11-18). Sections marked *earlier toolchain* were measured on 2026-11-17 with CPython 3.12.13 and Rust 1.97.1; the code they examine is unchanged.

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
| Build | CPython 3.14.6; the work runs in Pillow, NumPy and ORT C/C++ code | `go build` with Go 1.27.4 (default, optimised, no race detector) | `cargo run --release` with rustc 1.100.0 (opt-level 3, no profile overrides) |
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

## Same input, measured (earlier toolchain)

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
- **Model time is not perfectly constant across cases or days.** On 2026-11-17 Go's and Rust's `large JPEG` infer ran 0.3 to 0.4 ms below their small-image cases. In the 2026-11-18 runs every language's MobileNetV2 infer is lower in `large JPEG` (4.85 to 4.91 ms) than in the small cases, and Python's and Rust's infer is 0.3 to 0.6 ms lower than the day before for the same code and library. Same model and input size, so this is machine state (clocks, caches), not the languages. *Judgement: a run-condition effect of up to about 10%; compare languages within one run, not across days.*
- **Go's garbage collector can overlap `infer`.** In the 2026-11-18 runs Go's MobileNetV2 infer in the two small cases is 6.33 and 6.39 ms (median of five), against 5.03 to 5.37 ms for Python and Rust, which call the same ORT; four of Go's five runs are at about 6.3 ms, one at 5.7 ms. Go allocates about 5 MB per request in `resize`, so the GC's background workers often run while ORT's threads are busy. A diagnostic run with `GOGC=off` (not used for any published number) brought Go's small-case infer to 5.32 and 5.12 ms, level with Rust; a normal run right after gave 5.76 and 5.71 ms. *Judgement: a real cost of Go's runtime in this workload, worth 0.4 to 1.3 ms per small request and variable between runs; not an unfairness to fix, since turning off the GC would be tuning.*

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
| Python | 3.31 ms = 41 MP/s | 0.68 ms = 198 MP/s | 5.98 ms = 410 MP/s | 6.99 ms = 351 MP/s |
| Go | 6.12 ms = 22 MP/s | 1.65 ms = 82 MP/s | 19.33 ms = 127 MP/s | 28.03 ms = 87 MP/s |
| Rust | 1.75 ms = 77 MP/s | 0.77 ms = 175 MP/s | 8.06 ms = 304 MP/s | 9.88 ms = 248 MP/s |

Notes on the table:
- Hundreds of MP/s for libjpeg-turbo with NEON, somewhat less for zune-jpeg, and around 100 MP/s for scalar pure-Go JPEG are believable figures for one M1 Pro core.
- The small JPEG has a lower MP/s than the large one in every language because fixed per-file costs (open, header parsing, Huffman tables) dominate at 0.135 MP, and the enlarged image compresses unusually well.
- Rust's image steps are 5 to 10% faster than with Rust 1.97.1 (small PNG resize 1.89 to 1.71 ms, large JPEG decode 8.89 to 8.06 ms) with the same crate versions, which fits a compiler upgrade.

`go test -bench` on 2026-11-18: `Decode` 6.10 ms, `DecodeJPEG` 1.60 ms, `DecodeLargeJPEG` 19.47 ms, each at or a little below the in-program median. *Earlier toolchain* independent checks: Python `timeit` (best of 5) `decode` 3.16 / 0.55 / 5.93 ms for PNG / JPEG / large JPEG and `resize` 0.86 / 0.86 / 7.00 ms; a Rust loop (best of 5) `decode` 1.68 / 0.75 / 8.71 ms and `resize` 1.85 / 1.85 / 10.32 ms; `go test -bench` large-JPEG resize 28.1 ms. Python's small-JPEG decode is faster in a tight loop than inside the request loop, probably because caches are cold after each `infer`.

**Pure Python.** `matmul.py`'s pure-Python 200x200 multiplication takes 337 ms on CPython 3.14.6 (median of five) against 281 ms on 3.12.13; NumPy is unchanged at 0.058 ms. A micro-benchmark on this machine isolates the slowdown to the float multiply-accumulate in the inner loop (`s += y * y`: 130 ms on 3.12.13, 171 ms on 3.14.6), while integer loops and list indexing are within a few percent. Homebrew's CPython 3.14.4 is slower still (about 375 ms).

**Model.** `infer` median for ResNet50 against MobileNetV2 on the same image:
- Python: 36.59 / 5.37 = 6.8x
- Go: 37.39 / 6.33 = 5.9x (Go's small-case MobileNetV2 time is inflated by GC overlap, see above)
- Rust: 36.51 / 5.08 = 7.2x

ResNet50 is about 4.1 GMACs per image against about 0.3 GMACs for MobileNetV2, i.e. roughly 220 and 110 GFLOP/s across the CPU cores. That is plausible for ORT's MLAS kernels on an M1 Pro, and depthwise convolutions are known to use the hardware less efficiently. The three languages land within 2.4% of each other on ResNet50 (Go highest).

**Threads** (*earlier toolchain*), sampled with `ps -M` mid-benchmark (small PNG): 7 OS threads for Python and 7 for Rust (main thread plus ORT's pool); 17 for Go, the extra ones belonging to the Go runtime.

## Does "the model step costs the same; the differences come from image handling" hold?

Medians of the five runs, in ms (prep = decode + resize + normalize, per request):

| Case | model Py / Go / Rust | prep Py / Go / Rust | total Py / Go / Rust |
|---|---|---|---|
| small PNG | 5.37 / 6.33 / 5.08 | 5.06 / 9.00 / 3.58 | 10.44 / 15.33 / 8.70 |
| small JPEG | 5.15 / 6.39 / 5.03 | 2.43 / 4.90 / 2.60 | 7.60 / 11.32 / 7.66 |
| large JPEG | 4.91 / 4.88 / 4.85 | 13.88 / 47.58 / 18.07 | 18.93 / 52.55 / 22.95 |
| large model | 36.59 / 37.39 / 36.51 | 5.16 / 9.06 / 3.67 | 41.78 / 46.56 / 40.20 |

- **Python and Rust: yes, in all four cases.** Their model step is within 0.29 ms (6%) on MobileNetV2 and 0.08 ms (0.2%) on ResNet50, and their total gap is prep.
- **Go: mostly, not entirely.** In `large JPEG` all three run the model in 4.85 to 4.91 ms. In the two small cases Go's model step is 1.0 to 1.4 ms slower than the other two, and on ResNet50 0.8 to 0.9 ms slower, because Go's garbage collector overlaps `infer` (see above). Image handling is still most of Go's gap: in `small PNG` Go is 6.6 ms behind Rust, 5.4 ms of it prep; in `large JPEG` 29.6 ms behind, all of it prep.
- **Which language is fastest depends on the case.** Rust is fastest for the PNG and ResNet50 cases. Python is fastest for `large JPEG` by 4.0 ms, because Pillow's libjpeg-turbo decode and its integer resize beat Rust's `image` crate there. For `small JPEG` Python and Rust are tied (7.60 vs 7.66 ms; the run ranges 7.49 to 7.95 and 7.64 to 7.93 overlap). Go is slowest in every case, by 3.7 ms (small JPEG) to 33.6 ms (large JPEG).
- **Image handling's share of the request:**
  - With MobileNetV2 it is 32 to 91% depending on language, format and size.
  - With ResNet50 it drops to 9 to 19%, so a heavy model shrinks the language difference in relative terms. In absolute terms it stays: Go's prep costs about 5.4 ms more than Rust's in both PNG cases.
