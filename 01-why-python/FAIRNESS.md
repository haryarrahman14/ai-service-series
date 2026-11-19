# Is the comparison fair?

Each line is one finding; its footnote gives the evidence. The measurements behind them are in the [full audit](FAIRNESS-full.md).

## Verdict

- The three programs do the same work, and none is handicapped: each uses the libraries its own ecosystem reaches for first.[^stack]
- "The model step costs the same" holds for Python and Rust in all four cases.[^model-py-rust]
- For Go it holds on the large JPEG, but in the other three cases Go's model step is about 1 ms slower.[^model-go]
- Image handling is still most of every gap between the languages.[^prep]

## Identical in all three

- Same model files and same images.[^files]
- Same ONNX Runtime version, on CPU only, with default threads and default graph optimisation.[^ort]
- Go and Rust load the very same ONNX Runtime binary; Python's wheel is a separate build of the same commit.[^binary]
- Same measurement: 10 warm-ups, 200 timed requests, median and p95.[^timing]
- Same input to the model, to within the decoders' rounding, and the same answer.[^input]

## What differs, and who it favours

| Difference | Favours | Size |
|---|---|---|
| JPEG decoding with SIMD: Pillow and Rust have it, Go's standard library does not[^simd] | Python, Rust | the largest single reason Go's JPEG decode is slower |
| PNG inflate in pure Go[^inflate] | Python, Rust | more than half of Go's PNG decode |
| Go resizes 4 channels (RGBA) where the others resize 3[^rgba] | Python, Rust | up to a third more pixel work for Go |
| Go skips "fancy" chroma upsampling[^chroma] | Go | small; lower quality, cheaper per pixel |
| Three resize implementations of the same algorithm[^resize] | Python slightly | results agree to within one level |
| Go converts JPEG colour while resizing, not while decoding[^segments] | nobody | only moves time between two steps |
| Go's garbage collector overlaps the model step[^gc] | Python, Rust | 0.4 to 1.3 ms per small request, varies by run |
| Model time drifts with the machine's state[^drift] | nobody | up to about 10% between days |

## Checked and ruled out

- Pillow takes no shortcut on the large JPEG.[^draft]
- A faster Go resize exists, but it is a different algorithm, so it is not used.[^approx]
- The image numbers are plausible for one M1 Pro core.[^plausible]

## Fixed during the audit

- Python's `normalize` left a data copy to be paid inside the model step. It now pays for it itself.[^fixed]

[^stack]: Python: Pillow, NumPy and the official ONNX Runtime wheel. Go: `image/png`, `image/jpeg` and `golang.org/x/image/draw`. Rust: the `image` crate and `ort`, built with `--release`. [Full audit: "Is any language handicapped?"](FAIRNESS-full.md#is-any-language-handicapped)
[^model-py-rust]: Medians of five runs: within 0.29 ms (6%) of each other on MobileNetV2 and 0.08 ms (0.2%) on ResNet50. [Full audit, last section](FAIRNESS-full.md#does-the-model-step-costs-the-same-the-differences-come-from-image-handling-hold)
[^model-go]: Large JPEG: all three between 4.85 and 4.91 ms. Small PNG and small JPEG: Go 6.33 and 6.39 ms against 5.03 to 5.37 ms. ResNet50: Go 0.8 to 0.9 ms slower. Across all ten runs Go's small cases alternate between about 5.7 and 6.3 ms.
[^prep]: Small PNG: Go is 6.6 ms behind Rust, 5.4 ms of it image preparation. Large JPEG: 29.6 ms behind, all of it image preparation.
[^files]: `shared/export_model.py` writes both models and all three images; the JPEGs re-encode to byte-identical files. The large JPEG is the small picture enlarged to 1920x1277, not a native large photo. [Full audit: "The four cases"](FAIRNESS-full.md#the-four-cases)
[^ort]: ONNX Runtime 1.32.0. Python reports only `CPUExecutionProvider`; Rust is set to `AutoDevicePolicy::PreferCPU`, because `ort` would add CoreML on macOS by default. [Full audit: "Identical in all three"](FAIRNESS-full.md#identical-in-all-three)
[^binary]: Go and Rust load `shared/onnxruntime/lib/libonnxruntime` 1.32.0 (same Mach-O UUID). The Python wheel links its own copy statically; both report git commit `fab30a4`.
[^timing]: Clocks: `time.perf_counter()`, `time.Now()`, `std::time::Instant`. p95 uses the same linear interpolation in all three.
[^input]: Input pixels after resizing differ by at most 1 level out of 255 for the PNG and at most 8 for the JPEGs. Top-1 is class 285, Egyptian cat, in every language and case. Each program prints its tensor's mean and std so you can compare. [Full audit: "Same input, measured"](FAIRNESS-full.md#same-input-measured-earlier-toolchain)
[^simd]: Pillow bundles libjpeg-turbo 3.2.0 with NEON routines; Rust's `zune-jpeg` 0.5.15 uses NEON; Go's `image/jpeg` is scalar, with no assembly. Large JPEG decode: 5.98 ms Python, 8.06 ms Rust, 19.33 ms Go. [Full audit: "Which decoders actually run"](FAIRNESS-full.md#which-decoders-actually-run)
[^inflate]: Inflating the PNG's 223,692 bytes of zlib data alone takes 3.39 ms with Go's `compress/zlib` and 0.86 ms with zlib in C. Go's whole PNG decode is 6.12 ms.
[^rgba]: Go decodes an RGB PNG to `*image.RGBA`. It is what idiomatic Go code gets.
[^chroma]: Go's decoder keeps chroma at half resolution and picks the nearest sample, so colour edges differ by up to 14 levels at full size, 8 after downscaling.
[^resize]: Pillow `BILINEAR` (integers), `x/image/draw` `BiLinear` (float64), `image` `Triangle` (f32): all a separable triangle filter. Pillow's integer arithmetic helps Python; Go's float64 scratch buffer (4 to 16 MB per call) costs Go.
[^segments]: Go's decoder returns YCbCr and `x/image/draw` converts while it scales, so compare decode + resize together for Go's JPEG cases, not each step alone. Converting first does not help: 28.1 ms either way on the large JPEG.
[^gc]: Go allocates about 5 MB per request in `resize`. One diagnostic run with `GOGC=off`, not used for any published number, brought Go's small-case model step to 5.32 and 5.12 ms, level with Rust. This suggests the collector; it does not prove it.
[^drift]: With the same code and library, Python's and Rust's model step ran 0.3 to 0.6 ms faster on 18 Nov than on 17 Nov. Compare languages within one run, not across days.
[^draft]: Pillow can decode a JPEG at reduced size (`draft`), and `resize` can pre-shrink (`reducing_gap`). Neither is active: the decoded size is the full 1920x1277, and Pillow's pixels match Rust's full-size decode to within 4 levels. [Full audit: "What each timed segment contains"](FAIRNESS-full.md#what-each-timed-segment-contains)
[^approx]: `draw.ApproxBiLinear` takes 1.37 ms against 2.55 ms, but does no anti-aliasing when downscaling and changes pixels by up to 16 levels. A reused `NewScaler` (1.99 ms, identical output) is an optimisation, not the default, so it is not applied either.
[^plausible]: Large JPEG decode: 410 megapixels per second for libjpeg-turbo with NEON, 304 for zune-jpeg, 127 for scalar pure Go. `go test -bench` gives 6.10 ms for PNG decode against 6.12 ms in the program. [Full audit: "Plausibility"](FAIRNESS-full.md#plausibility)
[^fixed]: `normalize` returned a transposed NumPy view, so the copy into a real NCHW buffer happened inside the timed `infer`. It now ends with `np.ascontiguousarray(...)`. About 0.06 to 0.10 ms moved from `infer` to `normalize`; the total did not change.
