# 01 · Why Python, Go version

The same request as `../python/main.py`, rewritten in Go with the same ONNX Runtime (the official binding `github.com/microsoft/onnxruntime/go`, at the v1.32.0 tag commit). The functions have the same names and order as the Python and Rust versions, so the three files can be read side by side.

## Requirements

Go 1.27+ and the Xcode Command Line Tools (cgo). Once, from the `ai-series/` folder:

```sh
sh shared/get_onnxruntime.sh
uv run shared/export_model.py
```

## Running

From the `01-why-python/go/` folder (`main.go` and `../../shared` are read relative to it):

```sh
go build -o whypython && ./whypython
```

No env var needed: the `libonnxruntime.dylib` path is set in code with `ort.SetSharedLibraryPath`.

Four cases, in the same order and with the same names in all three languages: `small PNG` (`cat.png` + MobileNetV2), `small JPEG` (`cat.jpg`, the same picture as a quality-90 JPEG), `large JPEG` (`cat-large.jpg`, the same picture *enlarged* to 1920 px wide, not a native large photo), and `large model` (`cat.png` + ResNet50).

## How it measures

`benchmark` times each iteration with `time.Now()` (monotonic clock) per step: `decode`, `resize`, `normalize`, `infer`, then `softmax` + `argmax`; it prints the median and p95 of 200 iterations after 10 warm-ups per case.

`check` and `ms` at the end of the file are only helpers Go needs (error check, `time.Duration` to milliseconds).

`decode` uses `image.Decode` with the PNG and JPEG decoders from the standard library. As a cross-check, `main_test.go` times the same functions (including decoding both JPEGs) with Go's built-in benchmarks:

```sh
go test -bench . -benchmem -run '^$'
```
