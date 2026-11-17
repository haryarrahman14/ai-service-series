# 01 · Why Python, Python version

One image classification request with ONNX Runtime 1.32.0, NumPy and Pillow, without PyTorch. The functions have the same names and order as the Go and Rust versions, so the three files can be read side by side.

## Requirements

[`uv`](https://docs.astral.sh/uv/). `uv` takes care of Python and all dependencies. Once, from the `ai-series/` folder:

```sh
uv run shared/export_model.py
```

## Running

From the `01-why-python/python/` folder:

```sh
uv run main.py     # per case: [1] classification, [2] where the request time goes; then Summary
uv run matmul.py   # pure Python vs NumPy, 200x200 matrix multiplication
```

Four cases, in the same order and with the same names in all three languages: `small PNG` (`cat.png` + MobileNetV2), `small JPEG` (`cat.jpg`, the same picture as a quality-90 JPEG), `large JPEG` (`cat-large.jpg`, the same picture *enlarged* to 1920 px wide, not a native large photo), and `large model` (`cat.png` + ResNet50).

## How it measures

`benchmark` runs 200 requests after 10 warm-ups per case; each request is timed per step (`decode`, `resize`, `normalize`, `infer`, `softmax` + `argmax`) with `time.perf_counter()`, and the median and p95 are printed.
