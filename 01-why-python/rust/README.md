# 01 · Why Python, Rust version

The counterpart of `../python/main.py`: section [1] (classification) and [2] (where the request time goes) for each case, with the same model, images and ONNX Runtime 1.32.0. PNG and JPEG are read by the `image` crate (features `png` and `jpeg`). The functions have the same names and order as the Python and Go versions.

## Requirements

[Rust](https://rustup.rs/), then from the `ai-series/` folder:

```sh
sh shared/get_onnxruntime.sh
uv run shared/export_model.py
```

## Running

From the `01-why-python/rust/` folder:

```sh
cargo run --release
```

No env var needed: the `libonnxruntime` path is set in code with `ort::init_from`.

Four cases, in the same order and with the same names in all three languages: `small PNG` (`cat.png` + MobileNetV2), `small JPEG` (`cat.jpg`, the same picture as a quality-90 JPEG), `large JPEG` (`cat-large.jpg`, the same picture *enlarged* to 1920 px wide, not a native large photo), and `large model` (`cat.png` + ResNet50).

## How it measures

`benchmark` runs 200 requests after 10 warm-ups per case; each request is timed per step (`decode`, `resize`, `normalize`, `infer`, `softmax` + `argmax`) with `std::time::Instant`, and the median and p95 are printed.
