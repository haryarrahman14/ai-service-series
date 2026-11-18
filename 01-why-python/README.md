# 01 · Why Python

The code and article for part 1 of the AI Service series: why Python is used in AI, and where the time of one inference request actually goes. Serving needs only ONNX Runtime, NumPy and Pillow, no PyTorch.

## Requirements

[`uv`](https://docs.astral.sh/uv/). `uv` takes care of Python and all dependencies. The programs run on Python 3.14 (the newest version onnxruntime 1.32.0 ships wheels for); `shared/export_model.py` stays on Python 3.12.

## Running

From the repo root:

```sh
uv run shared/export_model.py
cd 01-why-python/python && uv run matmul.py && uv run main.py
```

## Expected output

- `export_model.py`: the paths of `shared/model/mobilenet_v2.onnx`, `resnet50.onnx`, `labels.txt`, `cat.png`, `cat.jpg` and `cat-large.jpg`, plus the model and image sizes.
- `python/matmul.py`: the time of a 200x200 matrix multiplication in pure Python vs NumPy, and the ratio.
- `python/main.py`, for each case: section 1 shows the top-3 guesses (top-1 must be a cat class) plus the mean and std of the input tensor; section 2 shows the median and p95 time of decode, resize + crop, normalize, infer and postprocess, with each step's share of the total per request.
- At the end: a `Summary` table per case (prep = decode + resize + normalize, model = infer, total).

Four cases, in the same order and with the same names in all three languages: `small PNG` (`cat.png` + MobileNetV2), `small JPEG` (`cat.jpg`, the same picture as a quality-90 JPEG), `large JPEG` (`cat-large.jpg`, the same picture *enlarged* to 1920 px wide, not a native large photo), and `large model` (`cat.png` + ResNet50).

## Three languages, one request

The same request is written in three folders, `python/`, `go/` and `rust/`, with the same model, images and ONNX Runtime. The three `main` files have the same functions, with the same names and order, so they can be read side by side line by line. Needs Go (tested with 1.27.4) and Rust; `rust/rust-toolchain.toml` pins Rust 1.100.0 for that crate only.

```sh
sh shared/get_onnxruntime.sh
sh 01-why-python/run_all.sh
```

`run_all.sh` runs `python/matmul.py` first, then the three programs. Each program runs the same four cases and times itself: warm-up, then median and p95 per step, and a `Summary` table at the end. The raw output of five runs is in `results/`. Close other heavy work first; a busy machine spoils the numbers. What was checked to keep the comparison fair is in `FAIRNESS.md`.

Cross-check for Go, from `go/`: `go test -bench . -benchmem -run '^$' -count 3`.

## Contents

- `python/`, `go/`, `rust/`: the same program in three languages.
- `FAIRNESS.md`: fairness audit of the comparison.
- `article.md`: the article in English.
- `article.id.md`: the Indonesian translation, same content.
- `img/id/`, `img/en/`: six article illustrations per language (PNG 3200x1800), including `6-four-cases.png` (scoreboard of the four cases).
- `results/`: raw output of the five `run_all.sh` runs used by the tables in the article.
- `proof/`: terminal screenshots of the runs used by the article.
