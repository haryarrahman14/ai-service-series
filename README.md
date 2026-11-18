# AI Service Series

One example, three languages. Each part has an article (`article.md`, with the Indonesian translation in `article.id.md`) and code you can run.

The shared example for parts 1–4: image classification with MobileNetV2. The model is exported to ONNX once (`shared/export_model.py`), and the same file is then run from Python, Go and Rust.

## Status

- [x] 01 · Why is almost every AI service written in Python? → `01-why-python/article.md`
- [ ] 02 · Python vs Go vs Rust as HTTP services under load
- [ ] 03 · To be decided
- [ ] 04 · To be decided

## Outline of parts 1–4

### 01 · Why is almost every AI service written in Python?

Where the time of one inference request actually goes, measured with the same model, images and ONNX Runtime in Python, Go and Rust.

### 02 · Python vs Go vs Rust as HTTP services under load

The same model, images and machine, served over HTTP from Python, Go and Rust, then measured: latency, throughput under many concurrent requests, memory, startup time, build size and lines of code.

### 03 · To be decided

### 04 · To be decided

## Series rules

- Every number comes from the code in this repo, run on the machine named in its article.
- Claims about libraries are checked against their documentation or source repo, and linked in the Sources section.
