# 02 · A thousand requests, Rust version

The counterpart of `../python/main.py`: the classifier from part 1 as an HTTP service, with the same model, image maths and ONNX Runtime 1.32.0. The server is `axum` on the default multi-thread `tokio` runtime. The functions have the same names and order as the Python version.

## Requirements

[Rust](https://rustup.rs/). `rust-toolchain.toml` in this folder pins Rust 1.100.0 for this crate only; rustup installs it on the first `cargo` call and your default toolchain stays as it is. Then, from the `ai-series/` folder:

```sh
sh shared/get_onnxruntime.sh
uv run shared/export_model.py
```

## Running

From the `02-thousand-requests/rust/` folder:

```sh
cargo run --release
curl -s -X POST --data-binary @../../shared/cat.jpg localhost:8000/classify
```

`POST /classify` takes the raw image bytes as the body and answers `{"label":"Egyptian cat","prob":0.226…}`, or 400 `cannot decode image`. `GET /ping` answers `{"ok":true}`.

Settings (environment variables): `PORT` (default 8000), `ORT_THREADS` (0 = ONNX Runtime's default), `SESSIONS` (number of ONNX Runtime sessions, default 1), `LIMIT` (requests worked on at once; 0 or unset = no limit of my own), `QUEUE` (requests allowed to wait when `LIMIT` is set; one more gets `503 busy`).

## How it handles many requests

Each request runs on a blocking thread (`tokio::task::spawn_blocking`, up to 512 threads by default), so decode, resize and normalize of different requests overlap. In `ort` 2.0.0-rc.15 `Session::run` takes `&mut self`, so one session runs one inference at a time: `classify` takes a session from a small pool (`Mutex<Vec<Session>>` + `Condvar`), runs the model, and puts it back. With `SESSIONS=1` that is one inference at a time; `ORT_THREADS=1 SESSIONS=8` trades threads inside one inference for eight inferences side by side.
