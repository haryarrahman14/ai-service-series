# 02 · A thousand requests, Python version

The classifier from part 1 as an HTTP service: FastAPI on uvicorn, with the same ONNX Runtime, NumPy and Pillow code as part 1. The Go and Rust versions have the same functions, with the same names and order.

## Running

From the `02-thousand-requests/python/` folder (`../../shared` is read relative to it; once beforehand, from `ai-series/`: `uv run shared/export_model.py`):

```sh
uv run main.py
```

Settings are environment variables: `PORT` (default 8000), `ORT_THREADS` (0 or unset = ONNX Runtime's default, above 0 = `intra_op_num_threads`) `WORKERS` (number of server processes, default 1), `LIMIT` (requests worked on at once; 0 or unset = no limit of my own) and `QUEUE` (requests allowed to wait when `LIMIT` is set; one more gets `503 busy`).

```sh
curl -s -X POST --data-binary @../../shared/cat.jpg localhost:8000/classify
# {"label":"Egyptian cat","prob":0.23050259053707123}
curl -s localhost:8000/ping
# {"ok":true}
```

A body that is not a PNG or JPEG gets `400 cannot decode image`.

## How it serves

Each server process loads one MobileNetV2 session, classifies `cat.jpg` 10 times as a warm-up and stops if the answer is not a cat. uvicorn's event loop reads the request; the blocking work (`classify`) then runs on a thread from the pool FastAPI uses for blocking calls, 40 threads by default, and all threads call `session.run` on the one session. With `WORKERS` above 1, uvicorn starts that many processes, each with its own session, event loop and thread pool. Access logging is off, because Go and Rust log nothing per request either.
