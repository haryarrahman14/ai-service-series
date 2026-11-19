# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#     "fastapi==0.163.2",
#     "uvicorn[standard]==0.54.0",
#     "onnxruntime==1.32.0",
#     "numpy==2.5.5",
#     "pillow==13.0.0",
# ]
# ///
# AI series, part 2, Python version: the classifier from part 1 as an HTTP service.
# Run from this folder (02-thousand-requests/python/): uv run main.py
# Settings (environment variables): PORT (default 8000), ORT_THREADS (0 = ONNX Runtime's default),
# WORKERS (number of server processes, default 1), LIMIT (requests worked on at once, 0 = no limit of my own),
# QUEUE (requests allowed to wait when LIMIT is set; one more gets 503).
import asyncio
import contextlib
import io
import os
import platform
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.concurrency import run_in_threadpool
from PIL import Image

SHARED = Path(__file__).parent / "../../shared"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CAT_CLASSES = {"tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat"}
PORT = int(os.environ.get("PORT", "8000"))
ORT_THREADS = int(os.environ.get("ORT_THREADS", "0"))
WORKERS = int(os.environ.get("WORKERS", "1"))
LIMIT = int(os.environ.get("LIMIT", "0"))
QUEUE = int(os.environ.get("QUEUE", "0"))

state = {"inside": 0}  # session and labels, loaded once per server process; inside = requests working or waiting


def decode(body):
    return Image.open(io.BytesIO(body)).convert("RGB")


def resize(img):
    w, h = img.size
    scale = 256 / min(w, h)
    w, h = round(w * scale), round(h * scale)
    img = img.resize((w, h), Image.BILINEAR)
    left, top = round((w - 224) / 2), round((h - 224) / 2)
    return img.crop((left, top, left + 224, top + 224))


def normalize(img):
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis])  # HWC -> NCHW


def preprocess(body):
    return normalize(resize(decode(body)))


def infer(session, x):
    return session.run(None, {"input": x})[0][0]


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def argmax(probs):
    return int(probs.argmax())


def classify(body):
    probs = softmax(infer(state["session"], preprocess(body)))
    best = argmax(probs)
    return state["labels"][best], float(probs[best])


def load():
    options = ort.SessionOptions()
    if ORT_THREADS > 0:
        options.intra_op_num_threads = ORT_THREADS
    state["session"] = ort.InferenceSession(
        SHARED / "model" / "mobilenet_v2.onnx", options, providers=["CPUExecutionProvider"]
    )
    state["labels"] = (SHARED / "model" / "labels.txt").read_text().splitlines()
    body = (SHARED / "cat.jpg").read_bytes()
    for _ in range(10):  # warm-up, and a check that the answer is still a cat
        label, _ = classify(body)
    assert label.lower() in CAT_CLASSES, f"top-1 is not a cat: {label}"


@asynccontextmanager
async def lifespan(app):
    load()
    state["slots"] = asyncio.Semaphore(LIMIT) if LIMIT else contextlib.nullcontext()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/classify")
async def handle_classify(request: Request):
    body = await request.body()
    if LIMIT and state["inside"] >= LIMIT + QUEUE:  # the queue is full: say so at once instead of letting it wait
        return Response("busy", status_code=503)
    state["inside"] += 1
    try:
        async with state["slots"]:  # with LIMIT set, at most LIMIT requests get past this line; the rest wait in order
            # The blocking work leaves the event loop and runs on a thread from the pool (40 threads by default).
            label, prob = await run_in_threadpool(classify, body)
    except OSError:  # Pillow could not read the image
        return Response("cannot decode image", status_code=400)
    finally:
        state["inside"] -= 1
    return {"label": label, "prob": prob}


@app.get("/ping")
async def handle_ping():
    return {"ok": True}


def main():
    print(
        f"Python {platform.python_version()}, fastapi + uvicorn {uvicorn.__version__}, onnxruntime {ort.__version__}, "
        f"ORT_THREADS={ORT_THREADS} (0 = default), WORKERS={WORKERS}, LIMIT={LIMIT}, QUEUE={QUEUE}, listening on :{PORT}",
        flush=True,
    )
    uvicorn.run("main:app", host="127.0.0.1", port=PORT, workers=WORKERS, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
