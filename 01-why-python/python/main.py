# /// script
# requires-python = ">=3.14,<3.15"
# dependencies = [
#     "onnxruntime==1.32.0",
#     "numpy==2.5.5",
#     "pillow==13.0.0",
# ]
# ///
# AI series, part 1, Python version: one classification request, and where its time goes.
# Run from this folder (01-why-python/python/): uv run main.py
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SHARED = Path(__file__).parent / "../../shared"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CAT_CLASSES = {"tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat"}
MODELS = {"MobileNetV2": "mobilenet_v2.onnx", "ResNet50": "resnet50.onnx"}
CASES = [  # name, image, model
    ("small PNG", "cat.png", "MobileNetV2"),
    ("small JPEG", "cat.jpg", "MobileNetV2"),
    ("large JPEG", "cat-large.jpg", "MobileNetV2"),  # cat.png enlarged to 1920 px, not a native large photo
    ("large model", "cat.png", "ResNet50"),
]


def decode(path):
    return Image.open(path).convert("RGB")


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
    return np.ascontiguousarray(x.transpose(2, 0, 1)[np.newaxis])  # HWC -> NCHW, copied here like Go/Rust


def preprocess(path):
    return normalize(resize(decode(path)))


def infer(session, x):
    return session.run(None, {"input": x})[0][0]


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def argmax(probs):
    return int(probs.argmax())


def median(values):
    return float(np.median(values))


def p95(values):
    return float(np.percentile(values, 95))  # linear interpolation


def classify(session, labels, image_path):
    x = preprocess(image_path)
    probs = softmax(infer(session, x))
    top = np.argsort(probs)[::-1][:3]
    print("[1] Classifying (top-3, ONNX Runtime CPU)")
    for i in top:
        print(f"    {labels[i]:<15} {probs[i] * 100:.2f}%")
    print(f"    input tensor: mean {x.mean(dtype=np.float64):.4f}, std {x.std(dtype=np.float64):.4f}")
    assert labels[top[0]].lower() in CAT_CLASSES, f"top-1 is not a cat: {labels[top[0]]}"
    print()


def benchmark(session, image_path):
    for _ in range(10):
        infer(session, preprocess(image_path))

    names = ["decode", "resize + crop", "normalize", "infer (ONNX Runtime)", "postprocess"]
    times, prep, total = [[] for _ in names], [], []
    for _ in range(200):
        t0 = time.perf_counter()
        img = decode(image_path)
        t1 = time.perf_counter()
        img = resize(img)
        t2 = time.perf_counter()
        x = normalize(img)
        t3 = time.perf_counter()
        logits = infer(session, x)
        t4 = time.perf_counter()
        argmax(softmax(logits))
        t5 = time.perf_counter()
        t = [t0, t1, t2, t3, t4, t5]
        for i in range(5):
            times[i].append((t[i + 1] - t[i]) * 1000)
        prep.append(sum(times[i][-1] for i in range(3)))
        total.append(sum(times[i][-1] for i in range(5)))

    print("[2] Where one request spends its time (200 requests after 10 warm-ups)")
    print(f"    {'':<20} {'median':>11} {'p95':>11} {'share':>8}")
    for name, ms in zip(names, times):
        print(f"    {name:<20} {median(ms):8.3f} ms {p95(ms):8.3f} ms {median(ms) / median(total) * 100:7.1f}%")
    print(f"    {'total':<20} {median(total):8.3f} ms {p95(total):8.3f} ms")
    return median(prep), median(times[3]), median(total)


def main():
    source = Path(__file__).read_text().split("# ///\n", 2)[-1]
    code_lines = [l for l in source.splitlines() if l.strip() and not l.strip().startswith("#")]
    print(f"main.py: {len(code_lines)} lines of code\n")

    labels = (SHARED / "model" / "labels.txt").read_text().splitlines()
    sessions = {
        name: ort.InferenceSession(SHARED / "model" / file, providers=["CPUExecutionProvider"])
        for name, file in MODELS.items()
    }

    summary = []
    for n, (name, file, model) in enumerate(CASES, 1):
        print(f"== case {n}: {name} ({file}, {model}) ==\n")
        image_path = SHARED / file
        classify(sessions[model], labels, image_path)
        summary.append(benchmark(sessions[model], image_path))
        print()

    threads = sessions["MobileNetV2"].get_session_options().intra_op_num_threads
    print(
        f"Env: Python {platform.python_version()}, onnxruntime {ort.__version__}, numpy {np.__version__}, "
        f"{platform.processor()}/{platform.machine()}, ORT intra_op_num_threads={threads} (0 = default ORT)\n"
    )
    print(f"{'Summary (median ms)':<22}{'prep':>9}{'model':>10}{'total':>10}")
    for (name, _, _), (prep, model, total) in zip(CASES, summary):
        print(f"    {name:<18}{prep:9.3f}{model:10.3f}{total:10.3f}")


if __name__ == "__main__":
    main()
