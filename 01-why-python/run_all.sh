#!/bin/sh
# Run the same request in Python, Go and Rust, one after another, on the same machine.
# Once beforehand, from ai-series/: uv run shared/export_model.py && sh shared/get_onnxruntime.sh
set -e
cd "$(dirname "$0")"
# Close other heavy work first: a busy machine inflates the numbers and makes them unusable.
echo "===== Pure Python vs NumPy ====="; (cd python && uv run matmul.py)
echo; echo "===== Python ====="; (cd python && uv run main.py)
echo; echo "===== Go ====="; (cd go && go build -o whypython && ./whypython)
echo; echo "===== Rust ====="; (cd rust && cargo run --release --quiet)
