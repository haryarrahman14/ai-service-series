#!/bin/sh
# Download the official ONNX Runtime 1.32.0 (macOS arm64) into shared/onnxruntime/. Used by the Go and Rust versions.
set -e
cd "$(dirname "$0")"
mkdir -p onnxruntime
curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.32.0/onnxruntime-osx-arm64-1.32.0.tgz \
  | tar -xz -C onnxruntime --strip-components 1
ls onnxruntime/lib
