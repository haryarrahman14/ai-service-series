#!/bin/sh
# Build the three servers, then load-test them one at a time on this machine.
# Once beforehand, from ai-series/: uv run shared/export_model.py && sh shared/get_onnxruntime.sh && brew install oha
#   sh run_all.sh          one pass   -> results/run1.txt, results/run1.json
#   sh run_all.sh 5        five passes -> results/run1..5, then results/summary.md (medians)
set -e
cd "$(dirname "$0")"
# Close other heavy work first: the load tool and the server share this machine's CPU, so anything else skews the numbers.
(cd go && go build -o thousandrequests)
(cd rust && cargo build --release --quiet)
(cd python && uv sync --script main.py --quiet)
for i in $(seq 1 "${1:-1}"); do
  caffeinate -i uv run bench.py "results/run$i"
done
uv run bench.py --summary results
