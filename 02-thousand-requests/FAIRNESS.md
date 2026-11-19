# Is the comparison fair?

Each line is one finding; its footnote gives the evidence. The measurements behind them, and the earlier attempts, are in the [full audit](FAIRNESS-full.md).

## Verdict

- The three servers do the same work per request and give the same answer.[^work]
- "Fair" here means each language at its own best settings, chosen by one rule, not identical numbers for all.[^rule]
- No request failed in any test, for any of the four settings.[^zero]
- Memory differences are solid. Throughput differences are not, and are not ranked.[^noise]

## Identical in all three

- Same photo, same model file, same ONNX Runtime 1.32.0 on CPU, one thread per request.[^ort]
- Same three rules: a limit on requests worked on at once, a queue of 100, `503 busy` beyond that.[^rules]
- Same load, same failure rule (no connection or no answer within 10 seconds), same harness.[^harness]
- Same selection rule for the settings: no failed request, then the most requests answered per second at 150% load.[^rule]

## What differs, and who it favours

| Difference | Favours | Size |
|---|---|---|
| Rust's binding gives one session to one request at a time; Python's and Go's let all requests share one[^session] | Python, Go | Rust by the book loads 20 copies of the model: 367 MB idle against 84 MB with one |
| "Rust, one shared copy" uses one `unsafe` line that its library forbids[^unsafe] | Rust | it becomes the lightest of all, at a risk this test cannot rule out |
| Python needs five processes to stop failing; each has its own limit and queue[^python] | Go, Rust | 852 MB idle; a full process says "busy" while another is idle |
| Rust drops a waiting request when its client hangs up; Python and Go keep it in the queue[^hangup] | nobody here | waits stayed under 5 seconds against a 10-second timeout |
| Rust by the book was searched up to 20 at once, the others up to 48[^range] | unknown | not measured |
| The image code is part 1's, untuned[^image] | Python, Rust | Go's decode and resize are slower, as in part 1 |

## About the measurements

- The load tool runs on the same 10 cores as the server.[^tool]
- The final set is the only one known to be on AC power with Low Power Mode off; sets are never compared with each other.[^power]
- Two sets were thrown away and kept: one made while the lid was closed, one made on battery.[^discarded]
- In the retry test the tool itself sends tens of thousands of requests per second, so its "answered per second" is shaped by the tool as much as by the servers.[^retry]

## Not established

- Whether sharing one session in Rust is safe beyond these runs.[^unsafe]
- Why Go takes longer than the others to get through a thousand simultaneous requests.[^burst]
- Whether a limit raises or lowers the ceiling compared with no limit.[^ceiling]

[^work]: `POST /classify` with the 35 KB `shared/cat.jpg`: decode, resize, normalize, MobileNetV2, softmax, argmax, JSON. Answers: `Egyptian cat` with probability 0.2305 (Python), 0.2230 (Go), 0.2265 (Rust); the decoders differ in the second decimal (part 1's audit).
[^rule]: `bench.py --sweep` then `bench.py --pick`: candidates with any failed request in any case of either pass are dropped, then the highest median of answered per second at 150% load wins. Results: `results/sweep/` (23 candidates) and `results/sweep2/` (13).
[^zero]: `results/final/run1..5`: `failed` is 0 in every case of every pass for `python-w5-l40`, `go-l32-t1`, `rust-l20-t1` and `rust-shared-l10`.
[^noise]: Answered per second at 150% load across the five passes: 320 to 491 (Python), 293 to 465 (Go), 360 to 545 (Rust by the book), 353 to 523 (Rust shared). Idle memory: 852, 99, 367, 84 MB, within a few MB per pass.
[^ort]: `ORT_THREADS=1` in all final settings; two threads per request with half as many requests at once was clearly worse in the first search (264 to 318 answered per second).
[^rules]: Python: `asyncio.Semaphore` plus a counter on the event loop. Go: a buffered channel plus `atomic.Int64`. Rust: `tokio::sync::Semaphore` plus `AtomicUsize`. All three serve waiters in arrival order.
[^harness]: `bench.py` with oha 1.17.0; `--stats-success-breakdown`, so latency covers answered requests only; client and connect timeout 10 s.
[^session]: `ort` 2.0.0-rc.15: `Session::run` takes `&mut self`. Idle memory by sessions: 84 MB (1), 220 MB (10), 367 MB (20); model file 13.6 MB. `with_prepacked_weights` shared across 20 sessions did not help: 371 MB against 374 MB.
[^unsafe]: `SharedSession` in `rust/src/main.rs`, enabled by `SHARE_SESSION=1`. The maintainer's reasons for forbidding it: [pykeio/ort#402](https://github.com/pykeio/ort/issues/402#issuecomment-2949993914), [#511](https://github.com/pykeio/ort/issues/511). Check made: 200 parallel requests, 200 identical answers. CPU provider only.
[^python]: First search: one process failed 402 and 414 connections in the 10-second retry case (20 at once); two processes up to 128; five and ten none. In `results/best/` one pass had 59,213 "busy" answers at 64 connections because connections landed unevenly on the processes.
[^hangup]: axum drops the handler future when the client disconnects; the Rust server counts and releases the slot through a guard. FastAPI and `net/http` run the handler to the end.
[^range]: `rust-l20-t1` was the top of the range tried in the first search and was carried into the final set unchanged.
[^image]: Part 1: Go's image step is 2 to 3 times slower than Python's and Rust's on these inputs; nothing in the server settings changes that.
[^tool]: In the `/classify` cases oha sends 35 KB and reads about 50 bytes per request; in the `ping` and retry cases its own CPU use is a large part of the picture.
[^power]: Low Power Mode was found on (also for AC) at 14:10 on 2026-11-19; when it was switched on is unknown. One request alone: 14.3 to 14.6 ms in the final set, 16.4 to 16.6 ms in the set made that morning.
[^discarded]: `results/discarded/run4-lid-closed.*` (macOS slept at 20:03 on 2026-11-18) and `results/discarded/sweep2-on-battery/`. The harness now refuses to start on battery.
[^retry]: Turned away per second: 21,900 (Python), 17,500 (Go), 41,400 and 52,500 (Rust).
[^burst]: `1000 at once`, Go: 0.8 to 4.1 seconds across the passes, against 0.4 to 0.9 for the others.
[^ceiling]: The sets with and without a limit were made under different, partly unknown power settings.
