# Fairness audit: three servers under load

What was checked to make sure `python/main.py`, `go/main.go` and `rust/src/main.rs` do the same work as HTTP services, and where they differ. Part 1's audit (`../01-why-python/FAIRNESS.md`) covers the image and model code, which is unchanged here apart from reading the image from the request body instead of a file.

Machine: Apple M1 Pro (8 performance + 2 efficiency cores), 32 GB, macOS, on AC power, under `caffeinate -i`. Versions: CPython 3.14.6, fastapi 0.163.2, uvicorn 0.54.0 (with uvloop 0.23.0 and httptools 0.9.0), starlette 1.10.0, anyio 4.16.0, onnxruntime 1.32.0, numpy 2.5.5, Pillow 13.0.0; Go 1.27.4 with `net/http`, the ONNX Runtime Go binding at the v1.32.0 commit, `golang.org/x/image` 0.48.0; Rust 1.100.0, axum 0.8.9, tokio 1.54.1, ort 2.0.0-rc.15, image 0.25.10; oha 1.17.0.

## Identical in all three

| | Python | Go | Rust |
|---|---|---|---|
| Request | `POST /classify`, body = the 35 KB `shared/cat.jpg` | same | same |
| Work per request | decode, resize, normalize, MobileNetV2, softmax, argmax, JSON answer | same | same |
| Answer | `Egyptian cat`, prob 0.2305 | `Egyptian cat`, prob 0.2230 | `Egyptian cat`, prob 0.2265 |
| ONNX Runtime | 1.32.0, CPU, default graph optimisation | same | same |
| Start | model loaded and 10 warm-up requests before the port opens | same | same |
| HTTP | HTTP/1.1 with keep-alive, no TLS, `127.0.0.1` | same | same |
| Logging per request | none (uvicorn's access log is switched off) | none | none |
| Limits and timeouts written by me | none | none | none |

The probabilities differ in the second decimal because the three JPEG decoders differ (part 1's audit, "Why JPEG pixels differ").

## The frameworks, and why these

Each is what a developer in that language reaches for first: FastAPI on uvicorn (installed as `uvicorn[standard]`, so uvicorn picks uvloop and httptools by itself), Go's standard `net/http`, and axum on tokio's default multi-thread runtime. Nothing is tuned in the `default` settings.

## Differences that remain

- **How many requests run the image and model code at the same time.** This is the largest difference, and it comes from each stack's defaults, not from a setting I chose.
  - Python: FastAPI runs blocking work on a thread pool of 40 threads (anyio's default limiter, read back at runtime: 40). A 41st request waits in the event loop.
  - Go: `net/http` starts a goroutine per request, with no limit. A thousand connections means a thousand requests decoding and calling the model together.
  - Rust: `spawn_blocking` uses tokio's blocking pool, up to 512 threads (`max_blocking_threads: 512` in tokio's `runtime/builder.rs`).
- **How the model session is shared.** Python and Go call `run` on one session from many threads; both bindings document that as safe. In `ort` 2.0.0-rc.15 every `run` method takes `&mut self`, so one session serves one request at a time. `ort`'s own comment (`src/session/mod.rs`) says this is because parts of ONNX Runtime are not thread safe, and recommends one session per thread. So the Rust server keeps a pool of sessions: 1 in `rust-default` (the model runs one request at a time, using all cores), 10 in `rust-tuned` (one per core, one thread each). *Judgement: a real property of the Rust binding today. `rust-default` is what a first version looks like; `rust-tuned` follows the binding's recommendation.*
- **Which waiting request gets the free session in Rust.** The pool wakes waiters with a `Condvar`, which does not promise arrival order.
- **Python's extra setting.** `python-tuned-4w` runs four server processes, each with its own model, event loop and 40 threads. Go and Rust have no such setting because one process already uses all cores.
- **ONNX Runtime threads.** In the `default` settings every `run` call may use all cores, and concurrent calls share the same thread pool. In the `tuned` settings each `run` uses one thread (`ORT_THREADS=1`), so parallelism comes from running several requests at once.
- **Connection queue.** uvicorn asks the system for a queue of 2,048 waiting connections; Go reads `kern.ipc.somaxconn`; tokio (through mio) asks for 128. macOS caps all of them at `kern.ipc.somaxconn`, which is 128 on this machine, so the three are equal here. It does mean that when 1,000 connections arrive together, how quickly the server accepts them matters.

## About the load test itself

- **The load tool runs on the same machine.** oha and the server compete for the same 10 cores. In the `/classify` cases the server does nearly all the work (oha only sends 35 KB and reads 50 bytes). In the `ping` case oha's own cost is a large part of the result, so read those numbers as a comparison between the three, not as what each server could do alone.
- **One image, one model.** Every request is the same small JPEG. Part 1 showed that Go's image code is slower on other inputs; this test does not repeat that.
- **Order.** The seven settings always run in the same order, one after another, each in a fresh server process.
- **Closed and open load.** In the `connections` cases a connection sends its next request only after the previous answer, so a slow server is automatically sent fewer requests. The `rate` cases send at a fixed rate regardless, and measure latency from the moment a request was due to be sent (`--latency-correction`), so time spent waiting to be sent counts.
- **Failures.** A request fails if it cannot connect, or gets no answer, within 10 seconds. Requests still in flight when a timed case ends are cut off by oha and are not counted as failures.
- **Memory** is RSS summed over the server's process tree (for Python that includes the `uv` launcher, about 21 MB), sampled four times a second; the peak is reported. Thread count is sampled the same way.

## The runs

Five full passes on 2026-11-18, each about 21 minutes: `results/run1.txt` to `run5.txt` (raw oha numbers in the `.json` files), medians and ranges in `results/summary.md`. The article quotes medians and gives the range where it matters.

- **One pass was thrown away and repeated.** During the original fourth pass the laptop lid was closed and macOS put the machine to sleep at 20:03 (`pmset -g log`: "Entering Sleep state due to 'Clamshell Sleep'"); `caffeinate -i` does not prevent that. The pass took 63 minutes instead of 21, and the settings running at that moment collapsed: `rust-default` answered 19 requests per second on one connection instead of about 115, then nothing at all. Its files are kept in `results/discarded/`, and `run4` is the repeat, made with the lid open.
- **Baseline against part 1.** One connection, default settings: 8.69 ms (Python), 11.06 ms (Go), 8.50 ms (Rust) per request over HTTP, against 7.60, 11.32 and 7.66 ms for the same image in part 1 without HTTP. Python and Rust pay about 1 ms for HTTP; Go's figure is slightly lower than in part 1, which fits part 1's note that Go's model step varies between runs.
- **Stable.** Throughput and median latency on 1, 8 and 64 connections vary by a few percent between passes.
- **Not stable.** The number of failed requests on 1,000 connections differs a lot from pass to pass (Python default 0 to 116, Go default 424 to 774 in a 20-second case). The `1000 at once` case is the least stable: Go default answered all 1,000 in four passes and lost 116 in one, and the time to answer them all ranged from 3.3 to 7.8 seconds. `rate 100%` sits exactly at the edge of what each server can do, so its latency swings widely; the article does not draw conclusions from it.

## What the failures are

oha reports two kinds, and both count as failed:

- `Operation timed out (os error 60)`: the connection was not accepted by the server within 10 seconds.
- `timeout`: the connection was open but the answer did not arrive within 10 seconds.

Over the five passes, in the `1000 connections` case: Python default 238 failures, all of the first kind; Go default 2,970 (1,100 of the first kind, 1,870 of the second); Rust default none.

**Slow connection setup only happens while `/classify` work is running.** In the `1000 connections` and `1000 at once` cases the slowest successful connection took about 5.87 seconds to open for Python and Go (the same value in almost every pass, which fits the client retrying a dropped connection attempt on a timer), against 0.06 to 1.07 seconds for Rust. As a check, the same 1,000 simultaneous connections were sent to `/ping`, which does no image or model work (`results/extra-ping-1000-at-once.txt`): all three servers accepted every connection within 0.1 to 0.36 seconds, with no failures.

**Not established.** Why Python's and Go's servers accept connections late under `/classify` load was not measured directly. The likely reason is where the accepting code runs: in Python on the one event-loop thread, which shares the interpreter lock with 40 busy worker threads; in Go as one goroutine among a thousand busy ones on 10 processor slots; in Rust on tokio's async threads, which are separate from the 512 threads doing the blocking work. Likewise, why Go's latency is so uneven on 1,000 connections (median 0.18 s, p99 9.4 s) was not investigated beyond observing that nothing in the Go server puts requests in an order.

## Memory

Go's server grew from 97 MB idle to between 1.7 and 2.1 GB on 1,000 connections, and stayed there for the remaining cases of each pass; whether it returns the memory later was not tested. With no limit, up to a thousand requests hold their decoded image and resize buffers at once (part 1 measured about 5 MB allocated per request in Go's resize). Python peaked at 541 to 566 MB with its 40 worker threads, Rust at 674 to 695 MB with 530 threads.

## The final set (2026-11-19 afternoon): the numbers part 2's article quotes

`results/final/run1.txt` to `run5.txt`, summary in `results/final/summary.md`. Four settings, chosen by `uv run bench.py --pick results/sweep2` from a second search (`results/sweep2/`, 13 candidates, two passes) with the same rule as before: no failed request, then the most requests answered per second at 150% load.

| Setting in the files | What it is |
|---|---|
| `python-w5-l40` | Python, 5 processes, 8 requests at once and 20 waiting per process |
| `go-l32-t1` | Go, 32 at once, 100 waiting |
| `rust-l20-t1` | Rust by the book: 20 at once, one session per request at once (20 sessions) |
| `rust-shared-l10` | Rust with `SHARE_SESSION=1`: 10 at once, every request in the same session |

- **No request failed** in any case of any pass, for all four.
- **`SHARE_SESSION=1` is the one `unsafe` in the Rust server.** It hands `&mut Session` to several threads at once, which `ort` forbids on purpose (`Session::run` takes `&mut self`; the maintainer's reasons are in pykeio/ort issue 402). It was run on the CPU provider only. Check made: 200 requests in parallel returned 200 identical answers. This shows it worked here; it does not show it is safe.
- **These runs are the only ones made with the machine known to be on AC power with Low Power Mode off.** Low Power Mode was found switched on (for AC as well) at 14:10 that day; it is not known when it was switched on, so the earlier sets (`results/run*`, `results/limit*`, `results/best/`, `results/sweep/`) may have been made under it. A single request takes 14.3 to 14.6 ms in the final set against 16.4 to 16.6 ms in `results/best/` the same morning, which fits. Numbers from different sets are therefore not compared with each other in the article.
- **Throughput in the final set is noisy.** Another session was working on the same laptop during part of it. Answered per second at 150% load ranged 320 to 491 (Python), 293 to 465 (Go), 360 to 545 (Rust by the book), 353 to 523 (Rust shared) across the five passes. The article does not rank throughput. Idle memory was stable: 852, 99, 367 and 84 MB.
- **The second search**: `go-l48-t1` had 14 failed requests over its two passes and was excluded; `go-l32-t1` (396, 344) beat `go-l20-t1` (271, 368); `python-w5-l40` (459, 465) beat `python-w5-l20` (412, 413); the seven `rust-shared` candidates from 8 to 48 at once were within each other's spread except 8, and 10 had the highest median (509, 501). Rust by the book was not searched beyond 20 at once. A first attempt at this search ran on battery and is in `results/discarded/sweep2-on-battery/`.

## Each language at its best, first pass (2026-11-19 morning; superseded by the final set above)

Part 2's article compares the three servers each at its own best settings, not at identical ones. Part 3 uses the default settings described above.

**The rules every server follows** (same code path as the "same limit" section below): work on at most `LIMIT` requests at once, let at most `QUEUE` wait in arrival order, answer `503 busy` beyond that. The queue is 100 in total for all three; it is treated as a promise about waiting time, not as something to tune.

**The search** (`uv run bench.py results/sweep/run1 --sweep --seconds 10`, two passes, `results/sweep/`): 23 candidates, each tested with 64 connections, 1,000 connections (clients that retry at once), 1,000 at once, and a fixed rate of 150% of its own ceiling.

- Python: 1, 2, 5 or 10 processes, with 10 or 20 requests at once in total (limit and queue divided evenly between the processes).
- Go and Rust: 8, 10, 12, 16 or 20 requests at once with one ONNX Runtime thread each, and 5 at once with two threads each; Rust always with one session per request at once. Go once more with `GOGC=400`.

**The selection rule**, the same for all three: no failed request in any case in either pass, then the highest median of requests answered per second at 150% load.

| | Chosen | Answered per second at 150% (two passes) | Runner-up |
|---|---|---|---|
| Python | 5 processes, 4 at once and 20 waiting each | 471, 457 | 2 processes with 10 at once each: 382, 460, but 128 failed connections in one pass |
| Go | 20 at once | 418, 420 | 12 at once: 405, 406 |
| Rust | 20 at once, 20 sessions | 486, 475 | 16 at once: 480, 470 |

What the search found, and what it did not settle:

- **Python with one process fails the retry case**: 402 and 414 failed connections in 10 seconds with 20 at once, 404 and 0 with 10 at once. Two processes: 87 and 0, 128 and 0. Five and ten processes: none in either pass. Ten processes answer fewer requests (233 to 349 per second at 150% load) and use about 1.6 GB idle.
- **Go and Rust had no failed request in any candidate.** For both, the differences between 10, 12, 16 and 20 at once are within the spread between the two passes (for example Rust: 428 to 490 at 10, 470 to 480 at 16, 475 to 486 at 20), so the choice of 20 follows the rule but is not a firm ranking.
- **Two ONNX Runtime threads per request with 5 at once** was clearly worse for both (264 to 318 per second). **`GOGC=400`** did not raise Go's throughput (389 to 411) and roughly doubled its memory in that case.
- Not tried: more than 20 at once, Python's free-threaded build, a shared queue in front of several Python processes (a separate load balancer), reusing image buffers in Go, any change to the image code.

**Why Rust uses more memory than Go.** Idle memory of the Rust server by number of sessions: 84 MB with 1 (`rust-default`), 220 MB with 10 (`rust-tuned`), 365 MB with 20 (`rust-best`), so about 15 MB per extra session, against a model file of 13.6 MB. Go idles at 98 MB with its single shared session. A one-off test on 2026-11-19 with `ort`'s `with_prepacked_weights` (one container shared by all 20 sessions) did not reduce it: 371 MB against 374 MB. Sharing the model's weights between sessions another way was not tried. The selection rule looks at failures and throughput only; by memory, 10 sessions (220 MB) would be the better choice for Rust at a throughput that the sweep could not tell apart from 20.

**The runs the article quotes**: `python-best`, `go-best`, `rust-best`, five passes, `results/best/run1.txt` to `run5.txt`, summary in `results/best/summary.md`. No request failed in any case of any pass. In one pass Python told 59,213 requests "busy" in the 64-connection case: the connections had landed unevenly on its five processes, and a process whose own queue is full says "busy" even when the others are idle.

## The same limit in all three (an earlier attempt at fairness, kept for reference)

The settings above leave every framework at its defaults, which answers "what happens if nobody sets a limit" and is unfair as a comparison of the languages: no production server should run without one. So each server got the same three rules, written in the handler in that language's usual way:

- work on at most `LIMIT` requests at once (10, one per core, with one ONNX Runtime thread per request and, for Rust, 10 sessions);
- let at most `QUEUE` more wait, in arrival order (100);
- answer anything beyond that `503 busy` immediately.

| | Python | Go | Rust |
|---|---|---|---|
| Limit | `asyncio.Semaphore(10)` around the thread-pool call | a buffered channel of 10 | `tokio::sync::Semaphore` with 10 permits |
| Count of working + waiting | an integer on the event loop | `atomic.Int64` | `AtomicUsize` |
| Waiting order | first come, first served (asyncio's semaphore) | first come, first served (channel send) | first come, first served (tokio's semaphore is fair) |

Runs: `results/limit/run1.txt` to `run5.txt`, summary in `results/limit/summary.md`. In these files `busy` is the number of 503 answers, `failed` is still "no connection or no answer within 10 seconds", and the latency columns cover the answered (200) requests only.

Differences that remain:

- **A client that hangs up while waiting.** In Rust, axum drops the request and it leaves the queue. In Python and Go the request keeps its place and is still worked on when its turn comes. With a 10-second client timeout and waits under 2 seconds this did not come into play.
- **The closed-loop case becomes a retry storm.** In `1000 connections`, a connection that gets `503` sends its next request at once, so the servers were answering 7,000 to 64,000 rejections per second, and oha was sending them from the same 10 cores. This is a real situation (clients that retry without waiting), but the "answered per second" figures in that case are shaped as much by the load tool's own CPU use as by the servers. The `rate` cases have no retries and are the cleaner test of overload.
- **Python with several processes** (`python-limited-5w`, `results/limit-python-5w/`): five processes, each working on 2 and letting 20 wait, so 10 and 100 in total. The limit is per process: a process whose queue is full answers `503` even when another process is idle, and connections are not spread evenly between processes. That is why this setting rejects requests even at half load (median 677 of about 4,600 at `rate 50%`).
- **Day-to-day drift.** These runs were made a day after the first set. One control pass of the three `tuned` settings on the same morning (`results/limit-control/`) gave ceilings of 434 (Python), 345 (Go) and 479 (Rust) requests per second, against 487, 419 and 469 the day before; the limited runs themselves vary from 285 to 471 between passes. So no conclusion is drawn about whether the limit raises or lowers the ceiling; failures, rejections, waiting time and memory differ by far more than this drift.
