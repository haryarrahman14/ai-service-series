# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# AI series, part 2: load test for the three servers. Starts one server at a time, sends it the same
# cases with oha, and records throughput, latency, errors, memory and thread count.
#   uv run bench.py results/run1          one full pass -> results/run1.txt and results/run1.json
#   uv run bench.py --summary results     medians over all results/run*.json -> results/summary.md
# Options for a quick look: --only python-default,go-default   --seconds 5
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
PORT = 8077
URL = f"http://127.0.0.1:{PORT}"
IMAGE = HERE / "../shared/cat.jpg"
TIMEOUT = "10s"  # a request that cannot connect, or is not answered, within this time counts as failed
CORES = os.cpu_count()

# name -> (folder, command, extra environment)
SERVERS = {
    "python-default": ("python", ["uv", "run", "main.py"], {}),
    "go-default": ("go", ["./thousandrequests"], {}),
    "rust-default": ("rust", ["target/release/thousand-requests-rust"], {}),
    "python-tuned": ("python", ["uv", "run", "main.py"], {"ORT_THREADS": "1"}),
    "go-tuned": ("go", ["./thousandrequests"], {"ORT_THREADS": "1"}),
    "rust-tuned": ("rust", ["target/release/thousand-requests-rust"], {"ORT_THREADS": "1", "SESSIONS": str(CORES)}),
    "python-tuned-4w": ("python", ["uv", "run", "main.py"], {"ORT_THREADS": "1", "WORKERS": "4"}),
    # each language at the settings the sweep (results/sweep/) found best for it: no failed request in any case, then the
    # most requests answered per second at 150% load. All three work on 20 requests at once and let 100 wait.
    "python-best": ("python", ["uv", "run", "main.py"], {"ORT_THREADS": "1", "WORKERS": "5", "LIMIT": "4", "QUEUE": "20"}),
    "go-best": ("go", ["./thousandrequests"], {"ORT_THREADS": "1", "LIMIT": "20", "QUEUE": "100"}),
    "rust-best": ("rust", ["target/release/thousand-requests-rust"],
                  {"ORT_THREADS": "1", "SESSIONS": "20", "LIMIT": "20", "QUEUE": "100"}),
    # the same limit in all three: work on one request per core, let 100 wait, answer 503 to the rest
    "python-limited": ("python", ["uv", "run", "main.py"], {"ORT_THREADS": "1", "LIMIT": str(CORES), "QUEUE": "100"}),
    # Python's usual answer to its interpreter lock: several processes. Five of them, 2 working + 20 waiting each = 10 + 100.
    "python-limited-5w": ("python", ["uv", "run", "main.py"], {"ORT_THREADS": "1", "WORKERS": "5", "LIMIT": "2", "QUEUE": "20"}),
    "go-limited": ("go", ["./thousandrequests"], {"ORT_THREADS": "1", "LIMIT": str(CORES), "QUEUE": "100"}),
    "rust-limited": ("rust", ["target/release/thousand-requests-rust"],
                     {"ORT_THREADS": "1", "SESSIONS": str(CORES), "LIMIT": str(CORES), "QUEUE": "100"}),
}
ONLY_CASES = None  # the sweep runs a short list of cases


def sweep_settings():
    """Candidate settings per language, to find each one's best (bench.py --sweep). The queue allows 100 waiting in all."""
    py, go, rs = ["uv", "run", "main.py"], ["./thousandrequests"], ["target/release/thousand-requests-rust"]
    found = {}
    for workers in (1, 2, 5, 10):  # Python: how many processes, and how many requests at once in total
        for total in (10, 20):
            found[f"python-w{workers}-l{total}"] = ("python", py, {
                "ORT_THREADS": "1", "WORKERS": str(workers), "LIMIT": str(total // workers), "QUEUE": str(100 // workers)})
    for limit, threads in ((8, 1), (10, 1), (12, 1), (16, 1), (20, 1), (5, 2)):  # requests at once x ONNX Runtime threads each
        env = {"ORT_THREADS": str(threads), "LIMIT": str(limit), "QUEUE": "100"}
        found[f"go-l{limit}-t{threads}"] = ("go", go, env)
        found[f"rust-l{limit}-t{threads}"] = ("rust", rs, {**env, "SESSIONS": str(limit)})
    for limit in (8, 10, 12, 16, 20, 32, 48):  # Rust with one session shared by every request (SHARE_SESSION=1)
        found[f"rust-shared-l{limit}"] = ("rust", rs, {"ORT_THREADS": "1", "LIMIT": str(limit), "QUEUE": "100", "SHARE_SESSION": "1"})
    found["python-w5-l40"] = ("python", py, {"ORT_THREADS": "1", "WORKERS": "5", "LIMIT": "8", "QUEUE": "20"})
    for limit in (32, 48):
        found[f"go-l{limit}-t1"] = ("go", go, {"ORT_THREADS": "1", "LIMIT": str(limit), "QUEUE": "100"})
    found["go-l10-t1-gogc400"] = ("go", go, {"ORT_THREADS": "1", "LIMIT": "10", "QUEUE": "100", "GOGC": "400"})  # collect garbage less often
    return found


CLASSIFY = ["-m", "POST", "-D", str(IMAGE), "-T", "image/jpeg", f"{URL}/classify"]


def cases(seconds, capacity):
    """name, oha arguments. `capacity` is the best requests/sec seen so far; the rate cases are built from it."""
    z = ["-z", f"{seconds}s"]
    if capacity is None:
        return [
            ("1 connection", ["-c", "1", *z, *CLASSIFY]),
            ("8 connections", ["-c", "8", *z, *CLASSIFY]),
            ("64 connections", ["-c", "64", *z, *CLASSIFY]),
            ("1000 connections", ["-c", "1000", *z, *CLASSIFY]),
            ("1000 at once", ["-c", "1000", "-n", "1000", *CLASSIFY]),
        ]
    rate = lambda pct: ["-c", "1000", "-q", str(round(capacity * pct / 100)), "--latency-correction", *z, *CLASSIFY]
    return [
        ("rate 50%", rate(50)),
        ("rate 100%", rate(100)),
        ("rate 150%", rate(150)),
        ("ping, 64 connections", ["-c", "64", "-z", f"{max(seconds // 2, 3)}s", f"{URL}/ping"]),
    ]


def ping(timeout=1.0):
    try:
        with urllib.request.urlopen(f"{URL}/ping", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def tree(pid):
    """pid plus all its descendants (uv starts python, uvicorn starts worker processes), with their RSS in KB."""
    out = subprocess.run(["ps", "-axo", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    rows = [tuple(map(int, line.split())) for line in out.splitlines() if line.strip()]
    pids, grew = {pid}, True
    while grew:
        more = {p for p, parent, _ in rows if parent in pids} - pids
        grew = bool(more)
        pids |= more
    return {p: rss for p, _, rss in rows if p in pids}


def threads(pids):
    out = subprocess.run(["ps", "-M", "-p", ",".join(map(str, pids))], capture_output=True, text=True).stdout
    return max(len(out.splitlines()) - 1, 0)


class Sampler(threading.Thread):
    """Peak memory (sum of RSS over the server's process tree) and peak OS thread count, sampled 4 times a second."""

    def __init__(self, pid):
        super().__init__(daemon=True)
        self.pid, self.peak_rss_mb, self.peak_threads, self.stop = pid, 0.0, 0, threading.Event()

    def run(self):
        while not self.stop.is_set():
            procs = tree(self.pid)
            if procs:
                self.peak_rss_mb = max(self.peak_rss_mb, sum(procs.values()) / 1024)
                self.peak_threads = max(self.peak_threads, threads(procs))
            self.stop.wait(0.25)


def run_case(pid, args):
    sampler = Sampler(pid)
    sampler.start()
    out = subprocess.run(
        ["oha", "--no-tui", "--output-format", "json", "--stats-success-breakdown", "-t", TIMEOUT, "--connect-timeout", TIMEOUT, *args], capture_output=True, text=True
    )
    sampler.stop.set()
    sampler.join()
    raw = json.loads(out.stdout)
    ok = raw["statusCodeDistribution"].get("200", 0)
    # oha lists requests it cut off itself when the test duration ended; they are not server failures.
    errors = {k: v for k, v in raw["errorDistribution"].items() if "deadline" not in k}
    rejected = raw["statusCodeDistribution"].get("503", 0)  # "busy": the server said no at once; not a failure
    other = {k: v for k, v in raw["statusCodeDistribution"].items() if k not in ("200", "503")}
    pct = raw["latencyPercentilesSuccessful"]  # latency of the answered (200) requests only
    ms = lambda v: None if v is None else round(v * 1000, 2)
    return {
        "seconds": round(raw["summary"]["total"], 2),
        "ok": ok,
        "rejected": rejected,
        "failed": sum(errors.values()) + sum(other.values()),
        "failures": {**errors, **{f"HTTP {k}": v for k, v in other.items()}},
        "rps": round(ok / raw["summary"]["total"], 1),
        "p50_ms": ms(pct.get("p50")),
        "p95_ms": ms(pct.get("p95")),
        "p99_ms": ms(pct.get("p99")),
        "max_ms": ms(pct.get("p99.99")),
        "connect_max_ms": ms(raw["details"]["DNSDialup"].get("slowest")),  # slowest time to open a connection
        "peak_rss_mb": round(sampler.peak_rss_mb, 1),
        "peak_threads": sampler.peak_threads,
    }


def settle(proc):
    """Wait until the server answers /ping again (it may still be working through a queue), then pause."""
    start = time.monotonic()
    while time.monotonic() - start < 60:
        if proc.poll() is not None:
            return False
        if ping():
            time.sleep(2)
            return True
        time.sleep(0.2)
    return False


def run_server(name, seconds, log):
    folder, command, env = SERVERS[name]
    assert not ping(), f"something else is already listening on port {PORT}"
    started = time.monotonic()
    proc = subprocess.Popen(
        command, cwd=HERE / folder, env={**os.environ, "PORT": str(PORT), **env},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
    )
    result = {"cases": {}}
    try:
        while not ping(0.2):
            assert proc.poll() is None, f"{name} exited during startup:\n{proc.stdout.read()}"
            assert time.monotonic() - started < 120, f"{name} did not start within 120 s"
            time.sleep(0.02)
        result["startup_s"] = round(time.monotonic() - started, 2)
        time.sleep(1)
        idle = tree(proc.pid)
        result["idle_rss_mb"] = round(sum(idle.values()) / 1024, 1)
        result["idle_threads"] = threads(idle)
        log(f"\n== {name} ==  settings: {env or 'defaults'}")
        log(f"   ready in {result['startup_s']} s, idle: {result['idle_rss_mb']} MB, {result['idle_threads']} threads")
        log(f"   {'case':<22}{'ok':>7}{'busy':>8}{'failed':>7}{'req/s':>9}{'p50 ms':>10}{'p95 ms':>10}{'p99 ms':>10}{'max ms':>10}{'conn ms':>10}{'MB':>8}{'threads':>8}")
        capacity = None
        for phase in range(2):
            for case, args in cases(seconds, capacity):
                if ONLY_CASES and case not in ONLY_CASES:
                    continue
                r = run_case(proc.pid, args)
                r["oha"] = " ".join(a.replace(str(IMAGE), "../shared/cat.jpg") for a in args)
                result["cases"][case] = r
                log(
                    f"   {case:<22}{r['ok']:>7}{r['rejected']:>8}{r['failed']:>7}{r['rps']:>9}{r['p50_ms']!s:>10}{r['p95_ms']!s:>10}"
                    f"{r['p99_ms']!s:>10}{r['max_ms']!s:>10}{r['connect_max_ms']!s:>10}{r['peak_rss_mb']:>8}{r['peak_threads']:>8}"
                    + (f"   {r['failures']}" if r["failures"] else "")
                )
                if not settle(proc):
                    log("   server stopped answering after this case; skipping the rest")
                    result["died_after"] = case
                    return result
            if phase == 0:  # "capacity" = the best throughput of the timed closed-loop cases above
                capacity = max(r["rps"] for c, r in result["cases"].items() if c.endswith(("connection", "connections")))
                result["capacity_rps"] = capacity
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        tail = proc.stdout.read().strip()
        result["server_output"] = tail[-2000:]
        time.sleep(2)
    return result


def summary(folder):
    runs = [json.loads(p.read_text()) for p in sorted(Path(folder).glob("run*.json"))]
    assert runs, f"no run*.json in {folder}"
    fields = ["ok", "rejected", "failed", "rps", "p50_ms", "p95_ms", "p99_ms", "max_ms", "connect_max_ms", "peak_rss_mb", "peak_threads"]
    lines = [f"# Summary: median of {len(runs)} runs", "", "`min..max` after a value is its range across the runs.", ""]
    for name in SERVERS:
        got = [r[name] for r in runs if name in r]
        if not got:
            continue
        med = lambda key: statistics.median(g[key] for g in got)
        lines += [f"## {name}", "", f"Ready in {med('startup_s')} s; idle {med('idle_rss_mb')} MB, {med('idle_threads')} threads; "
                  f"capacity {med('capacity_rps')} req/s.", "", "| case | " + " | ".join(fields) + " |", "|---|" + "---:|" * len(fields)]
        for case in got[0]["cases"]:
            cells = []
            for f in fields:
                values = [g["cases"][case].get(f) for g in got if case in g["cases"] and g["cases"][case].get(f) is not None]
                if not values:
                    cells.append("")
                    continue
                m = statistics.median(values)
                spread = f" ({min(values):g}..{max(values):g})" if f in ("rps", "p50_ms", "p99_ms", "failed", "rejected") and len(values) > 1 else ""
                cells.append(f"{m:g}{spread}")
            lines.append(f"| {case} | " + " | ".join(cells) + " |")
        lines.append("")
    Path(folder, "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def pick(folder):
    """The selection rule: no failed request in any case of any pass, then the highest median answered/s at 150% load.
    Groups: python-*, go-*, rust-shared-* and rust-l* (one session per request at once)."""
    runs = [json.loads(p.read_text()) for p in sorted(Path(folder).glob("run*.json"))]
    best = {}
    for name in runs[0]:
        group = "rust-shared" if name.startswith("rust-shared") else name.split("-")[0] if not name.startswith("rust") else "rust-sessions"
        if sum(c["failed"] for r in runs for c in r[name]["cases"].values()):
            continue
        score = statistics.median(r[name]["cases"]["rate 150%"]["rps"] for r in runs)
        if score > best.get(group, ("", 0))[1]:
            best[group] = (name, score)
    print(",".join(name for name, _ in best.values()))


def main():
    args = sys.argv[1:]
    if args[:1] == ["--summary"]:
        return summary(args[1])
    if args[:1] == ["--pick"]:
        return pick(args[1])
    out = Path(args[0])
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    assert "AC Power" in power, "on battery: the numbers would not be comparable with the runs made on AC power"
    only = args[args.index("--only") + 1].split(",") if "--only" in args else list(SERVERS)
    seconds = int(args[args.index("--seconds") + 1]) if "--seconds" in args else 20
    chosen = args[args.index("--only") + 1].split(",") if "--only" in args else None
    if "--sweep" in args:  # uv run bench.py results/sweep/run1 --sweep --seconds 10
        global ONLY_CASES
        if "--all-cases" not in args:  # --all-cases: the full list of cases, for candidates chosen from a sweep
            ONLY_CASES = {"64 connections", "1000 connections", "1000 at once", "rate 150%"}
        SERVERS.update(sweep_settings())
        only = chosen or list(sweep_settings())  # --only narrows the sweep to some candidates
    out.parent.mkdir(parents=True, exist_ok=True)
    text = open(f"{out}.txt", "w")

    def log(line):
        print(line, flush=True)
        text.write(line + "\n")
        text.flush()

    version = subprocess.run(["oha", "--version"], capture_output=True, text=True).stdout.strip()
    log(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {version}, {seconds} s per timed case, client timeout {TIMEOUT}, {CORES} CPU cores")
    log("POST /classify with shared/cat.jpg (35 KB). busy = answered 503 at once (queue full), latency = answered requests only, conn ms = slowest connection setup, MB = peak memory (RSS, whole server), threads = peak OS threads.")
    results = {}
    for name in only:
        results[name] = run_server(name, seconds, log)
        Path(f"{out}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
