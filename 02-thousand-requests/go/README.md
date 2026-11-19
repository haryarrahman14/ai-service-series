# 02 · A thousand requests, Go version

The same HTTP service as `../python/main.py`, rewritten in Go with the same ONNX Runtime (the official binding `github.com/microsoft/onnxruntime/go`, at the v1.32.0 tag commit) and the standard library's `net/http`. The functions have the same names and order as the Python version, so the files can be read side by side. The image code is part 1's, unchanged.

## Requirements

Go 1.27+ and the Xcode Command Line Tools (cgo). Once, from the `ai-series/` folder:

```sh
sh shared/get_onnxruntime.sh
uv run shared/export_model.py
```

## Running

From the `02-thousand-requests/go/` folder (`../../shared` is read relative to it):

```sh
go build -o thousandrequests && ./thousandrequests
```

Settings are environment variables: `PORT` (default 8000), `ORT_THREADS` (0 or unset = ONNX Runtime's default, above 0 = `SetIntraOpNumThreads`), `LIMIT` (requests worked on at once; 0 or unset = no limit of my own) and `QUEUE` (requests allowed to wait when `LIMIT` is set; one more gets `503 busy`). The `libonnxruntime.dylib` path is set in code with `ort.SetSharedLibraryPath`.

```sh
curl -s -X POST --data-binary @../../shared/cat.jpg localhost:8000/classify
# {"label":"Egyptian cat","prob":0.2229907512664795}
curl -s localhost:8000/ping
# {"ok":true}
```

A body that is not a PNG or JPEG gets `400 cannot decode image`.

## How it serves

`load` creates one MobileNetV2 session, classifies `cat.jpg` 10 times as a warm-up and stops if the answer is not a cat. `net/http` then runs every request in its own goroutine, and all of them call `Session.Run` on that one session, which the binding documents as safe for concurrent use. There is no worker pool, no limit on concurrent requests and no custom timeout: this is what Go does by default.

`check` and `envInt` at the end of the file are only helpers Go needs (error check, integer from an environment variable).
