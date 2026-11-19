// AI series, part 2, Go version: the classifier from part 1 as an HTTP service.
// Run from this folder (02-thousand-requests/go/): go build -o thousandrequests && ./thousandrequests
// Settings (environment variables): PORT (default 8000), ORT_THREADS (0 = ONNX Runtime's default),
// LIMIT (requests worked on at once, 0 = no limit of my own), QUEUE (requests allowed to wait when LIMIT is set).
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"image"
	_ "image/jpeg" // register the standard JPEG and PNG decoders for image.Decode
	_ "image/png"
	"io"
	"math"
	"net/http"
	"os"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
	"golang.org/x/image/draw"
)

const shared = "../../shared"

var (
	mean       = [3]float32{0.485, 0.456, 0.406}
	std        = [3]float32{0.229, 0.224, 0.225}
	catClasses = []string{"tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat"}
	port       = envInt("PORT", 8000)
	ortThreads = envInt("ORT_THREADS", 0)
	limit      = envInt("LIMIT", 0) // requests worked on at once, 0 = no limit of my own
	queue      = envInt("QUEUE", 0) // requests allowed to wait when LIMIT is set; one more gets 503
	slots      = make(chan struct{}, max(limit, 1))
	inside     atomic.Int64 // requests working or waiting

	session *ort.Session // session and labels, loaded once, shared by every request goroutine
	labels  []string
)

func decode(body []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(body))
	return img, err
}

func resize(img image.Image) *image.RGBA {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	scale := 256 / float64(min(w, h))
	w, h = int(math.RoundToEven(float64(w)*scale)), int(math.RoundToEven(float64(h)*scale))
	resized := image.NewRGBA(image.Rect(0, 0, w, h))
	draw.BiLinear.Scale(resized, resized.Bounds(), img, img.Bounds(), draw.Src, nil)
	left, top := int(math.RoundToEven(float64(w-224)/2)), int(math.RoundToEven(float64(h-224)/2))
	return resized.SubImage(image.Rect(left, top, left+224, top+224)).(*image.RGBA)
}

func normalize(img *image.RGBA) []float32 {
	b := img.Bounds()
	x := make([]float32, 3*224*224) // NCHW
	for y := 0; y < 224; y++ {
		for xx := 0; xx < 224; xx++ {
			p := img.Pix[img.PixOffset(b.Min.X+xx, b.Min.Y+y):]
			for c := 0; c < 3; c++ {
				x[c*224*224+y*224+xx] = (float32(p[c])/255 - mean[c]) / std[c]
			}
		}
	}
	return x
}

func preprocess(body []byte) ([]float32, error) {
	img, err := decode(body)
	if err != nil {
		return nil, err
	}
	return normalize(resize(img)), nil
}

func infer(session *ort.Session, x []float32) []float32 {
	input, err := ort.CreateTensor([]int64{1, 3, 224, 224}, x)
	check(err)
	defer input.Close()
	// Session.Run is safe for concurrent use: every request goroutine calls it on the same session.
	out, err := session.Run(context.Background(), map[string]*ort.Tensor{"input": input}, []string{"logits"})
	check(err)
	defer out["logits"].Close()
	logits, err := ort.TensorData[float32](out["logits"])
	check(err)
	return slices.Clone(logits)
}

func softmax(logits []float32) []float32 {
	m := slices.Max(logits)
	probs := make([]float32, len(logits))
	var sum float32
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v - m)))
		sum += probs[i]
	}
	for i := range probs {
		probs[i] /= sum
	}
	return probs
}

func argmax(v []float32) int {
	best := 0
	for i := range v {
		if v[i] > v[best] {
			best = i
		}
	}
	return best
}

func classify(body []byte) (string, float64, error) {
	x, err := preprocess(body)
	if err != nil {
		return "", 0, err
	}
	probs := softmax(infer(session, x))
	best := argmax(probs)
	return labels[best], float64(probs[best]), nil
}

func load() {
	ort.SetSharedLibraryPath(shared + "/onnxruntime/lib")
	check(ort.Init())
	options, err := ort.NewSessionOptions() // default options, CPU
	check(err)
	defer options.Close()
	if ortThreads > 0 {
		check(options.SetIntraOpNumThreads(ortThreads))
	}
	session, err = ort.NewSession(shared+"/model/mobilenet_v2.onnx", options)
	check(err)
	labelsText, err := os.ReadFile(shared + "/model/labels.txt")
	check(err)
	labels = strings.Split(strings.TrimRight(string(labelsText), "\n"), "\n")
	body, err := os.ReadFile(shared + "/cat.jpg")
	check(err)
	var label string
	for range 10 { // warm-up, and a check that the answer is still a cat
		label, _, err = classify(body)
		check(err)
	}
	if !slices.Contains(catClasses, strings.ToLower(label)) {
		check(fmt.Errorf("top-1 is not a cat: %s", label))
	}
}

// net/http runs every request in its own goroutine, so the blocking work simply runs here, with no limit.
func handleClassify(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "cannot read body", http.StatusBadRequest)
		return
	}
	if limit > 0 {
		if inside.Add(1) > int64(limit+queue) { // the queue is full: say so at once instead of letting it wait
			inside.Add(-1)
			http.Error(w, "busy", http.StatusServiceUnavailable)
			return
		}
		defer inside.Add(-1)
		slots <- struct{}{} // at most `limit` requests get past this line; the rest wait in order
		defer func() { <-slots }()
	}
	label, prob, err := classify(body)
	if err != nil { // image.Decode could not read the image
		http.Error(w, "cannot decode image", http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"label": label, "prob": prob})
}

func handlePing(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]bool{"ok": true})
}

func main() {
	load()
	ortVersion, err := ort.GetVersion()
	check(err)
	fmt.Printf("Go %s, net/http, onnxruntime %s, ORT_THREADS=%d (0 = default), LIMIT=%d, QUEUE=%d, listening on :%d\n",
		strings.TrimPrefix(runtime.Version(), "go"), ortVersion, ortThreads, limit, queue, port)
	http.HandleFunc("POST /classify", handleClassify)
	http.HandleFunc("GET /ping", handlePing)
	check(http.ListenAndServe(fmt.Sprintf("127.0.0.1:%d", port), nil))
}

// Helpers Go needs: read an integer setting from the environment, and stop the program on an error.
func envInt(name string, fallback int) int {
	s := os.Getenv(name)
	if s == "" {
		return fallback
	}
	n, err := strconv.Atoi(s)
	check(err)
	return n
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
