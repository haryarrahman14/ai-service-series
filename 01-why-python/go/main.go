// AI series, part 1, Go version: the same request, and where its time goes.
// Run from this folder (01-why-python/go/): go build -o whypython && ./whypython
package main

import (
	"cmp"
	"context"
	"fmt"
	"image"
	_ "image/jpeg" // register the standard JPEG and PNG decoders for image.Decode
	_ "image/png"
	"math"
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"strings"
	"time"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
	"golang.org/x/image/draw"
)

const shared = "../../shared"

var (
	mean       = [3]float32{0.485, 0.456, 0.406}
	std        = [3]float32{0.229, 0.224, 0.225}
	catClasses = []string{"tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat"}
	models     = map[string]string{"MobileNetV2": "mobilenet_v2.onnx", "ResNet50": "resnet50.onnx"}
	cases      = [][3]string{ // name, image, model
		{"small PNG", "cat.png", "MobileNetV2"},
		{"small JPEG", "cat.jpg", "MobileNetV2"},
		{"large JPEG", "cat-large.jpg", "MobileNetV2"}, // cat.png enlarged to 1920 px, not a native large photo
		{"large model", "cat.png", "ResNet50"},
	}
)

func decode(path string) image.Image {
	f, err := os.Open(path)
	check(err)
	defer f.Close()
	img, _, err := image.Decode(f)
	check(err)
	return img
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

func preprocess(path string) []float32 {
	return normalize(resize(decode(path)))
}

func infer(session *ort.Session, x []float32) []float32 {
	input, err := ort.CreateTensor([]int64{1, 3, 224, 224}, x)
	check(err)
	defer input.Close()
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

func median(v []float64) float64 {
	s := slices.Sorted(slices.Values(v))
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// p95: linear interpolation between the two nearest samples, same as np.percentile.
func p95(v []float64) float64 {
	s := slices.Sorted(slices.Values(v))
	r := 0.95 * float64(len(s)-1)
	i := int(r)
	if i+1 == len(s) {
		return s[i]
	}
	return s[i] + (r-float64(i))*(s[i+1]-s[i])
}

func classify(session *ort.Session, labels []string, imagePath string) {
	x := preprocess(imagePath)
	probs := softmax(infer(session, x))
	top := make([]int, len(probs))
	for i := range top {
		top[i] = i
	}
	slices.SortStableFunc(top, func(a, b int) int { return cmp.Compare(probs[b], probs[a]) })
	fmt.Println("[1] Classifying (top-3, ONNX Runtime CPU)")
	for _, i := range top[:3] {
		fmt.Printf("    %-15s %.2f%%\n", labels[i], probs[i]*100)
	}
	var sum, sq float64
	for _, v := range x {
		sum += float64(v)
	}
	mu := sum / float64(len(x))
	for _, v := range x {
		sq += (float64(v) - mu) * (float64(v) - mu)
	}
	fmt.Printf("    input tensor: mean %.4f, std %.4f\n", mu, math.Sqrt(sq/float64(len(x))))
	if !slices.Contains(catClasses, strings.ToLower(labels[top[0]])) {
		fmt.Fprintf(os.Stderr, "top-1 is not a cat: %s\n", labels[top[0]])
		os.Exit(1)
	}
	fmt.Println()
}

func benchmark(session *ort.Session, imagePath string) [3]float64 {
	for range 10 {
		infer(session, preprocess(imagePath))
	}

	names := []string{"decode", "resize + crop", "normalize", "infer (ONNX Runtime)", "postprocess"}
	times, prep, total := make([][]float64, len(names)), []float64{}, []float64{}
	for range 200 {
		t0 := time.Now()
		img := decode(imagePath)
		t1 := time.Now()
		cropped := resize(img)
		t2 := time.Now()
		x := normalize(cropped)
		t3 := time.Now()
		logits := infer(session, x)
		t4 := time.Now()
		_ = argmax(softmax(logits))
		t5 := time.Now()
		t := []time.Time{t0, t1, t2, t3, t4, t5}
		sum := 0.0
		for i := range 5 {
			times[i] = append(times[i], ms(t[i+1].Sub(t[i])))
			sum += times[i][len(times[i])-1]
			if i == 2 {
				prep = append(prep, sum)
			}
		}
		total = append(total, sum)
	}

	fmt.Println("[2] Where one request spends its time (200 requests after 10 warm-ups)")
	fmt.Printf("    %-20s %11s %11s %8s\n", "", "median", "p95", "share")
	for i, name := range names {
		fmt.Printf("    %-20s %8.3f ms %8.3f ms %7.1f%%\n", name, median(times[i]), p95(times[i]), median(times[i])/median(total)*100)
	}
	fmt.Printf("    %-20s %8.3f ms %8.3f ms\n", "total", median(total), p95(total))
	return [3]float64{median(prep), median(times[3]), median(total)}
}

func main() {
	source, err := os.ReadFile("main.go")
	check(err)
	n := 0
	for _, l := range strings.Split(string(source), "\n") {
		if l = strings.TrimSpace(l); l != "" && !strings.HasPrefix(l, "//") {
			n++
		}
	}
	fmt.Printf("main.go: %d lines of code\n\n", n)

	labelsText, err := os.ReadFile(shared + "/model/labels.txt")
	check(err)
	labels := strings.Split(strings.TrimRight(string(labelsText), "\n"), "\n")
	ort.SetSharedLibraryPath(shared + "/onnxruntime/lib")
	check(ort.Init())
	sessions := map[string]*ort.Session{}
	for name, file := range models {
		sessions[name], err = ort.NewSession(shared+"/model/"+file, nil) // nil = default options, CPU
		check(err)
	}

	summary := [][3]float64{}
	for n, c := range cases {
		name, file, model := c[0], c[1], c[2]
		fmt.Printf("== case %d: %s (%s, %s) ==\n\n", n+1, name, file, model)
		imagePath := shared + "/" + file
		classify(sessions[model], labels, imagePath)
		summary = append(summary, benchmark(sessions[model], imagePath))
		fmt.Println()
	}

	ortVersion, err := ort.GetVersion()
	check(err)
	binding := "?"
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, dep := range info.Deps {
			if dep.Path == "github.com/microsoft/onnxruntime/go" {
				binding = dep.Path + " " + dep.Version
			}
		}
	}
	fmt.Printf("Env: Go %s, %s, onnxruntime %s, %s/%s, SessionOptions default (intra_op_num_threads=0 = default ORT)\n\n",
		strings.TrimPrefix(runtime.Version(), "go"), binding, ortVersion, runtime.GOOS, runtime.GOARCH)
	fmt.Printf("%-22s%9s%10s%10s\n", "Summary (median ms)", "prep", "model", "total")
	for n, c := range cases {
		fmt.Printf("    %-18s%9.3f%10.3f%10.3f\n", c[0], summary[n][0], summary[n][1], summary[n][2])
	}
}

// Helpers Go needs: convert time.Duration to milliseconds, and stop the program on an error.
func ms(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
