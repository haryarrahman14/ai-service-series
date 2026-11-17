// Cross-check of the [2] timings with Go's standard tooling:
// go test -bench . -benchmem -run '^$'
package main

import (
	"os"
	"testing"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
)

const imagePath = shared + "/cat.png"

func TestMain(m *testing.M) {
	ort.SetSharedLibraryPath(shared + "/onnxruntime/lib")
	check(ort.Init())
	os.Exit(m.Run())
}

func BenchmarkDecode(b *testing.B) {
	for b.Loop() {
		decode(imagePath)
	}
}

func BenchmarkDecodeJPEG(b *testing.B) {
	for b.Loop() {
		decode(shared + "/cat.jpg")
	}
}

func BenchmarkDecodeLargeJPEG(b *testing.B) {
	for b.Loop() {
		decode(shared + "/cat-large.jpg")
	}
}

func BenchmarkResize(b *testing.B) {
	img := decode(imagePath)
	for b.Loop() {
		resize(img)
	}
}

func BenchmarkNormalize(b *testing.B) {
	img := resize(decode(imagePath))
	for b.Loop() {
		normalize(img)
	}
}

func BenchmarkPreprocess(b *testing.B) {
	for b.Loop() {
		preprocess(imagePath)
	}
}

func BenchmarkInfer(b *testing.B) {
	session, err := ort.NewSession(shared+"/model/mobilenet_v2.onnx", nil)
	check(err)
	x := preprocess(imagePath)
	for b.Loop() {
		infer(session, x)
	}
}
