# Kenapa AI service selalu pakai Python? Kenapa bukan Go yang lebih simpel, atau Rust yang lebih kencang?

*Seri AI Service, bagian 1. Semua saya ukur di mesin saya sendiri. Kodenya ada di repo.*

Jawaban singkat: Python memang lambat, tapi di AI service Python hampir tidak mengerjakan apa-apa. Langkah beratnya jalan di library C dan C++, dan library itu sebagus itu sampai Python jadi yang tercepat dari ketiganya untuk foto JPEG. Model juga dibuat di Python.

## Yang saya uji

Satu AI service kecil: satu foto masuk, model menebak isinya.

- **Model** adalah file hasil training. Di sini MobileNetV2, yang mengenali 1.000 jenis objek.
- **Fotonya** foto seekor kucing. Jawaban yang diharapkan: "Egyptian cat".
- **Satu request** punya empat langkah: decode (membaca foto), resize (mengubah ukurannya), normalize (mengubahnya jadi angka), dan menjalankan model.

![Foto kucing melewati decode, resize, normalize, dan model, lalu keluar sebagai jawaban Egyptian cat](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/img/id/1-request.png)

*Satu request: foto masuk, jawaban keluar.*

## Python memang lambat

Model AI pada dasarnya jutaan perkalian angka. Jadi saya ukur satu pekerjaan semacam itu: mengalikan dua tabel angka 200x200, yang butuh 8 juta perkalian. Saya kerjakan dua kali: sekali dengan loop Python biasa, sekali dengan NumPy.

```
pure Python  : 280.952 ms
NumPy (@)    :   0.058 ms
```

NumPy sekitar 5.000 kali lebih cepat, karena pekerjaannya diserahkan ke library yang ditulis dalam C.

## Tapi kerja beratnya bukan dikerjakan Python

Program Python saya, `python/main.py`, cuma 107 baris. Isinya hanya memanggil library:

- **Pillow** (C) membaca foto dan mengubah ukurannya.
- **NumPy** (C) mengubahnya jadi angka.
- **ONNX Runtime** (C++) menjalankan model.

Lebih dari 99% waktu satu request habis di dalam tiga library itu.

![main.py memanggil tiga library, Pillow, NumPy, dan ONNX Runtime, yang mengerjakan semuanya](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/img/id/2-who-does-the-work.png)

*Kode Python memberi perintah. Library C dan C++ yang bekerja.*

## Request yang sama di Go dan Rust

Request yang sama saya tulis di Go dan Rust: file model yang sama, foto yang sama, ONNX Runtime 1.32.0 yang sama, CPU saja.

Satu foto dan satu model itu bukti yang tipis, jadi saya jalankan empat kasus:

- **small PNG**: foto kucing tadi, 451x300, sebagai file PNG.
- **small JPEG**: foto yang sama, disimpan sebagai JPEG.
- **large JPEG**: foto yang sama, diperbesar ke 1920x1277 lalu disimpan sebagai JPEG. Ini hasil perbesaran, bukan foto besar asli.
- **large model**: small PNG lagi, tapi dengan ResNet50, model yang sekitar tujuh kali lebih berat dari MobileNetV2.

Total waktu satu request, median dalam milidetik:

```
                 Python       Go     Rust
small PNG        10.880   14.975    9.465
small JPEG        8.221   10.758    8.531
large JPEG       19.805   53.886   24.816
large model      42.863   46.965   41.402
```

Ketiga bahasa menjawab "Egyptian cat" di keempat kasus.

![Empat kasus, tiga bahasa: bahasa tercepat berganti tergantung kasusnya](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/img/id/6-four-cases.png)

*Pemenangnya tergantung kasus. Go terakhir di keempatnya.*

## Langkah model sama mahalnya di semua bahasa

Waktu untuk menjalankan model, dalam milidetik:

```
                 Python       Go     Rust
small PNG         5.668    5.805    5.628
small JPEG        5.674    5.763    5.643
large JPEG        5.686    5.388    5.323
large model      37.546   37.754   37.506
```

Di tiap kasus, ketiga angkanya hampir sama, karena ketiga bahasa memanggil mesin C++ yang sama.

![Langkah model makan waktu yang sama di Python, Go, dan Rust](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/img/id/3-same-model.png)

*Menjalankan model sama mahalnya di semua bahasa.*

## Bedanya ada di preprocessing gambar

Waktu untuk decode, resize, dan normalize, dalam milidetik:

```
                 Python       Go     Rust
small PNG         5.189    9.106    3.813
small JPEG        2.501    4.984    2.896
large JPEG       14.074   48.544   19.484
large model       5.256    9.133    3.872
```

Tiga hal yang menonjol:

- **Di JPEG, Python paling cepat.** Pillow men-decode JPEG dengan libjpeg-turbo, library C yang dioptimalkan untuk prosesor ini. Di sini itu lebih cepat dari library gambar Rust.
- **Go paling lambat di semua kasus.** Library gambarnya ditulis pakai Go biasa. Di large JPEG itu makan 48,5 ms, lawan 14,1 ms di Python.
- **Dengan model besar, selisihnya makin tidak terasa.** 5 ms yang hilang di Go untuk gambar itu sepertiga dari request kecil, tapi cuma sepersepuluh dari request ResNet50.

![Waktu preprocessing gambar untuk small PNG: Rust 3,81 ms, Python 5,19 ms, Go 9,11 ms](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/img/id/4-image-prep.png)

*Selisih antar bahasa datang dari library gambarnya.*

## Jadi kenapa Python?

Karena model dibuat di sana, dan sebuah model melewati dua tahap.

**Tahap 1: training.** Kita menulis modelnya, memberinya data, lalu komputer menghitung jutaan angka. Ini dikerjakan dengan PyTorch, yang merupakan library Python.

- Model ini ditulis dan di-training dengan PyTorch, di Python. API C++ milik PyTorch masih berlabel beta.
- Hampir setengah repo AI baru di GitHub memakai Python (Octoverse 2025).
- Tidak ada langkah compile, jadi mencoba perubahan itu cepat.

Hasil training adalah satu file. Di sini ukurannya 14,23 MB.

**Tahap 2: service.** Sebuah service membuka file itu, memberinya foto, dan mengembalikan jawabannya. File itu tidak peduli bahasa apa yang membukanya, jadi service-nya bisa ditulis di Python, Go, atau Rust.

![Tahap 1, training di Python, menghasilkan satu file; tahap 2, service di Python, Go, atau Rust, membukanya](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/img/id/5-one-file.png)

*Training terjadi di Python. Service-nya boleh di mana saja dari ketiganya.*

Binding ONNX Runtime belum sama matangnya:

- Python: resmi.
- Go: resmi sejak Agustus 2026, belum punya tag versi.
- Rust: crate komunitas `ort`, yang versi 2.0-nya masih release candidate.

## Cara saya mengukurnya

Kodenya ada di tiga folder, `python/`, `go/`, dan `rust/`. Tiap program punya dua belas fungsi yang sama dengan urutan yang sama: `decode`, `resize`, `normalize`, `preprocess`, `infer`, `softmax`, `argmax`, `median`, `p95`, `classify`, `benchmark`, `main`.

Tiap program mengukur dirinya sendiri dengan cara yang sama: 10 request warm-up, lalu 200 request yang diukur, lalu median dan p95 tiap langkah. Semuanya saya jalankan lima kali; tabel di atas memakai nilai tengah dari lima run itu. Selisih antar run di bawah 2%, kecuali langkah model di kasus large JPEG yang bergeser sampai 7%. Output mentah kelimanya ada di `results/`.

Ini loop yang diukur di Python:

```python
for _ in range(200):
    t0 = time.perf_counter()
    img = decode(image_path)
    t1 = time.perf_counter()
    img = resize(img)
    t2 = time.perf_counter()
    x = normalize(img)
    t3 = time.perf_counter()
    logits = infer(session, x)
    t4 = time.perf_counter()
    argmax(softmax(logits))
    t5 = time.perf_counter()
```

Angka gambar di Go terlihat terlalu lambat, jadi saya cek dengan tool benchmark bawaan Go:

```sh
go test -bench . -benchmem -run '^$' -count 3
```

Hasilnya 6,1 ms untuk decode dan 2,5 ms untuk resize, cocok dengan programnya.

## Apakah perbandingannya setara?

Saya cek bahwa ketiga program mengerjakan hal yang sama:

- Ketiganya memberi model input yang sama. Untuk PNG, tidak ada piksel yang beda lebih dari 1 tingkat dari 255 setelah resize. Untuk JPEG bedanya sedikit lebih besar, sampai 8 tingkat, karena decoder JPEG Go mengolah warna dengan cara yang lebih sederhana. Tiap program mencetak rata-rata dan sebaran inputnya supaya bisa dibandingkan.
- Model, foto, versi ONNX Runtime, setelan CPU, dan setelan thread-nya sama.
- Go memakai cara resize yang setara dengan dua bahasa lain. Ada cara yang lebih cepat di Go, tapi itu algoritma lain dan mengubah gambar sampai 16 tingkat, jadi tidak saya pakai.
- Lambatnya pengolahan gambar di Go itu nyata: membuka data PNG saja butuh 3,39 ms di Go, lawan 0,86 ms di C.
- Python tidak mengambil jalan pintas di large JPEG. Pillow bisa men-decode JPEG dalam ukuran yang diperkecil; saya cek bahwa itu tidak terjadi di sini.

Audit ini menemukan satu hal yang tidak setara, dan itu menguntungkan Python: satu penyalinan data terhitung di langkah model, bukan di normalize. Sudah saya perbaiki; totalnya tidak berubah.

Audit lengkapnya ada di [FAIRNESS.md](https://github.com/haryarrahman14/ai-service-series/blob/main/01-why-python/FAIRNESS.md).

Screenshot dari satu run lagi, diambil setelah lima run tadi:

![Run pure Python vs NumPy](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/proof/matmul.png)

![Run Python](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/proof/python.png)

![Run Go](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/proof/go.png)

![Run Rust](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/proof/rust.png)

![Benchmark decode Go](https://raw.githubusercontent.com/haryarrahman14/ai-service-series/main/01-why-python/proof/go-bench.png)

Untuk menjalankannya sendiri:

```sh
uv run shared/export_model.py
sh shared/get_onnxruntime.sh
sh 01-why-python/run_all.sh
```

Batasan uji ini:

- Yang diukur satu request dalam satu waktu, bukan service di bawah beban. Itu bagian 2.
- Keempat kasus memakai satu foto kucing. Large JPEG adalah perbesaran dari foto itu.
- Mesin yang sibuk merusak angka. Satu run saat ada pekerjaan lain memberi Python 54 ms, dan run itu saya buang.
- Di macOS, binding Rust memakai CoreML milik Apple secara default. Saya paksa ke CPU supaya ketiganya memakai mesin yang sama.

## Kesimpulan

Training model dikerjakan di Python, karena alatnya ada di sana.

Service yang menjalankan model bisa ditulis di mana saja dari ketiganya. Di empat kasus ini langkah modelnya sama mahal di semua bahasa, dan selisihnya datang dari library gambar: Rust paling cepat di PNG, Python paling cepat di JPEG, dan Go paling lambat di semuanya.

Jadi memilih Go atau Rust tidak membuat modelnya lebih cepat. Yang belum terlihat dari uji ini: bagaimana tiap bahasa bekerja saat banyak request datang bersamaan.

## Selanjutnya

Di bagian 2, versi Python dan Go saya jadikan service HTTP lalu saya uji di bawah beban.

## Mesin dan versi

- MacBook Pro, Apple M1 Pro, RAM 32 GB, macOS 26.6.2
- ONNX Runtime 1.32.0, CPU, jumlah thread default
- Python 3.12.13, numpy 2.5.5, pillow 13.0.0
- Go 1.27.4, `github.com/microsoft/onnxruntime/go` di commit v1.32.0, `golang.org/x/image` 0.48.0
- Rust 1.97.1, `ort` 2.0.0-rc.15, `image` 0.25.10

Kode: [github.com/haryarrahman14/ai-service-series](https://github.com/haryarrahman14/ai-service-series)

## Sumber

- [ONNX Runtime: bahasa yang didukung](https://onnxruntime.ai/docs/get-started/)
- [ONNX Runtime: binding Go](https://github.com/microsoft/onnxruntime/pull/29615)
- [ort](https://crates.io/crates/ort)
- [PyTorch C++ API](https://docs.pytorch.org/cppdocs/)
- [NumPy: BLAS, LAPACK](https://numpy.org/doc/stable/building/blas_lapack.html)
- [GitHub Octoverse 2025](https://github.blog/news-insights/octoverse/octoverse-a-new-developer-joins-github-every-second-as-ai-leads-typescript-to-1/)
