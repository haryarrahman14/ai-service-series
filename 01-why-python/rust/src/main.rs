// AI series, part 1, Rust version: the same request, and where its time goes.
// Run from this folder (01-why-python/rust/): cargo run --release
use std::collections::HashMap;
use std::time::Instant;

use image::RgbImage;
use image::imageops::{self, FilterType};
use ort::session::{Session, builder::AutoDevicePolicy};
use ort::value::Tensor;

const SHARED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../shared");
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const CAT_CLASSES: [&str; 5] = ["tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat"];
const MODELS: [(&str, &str); 2] = [("MobileNetV2", "mobilenet_v2.onnx"), ("ResNet50", "resnet50.onnx")];
const CASES: [(&str, &str, &str); 4] = [
    // name, image, model
    ("small PNG", "cat.png", "MobileNetV2"),
    ("small JPEG", "cat.jpg", "MobileNetV2"),
    ("large JPEG", "cat-large.jpg", "MobileNetV2"), // cat.png enlarged to 1920 px, not a native large photo
    ("large model", "cat.png", "ResNet50"),
];

fn decode(path: &str) -> RgbImage {
    image::open(path).unwrap().to_rgb8()
}

fn resize(img: &RgbImage) -> RgbImage {
    let (w, h) = img.dimensions();
    let scale = 256.0 / w.min(h) as f64;
    let (w, h) = ((w as f64 * scale).round_ties_even() as u32, (h as f64 * scale).round_ties_even() as u32);
    let img = imageops::resize(img, w, h, FilterType::Triangle);
    let left = ((w - 224) as f64 / 2.0).round_ties_even() as u32;
    let top = ((h - 224) as f64 / 2.0).round_ties_even() as u32;
    imageops::crop_imm(&img, left, top, 224, 224).to_image()
}

fn normalize(img: &RgbImage) -> Vec<f32> {
    let mut x = vec![0.0f32; 3 * 224 * 224]; // NCHW
    for (px, py, p) in img.enumerate_pixels() {
        for c in 0..3 {
            x[c * 224 * 224 + (py * 224 + px) as usize] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    x
}

fn preprocess(path: &str) -> Vec<f32> {
    normalize(&resize(&decode(path)))
}

fn infer(session: &mut Session, x: Vec<f32>) -> Vec<f32> {
    let input = Tensor::from_array(([1, 3, 224, 224], x)).unwrap();
    let outputs = session.run(ort::inputs!["input" => input]).unwrap();
    outputs["logits"].try_extract_tensor::<f32>().unwrap().1.to_vec()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = logits.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|v| v / sum).collect()
}

fn argmax(probs: &[f32]) -> usize {
    (0..probs.len()).max_by(|&a, &b| probs[a].total_cmp(&probs[b])).unwrap()
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

// p95: linear interpolation between the two nearest samples, same as np.percentile.
fn p95(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let r = 0.95 * (s.len() - 1) as f64;
    let i = r as usize;
    if i + 1 == s.len() { s[i] } else { s[i] + (r - i as f64) * (s[i + 1] - s[i]) }
}

fn classify(session: &mut Session, labels: &[&str], image_path: &str) {
    let x = preprocess(image_path);
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let std = (x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    let probs = softmax(&infer(session, x));
    let mut top: Vec<usize> = (0..probs.len()).collect();
    top.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    println!("[1] Classifying (top-3, ONNX Runtime CPU)");
    for &i in &top[..3] {
        println!("    {:<15} {:.2}%", labels[i], probs[i] * 100.0);
    }
    println!("    input tensor: mean {mean:.4}, std {std:.4}");
    assert!(CAT_CLASSES.contains(&labels[top[0]].to_lowercase().as_str()), "top-1 is not a cat: {}", labels[top[0]]);
    println!();
}

fn benchmark(session: &mut Session, image_path: &str) -> (f64, f64, f64) {
    for _ in 0..10 {
        infer(session, preprocess(image_path));
    }

    let names = ["decode", "resize + crop", "normalize", "infer (ONNX Runtime)", "postprocess"];
    let (mut times, mut prep, mut total) = (vec![vec![]; names.len()], vec![], vec![]);
    for _ in 0..200 {
        let t0 = Instant::now();
        let img = decode(image_path);
        let t1 = Instant::now();
        let img = resize(&img);
        let t2 = Instant::now();
        let x = normalize(&img);
        let t3 = Instant::now();
        let logits = infer(session, x);
        let t4 = Instant::now();
        let best = argmax(&softmax(&logits));
        let t5 = Instant::now();
        std::hint::black_box(best);
        let t = [t0, t1, t2, t3, t4, t5];
        let mut sum = 0.0;
        for i in 0..5 {
            times[i].push((t[i + 1] - t[i]).as_secs_f64() * 1000.0);
            sum += times[i].last().unwrap();
            if i == 2 {
                prep.push(sum);
            }
        }
        total.push(sum);
    }

    println!("[2] Where one request spends its time (200 requests after 10 warm-ups)");
    println!("    {:<20} {:>11} {:>11} {:>8}", "", "median", "p95", "share");
    for (name, ms) in names.iter().zip(&times) {
        println!("    {name:<20} {:8.3} ms {:8.3} ms {:7.1}%", median(ms), p95(ms), median(ms) / median(&total) * 100.0);
    }
    println!("    {:<20} {:8.3} ms {:8.3} ms", "total", median(&total), p95(&total));
    (median(&prep), median(&times[3]), median(&total))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = include_str!("main.rs");
    let code_lines = source.lines().filter(|l| !l.trim().is_empty() && !l.trim().starts_with("//")).count();
    println!("main.rs: {code_lines} lines of code\n");

    let labels_text = std::fs::read_to_string(format!("{SHARED}/model/labels.txt"))?;
    let labels: Vec<&str> = labels_text.lines().collect();
    let env = ort::init_from(format!("{SHARED}/onnxruntime/lib/libonnxruntime.1.32.0.dylib"))?.build()?;
    let mut sessions = HashMap::new();
    for (name, file) in MODELS {
        let session = Session::builder(&env)?
            .with_auto_device(AutoDevicePolicy::PreferCPU)? // CPU only; ort's default on macOS adds CoreML
            .commit_from_file(format!("{SHARED}/model/{file}"))?;
        sessions.insert(name, session);
    }

    let mut summary = vec![];
    for (n, (name, file, model)) in CASES.iter().enumerate() {
        println!("== case {}: {name} ({file}, {model}) ==\n", n + 1);
        let image_path = format!("{SHARED}/{file}");
        let session = sessions.get_mut(model).unwrap();
        classify(session, &labels, &image_path);
        summary.push(benchmark(session, &image_path));
        println!();
    }

    // ort does not expose the ONNX Runtime version string; the build commit is in the build info.
    let commit = ort::info().split(", ").find(|s| s.starts_with("git-commit-id")).unwrap_or("?");
    println!(
        "Env: ort 2.0.0-rc.15, image 0.25.10, onnxruntime dylib {commit}, {}/{}, ORT thread default\n",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("{:<22}{:>9}{:>10}{:>10}", "Summary (median ms)", "prep", "model", "total");
    for ((name, _, _), (prep, model, total)) in CASES.iter().zip(&summary) {
        println!("    {name:<18}{prep:9.3}{model:10.3}{total:10.3}");
    }
    Ok(())
}
