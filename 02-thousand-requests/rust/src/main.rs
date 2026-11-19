// AI series, part 2, Rust version: the classifier from part 1 as an HTTP service.
// Run from this folder (02-thousand-requests/rust/): cargo run --release
// Settings (environment variables): PORT (default 8000), ORT_THREADS (0 = ONNX Runtime's default),
// SESSIONS (number of ONNX Runtime sessions, default 1), LIMIT (requests worked on at once, 0 = no limit of my own),
// QUEUE (requests allowed to wait when LIMIT is set; one more gets 503), SHARE_SESSION (1 = all requests run the model
// on one session at the same time, which ort does not allow in safe code; see SharedSession).
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use axum::body::Bytes;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use image::imageops::{self, FilterType};
use image::{ImageResult, RgbImage};
use ort::session::{Session, builder::AutoDevicePolicy};
use ort::value::Tensor;
use serde_json::{Value, json};
use tokio::sync::Semaphore;

const SHARED: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../shared");
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
const CAT_CLASSES: [&str; 5] = ["tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat"];

// Sessions and labels, loaded once. `Session::run` takes `&mut self`, so one session serves one
// request at a time: a request takes a session out of `sessions`, runs it, and puts it back.
struct State {
    sessions: Mutex<Vec<Session>>,
    returned: Condvar, // signalled when a session is put back
    shared: Option<SharedSession>, // set with SHARE_SESSION=1: one session for every request, and `sessions` is empty
    labels: Vec<String>,
    slots: Semaphore,    // with LIMIT set: at most LIMIT requests hold a slot, the rest wait in order
    capacity: usize,     // LIMIT + QUEUE, or 0 for no limit of my own
    inside: AtomicUsize, // requests working or waiting
}

static STATE: OnceLock<State> = OnceLock::new();

// One session used by many threads at once. `ort` forbids this on purpose: `Session::run` takes `&mut self` because,
// in its maintainer's words, parts of ONNX Runtime are not thread safe, above all with GPU providers
// (https://github.com/pykeio/ort/issues/402). ONNX Runtime's own Python and Go bindings do share a session this way.
// Here it is opt-in, CPU only, and it is the one `unsafe` in this file: the compiler no longer checks it.
struct SharedSession(UnsafeCell<Session>);
unsafe impl Sync for SharedSession {}

// One request that is working or waiting; dropping it takes the request off the count.
struct Inside;
impl Drop for Inside {
    fn drop(&mut self) {
        STATE.get().unwrap().inside.fetch_sub(1, Ordering::SeqCst);
    }
}

fn env_number(name: &str, default: usize) -> usize {
    std::env::var(name).map_or(default, |v| v.parse().unwrap())
}

fn decode(body: &[u8]) -> ImageResult<RgbImage> {
    Ok(image::load_from_memory(body)?.to_rgb8())
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

fn preprocess(body: &[u8]) -> ImageResult<Vec<f32>> {
    Ok(normalize(&resize(&decode(body)?)))
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

fn classify(body: &[u8]) -> ImageResult<(&'static str, f32)> {
    let state = STATE.get().unwrap();
    let x = preprocess(body)?; // no session held here, so requests can overlap
    let logits = if let Some(shared) = &state.shared {
        infer(unsafe { &mut *shared.0.get() }, x) // many threads in the same session at once
    } else {
        let mut free = state.sessions.lock().unwrap();
        while free.is_empty() {
            free = state.returned.wait(free).unwrap(); // all sessions are busy: wait for one
        }
        let mut session = free.pop().unwrap();
        drop(free); // unlock while the model runs
        let logits = infer(&mut session, x);
        state.sessions.lock().unwrap().push(session);
        state.returned.notify_one();
        logits
    };
    let probs = softmax(&logits);
    let best = argmax(&probs);
    Ok((&state.labels[best], probs[best]))
}

fn load(ort_threads: usize, session_count: usize, limit: usize, queue: usize) -> Result<(), Box<dyn std::error::Error>> {
    let env = ort::init_from(format!("{SHARED}/onnxruntime/lib/libonnxruntime.1.32.0.dylib"))?.build()?;
    let mut sessions = vec![];
    for _ in 0..session_count {
        // CPU only; ort's default on macOS adds CoreML
        let mut builder = Session::builder(&env)?.with_auto_device(AutoDevicePolicy::PreferCPU)?;
        if ort_threads > 0 {
            builder = builder.with_intra_threads(ort_threads)?;
        }
        sessions.push(builder.commit_from_file(format!("{SHARED}/model/mobilenet_v2.onnx"))?);
    }
    let labels = std::fs::read_to_string(format!("{SHARED}/model/labels.txt"))?.lines().map(String::from).collect();
    let slots = Semaphore::new(if limit > 0 { limit } else { Semaphore::MAX_PERMITS });
    let capacity = if limit > 0 { limit + queue } else { 0 };
    let shared = if std::env::var("SHARE_SESSION").is_ok_and(|v| v == "1") { sessions.pop().map(|s| SharedSession(UnsafeCell::new(s))) } else { None };
    let state = State { sessions: Mutex::new(sessions), returned: Condvar::new(), shared, labels, slots, capacity, inside: AtomicUsize::new(0) };
    assert!(STATE.set(state).is_ok(), "load called twice");
    let body = std::fs::read(format!("{SHARED}/cat.jpg"))?;
    let mut label = "";
    for _ in 0..10 {
        // warm-up, and a check that the answer is still a cat
        (label, _) = classify(&body)?;
    }
    assert!(CAT_CLASSES.contains(&label.to_lowercase().as_str()), "top-1 is not a cat: {label}");
    Ok(())
}

async fn handle_classify(body: Bytes) -> Response {
    let state = STATE.get().unwrap();
    if state.inside.fetch_add(1, Ordering::SeqCst) >= state.capacity && state.capacity > 0 {
        state.inside.fetch_sub(1, Ordering::SeqCst); // the queue is full: say so at once instead of letting it wait
        return (StatusCode::SERVICE_UNAVAILABLE, "busy").into_response();
    }
    let inside = Inside; // counted until dropped, also when the client hangs up while this request waits
    let slot = state.slots.acquire().await.unwrap();
    // The blocking work leaves the async runtime and runs on a blocking thread (up to 512 by default).
    // The slot travels with the work, so it is given back when the work ends, not when the client leaves.
    let work = move || {
        let result = classify(&body);
        drop((slot, inside));
        result
    };
    match tokio::task::spawn_blocking(work).await.unwrap() {
        Ok((label, prob)) => Json(json!({"label": label, "prob": prob})).into_response(),
        Err(_) => (StatusCode::BAD_REQUEST, "cannot decode image").into_response(), // the image crate could not read it
    }
}

async fn handle_ping() -> Json<Value> {
    Json(json!({"ok": true}))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (port, ort_threads, session_count) = (env_number("PORT", 8000), env_number("ORT_THREADS", 0), env_number("SESSIONS", 1));
    let (limit, queue) = (env_number("LIMIT", 0), env_number("QUEUE", 0));
    load(ort_threads, session_count, limit, queue)?;
    println!(
        "Rust 1.100.0, axum 0.8.9 + tokio 1.54.1, ort 2.0.0-rc.15, image 0.25.10, \
         ORT_THREADS={ort_threads} (0 = default), SESSIONS={session_count}, SHARE_SESSION={}, LIMIT={limit}, QUEUE={queue}, listening on :{port}",
        STATE.get().unwrap().shared.is_some() as u8
    );
    let app = Router::new().route("/classify", post(handle_classify)).route("/ping", get(handle_ping));
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port as u16)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
