//! Performance tests for `index_folder` — requires real CLIP weights.
//!
//! Run with:
//!   cargo test -p twinpics_core --features real-clip -- --ignored --nocapture perf
//!
//! Environment variables:
//!   TWINPICS_PERF_N    Number of synthetic images to create (default: 50).
//!   TWINPICS_PERF_SRC  Path to an existing photo folder to index instead of
//!                      generating synthetic images.  When set, TWINPICS_PERF_N
//!                      is ignored.
//!   TWINPICS_CLIP_MODEL_DIR  Directory containing model.safetensors +
//!                            tokenizer.json (checked by the backend).

#![cfg(feature = "real-clip")]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serial_test::serial;
use twinpics_core::ml::{CandleClipBackend, EmbeddingBackend};
use twinpics_core::{
    index_folder, project_paths_for_source, IndexOptions, IndexOutcome, IndexProgress,
    Manifest, ProgressCallback,
};

use common::TestEnv;

// ── helpers ──────────────────────────────────────────────────────────────────

fn clip_backend() -> Arc<dyn EmbeddingBackend> {
    Arc::new(CandleClipBackend::new().expect(
        "CLIP backend failed — set TWINPICS_CLIP_MODEL_DIR or enable the real-clip feature \
         (which downloads from Hugging Face on first use)",
    ))
}

/// Create `n` synthetic 224×224 PNG images (solid colours, varied hues).
/// 224px matches CLIP's native input size, giving realistic decode overhead.
fn create_synthetic_images(dir: &Path, n: usize) {
    use image::{ImageBuffer, Rgb};
    fs::create_dir_all(dir).unwrap();
    for i in 0..n {
        // Spread colours evenly so images aren't all identical.
        let r = ((i * 97) % 256) as u8;
        let g = ((i * 137) % 256) as u8;
        let b = ((i * 193) % 256) as u8;
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_fn(224, 224, |_, _| Rgb([r, g, b]));
        img.save(dir.join(format!("img_{i:04}.png"))).unwrap();
    }
}

/// Stats collected via the progress callback.
#[derive(Default)]
struct RunStats {
    cached_count: usize,
    embedded_count: usize,
    /// Per-file embed times for newly-embedded files (ms).
    embed_times_ms: Vec<u64>,
}

/// Build a progress callback that collects `RunStats` alongside the standard outcome.
fn stats_callback() -> (ProgressCallback, Arc<Mutex<RunStats>>) {
    let stats = Arc::new(Mutex::new(RunStats::default()));
    let s = stats.clone();
    let cb: ProgressCallback = Arc::new(move |ev: IndexProgress| {
        if let IndexProgress::FileFinished { elapsed_ms, .. } = ev {
            let mut st = s.lock().unwrap();
            if elapsed_ms == 0 {
                st.cached_count += 1;
            } else {
                st.embedded_count += 1;
                st.embed_times_ms.push(elapsed_ms);
            }
        }
    });
    (cb, stats)
}

/// Pretty-print a single indexed run.
fn print_run(label: &str, outcome: &IndexOutcome, stats: &RunStats) {
    let n = outcome.file_count;
    let total_per = if n > 0 { outcome.total_ms as f64 / n as f64 } else { 0.0 };

    println!("  {label}");
    println!("    files   : {n}  ({} cached, {} embedded)",
        stats.cached_count, stats.embedded_count);
    println!("    scan    : {:>7} ms", outcome.scan_ms);
    println!("    process : {:>7} ms", outcome.process_ms);
    if !stats.embed_times_ms.is_empty() {
        let times = &stats.embed_times_ms;
        let sum: u64 = times.iter().sum();
        let avg = sum as f64 / times.len() as f64;
        let mut sorted = times.clone();
        sorted.sort_unstable();
        let p50 = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
        let min = sorted[0];
        let max = *sorted.last().unwrap();
        println!("      embed avg: {avg:.1} ms/img  p50={p50} p95={p95} min={min} max={max}");
    }
    println!("    finalize: {:>7} ms", outcome.finalize_ms);
    println!("    total   : {:>7} ms  ({total_per:.1} ms/img)", outcome.total_ms);
}

/// Resolve the source directory: real folder from env, or a freshly generated set.
fn resolve_source(env: &TestEnv) -> (PathBuf, bool) {
    if let Ok(p) = std::env::var("TWINPICS_PERF_SRC") {
        let src = PathBuf::from(&p);
        assert!(
            src.is_dir(),
            "TWINPICS_PERF_SRC={p} is not an accessible directory"
        );
        (src, false) // don't clean up — it's the user's folder
    } else {
        let n: usize = std::env::var("TWINPICS_PERF_N")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);
        let src = env.path().join("perf_photos");
        let t = Instant::now();
        print!("  Creating {n} synthetic 224×224 PNG images… ");
        std::io::Write::flush(&mut std::io::stdout()).ok();
        create_synthetic_images(&src, n);
        println!("{} ms", t.elapsed().as_millis());
        (src, true)
    }
}

// ── test ─────────────────────────────────────────────────────────────────────

/// Measures `index_folder` performance: cold run (fresh embed) then warm run
/// (all files cached from the first run).
///
/// Run with:
///   cargo test -p twinpics_core --features real-clip -- --ignored --nocapture perf_index_cold_and_warm
#[test]
#[serial]
#[ignore = "performance test — requires CLIP weights; run with --ignored --nocapture"]
fn perf_index_cold_and_warm() {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let _env = TestEnv::new();
    let (src, _synthetic) = resolve_source(&_env);

    println!("\n╔══ Performance: index_folder  ({cpus} logical CPUs) ══╗");

    // Count actual images before indexing.
    let all_images: Vec<_> = walkdir::WalkDir::new(&src)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().is_file()
                && e.path()
                    .extension()
                    .and_then(|x| x.to_str())
                    .map(|x| matches!(x.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "webp" | "bmp" | "tiff"))
                    .unwrap_or(false)
        })
        .collect();
    println!("  source  : {}", src.display());
    println!("  images  : {}", all_images.len());

    let paths = project_paths_for_source(&src).unwrap();

    // ── Cold run ─────────────────────────────────────────────────────────────
    println!("\n┌─ Cold run (no cache) ─────────────────────────────────");
    let (cb_cold, stats_cold_arc) = stats_callback();
    let backend_cold = clip_backend();
    let cold = index_folder(
        &src,
        backend_cold,
        &paths,
        IndexOptions {
            on_progress: Some(cb_cold),
            ..Default::default()
        },
    )
    .expect("cold indexing failed");
    let stats_cold = stats_cold_arc.lock().unwrap();
    print_run("", &cold, &stats_cold);
    println!("└───────────────────────────────────────────────────────");

    // ── Warm run ─────────────────────────────────────────────────────────────
    println!("\n┌─ Warm run (all files cached) ─────────────────────────");
    let (cb_warm, stats_warm_arc) = stats_callback();
    let backend_warm = clip_backend();
    let warm = index_folder(
        &src,
        backend_warm,
        &paths,
        IndexOptions {
            on_progress: Some(cb_warm),
            ..Default::default()
        },
    )
    .expect("warm indexing failed");
    let stats_warm = stats_warm_arc.lock().unwrap();
    print_run("", &warm, &stats_warm);
    println!("└───────────────────────────────────────────────────────");

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\n┌─ Summary ──────────────────────────────────────────────");
    if cold.total_ms > 0 && warm.total_ms > 0 {
        let speedup = cold.total_ms as f64 / warm.total_ms as f64;
        println!("  total speedup (cold → warm): {speedup:.1}x");
    }
    if cold.process_ms > 0 && warm.process_ms > 0 {
        let embed_speedup = cold.process_ms as f64 / warm.process_ms as f64;
        println!("  process speedup:             {embed_speedup:.1}x");
    }
    println!("└───────────────────────────────────────────────────────");
    println!("╚══════════════════════════════════════════════════════╝\n");

    // ── Assertions ────────────────────────────────────────────────────────────
    let manifest = Manifest::load_or_empty(&paths.manifest_path).unwrap();
    assert_eq!(manifest.entries.len(), cold.file_count, "manifest entry count mismatch");
    assert_eq!(cold.file_count, warm.file_count, "file count differs between runs");
    // Warm run must reuse all embeddings — zero new embeds.
    assert_eq!(
        stats_warm.embedded_count, 0,
        "warm run should embed 0 files (got {})",
        stats_warm.embedded_count
    );
    assert_eq!(
        stats_warm.cached_count,
        warm.file_count,
        "warm run should reuse all {} embeddings",
        warm.file_count
    );
    // Warm process phase must be faster than cold.
    assert!(
        warm.process_ms <= cold.process_ms,
        "warm process ({} ms) must not exceed cold process ({} ms)",
        warm.process_ms,
        cold.process_ms
    );
}
