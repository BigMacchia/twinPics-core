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
//!
//! ## Comparing CLI vs desktop index speed (manual)
//!
//! - Use **release** builds: `cargo build --release -p twinpics_cli` and a release desktop bundle.
//! - **PDF feature parity**: the CLI crate depends on `twinpics_core` without the `pdf` feature; the
//!   desktop crate enables `pdf`. If the folder contains PDFs, desktop does extra work unless you
//!   align features or exclude PDFs from the test folder.
//! - Example photo root: `D:\Marco\photos\2024` — set `TWINPICS_PERF_SRC` or use synthetic images (default).
//! - CLI prints total/scan/process/write ms; desktop `add_index` returns the same timing fields in its summary.

#![cfg(feature = "real-clip")]

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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

/// Compares indexing speed across option combinations to isolate cost of each phase.
///
/// Run with:
///   $env:TWINPICS_BENCH_DIR = "D:\Marco\photos\2024"
///   cargo test -p twinpics_core --features real-clip -- --ignored --nocapture bench_index_options
#[test]
#[serial]
#[ignore = "benchmark — requires CLIP weights and TWINPICS_BENCH_DIR; run with --ignored --nocapture"]
fn bench_index_options() {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let src = {
        let key = if std::env::var("TWINPICS_BENCH_DIR").is_ok() {
            "TWINPICS_BENCH_DIR"
        } else {
            "TWINPICS_PERF_SRC"
        };
        let p = std::env::var(key)
            .unwrap_or_else(|_| panic!("set TWINPICS_BENCH_DIR (or TWINPICS_PERF_SRC) to a photo folder"));
        let path = PathBuf::from(&p);
        assert!(path.is_dir(), "{key}={p} is not an accessible directory");
        path.canonicalize().unwrap()
    };

    println!("\n╔══ Benchmark: IndexOptions comparison  ({cpus} CPUs) ══╗");
    println!("  source: {}", src.display());

    struct BenchResult { label: &'static str, out: IndexOutcome }
    let mut results: Vec<BenchResult> = Vec::new();

    let configs: &[(&str, IndexOptions)] = &[
        ("baseline (no tags, no colors)", IndexOptions {
            skip_tagging: true,
            extract_colors: false,
            ..Default::default()
        }),
        ("tags, no colors", IndexOptions {
            extract_colors: false,
            ..Default::default()
        }),
        ("full  (tags + parallel colors)", IndexOptions::default()),
    ];

    for (label, opts) in configs {
        let paths = project_paths_for_source(&src).unwrap();
        twinpics_core::clean_project(&src).ok();
        let backend = clip_backend();
        let t0 = Instant::now();
        let out = index_folder(&src, backend, &paths, opts.clone())
            .unwrap_or_else(|e| panic!("index failed for {label}: {e}"));
        let wall = t0.elapsed().as_millis();
        println!(
            "\n  [{label}]\n    files={:>5}  wall={:>6}ms  scan={:>5}ms  process={:>6}ms  write={:>5}ms",
            out.file_count, wall, out.scan_ms, out.process_ms, out.finalize_ms
        );
        results.push(BenchResult { label, out });
    }

    println!("\n┌─ Overhead summary ────────────────────────────────────");
    if results.len() >= 2 {
        let base = results[0].out.total_ms as f64;
        for r in &results[1..] {
            let delta = r.out.total_ms as i64 - results[0].out.total_ms as i64;
            println!("  {} vs baseline: {:+} ms", r.label, delta);
        }
        if results.len() >= 3 {
            let tag_only = results[1].out.total_ms as f64;
            let full = results[2].out.total_ms as f64;
            if full > tag_only {
                println!("  Color extraction overhead: {:.0} ms  ({:.1}x vs tags-only)",
                    full - tag_only, full / tag_only.max(1.0));
            }
            let _ = base;
        }
    }
    println!("└───────────────────────────────────────────────────────");
    println!("╚═══════════════════════════════════════════════════════╝\n");
}

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

/// Mirrors desktop `add_index_work` progress filtering: skip `FileDiscovered` / `FileStarted`,
/// throttle `FileFinished` (~100 ms) with an atomic — no Tauri `emit`.
fn ui_like_throttle_allow(last_emit_ms: &AtomicU64, throttle_ms: u64) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let mut prev = last_emit_ms.load(Ordering::Relaxed);
    loop {
        if prev != 0 && now.saturating_sub(prev) < throttle_ms {
            return false;
        }
        match last_emit_ms.compare_exchange_weak(prev, now, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(_) => return true,
            Err(p) => prev = p,
        }
    }
}

fn ui_like_progress_callback() -> (ProgressCallback, Arc<AtomicUsize>) {
    let emitted = Arc::new(AtomicUsize::new(0));
    let last_emit_ms = Arc::new(AtomicU64::new(0));
    let e = emitted.clone();
    let cb: ProgressCallback = Arc::new(move |p: IndexProgress| {
        match &p {
            IndexProgress::FileDiscovered { .. } | IndexProgress::FileStarted { .. } => return,
            _ => {}
        }
        let always = matches!(
            p,
            IndexProgress::Scanning
                | IndexProgress::Discovered { .. }
                | IndexProgress::Finishing
                | IndexProgress::Done
        );
        const THROTTLE_MS: u64 = 100;
        if !always && !ui_like_throttle_allow(last_emit_ms.as_ref(), THROTTLE_MS) {
            return;
        }
        let counts = matches!(
            p,
            IndexProgress::Scanning
                | IndexProgress::Discovered { .. }
                | IndexProgress::FileFinished { .. }
                | IndexProgress::Finishing
                | IndexProgress::Done
        );
        if counts {
            e.fetch_add(1, Ordering::Relaxed);
        }
    });
    (cb, emitted)
}

/// Ensures a desktop-shaped progress callback does not materially slow `index_folder` vs no callback.
///
/// Run with:
///   $env:TWINPICS_PERF_SRC = "D:\Marco\photos\2024"
///   cargo test -p twinpics_core --features real-clip -- --ignored --nocapture perf_ui_progress_callback_overhead
#[test]
#[serial]
#[ignore = "performance test — requires CLIP weights; run with --ignored --nocapture"]
fn perf_ui_progress_callback_overhead() {
    let _env = TestEnv::new();
    let (src, _synthetic) = resolve_source(&_env);
    let paths = project_paths_for_source(&src).unwrap();

    twinpics_core::clean_project(&src).ok();

    let backend_none = clip_backend();
    let baseline = index_folder(
        &src,
        backend_none,
        &paths,
        IndexOptions {
            on_progress: None,
            ..Default::default()
        },
    )
    .expect("baseline index_folder failed");

    twinpics_core::clean_project(&src).ok();

    let backend_ui = clip_backend();
    let (ui_cb, emit_count) = ui_like_progress_callback();
    let with_ui_cb = index_folder(
        &src,
        backend_ui,
        &paths,
        IndexOptions {
            on_progress: Some(ui_cb),
            ..Default::default()
        },
    )
    .expect("ui_like index_folder failed");

    println!(
        "\n╔══ perf_ui_progress_callback_overhead ══╗\n  source: {}\n  baseline total_ms: {}\n  ui_like total_ms:  {}\n  simulated emits: {}\n╚════════════════════════════════════════╝\n",
        src.display(),
        baseline.total_ms,
        with_ui_cb.total_ms,
        emit_count.load(Ordering::Relaxed)
    );

    assert_eq!(
        baseline.file_count, with_ui_cb.file_count,
        "file_count must match between runs"
    );
    assert!(
        emit_count.load(Ordering::Relaxed) > 0,
        "ui_like callback should record at least one progress step"
    );

    assert!(
        with_ui_cb.total_ms <= baseline.total_ms.max(1).saturating_mul(2),
        "ui_like total_ms ({}) should not roughly double baseline ({} ms)",
        with_ui_cb.total_ms,
        baseline.total_ms
    );
}
