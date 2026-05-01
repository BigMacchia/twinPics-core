//! Walk a folder, embed images, build usearch index and project files.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use chrono::Utc;
use crate::ml::{build_index, save_index, EmbeddingBackend};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::error::CoreError;
use crate::manifest::{rel_path_posix, Manifest, ManifestEntry};
use crate::project::{ProjectConfig, ProjectPaths};
use crate::tags::{
    compute_vocab_hash, embed_vocab, load_vocab, tags_for_image, TagsDb,
};
#[cfg(feature = "pdf")]
use crate::pdf::{load_pdfium, render_pdf_pages};

/// Supported image extensions (lowercase, no dot).
pub const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "tiff"];

/// Progress reported while indexing; used by CLI, tests, and Tauri.
#[derive(Debug, Clone)]
pub enum IndexProgress {
    /// About to walk the directory tree to find images (no `total` yet).
    Scanning,
    /// One image file found during scanning; `count` is the running total so far.
    FileDiscovered {
        /// Running count of image files found so far.
        count: usize,
    },
    /// Image files discovered; `total` is the number of files to process.
    Discovered {
        /// Number of image files to index.
        total: usize,
        /// Wall time to walk, filter, and sort image paths, in milliseconds.
        scan_ms: u64,
    },
    /// A file is about to be processed (hash, mtime, embed, tags).
    FileStarted {
        /// Zero-based index in the ordered scan (0..total-1).
        index: usize,
        /// Total files (same as in [`IndexProgress::Discovered`]).
        total: usize,
        /// Absolute path to the image file.
        path: PathBuf,
    },
    /// One file finished; includes rolling average and ETA for remaining files.
    FileFinished {
        /// Index of the file that just completed (0-based).
        index: usize,
        /// Total files.
        total: usize,
        /// Path that was just processed.
        path: PathBuf,
        /// Wall time for that file only, milliseconds.
        elapsed_ms: u64,
        /// Average ms per file so far (including this one).
        avg_ms: f64,
        /// Estimated time to finish remaining files, ms.
        eta_ms: u64,
    },
    /// Embedding pass done; building usearch and writing project files.
    Finishing,
    /// All work complete (after artefacts are on disk).
    Done,
}

/// Optional callback (also [`Clone`] for cheap sharing into nested closures).
pub type ProgressCallback = Arc<dyn Fn(IndexProgress) + Send + Sync>;

/// Result of a successful `index_folder` run.
#[derive(Debug, Clone)]
pub struct IndexOutcome {
    /// Number of image files in the index (including skipped re-embeds from manifest).
    pub file_count: usize,
    /// Project artefact files that exist on disk after indexing (absolute paths).
    pub artefacts: Vec<PathBuf>,
    /// Milliseconds to discover and sort image paths.
    pub scan_ms: u64,
    /// Milliseconds in the per-image phase (load manifest, per-file hash/embed/tags).
    pub process_ms: u64,
    /// Milliseconds to build the vector index, write manifest, and config.
    pub finalize_ms: u64,
    /// Wall time for the full [`index_folder`] run from after project dir and tags DB setup, in ms.
    pub total_ms: u64,
}

/// Options for `index_folder`.
#[derive(Clone)]
pub struct IndexOptions {
    /// Recurse into subdirectories.
    pub recursive: bool,
    /// Model label stored in `config.json`.
    pub model: String,
    /// Optional vocabulary file (one tag per line); `None` uses the built-in list.
    pub tags_file: Option<PathBuf>,
    /// Minimum cosine similarity (dot product) for assigning a tag to an image.
    pub tag_threshold: f32,
    /// When true, do not build or update `tags.sqlite` (and remove an existing DB).
    pub skip_tagging: bool,
    /// Optional progress hook (e.g. CLI bar or Tauri events).
    pub on_progress: Option<ProgressCallback>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            recursive: true,
            model: "clip-vit-base-patch32".to_string(),
            tags_file: None,
            tag_threshold: 0.22,
            skip_tagging: false,
            on_progress: None,
        }
    }
}

fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let e = e.to_ascii_lowercase();
            IMAGE_EXTS.iter().any(|&x| x == e)
        })
        .unwrap_or(false)
}

fn is_pdf_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("pdf"))
        .unwrap_or(false)
}

/// One embeddable unit: either a plain image or a rendered PDF page.
struct FileEntry {
    /// Path passed to `backend.embed_image()`. For PDF pages: the rendered PNG.
    embed_path: PathBuf,
    /// Manifest key. For images: POSIX relative path. For PDF pages: absolute path string.
    rel_path: String,
    /// SHA-256 of the source file. For PDF pages: digest of the source PDF.
    sha256: String,
    /// mtime of the source file. For PDF pages: mtime of the source PDF.
    mtime_secs: i64,
    /// POSIX relative path of source PDF, or `None` for plain images.
    pdf_source: Option<String>,
    /// 0-based page index within the PDF, or `None` for plain images.
    pdf_page: Option<u32>,
}

fn hash_file(path: &Path) -> Result<String, CoreError> {
    let bytes = fs::read(path)?;
    let h = Sha256::digest(&bytes);
    Ok(format!("{:x}", h))
}

fn mtime_secs(path: &Path) -> Result<i64, CoreError> {
    let m = fs::metadata(path)?.modified()?;
    Ok(m.duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64)
}

/// Index all images under `source`, writing project files under `paths`.
pub fn index_folder(
    source: &Path,
    backend: Arc<dyn EmbeddingBackend>,
    paths: &ProjectPaths,
    opts: IndexOptions,
) -> Result<IndexOutcome, CoreError> {
    let source = source
        .canonicalize()
        .map_err(|e| CoreError::msg(format!("source path: {e}")))?;

    paths.ensure_dir()?;

    if opts.skip_tagging && paths.tags_db_path.is_file() {
        fs::remove_file(&paths.tags_db_path)?;
    }

    let mut tags_db_opt: Option<TagsDb> = if opts.skip_tagging {
        None
    } else {
        Some(TagsDb::open_or_init(&paths.tags_db_path)?)
    };
    let mut vocab_cache: Option<(Vec<String>, Vec<Vec<f32>>)> = None;

    let t_all = Instant::now();
    if let Some(ref cb) = opts.on_progress {
        cb(IndexProgress::Scanning);
    }

    let t_scan = Instant::now();
    let mut image_paths: Vec<PathBuf> = Vec::new();
    let mut pdf_paths: Vec<PathBuf> = Vec::new();
    let mut discovered_count = 0usize;

    if opts.recursive {
        for entry in WalkDir::new(&source).into_iter().filter_map(|e| e.ok()) {
            if entry.file_type().is_file() {
                let path = entry.path().to_path_buf();
                if is_image(&path) {
                    image_paths.push(path);
                    discovered_count += 1;
                    if let Some(ref cb) = opts.on_progress {
                        cb(IndexProgress::FileDiscovered { count: discovered_count });
                    }
                } else if is_pdf_file(&path) {
                    pdf_paths.push(path);
                    discovered_count += 1;
                    if let Some(ref cb) = opts.on_progress {
                        cb(IndexProgress::FileDiscovered { count: discovered_count });
                    }
                }
            }
        }
    } else {
        for entry in fs::read_dir(&source)?.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_file() {
                if is_image(&path) {
                    image_paths.push(path);
                    discovered_count += 1;
                    if let Some(ref cb) = opts.on_progress {
                        cb(IndexProgress::FileDiscovered { count: discovered_count });
                    }
                } else if is_pdf_file(&path) {
                    pdf_paths.push(path);
                    discovered_count += 1;
                    if let Some(ref cb) = opts.on_progress {
                        cb(IndexProgress::FileDiscovered { count: discovered_count });
                    }
                }
            }
        }
    }
    image_paths.sort();
    pdf_paths.sort();

    // ── Phase A: parallel hash + mtime for images and PDFs ──────────────────
    let image_meta_results: Vec<Result<(String, String, i64), CoreError>> = image_paths
        .par_iter()
        .map(|path| {
            let rel = rel_path_posix(&source, path)?;
            let hash = hash_file(path)?;
            let mtime = mtime_secs(path)?;
            Ok((rel, hash, mtime))
        })
        .collect();
    let image_metas: Vec<(String, String, i64)> =
        image_meta_results.into_iter().collect::<Result<_, _>>()?;

    // ── PDF rendering: sequential (pdfium page rendering is single-threaded) ─
    #[cfg(feature = "pdf")]
    let pdf_page_entries: Vec<FileEntry> = {
        if pdf_paths.is_empty() {
            Vec::new()
        } else {
            // Hash PDFs in parallel to get cache keys, then render pages sequentially.
            let pdf_meta_results: Vec<Result<(String, String, i64), CoreError>> = pdf_paths
                .par_iter()
                .map(|path| {
                    let rel = rel_path_posix(&source, path)?;
                    let hash = hash_file(path)?;
                    let mtime = mtime_secs(path)?;
                    Ok((rel, hash, mtime))
                })
                .collect();
            let pdf_metas: Vec<(String, String, i64)> =
                pdf_meta_results.into_iter().collect::<Result<_, _>>()?;

            let pdfium = load_pdfium()?;
            let mut entries = Vec::new();
            for (pdf_path, (pdf_rel, hash, mtime)) in pdf_paths.iter().zip(pdf_metas.iter()) {
                let rendered = render_pdf_pages(
                    &pdfium, pdf_path, hash, &paths.pdf_renders_path, false,
                )?;
                for page in rendered {
                    let abs_str = page.png_path.to_string_lossy().into_owned();
                    entries.push(FileEntry {
                        embed_path: page.png_path,
                        rel_path: abs_str,
                        sha256: hash.clone(),
                        mtime_secs: *mtime,
                        pdf_source: Some(pdf_rel.clone()),
                        pdf_page: Some(page.page_index),
                    });
                }
            }
            entries
        }
    };

    #[cfg(not(feature = "pdf"))]
    let pdf_page_entries: Vec<FileEntry> = {
        if !pdf_paths.is_empty() {
            tracing::warn!(
                "{} PDF file(s) found but skipped (compile with `--features pdf` to index PDFs)",
                pdf_paths.len()
            );
        }
        Vec::new()
    };

    // Combine: images first (stable sorted order), then PDF pages (ordered by file, then page).
    let mut all_entries: Vec<FileEntry> = image_paths
        .into_iter()
        .zip(image_metas.into_iter())
        .map(|(path, (rel, hash, mtime))| FileEntry {
            embed_path: path,
            rel_path: rel,
            sha256: hash,
            mtime_secs: mtime,
            pdf_source: None,
            pdf_page: None,
        })
        .collect();
    all_entries.extend(pdf_page_entries);

    let scan_ms = t_scan.elapsed().as_millis() as u64;
    let total = all_entries.len();
    if let Some(ref cb) = opts.on_progress {
        cb(IndexProgress::Discovered { total, scan_ms });
    }

    let t_process = Instant::now();
    let old_manifest = Manifest::load_or_empty(&paths.manifest_path)?;
    // O(1) lookup by rel_path instead of O(n) linear scan per file.
    let manifest_index: HashMap<&str, &ManifestEntry> =
        old_manifest.entries.iter().map(|e| (e.rel_path.as_str(), e)).collect();
    let old_index = if paths.index_path.is_file() {
        Some(crate::ml::load_index(&paths.index_path)?)
    } else {
        None
    };

    // ── Phase B: sequential manifest check + cached embed export ────────────
    // Kept sequential to avoid concurrent usearch index access.
    // Fires FileFinished immediately for files whose embedding is reused.
    let t_wall = Instant::now();
    let done_counter = Arc::new(AtomicU64::new(0));
    let mut cached_vecs: Vec<Option<Vec<f32>>> = Vec::with_capacity(total);
    for entry in &all_entries {
        let cached = manifest_index.get(entry.rel_path.as_str()).and_then(|me| {
            if me.sha256 == entry.sha256 && me.mtime_secs == entry.mtime_secs {
                // For PDF pages: also verify the rendered PNG still exists on disk.
                if entry.pdf_page.is_some() && !entry.embed_path.is_file() {
                    return None;
                }
                let idx = old_index.as_ref()?;
                let mut v = Vec::new();
                idx.export(me.embedding_id, &mut v)
                   .ok()
                   .filter(|&n| n > 0 && v.len() == crate::ml::CLIP_EMBED_DIM)
                   .map(|_| v)
            } else {
                None
            }
        });
        if cached.is_some() {
            let done = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
            let wall_ms = t_wall.elapsed().as_millis() as u64;
            let avg_ms = if done > 1 { wall_ms as f64 / done as f64 } else { 0.0 };
            let remaining = total.saturating_sub(done as usize);
            let eta_ms = (avg_ms * remaining as f64) as u64;
            if let Some(ref cb) = opts.on_progress {
                cb(IndexProgress::FileFinished {
                    index: (done - 1) as usize,
                    total,
                    path: entry.embed_path.clone(),
                    elapsed_ms: 0,
                    avg_ms,
                    eta_ms,
                });
            }
        }
        cached_vecs.push(cached);
    }

    // ── Phase C: parallel embed for uncached files ───────────────────────────
    let to_embed: Vec<(usize, PathBuf)> = all_entries
        .iter()
        .zip(cached_vecs.iter())
        .enumerate()
        .filter(|(_, (_, c))| c.is_none())
        .map(|(i, (entry, _))| (i, entry.embed_path.clone()))
        .collect();

    let dc = done_counter.clone();
    let new_embed_results: Vec<Result<(usize, Vec<f32>), CoreError>> = to_embed
        .par_iter()
        .map(|(idx, path)| {
            if let Some(ref cb) = opts.on_progress {
                cb(IndexProgress::FileStarted { index: *idx, total, path: path.clone() });
            }
            let t0 = Instant::now();
            let vec = backend.embed_image(path)?;
            let elapsed_ms = t0.elapsed().as_millis() as u64;
            let done = dc.fetch_add(1, Ordering::Relaxed) + 1;
            let wall_ms = t_wall.elapsed().as_millis() as u64;
            let avg_ms = wall_ms as f64 / done as f64;
            let remaining = total.saturating_sub(done as usize);
            let eta_ms = (avg_ms * remaining as f64) as u64;
            if let Some(ref cb) = opts.on_progress {
                cb(IndexProgress::FileFinished {
                    index: (done - 1) as usize,
                    total,
                    path: path.clone(),
                    elapsed_ms,
                    avg_ms,
                    eta_ms,
                });
            }
            Ok((*idx, vec))
        })
        .collect();

    for res in new_embed_results {
        let (idx, vec) = res?;
        cached_vecs[idx] = Some(vec);
    }

    let process_ms = t_process.elapsed().as_millis() as u64;

    // ── Phase D: tags, SQLite writes, manifest collection ───────────────────
    // Sequential: TagsDb requires &mut self; manifest ordering must be stable.
    let mut embeddings: Vec<(PathBuf, Vec<f32>)> = Vec::with_capacity(total);
    let mut manifest_entries: Vec<ManifestEntry> = Vec::with_capacity(total);

    for (i, (entry, opt_vec)) in all_entries.iter().zip(cached_vecs.iter()).enumerate() {
        let vec = opt_vec.as_ref().expect("all embeddings resolved in phase C");

        if let Some(ref mut db) = tags_db_opt {
            if vocab_cache.is_none() {
                let vocab = load_vocab(opts.tags_file.as_deref())?;
                let vocab_vecs = embed_vocab(backend.as_ref(), &vocab)?;
                let vhash = compute_vocab_hash(&vocab);
                db.set_meta("vocab_hash", &vhash)?;
                db.set_meta("tag_threshold", &opts.tag_threshold.to_string())?;
                db.set_meta("tags_updated_at", &Utc::now().to_rfc3339())?;
                vocab_cache = Some((vocab, vocab_vecs));
            }
            let (vocab, vocab_vecs) = vocab_cache.as_ref().expect("vocabulary initialised");
            let assigned = tags_for_image(vec, vocab, vocab_vecs, opts.tag_threshold);
            db.upsert_image_tags(&entry.rel_path, &assigned)?;
        }

        let id = i as u64;
        embeddings.push((entry.embed_path.clone(), vec.clone()));
        manifest_entries.push(ManifestEntry {
            rel_path: entry.rel_path.clone(),
            mtime_secs: entry.mtime_secs,
            sha256: entry.sha256.clone(),
            embedding_id: id,
            pdf_source: entry.pdf_source.clone(),
            pdf_page: entry.pdf_page,
        });
    }

    if let Some(ref cb) = opts.on_progress {
        cb(IndexProgress::Finishing);
    }

    let t_finalize = Instant::now();
    let index = build_index(&embeddings)?;
    save_index(&index, &paths.index_path)?;

    let manifest = Manifest {
        entries: manifest_entries,
    };
    manifest.save(&paths.manifest_path)?;

    let config = ProjectConfig {
        source_path: source.clone(),
        model: opts.model,
        created_at: Utc::now().to_rfc3339(),
    };
    let cfg_json = serde_json::to_string_pretty(&config)?;
    fs::write(&paths.config_path, cfg_json)?;

    let artefacts = project_artefact_paths(paths, opts.skip_tagging);

    let finalize_ms = t_finalize.elapsed().as_millis() as u64;
    let total_ms = t_all.elapsed().as_millis() as u64;

    if let Some(ref cb) = opts.on_progress {
        cb(IndexProgress::Done);
    }

    Ok(IndexOutcome {
        file_count: total,
        artefacts,
        scan_ms,
        process_ms,
        finalize_ms,
        total_ms,
    })
}

/// Paths to project files that exist on disk (after an index), in stable order.
///
/// When `skip_tagging` is true, `tags.sqlite` is not included even if present.
pub fn project_artefact_paths(paths: &ProjectPaths, skip_tagging: bool) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for p in [&paths.index_path, &paths.manifest_path, &paths.config_path] {
        if p.is_file() {
            out.push(p.to_path_buf());
        }
    }
    if !skip_tagging && paths.tags_db_path.is_file() {
        out.push(paths.tags_db_path.clone());
    }
    out
}

/// Remove the project directory for `source` if it exists.
pub fn clean_project(source: &Path) -> Result<(), CoreError> {
    let paths = ProjectPaths::for_source(source)?;
    if paths.project_dir.is_dir() {
        fs::remove_dir_all(&paths.project_dir)?;
    }
    Ok(())
}

/// Delete the entire project root directory (all indices). Use with care.
pub fn clean_all_projects() -> Result<(), CoreError> {
    let root = crate::project::default_project_root()?;
    if root.is_dir() {
        fs::remove_dir_all(&root)?;
    }
    Ok(())
}

/// Return set of relative paths present in `source` scan (for structure checks in tests).
pub fn list_rel_paths_for_test(
    source: &Path,
    recursive: bool,
) -> Result<HashSet<String>, CoreError> {
    let source = source
        .canonicalize()
        .map_err(|e| CoreError::msg(e.to_string()))?;
    let collected: Vec<PathBuf> = if recursive {
        WalkDir::new(&source)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .filter(|p| is_image(p))
            .collect()
    } else {
        fs::read_dir(&source)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file() && is_image(p))
            .collect()
    };
    collected
        .into_iter()
        .map(|p| rel_path_posix(&source, &p))
        .collect()
}

#[cfg(test)]
mod progress_tests {
    use std::fs;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use image::{ImageBuffer, Rgb};
    use crate::ml::MockEmbeddingBackend;
    use serial_test::serial;
    use tempfile::tempdir;

    use super::{index_folder, IndexOptions, IndexProgress, ProgressCallback};
    use crate::project::ProjectPaths;

    /// Override `TWINPICS_PROJECT_DIR` for a test, restored in [`Drop`].
    struct ProjectDirEnv;

    impl ProjectDirEnv {
        fn new(root: &Path) -> Self {
            std::env::set_var("TWINPICS_PROJECT_DIR", root.as_os_str());
            Self
        }
    }

    impl Drop for ProjectDirEnv {
        fn drop(&mut self) {
            std::env::remove_var("TWINPICS_PROJECT_DIR");
        }
    }

    fn write_png(path: &Path) {
        let img: ImageBuffer<Rgb<u8>, _> =
            ImageBuffer::from_fn(2, 2, |_, _| Rgb([10, 20, 30]));
        img.save(path).expect("save");
    }

    #[test]
    #[serial]
    fn index_emits_progress_and_artefacts() {
        let tmp = tempdir().expect("temp");
        let _env = ProjectDirEnv::new(tmp.path());
        let n = 3_usize;
        let root = tmp.path().join("src_photos");
        fs::create_dir_all(&root).expect("mkdir");
        for i in 0..n {
            write_png(&root.join(format!("a{i}.png")));
        }
        let source = root.canonicalize().expect("canonicalize");
        let paths = ProjectPaths::for_source(&source).expect("paths");
        let acc: Arc<Mutex<Vec<IndexProgress>>> = Arc::new(Mutex::new(Vec::new()));
        let a2 = acc.clone();
        let cb: ProgressCallback = Arc::new(move |p| {
            a2.lock().expect("lock").push(p);
        });
        let outcome = index_folder(
            &source,
            Arc::new(MockEmbeddingBackend::new()),
            &paths,
            IndexOptions {
                on_progress: Some(cb),
                ..Default::default()
            },
        )
        .expect("index");
        assert_eq!(outcome.file_count, n);
        for p in [
            &paths.index_path,
            &paths.manifest_path,
            &paths.config_path,
            &paths.tags_db_path,
        ] {
            assert!(p.is_file(), "missing {p:?}");
        }
        assert_eq!(outcome.artefacts.len(), 4);
        assert!(outcome.total_ms >= outcome.scan_ms);
        assert!(outcome.total_ms >= outcome.process_ms);
        assert!(outcome.total_ms >= outcome.finalize_ms);

        let v = acc.lock().expect("lock");
        // Scanning fires first.
        assert!(matches!(&v[0], IndexProgress::Scanning), "expected Scanning at [0]");
        // Some number of FileDiscovered events follow (one per image found).
        let discovered_pos = v
            .iter()
            .position(|e| matches!(e, IndexProgress::Discovered { .. }))
            .expect("Discovered event missing");
        // All events between Scanning and Discovered must be FileDiscovered.
        for ev in &v[1..discovered_pos] {
            assert!(
                matches!(ev, IndexProgress::FileDiscovered { .. }),
                "unexpected event between Scanning and Discovered: {ev:?}"
            );
        }
        assert!(
            matches!(&v[discovered_pos], IndexProgress::Discovered { total, .. } if *total == n)
        );
        // After Discovered: n FileFinished events (parallel order, not necessarily sequential).
        let finished_count = v
            .iter()
            .filter(|e| matches!(e, IndexProgress::FileFinished { .. }))
            .count();
        assert_eq!(finished_count, n, "expected {n} FileFinished events");
        // Finishing then Done close the sequence.
        let finishing_pos = v
            .iter()
            .position(|e| matches!(e, IndexProgress::Finishing))
            .expect("Finishing event missing");
        assert!(
            matches!(&v[finishing_pos + 1], IndexProgress::Done),
            "expected Done after Finishing"
        );
    }
}
