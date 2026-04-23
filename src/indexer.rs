//! Walk a folder, embed images, build usearch index and project files.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

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

/// Supported image extensions (lowercase, no dot).
pub const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "tiff"];

/// Progress reported while indexing; used by CLI, tests, and Tauri.
#[derive(Debug, Clone)]
pub enum IndexProgress {
    /// About to walk the directory tree to find images (no `total` yet).
    Scanning,
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
    let mut collected: Vec<PathBuf> = if opts.recursive {
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

    collected.sort();
    if collected.len() < 20 {
        // Spec asks for 20+ images in some tests; we still allow smaller sets for dev.
    }

    let scan_ms = t_scan.elapsed().as_millis() as u64;
    let total = collected.len();
    if let Some(ref cb) = opts.on_progress {
        cb(IndexProgress::Discovered { total, scan_ms });
    }
    let mut total_elapsed_ms: u128 = 0;

    let t_process = Instant::now();
    let old_manifest = Manifest::load_or_empty(&paths.manifest_path)?;
    let old_index = if paths.index_path.is_file() {
        Some(crate::ml::load_index(&paths.index_path)?)
    } else {
        None
    };

    let mut embeddings: Vec<(PathBuf, Vec<f32>)> = Vec::with_capacity(collected.len());
    let mut manifest_entries: Vec<ManifestEntry> = Vec::with_capacity(collected.len());

    for (i, path) in collected.iter().enumerate() {
        if let Some(ref cb) = opts.on_progress {
            cb(IndexProgress::FileStarted {
                index: i,
                total,
                path: path.clone(),
            });
        }
        let t0 = Instant::now();

        let rel = rel_path_posix(&source, path)?;
        let hash = hash_file(path)?;
        let mtime = mtime_secs(path)?;

        let skip_embed = old_manifest.find_by_rel_path(&rel).and_then(|e| {
            if e.sha256 == hash && e.mtime_secs == mtime {
                Some(e.embedding_id)
            } else {
                None
            }
        });

        let vec = if let (Some(key), Some(ref idx)) = (skip_embed, &old_index) {
            let mut v = Vec::new();
            let n = idx
                .export(key, &mut v)
                .map_err(|e| CoreError::msg(format!("index export: {e}")))?;
            if n == 0 || v.len() != crate::ml::CLIP_EMBED_DIM {
                backend.embed_image(path)?
            } else {
                v
            }
        } else {
            backend.embed_image(path)?
        };

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
            let assigned = tags_for_image(&vec, vocab, vocab_vecs, opts.tag_threshold);
            db.upsert_image_tags(&rel, &assigned)?;
        }

        let id = embeddings.len() as u64;
        embeddings.push((path.clone(), vec));
        manifest_entries.push(ManifestEntry {
            rel_path: rel,
            mtime_secs: mtime,
            sha256: hash,
            embedding_id: id,
        });

        if let Some(ref cb) = opts.on_progress {
            let elapsed_ms = t0.elapsed().as_millis() as u64;
            total_elapsed_ms += elapsed_ms as u128;
            let done = i + 1;
            let avg_ms = total_elapsed_ms as f64 / done as f64;
            let remaining = total.saturating_sub(done);
            let eta_ms = (avg_ms * remaining as f64) as u64;
            cb(IndexProgress::FileFinished {
                index: i,
                total,
                path: path.clone(),
                elapsed_ms,
                avg_ms,
                eta_ms,
            });
        }
    }

    let process_ms = t_process.elapsed().as_millis() as u64;

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
        let mut i = 0;
        assert!(matches!(&v[i], IndexProgress::Scanning));
        i += 1;
        assert!(matches!(&v[i], IndexProgress::Discovered { total, .. } if *total == n));
        i += 1;
        for idx in 0..n {
            assert!(matches!(
                &v[i],
                IndexProgress::FileStarted { index, total, .. } if *index == idx && *total == n
            ));
            i += 1;
            assert!(matches!(
                &v[i],
                IndexProgress::FileFinished { index, total, .. } if *index == idx && *total == n
            ));
            i += 1;
        }
        assert!(matches!(&v[i], IndexProgress::Finishing));
        i += 1;
        assert!(matches!(&v[i], IndexProgress::Done));
        i += 1;
        assert_eq!(i, v.len());
    }
}
