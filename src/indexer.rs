//! Walk a folder, embed images, build usearch index and project files.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use mll::{build_index, save_index, EmbeddingBackend};
use sha2::{Digest, Sha256};
use walkdir::WalkDir;

use crate::error::CoreError;
use crate::manifest::{rel_path_posix, Manifest, ManifestEntry};
use crate::project::{ProjectConfig, ProjectPaths};

/// Supported image extensions (lowercase, no dot).
pub const IMAGE_EXTS: &[&str] = &["jpg", "jpeg", "png", "webp", "bmp", "tiff"];

/// Options for `index_folder`.
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Recurse into subdirectories.
    pub recursive: bool,
    /// Model label stored in `config.json`.
    pub model: String,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            recursive: true,
            model: "clip-vit-base-patch32".to_string(),
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
) -> Result<(), CoreError> {
    let source = source
        .canonicalize()
        .map_err(|e| CoreError::msg(format!("source path: {e}")))?;

    paths.ensure_dir()?;

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

    let old_manifest = Manifest::load_or_empty(&paths.manifest_path)?;
    let old_index = if paths.index_path.is_file() {
        Some(mll::load_index(&paths.index_path)?)
    } else {
        None
    };

    let mut embeddings: Vec<(PathBuf, Vec<f32>)> = Vec::with_capacity(collected.len());
    let mut manifest_entries: Vec<ManifestEntry> = Vec::with_capacity(collected.len());

    for path in &collected {
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
            if n == 0 || v.len() != mll::CLIP_EMBED_DIM {
                backend.embed_image(path)?
            } else {
                v
            }
        } else {
            backend.embed_image(path)?
        };

        let id = embeddings.len() as u64;
        embeddings.push((path.clone(), vec));
        manifest_entries.push(ManifestEntry {
            rel_path: rel,
            mtime_secs: mtime,
            sha256: hash,
            embedding_id: id,
        });
    }

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

    Ok(())
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
