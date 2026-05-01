//! Project directory layout: `~/.twinpics/projects/<sanitised_source>/`.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

const ENV_PROJECT_DIR: &str = "TWINPICS_PROJECT_DIR";

/// Root directory for all twinpics project folders (override with env var `TWINPICS_PROJECT_DIR`).
pub fn default_project_root() -> Result<PathBuf, CoreError> {
    if let Ok(p) = env::var(ENV_PROJECT_DIR) {
        return Ok(PathBuf::from(p));
    }
    let home =
        dirs::home_dir().ok_or_else(|| CoreError::msg("could not resolve home directory"))?;
    Ok(home.join(".twinpics").join("projects"))
}

/// Sanitise a source path into a single directory name under the project root.
pub fn sanitise_source_path(source: &Path) -> String {
    let s = source.to_string_lossy();
    s.chars()
        .map(|c| match c {
            ':' | '\\' | '/' => '_',
            c if c.is_ascii_alphanumeric() => c,
            '_' | '-' | '.' => c,
            _ => '_',
        })
        .collect()
}

/// Paths to `index.usearch`, `manifest.json`, and `config.json` for a source tree.
#[derive(Debug, Clone)]
pub struct ProjectPaths {
    /// Root of this project (e.g. `~/.twinpics/projects/c_photos_italy`).
    pub project_dir: PathBuf,
    /// Vector index file.
    pub index_path: PathBuf,
    /// Sidecar manifest mapping files to embedding ids.
    pub manifest_path: PathBuf,
    /// Index metadata (model, creation time, source path).
    pub config_path: PathBuf,
    /// SQLite DB of per-image CLIP tags (`tags.sqlite`).
    pub tags_db_path: PathBuf,
    /// Original source folder this project indexes.
    pub source_path: PathBuf,
    /// Cache directory for rendered PDF page images: `<project_dir>/pdf_renders/`.
    pub pdf_renders_path: PathBuf,
}

impl ProjectPaths {
    /// Resolve project paths for a `source` directory (canonicalised).
    pub fn for_source(source: &Path) -> Result<Self, CoreError> {
        let source = source
            .canonicalize()
            .map_err(|e| CoreError::msg(format!("source path: {e}")))?;
        let root = default_project_root()?;
        let name = sanitise_source_path(&source);
        let project_dir = root.join(&name);
        Ok(Self {
            index_path: project_dir.join("index.usearch"),
            manifest_path: project_dir.join("manifest.json"),
            config_path: project_dir.join("config.json"),
            tags_db_path: project_dir.join("tags.sqlite"),
            pdf_renders_path: project_dir.join("pdf_renders"),
            project_dir,
            source_path: source,
        })
    }

    /// Ensure the project directory exists.
    pub fn ensure_dir(&self) -> Result<(), CoreError> {
        fs::create_dir_all(&self.project_dir)?;
        Ok(())
    }
}

/// Metadata stored in `config.json`.
#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Absolute normalised path to the indexed source folder.
    pub source_path: PathBuf,
    /// Model identifier used for embeddings.
    pub model: String,
    /// RFC3339 timestamp when the index was created or last fully rebuilt.
    pub created_at: String,
}

/// Discover project paths for a source folder if `config.json` exists.
pub fn find_project_for_source(source: &Path) -> Result<Option<ProjectPaths>, CoreError> {
    let paths = ProjectPaths::for_source(source)?;
    if paths.config_path.is_file() {
        Ok(Some(paths))
    } else {
        Ok(None)
    }
}

/// List all projects under the project root (subdirs containing `config.json`).
pub fn list_projects() -> Result<Vec<(ProjectPaths, ProjectConfig)>, CoreError> {
    let root = default_project_root()?;
    if !root.is_dir() {
        return Ok(vec![]);
    }
    let mut out = Vec::new();
    for ent in fs::read_dir(&root)? {
        let ent = ent?;
        let p = ent.path();
        if !p.is_dir() {
            continue;
        }
        let config_path = p.join("config.json");
        if !config_path.is_file() {
            continue;
        }
        let config: ProjectConfig = serde_json::from_slice(&fs::read(&config_path)?)?;
        let source_path = config.source_path.clone();
        let paths = ProjectPaths {
            project_dir: p.clone(),
            index_path: p.join("index.usearch"),
            manifest_path: p.join("manifest.json"),
            config_path,
            tags_db_path: p.join("tags.sqlite"),
            pdf_renders_path: p.join("pdf_renders"),
            source_path,
        };
        out.push((paths, config));
    }
    out.sort_by(|a, b| a.0.source_path.cmp(&b.0.source_path));
    Ok(out)
}

/// Resolve `source` folder from optional `--index` hint: if `None`, walk `start` ancestors to find a matching project.
pub fn resolve_source_from_index_hint(
    hint: Option<&Path>,
    start: &Path,
) -> Result<PathBuf, CoreError> {
    if let Some(h) = hint {
        let canon = h
            .canonicalize()
            .map_err(|e| CoreError::msg(format!("index path: {e}")))?;
        if find_project_for_source(&canon)?.is_some() {
            return Ok(canon);
        }
        return Err(CoreError::msg(format!(
            "no index found for source {}",
            canon.display()
        )));
    }
    let mut cur = start.to_path_buf();
    loop {
        if let Ok(canon) = cur.canonicalize() {
            if find_project_for_source(&canon)?.is_some() {
                return Ok(canon);
            }
        }
        if !cur.pop() {
            break;
        }
    }
    Err(CoreError::msg(
        "could not auto-detect index from current directory; pass --index <SOURCE_FOLDER>",
    ))
}
