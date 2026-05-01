//! Sidecar manifest: file path → embedding id, hash, mtime.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::CoreError;

/// One indexed file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ManifestEntry {
    /// Path relative to the source root, POSIX-style for regular images.
    /// For PDF page entries: absolute path to the rendered PNG in `pdf_renders/`.
    pub rel_path: String,
    /// File modification time (seconds since UNIX epoch).
    /// For PDF pages: mtime of the source PDF.
    pub mtime_secs: i64,
    /// SHA-256 hex digest of file contents.
    /// For PDF pages: digest of the source PDF.
    pub sha256: String,
    /// Key in the usearch index (0..n-1 after rebuild).
    pub embedding_id: u64,
    /// POSIX relative path of the source PDF within the indexed folder.
    /// `None` for regular image entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pdf_source: Option<String>,
    /// 0-based page index within the source PDF.
    /// `None` for regular image entries.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pdf_page: Option<u32>,
}

/// Manifest file contents.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    /// Indexed files in arbitrary order; ids reference usearch keys.
    pub entries: Vec<ManifestEntry>,
}

impl Manifest {
    /// Load from `path` or empty manifest if missing.
    pub fn load_or_empty(path: &Path) -> Result<Self, CoreError> {
        if !path.is_file() {
            return Ok(Self::default());
        }
        let data = fs::read_to_string(path)?;
        Ok(serde_json::from_str(&data)?)
    }

    /// Save to `path`.
    pub fn save(&self, path: &Path) -> Result<(), CoreError> {
        let s = serde_json::to_string_pretty(self)?;
        fs::write(path, s)?;
        Ok(())
    }

    /// Find entry by relative path.
    pub fn find_by_rel_path(&self, rel: &str) -> Option<&ManifestEntry> {
        self.entries.iter().find(|e| e.rel_path == rel)
    }
}

/// Relative path string from `source_root` to `file` (POSIX `/`).
pub fn rel_path_posix(source_root: &Path, file: &Path) -> Result<String, CoreError> {
    let rel = file.strip_prefix(source_root).map_err(|_| {
        CoreError::msg(format!(
            "file {} is not under source root {}",
            file.display(),
            source_root.display()
        ))
    })?;
    Ok(rel
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/"))
}
