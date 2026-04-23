//! Search orchestration: load index, query, map keys to paths.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ml::{load_index, search_index, EmbeddingBackend};

use crate::error::CoreError;
use crate::manifest::Manifest;
use crate::project::ProjectPaths;

/// One search hit with rank and resolved path.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchHit {
    /// 1-based rank.
    pub rank: usize,
    /// Cosine similarity score.
    pub score: f32,
    /// Absolute path to the image file.
    pub path: PathBuf,
}

/// Parameters for `search_project`.
#[derive(Debug, Clone)]
pub struct SearchParams {
    /// Minimum cosine similarity.
    pub min_score: f32,
    /// Maximum number of results.
    pub output_max: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            min_score: 0.2,
            output_max: 10,
        }
    }
}

/// Run text search against the project for `source_root`.
pub fn search_project_text(
    paths: &ProjectPaths,
    backend: Arc<dyn EmbeddingBackend>,
    tags: &[String],
    params: SearchParams,
) -> Result<Vec<SearchHit>, CoreError> {
    let refs: Vec<&str> = tags.iter().map(String::as_str).collect();
    let query = backend.embed_text(&refs)?;
    search_project_with_query(paths, &query, params)
}

/// Run image-to-image search.
pub fn search_project_image(
    paths: &ProjectPaths,
    backend: Arc<dyn EmbeddingBackend>,
    image_path: &Path,
    params: SearchParams,
) -> Result<Vec<SearchHit>, CoreError> {
    let query = backend.embed_image(image_path)?;
    search_project_with_query(paths, &query, params)
}

fn search_project_with_query(
    paths: &ProjectPaths,
    query: &[f32],
    params: SearchParams,
) -> Result<Vec<SearchHit>, CoreError> {
    let manifest = Manifest::load_or_empty(&paths.manifest_path)?;
    let index = load_index(&paths.index_path)?;

    let mut id_to_path: HashMap<u64, PathBuf> = HashMap::new();
    for e in &manifest.entries {
        let rel = std::path::PathBuf::from(e.rel_path.replace('/', std::path::MAIN_SEPARATOR_STR));
        let abs = paths.source_path.join(rel);
        id_to_path.insert(e.embedding_id, abs);
    }

    let results = search_index(&index, query, params.output_max, params.min_score)?;

    let mut hits = Vec::new();
    for (i, r) in results.iter().enumerate() {
        let path = id_to_path
            .get(&r.key)
            .cloned()
            .ok_or_else(|| CoreError::msg(format!("unknown embedding id {}", r.key)))?;
        hits.push(SearchHit {
            rank: i + 1,
            score: r.score,
            path,
        });
    }
    Ok(hits)
}
