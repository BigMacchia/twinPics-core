//! usearch index wrapper: build, persist, search.

use std::path::Path;

use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

use super::backend::{MllError, SearchResult};
use super::CLIP_EMBED_DIM;

fn index_options() -> IndexOptions {
    IndexOptions {
        dimensions: CLIP_EMBED_DIM,
        metric: MetricKind::Cos,
        quantization: ScalarKind::F32,
        ..Default::default()
    }
}

/// Build a usearch index from embeddings. Keys are `0..n` in order.
pub fn build_index(embeddings: &[(std::path::PathBuf, Vec<f32>)]) -> Result<Index, MllError> {
    let index = Index::new(&index_options()).map_err(|e| MllError::Index(e.to_string()))?;
    if !embeddings.is_empty() {
        index
            .reserve(embeddings.len())
            .map_err(|e| MllError::Index(e.to_string()))?;
    }
    for (i, (_path, vec)) in embeddings.iter().enumerate() {
        if vec.len() != CLIP_EMBED_DIM {
            return Err(MllError::Invalid(format!(
                "expected embedding dim {}, got {}",
                CLIP_EMBED_DIM,
                vec.len()
            )));
        }
        let key = i as u64;
        index
            .add(key, vec.as_slice())
            .map_err(|e| MllError::Index(e.to_string()))?;
    }
    Ok(index)
}

/// Persist `index` to `path` (typically `index.usearch`).
pub fn save_index(index: &Index, path: &Path) -> Result<(), MllError> {
    let p = path
        .to_str()
        .ok_or_else(|| MllError::Invalid("non-utf8 path".into()))?;
    index.save(p).map_err(|e| MllError::Index(e.to_string()))
}

/// Load a usearch index from disk.
pub fn load_index(path: &Path) -> Result<Index, MllError> {
    let index = Index::new(&index_options()).map_err(|e| MllError::Index(e.to_string()))?;
    let p = path
        .to_str()
        .ok_or_else(|| MllError::Invalid("non-utf8 path".into()))?;
    index.load(p).map_err(|e| MllError::Index(e.to_string()))?;
    Ok(index)
}

/// Cosine distance from usearch is converted to similarity: `1 - distance`.
pub fn search_index(
    index: &Index,
    query: &[f32],
    top_k: usize,
    min_score: f32,
) -> Result<Vec<SearchResult>, MllError> {
    if query.len() != CLIP_EMBED_DIM {
        return Err(MllError::Invalid(format!(
            "expected query dim {}, got {}",
            CLIP_EMBED_DIM,
            query.len()
        )));
    }
    if top_k == 0 {
        return Ok(vec![]);
    }
    let n = index.size().max(1);
    let matches = index
        .exact_search(query, n)
        .map_err(|e| MllError::Index(e.to_string()))?;
    let mut out = Vec::new();
    for (key, distance) in matches.keys.iter().zip(matches.distances.iter()) {
        let score = (1.0 - *distance).clamp(0.0, 1.0);
        if score >= min_score {
            out.push(SearchResult { key: *key, score });
        }
    }
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(top_k);
    Ok(out)
}
