//! Embedding backend trait and shared types.

use std::path::Path;

use thiserror::Error;

/// Errors returned by the `mll` crate.
#[derive(Debug, Error)]
pub enum MllError {
    /// I/O or file system failure.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// Candle / tensor error.
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    /// Tokenizer error.
    #[error("tokenizers: {0}")]
    Tokenizers(#[from] tokenizers::Error),
    /// Image decode error.
    #[error("image: {0}")]
    Image(String),
    /// HF Hub / model download error.
    #[error("hf hub: {0}")]
    HfHub(String),
    /// Vector index / usearch error.
    #[error("index: {0}")]
    Index(String),
    /// Invalid input or state.
    #[error("invalid: {0}")]
    Invalid(String),
    /// CLIP weight directory is missing one or both files.
    #[error(
        "CLIP weights incomplete in {0} (expected model.safetensors and tokenizer.json)"
    )]
    ClipWeightsNotFound(String),
    /// No candidate directory contained CLIP weights.
    #[error("CLIP weights not found (need model.safetensors and tokenizer.json). Tried:\n{0}")]
    ClipWeightsNotFoundAnywhere(String),
}

/// One neighbour returned from vector search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Key stored with the vector (embedding id in twinpics).
    pub key: u64,
    /// Cosine similarity in \[0, 1\] (higher is more similar).
    pub score: f32,
}

/// Pluggable embedding backend so `core` stays implementation-agnostic.
pub trait EmbeddingBackend: Send + Sync {
    /// Embed an image file at `path` as a 512-dimensional L2-normalised vector.
    fn embed_image(&self, path: &Path) -> Result<Vec<f32>, MllError>;

    /// Embed one or more text tags/phrases (joined with spaces) as a 512-dimensional L2-normalised vector.
    fn embed_text(&self, tags: &[&str]) -> Result<Vec<f32>, MllError>;
}
