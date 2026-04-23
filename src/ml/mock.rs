//! Deterministic mock embeddings for fast tests.

use std::path::Path;

use super::backend::{EmbeddingBackend, MllError};
use super::CLIP_EMBED_DIM;

/// Produces L2-normalised 512-D vectors from a deterministic mix of seed bytes.
pub struct MockEmbeddingBackend;

impl MockEmbeddingBackend {
    /// Creates a new mock backend.
    pub fn new() -> Self {
        Self
    }

    fn vec_from_seed(seed: &[u8]) -> Vec<f32> {
        let mut x = 0xcbf29ce484222325u64;
        for b in seed {
            x ^= *b as u64;
            x = x.wrapping_mul(0x100000001b3);
        }
        let mut out = vec![0f32; CLIP_EMBED_DIM];
        for (i, slot) in out.iter_mut().enumerate() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1 + i as u64);
            let v = (x >> 33) ^ (x >> 17) ^ x;
            *slot = ((v as f32) / (u64::MAX as f32)) * 2.0 - 1.0;
        }
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut out {
                *v /= norm;
            }
        }
        out
    }
}

impl Default for MockEmbeddingBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingBackend for MockEmbeddingBackend {
    fn embed_image(&self, path: &Path) -> Result<Vec<f32>, MllError> {
        let s = path
            .canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .into_owned();
        Ok(Self::vec_from_seed(s.as_bytes()))
    }

    fn embed_text(&self, tags: &[&str]) -> Result<Vec<f32>, MllError> {
        let joined = tags.join("\0");
        Ok(Self::vec_from_seed(joined.as_bytes()))
    }
}
