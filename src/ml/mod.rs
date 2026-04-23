//! CLIP inference and usearch index helpers (merged from the former `mll` crate).
//!
//! ## Main types
//! - [`CandleClipBackend`] — CLIP ViT-B/32 inference on CPU via `candle` (offline weights on disk).
//! - [`EmbeddingBackend`] — trait implemented by [`CandleClipBackend`] and (with the `mock` feature) `MockEmbeddingBackend`.
//! - [`build_index`], [`save_index`], [`load_index`], [`search_index`] — usearch helpers.
//!
//! ## CLIP weights
//! Place `model.safetensors` and `tokenizer.json` under `TWINPICS_CLIP_MODEL_DIR`, or under
//! `models/clip-vit-base-patch32/` next to the executable, or under `~/.twinpics/models/clip-vit-base-patch32/`.
//! Enable the optional `model-download` feature to fetch weights from Hugging Face into the home
//! directory path when missing (requires network).

#![warn(missing_docs)]

mod backend;
mod clip;
mod embd_index;
mod model_paths;

#[cfg(any(test, feature = "mock"))]
mod mock;

pub use backend::{EmbeddingBackend, MllError, SearchResult};
pub use clip::CandleClipBackend;
pub use embd_index::{build_index, load_index, save_index, search_index};

#[cfg(any(test, feature = "mock"))]
pub use mock::MockEmbeddingBackend;

/// CLIP ViT-B/32 projection dimension (embedding size).
pub const CLIP_EMBED_DIM: usize = 512;
