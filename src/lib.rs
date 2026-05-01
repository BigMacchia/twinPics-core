//! twinpics core: indexing, search orchestration, and project layout.
//!
//! The `twinpics_cli` binary lives in `crates/cli`. Library users typically call
//! [`index_folder`], [`search_project_text`], and [`search_project_image`] with an
//! [`std::sync::Arc`] to an [`ml::EmbeddingBackend`]. Indexing optionally writes CLIP-derived
//! tags to [`tags::TagsDb`] (`tags.sqlite` in the project directory).

#![warn(missing_docs)]

pub mod db;
pub mod error;
pub mod indexer;
#[cfg(feature = "pdf")]
pub mod pdf;
pub mod manifest;
/// CLIP embeddings and vector index (merged from the former `mll` crate).
pub mod ml;
pub mod project;
pub mod search;
pub mod tags;

pub use error::CoreError;
pub use indexer::{
    clean_all_projects, clean_project, index_folder, list_rel_paths_for_test, project_artefact_paths,
    IndexOptions, IndexOutcome, IndexProgress, ProgressCallback,
};
pub use manifest::{Manifest, ManifestEntry};
pub use project::{
    default_project_root, find_project_for_source, list_projects, resolve_source_from_index_hint,
    ProjectConfig, ProjectPaths,
};
pub use search::{search_project_image, search_project_text, SearchHit, SearchParams};
pub use tags::{
    list_tag_counts, list_tag_counts_with_images, rel_path_to_abs, TagCountWithImages, TagsDb,
    DEFAULT_VOCAB,
};

/// Build [`ProjectPaths`] for a source directory (see [`ProjectPaths::for_source`]).
pub fn project_paths_for_source(source: &std::path::Path) -> Result<ProjectPaths, CoreError> {
    ProjectPaths::for_source(source)
}
