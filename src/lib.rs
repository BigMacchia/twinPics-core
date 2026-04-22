//! twinpics core: indexing, search orchestration, and project layout.
//!
//! The `twinpics_cli` binary lives in `src/bin/twinpics_cli.rs`. Library users typically call
//! [`index_folder`], [`search_project_text`], and [`search_project_image`] with an
//! [`std::sync::Arc`] to an [`mll::EmbeddingBackend`].

#![warn(missing_docs)]

pub mod error;
pub mod indexer;
pub mod manifest;
pub mod project;
pub mod search;

pub use error::CoreError;
pub use indexer::{
    clean_all_projects, clean_project, index_folder, list_rel_paths_for_test, IndexOptions,
};
pub use manifest::{Manifest, ManifestEntry};
pub use project::{
    default_project_root, find_project_for_source, list_projects, resolve_source_from_index_hint,
    ProjectConfig, ProjectPaths,
};
pub use search::{search_project_image, search_project_text, SearchHit, SearchParams};

/// Build [`ProjectPaths`] for a source directory (see [`ProjectPaths::for_source`]).
pub fn project_paths_for_source(source: &std::path::Path) -> Result<ProjectPaths, CoreError> {
    ProjectPaths::for_source(source)
}
