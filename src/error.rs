//! Core error type.

use thiserror::Error;

/// Errors from indexing, search, and project I/O.
#[derive(Debug, Error)]
pub enum CoreError {
    /// Wrapped ML layer error.
    #[error("mll: {0}")]
    Mll(#[from] mll::MllError),
    /// I/O error.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// JSON (de)serialization.
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    /// Invalid user input or inconsistent state.
    #[error("{0}")]
    Msg(String),
}

impl CoreError {
    /// Constructs a message error.
    pub fn msg(s: impl Into<String>) -> Self {
        CoreError::Msg(s.into())
    }
}
