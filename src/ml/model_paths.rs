//! Resolve directories containing CLIP `model.safetensors` and `tokenizer.json`.

use std::path::{Path, PathBuf};

use super::backend::MllError;

/// Environment variable: directory that **directly** contains `model.safetensors` and `tokenizer.json`.
pub const ENV_CLIP_MODEL_DIR: &str = "TWINPICS_CLIP_MODEL_DIR";

/// Subdirectory under `models/` (next to the executable) or under `~/.twinpics/models/`.
pub const CLIP_MODEL_SUBDIR: &str = "clip-vit-base-patch32";

const MODEL_FILE: &str = "model.safetensors";
const TOKENIZER_FILE: &str = "tokenizer.json";

/// Ordered search locations for bundled/offline CLIP weights.
pub fn clip_model_dir_candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Ok(p) = std::env::var(ENV_CLIP_MODEL_DIR) {
        if !p.is_empty() {
            v.push(PathBuf::from(p));
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            v.push(
                parent
                    .join("models")
                    .join(CLIP_MODEL_SUBDIR),
            );
        }
    }
    if let Some(home) = dirs::home_dir() {
        v.push(
            home.join(".twinpics")
                .join("models")
                .join(CLIP_MODEL_SUBDIR),
        );
    }
    v
}

/// `~/.twinpics/models/<CLIP_MODEL_SUBDIR>/` — target for optional Hub download (`model-download` feature).
#[cfg(feature = "model-download")]
pub(crate) fn default_clip_home_model_dir() -> Option<PathBuf> {
    Some(
        dirs::home_dir()?.join(".twinpics").join("models").join(CLIP_MODEL_SUBDIR),
    )
}

/// First candidate directory that contains both weight files.
pub fn first_existing_clip_model_dir(candidates: &[PathBuf]) -> Option<PathBuf> {
    for d in candidates {
        if clip_weight_files_exist(d) {
            return Some(d.clone());
        }
    }
    None
}

/// Returns `(model.safetensors, tokenizer.json)` if both exist under `dir`.
pub fn clip_weight_paths_in(dir: &Path) -> Result<(PathBuf, PathBuf), MllError> {
    let model = dir.join(MODEL_FILE);
    let tokenizer = dir.join(TOKENIZER_FILE);
    if model.is_file() && tokenizer.is_file() {
        return Ok((model, tokenizer));
    }
    Err(MllError::ClipWeightsNotFound(
        dir.display().to_string(),
    ))
}

fn clip_weight_files_exist(dir: &Path) -> bool {
    dir.join(MODEL_FILE).is_file() && dir.join(TOKENIZER_FILE).is_file()
}

/// Resolve a directory with CLIP files, or return an error listing candidates tried.
pub fn resolve_clip_model_dir() -> Result<PathBuf, MllError> {
    let candidates = clip_model_dir_candidates();
    if let Some(dir) = first_existing_clip_model_dir(&candidates) {
        return Ok(dir);
    }
    let listing = candidates
        .iter()
        .map(|p| p.display().to_string())
        .collect::<Vec<_>>()
        .join("\n");
    Err(MllError::ClipWeightsNotFoundAnywhere(listing))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::Mutex;

    use tempfile::tempdir;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn first_existing_prefers_first_complete_dir() {
        let tmp = tempdir().unwrap();
        let d1 = tmp.path().join("a");
        let d2 = tmp.path().join("b");
        fs::create_dir_all(&d1).unwrap();
        fs::create_dir_all(&d2).unwrap();
        fs::write(d1.join(MODEL_FILE), "").unwrap();
        fs::write(d1.join(TOKENIZER_FILE), "").unwrap();
        fs::write(d2.join(MODEL_FILE), "").unwrap();
        fs::write(d2.join(TOKENIZER_FILE), "").unwrap();
        let cands = vec![d1.clone(), d2.clone()];
        assert_eq!(first_existing_clip_model_dir(&cands), Some(d1));
    }

    #[test]
    fn first_existing_skips_incomplete() {
        let tmp = tempdir().unwrap();
        let d1 = tmp.path().join("a");
        let d2 = tmp.path().join("b");
        fs::create_dir_all(&d1).unwrap();
        fs::create_dir_all(&d2).unwrap();
        fs::write(d1.join(MODEL_FILE), "").unwrap();
        fs::write(d2.join(MODEL_FILE), "").unwrap();
        fs::write(d2.join(TOKENIZER_FILE), "").unwrap();
        let cands = vec![d1, d2.clone()];
        assert_eq!(first_existing_clip_model_dir(&cands), Some(d2));
    }

    #[test]
    fn env_var_is_first_candidate() {
        let _g = ENV_LOCK.lock().unwrap();
        let tmp = tempdir().unwrap();
        let p = tmp.path().to_string_lossy().to_string();
        std::env::set_var(ENV_CLIP_MODEL_DIR, &p);
        let c = clip_model_dir_candidates();
        std::env::remove_var(ENV_CLIP_MODEL_DIR);
        assert_eq!(c.first().map(|x| x.as_path()), Some(tmp.path()));
    }
}
