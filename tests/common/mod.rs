//! Shared helpers for integration tests.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use image::{ImageBuffer, Rgb};
use mll::MockEmbeddingBackend;
use tempfile::TempDir;

/// Temp project root + env var set for twinpics.
pub struct TestEnv {
    pub temp: TempDir,
}

impl TestEnv {
    pub fn new() -> Self {
        let temp = tempfile::tempdir().expect("tempdir");
        std::env::set_var("TWINPICS_PROJECT_DIR", temp.path().as_os_str());
        std::env::set_var("TWINPICS_CONFIRM", "yes");
        Self { temp }
    }

    pub fn path(&self) -> &Path {
        self.temp.path()
    }
}

impl Drop for TestEnv {
    fn drop(&mut self) {
        std::env::remove_var("TWINPICS_PROJECT_DIR");
        // keep TWINPICS_CONFIRM for other tests if needed
    }
}

/// Mock backend wrapped for indexing tests.
#[allow(dead_code)] // Used only by integration tests that import `common`; not every test binary uses it.
pub fn mock_backend() -> Arc<dyn mll::EmbeddingBackend> {
    Arc::new(MockEmbeddingBackend::new())
}

/// Write a solid-colour PNG at `path` (size 4x4).
pub fn write_solid_png(path: &Path, rgb: [u8; 3]) {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(4, 4, |_, _| Rgb(rgb));
    img.save(path).expect("save png");
}

/// Build a fixture tree: `root/a/`, `root/b/`, `root/c/` with 7+7+7 PNGs each (21 total).
#[allow(dead_code)] // Not every integration test binary uses this helper.
pub fn build_fixture_tree(root: &Path) -> Vec<PathBuf> {
    let dirs = ["a", "b", "c"];
    let mut paths = Vec::new();
    for d in dirs {
        let sub = root.join(d);
        std::fs::create_dir_all(&sub).expect("mkdir");
        for i in 0..7 {
            let p = sub.join(format!("img_{i}.png"));
            let base = d.as_bytes()[0];
            let shade = base.wrapping_add((i * 30) as u8);
            write_solid_png(&p, [shade, shade.wrapping_add(10), shade.wrapping_add(20)]);
            paths.push(p);
        }
    }
    paths
}
