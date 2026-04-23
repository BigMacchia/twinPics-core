//! CLIP-based tests: run with `cargo test -p twinpics_core --features real-clip -- --ignored`.
//! Requires CLIP weights (`model.safetensors` + `tokenizer.json`): set `TWINPICS_CLIP_MODEL_DIR`, or
//! place them under `~/.twinpics/models/clip-vit-base-patch32/`, or let the `real-clip` feature
//! enable `twinpics_core/model-download` to fetch from Hugging Face when missing.

#![cfg(feature = "real-clip")]

mod common;

use std::fs;

use serial_test::serial;
use twinpics_core::ml::{CandleClipBackend, EmbeddingBackend};
use twinpics_core::{
    index_folder, project_paths_for_source, search_project_text,
    IndexOptions, SearchParams,
};

use common::{build_fixture_tree, TestEnv};

fn clip_backend() -> std::sync::Arc<dyn EmbeddingBackend> {
    std::sync::Arc::new(CandleClipBackend::new().expect("clip"))
}

#[test]
#[serial]
#[ignore = "requires CLIP weights on disk or Hub download via real-clip feature (run with --ignored)"]
fn test_search_by_text() {
    let _env = TestEnv::new();
    let root = _env.path().join("clip_text");
    fs::create_dir_all(&root).unwrap();
    build_fixture_tree(&root);
    let paths = project_paths_for_source(&root).unwrap();
    index_folder(&root, clip_backend(), &paths, IndexOptions::default()).unwrap();

    let hits = search_project_text(
        &paths,
        clip_backend(),
        &["solid".to_string(), "red".to_string()],
        SearchParams {
            min_score: 0.2,
            output_max: 5,
        },
    )
    .unwrap();
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.score >= 0.2));
}

#[test]
#[serial]
#[ignore = "requires CLIP weights on disk or Hub download via real-clip feature (run with --ignored)"]
fn test_search_solid_colour() {
    let _env = TestEnv::new();
    let root = _env.path().join("solid");
    fs::create_dir_all(&root).unwrap();
    let red = root.join("red.png");
    let green = root.join("green.png");
    let blue = root.join("blue.png");
    common::write_solid_png(&red, [255, 0, 0]);
    common::write_solid_png(&green, [0, 255, 0]);
    common::write_solid_png(&blue, [0, 0, 255]);

    let paths = project_paths_for_source(&root).unwrap();
    index_folder(&root, clip_backend(), &paths, IndexOptions::default()).unwrap();

    let hits = search_project_text(
        &paths,
        clip_backend(),
        &["solid".to_string(), "red".to_string()],
        SearchParams {
            min_score: 0.0,
            output_max: 5,
        },
    )
    .unwrap();
    let top = &hits[0];
    assert!(top.score > 0.3);
    assert!(
        top.path == red.canonicalize().unwrap(),
        "red image should rank at top: got {:?}",
        top.path
    );
}
