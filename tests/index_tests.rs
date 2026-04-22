mod common;

use std::fs;

use serial_test::serial;
use twinpics_core::indexer::list_rel_paths_for_test;
use twinpics_core::{index_folder, project_paths_for_source, IndexOptions, Manifest};

use common::{build_fixture_tree, mock_backend, TestEnv};

#[test]
#[serial]
fn test_index_creation() {
    let _env = TestEnv::new();
    let root = _env.path().join("photos");
    fs::create_dir_all(&root).unwrap();
    assert_eq!(build_fixture_tree(&root).len(), 21);

    let paths = project_paths_for_source(&root).unwrap();
    index_folder(&root, mock_backend(), &paths, IndexOptions::default()).unwrap();

    assert!(paths.index_path.is_file(), "index.usearch missing");
    assert!(paths.manifest_path.is_file(), "manifest.json missing");
    let m = Manifest::load_or_empty(&paths.manifest_path).unwrap();
    assert_eq!(m.entries.len(), 21);

    assert!(root.join("a").is_dir());
    assert!(root.join("b").is_dir());
    assert!(root.join("c").is_dir());

    let rels: std::collections::HashSet<String> = list_rel_paths_for_test(&root, true).unwrap();
    assert_eq!(rels.len(), 21);
}

#[test]
#[serial]
fn test_index_recursive_false() {
    let _env = TestEnv::new();
    let root = _env.path().join("flat");
    fs::create_dir_all(&root).unwrap();
    common::write_solid_png(&root.join("top1.png"), [1, 2, 3]);
    common::write_solid_png(&root.join("top2.png"), [4, 5, 6]);
    let sub = root.join("nested");
    fs::create_dir_all(&sub).unwrap();
    common::write_solid_png(&sub.join("deep.png"), [7, 8, 9]);

    let paths = project_paths_for_source(&root).unwrap();
    index_folder(
        &root,
        mock_backend(),
        &paths,
        IndexOptions {
            recursive: false,
            ..Default::default()
        },
    )
    .unwrap();

    let m = Manifest::load_or_empty(&paths.manifest_path).unwrap();
    assert_eq!(m.entries.len(), 2);
    assert!(
        m.entries.iter().all(|e| !e.rel_path.contains("nested")),
        "subdirectory images must not be indexed"
    );
}
