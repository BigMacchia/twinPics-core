mod common;

use std::fs;

use serial_test::serial;
use twinpics_core::{
    clean_all_projects, clean_project, index_folder, list_projects, project_paths_for_source,
    IndexOptions,
};

use common::{build_fixture_tree, mock_backend, TestEnv};

#[test]
#[serial]
fn test_list_index() {
    let _env = TestEnv::new();
    let a = _env.path().join("src_a");
    let b = _env.path().join("src_b");
    fs::create_dir_all(&a).unwrap();
    fs::create_dir_all(&b).unwrap();
    build_fixture_tree(&a);
    build_fixture_tree(&b);

    let pa = project_paths_for_source(&a).unwrap();
    let pb = project_paths_for_source(&b).unwrap();
    index_folder(&a, mock_backend(), &pa, IndexOptions::default()).unwrap();
    index_folder(&b, mock_backend(), &pb, IndexOptions::default()).unwrap();

    let listed = list_projects().unwrap();
    assert_eq!(listed.len(), 2);
    let sources: Vec<_> = listed.iter().map(|(p, _)| p.source_path.clone()).collect();
    assert!(sources.contains(&a.canonicalize().unwrap()));
    assert!(sources.contains(&b.canonicalize().unwrap()));
}

#[test]
#[serial]
fn test_clean_single() {
    let _env = TestEnv::new();
    let a = _env.path().join("only");
    fs::create_dir_all(&a).unwrap();
    build_fixture_tree(&a);
    let pa = project_paths_for_source(&a).unwrap();
    index_folder(&a, mock_backend(), &pa, IndexOptions::default()).unwrap();
    assert!(pa.project_dir.is_dir());

    clean_project(&a.canonicalize().unwrap()).unwrap();
    assert!(!pa.project_dir.exists());
}

#[test]
#[serial]
fn test_clean_all() {
    let _env = TestEnv::new();
    let a = _env.path().join("x");
    let b = _env.path().join("y");
    fs::create_dir_all(&a).unwrap();
    fs::create_dir_all(&b).unwrap();
    build_fixture_tree(&a);
    build_fixture_tree(&b);
    let pa = project_paths_for_source(&a).unwrap();
    let pb = project_paths_for_source(&b).unwrap();
    index_folder(&a, mock_backend(), &pa, IndexOptions::default()).unwrap();
    index_folder(&b, mock_backend(), &pb, IndexOptions::default()).unwrap();

    clean_all_projects().unwrap();
    let root = twinpics_core::default_project_root().unwrap();
    assert!(!root.exists());
}
