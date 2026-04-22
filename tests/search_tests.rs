mod common;

use std::fs;

use serial_test::serial;
use twinpics_core::{
    index_folder, project_paths_for_source, search_project_image, search_project_text,
    IndexOptions, ProjectPaths, SearchParams,
};

use common::{build_fixture_tree, mock_backend, TestEnv};

fn index_small_fixture() -> (TestEnv, std::path::PathBuf, ProjectPaths) {
    let env = TestEnv::new();
    let root = env.path().join("src");
    fs::create_dir_all(&root).unwrap();
    build_fixture_tree(&root);
    let paths = project_paths_for_source(&root).unwrap();
    index_folder(&root, mock_backend(), &paths, IndexOptions::default()).unwrap();
    (env, root, paths)
}

#[test]
#[serial]
fn test_search_min_score_filter() {
    let (_e, _root, paths) = index_small_fixture();
    let hits = search_project_text(
        &paths,
        mock_backend(),
        &["anything".to_string()],
        SearchParams {
            min_score: 0.99,
            output_max: 50,
        },
    )
    .unwrap();
    assert!(
        hits.iter().all(|h| h.score >= 0.99) || hits.is_empty(),
        "no scores below threshold"
    );
}

#[test]
#[serial]
fn test_search_output_max() {
    let (_e, _root, paths) = index_small_fixture();
    let hits = search_project_text(
        &paths,
        mock_backend(),
        &["query".to_string()],
        SearchParams {
            min_score: 0.0,
            output_max: 3,
        },
    )
    .unwrap();
    assert!(hits.len() <= 3);
}

#[test]
#[serial]
fn test_search_by_image() {
    let (_e, root, paths) = index_small_fixture();
    let query_img = root.join("a").join("img_0.png");
    let hits = search_project_image(
        &paths,
        mock_backend(),
        &query_img,
        SearchParams {
            min_score: 0.0,
            output_max: 5,
        },
    )
    .unwrap();
    assert!(!hits.is_empty());
    let top = &hits[0];
    assert_eq!(top.path, query_img.canonicalize().unwrap());
    assert!(top.score >= 0.95);
}
