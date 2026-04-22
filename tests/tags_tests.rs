mod common;

use std::fs;
use std::path::Path;
use std::sync::Arc;

use assert_cmd::Command;
use mll::{EmbeddingBackend, MllError};
use predicates::str::contains;
use serial_test::serial;
use twinpics_core::{index_folder, list_tag_counts, project_paths_for_source, IndexOptions, TagsDb};

use common::{write_solid_png, TestEnv};

/// Maps filenames / vocabulary terms to orthonormal axes so cosine scores are 0 or 1.
struct AxisBackend;

fn unit_axis(idx: usize) -> Vec<f32> {
    let mut v = vec![0f32; mll::CLIP_EMBED_DIM];
    if idx < v.len() {
        v[idx] = 1.0;
    }
    v
}

impl EmbeddingBackend for AxisBackend {
    fn embed_image(&self, path: &Path) -> Result<Vec<f32>, MllError> {
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let idx = if name.contains("red") {
            0
        } else if name.contains("green") {
            1
        } else if name.contains("blue") {
            2
        } else {
            99
        };
        Ok(unit_axis(idx))
    }

    fn embed_text(&self, tags: &[&str]) -> Result<Vec<f32>, MllError> {
        let term = tags.join(" ").to_ascii_lowercase();
        let idx = match term.as_str() {
            "red" => 0,
            "green" => 1,
            "blue" => 2,
            "cat" => 3,
            _ => 99,
        };
        Ok(unit_axis(idx))
    }
}

fn axis_backend() -> Arc<dyn mll::EmbeddingBackend> {
    Arc::new(AxisBackend)
}

fn write_vocab(path: &Path) {
    fs::write(path, "red\ngreen\nblue\ncat\n").expect("vocab");
}

#[test]
#[serial]
fn test_index_populates_tags() {
    let _env = TestEnv::new();
    let root = _env.path().join("tagged");
    fs::create_dir_all(&root).unwrap();
    write_solid_png(&root.join("scene_red.png"), [255, 0, 0]);
    write_solid_png(&root.join("scene_green.png"), [0, 255, 0]);
    write_solid_png(&root.join("scene_blue.png"), [0, 0, 255]);

    let vocab_path = root.join("vocab.txt");
    write_vocab(&vocab_path);

    let paths = project_paths_for_source(&root).unwrap();
    index_folder(
        &root,
        axis_backend(),
        &paths,
        IndexOptions {
            tags_file: Some(vocab_path),
            tag_threshold: 0.99,
            ..Default::default()
        },
    )
    .unwrap();

    let db = TagsDb::open_existing(&paths.tags_db_path).unwrap();
    assert_eq!(db.image_count().unwrap(), 3);

    let counts: std::collections::HashMap<String, usize> =
        list_tag_counts(&paths.tags_db_path).unwrap().into_iter().collect();
    assert!(counts.get("red").copied().unwrap_or(0) >= 1);
    assert!(counts.get("green").copied().unwrap_or(0) >= 1);
    assert!(counts.get("blue").copied().unwrap_or(0) >= 1);
    assert_eq!(counts.get("cat").copied().unwrap_or(0), 0);
}

#[test]
#[serial]
fn test_list_tags_cli_table() {
    let _env = TestEnv::new();
    let root = _env.path().join("cli_tags");
    fs::create_dir_all(&root).unwrap();
    write_solid_png(&root.join("pic_red.png"), [1, 0, 0]);
    let vocab_path = root.join("vocab.txt");
    write_vocab(&vocab_path);
    let paths = project_paths_for_source(&root).unwrap();
    index_folder(
        &root,
        axis_backend(),
        &paths,
        IndexOptions {
            tags_file: Some(vocab_path),
            tag_threshold: 0.99,
            ..Default::default()
        },
    )
    .unwrap();

    Command::cargo_bin("twinpics_cli")
        .unwrap()
        .args(["list-tags", "--index"])
        .arg(&root)
        .assert()
        .success()
        .stdout(contains("red"));
}

#[test]
#[serial]
fn test_list_tags_cli_json() {
    let _env = TestEnv::new();
    let root = _env.path().join("cli_tags_json");
    fs::create_dir_all(&root).unwrap();
    write_solid_png(&root.join("x_red.png"), [2, 0, 0]);
    let vocab_path = root.join("vocab.txt");
    write_vocab(&vocab_path);
    let paths = project_paths_for_source(&root).unwrap();
    index_folder(
        &root,
        axis_backend(),
        &paths,
        IndexOptions {
            tags_file: Some(vocab_path),
            tag_threshold: 0.99,
            ..Default::default()
        },
    )
    .unwrap();

    let out = Command::cargo_bin("twinpics_cli")
        .unwrap()
        .args(["list-tags", "--index"])
        .arg(&root)
        .args(["--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).expect("json");
    assert!(v.is_array());
    let arr = v.as_array().unwrap();
    assert!(!arr.is_empty());
    assert!(arr[0].get("tag").is_some());
    assert!(arr[0].get("count").is_some());
}

#[test]
#[serial]
fn test_list_tags_cli_json_include_paths() {
    let _env = TestEnv::new();
    let root = _env.path().join("cli_tags_paths");
    fs::create_dir_all(&root).unwrap();
    write_solid_png(&root.join("shot_red.png"), [5, 0, 0]);
    let vocab_path = root.join("vocab.txt");
    write_vocab(&vocab_path);
    let paths = project_paths_for_source(&root).unwrap();
    index_folder(
        &root,
        axis_backend(),
        &paths,
        IndexOptions {
            tags_file: Some(vocab_path),
            tag_threshold: 0.99,
            ..Default::default()
        },
    )
    .unwrap();

    let out = Command::cargo_bin("twinpics_cli")
        .unwrap()
        .args(["list-tags", "--index"])
        .arg(&root)
        .args(["--format", "json", "--include-image-paths"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).expect("json");
    let arr = v.as_array().expect("array");
    let red = arr
        .iter()
        .find(|o| o.get("tag").and_then(|t| t.as_str()) == Some("red"))
        .expect("red tag");
    let imgs = red.get("images").and_then(|x| x.as_array()).expect("images");
    assert_eq!(imgs.len(), 1);
    assert_eq!(
        imgs[0].get("rel_path").and_then(|x| x.as_str()),
        Some("shot_red.png")
    );
    assert!(imgs[0].get("path").and_then(|x| x.as_str()).is_some());
    assert!(imgs[0].get("score").and_then(|x| x.as_f64()).is_some());
}

#[test]
#[serial]
fn test_list_tags_include_paths_global_before_subcommand() {
    let _env = TestEnv::new();
    let root = _env.path().join("cli_tags_global_flag");
    fs::create_dir_all(&root).unwrap();
    write_solid_png(&root.join("g_red.png"), [9, 0, 0]);
    let vocab_path = root.join("vocab.txt");
    write_vocab(&vocab_path);
    let paths = project_paths_for_source(&root).unwrap();
    index_folder(
        &root,
        axis_backend(),
        &paths,
        IndexOptions {
            tags_file: Some(vocab_path),
            tag_threshold: 0.99,
            ..Default::default()
        },
    )
    .unwrap();

    let out = Command::cargo_bin("twinpics_cli")
        .unwrap()
        .arg("--include-image-paths")
        .args(["list-tags", "--index"])
        .arg(&root)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).expect("json");
    assert!(v.get(0).and_then(|x| x.get("images")).is_some());
}

#[test]
#[serial]
fn test_list_tags_include_paths_without_format_implies_json() {
    let _env = TestEnv::new();
    let root = _env.path().join("cli_tags_paths_no_format");
    fs::create_dir_all(&root).unwrap();
    write_solid_png(&root.join("solo_red.png"), [8, 0, 0]);
    let vocab_path = root.join("vocab.txt");
    write_vocab(&vocab_path);
    let paths = project_paths_for_source(&root).unwrap();
    index_folder(
        &root,
        axis_backend(),
        &paths,
        IndexOptions {
            tags_file: Some(vocab_path),
            tag_threshold: 0.99,
            ..Default::default()
        },
    )
    .unwrap();

    let out = Command::cargo_bin("twinpics_cli")
        .unwrap()
        .args(["list-tags", "--index"])
        .arg(&root)
        .args(["--include-image-paths"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let s = String::from_utf8(out).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).expect("json");
    let arr = v.as_array().expect("array");
    let red = arr
        .iter()
        .find(|o| o.get("tag").and_then(|t| t.as_str()) == Some("red"))
        .expect("red tag");
    assert!(red.get("images").and_then(|x| x.as_array()).is_some());
}

#[test]
#[serial]
fn test_list_tags_include_paths_with_explicit_table_fails() {
    let _env = TestEnv::new();
    let root = _env.path().join("cli_tags_table_paths");
    fs::create_dir_all(&root).unwrap();
    write_solid_png(&root.join("one.png"), [1, 1, 1]);
    let vocab_path = root.join("vocab.txt");
    write_vocab(&vocab_path);
    let paths = project_paths_for_source(&root).unwrap();
    index_folder(
        &root,
        axis_backend(),
        &paths,
        IndexOptions {
            tags_file: Some(vocab_path),
            tag_threshold: 0.99,
            ..Default::default()
        },
    )
    .unwrap();

    Command::cargo_bin("twinpics_cli")
        .unwrap()
        .args(["list-tags", "--index"])
        .arg(&root)
        .args(["--include-image-paths", "--format", "table"])
        .assert()
        .failure()
        .stderr(contains("include-image-paths"));
}

#[test]
#[serial]
fn test_list_tags_min_count() {
    let _env = TestEnv::new();
    let root = _env.path().join("min_count");
    fs::create_dir_all(&root).unwrap();
    write_solid_png(&root.join("a_red.png"), [3, 0, 0]);
    write_solid_png(&root.join("b_red.png"), [4, 0, 0]);
    write_solid_png(&root.join("c_green.png"), [0, 5, 0]);
    let vocab_path = root.join("vocab.txt");
    write_vocab(&vocab_path);
    let paths = project_paths_for_source(&root).unwrap();
    index_folder(
        &root,
        axis_backend(),
        &paths,
        IndexOptions {
            tags_file: Some(vocab_path),
            tag_threshold: 0.99,
            ..Default::default()
        },
    )
    .unwrap();

    let rows = list_tag_counts(&paths.tags_db_path).unwrap();
    let filtered: Vec<_> = rows.into_iter().filter(|(_, c)| *c >= 2).collect();
    assert!(
        filtered.iter().any(|(t, _)| t == "red"),
        "red should appear twice"
    );
    assert!(
        !filtered.iter().any(|(t, _)| t == "green"),
        "green should be filtered at min_count 2"
    );
}

#[test]
#[serial]
fn test_list_tags_missing_db() {
    let _env = TestEnv::new();
    let root = _env.path().join("no_tags_db");
    fs::create_dir_all(&root).unwrap();
    write_solid_png(&root.join("solo.png"), [9, 9, 9]);
    let paths = project_paths_for_source(&root).unwrap();
    index_folder(
        &root,
        axis_backend(),
        &paths,
        IndexOptions {
            skip_tagging: true,
            ..Default::default()
        },
    )
    .unwrap();

    assert!(!paths.tags_db_path.is_file());

    Command::cargo_bin("twinpics_cli")
        .unwrap()
        .args(["list-tags", "--index"])
        .arg(&root)
        .assert()
        .failure()
        .stderr(contains("no tags database"));
}
