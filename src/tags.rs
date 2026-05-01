//! Tag vocabulary loading, CLIP similarity tagging, and `tags.sqlite` persistence.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::color::DominantColor;
use crate::ml::EmbeddingBackend;
use rusqlite::Connection;
use sha2::{Digest, Sha256};

use crate::error::CoreError;

/// Tag name, number of matching images, and each image's manifest `rel_path` with CLIP score.
pub type TagCountWithImages = (String, usize, Vec<(String, f32)>);

/// Embedded built-in vocabulary (one tag per non-comment line).
pub const DEFAULT_VOCAB: &str = include_str!("default_vocab.txt");

/// Opens an existing tags DB and returns `(tag, image_count)` rows sorted by count descending.
pub fn list_tag_counts(db_path: &Path) -> Result<Vec<(String, usize)>, CoreError> {
    let db = TagsDb::open_existing(db_path)?;
    db.tag_counts()
}

/// Like [`list_tag_counts`], but each tag includes `(rel_path, score)` pairs for every matching image.
pub fn list_tag_counts_with_images(db_path: &Path) -> Result<Vec<TagCountWithImages>, CoreError> {
    let db = TagsDb::open_existing(db_path)?;
    db.tag_counts_with_images()
}

/// Resolve a manifest-style POSIX `rel_path` under `source_root` to a filesystem path.
pub fn rel_path_to_abs(source_root: &Path, rel_posix: &str) -> PathBuf {
    rel_posix
        .split('/')
        .fold(source_root.to_path_buf(), |p, seg| p.join(seg))
}

/// Stable hash of the vocabulary (sorted lines joined with newlines) for `meta` bookkeeping.
pub fn compute_vocab_hash(vocab: &[String]) -> String {
    let mut v = vocab.to_vec();
    v.sort();
    let joined = v.join("\n");
    format!("{:x}", Sha256::digest(joined.as_bytes()))
}

/// Load vocabulary from `path` (one term per line, `#` comments) or the built-in list if `None`.
pub fn load_vocab(path: Option<&Path>) -> Result<Vec<String>, CoreError> {
    let raw = if let Some(p) = path {
        fs::read_to_string(p).map_err(CoreError::from)?
    } else {
        DEFAULT_VOCAB.to_string()
    };
    parse_vocab_lines(&raw)
}

fn parse_vocab_lines(s: &str) -> Result<Vec<String>, CoreError> {
    let mut out: Vec<String> = s
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(std::string::ToString::to_string)
        .collect();
    out.sort();
    out.dedup();
    if out.is_empty() {
        return Err(CoreError::msg(
            "tag vocabulary is empty after parsing (add non-comment lines)",
        ));
    }
    Ok(out)
}

/// Embed each vocabulary term as its own CLIP text vector (L2-normalised by the backend).
/// Runs in parallel (rayon) since the backend is `Send + Sync`.
pub fn embed_vocab(
    backend: &dyn EmbeddingBackend,
    vocab: &[String],
) -> Result<Vec<Vec<f32>>, CoreError> {
    use rayon::prelude::*;
    let results: Vec<Result<Vec<f32>, CoreError>> = vocab
        .par_iter()
        .map(|term| Ok(backend.embed_text(&[term.as_str()])?))
        .collect();
    results.into_iter().collect()
}

/// Cosine similarity (dot product) between L2-normalised `image_vec` and each vocab vector;
/// keep pairs with score `>= threshold`, sorted by score descending.
pub fn tags_for_image(
    image_vec: &[f32],
    vocab: &[String],
    vocab_vecs: &[Vec<f32>],
    threshold: f32,
) -> Vec<(String, f32)> {
    let dim = crate::ml::CLIP_EMBED_DIM;
    let mut scored: Vec<(String, f32)> = Vec::new();
    for (name, tv) in vocab.iter().zip(vocab_vecs.iter()) {
        if image_vec.len() != dim || tv.len() != dim {
            continue;
        }
        let s: f32 = image_vec.iter().zip(tv.iter()).map(|(a, b)| a * b).sum();
        if s >= threshold {
            scored.push((name.clone(), s));
        }
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

/// SQLite tags database for one project directory.
pub struct TagsDb {
    conn: Connection,
}

impl TagsDb {
    fn apply_schema(conn: &Connection) -> Result<(), CoreError> {
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             CREATE TABLE IF NOT EXISTS images (
               id INTEGER PRIMARY KEY,
               rel_path TEXT UNIQUE NOT NULL
             );
             CREATE TABLE IF NOT EXISTS tags (
               id INTEGER PRIMARY KEY,
               name TEXT UNIQUE NOT NULL
             );
             CREATE TABLE IF NOT EXISTS image_tags (
               image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
               tag_id   INTEGER NOT NULL REFERENCES tags(id)   ON DELETE CASCADE,
               score    REAL    NOT NULL,
               PRIMARY KEY (image_id, tag_id)
             );
             CREATE INDEX IF NOT EXISTS idx_image_tags_tag ON image_tags(tag_id);
             CREATE TABLE IF NOT EXISTS meta (
               k TEXT PRIMARY KEY,
               v TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS colors (
               image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
               r        INTEGER NOT NULL,
               g        INTEGER NOT NULL,
               b        INTEGER NOT NULL,
               pct      REAL    NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_colors_image ON colors(image_id);",
        )?;
        Ok(())
    }

    /// Create or open `tags.sqlite`, ensuring schema exists.
    pub fn open_or_init(path: &Path) -> Result<Self, CoreError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        Self::apply_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Open an existing database file (fails if missing).
    pub fn open_existing(path: &Path) -> Result<Self, CoreError> {
        if !path.is_file() {
            return Err(CoreError::msg(format!(
                "no tags database at {} (run `twinpics_cli index` without --no-tags to create tags.sqlite)",
                path.display()
            )));
        }
        let conn = Connection::open(path)?;
        Self::apply_schema(&conn)?;
        Ok(Self { conn })
    }

    /// Open for color search — same as `open_existing` but returns `Ok(None)` if the file is missing.
    pub fn open_for_color_search(path: &Path) -> Result<Self, CoreError> {
        Self::open_existing(path)
    }

    /// Upsert meta key (e.g. `vocab_hash`, `tag_threshold`).
    pub fn set_meta(&self, key: &str, value: &str) -> Result<(), CoreError> {
        self.conn.execute(
            "INSERT INTO meta (k, v) VALUES (?1, ?2)
             ON CONFLICT(k) DO UPDATE SET v = excluded.v",
            rusqlite::params![key, value],
        )?;
        Ok(())
    }

    /// Replace tag assignments for one image (by manifest `rel_path`).
    pub fn upsert_image_tags(&mut self, rel_path: &str, tags: &[(String, f32)]) -> Result<(), CoreError> {
        let tx = self.conn.transaction()?;
        tx.execute("INSERT OR IGNORE INTO images (rel_path) VALUES (?1)", [rel_path])?;
        let image_id: i64 =
            tx.query_row("SELECT id FROM images WHERE rel_path = ?1", [rel_path], |r| r.get(0))?;
        tx.execute("DELETE FROM image_tags WHERE image_id = ?1", [image_id])?;
        for (name, score) in tags {
            tx.execute("INSERT OR IGNORE INTO tags (name) VALUES (?1)", [name])?;
            let tag_id: i64 =
                tx.query_row("SELECT id FROM tags WHERE name = ?1", [name], |r| r.get(0))?;
            tx.execute(
                "INSERT INTO image_tags (image_id, tag_id, score) VALUES (?1, ?2, ?3)",
                rusqlite::params![image_id, tag_id, score],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Per-tag number of distinct images that have that tag (count &gt; 0 only).
    pub fn tag_counts(&self) -> Result<Vec<(String, usize)>, CoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT t.name, COUNT(it.image_id) AS c
             FROM tags t
             LEFT JOIN image_tags it ON it.tag_id = t.id
             GROUP BY t.id
             HAVING c > 0
             ORDER BY c DESC, t.name ASC",
        )?;
        let rows = stmt.query_map([], |r| {
            let name: String = r.get(0)?;
            let c: i64 = r.get(1)?;
            Ok((name, c as usize))
        })?;
        let mut out = Vec::new();
        for row in rows {
            out.push(row?);
        }
        Ok(out)
    }

    /// Per-tag image list and counts; same tag ordering as [`Self::tag_counts`].
    ///
    /// Within each tag, images are sorted by score descending, then `rel_path` ascending.
    pub fn tag_counts_with_images(&self) -> Result<Vec<TagCountWithImages>, CoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT t.name, i.rel_path, it.score
             FROM image_tags it
             JOIN tags t ON t.id = it.tag_id
             JOIN images i ON i.id = it.image_id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, f32>(2)?,
            ))
        })?;
        let mut map: HashMap<String, Vec<(String, f32)>> = HashMap::new();
        for row in rows {
            let (tag, rel_path, score) = row?;
            map.entry(tag).or_default().push((rel_path, score));
        }
        let mut out: Vec<TagCountWithImages> = map
            .into_iter()
            .map(|(tag, mut images)| {
                images.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.0.cmp(&b.0))
                });
                let c = images.len();
                (tag, c, images)
            })
            .filter(|(_, c, _)| *c > 0)
            .collect();
        out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        Ok(out)
    }

    /// Number of rows in `images`.
    pub fn image_count(&self) -> Result<usize, CoreError> {
        let n: i64 = self.conn.query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    /// Replace dominant-color entries for one image.
    pub fn upsert_image_colors(
        &mut self,
        rel_path: &str,
        colors: &[DominantColor],
    ) -> Result<(), CoreError> {
        let tx = self.conn.transaction()?;
        tx.execute(
            "INSERT OR IGNORE INTO images (rel_path) VALUES (?1)",
            [rel_path],
        )?;
        let image_id: i64 = tx.query_row(
            "SELECT id FROM images WHERE rel_path = ?1",
            [rel_path],
            |r| r.get(0),
        )?;
        tx.execute("DELETE FROM colors WHERE image_id = ?1", [image_id])?;
        for c in colors {
            tx.execute(
                "INSERT INTO colors (image_id, r, g, b, pct) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![image_id, c.r as i64, c.g as i64, c.b as i64, c.pct],
            )?;
        }
        tx.commit()?;
        Ok(())
    }

    /// Load all image palettes as `(rel_path, colors)` pairs.
    pub fn load_all_colors(&self) -> Result<Vec<(String, Vec<DominantColor>)>, CoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT i.rel_path, c.r, c.g, c.b, c.pct
             FROM colors c
             JOIN images i ON i.id = c.image_id
             ORDER BY i.id, c.pct DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, i64>(1)? as u8,
                r.get::<_, i64>(2)? as u8,
                r.get::<_, i64>(3)? as u8,
                r.get::<_, f64>(4)? as f32,
            ))
        })?;
        let mut result: Vec<(String, Vec<DominantColor>)> = Vec::new();
        for row in rows {
            let (rel_path, r, g, b, pct) = row?;
            let dc = DominantColor { r, g, b, pct };
            if let Some(last) = result.last_mut() {
                if last.0 == rel_path {
                    last.1.push(dc);
                    continue;
                }
            }
            result.push((rel_path, vec![dc]));
        }
        Ok(result)
    }

    /// Set of `rel_path` values that already have color data stored.
    pub fn rel_paths_with_colors(&self) -> Result<HashSet<String>, CoreError> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT i.rel_path FROM colors c JOIN images i ON i.id = c.image_id",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        let mut set = HashSet::new();
        for row in rows {
            set.insert(row?);
        }
        Ok(set)
    }
}
