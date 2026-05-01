//! Dominant-color extraction (k-means in CIE L*a*b* space) and color-palette search.
//!
//! No extra crates — uses the `image` crate already in the dependency tree.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::CoreError;
use crate::manifest::Manifest;
use crate::project::ProjectPaths;
use crate::search::SearchParams;
use crate::tags::TagsDb;

// ── public types ────────────────────────────────────────────────────────────

/// One dominant color: sRGB 0-255 bytes and its share of image pixels (0.0–1.0).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominantColor {
    /// Red channel (0–255).
    pub r: u8,
    /// Green channel (0–255).
    pub g: u8,
    /// Blue channel (0–255).
    pub b: u8,
    /// Fraction of pixels belonging to this cluster (0.0–1.0).
    pub pct: f32,
}

/// Color-search hit: resolved path + overall score + the image's extracted palette.
#[derive(Debug, Clone)]
pub struct ColorHit {
    /// 1-based rank in the result list.
    pub rank: usize,
    /// Palette-match score (0.0–1.0).
    pub score: f32,
    /// Absolute path to the matched file.
    pub path: std::path::PathBuf,
    /// 0-based page index for PDF hits; `None` for regular images.
    pub pdf_page: Option<u32>,
    /// Dominant colors extracted from this image.
    pub palette: Vec<DominantColor>,
}

// ── color math ──────────────────────────────────────────────────────────────

/// sRGB byte (0–255) → CIE L*a*b* (L in 0–100, a/b roughly ±128).
pub(crate) fn srgb_to_lab(r: u8, g: u8, b: u8) -> [f32; 3] {
    let lin = |c: u8| -> f32 {
        let f = c as f32 / 255.0;
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055_f32).powf(2.4)
        }
    };
    let lr = lin(r);
    let lg = lin(g);
    let lb = lin(b);

    // Linear RGB → XYZ (D65)
    let x = lr * 0.4124564 + lg * 0.3575761 + lb * 0.1804375;
    let y = lr * 0.2126729 + lg * 0.7151522 + lb * 0.0721750;
    let z = lr * 0.0193339 + lg * 0.1191920 + lb * 0.9503041;

    // XYZ → L*a*b*
    let f = |t: f32| -> f32 {
        if t > 0.008856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };
    let fx = f(x / 0.950456);
    let fy = f(y);
    let fz = f(z / 1.089058);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIE L*a*b* → sRGB bytes (clamped to 0-255).
fn lab_to_srgb(lab: [f32; 3]) -> (u8, u8, u8) {
    let [l, a, b] = lab;
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    let inv = |t: f32| -> f32 {
        let t3 = t * t * t;
        if t3 > 0.008856 {
            t3
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };
    let x = inv(fx) * 0.950456;
    let y = inv(fy);
    let z = inv(fz) * 1.089058;

    let r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
    let g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;

    let enc = |c: f32| -> u8 {
        let c = c.clamp(0.0, 1.0);
        let s = if c <= 0.0031308 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (s * 255.0).round() as u8
    };
    (enc(r_lin), enc(g_lin), enc(b_lin))
}

fn lab_sq_dist(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dl = a[0] - b[0];
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    dl * dl + da * da + db * db
}

/// CIE76 deltaE — perceptual color distance (~0 identical, ~100 very different).
pub fn delta_e76(a: [f32; 3], b: [f32; 3]) -> f32 {
    lab_sq_dist(&a, &b).sqrt()
}

// ── k-means ──────────────────────────────────────────────────────────────────

fn kmeans(pixels: &[[f32; 3]], k: usize, iters: usize) -> Vec<([f32; 3], usize)> {
    if pixels.is_empty() || k == 0 {
        return Vec::new();
    }
    let k = k.min(pixels.len());
    let step = pixels.len() / k;
    let mut centroids: Vec<[f32; 3]> = (0..k).map(|i| pixels[i * step]).collect();
    let mut assignments = vec![0usize; pixels.len()];

    for _ in 0..iters {
        let mut changed = false;
        for (i, px) in pixels.iter().enumerate() {
            let nearest = centroids
                .iter()
                .enumerate()
                .map(|(j, c)| (j, lab_sq_dist(px, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
                .map(|(j, _)| j)
                .unwrap_or(0);
            if assignments[i] != nearest {
                changed = true;
                assignments[i] = nearest;
            }
        }
        if !changed {
            break;
        }
        let mut sums = vec![[0.0f32; 3]; k];
        let mut counts = vec![0usize; k];
        for (i, px) in pixels.iter().enumerate() {
            let j = assignments[i];
            sums[j][0] += px[0];
            sums[j][1] += px[1];
            sums[j][2] += px[2];
            counts[j] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                centroids[j] = [
                    sums[j][0] / counts[j] as f32,
                    sums[j][1] / counts[j] as f32,
                    sums[j][2] / counts[j] as f32,
                ];
            }
        }
    }
    let mut counts = vec![0usize; k];
    for &j in &assignments {
        counts[j] += 1;
    }
    centroids
        .into_iter()
        .zip(counts)
        .filter(|(_, c)| *c > 0)
        .collect()
}

// ── public API ───────────────────────────────────────────────────────────────

/// Extract up to 6 dominant colors from an image.
///
/// Downsamples to 64×64 first, so this is typically <5 ms per image.
pub fn extract_dominant_colors(path: &Path) -> Result<Vec<DominantColor>, CoreError> {
    let img = image::open(path)
        .map_err(|e| CoreError::msg(format!("color extraction: {e}")))?;
    let thumb = img.thumbnail(64, 64);
    let rgb8 = thumb.to_rgb8();

    let pixels: Vec<[f32; 3]> = rgb8
        .pixels()
        .map(|p| srgb_to_lab(p[0], p[1], p[2]))
        .collect();

    if pixels.is_empty() {
        return Ok(Vec::new());
    }

    let clusters = kmeans(&pixels, 6, 20);
    let total = pixels.len() as f32;

    let mut colors: Vec<DominantColor> = clusters
        .iter()
        .map(|(lab, count)| {
            let (r, g, b) = lab_to_srgb(*lab);
            DominantColor {
                r,
                g,
                b,
                pct: *count as f32 / total,
            }
        })
        .collect();

    colors.sort_by(|a, b| b.pct.partial_cmp(&a.pct).unwrap_or(Ordering::Equal));
    Ok(colors)
}

/// Score a query palette against a stored palette.
///
/// For each query color, finds the best-matching palette entry weighted by coverage,
/// then averages across query colors. Result in [0, 1].
///
/// `tolerance` is the max CIE76 deltaE for a full match (~25 = same-hue family).
pub fn color_match_score(
    query: &[[u8; 3]],
    palette: &[DominantColor],
    tolerance: f32,
) -> f32 {
    if query.is_empty() || palette.is_empty() {
        return 0.0;
    }
    let mut total = 0.0f32;
    for &[qr, qg, qb] in query {
        let qlab = srgb_to_lab(qr, qg, qb);
        let best = palette
            .iter()
            .map(|dc| {
                let dlab = srgb_to_lab(dc.r, dc.g, dc.b);
                let de = delta_e76(qlab, dlab);
                let strength = (1.0 - de / tolerance).max(0.0);
                dc.pct * strength
            })
            .fold(0.0f32, f32::max);
        total += best;
    }
    total / query.len() as f32
}

/// Search the index at `paths` by color palette.
///
/// `query_colors`: RGB triples the user wants to find.
/// `tolerance`: max CIE76 deltaE for a "match" (25 ≈ same-hue family, 10 ≈ close match).
pub fn search_project_colors(
    paths: &ProjectPaths,
    query_colors: &[[u8; 3]],
    tolerance: f32,
    params: SearchParams,
) -> Result<Vec<ColorHit>, CoreError> {
    if !paths.tags_db_path.is_file() {
        return Ok(Vec::new());
    }
    let db = TagsDb::open_for_color_search(&paths.tags_db_path)?;
    let palettes = db.load_all_colors()?;

    if palettes.is_empty() || query_colors.is_empty() {
        return Ok(Vec::new());
    }

    let manifest = Manifest::load_or_empty(&paths.manifest_path)?;
    let rel_to_entry: HashMap<&str, &crate::manifest::ManifestEntry> =
        manifest.entries.iter().map(|e| (e.rel_path.as_str(), e)).collect();

    let mut scored: Vec<(String, f32, Vec<DominantColor>)> = palettes
        .into_iter()
        .filter_map(|(rel_path, palette)| {
            let score = color_match_score(query_colors, &palette, tolerance);
            if score >= params.min_score {
                Some((rel_path, score, palette))
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.truncate(params.output_max);

    let hits = scored
        .into_iter()
        .enumerate()
        .map(|(i, (rel_path, score, palette))| {
            let abs_path =
                if let Some(entry) = rel_to_entry.get(rel_path.as_str()) {
                    if let Some(ref pdf_rel) = entry.pdf_source {
                        let rel = std::path::PathBuf::from(
                            pdf_rel.replace('/', std::path::MAIN_SEPARATOR_STR),
                        );
                        paths.source_path.join(rel)
                    } else {
                        let rel = std::path::PathBuf::from(
                            rel_path.replace('/', std::path::MAIN_SEPARATOR_STR),
                        );
                        paths.source_path.join(rel)
                    }
                } else {
                    let rel = std::path::PathBuf::from(
                        rel_path.replace('/', std::path::MAIN_SEPARATOR_STR),
                    );
                    paths.source_path.join(rel)
                };
            let pdf_page = rel_to_entry
                .get(rel_path.as_str())
                .and_then(|e| e.pdf_page);
            ColorHit {
                rank: i + 1,
                score,
                path: abs_path,
                pdf_page,
                palette,
            }
        })
        .collect();

    Ok(hits)
}
