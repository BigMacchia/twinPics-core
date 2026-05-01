//! PDF page rendering via pdfium-render. Compiled only with the `pdf` feature.

use std::fs;
use std::path::{Path, PathBuf};

use pdfium_render::prelude::*;

use crate::error::CoreError;

/// Render at 150 DPI — enough for CLIP's 224×224 resize without excessive file size.
const RENDER_DPI: f32 = 150.0;

/// One rendered page from a PDF.
#[derive(Debug, Clone)]
pub struct RenderedPage {
    /// Absolute path to the rendered PNG in the project's `pdf_renders/` dir.
    pub png_path: PathBuf,
    /// 0-based page index within the source PDF.
    pub page_index: u32,
}

/// Load pdfium dynamically.
///
/// Search order:
/// 1. `TWINPICS_PDFIUM_PATH` env var — explicit path to the `.dll` / `.so` / `.dylib`
/// 2. `<workspace>/pdfium/<lib>` — project-local copy next to the `pdfium/` build-scripts dir
/// 3. Directory of the current executable
/// 4. Current working directory
/// 5. System PATH / LD_LIBRARY_PATH (OS default search)
pub fn load_pdfium() -> Result<Pdfium, CoreError> {
    let mut candidates: Vec<PathBuf> = Vec::new();

    // 1. Explicit env var override.
    if let Ok(p) = std::env::var("TWINPICS_PDFIUM_PATH") {
        candidates.push(PathBuf::from(p));
    }

    // 2. Project-local pdfium/ directory: walk up from the exe until we find it.
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(PathBuf::from).unwrap_or_default();
        for _ in 0..6 {
            let candidate = dir.join("pdfium").join(PDFIUM_LIB_NAME);
            if candidate.is_file() {
                candidates.push(candidate);
                break;
            }
            if !dir.pop() { break; }
        }
    }

    // 3. Next to the executable.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(PDFIUM_LIB_NAME));
        }
    }

    // 4. Current working directory.
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(PDFIUM_LIB_NAME));
    }

    for path in &candidates {
        if path.is_file() {
            if let Ok(b) = Pdfium::bind_to_library(path) {
                tracing::debug!("pdfium loaded from {}", path.display());
                return Ok(Pdfium::new(b));
            }
        }
    }

    // 5. OS library search (PATH on Windows, LD_LIBRARY_PATH on Linux).
    Pdfium::bind_to_system_library()
        .map(|b| Pdfium::new(b))
        .map_err(|_| {
            CoreError::msg(format!(
                "{PDFIUM_LIB_NAME} not found.\n\
                 \n\
                 Searched:\n\
                   - TWINPICS_PDFIUM_PATH env var\n\
                   - <workspace>/pdfium/{PDFIUM_LIB_NAME}\n\
                   - next to the binary\n\
                   - current directory\n\
                   - system PATH\n\
                 \n\
                 The file is already in D:\\dev\\ext\\twinPics\\pdfium\\{PDFIUM_LIB_NAME}.\n\
                 Set the env var to use it directly:\n\
                   $env:TWINPICS_PDFIUM_PATH = 'D:\\dev\\ext\\twinPics\\pdfium\\{PDFIUM_LIB_NAME}'"
            ))
        })
}

#[cfg(target_os = "windows")]
const PDFIUM_LIB_NAME: &str = "pdfium.dll";
#[cfg(target_os = "macos")]
const PDFIUM_LIB_NAME: &str = "libpdfium.dylib";
#[cfg(not(any(target_os = "windows", target_os = "macos")))]
const PDFIUM_LIB_NAME: &str = "libpdfium.so";

/// Render all pages of `pdf_path` to PNGs in `renders_dir`.
///
/// Files are named `<sha256>_p<N:04>.png`. Skips existing files when `force` is false.
/// `sha256` is the pre-computed hex digest of the PDF (passed in to avoid double I/O).
pub fn render_pdf_pages(
    pdfium: &Pdfium,
    pdf_path: &Path,
    sha256: &str,
    renders_dir: &Path,
    force: bool,
) -> Result<Vec<RenderedPage>, CoreError> {
    fs::create_dir_all(renders_dir)?;

    let document = pdfium
        .load_pdf_from_file(pdf_path, None)
        .map_err(|e| CoreError::msg(format!("open PDF {}: {e}", pdf_path.display())))?;

    let page_count = document.pages().len() as u32;
    let mut pages = Vec::with_capacity(page_count as usize);

    for page_index in 0..page_count {
        let png_path = renders_dir.join(format!("{sha256}_p{page_index:04}.png"));

        if !force && png_path.is_file() {
            pages.push(RenderedPage { png_path, page_index });
            continue;
        }

        let page = document
            .pages()
            .get(page_index as u16)
            .map_err(|e| CoreError::msg(format!("get page {page_index}: {e}")))?;

        // Convert PDF points (1 pt = 1/72 inch) to pixels at RENDER_DPI.
        let px_w = ((page.width().value * RENDER_DPI / 72.0) as i32).max(1);
        let px_h = ((page.height().value * RENDER_DPI / 72.0) as i32).max(1);

        let config = PdfRenderConfig::new()
            .set_target_width(px_w)
            .set_target_height(px_h);

        let bitmap = page
            .render_with_config(&config)
            .map_err(|e| CoreError::msg(format!("render page {page_index}: {e}")))?;

        bitmap
            .as_image()
            .into_rgb8()
            .save(&png_path)
            .map_err(|e| CoreError::msg(format!("save PNG {}: {e}", png_path.display())))?;

        pages.push(RenderedPage { png_path, page_index });
    }

    Ok(pages)
}

/// Returns true if `path` has a `.pdf` extension (case-insensitive).
pub fn is_pdf(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("pdf"))
        .unwrap_or(false)
}
