//! twinpics CLI: `index`, `search`, `list-index`, `list-tags`, `clean`.

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Context;
use clap::builder::BoolishValueParser;
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressStyle};
use mll::CandleClipBackend;
use serde::Serialize;

use twinpics_core::{
    clean_all_projects, clean_project, default_project_root, index_folder, list_projects,
    list_tag_counts, list_tag_counts_with_images, manifest::Manifest, project_paths_for_source,
    rel_path_to_abs,
    resolve_source_from_index_hint, search_project_image, search_project_text, IndexOptions,
    IndexProgress, ProgressCallback, SearchHit, SearchParams,
};

/// Format `ms` as `mm:ss` for ETA display.
fn fmt_eta_ms(ms: u64) -> String {
    let s = ms / 1000;
    let m = s / 60;
    let s = s % 60;
    format!("{m:02}:{s:02}")
}

#[derive(Parser)]
#[command(name = "twinpics_cli")]
#[command(about = "Local CLIP-based image search", version)]
struct Cli {
    /// With `list-tags` only: JSON output includes each tag's `images` (`rel_path`, absolute `path`, `score`). May appear before or after `list-tags`. Ignored by other subcommands.
    #[arg(long, global = true)]
    include_image_paths: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build or update an index for a folder of images.
    Index {
        /// Folder containing images to index.
        folder: PathBuf,
        /// Recurse into subdirectories (default: true).
        #[arg(long, default_value = "true", value_parser = BoolishValueParser::new())]
        recursive: bool,
        /// Root directory for index projects (default: ~/.twinpics/projects or TWINPICS_PROJECT_DIR).
        #[arg(long)]
        project_dir: Option<PathBuf>,
        /// Vocabulary file (one tag per line, `#` comments); overrides the built-in vocabulary.
        #[arg(long)]
        tags_file: Option<PathBuf>,
        /// Minimum cosine similarity (dot product) to assign a tag to an image.
        #[arg(long, default_value_t = 0.22)]
        tag_threshold: f32,
        /// Skip building `tags.sqlite` (removes an existing tags DB for this project).
        #[arg(long)]
        no_tags: bool,
    },
    /// Search by query image or text tags.
    Search {
        /// Path to a query image.
        #[arg(long, group = "mode")]
        image: Option<PathBuf>,
        /// Text tags or phrases.
        #[arg(long, group = "mode", num_args = 1..)]
        text: Vec<String>,
        /// Source folder whose index to use (default: auto-detect from CWD).
        #[arg(long)]
        index: Option<PathBuf>,
        #[arg(long, default_value_t = 0.2)]
        min_score: f32,
        #[arg(long, default_value_t = 10)]
        output_max: usize,
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },
    /// List registered indices.
    ListIndex {
        #[arg(long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },
    /// List tags stored for an index (from `tags.sqlite`) with image counts.
    ListTags {
        /// Source folder whose index to use (default: auto-detect from CWD).
        #[arg(long)]
        index: Option<PathBuf>,
        /// Only include tags matched by at least this many images.
        #[arg(long, default_value_t = 1)]
        min_count: usize,
        /// Maximum number of tag rows to print.
        #[arg(long)]
        limit: Option<usize>,
        /// Output format (default: table). With `--include-image-paths`, JSON is used if you omit `--format`.
        #[arg(long, value_enum)]
        format: Option<OutputFormat>,
    },
    /// Remove index data for a source folder, or wipe all indices.
    Clean {
        /// Source folder whose index to delete.
        folder: Option<PathBuf>,
        #[arg(long)]
        all: bool,
        /// Skip the interactive confirmation prompt for `--all`.
        #[arg(long, short = 'y')]
        yes: bool,
    },
}

#[derive(Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
enum OutputFormat {
    #[default]
    Table,
    Json,
    Paths,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();
    let include_image_paths = cli.include_image_paths;
    match cli.command {
        Command::Index {
            folder,
            recursive,
            project_dir,
            tags_file,
            tag_threshold,
            no_tags,
        } => {
            if let Some(p) = project_dir {
                std::env::set_var("TWINPICS_PROJECT_DIR", p.as_os_str());
            }
            let source = folder
                .canonicalize()
                .with_context(|| format!("index folder {}", folder.display()))?;
            let paths = project_paths_for_source(&source)?;
            let backend: Arc<dyn mll::EmbeddingBackend> = Arc::new(CandleClipBackend::new()?);
            let bar: Arc<Mutex<Option<ProgressBar>>> = Arc::new(Mutex::new(None));
            let bar_handle = bar.clone();
            let cb: ProgressCallback = Arc::new(move |ev: IndexProgress| {
                match ev {
                    IndexProgress::Scanning => {
                        eprintln!("Scanning for images…");
                    }
                    IndexProgress::Discovered { total, scan_ms } => {
                        eprintln!("Found {total} image(s) in {scan_ms} ms");
                        if total == 0 {
                            return;
                        }
                        let pb = ProgressBar::new(total as u64);
                        if let Ok(style) = ProgressStyle::with_template(
                            "{bar:40.cyan/blue} {pos}/{len} {msg}",
                        ) {
                            pb.set_style(style);
                        }
                        *bar_handle.lock().expect("pb mutex") = Some(pb);
                    }
                    IndexProgress::FileFinished { avg_ms, eta_ms, .. } => {
                        let g = bar_handle.lock().expect("pb mutex");
                        if let Some(pb) = g.as_ref() {
                            pb.set_message(format!(
                                "{:.0}ms  ETA {}",
                                avg_ms,
                                fmt_eta_ms(eta_ms)
                            ));
                            pb.inc(1);
                        }
                    }
                    IndexProgress::Finishing => {
                        if let Some(pb) = bar_handle.lock().expect("pb mutex").as_ref() {
                            pb.set_message("writing index…".to_string());
                        }
                    }
                    IndexProgress::Done => {
                        if let Some(pb) = bar_handle.lock().expect("pb mutex").take() {
                            pb.finish_and_clear();
                        }
                    }
                    _ => {}
                }
            });
            let outcome = index_folder(
                &source,
                backend,
                &paths,
                IndexOptions {
                    on_progress: Some(cb),
                    recursive,
                    model: "clip-vit-base-patch32".to_string(),
                    tags_file,
                    tag_threshold,
                    skip_tagging: no_tags,
                },
            )?;
            println!(
                "Indexed {} file(s) from {} into {} — done in {}",
                outcome.file_count,
                source.display(),
                paths.project_dir.display(),
                fmt_eta_ms(outcome.total_ms)
            );
            println!(
                "Time: total {} ms (scan {} ms, process {} ms, write {} ms)",
                outcome.total_ms, outcome.scan_ms, outcome.process_ms, outcome.finalize_ms
            );
            println!("Created:");
            for f in &outcome.artefacts {
                println!("  {}", f.display());
            }
        }
        Command::Search {
            image,
            text,
            index,
            min_score,
            output_max,
            format,
        } => {
            let cwd = std::env::current_dir()?;
            let source = resolve_source_from_index_hint(index.as_deref(), &cwd)?;
            let paths = project_paths_for_source(&source)?;
            let backend: Arc<dyn mll::EmbeddingBackend> = Arc::new(CandleClipBackend::new()?);
            let params = SearchParams {
                min_score,
                output_max,
            };
            let hits = if let Some(img) = image {
                search_project_image(&paths, backend, &img, params)?
            } else if !text.is_empty() {
                search_project_text(&paths, backend, &text, params)?
            } else {
                anyhow::bail!("either --image or --text is required");
            };
            print_hits(&hits, format)?;
        }
        Command::ListTags {
            index,
            min_count,
            limit,
            format,
        } => {
            if format == Some(OutputFormat::Paths) {
                anyhow::bail!("--format paths is not supported for list-tags");
            }
            if include_image_paths && format == Some(OutputFormat::Table) {
                anyhow::bail!(
                    "--include-image-paths cannot be combined with --format table (use JSON or omit --format)"
                );
            }
            let cwd = std::env::current_dir()?;
            let source = resolve_source_from_index_hint(index.as_deref(), &cwd)?;
            let paths = project_paths_for_source(&source)?;

            let table_output = !include_image_paths
                && (format.is_none() || format == Some(OutputFormat::Table));
            let detailed_json = include_image_paths;
            let simple_json =
                !include_image_paths && format == Some(OutputFormat::Json);

            if table_output {
                let mut rows = list_tag_counts(&paths.tags_db_path)?;
                rows.retain(|(_, c)| *c >= min_count);
                if let Some(lim) = limit {
                    rows.truncate(lim);
                }
                println!("{:<40} {:>8}", "TAG", "COUNT");
                for (tag, c) in rows {
                    println!("{:<40} {:>8}", truncate(&tag, 40), c);
                }
            } else if detailed_json || simple_json {
                if detailed_json {
                    let mut rows = list_tag_counts_with_images(&paths.tags_db_path)?;
                    rows.retain(|(_, c, _)| *c >= min_count);
                    if let Some(lim) = limit {
                        rows.truncate(lim);
                    }
                    #[derive(Serialize)]
                    struct ImageRow {
                        rel_path: String,
                        path: PathBuf,
                        score: f32,
                    }
                    #[derive(Serialize)]
                    struct Row {
                        tag: String,
                        count: usize,
                        images: Vec<ImageRow>,
                    }
                    let j: Vec<Row> = rows
                        .into_iter()
                        .map(|(tag, count, images)| Row {
                            tag,
                            count,
                            images: images
                                .into_iter()
                                .map(|(rel_path, score)| ImageRow {
                                    path: rel_path_to_abs(&paths.source_path, &rel_path),
                                    rel_path,
                                    score,
                                })
                                .collect(),
                        })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&j)?);
                } else {
                    let mut rows = list_tag_counts(&paths.tags_db_path)?;
                    rows.retain(|(_, c)| *c >= min_count);
                    if let Some(lim) = limit {
                        rows.truncate(lim);
                    }
                    #[derive(Serialize)]
                    struct Row {
                        tag: String,
                        count: usize,
                    }
                    let j: Vec<Row> = rows
                        .into_iter()
                        .map(|(tag, count)| Row { tag, count })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&j)?);
                }
            } else {
                anyhow::bail!("unexpected list-tags options (this is a bug)");
            }
        }
        Command::ListIndex { format } => {
            let items = list_projects()?;
            match format {
                OutputFormat::Table => {
                    println!(
                        "{:<40} {:<45} {:>8} {:>12}",
                        "SOURCE FOLDER", "INDEX PATH", "FILES", "CREATED"
                    );
                    for (paths, cfg) in &items {
                        let n = ManifestCount::load(&paths.manifest_path)?;
                        println!(
                            "{:<40} {:<45} {:>8} {:>12}",
                            truncate(&paths.source_path.display().to_string(), 40),
                            truncate(&paths.project_dir.display().to_string(), 45),
                            n,
                            truncate(&cfg.created_at, 12)
                        );
                    }
                }
                OutputFormat::Json => {
                    let rows: Vec<ListJson> = items
                        .iter()
                        .map(|(paths, cfg)| {
                            let file_count = ManifestCount::load(&paths.manifest_path).unwrap_or(0);
                            ListJson {
                                source_folder: paths.source_path.clone(),
                                index_path: paths.project_dir.clone(),
                                files: file_count,
                                created: cfg.created_at.clone(),
                            }
                        })
                        .collect();
                    println!("{}", serde_json::to_string_pretty(&rows)?);
                }
                OutputFormat::Paths => {
                    for (paths, _) in &items {
                        println!("{}", paths.project_dir.display());
                    }
                }
            }
        }
        Command::Clean { folder, all, yes } => {
            if all {
                if !yes && !confirm_clean_all()? {
                    eprintln!("Aborted.");
                    return Ok(());
                }
                clean_all_projects()?;
                println!("Removed all projects under {:?}", default_project_root()?);
            } else {
                let folder = folder.ok_or_else(|| anyhow::anyhow!("pass FOLDER_PATH or --all"))?;
                let source = folder
                    .canonicalize()
                    .with_context(|| format!("clean folder {}", folder.display()))?;
                clean_project(&source)?;
                println!("Removed index for {}", source.display());
            }
        }
    }
    Ok(())
}

struct ManifestCount;
impl ManifestCount {
    fn load(path: &std::path::Path) -> anyhow::Result<usize> {
        let m = Manifest::load_or_empty(path)?;
        Ok(m.entries.len())
    }
}

#[derive(Serialize)]
struct ListJson {
    source_folder: PathBuf,
    index_path: PathBuf,
    files: usize,
    created: String,
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
}

fn print_hits(hits: &[SearchHit], fmt: OutputFormat) -> anyhow::Result<()> {
    match fmt {
        OutputFormat::Table => {
            println!("{:<5} {:<8} PATH", "RANK", "SCORE");
            for h in hits {
                println!("{:<5} {:<8.3} {}", h.rank, h.score, h.path.display());
            }
        }
        OutputFormat::Json => {
            #[derive(Serialize)]
            struct Row {
                rank: usize,
                score: f32,
                path: PathBuf,
            }
            let rows: Vec<Row> = hits
                .iter()
                .map(|h| Row {
                    rank: h.rank,
                    score: h.score,
                    path: h.path.clone(),
                })
                .collect();
            println!("{}", serde_json::to_string_pretty(&rows)?);
        }
        OutputFormat::Paths => {
            for h in hits {
                println!("{}", h.path.display());
            }
        }
    }
    Ok(())
}

fn confirm_clean_all() -> anyhow::Result<bool> {
    if std::env::var("TWINPICS_CONFIRM")
        .map(|v| v == "yes" || v == "1")
        .unwrap_or(false)
    {
        return Ok(true);
    }
    eprint!("Delete ALL indices under ~/.twinpics/projects? [y/N] ");
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().read_line(&mut line)?;
    Ok(line.trim().eq_ignore_ascii_case("y") || line.trim().eq_ignore_ascii_case("yes"))
}
