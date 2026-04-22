//! twinpics CLI: `index`, `search`, `list-index`, `clean`.

use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::builder::BoolishValueParser;
use clap::{Parser, Subcommand};
use mll::CandleClipBackend;
use serde::Serialize;

use twinpics_core::{
    clean_all_projects, clean_project, default_project_root, index_folder, list_projects,
    manifest::Manifest, project_paths_for_source, resolve_source_from_index_hint,
    search_project_image, search_project_text, IndexOptions, SearchHit, SearchParams,
};

#[derive(Parser)]
#[command(name = "twinpics_cli")]
#[command(about = "Local CLIP-based image search", version)]
struct Cli {
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
    /// Remove index data for a source folder, or wipe all indices.
    Clean {
        /// Source folder whose index to delete.
        folder: Option<PathBuf>,
        #[arg(long)]
        all: bool,
    },
}

#[derive(Clone, Copy, Default, clap::ValueEnum)]
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
    match cli.command {
        Command::Index {
            folder,
            recursive,
            project_dir,
        } => {
            if let Some(p) = project_dir {
                std::env::set_var("TWINPICS_PROJECT_DIR", p.as_os_str());
            }
            let source = folder
                .canonicalize()
                .with_context(|| format!("index folder {}", folder.display()))?;
            let paths = project_paths_for_source(&source)?;
            let backend: Arc<dyn mll::EmbeddingBackend> = Arc::new(CandleClipBackend::new()?);
            index_folder(
                &source,
                backend,
                &paths,
                IndexOptions {
                    recursive,
                    model: "clip-vit-base-patch32".to_string(),
                },
            )?;
            println!(
                "Indexed {} -> {}",
                source.display(),
                paths.project_dir.display()
            );
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
        Command::Clean { folder, all } => {
            if all {
                if !confirm_clean_all()? {
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
