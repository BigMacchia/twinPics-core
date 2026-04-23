//! CLIP ViT-B/32 inference via `candle` with offline weights on disk.

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::{self, ClipModel};
use tokenizers::Tokenizer;

use super::backend::{EmbeddingBackend, MllError};
use super::model_paths::{clip_weight_paths_in, resolve_clip_model_dir};
#[cfg(feature = "model-download")]
use super::model_paths::default_clip_home_model_dir;
use super::CLIP_EMBED_DIM;

#[cfg(feature = "model-download")]
const HF_REPO: &str = "openai/clip-vit-base-patch32";
#[cfg(feature = "model-download")]
const HF_REVISION: &str = "refs/pr/15";

/// CLIP backend using Candle (CPU). Loads weights from disk (bundled next to the executable,
/// `TWINPICS_CLIP_MODEL_DIR`, or `~/.twinpics/models/clip-vit-base-patch32/`).
pub struct CandleClipBackend {
    model: ClipModel,
    tokenizer: Tokenizer,
    device: Device,
    config: clip::ClipConfig,
}

impl CandleClipBackend {
    /// Load CLIP ViT-B/32 from the first directory that contains both weight files.
    ///
    /// With the `model-download` feature, if none are found, downloads into
    /// `~/.twinpics/models/clip-vit-base-patch32/` and loads from there.
    pub fn new() -> Result<Self, MllError> {
        let dir = match resolve_clip_model_dir() {
            Ok(d) => d,
            Err(err) => {
                #[cfg(feature = "model-download")]
                {
                    let Some(home_dir) = default_clip_home_model_dir() else {
                        return Err(err);
                    };
                    download_clip_weights_to(&home_dir)?;
                    home_dir
                }
                #[cfg(not(feature = "model-download"))]
                {
                    return Err(err);
                }
            }
        };
        Self::load_from_dir(&dir)
    }

    fn load_from_dir(dir: &Path) -> Result<Self, MllError> {
        let device = Device::Cpu;
        let config = clip::ClipConfig::vit_base_patch32();
        let (model_file, tokenizer_file) = clip_weight_paths_in(dir)?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[model_file.as_path()], DType::F32, &device)
                .map_err(MllError::Candle)?
        };
        let model = ClipModel::new(vb, &config).map_err(MllError::Candle)?;
        let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(MllError::Tokenizers)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    fn load_image_tensor(&self, path: &Path) -> Result<Tensor, MllError> {
        let image_size = self.config.image_size;
        let img = image::ImageReader::open(path)
            .map_err(|e| MllError::Image(e.to_string()))?
            .decode()
            .map_err(|e| MllError::Image(e.to_string()))?;
        let (height, width) = (image_size, image_size);
        let img = img.resize_to_fill(
            width as u32,
            height as u32,
            image::imageops::FilterType::Triangle,
        );
        let img = img.to_rgb8();
        let img = img.into_raw();
        let tensor = Tensor::from_vec(img, (height, width, 3), &self.device)
            .map_err(MllError::Candle)?
            .permute((2, 0, 1))
            .map_err(MllError::Candle)?
            .to_dtype(DType::F32)
            .map_err(MllError::Candle)?
            .affine(2. / 255., -1.)
            .map_err(MllError::Candle)?;
        Ok(tensor)
    }

    fn tokenize_text(&self, text: &str) -> Result<Tensor, MllError> {
        let pad_id = *self
            .tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .ok_or_else(|| MllError::Invalid("missing <|endoftext|> in tokenizer".into()))?;
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(MllError::Tokenizers)?;
        let mut ids = encoding.get_ids().to_vec();
        let max_len = 77usize;
        if ids.len() > max_len {
            ids.truncate(max_len);
        } else if ids.len() < max_len {
            ids.extend(std::iter::repeat_n(pad_id, max_len - ids.len()));
        }
        Tensor::from_vec(ids, (1, max_len), &self.device).map_err(MllError::Candle)
    }
}

#[cfg(feature = "model-download")]
fn download_clip_weights_to(dir: &Path) -> Result<(), MllError> {
    use std::fs;

    fs::create_dir_all(dir)?;
    let api = hf_hub::api::sync::Api::new().map_err(|e| MllError::HfHub(e.to_string()))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        HF_REPO.to_string(),
        hf_hub::RepoType::Model,
        HF_REVISION.to_string(),
    ));
    let model_src = repo
        .get("model.safetensors")
        .map_err(|e| MllError::HfHub(format!("model.safetensors: {e}")))?;
    let tok_src = repo
        .get("tokenizer.json")
        .map_err(|e| MllError::HfHub(format!("tokenizer.json: {e}")))?;

    let model_dst = dir.join("model.safetensors");
    let tok_dst = dir.join("tokenizer.json");
    if !model_dst.is_file() {
        fs::copy(&model_src, &model_dst)?;
    }
    if !tok_dst.is_file() {
        fs::copy(&tok_src, &tok_dst)?;
    }
    Ok(())
}

impl EmbeddingBackend for CandleClipBackend {
    fn embed_image(&self, path: &Path) -> Result<Vec<f32>, MllError> {
        let t = self.load_image_tensor(path)?;
        let batch = t.unsqueeze(0).map_err(MllError::Candle)?;
        let features = self.model.get_image_features(&batch)?;
        let normed = clip::div_l2_norm(&features)?;
        let v = normed.flatten_all()?.to_vec1::<f32>()?;
        if v.len() != CLIP_EMBED_DIM {
            return Err(MllError::Invalid(format!(
                "unexpected image embedding len {}",
                v.len()
            )));
        }
        Ok(v)
    }

    fn embed_text(&self, tags: &[&str]) -> Result<Vec<f32>, MllError> {
        let text = tags.join(" ");
        let input_ids = self.tokenize_text(&text)?;
        let features = self.model.get_text_features(&input_ids)?;
        let normed = clip::div_l2_norm(&features)?;
        let v = normed.flatten_all()?.to_vec1::<f32>()?;
        if v.len() != CLIP_EMBED_DIM {
            return Err(MllError::Invalid(format!(
                "unexpected text embedding len {}",
                v.len()
            )));
        }
        Ok(v)
    }
}
