use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{anyhow, Context};
use candle::{DType, Device, IndexOp, Module, Tensor, D};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::clip;
use candle_transformers::models::stable_diffusion::unet_2d::UNet2DConditionModel;
use candle_transformers::models::stable_diffusion::vae::AutoEncoderKL;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use tokenizers::Tokenizer;

const HEIGHT: usize = 512;
const WIDTH: usize = 512;
const VAE_SCALE: f64 = 0.13025;

static CONTEXT: OnceLock<Mutex<SdxlBridge>> = OnceLock::new();
static EMPTY_C_STRING: &[u8; 1] = b"\0";

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

struct SdxlBridge {
    device: Device,
    dtype: DType,
    config: stable_diffusion::StableDiffusionConfig,
    tokenizer_primary: Tokenizer,
    tokenizer_secondary: Tokenizer,
    clip_primary: clip::ClipTextTransformer,
    clip_secondary: clip::ClipTextTransformer,
    unet: UNet2DConditionModel,
    vae: AutoEncoderKL,
}

struct AssetLayout {
    tokenizer_primary: PathBuf,
    tokenizer_secondary: PathBuf,
    clip_primary: PathBuf,
    clip_secondary: PathBuf,
    unet: PathBuf,
    vae: PathBuf,
}

impl AssetLayout {
    fn new(root: PathBuf) -> Result<Self, CandleSdxlStatusCode> {
        if !root.exists() {
            return Err(set_error(
                CandleSdxlStatusCode::AssetRootMissing,
                format!("asset directory {:?} does not exist", root),
            ));
        }
        let layout = Self {
            tokenizer_primary: root.join("tokenizer/tokenizer.json"),
            tokenizer_secondary: root.join("tokenizer_2/tokenizer.json"),
            clip_primary: root.join("text_encoder/model.fp16.safetensors"),
            clip_secondary: root.join("text_encoder_2/model.fp16.safetensors"),
            unet: root.join("unet/diffusion_pytorch_model.fp16.safetensors"),
            vae: root.join("vae/diffusion_pytorch_model.fp16.safetensors"),
        };
        if !layout.tokenizer_primary.exists() {
            return Err(set_error(
                CandleSdxlStatusCode::AssetTokenizerPrimaryMissing,
                format!(
                    "missing asset tokenizer/tokenizer.json at {:?}",
                    layout.tokenizer_primary
                ),
            ));
        }
        if !layout.tokenizer_secondary.exists() {
            return Err(set_error(
                CandleSdxlStatusCode::AssetTokenizerSecondaryMissing,
                format!(
                    "missing asset tokenizer_2/tokenizer.json at {:?}",
                    layout.tokenizer_secondary
                ),
            ));
        }
        if !layout.clip_primary.exists() {
            return Err(set_error(
                CandleSdxlStatusCode::AssetClipPrimaryMissing,
                format!(
                    "missing asset text_encoder/model.fp16.safetensors at {:?}",
                    layout.clip_primary
                ),
            ));
        }
        if !layout.clip_secondary.exists() {
            return Err(set_error(
                CandleSdxlStatusCode::AssetClipSecondaryMissing,
                format!(
                    "missing asset text_encoder_2/model.fp16.safetensors at {:?}",
                    layout.clip_secondary
                ),
            ));
        }
        if !layout.unet.exists() {
            return Err(set_error(
                CandleSdxlStatusCode::AssetUnetMissing,
                format!(
                    "missing asset unet/diffusion_pytorch_model.fp16.safetensors at {:?}",
                    layout.unet
                ),
            ));
        }
        if !layout.vae.exists() {
            return Err(set_error(
                CandleSdxlStatusCode::AssetVaeMissing,
                format!(
                    "missing asset vae/diffusion_pytorch_model.fp16.safetensors at {:?}",
                    layout.vae
                ),
            ));
        }
        Ok(layout)
    }
}

#[repr(C)]
pub struct CandleSdxlInitOptions {
    pub asset_dir: *const c_char,
    pub use_metal: u8,
}

#[repr(C)]
pub struct CandleSdxlRequest {
    pub prompt: *const c_char,
    pub steps: u32,
    pub seed: u64,
    pub use_seed: u8,
}

#[repr(C)]
pub struct CandleSdxlImage {
    pub data: *mut u8,
    pub len: usize,
    pub capacity: usize,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CandleSdxlStatusCode {
    Ok = 0,
    InvalidArgument = 1,
    NotInitialized = 2,
    AlreadyInitialized = 3,
    RuntimeError = 4,
    AssetDirInvalid = 5,
    AssetRootMissing = 6,
    AssetTokenizerPrimaryMissing = 7,
    AssetTokenizerSecondaryMissing = 8,
    AssetClipPrimaryMissing = 9,
    AssetClipSecondaryMissing = 10,
    AssetUnetMissing = 11,
    AssetVaeMissing = 12,
    LoadTokenizerPrimaryFailed = 13,
    LoadTokenizerSecondaryFailed = 14,
    LoadClipPrimaryFailed = 15,
    LoadClipSecondaryFailed = 16,
    LoadUnetFailed = 17,
    LoadVaeFailed = 18,
    MetalUnavailable = 19,
}

#[no_mangle]
pub unsafe extern "C" fn candle_sdxl_init(
    options: *const CandleSdxlInitOptions,
) -> CandleSdxlStatusCode {
    if options.is_null() {
        return set_error(
            CandleSdxlStatusCode::InvalidArgument,
            "options pointer was null",
        );
    }
    if CONTEXT.get().is_some() {
        return set_error(
            CandleSdxlStatusCode::AlreadyInitialized,
            "bridge already initialised",
        );
    }
    let options = &*options;
    let asset_dir = match c_string_to_path(options.asset_dir) {
        Ok(path) => path,
        Err(_code) => {
            return set_error(
                CandleSdxlStatusCode::AssetDirInvalid,
                "invalid asset_dir path",
            )
        }
    };
    let use_metal = options.use_metal != 0;

    match initialise(&asset_dir, use_metal) {
        Ok(context) => {
            let _ = CONTEXT.set(Mutex::new(context));
            CandleSdxlStatusCode::Ok
        }
        Err(code) => code,
    }
}

#[no_mangle]
pub extern "C" fn candle_sdxl_is_ready() -> bool {
    CONTEXT.get().is_some()
}

#[no_mangle]
pub unsafe extern "C" fn candle_sdxl_generate(
    request: *const CandleSdxlRequest,
    out_image: *mut CandleSdxlImage,
) -> CandleSdxlStatusCode {
    if request.is_null() {
        return set_error(
            CandleSdxlStatusCode::InvalidArgument,
            "request pointer was null",
        );
    }
    if out_image.is_null() {
        return set_error(
            CandleSdxlStatusCode::InvalidArgument,
            "out_image pointer was null",
        );
    }
    let context = match CONTEXT.get() {
        Some(ctx) => ctx,
        None => {
            return set_error(
                CandleSdxlStatusCode::NotInitialized,
                "bridge not initialised",
            )
        }
    };
    let request = &*request;
    let prompt = match c_string_to_string(request.prompt) {
        Ok(p) => p,
        Err(code) => return code,
    };
    if prompt.trim().is_empty() {
        return set_error(
            CandleSdxlStatusCode::InvalidArgument,
            "prompt must not be empty",
        );
    }
    let steps = request.steps.max(1) as usize;
    let seed = if request.use_seed != 0 {
        Some(request.seed)
    } else {
        None
    };

    let mut guard = context.lock().expect("mutex poisoned");
    match generate_image(&mut guard, &prompt, steps, seed) {
        Ok(buffer) => {
            let mut buffer = buffer;
            let image = CandleSdxlImage {
                data: buffer.as_mut_ptr(),
                len: buffer.len(),
                capacity: buffer.capacity(),
                width: WIDTH as u32,
                height: HEIGHT as u32,
                channels: 4,
            };
            std::mem::forget(buffer);
            unsafe { *out_image = image };
            CandleSdxlStatusCode::Ok
        }
        Err(err) => set_error(CandleSdxlStatusCode::RuntimeError, err),
    }
}

#[no_mangle]
pub unsafe extern "C" fn candle_sdxl_free_image(image: *mut CandleSdxlImage) {
    if image.is_null() {
        return;
    }
    let image = &mut *image;
    if image.data.is_null() {
        return;
    }
    let data = image.data;
    let len = image.len;
    let capacity = image.capacity;
    let _ = Vec::from_raw_parts(data, len, capacity);
    image.data = std::ptr::null_mut();
    image.len = 0;
    image.capacity = 0;
}

#[no_mangle]
pub unsafe extern "C" fn candle_sdxl_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        if let Some(message) = slot.borrow().as_ref() {
            message.as_ptr()
        } else {
            EMPTY_C_STRING.as_ptr() as *const c_char
        }
    })
}

fn initialise(asset_dir: &Path, use_metal: bool) -> Result<SdxlBridge, CandleSdxlStatusCode> {
    let layout = AssetLayout::new(asset_dir.to_path_buf())?;
    let device = if use_metal {
        #[cfg(feature = "metal")]
        {
            match Device::new_metal(0) {
                Ok(d) => d,
                Err(e) => return Err(set_error(CandleSdxlStatusCode::MetalUnavailable, e)),
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            return Err(set_error(
                CandleSdxlStatusCode::MetalUnavailable,
                "bridge compiled without Metal support",
            ));
        }
    } else {
        Device::Cpu
    };

    // Mixed precision: UNet in F16 (on Metal) for perf, VAE in F32 for stability
    let (unet_dtype, vae_dtype) = if use_metal {
        (DType::F16, DType::F32)
    } else {
        (DType::F32, DType::F32)
    };
    let config =
        stable_diffusion::StableDiffusionConfig::sdxl_turbo(None, Some(HEIGHT), Some(WIDTH));

    let tokenizer_primary = match Tokenizer::from_file(&layout.tokenizer_primary)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("failed to load tokenizer at {:?}", layout.tokenizer_primary))
    {
        Ok(v) => v,
        Err(e) => {
            return Err(set_error(
                CandleSdxlStatusCode::LoadTokenizerPrimaryFailed,
                e,
            ))
        }
    };
    let tokenizer_secondary = match Tokenizer::from_file(&layout.tokenizer_secondary)
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "failed to load tokenizer at {:?}",
                layout.tokenizer_secondary
            )
        }) {
        Ok(v) => v,
        Err(e) => {
            return Err(set_error(
                CandleSdxlStatusCode::LoadTokenizerSecondaryFailed,
                e,
            ))
        }
    };

    let clip_primary = match stable_diffusion::build_clip_transformer(
        &config.clip,
        &layout.clip_primary,
        &device,
        DType::F32,
    ) {
        Ok(v) => v,
        Err(e) => return Err(set_error(CandleSdxlStatusCode::LoadClipPrimaryFailed, e)),
    };
    let clip2_config = match config.clip2.as_ref() {
        Some(c) => c,
        None => {
            return Err(set_error(
                CandleSdxlStatusCode::LoadClipSecondaryFailed,
                "sdxl turbo requires secondary text encoder",
            ))
        }
    };
    let clip_secondary = match stable_diffusion::build_clip_transformer(
        clip2_config,
        &layout.clip_secondary,
        &device,
        DType::F32,
    ) {
        Ok(v) => v,
        Err(e) => return Err(set_error(CandleSdxlStatusCode::LoadClipSecondaryFailed, e)),
    };

    let vae = match config.build_vae(&layout.vae, &device, vae_dtype) {
        Ok(v) => v,
        Err(e) => return Err(set_error(CandleSdxlStatusCode::LoadVaeFailed, e)),
    };
    let unet = match config.build_unet(&layout.unet, &device, 4, false, unet_dtype) {
        Ok(v) => v,
        Err(e) => return Err(set_error(CandleSdxlStatusCode::LoadUnetFailed, e)),
    };

    Ok(SdxlBridge {
        device,
        dtype: unet_dtype,
        config,
        tokenizer_primary,
        tokenizer_secondary,
        clip_primary,
        clip_secondary,
        unet,
        vae,
    })
}

fn generate_image(
    context: &mut SdxlBridge,
    prompt: &str,
    steps: usize,
    seed: Option<u64>,
) -> anyhow::Result<Vec<u8>> {
    let mut scheduler = context.config.build_scheduler(steps)?;
    let text_embeddings = build_text_embeddings(context, prompt)?;

    let latent_shape = (1, 4, context.config.height / 8, context.config.width / 8);
    let mut latents = sample_latents(latent_shape, &context.device, context.dtype, seed)?;
    latents = (latents * scheduler.init_noise_sigma())?.to_dtype(context.dtype)?;

    let timesteps = scheduler.timesteps().to_vec();
    for &timestep in &timesteps {
        let latent_model_input = scheduler.scale_model_input(latents.clone(), timestep)?;
        let noise_pred =
            context
                .unet
                .forward(&latent_model_input, timestep as f64, &text_embeddings)?;
        latents = scheduler
            .step(&noise_pred, timestep, &latents)?
            .to_dtype(context.dtype)?;
    }

    let latents = (latents / VAE_SCALE)?.to_dtype(DType::F32)?;
    let image = context.vae.decode(&latents)?;
    let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
    let image = (image.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;
    let image = image.i(0)?; // (3, H, W)
    let image = image.permute((1, 2, 0))?.flatten_all()?;
    let rgb = image.to_vec1::<u8>()?;
    #[cfg(debug_assertions)]
    {
        let non_zero = rgb.iter().filter(|v| **v != 0).count();
        println!("DEBUG: rgb bytes non-zero {}/{}", non_zero, rgb.len());
    }
    Ok(to_rgba(&rgb))
}

fn build_text_embeddings(context: &SdxlBridge, prompt: &str) -> anyhow::Result<Tensor> {
    let prompt = prompt.trim();
    let tokens_primary = tokenize_prompt(
        &context.tokenizer_primary,
        &context.config.clip,
        prompt,
        &context.device,
    )?;
    let embed_primary = context.clip_primary.forward(&tokens_primary)?;

    let clip2_config = context
        .config
        .clip2
        .as_ref()
        .ok_or_else(|| anyhow!("sdxl turbo requires secondary encoder"))?;
    let tokens_secondary = tokenize_prompt(
        &context.tokenizer_secondary,
        clip2_config,
        prompt,
        &context.device,
    )?;
    let embed_secondary = context.clip_secondary.forward(&tokens_secondary)?;

    let embed_primary = embed_primary.to_dtype(context.dtype)?;
    let embed_secondary = embed_secondary.to_dtype(context.dtype)?;
    Ok(Tensor::cat(&[&embed_primary, &embed_secondary], D::Minus1)?)
}

fn tokenize_prompt(
    tokenizer: &Tokenizer,
    config: &clip::Config,
    prompt: &str,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)
        .context("failed to encode prompt")?
        .get_ids()
        .to_vec();
    let pad_id = match &config.pad_with {
        Some(pad) => tokenizer
            .get_vocab(true)
            .get(pad.as_str())
            .copied()
            .ok_or_else(|| anyhow!("pad token {} not found in tokenizer", pad))?,
        None => tokenizer
            .get_vocab(true)
            .get("<|endoftext|>")
            .copied()
            .ok_or_else(|| anyhow!("<|endoftext|> not found in tokenizer"))?,
    };
    if tokens.len() > config.max_position_embeddings {
        return Err(anyhow!(
            "prompt is too long: {} tokens > max {}",
            tokens.len(),
            config.max_position_embeddings
        ));
    }
    while tokens.len() < config.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
    Ok(tokens)
}

fn to_rgba(rgb: &[u8]) -> Vec<u8> {
    let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
    for chunk in rgb.chunks_exact(3) {
        rgba.extend_from_slice(chunk);
        rgba.push(255);
    }
    rgba
}

fn sample_latents(
    shape: (usize, usize, usize, usize),
    device: &Device,
    dtype: DType,
    seed: Option<u64>,
) -> anyhow::Result<Tensor> {
    if let Some(seed) = seed {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = StandardNormal;
        let elem = shape.0 * shape.1 * shape.2 * shape.3;
        let mut data = vec![0f32; elem];
        for value in &mut data {
            *value = normal.sample(&mut rng);
        }
        let tensor = Tensor::from_vec(data, shape, &Device::Cpu)?
            .to_dtype(dtype)?
            .to_device(device)?;
        Ok(tensor)
    } else {
        let tensor = Tensor::randn(0f32, 1f32, shape, device)?;
        Ok(tensor.to_dtype(dtype)?)
    }
}

fn c_string_to_path(ptr: *const c_char) -> Result<PathBuf, CandleSdxlStatusCode> {
    c_string_to_string(ptr).map(PathBuf::from)
}

fn c_string_to_string(ptr: *const c_char) -> Result<String, CandleSdxlStatusCode> {
    if ptr.is_null() {
        return Err(set_error(
            CandleSdxlStatusCode::InvalidArgument,
            "received null string pointer",
        ));
    }
    let c_str = unsafe { CStr::from_ptr(ptr) };
    match c_str.to_str() {
        Ok(s) => Ok(s.to_string()),
        Err(_) => Err(set_error(
            CandleSdxlStatusCode::InvalidArgument,
            "string was not valid UTF-8",
        )),
    }
}

fn set_error(code: CandleSdxlStatusCode, message: impl std::fmt::Display) -> CandleSdxlStatusCode {
    let msg = message.to_string();
    let cstring =
        CString::new(msg).unwrap_or_else(|_| CString::new("error message contained nul").unwrap());
    LAST_ERROR.with(|slot| {
        slot.replace(Some(cstring));
    });
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_conversion_appends_alpha() {
        let rgb = vec![1, 2, 3, 4, 5, 6];
        let rgba = to_rgba(&rgb);
        assert_eq!(rgba, vec![1, 2, 3, 255, 4, 5, 6, 255]);
    }
}
