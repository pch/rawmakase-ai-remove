//! AI Retouch: AI Remove and AI Replace for RAWmakase's AI Retouch tool through
//! a hosted model, with the user's own API key.
//!
//! The app sends `retouch.generate` with a region of the photo (8-bit sRGB),
//! how much of each pixel the user painted, a result buffer of the same size
//! and, when the user wrote one, the request's text. This program turns the
//! painted area into a binary mask, sends the region, the mask and the text
//! to the chosen provider ([`fal`] by default, or [`replicate`]), and writes
//! the provider's image back into the result. The app owns all colour work,
//! the patch and how it blends; this program only turns pixels into pixels.
//!
//! Which model runs follows what is asked ([`Retouch::provider`]): a removal
//! without text goes to an eraser, one with text ("remove the dogs, keep the
//! people") to an instruction-following editor, and a replacement to the
//! fill model the settings choose.
//!
//! Every request is one paid call to the provider, so nothing here retries.
mod fal;
mod http;
mod local;
mod panel;
mod pixels;
mod provider;
mod replicate;
mod sanitize;

use provider::{Provider, ProviderError};
use rawmakase_extension::protocol::method::RetouchGenerated;
use rawmakase_extension::{
    Context, ErrorCode, Extension, RemoteError, RetouchAsk, RetouchGenerator,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

pub use local::ModelFile;
pub use sanitize::sanitize;

/// The manifest's `[[retouch-generator]]` id.
pub const GENERATOR: &str = "retouch";
/// The manifest's `[[retouch-generator]]` id for AI Remove on this computer:
/// no service, no key, nothing uploaded.
pub const LOCAL_GENERATOR: &str = "local";
pub use panel::{MODE, PANEL, PROMPT, REMOVE, REPLACE};

/// The manifest's `[[setting]]` ids.
pub mod setting {
    /// Which service runs the removal: [`super::FAL`] or [`super::REPLICATE`].
    pub const PROVIDER: &str = "provider";
    /// The fal.ai API key (a secret).
    pub const FAL_KEY: &str = "fal-key";
    /// The Replicate API token (a secret).
    pub const REPLICATE_TOKEN: &str = "replicate-token";
    /// The Replicate model, `owner/name` or `owner/name:version`.
    pub const REPLICATE_MODEL: &str = "replicate-model";
    /// The fal.ai model AI Replace uses: [`super::NANO_BANANA`],
    /// [`super::GENFILL`], [`super::FLUX_FILL`] or [`super::FIBO_EDIT`].
    pub const REPLACE_MODEL: &str = "replace-model";
}

/// The `provider` setting's value for fal.ai, the default.
pub const FAL: &str = "fal.ai";
/// The `provider` setting's value for Replicate.
pub const REPLICATE: &str = "Replicate";
/// The `replicate-model` setting's default.
pub const DEFAULT_REPLICATE_MODEL: &str = replicate::DEFAULT_MODEL;
/// The fal.ai model a removal without text calls.
pub const FAL_MODEL: &str = fal::ERASER;
/// The fal.ai model a removal with text calls.
pub const FAL_EDIT_MODEL: &str = fal::EDIT;
/// The Replicate model a request with text calls.
pub const REPLICATE_EDIT_MODEL: &str = replicate::EDIT_MODEL;

/// The `replace-model` setting's value for Nano Banana, the default.
pub const NANO_BANANA: &str = "Nano Banana";
/// The `replace-model` setting's value for Bria GenFill.
pub const GENFILL: &str = "Bria GenFill";
/// The `replace-model` setting's value for FLUX.1 Fill \[pro].
pub const FLUX_FILL: &str = "FLUX.1 Fill [pro]";
/// The `replace-model` setting's value for Bria FIBO Edit.
pub const FIBO_EDIT: &str = "Bria FIBO Edit";

/// The environment variable read when the `fal-key` setting is empty.
pub const FAL_KEY_ENV: &str = "FAL_KEY";
/// The environment variable read when the `replicate-token` setting is empty.
pub const REPLICATE_TOKEN_ENV: &str = "REPLICATE_API_TOKEN";

/// Where to tell the user to set a key.
const WHERE: &str = "Preferences > Extensions > AI Retouch";

/// Reads an environment variable.
pub type Env = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// How the program reaches its providers and its environment. [`extension`]
/// uses the real services; tests point it at a local server.
#[derive(Clone)]
pub struct Config {
    /// fal.ai's synchronous endpoint, without a trailing `/`
    /// (`https://fal.run`).
    pub fal_url: String,
    /// Replicate's API, without a trailing `/` (`https://api.replicate.com`).
    pub replicate_url: String,
    /// Reads an environment variable, for the key fallbacks.
    pub env: Env,
    /// How often a Replicate prediction still running is polled.
    pub poll_every: Duration,
    /// How long one removal may take, from the request to the provider's
    /// image. Below the app's 200 s deadline for `retouch.generate`, so the
    /// program answers before the app gives up on it.
    pub deadline: Duration,
    /// The model AI Remove on this computer downloads, once.
    pub lama: ModelFile,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            fal_url: "https://fal.run".into(),
            replicate_url: "https://api.replicate.com".into(),
            env: Arc::new(|name| std::env::var(name).ok()),
            poll_every: Duration::from_secs(1),
            deadline: Duration::from_secs(180),
            lama: ModelFile::lama(),
        }
    }
}

/// The extension, talking to the real services.
pub fn extension() -> Extension {
    extension_with(Config::default())
}

/// The extension, with `config`'s endpoints and environment.
pub fn extension_with(config: Config) -> Extension {
    let asked = panel::Shared::default();
    Extension::new()
        .retouch_generator(
            GENERATOR,
            Retouch {
                config: config.clone(),
                asked: asked.clone(),
                local: false,
            },
        )
        .retouch_generator(
            LOCAL_GENERATOR,
            Retouch {
                config,
                asked: asked.clone(),
                local: true,
            },
        )
        .panel(PANEL, panel::RetouchPanel(asked))
}

/// What a generator does with the painted area.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Removes it, or edits it as the text says.
    Remove,
    /// Fills it with what the text describes, which it needs.
    Replace,
}

/// The retouch generator: AI Remove or AI Replace, as its panel says.
struct Retouch {
    config: Config,
    asked: panel::Shared,
    /// The generator on this computer: always a plain removal by LaMa, which
    /// the panel and the settings don't touch.
    local: bool,
}

impl RetouchGenerator for Retouch {
    fn generate(
        &mut self,
        cx: &Context,
        ask: RetouchAsk<'_>,
    ) -> Result<RetouchGenerated, RemoteError> {
        // Whatever a setting holds, a key pasted in the wrong field included,
        // is cleaned out of what choosing the provider says.
        // What the panel holds: the mode, and the prompt on one line.
        let asked = if self.local {
            panel::Asked::default()
        } else {
            self.asked.get(cx.data_dir())
        };
        let kind = if asked.replace {
            Kind::Replace
        } else {
            Kind::Remove
        };
        let text = asked.text();
        let summary = text.as_deref().map(summary);
        let chosen = self.provider(cx, kind, text).map_err(|mut e| {
            let secrets: Vec<&str> = [
                setting::FAL_KEY,
                setting::REPLICATE_TOKEN,
                setting::REPLICATE_MODEL,
            ]
            .iter()
            .map(|id| text_setting(cx.settings(), id))
            .filter(|v| !v.is_empty())
            .collect();
            e.message = sanitize(&e.message, &secrets);
            e
        })?;
        let (name, secret) = (chosen.provider.name(), chosen.secret.clone());
        let fail = |e: ProviderError| to_remote(e, name, &secret);
        let request = &ask.request;
        let (width, height) = (request.image.width as usize, request.image.height as usize);
        let image = pixels::packed(ask.image, width, height, request.image.stride as usize, 3);
        let coverage = pixels::packed(
            ask.coverage,
            width,
            height,
            request.coverage.stride as usize,
            1,
        );
        let mask =
            pixels::binary_mask(&coverage, width, height, pixels::grow_radius(width, height));
        if !mask.contains(&255) {
            return Err(RemoteError::new(
                ErrorCode::InvalidRequest,
                "nothing is painted in the region",
            ));
        }
        let image_png = pixels::encode_rgb_png(&image, width, height).map_err(fail)?;
        let mask_png = pixels::encode_mask_png(&mask, width, height).map_err(fail)?;
        let answer = RetouchGenerated {
            model: chosen.provider.model(),
            model_key: chosen.provider.model_key(),
            summary,
        };
        if let Err(e) = answer.validate() {
            return Err(RemoteError::new(
                ErrorCode::InvalidRequest,
                format!("the model's name doesn't fit: {}", e.message),
            ));
        }
        let doing = match kind {
            Kind::Remove => "Removing",
            Kind::Replace => "Replacing",
        };
        if self.local && !local::cached(cx.data_dir(), &self.config.lama) {
            let mib = self.config.lama.bytes >> 20;
            cx.progress(
                None,
                Some(&format!("Downloading the removal model, once ({mib} MiB)")),
            );
        } else {
            cx.progress(None, Some(&format!("{doing} with {name}")));
        }
        let encoded = call(
            cx,
            chosen.provider,
            image_png,
            mask_png,
            self.config.deadline,
        )
        .map_err(fail)?;
        let rgb = pixels::decode_to_size(&encoded, width, height).map_err(fail)?;
        pixels::write_rgb(
            &rgb,
            width,
            height,
            ask.result,
            request.result.stride as usize,
        );
        Ok(answer)
    }
}

/// The provider a request goes to, with the secret its errors must never
/// show.
struct Chosen {
    provider: Box<dyn Provider>,
    secret: String,
}

impl Retouch {
    /// The provider the settings choose, with its key (from the setting, or
    /// else the environment; no key at all is `needs_setup`), and the model
    /// for what is asked: a removal without text the eraser, one with text
    /// the instruction-following editor, a replacement the fill model the
    /// settings choose on fal.ai and the editor on Replicate.
    fn provider(
        &self,
        cx: &Context,
        kind: Kind,
        text: Option<String>,
    ) -> Result<Chosen, RemoteError> {
        let (config, settings) = (&self.config, cx.settings());
        if self.local {
            if !local::SUPPORTED {
                return Err(RemoteError::new(
                    ErrorCode::Unavailable,
                    "AI Remove on this computer isn't available on Intel Macs: ONNX Runtime has \
                     no build for them",
                ));
            }
            return Ok(Chosen {
                provider: Box::new(local::Lama::new(
                    config.lama.clone(),
                    cx.data_dir(),
                    config.deadline,
                )),
                secret: String::new(),
            });
        }
        // Nothing wrong with the request: the user has a prompt to write.
        if kind == Kind::Replace && text.is_none() {
            return Err(RemoteError::new(
                ErrorCode::NeedsInput,
                "AI Replace needs a prompt: say what to put where you paint",
            ));
        }
        if text_setting(settings, setting::PROVIDER) == REPLICATE {
            let token = key(
                settings,
                setting::REPLICATE_TOKEN,
                REPLICATE_TOKEN_ENV,
                config,
            )
            .ok_or_else(|| {
                needs_setup(format!(
                    "No Replicate API token is set. Set one in {WHERE}; the link under its \
                     field shows how to get one."
                ))
            })?;
            let model = match (&text, text_setting(settings, setting::REPLICATE_MODEL)) {
                (Some(_), _) => REPLICATE_EDIT_MODEL,
                (None, "") => DEFAULT_REPLICATE_MODEL,
                (None, model) => model,
            };
            let provider = replicate::Replicate::new(
                &config.replicate_url,
                &token,
                model,
                text,
                config.deadline,
                config.poll_every,
            )
            .map_err(|why| RemoteError::new(ErrorCode::InvalidRequest, why))?;
            return Ok(Chosen {
                provider: Box::new(provider),
                secret: token,
            });
        }
        let key = key(settings, setting::FAL_KEY, FAL_KEY_ENV, config).ok_or_else(|| {
            needs_setup(format!(
                "No fal.ai API key is set. Set one in {WHERE}; the link under its field \
                 shows how to get one."
            ))
        })?;
        let model = match (kind, text) {
            (Kind::Remove, None) => fal::FalModel::Eraser,
            (Kind::Remove, Some(text)) => fal::FalModel::Edit(text),
            (Kind::Replace, text) => {
                let text = text.unwrap_or_default();
                match text_setting(settings, setting::REPLACE_MODEL) {
                    GENFILL => fal::FalModel::GenFill(text),
                    FLUX_FILL => fal::FalModel::FluxFill(text),
                    FIBO_EDIT => fal::FalModel::Edit(text),
                    _ => fal::FalModel::NanoBanana(text),
                }
            }
        };
        Ok(Chosen {
            provider: Box::new(fal::Fal::new(&config.fal_url, &key, model, config.deadline)),
            secret: key,
        })
    }
}

/// `text` as an answer's summary: at most [`MAX_SUMMARY_LEN`] bytes, cut
/// at a character.
///
/// [`MAX_SUMMARY_LEN`]: rawmakase_extension::protocol::method::MAX_SUMMARY_LEN
fn summary(text: &str) -> String {
    let max = rawmakase_extension::protocol::method::MAX_SUMMARY_LEN;
    let mut end = text.len().min(max);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].trim_end().to_string()
}

/// A text setting, trimmed; `""` when it is missing or not text.
fn text_setting<'a>(settings: &'a Map<String, Value>, id: &str) -> &'a str {
    settings
        .get(id)
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("")
}

/// A key from its secret setting, or else from the environment variable
/// `env`. A GUI app started from a desktop launcher usually doesn't inherit
/// a shell's variables, so the setting is the reliable place.
fn key(settings: &Map<String, Value>, id: &str, env: &str, config: &Config) -> Option<String> {
    let set = text_setting(settings, id);
    if !set.is_empty() {
        return Some(set.to_string());
    }
    (config.env)(env)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

fn needs_setup(message: String) -> RemoteError {
    RemoteError::new(ErrorCode::NeedsSetup, message)
}

/// Runs the provider's call on a thread of its own and polls cancellation
/// meanwhile, so a cancel is answered within a few tens of milliseconds even
/// while a request is in flight.
///
/// Once cancelled (or past `deadline`) this returns at once. The thread is
/// left to finish on its own: it sees the flag between steps (Replicate also
/// cancels its prediction), its HTTP calls are bounded by `deadline`, and
/// what it produces is dropped. It never touches the channel to the app.
fn call(
    cx: &Context,
    provider: Box<dyn Provider>,
    image_png: Vec<u8>,
    mask_png: Vec<u8>,
    deadline: Duration,
) -> Result<Vec<u8>, ProviderError> {
    let cancelled = Arc::new(AtomicBool::new(false));
    let (send, receive) = mpsc::channel();
    let flag = cancelled.clone();
    std::thread::Builder::new()
        .name("ai-retouch-call".into())
        .spawn(move || {
            let is_cancelled = || flag.load(Ordering::SeqCst);
            let _ = send.send(provider.erase(&image_png, &mask_png, &is_cancelled));
        })
        .map_err(|e| ProviderError::Unavailable(format!("couldn't start the request: {e}")))?;
    // A little past the provider's own deadline, which should end it first.
    let until = Instant::now() + deadline + Duration::from_secs(5);
    loop {
        if cx.is_cancelled() {
            cancelled.store(true, Ordering::SeqCst);
            return Err(ProviderError::Cancelled);
        }
        match receive.recv_timeout(Duration::from_millis(20)) {
            Ok(_) if cx.is_cancelled() => {
                cancelled.store(true, Ordering::SeqCst);
                return Err(ProviderError::Cancelled);
            }
            Ok(result) => return result,
            Err(RecvTimeoutError::Timeout) if Instant::now() > until => {
                cancelled.store(true, Ordering::SeqCst);
                return Err(ProviderError::Unavailable("the request timed out".into()));
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err(ProviderError::Unavailable(
                    "the request stopped without an answer".into(),
                ));
            }
        }
    }
}

/// The error the app sees: its code, and a message with every key-shaped
/// thing removed ([`sanitize`]).
fn to_remote(error: ProviderError, provider: &str, secret: &str) -> RemoteError {
    let clean = |m: &str| sanitize(m, &[secret]);
    match error {
        ProviderError::Auth { status, message } => {
            let detail = clean(&message);
            let detail = if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            };
            needs_setup(format!(
                "{provider} didn't accept the API key (HTTP {status}{detail}). \
                 Check it in {WHERE}."
            ))
        }
        ProviderError::Unavailable(message) => RemoteError::new(
            ErrorCode::Unavailable,
            format!("{provider}: {}", clean(&message)),
        ),
        ProviderError::Refused(message) => RemoteError::new(
            ErrorCode::Refused,
            format!("{provider} declined this image: {}", clean(&message)),
        ),
        ProviderError::Invalid(message) => RemoteError::new(
            ErrorCode::InvalidRequest,
            format!("{provider}: {}", clean(&message)),
        ),
        ProviderError::Cancelled => RemoteError::new(ErrorCode::Cancelled, "cancelled"),
    }
}
