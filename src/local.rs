//! AI Local Remove: LaMa ("big-lama", Apache-2.0), run by ONNX
//! Runtime on the CPU. Nothing is uploaded and no key is needed.
//!
//! The weights are not part of this package. The first request downloads
//! them, once, from a pinned revision of the ONNX port at
//! <https://huggingface.co/Carve/LaMa-ONNX> (`lama_fp32.onnx`, 198 MiB) into
//! the extension's data folder, and refuses anything that doesn't match the
//! pinned SHA-256. The model's training data is Places; check that its terms
//! suit what you use it for.
//!
//! What this relies on, from the model's card and the exported graph
//! (checked 2026-11): inputs `image` (`[batch, 3, 512, 512]`, float, 0 to 1)
//! and `mask` (`[batch, 1, 512, 512]`, float, 1 where to fill), output
//! `output` (`[batch, 3, 512, 512]`, float, 0 to 255). The input size is
//! fixed at 512, so the region is padded to a square (its edges mirrored),
//! shrunk to 512, and the answer scaled back and cropped. A mask pixel is a
//! hole when any painted pixel falls in it, so a one-pixel wire survives
//! the shrink. Outside the mask the answer is the input's own pixels.
use crate::pixels;
use crate::provider::{Provider, ProviderError};
use image::RgbImage;
use image::imageops::{FilterType, resize};
use sha2::{Digest, Sha256};
use std::io::{Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The model's input side.
const NET: usize = 512;

/// Whether this build can run the model: ONNX Runtime has no prebuilt binary
/// for Intel Macs.
pub(crate) const SUPPORTED: bool = cfg!(not(all(target_os = "macos", target_arch = "x86_64")));

/// A file this program downloads, and what it must be.
#[derive(Clone, Debug)]
pub struct ModelFile {
    /// Where to download it.
    pub url: String,
    /// Its SHA-256, in lowercase hex. Nothing else is kept.
    pub sha256: String,
    /// Its size in bytes.
    pub bytes: u64,
}

impl ModelFile {
    /// `lama_fp32.onnx` at a pinned revision of `Carve/LaMa-ONNX`.
    pub fn lama() -> Self {
        Self {
            url: "https://huggingface.co/Carve/LaMa-ONNX/resolve/\
                  c3c0c9e468934d62e79c329e35d82dd09ff8c444/lama_fp32.onnx"
                .into(),
            sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6".into(),
            bytes: 208_044_816,
        }
    }

    /// The first 12 characters of the hash, for names.
    fn short(&self) -> &str {
        self.sha256.get(..12).unwrap_or(&self.sha256)
    }

    /// Where the verified file lives in `dir`; named by its hash, so a new
    /// pin is a new file.
    fn path(&self, dir: &Path) -> PathBuf {
        dir.join(format!("lama-{}.onnx", self.short()))
    }
}

/// Whether the model has been downloaded and checked already.
pub(crate) fn cached(dir: &Path, file: &ModelFile) -> bool {
    std::fs::metadata(file.path(dir)).is_ok_and(|m| m.len() == file.bytes)
}

/// LaMa as a [`Provider`].
pub(crate) struct Lama {
    file: ModelFile,
    dir: PathBuf,
    /// How long the download may take in one request: most of the deadline,
    /// leaving the rest for loading the model and running it.
    download: Duration,
}

impl Lama {
    pub(crate) fn new(file: ModelFile, dir: &Path, deadline: Duration) -> Self {
        Self {
            file,
            dir: dir.to_path_buf(),
            download: deadline * 3 / 4,
        }
    }
}

impl Provider for Lama {
    fn name(&self) -> &'static str {
        "LaMa"
    }
    fn model(&self) -> String {
        "lama".into()
    }
    fn model_key(&self) -> String {
        format!("local|lama|{}|{NET}|1", self.file.short())
    }
    fn erase(
        &self,
        image_png: &[u8],
        mask_png: &[u8],
        cancel: &dyn Fn() -> bool,
    ) -> Result<Vec<u8>, ProviderError> {
        let model = ensure(&self.dir, &self.file, self.download, cancel)?;
        let (rgb, width, height) = pixels::decode(image_png)?;
        let (mask, ..) = pixels::decode(mask_png)?;
        let mask: Vec<u8> = mask.iter().step_by(3).copied().collect();
        let filled = fill(&model, &rgb, &mask, width, height)?;
        if cancel() {
            return Err(ProviderError::Cancelled);
        }
        pixels::encode_rgb_png(&filled, width, height)
    }
}

/// The verified model in `dir`, downloading what is missing for up to
/// `budget`. A download that doesn't finish keeps what it has, and the next
/// request continues from there. A saved model is hashed again before it is
/// used (a fraction of a second), so a damaged file is never trusted.
fn ensure(
    dir: &Path,
    file: &ModelFile,
    budget: Duration,
    cancel: &dyn Fn() -> bool,
) -> Result<PathBuf, ProviderError> {
    let done = file.path(dir);
    if cached(dir, file) {
        if sha256_of(&done).is_ok_and(|hash| hash == file.sha256) {
            return Ok(done);
        }
        // Damaged since it was checked: fetch it again.
        let _ = std::fs::remove_file(&done);
    }
    let io = |e: std::io::Error| unavailable(format!("couldn't save the model: {e}"));
    std::fs::create_dir_all(dir).map_err(io)?;
    let part = done.with_extension("part");
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&part)
        .map_err(io)?;
    // One writer at a time: a request cancelled mid-download can leave its
    // thread writing for a moment, and a retry must not interleave with it.
    out.try_lock()
        .map_err(|_| unavailable("the model is already downloading; send it again".into()))?;
    let mut have = out.metadata().map_err(io)?.len();
    if have > file.bytes {
        out.set_len(0).map_err(io)?;
        have = 0;
    }
    if have < file.bytes {
        have = download(file, &mut out, have, budget, cancel)?;
    }
    if have < file.bytes {
        let mib = |b: u64| b / (1 << 20);
        return Err(unavailable(format!(
            "downloaded {} of {} MiB of the model so far; send it again to continue",
            mib(have),
            mib(file.bytes)
        )));
    }
    out.sync_all().map_err(io)?;
    // Read back through the handle that holds the lock: on Windows another
    // handle could not read the locked part.
    let hash = hash_of(&mut out).map_err(io)?;
    drop(out);
    if hash != file.sha256 {
        let _ = std::fs::remove_file(&part);
        return Err(unavailable(
            "the downloaded model didn't match its checksum (SHA-256), so it was deleted; \
             try again later"
                .into(),
        ));
    }
    std::fs::rename(&part, &done).map_err(io)?;
    Ok(done)
}

fn unavailable(why: String) -> ProviderError {
    ProviderError::Unavailable(why)
}

/// Appends the rest of `file` to `out`, which holds `have` bytes of it, until
/// the answer ends, `budget` runs out or `cancel` is true; returns how many
/// bytes `out` holds then. A server that ignores the range sends the whole
/// file again, and `out` starts over.
fn download(
    file: &ModelFile,
    out: &mut std::fs::File,
    mut have: u64,
    budget: Duration,
    cancel: &dyn Fn() -> bool,
) -> Result<u64, ProviderError> {
    let io = |e: std::io::Error| unavailable(format!("couldn't save the model: {e}"));
    let started = Instant::now();
    let agent = crate::http::agent(budget);
    let mut request = agent.get(&file.url);
    if have > 0 {
        request = request.header("Range", format!("bytes={have}-"));
    }
    let mut response = request
        .call()
        .map_err(|e| unavailable(format!("couldn't download the model: {e}")))?;
    match response.status().as_u16() {
        206 => {}
        200 => {
            out.set_len(0).map_err(io)?;
            have = 0;
        }
        // The part is past what the server has; start over next time.
        416 => {
            out.set_len(0).map_err(io)?;
            return Err(unavailable(
                "the model download restarts; send it again".into(),
            ));
        }
        status => {
            return Err(unavailable(format!(
                "downloading the model failed (HTTP {status})"
            )));
        }
    }
    // Not append mode: Windows refuses to resize a file opened only to append.
    out.seek(std::io::SeekFrom::Start(have)).map_err(io)?;
    let mut body = response.body_mut().as_reader();
    let mut chunk = vec![0u8; 1 << 16];
    while have < file.bytes && started.elapsed() <= budget {
        // A dropped connection or a timeout ends the read: keep what came.
        let n = match body.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        if cancel() {
            return Err(ProviderError::Cancelled);
        }
        out.write_all(&chunk[..n]).map_err(io)?;
        have += n as u64;
    }
    out.flush().map_err(io)?;
    Ok(have)
}

fn sha256_of(path: &Path) -> std::io::Result<String> {
    hash_of(&mut std::fs::File::open(path)?)
}

/// The SHA-256 of all of `file`, as lowercase hex.
fn hash_of(file: &mut std::fs::File) -> std::io::Result<String> {
    file.seek(std::io::SeekFrom::Start(0))?;
    let mut hash = Sha256::new();
    let mut chunk = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        hash.update(&chunk[..n]);
    }
    Ok(hash.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

/// The region with `mask` (255 where to fill) filled by the model at
/// `model`, as packed RGB; the pixels outside the mask are the input's.
fn fill(
    model: &Path,
    rgb: &[u8],
    mask: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u8>, ProviderError> {
    // Nothing the user sent was wrong: a model that won't load or run is the
    // machine's or the file's, so it is unavailable, not invalid.
    let failed = |e: &dyn std::fmt::Display| unavailable(format!("the model failed: {e}"));
    let square = pixels::pad_to_aspect(rgb, width, height, &["1:1"]);
    let side = square.width;
    let net = shrink(&square.rgb, side);
    let hole = shrink_mask(mask, width, height, (square.x, square.y), side);
    let image: Vec<f32> = (0..3)
        .flat_map(|c| net.iter().skip(c).step_by(3).map(|&v| f32::from(v) / 255.0))
        .collect();
    let mask_in: Vec<f32> = hole.iter().map(|&m| f32::from(m)).collect();
    let out = infer(model, image, mask_in).map_err(|e| failed(&e))?;
    if out.len() != 3 * NET * NET {
        return Err(failed(&"answered with an unexpected shape"));
    }
    let plane = NET * NET;
    let small: Vec<u8> = (0..plane)
        .flat_map(|i| (0..3).map(move |c| (c, i)))
        .map(|(c, i)| out[c * plane + i].round().clamp(0.0, 255.0) as u8)
        .collect();
    let big = grow(&small, side);
    let filled = pixels::crop(&big, side, square.x, square.y, width, height);
    // The mask is the only thing the model changes.
    let mut result = rgb.to_vec();
    for (i, _) in mask.iter().enumerate().filter(|&(_, &m)| m != 0) {
        result[i * 3..][..3].copy_from_slice(&filled[i * 3..][..3]);
    }
    Ok(result)
}

/// Runs the model on `image` and `mask`, as `NET` x `NET` planes, and returns
/// its `output` plane by plane.
#[cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
fn infer(model: &Path, image: Vec<f32>, mask: Vec<f32>) -> Result<Vec<f32>, String> {
    use ort::environment::Environment;
    use std::sync::OnceLock;
    // ONNX Runtime's environment is made once: it allows one per process, and
    // a request after the first would otherwise find it taken.
    static ENV: OnceLock<Result<Environment, String>> = OnceLock::new();
    let env = ENV
        .get_or_init(|| {
            ort::init()
                .with_name("rawmakase-ai-retouch")
                .build()
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(Clone::clone)?;
    let run = || -> ort::Result<Vec<f32>> {
        let threads = std::thread::available_parallelism().map_or(4, usize::from);
        let mut session = ort::session::Session::builder(env)?
            .with_intra_threads(threads)?
            .commit_from_file(model)?;
        let image = ort::value::Tensor::from_array(([1, 3, NET, NET], image))?;
        let mask = ort::value::Tensor::from_array(([1, 1, NET, NET], mask))?;
        let outputs = session.run(ort::inputs!["image" => image, "mask" => mask])?;
        let (_, out) = outputs["output"].try_extract_tensor::<f32>()?;
        Ok(out.to_vec())
    };
    run().map_err(|e| e.to_string())
}

/// ONNX Runtime has no prebuilt binary for Intel Macs.
#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
fn infer(_: &Path, _: Vec<f32>, _: Vec<f32>) -> Result<Vec<f32>, String> {
    Err("AI Local Remove isn't available on Intel Macs".into())
}

/// A `side` x `side` RGB image at `NET` x `NET`.
fn shrink(rgb: &[u8], side: usize) -> Vec<u8> {
    if side == NET {
        return rgb.to_vec();
    }
    let image = RgbImage::from_raw(side as u32, side as u32, rgb.to_vec()).expect("sized");
    resize(&image, NET as u32, NET as u32, FilterType::Triangle).into_raw()
}

/// A `NET` x `NET` image back at `side` x `side`.
fn grow(rgb: &[u8], side: usize) -> Vec<u8> {
    if side == NET {
        return rgb.to_vec();
    }
    let image = RgbImage::from_raw(NET as u32, NET as u32, rgb.to_vec()).expect("sized");
    resize(&image, side as u32, side as u32, FilterType::CatmullRom).into_raw()
}

/// The mask on the padded square at `NET` x `NET`: 1.0 in each cell where
/// any mask pixel (255) falls, 0.0 elsewhere. The padding is never a hole.
fn shrink_mask(
    mask: &[u8],
    width: usize,
    height: usize,
    (x0, y0): (usize, usize),
    side: usize,
) -> Vec<u8> {
    let mut hole = vec![0u8; NET * NET];
    for y in 0..height {
        for x in 0..width {
            if mask[y * width + x] == 0 {
                continue;
            }
            // The cells the pixel's span [p, p+1) of `side` touches.
            let (px, py) = (x + x0, y + y0);
            let span = |p: usize| (p * NET / side, ((p + 1) * NET).div_ceil(side));
            let ((ax, bx), (ay, by)) = (span(px), span(py));
            for cy in ay..by.min(NET) {
                hole[cy * NET + ax..cy * NET + bx.min(NET)].fill(1);
            }
        }
    }
    hole
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_one_pixel_line_survives_the_shrink() {
        let (w, h) = (2048, 2048);
        let mut mask = vec![0u8; w * h];
        for y in 0..h {
            mask[y * w + 1000] = 255;
        }
        let hole = shrink_mask(&mask, w, h, (0, 0), 2048);
        // 2048 px shrink 4x: the wire's column is one cell wide, all the way down.
        assert!((0..NET).all(|y| hole[y * NET + 250] == 1));
        assert_eq!(hole.iter().filter(|&&v| v == 1).count(), NET);
    }

    #[test]
    fn a_small_region_enlarges_its_cells_and_padding_is_never_a_hole() {
        let (w, h) = (64, 32);
        let mask = vec![255u8; w * h];
        // Padded to 64 x 64 with the region in rows 16..48.
        let hole = shrink_mask(&mask, w, h, (0, 16), 64);
        assert_eq!(hole[0], 0);
        assert_eq!(hole[(NET / 2) * NET], 1);
        assert_eq!(hole.iter().filter(|&&v| v == 1).count(), NET * NET / 2);
    }

    #[test]
    fn the_model_file_is_named_by_its_hash() {
        let file = ModelFile::lama();
        assert_eq!(file.sha256.len(), 64);
        assert!(
            file.path(Path::new("/d"))
                .ends_with("lama-1faef5301d78.onnx")
        );
    }
}
