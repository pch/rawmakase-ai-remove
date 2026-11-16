//! The real LaMa weights through the whole path, on a photo of your own.
//! Ignored: it needs the 198 MiB model and two images. For checking a new pin
//! or a new `ort`; the other tests use a stand-in model.
//!
//! ```sh
//! LAMA_ONNX=lama_fp32.onnx REGION=region.png PAINT=paint.png OUT=out.png \
//!     cargo test --release --test real_model -- --ignored --nocapture
//! ```
//! `REGION` is an RGB image (at most 2048 pixels a side), `PAINT` a greyscale
//! image the same size (white where to remove); `OUT` receives the result.
// ONNX Runtime has no binary for Intel Macs, where the generator only says so.
#![cfg(not(all(target_os = "macos", target_arch = "x86_64")))]
mod support;

use rawmakase_extension::protocol::method::RetouchGenerate;
use sha2::{Digest, Sha256};
use support::{Reply, Server};

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}"))
}

/// `pixels` (`channels` per pixel) with each row padded by `pad` bytes.
fn padded(pixels: &[u8], width: usize, channels: usize, pad: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for row in pixels.chunks(width * channels) {
        out.extend_from_slice(row);
        out.extend(std::iter::repeat_n(0xEE, pad));
    }
    out
}

#[test]
#[ignore = "needs the real model and a photo; see the file's comment"]
fn the_real_model_removes_what_is_painted_on_a_photo() {
    let model = std::fs::read(env("LAMA_ONNX")).unwrap();
    let region = image::open(env("REGION")).unwrap().into_rgb8();
    let paint = image::open(env("PAINT")).unwrap().into_luma8();
    let (w, h) = region.dimensions();
    assert_eq!(paint.dimensions(), (w, h));

    let served = model.clone();
    let server = Server::start(move |_| Reply::bytes("application/octet-stream", served.clone()));
    let mut config = support::config(&server, &[]);
    config.deadline = std::time::Duration::from_secs(120);
    config.lama = rawmakase_ai_retouch::ModelFile {
        url: server.url.clone(),
        sha256: Sha256::digest(&model)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
        bytes: model.len() as u64,
    };
    let host = support::start(config, &[]);

    let base = support::request();
    let (stride, cover) = (w * 3 + 5, w + 3);
    let request = RetouchGenerate {
        generator: rawmakase_ai_retouch::LOCAL_GENERATOR.into(),
        image: rawmakase_extension::protocol::buffer::Buffer {
            width: w,
            height: h,
            stride,
            bytes: u64::from(stride * h),
            ..base.image
        },
        coverage: rawmakase_extension::protocol::buffer::Buffer {
            width: w,
            height: h,
            stride: cover,
            bytes: u64::from(cover * h),
            ..base.coverage
        },
        result: rawmakase_extension::protocol::buffer::Buffer {
            width: w,
            height: h,
            stride: w * 3 + 7,
            bytes: u64::from((w * 3 + 7) * h),
            ..base.result
        },
        ..base
    };
    let started = std::time::Instant::now();
    let (answer, result) = host
        .generate_retouch(
            request,
            &padded(region.as_raw(), w as usize, 3, 5),
            &padded(paint.as_raw(), w as usize, 1, 3),
        )
        .unwrap();
    eprintln!(
        "{} in {:.1} s",
        answer.model_key,
        started.elapsed().as_secs_f64()
    );

    let row = (w * 3 + 7) as usize;
    let mut out = image::RgbImage::new(w, h);
    for (x, y, p) in out.enumerate_pixels_mut() {
        p.0 = result[y as usize * row + x as usize * 3..][..3]
            .try_into()
            .unwrap();
    }
    out.save(env("OUT")).unwrap();
    host.finish().unwrap();
}
