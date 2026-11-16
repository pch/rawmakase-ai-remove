//! A local HTTP server standing in for fal.ai and Replicate, and requests
//! shaped as the app sends them.
#![allow(dead_code)]

use rawmakase_ai_retouch::Config;
use rawmakase_extension::ServeOptions;
use rawmakase_extension::mock::MockHost;
use rawmakase_extension::protocol::buffer::{
    Access, Alpha, Buffer, Meaning, Primaries, Region, Sample, Space, Transfer, White,
};
use rawmakase_extension::protocol::method::{Hello, RetouchGenerate};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// A request the server received.
#[derive(Clone, Debug)]
pub struct Request {
    pub method: String,
    pub path: String,
    /// Names lowercased.
    pub headers: HashMap<String, String>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(String::as_str)
    }
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).expect("a JSON body")
    }
}

/// What the server answers.
pub struct Reply {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
    /// How long to wait before answering.
    pub delay: Duration,
}

impl Reply {
    pub fn json(status: u16, body: Value) -> Self {
        Self {
            status,
            content_type: "application/json",
            body: body.to_string().into_bytes(),
            delay: Duration::ZERO,
        }
    }
    pub fn bytes(content_type: &'static str, body: Vec<u8>) -> Self {
        Self {
            status: 200,
            content_type,
            body,
            delay: Duration::ZERO,
        }
    }
    pub fn after(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }
}

type Handler = dyn Fn(&Request) -> Reply + Send + Sync;

/// The server: every connection is one request, answered by the handler.
pub struct Server {
    pub url: String,
    seen: Arc<Mutex<Vec<Request>>>,
    arrived: Mutex<Receiver<Request>>,
}

impl Server {
    pub fn start(handler: impl Fn(&Request) -> Reply + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("a local port");
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (tell, arrived) = mpsc::channel();
        let handler: Arc<Handler> = Arc::new(handler);
        {
            let seen = seen.clone();
            std::thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let (handler, seen, tell) = (handler.clone(), seen.clone(), tell.clone());
                    std::thread::spawn(move || serve(stream, &*handler, &seen, &tell));
                }
            });
        }
        Self {
            url,
            seen,
            arrived: Mutex::new(arrived),
        }
    }

    /// Every request so far.
    pub fn requests(&self) -> Vec<Request> {
        self.seen.lock().unwrap().clone()
    }

    /// Waits for the next request to arrive (before it is answered).
    pub fn next_arrival(&self, within: Duration) -> Request {
        self.arrived
            .lock()
            .unwrap()
            .recv_timeout(within)
            .expect("a request in time")
    }
}

fn serve(stream: TcpStream, handler: &Handler, seen: &Mutex<Vec<Request>>, tell: &Sender<Request>) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    if reader.read_line(&mut line).unwrap_or(0) == 0 {
        return;
    }
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut headers = HashMap::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).unwrap_or(0) == 0 || header.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    assert!(
        !headers.contains_key("transfer-encoding"),
        "the mock reads Content-Length bodies only"
    );
    let length: usize = headers
        .get("content-length")
        .and_then(|l| l.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0; length];
    if reader.read_exact(&mut body).is_err() {
        return;
    }
    let request = Request {
        method,
        path,
        headers,
        body,
    };
    seen.lock().unwrap().push(request.clone());
    let _ = tell.send(request.clone());
    let reply = handler(&request);
    std::thread::sleep(reply.delay);
    let mut stream = stream;
    let head = format!(
        "HTTP/1.1 {} X\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reply.status,
        reply.content_type,
        reply.body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(&reply.body);
    let _ = stream.flush();
}

/// A config for `server`, with no environment unless `env` gives some.
pub fn config(server: &Server, env: &[(&str, &str)]) -> Config {
    let env: HashMap<String, String> = env
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    Config {
        fal_url: server.url.clone(),
        replicate_url: server.url.clone(),
        env: Arc::new(move |name| env.get(name).cloned()),
        poll_every: Duration::from_millis(10),
        deadline: Duration::from_secs(8),
        lama: Config::default().lama,
    }
}

/// The extension served as the app serves an `api = 3` manifest, after the
/// handshake with `settings`.
pub fn start(config: Config, settings: &[(&str, &str)]) -> MockHost {
    // A data folder of its own: the panel keeps its values there.
    static HOSTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = HOSTS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let data_dir = std::env::temp_dir().join(format!(
        "rawmakase-ai-retouch-test-{}-{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&data_dir);
    start_in(config, settings, data_dir)
}

/// [`start`] with the data folder given, to set up or look at what it holds.
pub fn start_in(
    config: Config,
    settings: &[(&str, &str)],
    data_dir: std::path::PathBuf,
) -> MockHost {
    let host = MockHost::start_with_options(
        rawmakase_ai_retouch::extension_with(config),
        ServeOptions { api: 3 },
        |_, _| Ok(Value::Null),
    );
    let settings: Map<String, Value> = settings
        .iter()
        .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
        .collect();
    let hello = host
        .hello_with(Hello {
            settings,
            extension: "dev.pch.ai-remove".into(),
            contributions: vec![
                "retouch-generator/retouch".into(),
                "retouch-generator/local".into(),
                "panel/retouch".into(),
            ],
            data_dir,
            ..MockHost::hello_params()
        })
        .unwrap();
    assert_eq!(
        hello.implements,
        [
            "panel/retouch",
            "retouch-generator/local",
            "retouch-generator/retouch"
        ]
    );
    host
}

/// Sets the panel as the user would: the mode (`remove` or `replace`) and
/// the prompt.
pub fn ask(host: &MockHost, mode: &str, prompt: &str) {
    let panel = rawmakase_ai_retouch::PANEL;
    host.change(panel, rawmakase_ai_retouch::MODE, Value::from(mode))
        .unwrap();
    host.change(panel, rawmakase_ai_retouch::PROMPT, Value::from(prompt))
        .unwrap();
}

pub const W: u32 = 64;
pub const H: u32 = 48;

/// A `W` x `H` retouch request, its rows padded (strides wider than the
/// rows) to check they are honoured.
pub fn request() -> RetouchGenerate {
    let image = Buffer {
        handle: 0,
        bytes: u64::from((W * 3 + 5) * H),
        width: W,
        height: H,
        stride: W * 3 + 5,
        channels: 3,
        sample: Sample::U8,
        meaning: Meaning::Color {
            primaries: Primaries::Srgb,
            white: White::D65,
            transfer: Transfer::Srgb,
            alpha: Alpha::None,
        },
        region: Region {
            space: Space::ImageFrame,
            rect: [0.25, 0.25, 0.5, 0.5],
            scale: 0.5,
            overlap: 0,
        },
        access: Access::Read,
    };
    let coverage = Buffer {
        handle: 1,
        bytes: u64::from((W + 3) * H),
        stride: W + 3,
        channels: 1,
        meaning: Meaning::Coverage,
        ..image
    };
    let result = Buffer {
        handle: 2,
        bytes: u64::from((W * 3 + 7) * H),
        stride: W * 3 + 7,
        access: Access::Write,
        ..image
    };
    RetouchGenerate {
        generator: rawmakase_ai_retouch::GENERATOR.into(),
        image,
        coverage,
        result,
        image_key: "0123456789abcdef".into(),
    }
}

/// A `w` x `h` request, its rows padded like [`request`]'s, with the region's
/// pixels from `pixel(x, y)` and the painted area from `paint(x, y)`, each in
/// the request's padded layout.
pub fn sized(
    w: u32,
    h: u32,
    pixel: impl Fn(usize, usize) -> [u8; 3],
    paint: impl Fn(usize, usize) -> u8,
) -> (RetouchGenerate, Vec<u8>, Vec<u8>) {
    let base = request();
    let (image_stride, coverage_stride, result_stride) = (w * 3 + 5, w + 3, w * 3 + 7);
    let request = RetouchGenerate {
        image: Buffer {
            width: w,
            height: h,
            stride: image_stride,
            bytes: u64::from(image_stride * h),
            ..base.image
        },
        coverage: Buffer {
            width: w,
            height: h,
            stride: coverage_stride,
            bytes: u64::from(coverage_stride * h),
            ..base.coverage
        },
        result: Buffer {
            width: w,
            height: h,
            stride: result_stride,
            bytes: u64::from(result_stride * h),
            ..base.result
        },
        ..base
    };
    let mut image = vec![0xEE; (image_stride * h) as usize];
    let mut coverage = vec![0; (coverage_stride * h) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let i = y * image_stride as usize + x * 3;
            image[i..i + 3].copy_from_slice(&pixel(x, y));
            coverage[y * coverage_stride as usize + x] = paint(x, y);
        }
    }
    (request, image, coverage)
}

/// The result's pixel at (x, y) of a `w`-wide request made by [`sized`].
pub fn sized_pixel(result: &[u8], w: u32, x: usize, y: usize) -> [u8; 3] {
    let stride = (w * 3 + 7) as usize;
    result[y * stride + x * 3..][..3].try_into().unwrap()
}

/// The region's pixels, in the request's padded layout.
pub fn image() -> Vec<u8> {
    let r = request();
    let mut data = vec![0xEE; r.image.bytes as usize];
    for y in 0..H as usize {
        for x in 0..W as usize {
            let i = y * r.image.stride as usize + x * 3;
            data[i..i + 3].copy_from_slice(&[(x * 4) as u8, (y * 5) as u8, 200]);
        }
    }
    data
}

/// The painted area: a feathered disc of radius 8 at (20, 24), in the
/// request's padded layout.
pub fn coverage() -> Vec<u8> {
    let r = request();
    let mut data = vec![0; r.coverage.bytes as usize];
    for y in 0..H as usize {
        for x in 0..W as usize {
            let d = (((x as f64 - 20.0).powi(2) + (y as f64 - 24.0).powi(2)).sqrt()) as i64;
            let value = (255 - (d - 6).max(0) * 120).clamp(0, 255);
            data[y * r.coverage.stride as usize + x] = value as u8;
        }
    }
    data
}

/// Whether (x, y) is painted at all in [`coverage`].
pub fn painted(x: usize, y: usize) -> bool {
    let r = request();
    coverage()[y * r.coverage.stride as usize + x] > 0
}

/// A solid `width` x `height` PNG of `rgb`.
pub fn png(width: u32, height: u32, rgb: [u8; 3]) -> Vec<u8> {
    let image = image::RgbImage::from_pixel(width, height, image::Rgb(rgb));
    let mut out = std::io::Cursor::new(Vec::new());
    image.write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

/// The result's pixel at (x, y), from its padded layout.
pub fn pixel(result: &[u8], x: usize, y: usize) -> [u8; 3] {
    let stride = request().result.stride as usize;
    result[y * stride + x * 3..][..3].try_into().unwrap()
}

/// The bytes of a base64 data URI.
pub fn from_data_uri(uri: &str) -> Vec<u8> {
    use base64::Engine;
    let (_, data) = uri.split_once(";base64,").expect("a base64 data URI");
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .unwrap()
}
