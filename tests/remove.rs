//! AI Remove end to end: the SDK's mock host stands in for the app (serving
//! `api = 3`, through shared memory), and a local HTTP server for fal.ai and
//! Replicate. Nothing here reaches the network.
mod support;

use rawmakase_ai_retouch::{
    DEFAULT_REPLICATE_MODEL, FAL, FAL_EDIT_MODEL, FAL_MODEL, FIBO_EDIT, FLUX_FILL, GENFILL,
    NANO_BANANA, REPLICATE, REPLICATE_EDIT_MODEL,
};
use rawmakase_extension::ErrorCode;
use rawmakase_extension::protocol::manifest::{Manifest, RETOUCH_GENERATOR_API, SettingControl};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use support::{H, Reply, Request, Server, W};

const FAL_PATH: &str = "/bria/fibo-edit-1.5/eraser";
const KEY: &str = "fal-test-key-0001";
const TOKEN: &str = "r8_test0token0001";
const SENTINEL: &str = "sk-SENTINEL-1234567890abcdef";

fn data_png(width: u32, height: u32, rgb: [u8; 3]) -> String {
    use base64::Engine;
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(support::png(width, height, rgb))
    )
}

/// fal's answer with `url` as the image.
fn fal_answer(url: &str) -> Reply {
    Reply::json(
        200,
        json!({"image": {"url": url, "width": W, "height": H, "content_type": "image/png"}, "seed": 7}),
    )
}

fn every_pixel_is(result: &[u8], rgb: [u8; 3]) {
    for y in 0..H as usize {
        for x in 0..W as usize {
            assert_eq!(support::pixel(result, x, y), rgb, "{x},{y}");
        }
    }
}

#[test]
fn fal_writes_its_image_into_the_result_and_is_sent_a_grown_binary_mask() {
    let server = Server::start(|r| match r.path.as_str() {
        FAL_PATH => fal_answer(&data_png(W, H, [10, 20, 30])),
        _ => Reply::json(404, json!({})),
    });
    let host = support::start(support::config(&server, &[]), &[("fal-key", KEY)]);
    let (answer, result) = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    assert_eq!(answer.model, FAL_MODEL);
    assert_eq!(
        answer.model_key,
        "fal|bria/fibo-edit-1.5/eraser|composite|1"
    );
    every_pixel_is(&result, [10, 20, 30]);

    let [call] = &server.requests()[..] else {
        panic!("one call: {:?}", server.requests());
    };
    assert_eq!(call.method, "POST");
    assert_eq!(call.header("authorization"), Some(&*format!("Key {KEY}")));
    assert_eq!(call.header("x-fal-store-io"), Some("0"));
    let body = call.json();
    assert_eq!(body["composite"], json!(true));
    assert_eq!(body["sync_mode"], json!(true));

    // The region as given, unpadded.
    let sent =
        image::load_from_memory(&support::from_data_uri(body["image_url"].as_str().unwrap()))
            .unwrap()
            .into_rgb8();
    assert_eq!(sent.dimensions(), (W, H));
    let source = support::image();
    let stride = support::request().image.stride as usize;
    for (x, y, p) in sent.enumerate_pixels() {
        let i = y as usize * stride + x as usize * 3;
        assert_eq!(p.0, source[i..i + 3], "{x},{y}");
    }

    // The mask: the image's size, binary, everything painted and 6 px more.
    let mask_png = support::from_data_uri(body["mask_url"].as_str().unwrap());
    assert!(mask_png.starts_with(b"\x89PNG"), "a lossless mask");
    let mask = image::load_from_memory(&mask_png).unwrap().into_luma8();
    assert_eq!(mask.dimensions(), (W, H));
    assert!(mask.pixels().all(|p| p.0[0] == 0 || p.0[0] == 255));
    for y in 0..H as usize {
        for x in 0..W as usize {
            if support::painted(x, y) {
                assert_eq!(mask.get_pixel(x as u32, y as u32).0[0], 255, "{x},{y}");
            }
        }
    }
    // The faintest paint reaches (28, 24); the mask 6 px past it.
    assert!(support::painted(28, 24) && !support::painted(29, 24));
    assert_eq!(mask.get_pixel(34, 24).0[0], 255);
    assert_eq!(mask.get_pixel(35, 24).0[0], 0);
    assert_eq!(mask.get_pixel(60, 5).0[0], 0);
    host.finish().unwrap();
}

#[test]
fn an_image_at_another_size_is_fetched_without_the_key_and_resized() {
    let server = Server::start(|r| match r.path.as_str() {
        FAL_PATH => fal_answer(&format!(
            "http://{}/files/out.png",
            r.header("host").unwrap()
        )),
        "/files/out.png" => Reply::bytes("image/png", support::png(W / 2, H / 2, [200, 100, 50])),
        _ => Reply::json(404, json!({})),
    });
    let host = support::start(support::config(&server, &[]), &[("fal-key", KEY)]);
    let (_, result) = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    every_pixel_is(&result, [200, 100, 50]);
    let requests = server.requests();
    let download = requests
        .iter()
        .find(|r| r.path == "/files/out.png")
        .unwrap();
    assert_eq!(download.header("authorization"), None);
    host.finish().unwrap();
}

#[test]
fn without_a_key_it_needs_setup_and_calls_nothing() {
    let server = Server::start(|_| Reply::json(500, json!({})));
    let host = support::start(support::config(&server, &[]), &[("fal-key", "  ")]);
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::NeedsSetup);
    assert!(error.message.contains("fal.ai API key"), "{error}");
    assert!(
        error
            .message
            .contains("Preferences > Extensions > AI Retouch"),
        "{error}"
    );
    host.finish().unwrap();

    let host = support::start(
        support::config(&server, &[]),
        &[("provider", REPLICATE), ("replicate-token", "")],
    );
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::NeedsSetup);
    assert!(error.message.contains("Replicate API token"), "{error}");
    host.finish().unwrap();
    assert!(server.requests().is_empty());
}

#[test]
fn an_empty_setting_falls_back_to_the_environment() {
    let server = Server::start(|r| match r.path.as_str() {
        FAL_PATH => fal_answer(&data_png(W, H, [1, 2, 3])),
        _ => Reply::json(404, json!({})),
    });
    let config = support::config(&server, &[("FAL_KEY", " env-fal-key-1 ")]);
    let host = support::start(config, &[("fal-key", "")]);
    host.generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    assert_eq!(
        server.requests()[0].header("authorization"),
        Some("Key env-fal-key-1")
    );
    host.finish().unwrap();

    // The setting wins over the environment.
    let config = support::config(&server, &[("FAL_KEY", "env-fal-key-1")]);
    let host = support::start(config, &[("fal-key", KEY)]);
    host.generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    assert_eq!(
        server.requests()[1].header("authorization"),
        Some(&*format!("Key {KEY}"))
    );
    host.finish().unwrap();
}

/// The error a fal call answered with `reply` turns into.
fn fal_error(
    reply: impl Fn(&Request) -> Reply + Send + Sync + 'static,
) -> rawmakase_extension::RemoteError {
    let server = Server::start(reply);
    let host = support::start(support::config(&server, &[]), &[("fal-key", SENTINEL)]);
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    host.finish().unwrap();
    error
}

#[test]
fn fal_statuses_map_to_what_the_user_can_do() {
    for status in [401, 403] {
        let error = fal_error(move |_| {
            Reply::json(
                status,
                json!({"detail": "No user found for Key ID and Secret"}),
            )
        });
        assert_eq!(error.code, ErrorCode::NeedsSetup, "{status}");
        assert!(
            error
                .message
                .contains("Preferences > Extensions > AI Retouch"),
            "{error}"
        );
    }
    let error = fal_error(|_| {
        Reply::json(
            429,
            json!({"detail": "Too many concurrent requests", "type": "concurrent_requests_limit"}),
        )
    });
    assert_eq!(error.code, ErrorCode::Unavailable);
    assert!(
        error.message.contains("Too many concurrent requests"),
        "{error}"
    );
    for status in [500, 503, 504] {
        let error = fal_error(move |_| {
            Reply::json(
                status,
                json!({"detail": "Runner failed", "error_type": "runner_error"}),
            )
        });
        assert_eq!(error.code, ErrorCode::Unavailable, "{status}");
    }
    let error = fal_error(|_| {
        Reply::json(
            422,
            json!({"detail": [{
                "loc": ["body", "image_url"], "msg": "The content could not be processed because it contained material flagged by a content checker.",
                "type": "content_policy_violation", "url": "https://docs.fal.ai/errors#content_policy_violation",
                "categories": ["violence"]
            }]}),
        )
    });
    assert_eq!(error.code, ErrorCode::Refused);
    assert!(
        error.message.contains("flagged by a content checker"),
        "{error}"
    );
    let error = fal_error(|_| {
        Reply::json(
            422,
            json!({"detail": [{"loc": ["body", "mask_url"], "msg": "Mask size must match image", "type": "image_size_mismatch"}]}),
        )
    });
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert!(
        error.message.contains("Mask size must match image"),
        "{error}"
    );
    // Nothing listening: the network failure is unavailable, not a crash.
    let closed = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", listener.local_addr().unwrap())
    };
    let mut config = support::config(&Server::start(|_| Reply::json(500, json!({}))), &[]);
    config.fal_url = closed;
    let host = support::start(config, &[("fal-key", KEY)]);
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Unavailable, "{error}");
    host.finish().unwrap();
}

#[test]
fn a_key_never_reaches_an_error_even_when_the_service_echoes_it() {
    // Each answer repeats the request's Authorization header, as a careless
    // service might.
    let echo = |status: u16| {
        move |r: &Request| {
            let said = format!(
                "bad credentials: {}",
                r.header("authorization").unwrap_or("")
            );
            match status {
                422 => Reply::json(
                    422,
                    json!({"detail": [{"loc": [], "msg": said, "type": "content_policy_violation"}]}),
                ),
                s => Reply::json(s, json!({"detail": said})),
            }
        }
    };
    for status in [400, 401, 403, 404, 422, 429, 500] {
        let error = fal_error(echo(status));
        assert!(!error.message.contains("SENTINEL"), "{status}: {error}");
        assert!(
            !error.message.contains("1234567890abcdef"),
            "{status}: {error}"
        );
        assert!(error.message.len() <= 400, "{status}: {error}");
    }
    // A plain-text body too.
    let error = fal_error(|r| Reply {
        status: 500,
        content_type: "text/plain",
        body: format!("upstream said {}", r.header("authorization").unwrap()).into_bytes(),
        delay: Duration::ZERO,
    });
    assert!(!error.message.contains("SENTINEL"), "{error}");

    // Replicate: a failed prediction quoting the token.
    let server = replicate_server(
        |_, token, _| json!({"id": "p1", "status": "failed", "error": format!("auth {token} rejected")}),
    );
    let host = support::start(
        support::config(&server, &[]),
        &[("provider", REPLICATE), ("replicate-token", SENTINEL)],
    );
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert!(!error.message.contains("SENTINEL"), "{error}");
    assert!(!error.message.contains("1234567890abcdef"), "{error}");
    host.finish().unwrap();
}

#[test]
fn cancelling_a_fal_call_in_flight_answers_cancelled_at_once() {
    let server =
        Server::start(|_| fal_answer(&data_png(W, H, [0, 0, 0])).after(Duration::from_secs(30)));
    let host = support::start(support::config(&server, &[]), &[("fal-key", KEY)]);
    let id = host
        .send_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    server.next_arrival(Duration::from_secs(5));
    let asked = Instant::now();
    host.cancel(id);
    let error = host.wait_retouch(id).unwrap_err();
    assert_eq!(error.code, ErrorCode::Cancelled);
    assert!(
        asked.elapsed() < Duration::from_secs(1),
        "{:?}",
        asked.elapsed()
    );
    host.finish().unwrap();
}

/// A Replicate stand-in: uploads, deletes and cancels answered as the API
/// does; a prediction's state from `state(polls so far, token, delivery)`, 0
/// for the create call's answer. `delivery` is the address of another server
/// standing in for `replicate.delivery`, which serves the output at
/// `/delivery/out.png`, and refuses a request carrying a token.
fn replicate_server(state: impl Fn(usize, &str, &str) -> Value + Send + Sync + 'static) -> Server {
    let delivery = Server::start(|r| match r.header("authorization") {
        Some(_) => Reply::json(
            400,
            json!({"detail": "a token was sent to the delivery host"}),
        ),
        None if r.path == "/delivery/out.png" => {
            Reply::bytes("image/png", support::png(W, H, [9, 8, 7]))
        }
        None => Reply::json(404, json!({})),
    });
    let delivery = delivery.url.trim_start_matches("http://").to_string();
    let files = AtomicUsize::new(0);
    let polls = AtomicUsize::new(0);
    Server::start(move |r| {
        let token = r
            .header("authorization")
            .and_then(|a| a.strip_prefix("Bearer "))
            .unwrap_or("");
        let host = delivery.as_str();
        match (r.method.as_str(), r.path.as_str()) {
            ("POST", "/v1/files") => {
                let n = files.fetch_add(1, Ordering::SeqCst);
                Reply::json(
                    201,
                    json!({"id": format!("file{n}"), "urls": {"get": format!("https://api.replicate.com/v1/files/file{n}")}}),
                )
            }
            ("DELETE", _) => Reply::json(204, json!({})),
            ("POST", p) if p.ends_with("/cancel") => {
                Reply::json(200, json!({"id": "p1", "status": "canceled"}))
            }
            ("POST", _) => Reply::json(201, state(0, token, host)),
            ("GET", "/v1/predictions/p1") => Reply::json(
                200,
                state(polls.fetch_add(1, Ordering::SeqCst) + 1, token, host),
            ),
            _ => Reply::json(404, json!({"detail": "not found"})),
        }
    })
}

/// A prediction that succeeds at poll `after`, its output on `host`.
fn succeeds_at(after: usize) -> impl Fn(usize, &str, &str) -> Value + Send + Sync + 'static {
    move |polls, _, host| match polls {
        0 => json!({"id": "p1", "status": "starting"}),
        n if n < after => json!({"id": "p1", "status": "processing"}),
        _ => {
            json!({"id": "p1", "status": "succeeded", "output": format!("http://{host}/delivery/out.png")})
        }
    }
}

fn replicate_host(
    server: &Server,
    settings: &[(&str, &str)],
) -> rawmakase_extension::mock::MockHost {
    let mut all = vec![("provider", REPLICATE), ("replicate-token", TOKEN)];
    all.extend_from_slice(settings);
    support::start(support::config(server, &[]), &all)
}

#[test]
fn replicate_uploads_waits_and_polls_past_prefer_wait() {
    let server = replicate_server(succeeds_at(3));
    let host = replicate_host(&server, &[("replicate-model", "")]);
    let (answer, result) = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    assert_eq!(answer.model, DEFAULT_REPLICATE_MODEL);
    assert_eq!(answer.model_key, "replicate|bria/eraser|1");
    every_pixel_is(&result, [9, 8, 7]);

    let requests = server.requests();
    let bearer = format!("Bearer {TOKEN}");
    let uploads: Vec<&Request> = requests.iter().filter(|r| r.path == "/v1/files").collect();
    assert_eq!(uploads.len(), 2);
    for upload in &uploads {
        assert_eq!(upload.header("authorization"), Some(&*bearer));
        assert!(
            upload
                .header("content-type")
                .unwrap()
                .starts_with("multipart/form-data; boundary=")
        );
        assert!(upload.body.windows(4).any(|w| w == b"\x89PNG"));
        assert!(String::from_utf8_lossy(&upload.body).contains("name=\"content\""));
    }
    let create = requests
        .iter()
        .find(|r| r.path == "/v1/models/bria/eraser/predictions")
        .expect("a prediction of the official model");
    assert_eq!(create.header("prefer"), Some("wait=60"));
    // What is left of the 8 s deadline after the uploads, not a fresh one.
    let left: u64 = create
        .header("cancel-after")
        .and_then(|v| v.strip_suffix('s'))
        .and_then(|v| v.parse().ok())
        .unwrap();
    assert!((5..=8).contains(&left), "{left}");
    assert_eq!(create.header("authorization"), Some(&*bearer));
    let input = &create.json()["input"];
    assert_eq!(
        input["image"],
        json!("https://api.replicate.com/v1/files/file0")
    );
    assert_eq!(
        input["mask"],
        json!("https://api.replicate.com/v1/files/file1")
    );
    let polls = requests
        .iter()
        .filter(|r| r.method == "GET" && r.path == "/v1/predictions/p1")
        .count();
    assert_eq!(polls, 3);
    let deletes: Vec<&str> = requests
        .iter()
        .filter(|r| r.method == "DELETE")
        .map(|r| r.path.as_str())
        .collect();
    assert_eq!(deletes, ["/v1/files/file0", "/v1/files/file1"]);
    // The output came from the delivery host, which refuses a token.
    host.finish().unwrap();
}

#[test]
fn a_replicate_model_with_a_version_and_a_bad_one() {
    let server = replicate_server(succeeds_at(0));
    let host = replicate_host(&server, &[("replicate-model", "someone/eraser:0123abcd")]);
    let (answer, _) = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    assert_eq!(answer.model, "someone/eraser:0123abcd");
    let create = server
        .requests()
        .into_iter()
        .find(|r| r.path == "/v1/predictions")
        .unwrap();
    assert_eq!(create.json()["version"], json!("0123abcd"));
    host.finish().unwrap();

    let server = replicate_server(succeeds_at(0));
    let host = replicate_host(&server, &[("replicate-model", "not a model")]);
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert!(error.message.contains("Replicate model"), "{error}");
    assert!(server.requests().is_empty());
    host.finish().unwrap();
}

#[test]
fn replicate_failures_map_to_what_the_user_can_do() {
    let error_with = |server: Server| {
        let host = replicate_host(&server, &[]);
        let error = host
            .generate_retouch(support::request(), &support::image(), &support::coverage())
            .unwrap_err();
        host.finish().unwrap();
        (error, server.requests())
    };
    // A content refusal: the model fails the prediction saying so.
    let (error, requests) = error_with(replicate_server(|polls, _, _| match polls {
        0 => json!({"id": "p1", "status": "processing"}),
        _ => {
            json!({"id": "p1", "status": "failed", "error": "Request blocked: the image did not pass content moderation"})
        }
    }));
    assert_eq!(error.code, ErrorCode::Refused);
    assert!(error.message.contains("content moderation"), "{error}");
    assert_eq!(requests.iter().filter(|r| r.method == "DELETE").count(), 2);
    // Any other failure is unavailable.
    let (error, _) = error_with(replicate_server(
        |_, _, _| json!({"id": "p1", "status": "failed", "error": "CUDA out of memory"}),
    ));
    assert_eq!(error.code, ErrorCode::Unavailable);
    // A token Replicate doesn't accept, at the first call.
    let (error, requests) = error_with(Server::start(|_| {
        Reply::json(
            401,
            json!({"title": "Unauthenticated", "detail": "Invalid token", "status": 401}),
        )
    }));
    assert_eq!(error.code, ErrorCode::NeedsSetup);
    assert!(
        error
            .message
            .contains("Preferences > Extensions > AI Retouch"),
        "{error}"
    );
    assert_eq!(requests.len(), 1, "no retry: {requests:?}");
    // Throttled.
    let (error, _) = error_with(Server::start(|_| {
        Reply::json(
            429,
            json!({"detail": "Request was throttled. Expected available in 1 second."}),
        )
    }));
    assert_eq!(error.code, ErrorCode::Unavailable);
    assert!(error.message.contains("throttled"), "{error}");
}

#[test]
fn the_replicate_token_falls_back_to_the_environment() {
    let server = replicate_server(succeeds_at(0));
    let config = support::config(&server, &[("REPLICATE_API_TOKEN", "r8_fromtheenv")]);
    let host = support::start(config, &[("provider", REPLICATE)]);
    host.generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    assert!(
        server
            .requests()
            .iter()
            .all(|r| r.header("authorization") == Some("Bearer r8_fromtheenv"))
    );
    host.finish().unwrap();
}

#[test]
fn cancelling_while_replicate_runs_answers_at_once_and_cancels_the_prediction() {
    let server = replicate_server(|_, _, _| json!({"id": "p1", "status": "processing"}));
    let host = replicate_host(&server, &[]);
    let id = host
        .send_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    // Wait until it polls.
    while server.next_arrival(Duration::from_secs(5)).method != "GET" {}
    let asked = Instant::now();
    host.cancel(id);
    let error = host.wait_retouch(id).unwrap_err();
    assert_eq!(error.code, ErrorCode::Cancelled);
    assert!(
        asked.elapsed() < Duration::from_secs(1),
        "{:?}",
        asked.elapsed()
    );
    // The prediction is cancelled on Replicate, and the uploads deleted.
    let until = Instant::now() + Duration::from_secs(5);
    while Instant::now() < until {
        let requests = server.requests();
        if requests
            .iter()
            .any(|r| r.path == "/v1/predictions/p1/cancel")
            && requests.iter().filter(|r| r.method == "DELETE").count() == 2
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let requests = server.requests();
    assert!(
        requests
            .iter()
            .any(|r| r.path == "/v1/predictions/p1/cancel")
    );
    assert_eq!(requests.iter().filter(|r| r.method == "DELETE").count(), 2);
    host.finish().unwrap();
}

#[test]
fn a_prediction_past_the_deadline_is_cancelled_and_unavailable() {
    let server = replicate_server(|_, _, _| json!({"id": "p1", "status": "processing"}));
    let mut config = support::config(&server, &[]);
    // Just over the least Cancel-After, so the prediction starts, then runs out.
    config.deadline = Duration::from_millis(5300);
    let host = support::start(
        config,
        &[("provider", REPLICATE), ("replicate-token", TOKEN)],
    );
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Unavailable);
    assert!(error.message.contains("timed out"), "{error}");
    assert!(
        server
            .requests()
            .iter()
            .any(|r| r.path == "/v1/predictions/p1/cancel")
    );
    host.finish().unwrap();
}

#[test]
fn the_manifest_declares_what_the_program_implements() {
    let manifest = Manifest::parse(include_str!("../rawmakase-extension.toml")).unwrap();
    let e = &manifest.extension;
    assert_eq!(e.id, "dev.pch.ai-remove");
    assert_eq!(e.name, "AI Retouch");
    assert_eq!(e.version.to_string(), env!("CARGO_PKG_VERSION"));
    assert_eq!(e.api, RETOUCH_GENERATOR_API);
    // RAWmakase 0.5.0 and later.
    assert_eq!(e.rawmakase.to_string(), ">=0.5");
    assert_eq!(
        e.repository.as_deref(),
        Some("https://github.com/pch/rawmakase-ai-retouch")
    );
    assert!(manifest.access.network);
    let program = manifest.program.as_ref().unwrap();
    assert_eq!(program.path, concat!("bin/", env!("CARGO_PKG_NAME")));
    use rawmakase_extension::protocol::manifest::PanelPlace;
    let [generator, local] = &manifest.retouch_generators[..] else {
        panic!("two retouch generators");
    };
    // AI Retouch first: the rail's tool is drawn and named by the first, and
    // each is a tab in its drawer, saying below the tabs what it is.
    assert!(
        generator
            .description
            .as_deref()
            .is_some_and(|d| d.contains("API key"))
    );
    assert!(
        local
            .description
            .as_deref()
            .is_some_and(|d| d.contains("nothing is uploaded"))
    );
    // The one on this computer uploads nothing, so the app never asks the
    // user's consent for it, and it has no drawer: nothing to choose.
    assert_eq!(local.id, rawmakase_ai_retouch::LOCAL_GENERATOR);
    assert_eq!(local.name, "AI Local Remove");
    assert!(!local.uploads);
    assert_eq!(local.panel, None);
    assert_eq!(generator.id, rawmakase_ai_retouch::GENERATOR);
    assert_eq!(generator.name, "AI Retouch");
    assert!(generator.uploads);
    // Remove or Replace, and the prompt, in the tool's drawer.
    assert_eq!(
        generator.panel.as_deref(),
        Some(rawmakase_ai_retouch::PANEL)
    );
    let [panel] = &manifest.panels[..] else {
        panic!("one panel");
    };
    assert_eq!(panel.place, PanelPlace::Generator);

    // The program implements exactly what the manifest contributes.
    let server = Server::start(|_| Reply::json(500, json!({})));
    let host = rawmakase_extension::mock::MockHost::start_with_options(
        rawmakase_ai_retouch::extension_with(support::config(&server, &[])),
        rawmakase_extension::ServeOptions { api: e.api },
        |_, _| Ok(Value::Null),
    );
    let hello = host
        .hello_with(rawmakase_extension::protocol::method::Hello {
            contributions: manifest.contributions(),
            ..rawmakase_extension::mock::MockHost::hello_params()
        })
        .unwrap();
    // The same contributions, in whatever order each lists them.
    let sorted = |mut v: Vec<String>| {
        v.sort();
        v
    };
    assert_eq!(sorted(hello.implements), sorted(manifest.contributions()));
    host.finish().unwrap();

    // The settings the program reads, with the defaults it assumes.
    use rawmakase_ai_retouch::setting;
    let control = |id: &str| {
        let s = manifest
            .settings
            .iter()
            .find(|s| s.id == id)
            .unwrap_or_else(|| panic!("{id}"));
        assert!(s.help.is_some(), "{id} has help");
        (s.control.clone(), s.default.clone())
    };
    assert_eq!(
        control(setting::PROVIDER),
        (
            SettingControl::Segmented(vec![FAL.into(), REPLICATE.into()]),
            json!(FAL)
        )
    );
    assert_eq!(
        control(setting::FAL_KEY),
        (SettingControl::Secret, json!(""))
    );
    assert_eq!(
        control(setting::REPLICATE_TOKEN),
        (SettingControl::Secret, json!(""))
    );
    assert_eq!(
        control(setting::REPLICATE_MODEL),
        (SettingControl::Text, json!(DEFAULT_REPLICATE_MODEL))
    );
    assert_eq!(
        control(setting::REPLACE_MODEL),
        (
            SettingControl::Choice(vec![
                NANO_BANANA.into(),
                GENFILL.into(),
                FLUX_FILL.into(),
                FIBO_EDIT.into()
            ]),
            json!(NANO_BANANA)
        )
    );
    assert_eq!(manifest.settings.len(), 5);
    // Each service's settings show only while it is chosen.
    let shown_for = |id: &str| {
        let s = manifest.settings.iter().find(|s| s.id == id).unwrap();
        s.visible_when.as_ref().map(|w| {
            assert_eq!(w.setting, setting::PROVIDER);
            w.equals.as_str().unwrap().to_string()
        })
    };
    assert_eq!(shown_for(setting::PROVIDER), None);
    assert_eq!(shown_for(setting::FAL_KEY).as_deref(), Some(FAL));
    assert_eq!(shown_for(setting::REPLACE_MODEL).as_deref(), Some(FAL));
    assert_eq!(
        shown_for(setting::REPLICATE_TOKEN).as_deref(),
        Some(REPLICATE)
    );
    assert_eq!(
        shown_for(setting::REPLICATE_MODEL).as_deref(),
        Some(REPLICATE)
    );
}

/// Replicate's files and predictions, each upload answered after `upload`,
/// every poll with `poll`'s status.
fn slow_replicate(upload: Duration, poll: u16) -> Server {
    let files = AtomicUsize::new(0);
    Server::start(move |r| match (r.method.as_str(), r.path.as_str()) {
        ("POST", "/v1/files") => {
            let n = files.fetch_add(1, Ordering::SeqCst);
            Reply::json(
                201,
                json!({"id": format!("file{n}"), "urls": {"get": format!("https://api.replicate.com/v1/files/file{n}")}}),
            )
            .after(upload)
        }
        ("DELETE", _) => Reply::json(204, json!({})),
        ("POST", p) if p.ends_with("/cancel") => {
            Reply::json(200, json!({"id": "p1", "status": "canceled"}))
        }
        ("POST", _) => Reply::json(201, json!({"id": "p1", "status": "starting"})),
        ("GET", "/v1/predictions/p1") => Reply::json(
            poll,
            json!({"id": "p1", "status": "processing", "detail": "down"}),
        ),
        _ => Reply::json(404, json!({"detail": "not found"})),
    })
}

fn creates(server: &Server) -> usize {
    server
        .requests()
        .iter()
        .filter(|r| {
            r.method == "POST" && r.path.contains("/predictions") && !r.path.ends_with("/cancel")
        })
        .count()
}

/// A token pasted into the model setting by mistake never reaches the error.
#[test]
fn a_token_in_the_model_setting_never_reaches_the_error() {
    let server = Server::start(|_| Reply::json(500, json!({})));
    let host = replicate_host(
        &server,
        &[("replicate-model", SENTINEL), ("replicate-token", SENTINEL)],
    );
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::InvalidRequest);
    assert!(!error.message.contains("SENTINEL"), "{error}");
    assert!(server.requests().is_empty());
    host.finish().unwrap();
}

/// Cancelled during the first upload: nothing more is uploaded, and no
/// prediction is ever started.
#[test]
fn a_cancel_during_an_upload_starts_nothing_more() {
    let server = slow_replicate(Duration::from_millis(300), 200);
    let host = replicate_host(&server, &[]);
    let id = host
        .send_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    while server.next_arrival(Duration::from_secs(5)).path != "/v1/files" {}
    host.cancel(id);
    assert_eq!(
        host.wait_retouch(id).unwrap_err().code,
        ErrorCode::Cancelled
    );
    std::thread::sleep(Duration::from_millis(800));
    let uploads = server
        .requests()
        .iter()
        .filter(|r| r.path == "/v1/files")
        .count();
    assert_eq!(uploads, 1);
    assert_eq!(creates(&server), 0);
    host.finish().unwrap();
}

/// The deadline is one for the whole request: uploads that use it up leave
/// no time to start a paid prediction, and none is started.
#[test]
fn uploads_that_use_up_the_deadline_start_no_prediction() {
    let server = slow_replicate(Duration::from_millis(200), 200);
    let mut config = support::config(&server, &[]);
    config.deadline = Duration::from_millis(300);
    let host = support::start(
        config,
        &[("provider", REPLICATE), ("replicate-token", TOKEN)],
    );
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Unavailable);
    std::thread::sleep(Duration::from_millis(500));
    assert_eq!(creates(&server), 0);
    host.finish().unwrap();
}

/// A poll that fails leaves the known prediction cancelled on Replicate.
#[test]
fn a_failed_poll_cancels_the_prediction() {
    let server = slow_replicate(Duration::ZERO, 503);
    let host = replicate_host(&server, &[]);
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Unavailable);
    assert!(
        server
            .requests()
            .iter()
            .any(|r| r.path == "/v1/predictions/p1/cancel")
    );
    host.finish().unwrap();
}

/// With less than Replicate's least `Cancel-After` (5 s) left once uploaded,
/// no prediction is started: it couldn't be bounded.
#[test]
fn too_little_time_left_for_a_bounded_prediction_starts_none() {
    let server = slow_replicate(Duration::from_millis(600), 200);
    let mut config = support::config(&server, &[]);
    config.deadline = Duration::from_secs(6);
    let host = support::start(
        config,
        &[("provider", REPLICATE), ("replicate-token", TOKEN)],
    );
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::Unavailable);
    assert!(error.message.contains("timed out"), "{error}");
    assert_eq!(creates(&server), 0);
    assert_eq!(
        server
            .requests()
            .iter()
            .filter(|r| r.method == "DELETE")
            .count(),
        2
    );
    host.finish().unwrap();
}

/// The one fal.ai call a request made, answered with a flat image, and its
/// answer.
fn fal_call(
    settings: &[(&str, &str)],
    mode: &str,
    prompt: Option<&str>,
) -> (
    rawmakase_extension::protocol::method::RetouchGenerated,
    Request,
) {
    // Each model's documented answer: Bria's an `image`, FLUX's and Nano
    // Banana's `images` (Nano Banana's at a size of its own).
    let server = Server::start(|r| match r.path.as_str() {
        "/bria/fibo-edit-1.5/eraser" | "/bria/fibo-edit-1.5/edit" | "/bria/genfill/v2" => {
            fal_answer(&data_png(W, H, [40, 50, 60]))
        }
        "/fal-ai/nano-banana/edit" => Reply::json(
            200,
            json!({"images": [{"url": data_png(1024, 768, [40, 50, 60]),
                               "content_type": "image/png"}],
                   "description": "done"}),
        ),
        "/fal-ai/flux-pro/v1/fill" => Reply::json(
            200,
            json!({"images": [{"url": data_png(W, H, [40, 50, 60]), "width": W, "height": H,
                               "content_type": "image/png"}],
                   "prompt": "a bench", "seed": 7, "has_nsfw_concepts": [false]}),
        ),
        _ => Reply::json(404, json!({})),
    });
    let mut all = vec![("fal-key", KEY)];
    all.extend_from_slice(settings);
    let host = support::start(support::config(&server, &[]), &all);
    support::ask(&host, mode, prompt.unwrap_or(""));
    let (answer, result) = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap();
    every_pixel_is(&result, [40, 50, 60]);
    host.finish().unwrap();
    let [call] = &server.requests()[..] else {
        panic!("one call: {:?}", server.requests());
    };
    (answer, call.clone())
}

#[test]
fn a_removal_with_text_goes_to_an_editor_that_follows_it() {
    // Without text: the eraser, as before.
    let (answer, call) = fal_call(&[], "remove", None);
    assert_eq!(
        (answer.model.as_str(), call.path.as_str()),
        (FAL_MODEL, FAL_PATH)
    );
    assert!(call.json().get("instruction").is_none());

    // With: FIBO Edit, the region its one reference and the mask what to edit.
    let (answer, call) = fal_call(&[], "remove", Some("remove the dogs, keep the people"));
    assert_eq!(answer.model, FAL_EDIT_MODEL);
    assert_eq!(answer.model_key, "fal|bria/fibo-edit-1.5/edit|1");
    assert_eq!(call.path, "/bria/fibo-edit-1.5/edit");
    let body = call.json();
    assert_eq!(
        body["instruction"],
        json!("remove the dogs, keep the people")
    );
    let references = body["image_urls"].as_array().unwrap();
    assert_eq!(references.len(), 1);
    assert!(
        references[0]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
    assert!(
        body["mask_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
    assert_eq!(body["sync_mode"], json!(true));
    assert!(body.get("composite").is_none());
    assert_eq!(call.header("x-fal-store-io"), Some("0"));
}

#[test]
fn a_replacement_goes_to_the_model_chosen() {
    // Nano Banana by default: the region with the prompt, no mask, at the
    // aspect ratio of its list it is padded to (4:3 already); its answer, at
    // a size of its own, comes back to the region's.
    let (answer, call) = fal_call(&[], "replace", Some("change the shoes to high heels"));
    assert_eq!(answer.model, "fal-ai/nano-banana/edit");
    assert_eq!(answer.model_key, "fal|fal-ai/nano-banana/edit|png|padded|1");
    assert_eq!(
        answer.summary.as_deref(),
        Some("change the shoes to high heels")
    );
    assert_eq!(call.path, "/fal-ai/nano-banana/edit");
    let body = call.json();
    // Told where: the painted area, white, as a second image of the padded
    // size, and to keep the change inside it.
    let prompt = body["prompt"].as_str().unwrap();
    assert!(
        prompt.contains("change the shoes to high heels"),
        "{prompt}"
    );
    assert!(
        prompt.contains("white area of the second image"),
        "{prompt}"
    );
    let urls = body["image_urls"].as_array().unwrap();
    assert_eq!(urls.len(), 2);
    let mask = image::load_from_memory(&support::from_data_uri(urls[1].as_str().unwrap()))
        .unwrap()
        .into_luma8();
    assert_eq!(mask.dimensions(), (W, H));
    // White where painted, black far from it (the mask is grown a little).
    let all = |pick: &dyn Fn(usize, usize) -> bool| {
        (0..H as usize)
            .flat_map(|y| (0..W as usize).map(move |x| (x, y)))
            .find(|(x, y)| pick(*x, *y))
            .unwrap()
    };
    let (px, py) = all(&|x, y| support::painted(x, y));
    assert_eq!(mask.get_pixel(px as u32, py as u32).0, [255]);
    let (fx, fy) = all(&|x, y| {
        (0..H as usize).all(|j| {
            (0..W as usize).all(|i| !support::painted(i, j) || x.abs_diff(i) + y.abs_diff(j) > 20)
        })
    });
    assert_eq!(mask.get_pixel(fx as u32, fy as u32).0, [0]);
    assert!(body.get("mask_url").is_none());
    assert_eq!(body["aspect_ratio"], json!("4:3"));
    assert_eq!(body["output_format"], json!("png"));
    assert_eq!(call.header("x-fal-store-io"), Some("0"));

    let (answer, call) = fal_call(
        &[("replace-model", GENFILL)],
        "replace",
        Some("a wooden bench"),
    );
    assert_eq!(answer.model, "bria/genfill/v2");
    assert_eq!(call.path, "/bria/genfill/v2");
    let body = call.json();
    assert_eq!(body["instruction"], json!("a wooden bench"));
    assert!(
        body["image_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );
    assert!(
        body["mask_url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );

    let (answer, call) = fal_call(&[("replace-model", FLUX_FILL)], "replace", Some("a bench"));
    assert_eq!(answer.model, "fal-ai/flux-pro/v1/fill");
    assert_eq!(answer.model_key, "fal|fal-ai/flux-pro/v1/fill|png|1");
    let body = call.json();
    assert_eq!(body["prompt"], json!("a bench"));
    // Lossless back, not FLUX's JPEG default.
    assert_eq!(body["output_format"], json!("png"));

    let (answer, call) = fal_call(&[("replace-model", FIBO_EDIT)], "replace", Some("a bench"));
    assert_eq!(answer.model, FAL_EDIT_MODEL);
    assert_eq!(call.json()["instruction"], json!("a bench"));
}

#[test]
fn a_replacement_needs_a_prompt_and_the_panel_keeps_it() {
    let server = Server::start(|_| fal_answer(&data_png(W, H, [0, 0, 0])));
    let host = support::start(support::config(&server, &[]), &[("fal-key", KEY)]);
    // Nothing typed, or only spaces: the user has a prompt to write, which
    // costs nothing and ends nothing.
    for prompt in ["", "  \n "] {
        support::ask(&host, "replace", prompt);
        let error = host
            .generate_retouch(support::request(), &support::image(), &support::coverage())
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::NeedsInput, "{error}");
        assert!(error.message.contains("AI Replace"), "{error}");
    }
    assert!(server.requests().is_empty());
    // The panel shows what it keeps, on one line when sent.
    support::ask(&host, "replace", "  a wooden\nbench ");
    let controls = host.open_panel(rawmakase_ai_retouch::PANEL).unwrap();
    let prompt = controls
        .iter()
        .find(|c| c.id.as_deref() == Some(rawmakase_ai_retouch::PROMPT))
        .unwrap();
    assert!(matches!(&prompt.kind,
        rawmakase_extension::protocol::controls::ControlKind::Text { value, multiline: true, .. }
            if value == "  a wooden\nbench "));
    // The tool says what it is itself, first: RAWmakase doesn't.
    assert!(matches!(&controls[0].kind,
        rawmakase_extension::protocol::controls::ControlKind::Help { text }
            if text.contains("Experimental") && text.contains("paid")));
    host.finish().unwrap();
    // A missing key names what needs it.
    let host = support::start(support::config(&server, &[]), &[]);
    support::ask(&host, "replace", "a bench");
    let error = host
        .generate_retouch(support::request(), &support::image(), &support::coverage())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::NeedsSetup);
    // Said without an address, which the key scrubber would mangle: the
    // setting's help has the link.
    assert_eq!(
        error.message,
        "No fal.ai API key is set. Set one in Preferences > Extensions > AI Retouch; \
         the link under its field shows how to get one."
    );
}

#[test]
fn replicate_takes_text_to_bria_fibo_edit() {
    for mode in ["remove", "replace"] {
        let server = replicate_server(succeeds_at(0));
        // The removal model setting is for removals without text.
        let host = replicate_host(&server, &[("replicate-model", "bria/eraser")]);
        support::ask(&host, mode, "a wooden bench");
        let (answer, result) = host
            .generate_retouch(support::request(), &support::image(), &support::coverage())
            .unwrap();
        assert_eq!(answer.model, REPLICATE_EDIT_MODEL);
        every_pixel_is(&result, [9, 8, 7]);
        let requests = server.requests();
        let create = requests
            .iter()
            .find(|r| r.path == "/v1/models/bria/fibo-edit/predictions")
            .expect("a prediction of bria/fibo-edit");
        let input = &create.json()["input"];
        assert_eq!(input["instruction"], json!("a wooden bench"));
        assert!(input["image"].is_string() && input["mask"].is_string());
        host.finish().unwrap();
    }
}
