# AI Retouch for RAWmakase

**Experimental.** A [RAWmakase](https://github.com/pch/rawmakase) extension
that adds an **AI Retouch** tool to Develop's rail (a magic wand), run by a
hosted AI model with your own account and API key. Its drawer has a **Mode**
and a **Prompt**:

- **AI Remove**: paint over an object and the model fills the area with what
  could plausibly be behind it. You can also say what to do, for example
  *remove the dogs, keep the people*, when you paint over both.
- **AI Replace**: say what to do with the area, for example *change the shoes
  to elegant high heels* or *a wooden bench*.

A second tool, **AI Remove on this computer** (an eraser), removes what you
paint with a model that runs on your own machine: no account, no key, nothing
uploaded. See [AI Remove on this computer](#ai-remove-on-this-computer).

Paint in as many strokes as it takes, then send the area with the drawer's
button or Enter. The prompt and mode are kept, between photos and after
RAWmakase restarts.

RAWmakase does all the colour work. It renders the region around what you
painted, sends it to this extension as 8-bit sRGB with the painted area, and
keeps the pixels it gets back as a patch in your edit,
which renders without the extension or the service from then on. The
extension only turns pixels into pixels through the service.

## Services and models

The model follows what you ask for:

| You ask for | fal.ai (default) | Replicate |
| --- | --- | --- |
| AI Remove, nothing written | Bria FIBO Eraser, [`bria/fibo-edit-1.5/eraser`](https://fal.ai/models/bria/fibo-edit-1.5/eraser), pixel-perfect mode: only the masked area is rebuilt. $0.04 | Bria Eraser, [`bria/eraser`](https://replicate.com/bria/eraser), or another model that takes `image` and `mask` named in the settings. $0.04 |
| AI Remove, with what to do | Bria FIBO Edit, [`bria/fibo-edit-1.5/edit`](https://fal.ai/models/bria/fibo-edit-1.5/edit), with your text as its instruction. $0.04 | Bria FIBO Edit, [`bria/fibo-edit`](https://replicate.com/bria/fibo-edit) |
| AI Replace | The **Replace model** setting: Nano Banana (Gemini 2.5 Flash Image), [`fal-ai/nano-banana/edit`](https://fal.ai/models/fal-ai/nano-banana/edit) ($0.04, the default), which edits what you paint as the prompt says; Bria GenFill, [`bria/genfill/v2`](https://fal.ai/models/bria/genfill/v2) ($0.04 a megapixel) or FLUX.1 Fill \[pro], [`fal-ai/flux-pro/v1/fill`](https://fal.ai/models/fal-ai/flux-pro/v1/fill) ($0.05 a megapixel, rounded up), which fill it with what you describe; or Bria FIBO Edit ($0.04) | Bria FIBO Edit, [`bria/fibo-edit`](https://replicate.com/bria/fibo-edit) |

A region is at most 2048 pixels a side, about 4 megapixels, so a
per-megapixel model costs at most about $0.17 (GenFill) or $0.25 (FLUX) a
request. The Bria models are trained on licensed data. Prices are the
services' as of November 2026; check their pages for the current ones.

**Text and the edit models.** The eraser rebuilds only the painted area. FIBO
Edit and the fill models generate the painted area from your text. Nano Banana
has no mask input: it gets the region (padded to an aspect ratio it takes, then
cut back) and, as a second image, the painted area in white, with an
instruction to change only that area; it may still redraw more of the region.
Either way, RAWmakase keeps only what comes back inside what you painted.
Inside the area, an instruction such as *keep the people* is followed as well
as the model follows it: check the
result, and undo it if it changed what it should have kept.

## AI Remove on this computer

The **AI Remove on this computer** tool needs no account and no key. It runs
[LaMa](https://github.com/advimman/lama) ("big-lama", Apache-2.0), in
[Carve's ONNX port](https://huggingface.co/Carve/LaMa-ONNX), on the CPU with
ONNX Runtime, inside the extension's own process. Nothing is uploaded, so
RAWmakase asks no consent question for it.

- **The weights are not in this package.** The first removal downloads
  `lama_fp32.onnx` (198 MiB) from a pinned revision of that repository into
  the extension's data folder, checks it against a pinned SHA-256, and keeps
  nothing that doesn't match. A download that stops early continues on the
  next request. Without a network the tool says so until the model is there.
  LaMa's training data is Places: check that its terms suit your use.
- **Speed.** A removal takes about 3 to 4.5 s on a recent laptop CPU (Apple
  M1 to M2 Max, 8 to 12 threads), plus about 1 s to load the model, and about
  1.5 GB of memory while it runs. The region is padded to a square and
  shrunk to the model's fixed 512 pixels, so a large region loses detail
  inside what you painted; the pixels outside it stay as they were.
- **What it is good at.** Small objects and texture: sand, water, walls,
  foliage, gravel. There it comes close to the hosted eraser.
- **What it is not.** It has no prompt, and it fills from what is around the
  area, so it does worse than the hosted model where a person or a structure
  crosses a sharp edge: a standing person's legs can leave a ghost on a
  platform, dark clothing on a wall can leave a dark blob. Use the hosted
  **AI Retouch** tool for those, and undo what looks wrong.
- AI Replace and removal with a prompt stay with the hosted tool.

## Installing

Install it from RAWmakase: **Preferences > Extensions > Available**, or
**Install from GitHub…** with `https://github.com/pch/rawmakase-ai-retouch`. It
needs a RAWmakase with AI Remove ([pch/rawmakase#1093](https://github.com/pch/rawmakase/pull/1093)),
which isn't in a release yet: RAWmakase 0.5.0 reads manifests up to `api = 2`
and refuses this one as written for a newer RAWmakase.

## Setting it up

You need an account with fal.ai or Replicate, with billing set up, and its
API key. [Getting an API key](docs/getting-a-key.md) walks through both
services step by step and where the key goes in RAWmakase; the extension's
settings link to it. Until a key is set, AI Remove says so and offers a
button to its settings.

**Environment variables.** When the key setting is empty, the extension reads
`FAL_KEY` (fal.ai) or `REPLICATE_API_TOKEN` (Replicate) from its environment.
RAWmakase passes its own environment on, but a GUI app started from a desktop
launcher, the Dock or the Start menu usually doesn't see variables set in a
shell profile (`.zshrc`, `.bashrc`), so the setting is the reliable place.

## Privacy and cost

- **AI Remove on this computer sends nothing**, besides the one download of
  the model.
- **The hosted tool's region leaves your computer.** Each request uploads the rendered
  region around what you painted (at most 2048 pixels a side), a mask and
  what you wrote to the service you chose. RAWmakase asks once before the
  first upload. The extension asks fal.ai not to store the request's
  payloads (`X-Fal-Store-IO: 0`) and deletes the files it uploads to
  Replicate once the prediction ends; what each service keeps beyond that is
  up to its own policy.
- **Each request is one paid call.** Nothing is retried automatically, not
  even after a timeout or a rate limit: you decide whether to try again.
- Your key is kept in the system's credential store by RAWmakase, never in
  its settings file. The extension never logs it, and removes anything shaped
  like a key from the services' error messages before you see them.

## Limitations

- **8-bit.** The region travels as 8-bit sRGB both ways, so a removal over
  deep shadows or smooth gradients may band where a raw edit wouldn't.
- **AI Remove on this computer is not available on Intel Macs**, for which
  ONNX Runtime has no prebuilt binary; it says so there.
- **At most 2048 pixels a side.** A larger region is sent at a lower
  resolution and scaled back by RAWmakase.
- **No HDR.** RAWmakase refuses AI Remove and AI Replace on HDR files and HDR
  edits.
- **Lightroom never sees these spots.** Their patches are RAWmakase's
  own: they aren't written to XMP, so Lightroom and Camera Raw show the photo
  without them.
- The result depends on the service: it can change between calls and between
  model versions, and a service may decline an image on its content.

## Building and releasing

```sh
cargo test                         # no network, no key needed
# The RAWmakase commit the SDK is pinned to: it reads `api = 3`.
cargo install --locked --git https://github.com/pch/rawmakase \
    --rev 8335cb25647c5dbcb929272f875b2c77f3feda70 cargo-rawmakase
cargo rawmakase dev                # try it in RAWmakase's developer mode
cargo rawmakase build && cargo rawmakase package
```

To release, raise `version` in both `Cargo.toml` and
`rawmakase-extension.toml`, commit, and push a tag `v<version>`.
`.github/workflows/rawmakase-release.yml` builds the extension for Linux,
macOS and Windows (x86-64 and ARM each), packages it with `cargo rawmakase
package` and attaches the `.rmext` files to the tag's GitHub release, where
RAWmakase's Install from GitHub finds them.

## How it works

- The painted area is made binary (anything painted at all counts) and grown
  by 6 pixels, so the model also replaces the soft fringe a brush tends to
  miss. The mask is sent as a lossless PNG the region's size, white where to
  change.
- fal.ai: one request to `https://fal.run/<model>` with the region and the
  mask as base64 `data:` URIs, which fal's model APIs accept, what you wrote
  as the model's `instruction` or `prompt`, and `sync_mode`, which returns
  the image in the answer. FLUX.1 Fill is asked for a PNG, not its default
  JPEG.
- Replicate: the region and the mask are uploaded with the files API, a
  prediction is started with `Prefer: wait` and polled until it ends, then
  the uploads are deleted. What you wrote goes as `bria/fibo-edit`'s
  `instruction`.
- A result at another size than the region is resized to it. Cancelling in
  RAWmakase answers at once; a Replicate prediction is also cancelled on
  Replicate. A request gives up after 3 minutes.

The source has the details and the documentation each fact was checked
against: `src/fal.rs` and `src/replicate.rs`.

## License

MIT. `icons/retouch.svg` is [Lucide](https://lucide.dev)'s `wand-sparkles` and
`icons/local.svg` its `eraser`, under the ISC license. The model it downloads
is Apache-2.0; ONNX Runtime is MIT.
