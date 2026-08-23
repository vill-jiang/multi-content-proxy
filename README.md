# multi-content-proxy

A [Pi](https://github.com/earendil-works/pi-coding-agent) agent extension that **parses image, audio, and video
inputs** and forwards them to an OpenAI-compatible multimodal proxy so the agent can "see", "hear", and "watch"
content that the underlying model wouldn't otherwise ingest natively.

It is the generalized sibling of the image-only **vision-proxy**: the same interception pattern
(`<input>` hook → describe media through a proxy → inject text into the prompt) is extended to audio and
video, with per-modality provider configuration so you can point each modality at the model that actually
supports it.

## Why

Different multimodal models accept different input modalities. Some do images only, some add audio, some add
video. The [dots.ai platform](https://dots.ai) exposes an OpenAI-compatible chat-completions endpoint
(described in its `#tool-calling-response-example` docs) that accepts a mix of **text + image + video + audio**
content parts. This extension maps each detected media file to the right content-part format and sends it to
the right endpoint, so a non-multimodal coding model can still reason about screenshots, voice memos, and
screen recordings.

## Install / enable

### From npm (recommended for users)

```bash
# global install (applies to all your pi projects)
pi install npm:multi-content-proxy

# or project-local install (scoped to the current project, adds to .pi/settings.json)
pi install npm:multi-content-proxy -l
```

This downloads the package into pi's managed `npm` directory and registers it in your
`settings.json` `packages` list. Restart (or start a new) pi session and the `analyze_media`
tool + `/multi-content-proxy` command become available.

### From a local folder (for development)

```bash
pi install /absolute/path/to/multi-content-proxy
```

pi treats an absolute path as a `local:` source — no network, no copy; it just records the
path in `settings.json` and loads `index.ts` via jiti. Great for iterating on the source.

### Manual

You can also drop the folder into `~/.pi/agent/extensions/` (auto-discovered) or add the
path to `settings.json` → `packages` yourself.

`typebox` is the only runtime dependency (Pi already provides it; the dependency is declared
for standalone installs).

## Concepts

### Providers (per-modality)

Because modality support is uneven across models, the extension keeps a **base image provider** plus optional
dedicated **audio** and **video** providers.

- `image` is the base and is always required.
- `audio` and `video` are optional. When they are **not** configured they transparently **fall back to the
  image provider** (its `baseUrl` / `apiKey` / `model`). This is exactly what you want for an endpoint like
  dots.ai's that handles all three with one model — set it once on `image` and everything works.
- To use a different model for audio or video, configure that modality's provider and only the fields you
  want; the rest inherits from `image`. Clear it again (`audio-model clear`) to revert to fallback.

Each provider is an OpenAI-compatible chat-completions endpoint:

| Field     | Meaning                                            |
| --------- | -------------------------------------------------- |
| `baseUrl` | chat-completions URL (e.g. dots.ai's)              |
| `apiKey`  | bearer token (optional for some endpoints)         |
| `model`   | multimodal model id understood by the endpoint     |

### Modes

- `fallback` (default): only send media to the proxy when the *active* model lacks that modality. Images
  attached by Pi are still passed through if the model supports `image`. Audio/video are always proxied
  (Pi models can't ingest them natively).
- `always`: proxy every media input regardless of model capabilities (useful to force descriptions).
- `off`: disable the extension.

### Strategies

- **Video**
  - `native` (default): send the whole file as a single `video_url` part (only works with endpoints
    that accept video natively, e.g. dots.ai). If the provider **rejects** the payload (dots.ai returns
    HTTP 400 for oversized video), the extension **automatically retries with `frames`** — so you get the
    best quality when it works and a graceful fallback when it doesn't.
  - `frames`: `ffmpeg` extracts up to N preview frames (PNG) + the audio track (WAV). Frames are sent as
    `image_url` parts and the audio is sent as an `audio_url` data URI (or transcribed if
    `audio-strategy=transcribe`).
- **Audio**
  - `describe` (default): send the audio inline as an `audio_url` data URI. **Note:** dots.ai (and this
    extension's default content-part format) use `audio_url`, *not* OpenAI's `input_audio` shape, which
    dots.ai rejects.
  - `transcribe`: POST to the endpoint's `audio/transcriptions` route (OpenAI Whisper-compatible) and inject
    the transcript text.

## Usage

When enabled, any image/audio/video file you reference in a message is detected and handled automatically.
References can be a local path, an `http(s)` URL, or Pi's file-mention syntax `@file` (e.g. `@pic.jpg`).
For example:

```
explain what this diagram shows: ./arch.png
summarize this meeting recording: ./standup.mp3
what is happening in this clip: ./demo.mov
```

You can also drive it deliberately with the built-in tool and command (see below).

### The `analyze_media` tool

The agent can call `analyze_media` to inspect specific files or URLs on demand:

- `media` — array of 1–20 references: local file paths or `http(s)` URLs (image/audio/video)
- `question` — what to ask about the media (required)
- `frames` — number of video frames to sample (frames strategy)
- `crop` — per-image crop `{ image_index, region | box }` (requires ffmpeg)

### The `/multi-content-proxy` command

```
/multi-content-proxy                          # show current config
/multi-content-proxy fallback|always|off      # set mode

# per-modality provider (audio/video fall back to image when unset)
/multi-content-proxy image-model <id>
/multi-content-proxy image-base-url <url>
/multi-content-proxy image-api-key <key>
/multi-content-proxy audio-model <id>          # or `audio-model clear`
/multi-content-proxy audio-base-url <url>
/multi-content-proxy audio-api-key <key>
/multi-content-proxy video-model <id>          # or `video-model clear`
/multi-content-proxy video-base-url <url>
/multi-content-proxy video-api-key <key>
# shortcuts: model / base-url / api-key  ==  image-*

/multi-content-proxy video-strategy native|frames
/multi-content-proxy audio-strategy describe|transcribe
/multi-content-proxy max-frames <n>
/multi-content-proxy max-bytes <n|Nmb>
/multi-content-proxy ffmpeg <path>
/multi-content-proxy status on|off
/multi-content-proxy consent yes|no|ask
/multi-content-proxy folders add|remove|clear <path>
/multi-content-proxy test <path|url>           # call the proxy once and print the result
/multi-content-proxy reset-consent
```

## Configuration

Precedence (highest first): **environment variables → persisted JSON
(`~/.pi/agent/multi-content-proxy.json`) → built-in defaults**.

### Environment variables

| Variable                                         | Maps to                         |
| ------------------------------------------------ | ------------------------------- |
| `MULTI_CONTENT_PROXY_MODE`                       | mode                            |
| `MULTI_CONTENT_PROXY_IMAGE_BASE_URL`             | image.baseUrl (falls back to `MULTI_CONTENT_PROXY_BASE_URL`) |
| `MULTI_CONTENT_PROXY_IMAGE_API_KEY`              | image.apiKey (falls back to `MULTI_CONTENT_PROXY_API_KEY`) |
| `MULTI_CONTENT_PROXY_IMAGE_MODEL`                | image.model (falls back to `MULTI_CONTENT_PROXY_MODEL`) |
| `MULTI_CONTENT_PROXY_AUDIO_BASE_URL/API_KEY/MODEL` | audio provider (inherits image when omitted) |
| `MULTI_CONTENT_PROXY_VIDEO_BASE_URL/API_KEY/MODEL` | video provider (inherits image when omitted) |
| `MULTI_CONTENT_PROXY_VIDEO_STRATEGY`             | `native` \| `frames`            |
| `MULTI_CONTENT_PROXY_AUDIO_STRATEGY`             | `describe` \| `transcribe`      |
| `MULTI_CONTENT_PROXY_MAX_BYTES`                  | size guard (bytes)              |
| `MULTI_CONTENT_PROXY_MAX_FRAMES`                 | video frame count               |
| `MULTI_CONTENT_PROXY_FFMPEG`                     | ffmpeg executable path          |
| `MULTI_CONTENT_PROXY_CONSENT`                    | `yes` \| `no` \| `ask`          |
| `MULTI_CONTENT_PROXY_STATUS_LINE`                | `on` \| `off`                   |
| `MULTI_CONTENT_PROXY_INCLUDE_CONTEXT`            | `true` \| `false`               |
| `MULTI_CONTENT_PROXY_ENABLE_IMAGE/AUDIO/VIDEO`   | per-kind enable toggles         |

Defaults: mode `fallback`, image base URL `https://dots.ai/api/v1/chat/completions`,
image model `dots3-note-prev`.

## Privacy & consent

Media is base64-encoded and sent to the configured endpoint. Before the first upload you'll be asked for
consent (`consent` config: `ask` default, or set `yes`/`no`). Local file access is restricted to the cwd and
(optionally) `allowedFolders`; per-file size is capped by `maxBytes`. The proxy result is injected into the
prompt inside a clearly delimited `<multi_content_proxy>` block, and the original media is removed from the
message when the model can't use it natively.

## How it works (architecture)

```
src/types.ts    MediaKind / MediaFile / MediaPart / ProxyResult
src/config.ts   resolveConfig (env > persisted > defaults), per-modality providers, providerFor()
src/media.ts    path/URL/attachment extraction, size+folder guards, ffmpeg frame/audio extraction,
                long-audio chunking, buildParts() → OpenAI multimodal content parts
                (image_url / video_url / audio_url)
src/proxy.ts    callMultimodalProxy() (chat/completions) + callStt() (audio/transcriptions)
index.ts        <input> hook (auto media), analyze_media tool, /multi-content-proxy command,
                before_agent_start / session_start hooks, consent + rate-limit + LRU cache
```

Media is mapped to the canonical OpenAI-compatible multimodal content-part format (data-URI parts):

- image → `{ type: "image_url", image_url: { url: "data:..." } }`
- video (native) → `{ type: "video_url", video_url: { url: "data:..." } }`
- video (frames) → N `image_url` parts + one `audio_url` part
- audio → `{ type: "audio_url", audio_url: { url: "data:..." } }` (or a transcription)

## dots.ai notes

The reference endpoint is dots.ai's platform API. Working configuration (also the default `image` provider):

- `baseUrl`: `https://note3-prev-api.askdiandian.com/v1/chat/completions`
- `model`: `dots3-note-prev`
- `apiKey`: a dots-ai API key (bearer token)

dots.ai-specific behaviours this extension handles for you:

- **`audio_url`, not `input_audio`.** dots.ai's chat-completions endpoint accepts audio as
  `{ type: "audio_url", audio_url: { url: "data:..." } }`. The OpenAI `input_audio` shape is rejected.
- **Long-audio chunking.** dots.ai returns an empty response for a single huge audio payload, so audio
  longer than ~110s is automatically split into segments, analyzed per segment, then merged.
- **Video native → frames fallback.** Very large `video_url` payloads are rejected (HTTP 400); the
  extension retries with the `frames` strategy automatically.
- **Higher token budget.** dots.ai is a reasoning model; audio/video calls use `max_tokens: 8192` or it
  spends the whole budget thinking and returns empty `content`.

## License

MIT.
