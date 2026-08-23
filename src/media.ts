/**
 * Media parsing for multi-content-proxy.
 *
 * Responsibilities:
 *   - classify files by extension into image / audio / video
 *   - discover media references in free text (paths + URLs, quoted or bare)
 *   - read & base64-encode local files with size + folder guards
 *   - use ffmpeg to extract video frames and the audio track
 *   - assemble OpenAI-compatible content parts for the proxy
 */

import { execFile } from "node:child_process";
import { readFile, stat, mkdtemp, readdir, rm } from "node:fs/promises";
import { statSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, extname, isAbsolute, join, resolve as resolvePath, sep } from "node:path";
import type { ImageContent } from "@earendil-works/pi-ai";
import type { MediaFile, MediaKind, MediaPart } from "./types.js";
import type { MultiContentConfig } from "./config.js";

// ── Extension → MIME maps ──────────────────────────────────────────────────

const IMAGE_EXT: Record<string, string> = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
  ".svg": "image/svg+xml",
  ".heic": "image/heic",
  ".heif": "image/heif",
  ".tif": "image/tiff",
  ".tiff": "image/tiff",
  ".avif": "image/avif",
};
const AUDIO_EXT: Record<string, string> = {
  ".wav": "audio/wav",
  ".mp3": "audio/mpeg",
  ".m4a": "audio/mp4",
  ".aac": "audio/aac",
  ".flac": "audio/flac",
  ".ogg": "audio/ogg",
  ".oga": "audio/ogg",
  ".opus": "audio/opus",
  ".webm": "audio/webm",
  ".wma": "audio/x-ms-wma",
};
const VIDEO_EXT: Record<string, string> = {
  ".mp4": "video/mp4",
  ".mov": "video/quicktime",
  ".mkv": "video/x-matroska",
  ".webm": "video/webm",
  ".avi": "video/x-msvideo",
  ".flv": "video/x-flv",
  ".m4v": "video/mp4",
  ".wmv": "video/x-ms-wmv",
  ".3gp": "video/3gpp",
};

const ALL_EXTS = [
  ...Object.keys(IMAGE_EXT),
  ...Object.keys(AUDIO_EXT),
  ...Object.keys(VIDEO_EXT),
].map((e) => e.slice(1));

const EXT_RE = new RegExp(`\\.(${ALL_EXTS.join("|")})$`, "i");

/** Classify a file extension (with or without leading dot). */
export function classifyExt(ext: string): MediaKind | undefined {
  const e = ext.toLowerCase().startsWith(".") ? ext.toLowerCase() : "." + ext.toLowerCase();
  if (IMAGE_EXT[e]) return "image";
  if (AUDIO_EXT[e]) return "audio";
  if (VIDEO_EXT[e]) return "video";
  return undefined;
}

export function mimeFor(kind: MediaKind, ext: string): string {
  const e = ext.toLowerCase().startsWith(".") ? ext.toLowerCase() : "." + ext.toLowerCase();
  const map = kind === "image" ? IMAGE_EXT : kind === "audio" ? AUDIO_EXT : VIDEO_EXT;
  return map[e] || "application/octet-stream";
}

function mb(bytes: number): string {
  return `${(bytes / 1024 / 1024).toFixed(1)}MB`;
}

// ── Text discovery ─────────────────────────────────────────────────────────

/**
 * Find media references in free text. Matches absolute/relative/Windows paths
 * and http(s) URLs (bare or quoted) that end in a known media extension.
 * Local matches are verified to exist on disk before being returned.
 */
export function extractCandidatePaths(text: string, cwd: string): MediaFile[] {
  const found = new Map<string, MediaFile>();
  const extGroup = ALL_EXTS.join("|");

  // Unquoted path/url tokens. A leading `@` is pi's file-mention syntax
  // (e.g. "@pic.jpg"); strip it so the token resolves as a relative path.
  const tokenRe = new RegExp(
    `(?:[A-Za-z]:\\\\|~?/|\\.\\.?(?:/|\\\\)|@|https?://)[^\\s"'\`<>]+\\.(?:${extGroup})\\b`,
    "gi",
  );
  // Quoted tokens (handles spaces).
  const quotedRe = new RegExp(`["'\`]([^"'\`<>]+\\.(?:${extGroup})\\b)["'\`]`, "gi");

  const pushRaw = (raw: string) => {
    let s = raw.trim();
    if (s.startsWith("@")) s = s.slice(1).trim();
    if (s.length === 0) return;
    const isUrl = /^https?:\/\//i.test(s);
    const m = EXT_RE.exec(s);
    const ext = m ? m[0] : "";
    const kind = classifyExt(ext);
    if (!kind) return;
    if (isUrl) {
      found.set(`url:${s}`, { kind, url: s, label: s, source: "url" });
      return;
    }
    let p = s;
    if (s.startsWith("~")) p = resolvePath(process.env.HOME || "", s.slice(1));
    else if (!isAbsolute(s)) p = resolvePath(cwd, s);
    found.set(`path:${p}`, { kind, path: p, label: basename(p), source: "path" });
  };

  for (const m of text.match(tokenRe) || []) pushRaw(m);
  for (const m of text.match(quotedRe) || []) pushRaw(m);

  // Verify local paths exist; drop URLs we can't validate (proxy will fail later).
  const out: MediaFile[] = [];
  for (const media of found.values()) {
    if (media.source === "url") {
      out.push(media);
      continue;
    }
    try {
      if (statSyncSafe(media.path!)) out.push(media);
    } catch {
      /* not a real path — skip */
    }
  }
  return out;
}

function statSyncSafe(p: string): boolean {
  try {
    return statSync(p).isFile();
  } catch {
    return false;
  }
}

// ── Reading & guards ────────────────────────────────────────────────────────

async function assertAllowed(path: string, config: MultiContentConfig): Promise<void> {
  if (config.allowedFolders.length === 0) return;
  const abs = resolvePath(path);
  const ok = config.allowedFolders.some((f) => {
    const af = resolvePath(f);
    return abs === af || abs.startsWith(af + sep);
  });
  if (!ok) throw new Error(`path outside allowed folders (grant with /multi-content-proxy folders add <path>)`);
}

/** Read a local media file into base64 with size + folder guards. */
export async function readMediaBytes(
  path: string,
  config: MultiContentConfig,
): Promise<{ data: string; mimeType: string; size: number }> {
  await assertAllowed(path, config);
  let st;
  try {
    st = await stat(path);
  } catch (err: any) {
    throw new Error(`cannot read ${path}: ${err?.message ?? err}`);
  }
  if (!st.isFile()) throw new Error(`not a file: ${path}`);
  if (st.size === 0) throw new Error(`empty file: ${path}`);
  if (st.size > config.maxBytes) {
    throw new Error(`file too large (${mb(st.size)} > ${mb(config.maxBytes)}; raise MULTI_CONTENT_PROXY_MAX_BYTES)`);
  }
  const buf = await readFile(path);
  const ext = extname(path);
  const kind = classifyExt(ext);
  if (!kind) throw new Error(`unsupported extension: ${ext}`);
  return { data: buf.toString("base64"), mimeType: mimeFor(kind, ext), size: st.size };
}

/** Convert an attached ImageContent into a MediaFile. */
export function mediaFromAttachment(img: ImageContent): MediaFile {
  return {
    kind: "image",
    data: img.data,
    mimeType: img.mimeType,
    label: "attached image",
    source: "attachment",
  };
}

// ── ffmpeg helpers ──────────────────────────────────────────────────────────

let ffmpegOk: boolean | undefined;
let ffprobeOk: boolean | undefined;

export async function ffmpegAvailable(ffmpegPath: string): Promise<boolean> {
  if (ffmpegOk !== undefined) return ffmpegOk;
  try {
    await runFfmpeg(ffmpegPath, ["-version"], undefined);
    ffmpegOk = true;
  } catch {
    ffmpegOk = false;
  }
  return ffmpegOk;
}

async function runFfmpeg(
  bin: string,
  args: string[],
  signal: AbortSignal | undefined,
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const child = execFile(bin, args, { maxBuffer: 64 * 1024 * 1024 }, (err) => {
      if (err && (err as any).killed) return reject(new AbortError("cancelled"));
      if (err) return reject(err);
      resolve();
    });
    if (signal) {
      if (signal.aborted) {
        child.kill();
        reject(new AbortError("cancelled"));
        return;
      }
      signal.addEventListener(
        "abort",
        () => {
          child.kill();
        },
        { once: true },
      );
    }
  });
}

class AbortError extends Error {
  constructor(message = "cancelled") {
    super(message);
    this.name = "AbortError";
  }
}

/** Duration of a media file in seconds (best-effort; undefined if ffprobe missing). */
export function getMediaDuration(
  path: string,
  ffmpegPath: string,
  signal?: AbortSignal,
): Promise<number | undefined> {
  return probeDuration(path, ffmpegPath, signal);
}

/**
 * Split a (long) audio/video file into <=maxSeconds re-encoded mp3 segments.
 * dots.ai (and similar) reject a single huge audio payload, so long audio is
 * chunked and analyzed segment-by-segment, then merged — mirroring the
 * vision-proxy behaviour.
 */
export async function extractAudioSegments(
  path: string,
  maxSeconds: number,
  ffmpegPath: string,
  signal?: AbortSignal,
): Promise<{ data: string; mimeType: string }[]> {
  const dir = await mkdtemp(join(tmpdir(), "mcp-seg-"));
  const outPattern = join(dir, "seg_%03d.mp3");
  try {
    await runFfmpeg(
      ffmpegPath,
      [
        "-y",
        "-i",
        path,
        "-vn",
        "-acodec",
        "libmp3lame",
        "-b:a",
        "128k",
        "-f",
        "segment",
        "-segment_time",
        String(maxSeconds),
        outPattern,
      ],
      signal,
    );
    const files = (await readdir(dir)).filter((f) => f.endsWith(".mp3")).sort().map((f) => join(dir, f));
    if (files.length === 0) throw new Error("ffmpeg produced no audio segments");
    const out: { data: string; mimeType: string }[] = [];
    for (const f of files) {
      const buf = await readFile(f);
      out.push({ data: buf.toString("base64"), mimeType: "audio/mpeg" });
    }
    return out;
  } finally {
    try {
      await rm(dir, { recursive: true, force: true });
    } catch {
      /* ignore */
    }
  }
}

async function probeDuration(path: string, ffmpegPath: string, signal?: AbortSignal): Promise<number | undefined> {
  if (ffprobeOk === false) return undefined;
  try {
    const out = await new Promise<string>((resolve, reject) => {
      execFile(
        "ffprobe",
        ["-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        (err, stdout) => (err ? reject(err) : resolve(stdout.trim())),
      );
    });
    const n = parseFloat(out);
    if (Number.isFinite(n)) return n;
  } catch {
    ffprobeOk = false;
  }
  return undefined;
}

/** Extract `count` evenly-spaced PNG frames from a video. */
export async function extractVideoFrames(
  path: string,
  ffmpegPath: string,
  count: number,
  signal?: AbortSignal,
): Promise<string[]> {
  const dir = await mkdtemp(join(tmpdir(), "mcp-frames-"));
  const dur = await probeDuration(path, ffmpegPath, signal);
  // fps chosen so the clip yields ~`count` frames. Fall back to count fps if unknown.
  const fps = dur && dur > 0 ? (count / dur).toFixed(4) : String(count);
  const outPattern = join(dir, "frame_%03d.png");
  await runFfmpeg(
    ffmpegPath,
    ["-y", "-i", path, "-vf", `fps=${fps},scale='min(1280,iw)':-2`, "-frames:v", String(count), outPattern],
    signal,
  );
  const files = (await readdir(dir))
    .filter((f) => f.endsWith(".png"))
    .sort()
    .map((f) => join(dir, f));
  if (files.length === 0) throw new Error("ffmpeg produced no frames");
  return files;
}

/** Extract the audio track of a video to 16kHz mono WAV, sent to the proxy as an `audio_url` data URI. */
export async function extractVideoAudio(
  path: string,
  ffmpegPath: string,
  signal?: AbortSignal,
): Promise<{ data: string; format: string }> {
  const dir = await mkdtemp(join(tmpdir(), "mcp-audio-"));
  const out = join(dir, "audio.wav");
  await runFfmpeg(ffmpegPath, ["-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", out], signal);
  const buf = await readFile(out);
  return { data: buf.toString("base64"), format: "wav" };
}

async function probeDimensions(path: string, ffmpegPath: string, signal?: AbortSignal): Promise<{ width: number; height: number } | undefined> {
  if (ffprobeOk === false) return undefined;
  try {
    const out = await new Promise<string>((resolve, reject) => {
      execFile(
        "ffprobe",
        ["-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        (err, stdout) => (err ? reject(err) : resolve(stdout.trim())),
      );
    });
    const [w, h] = out.split(",").map(Number);
    if (Number.isFinite(w) && Number.isFinite(h)) return { width: w, height: h };
  } catch {
    ffprobeOk = false;
  }
  return undefined;
}

/** A per-image crop spec (mirrors the analyze_image tool). */
export type CropSpec = { image_index: number } & (
  | { region: string }
  | { normalized: { x: number; y: number; width: number; height: number } }
  | { pixels: { x: number; y: number; width: number; height: number } }
);

/** Crop an image file to a temp PNG using ffmpeg; returns the temp path. Best-effort. */
export async function cropImageFile(
  path: string,
  crop: CropSpec,
  ffmpegPath: string,
  signal?: AbortSignal,
): Promise<string> {
  const dim = await probeDimensions(path, ffmpegPath, signal);
  if (!dim) throw new Error("could not read image dimensions for crop (need ffprobe)");
  const W = dim.width;
  const H = dim.height;
  let x: number, y: number, w: number, h: number;
  if ("pixels" in crop) {
    x = crop.pixels.x;
    y = crop.pixels.y;
    w = crop.pixels.width;
    h = crop.pixels.height;
  } else if ("normalized" in crop) {
    w = Math.round(crop.normalized.width * W);
    h = Math.round(crop.normalized.height * H);
    x = Math.round(crop.normalized.x * W);
    y = Math.round(crop.normalized.y * H);
  } else {
    const REG: Record<string, [number, number]> = {
      "top-left": [0, 0],
      "top-right": [W / 2, 0],
      "bottom-left": [0, H / 2],
      "bottom-right": [W / 2, H / 2],
      top: [0, 0],
      bottom: [0, H / 2],
      left: [0, 0],
      right: [W / 2, 0],
      center: [W / 4, H / 4],
      "top-half": [0, 0],
      "bottom-half": [0, H / 2],
      "left-half": [0, 0],
      "right-half": [W / 2, 0],
    };
    const [bx, by] = REG[crop.region] || [0, 0];
    w = Math.round(W / 2);
    h = Math.round(H / 2);
    x = Math.round(bx);
    y = Math.round(by);
    if (crop.region === "top" || crop.region === "bottom") {
      w = W;
      x = 0;
    }
    if (crop.region === "left" || crop.region === "right") {
      h = H;
      y = 0;
    }
    if (crop.region === "top-half" || crop.region === "bottom-half") {
      w = W;
      x = 0;
    }
    if (crop.region === "left-half" || crop.region === "right-half") {
      h = H;
      y = 0;
    }
  }
  x = Math.max(0, Math.min(Math.round(x), W - 1));
  y = Math.max(0, Math.min(Math.round(y), H - 1));
  w = Math.max(1, Math.min(Math.round(w), W - x));
  h = Math.max(1, Math.min(Math.round(h), H - y));
  const dir = await mkdtemp(join(tmpdir(), "mcp-crop-"));
  const out = join(dir, "crop.png");
  await runFfmpeg(ffmpegPath, ["-y", "-i", path, "-vf", `crop=${w}:${h}:${x}:${y}`, out], signal);
  return out;
}

export function audioFormatFromMime(mime: string): string {
  if (mime.includes("wav")) return "wav";
  if (mime.includes("mp3") || mime.includes("mpeg")) return "mp3";
  if (mime.includes("ogg")) return "ogg";
  if (mime.includes("flac")) return "flac";
  if (mime.includes("webm")) return "webm";
  if (mime.includes("m4a") || mime.includes("mp4")) return "m4a";
  return "mp3";
}

// ── Content-part assembly ───────────────────────────────────────────────────

/**
 * Build OpenAI-compatible content parts for one media item.
 * `frames` overrides the per-video frame count (used by the analyze tool).
 */
export async function buildParts(
  media: MediaFile,
  config: MultiContentConfig,
  opts: { frames?: number; signal?: AbortSignal } = {},
): Promise<MediaPart[]> {
  if (media.kind === "image") {
    if (media.source === "url") return [{ type: "image_url", image_url: { url: media.url! } }];
    let data = media.data;
    let mime = media.mimeType;
    if (media.source === "path") {
      const r = await readMediaBytes(media.path!, config);
      data = r.data;
      mime = r.mimeType;
    }
    return [{ type: "image_url", image_url: { url: `data:${mime};base64,${data}` } }];
  }

  if (media.kind === "audio") {
    if (media.source === "url") {
      throw new Error("remote audio URLs are not supported; download first");
    }
    let data = media.data;
    let mime = media.mimeType;
    if (media.source === "path") {
      const r = await readMediaBytes(media.path!, config);
      data = r.data;
      mime = r.mimeType;
    }
    // dots.ai ingests audio via `audio_url` with a data URI (the OpenAI
    // `input_audio` shape is rejected by dots.ai).
    return [{ type: "audio_url", audio_url: { url: `data:${mime};base64,${data}` } }];
  }

  // video
  if (config.videoStrategy === "native") {
    if (media.source === "url") return [{ type: "video_url", video_url: { url: media.url! } }];
    const r = await readMediaBytes(media.path!, config);
    return [{ type: "video_url", video_url: { url: `data:${r.mimeType};base64,${r.data}` } }];
  }

  // frames strategy
  if (media.source === "url") return [{ type: "video_url", video_url: { url: media.url! } }];
  const count = opts.frames ?? config.maxFrames;
  const frames = await extractVideoFrames(media.path!, config.ffmpegPath, count, opts.signal);
  const parts: MediaPart[] = [];
  for (const f of frames) {
    const r = await readMediaBytes(f, config);
    parts.push({ type: "image_url", image_url: { url: `data:${r.mimeType};base64,${r.data}` } });
  }
  try {
    const a = await extractVideoAudio(media.path!, config.ffmpegPath, opts.signal);
    parts.push({ type: "audio_url", audio_url: { url: `data:audio/wav;base64,${a.data}` } });
  } catch {
    /* video has no audio track — fine */
  }
  if (parts.length === 0) throw new Error("no frames extracted from video");
  return parts;
}
