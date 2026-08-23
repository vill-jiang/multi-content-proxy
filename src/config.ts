/**
 * Configuration resolution for multi-content-proxy.
 *
 * Multimodal models differ wildly in what they accept: some do images only,
 * some add audio, some add video. So we keep a *base* image provider plus
 * optional dedicated audio/video providers. When audio/video are not
 * configured they transparently fall back to the image (base) provider —
 * which is exactly what you want for an endpoint like dots.ai's that handles
 * all three with one model.
 *
 * Precedence (highest first):
 *   1. Environment variables (override everything)
 *   2. Persisted JSON file (~/.pi/agent/multi-content-proxy.json)
 *   3. Built-in defaults
 */

import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { readFileSync } from "node:fs";
import type { MediaKind } from "./types.js";

export type ProxyMode = "fallback" | "always" | "off";
export type VideoStrategy = "native" | "frames";
export type AudioStrategy = "describe" | "transcribe";

/** A single OpenAI-compatible multimodal endpoint. */
export interface ProviderConfig {
  /** OpenAI-compatible chat-completions base URL. */
  baseUrl: string;
  apiKey: string;
  /** Multimodal model id understood by the endpoint. */
  model: string;
}

export interface MultiContentConfig {
  /** fallback: only proxy when the active model lacks the modality. always: proxy everything. off: disabled. */
  mode: ProxyMode;
  /** Base provider — required. Audio/video inherit from this when not set. */
  image: ProviderConfig;
  /** Optional dedicated audio provider; falls back to image when undefined. */
  audio?: ProviderConfig;
  /** Optional dedicated video provider; falls back to image when undefined. */
  video?: ProviderConfig;
  videoStrategy: VideoStrategy;
  audioStrategy: AudioStrategy;
  maxBytes: number;
  ffmpegPath: string;
  consent: "yes" | "no" | "ask";
  statusLine: "on" | "off";
  includeContext: boolean;
  enableImage: boolean;
  enableAudio: boolean;
  enableVideo: boolean;
  allowedFolders: string[];
  maxFrames: number;
}

/** OpenAI-compatible base URL (the `…/v1` root). The request path
 *  (`/chat/completions`, `/audio/transcriptions`) is appended in proxy.ts.
 *  Override via MULTI_CONTENT_PROXY_*_BASE_URL or the persisted JSON config. */
export const DEFAULT_IMAGE_BASE_URL = "https://note3-prev-api.askdiandian.com/v1";
export const DEFAULT_IMAGE_MODEL = "dots3-note-prev";

export const DEFAULT_IMAGE_PROVIDER: ProviderConfig = {
  baseUrl: DEFAULT_IMAGE_BASE_URL,
  apiKey: "",
  model: DEFAULT_IMAGE_MODEL,
};

export const DEFAULT_CONFIG: MultiContentConfig = {
  mode: "fallback",
  image: { ...DEFAULT_IMAGE_PROVIDER },
  audio: undefined,
  video: undefined,
  videoStrategy: "native", // try whole-file first; auto-fall back to frames on failure
  audioStrategy: "describe",
  maxBytes: 25 * 1024 * 1024,
  ffmpegPath: "ffmpeg",
  consent: "ask",
  statusLine: "on",
  includeContext: true,
  enableImage: true,
  enableAudio: true,
  enableVideo: true,
  allowedFolders: [],
  maxFrames: 4,
};

const CONFIG_FILE = join(homedir(), ".pi", "agent", "multi-content-proxy.json");

function envBool(v: string | undefined, def: boolean): boolean {
  if (v === undefined) return def;
  return v === "1" || v.toLowerCase() === "true" || v.toLowerCase() === "yes";
}

function num(v: string | undefined, def: number): number {
  if (v === undefined) return def;
  const n = Number(v);
  return Number.isFinite(n) ? n : def;
}

function loadPersistedSync(): Partial<MultiContentConfig> {
  try {
    return JSON.parse(readFileSync(CONFIG_FILE, "utf8")) as Partial<MultiContentConfig>;
  } catch {
    return {};
  }
}

/** Resolve the effective provider for a media kind (audio/video fall back to image). */
export function providerFor(kind: MediaKind, config: MultiContentConfig): ProviderConfig {
  if (kind === "audio" && config.audio) return config.audio;
  if (kind === "video" && config.video) return config.video;
  return config.image;
}

/** Resolve the effective config (env overrides persisted overrides defaults). */
export function resolveConfig(env: NodeJS.ProcessEnv = process.env): MultiContentConfig {
  const p = loadPersistedSync();

  const base: MultiContentConfig = {
    ...DEFAULT_CONFIG,
    ...p,
    image: { ...DEFAULT_CONFIG.image, ...(p.image || {}) },
    audio: p.audio ? { ...DEFAULT_CONFIG.image, ...p.audio } : undefined,
    video: p.video ? { ...DEFAULT_CONFIG.image, ...p.video } : undefined,
  };

  // Image provider (also accepts generic MULTI_CONTENT_PROXY_* as image defaults).
  const image: ProviderConfig = { ...base.image };
  image.baseUrl = env.MULTI_CONTENT_PROXY_IMAGE_BASE_URL || env.MULTI_CONTENT_PROXY_BASE_URL || image.baseUrl;
  image.apiKey = env.MULTI_CONTENT_PROXY_IMAGE_API_KEY || env.MULTI_CONTENT_PROXY_API_KEY || image.apiKey;
  image.model = env.MULTI_CONTENT_PROXY_IMAGE_MODEL || env.MULTI_CONTENT_PROXY_MODEL || image.model;
  base.image = image;

  // Audio provider (optional; inherits image when a field is omitted).
  if (
    env.MULTI_CONTENT_PROXY_AUDIO_BASE_URL ||
    env.MULTI_CONTENT_PROXY_AUDIO_API_KEY ||
    env.MULTI_CONTENT_PROXY_AUDIO_MODEL
  ) {
    base.audio = {
      baseUrl: env.MULTI_CONTENT_PROXY_AUDIO_BASE_URL || image.baseUrl,
      apiKey: env.MULTI_CONTENT_PROXY_AUDIO_API_KEY || image.apiKey,
      model: env.MULTI_CONTENT_PROXY_AUDIO_MODEL || image.model,
    };
  }

  // Video provider (optional; inherits image when a field is omitted).
  if (
    env.MULTI_CONTENT_PROXY_VIDEO_BASE_URL ||
    env.MULTI_CONTENT_PROXY_VIDEO_API_KEY ||
    env.MULTI_CONTENT_PROXY_VIDEO_MODEL
  ) {
    base.video = {
      baseUrl: env.MULTI_CONTENT_PROXY_VIDEO_BASE_URL || image.baseUrl,
      apiKey: env.MULTI_CONTENT_PROXY_VIDEO_API_KEY || image.apiKey,
      model: env.MULTI_CONTENT_PROXY_VIDEO_MODEL || image.model,
    };
  }

  // Scalar overrides.
  base.mode = (env.MULTI_CONTENT_PROXY_MODE as ProxyMode) || p.mode || DEFAULT_CONFIG.mode;
  base.videoStrategy =
    (env.MULTI_CONTENT_PROXY_VIDEO_STRATEGY as VideoStrategy) ||
    p.videoStrategy ||
    DEFAULT_CONFIG.videoStrategy;
  base.audioStrategy =
    (env.MULTI_CONTENT_PROXY_AUDIO_STRATEGY as AudioStrategy) ||
    p.audioStrategy ||
    DEFAULT_CONFIG.audioStrategy;
  base.maxBytes = num(env.MULTI_CONTENT_PROXY_MAX_BYTES, p.maxBytes ?? DEFAULT_CONFIG.maxBytes);
  base.ffmpegPath = env.MULTI_CONTENT_PROXY_FFMPEG || p.ffmpegPath || DEFAULT_CONFIG.ffmpegPath;
  base.consent = (env.MULTI_CONTENT_PROXY_CONSENT as "yes" | "no" | "ask") || p.consent || DEFAULT_CONFIG.consent;
  base.statusLine = (env.MULTI_CONTENT_PROXY_STATUS_LINE as "on" | "off") || p.statusLine || DEFAULT_CONFIG.statusLine;
  base.includeContext = envBool(env.MULTI_CONTENT_PROXY_INCLUDE_CONTEXT, p.includeContext ?? DEFAULT_CONFIG.includeContext);
  base.enableImage = envBool(env.MULTI_CONTENT_PROXY_ENABLE_IMAGE, p.enableImage ?? DEFAULT_CONFIG.enableImage);
  base.enableAudio = envBool(env.MULTI_CONTENT_PROXY_ENABLE_AUDIO, p.enableAudio ?? DEFAULT_CONFIG.enableAudio);
  base.enableVideo = envBool(env.MULTI_CONTENT_PROXY_ENABLE_VIDEO, p.enableVideo ?? DEFAULT_CONFIG.enableVideo);
  base.allowedFolders = p.allowedFolders ?? DEFAULT_CONFIG.allowedFolders;
  base.maxFrames = num(env.MULTI_CONTENT_PROXY_MAX_FRAMES, p.maxFrames ?? DEFAULT_CONFIG.maxFrames);

  return base;
}

/** Persist a partial config patch (shallow-merged with the existing file). */
export async function savePersisted(patch: Partial<MultiContentConfig>): Promise<void> {
  const cur = loadPersistedSync();
  const next = { ...cur, ...patch };
  await mkdir(dirname(CONFIG_FILE), { recursive: true });
  await writeFile(CONFIG_FILE, JSON.stringify(next, null, 2), "utf8");
}
