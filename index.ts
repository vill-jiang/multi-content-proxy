/**
 * multi-content-proxy — a pi extension that parses image, audio, and video
 * file inputs and forwards them to an OpenAI-compatible multimodal endpoint
 * (dots.ai / OpenRouter / vLLM / Ollama) so a text-only model can still
 * reason about them.
 *
 * Inspired by the vision-proxy plugin (which handles images) but generalizes
 * to audio and video as well, matching dots.ai's platform docs where a single
 * messages array accepts text + image + video + audio input.
 *
 * Flow:
 *   1. The `input` hook scans the user prompt and any attached images for media.
 *   2. Rather than analyzing synchronously (slow → unresponsive), it injects a
 *      short hint listing the media references and instructs the model to call
 *      the `analyze_media` tool, so analysis happens in the model's own loop as
 *      a visible, progress-reporting tool call.
 *   3. The tool reads/encodes each media file (video → ffmpeg frames + audio
 *      track), sends the rich content parts to the proxy, and returns a
 *      description / transcript the model can use.
 *
 * Install / load:
 *   pi -e ./index.ts
 *   or copy this folder to ~/.pi/agent/extensions/multi-content-proxy/
 */

import type {
  ExtensionAPI,
  ExtensionContext,
  InputEvent,
  BeforeAgentStartEvent,
  SessionStartEvent,
} from "@earendil-works/pi-coding-agent";
import type { ImageContent } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { basename, resolve, isAbsolute, join } from "node:path";
import { statSync } from "node:fs";
import { writeFile, mkdir } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";

import {
  resolveConfig,
  savePersisted,
  providerFor,
  type MultiContentConfig,
  type ProxyMode,
  type VideoStrategy,
  type AudioStrategy,
} from "./src/config.js";
import {
  extractCandidatePaths,
  mediaFromAttachment,
  buildParts,
  readMediaBytes,
  cropImageFile,
  audioFormatFromMime,
  ffmpegAvailable,
  getMediaDuration,
  extractAudioSegments,
  classifyExt,
  type CropSpec,
  type MediaFile,
  type MediaKind,
} from "./src/media.js";
import { callMultimodalProxy, callStt } from "./src/proxy.js";
import type { MediaPart } from "./src/types.js";

// ── Constants & per-session state ───────────────────────────────────────────

/** Maximum analyze_media calls per agent turn (cost runaway guard). */
const MAX_TOOL_CALLS_PER_TURN = 10;
const TOOL_CACHE_SIZE = 50;

class LRU<V> {
  private map = new Map<string, V>();
  constructor(private cap: number) {}
  get(k: string): V | undefined {
    const v = this.map.get(k);
    if (v === undefined) return undefined;
    this.map.delete(k);
    this.map.set(k, v);
    return v;
  }
  set(k: string, v: V): void {
    if (this.map.has(k)) this.map.delete(k);
    this.map.set(k, v);
    if (this.map.size > this.cap) {
      const first = this.map.keys().next().value;
      if (first !== undefined) this.map.delete(first as string);
    }
  }
}

interface SessionState {
  toolCache: LRU<string, string>;
  toolCallCount: number;
}

const sessionStates = new WeakMap<object, SessionState>();

function getSessionState(ctx: ExtensionContext): SessionState {
  const key = ctx.sessionManager as unknown as object;
  let s = sessionStates.get(key);
  if (!s) {
    s = { toolCache: new LRU<string, string>(TOOL_CACHE_SIZE), toolCallCount: 0 };
    sessionStates.set(key, s);
  }
  return s;
}

// ── Prompts & rendering ─────────────────────────────────────────────────────

function defaultMediaPrompt(kind: MediaKind): string {
  if (kind === "image") {
    return "Describe this image in detail and factually. Transcribe all visible text, code, UI labels, numbers, diagrams, and the spatial layout. If it is a screenshot, name the app and note any errors or warnings.";
  }
  if (kind === "audio") {
    return "Transcribe this audio verbatim. Mark distinct speakers if identifiable (e.g. Speaker 1 / Speaker 2). Then add a one-line topic summary.";
  }
  return "These are sampled frames and the audio track from a video. Describe what happens across the frames in chronological order, transcribe any speech, and give a concise summary of the video's content and any on-screen text.";
}

/** Materialize an attachment (which has no path/url) to a temp file so the
 *  analyze_media tool can read it by reference. Returns the temp path, or
 *  undefined if the media is not an analyzable attachment. */
async function attachmentToTempPath(media: MediaFile): Promise<string | undefined> {
  if (media.source !== "attachment" || !media.data) return undefined;
  const mime = media.mimeType || "image/png";
  const ext = mime.includes("jpeg") || mime.includes("jpg")
    ? "jpg"
    : mime.includes("gif")
      ? "gif"
      : mime.includes("webp")
        ? "webp"
        : "png";
  const dir = join(tmpdir(), "multi-content-proxy");
  await mkdir(dir, { recursive: true });
  const p = join(dir, `attached-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.${ext}`);
  await writeFile(p, Buffer.from(media.data, "base64"));
  return p;
}

function maskKey(key: string): string {
  if (!key) return "(none)";
  return key.length <= 8 ? "****" : key.slice(0, 4) + "****" + key.slice(-4);
}

// ── Core processing ─────────────────────────────────────────────────────────

/**
 * Send one media item to the proxy and return the resulting text.
 * Audio + transcribe strategy uses the dedicated STT endpoint.
 * Long audio (>AUDIO_CHUNK_SECONDS) is chunked — dots.ai returns an empty
 * response for a single huge audio payload — then merged.
 */
const AUDIO_CHUNK_SECONDS = 110; // dots-ai audio chunk threshold

async function callForMedia(
  media: MediaFile,
  prompt: string,
  config: MultiContentConfig,
  ctx: ExtensionContext,
  opts: { frames?: number } = {},
): Promise<string> {
  const provider = providerFor(media.kind, config);
  if (media.kind === "audio" && config.audioStrategy === "transcribe") {
    let data = media.data;
    let format = audioFormatFromMime(media.mimeType || "audio/mpeg");
    if (media.source === "path") {
      const r = await readMediaBytes(media.path!, config);
      data = r.data;
      format = audioFormatFromMime(r.mimeType);
    } else if (media.source === "attachment") {
      format = audioFormatFromMime(media.mimeType || "audio/mpeg");
    } else {
      throw new Error("cannot transcribe a remote audio URL (download it first)");
    }
    return callStt({ provider, data: data!, format, signal: ctx.signal });
  }

  // Long audio: chunk into <=AUDIO_CHUNK_SECONDS slices so dots.ai doesn't drop
  // the whole payload, analyze each, then merge.
  if (media.kind === "audio" && media.source === "path") {
    const dur = await getMediaDuration(media.path!, config.ffmpegPath, ctx.signal);
    if (dur && dur > AUDIO_CHUNK_SECONDS) {
      const segs = await extractAudioSegments(media.path!, AUDIO_CHUNK_SECONDS, config.ffmpegPath, ctx.signal);
      const out: string[] = [];
      for (let i = 0; i < segs.length; i++) {
        const parts: MediaPart[] = [
          { type: "text", text: `${prompt} (segment ${i + 1}/${segs.length})` },
          { type: "audio_url", audio_url: { url: `data:${segs[i].mimeType};base64,${segs[i].data}` } },
        ];
        const r = await callMultimodalProxy({ provider, parts, prompt, maxTokens: 8192, signal: ctx.signal });
        out.push(r.text);
      }
      return out.join("\n\n— — —\n\n");
    }
  }

  // Video: native→frames auto-fallback for large/rejected payloads.
  if (media.kind === "video") return callVideo(media, prompt, config, ctx, opts);

  const parts = await buildParts(media, config, { frames: opts.frames, signal: ctx.signal });
  // dots.ai is a reasoning model: audio/video need a large budget or it spends
  // it all thinking and returns empty content.
  const maxTokens = media.kind === "audio" ? 8192 : 4096;
  const r = await callMultimodalProxy({ provider, parts, prompt, maxTokens, signal: ctx.signal });
  return r.text;
}

/** Video: prefer native (whole clip → best motion understanding) but fall back
 *  to frames if the provider rejects the large video_url payload (dots.ai
 *  returns HTTP 400 for big files). Returns the first successful description. */
async function callVideo(
  media: MediaFile,
  prompt: string,
  config: MultiContentConfig,
  ctx: ExtensionContext,
  opts: { frames?: number } = {},
): Promise<string> {
  const provider = providerFor("video", config);
  const strategies: VideoStrategy[] = config.videoStrategy === "native" ? ["native", "frames"] : ["frames"];
  let lastErr: any;
  for (const strat of strategies) {
    try {
      const parts = await buildParts(media, { ...config, videoStrategy: strat }, { frames: opts.frames, signal: ctx.signal });
      const r = await callMultimodalProxy({ provider, parts, prompt, maxTokens: 8192, signal: ctx.signal });
      return r.text;
    } catch (e: any) {
      lastErr = e;
      if (strategies.length > 1) {
        ctx.ui.notify(`[multi-content-proxy] video ${strat} failed (${e?.message ?? e}); retrying with frames…`, "warning");
      }
    }
  }
  throw lastErr ?? new Error("video analysis failed");
}

/** First-use data-egress consent gate. */
async function ensureConsent(ctx: ExtensionContext, config: MultiContentConfig): Promise<boolean> {
  if (config.consent === "yes") return true;
  if (config.consent === "no") return false;
  if (!ctx.hasUI) return false;
  const ok = await ctx.ui.confirm(
    "multi-content-proxy",
    "Send the referenced media files to the configured multimodal proxy to describe/transcribe them?",
  );
  if (ok) await savePersisted({ consent: "yes" });
  return ok;
}

// ── Extension entry point ───────────────────────────────────────────────────

export default function (pi: ExtensionAPI) {
  // Reset per-turn tool-call counter.
  pi.on("before_agent_start", (_event: BeforeAgentStartEvent, ctx: ExtensionContext) => {
    getSessionState(ctx).toolCallCount = 0;
  });

  // Steady status line.
  pi.on("session_start", (_event: SessionStartEvent, ctx: ExtensionContext) => {
    const config = resolveConfig();
    if (config.mode !== "off" && config.statusLine === "on") {
      ctx.ui.setStatus(
        "multi-content-proxy",
        "📎",
      );
    } else {
      ctx.ui.setStatus("multi-content-proxy", undefined);
    }
  });

  // ── Input hook: detect media and ask the model to analyze it via the tool ──
  // We deliberately do NOT analyze media here — that work is slow and would
  // make the user wait with no feedback. Instead we inject a short hint listing
  // the media references and instruct the model to call `analyze_media` in its
  // own loop, so analysis happens as a visible, progress-reporting tool call.
  pi.on("input", async (event: InputEvent, ctx: ExtensionContext) => {
    const config = resolveConfig();
    if (config.mode === "off") return { action: "continue" };
    if (event.source === "extension") return { action: "continue" };
    if (config.consent === "no") return { action: "continue" };

    const modelInput = (ctx.model?.input as string[] | undefined) ?? [];
    const modelSupportsImage = modelInput.includes("image");

    // Gather candidates.
    const candidates: MediaFile[] = [];
    for (const img of event.images ?? []) {
      if (config.enableImage) candidates.push(mediaFromAttachment(img as ImageContent));
    }
    for (const c of extractCandidatePaths(event.text, ctx.cwd)) {
      if (c.kind === "image" && config.enableImage) candidates.push(c);
      else if (c.kind === "audio" && config.enableAudio) candidates.push(c);
      else if (c.kind === "video" && config.enableVideo) candidates.push(c);
    }
    if (candidates.length === 0) return { action: "continue" };

    // Decide which media the model cannot ingest natively and must analyze via
    // the tool. In fallback mode, natively-seen images stay as attachments.
    let toolMedia = candidates;
    if (config.mode === "fallback") {
      toolMedia = candidates.filter((c) => !(c.kind === "image" && modelSupportsImage));
    }
    if (toolMedia.length === 0) return { action: "continue" };

    // Resolve each to a reference the model can pass to analyze_media.
    // Attachments (no path/url) are materialized to a temp file first.
    const refs: string[] = [];
    for (const media of toolMedia) {
      if (media.source === "attachment") {
        const tmp = await attachmentToTempPath(media);
        if (tmp) refs.push(`- ${media.kind}: ${tmp}`);
      } else if (media.url) {
        refs.push(`- ${media.kind}: ${media.url}`);
      } else if (media.path) {
        refs.push(`- ${media.kind}: ${media.path}`);
      }
    }
    if (refs.length === 0) return { action: "continue" };

    const hint =
      "<multi_content_proxy>\n" +
      "The user referenced media that cannot be ingested natively. To perceive it, " +
      "call the `analyze_media` tool with the references below (pair each with a " +
      "question suited to the user's request). Analyze before answering whenever " +
      "the media is relevant:\n" +
      refs.join("\n") +
      "\n</multi_content_proxy>";

    const newText = event.text ? `${event.text}\n\n${hint}` : hint;

    // If images are being proxied (model can't see them natively), drop the raw
    // attachments so we don't send the same image twice.
    const proxiedImages = toolMedia.some((c) => c.kind === "image");
    const images = proxiedImages && !modelSupportsImage ? [] : event.images;

    return { action: "transform", text: newText, images };
  });

  // ── On-demand analyze_media tool ───────────────────────────────────────────
  const CropEntrySchema = Type.Union([
    Type.Object({
      image_index: Type.Integer({ minimum: 0 }),
      region: Type.String({ description: "top-left|top-right|bottom-left|bottom-right|top|bottom|left|right|center|*-half" }),
    }),
    Type.Object({
      image_index: Type.Integer({ minimum: 0 }),
      normalized: Type.Object({ x: Type.Number(), y: Type.Number(), width: Type.Number(), height: Type.Number() }),
    }),
    Type.Object({
      image_index: Type.Integer({ minimum: 0 }),
      pixels: Type.Object({ x: Type.Number(), y: Type.Number(), width: Type.Number(), height: Type.Number() }),
    }),
  ]);

  pi.registerTool({
    name: "analyze_media",
    label: "Analyze Media",
    description: [
      "Use `analyze_media` to describe, transcribe, or ask a specific question about image, audio, or video files.",
      "Pass file paths or http(s) URLs. Images can be cropped via `crop`; videos accept a `frames` sample count.",
      "The tool sends the parsed media to the configured multimodal proxy and returns the model's text answer.",
    ].join("\n"),
    parameters: Type.Object({
      media: Type.Array(Type.String(), {
        description: "1..20 media references: local file paths or http(s) URLs (image/audio/video).",
        minItems: 1,
        maxItems: 20,
      }),
      question: Type.String({ description: "Question or instruction about the media. Required." }),
      crop: Type.Optional(Type.Array(CropEntrySchema, { description: "Optional per-image crop, indexed by image_index." })),
      frames: Type.Optional(Type.Integer({ minimum: 1, maximum: 30, description: "Frames to sample per video (frames strategy)." })),
      reason: Type.Optional(Type.String({ description: "Optional; logged for analytics only." })),
    }),
    async execute(_toolCallId, params: any, _signal, _onUpdate, ctx: ExtensionContext) {
      const config = resolveConfig();
      if (config.mode === "off") {
        return { content: [{ type: "text", text: "multi-content-proxy is disabled (mode=off)." }], details: {} };
      }
      const state = getSessionState(ctx);
      if (state.toolCallCount >= MAX_TOOL_CALLS_PER_TURN) {
        return {
          content: [{ type: "text", text: `analyze_media rate limit reached (${MAX_TOOL_CALLS_PER_TURN}/turn).` }],
          details: {},
        };
      }
      state.toolCallCount++;

      if (!(await ensureConsent(ctx, config))) {
        return { content: [{ type: "text", text: "Consent denied — cannot send media to the proxy." }], details: {} };
      }
      if (!config.image.baseUrl) {
        return {
          content: [{ type: "text", text: "multi-content-proxy image base URL is not configured. Set MULTI_CONTENT_PROXY_IMAGE_BASE_URL or use /multi-content-proxy image-base-url <url>." }],
          details: {},
        };
      }

      // Cache key.
      const cacheKey = JSON.stringify({ m: params.media, q: params.question, c: params.crop, f: params.frames });
      const cached = state.toolCache.get(cacheKey);
      if (cached) return { content: [{ type: "text", text: cached }], details: {} };

      // Resolve media refs.
      const mediaList: MediaFile[] = [];
      for (const ref of params.media as string[]) {
        const s = ref.trim();
        if (!s) continue;
        if (/^https?:\/\//i.test(s)) {
          const m = /\.([a-z0-9]+)$/i.exec(s);
          const kind = m ? classifyExt("." + m[1]) : undefined;
          if (kind) mediaList.push({ kind, url: s, label: s, source: "url" });
          continue;
        }
        const resolved = resolveLocalMedia(s, ctx.cwd);
        if (resolved) mediaList.push(resolved);
        else ctx.ui.notify(`[multi-content-proxy] skipped unreadable media: ${s}`, "warning");
      }
      if (mediaList.length === 0) {
        return { content: [{ type: "text", text: "No valid media references found." }], details: {} };
      }

      // Apply crops (images only).
      if (params.crop && Array.isArray(params.crop)) {
        if (await ffmpegAvailable(config.ffmpegPath)) {
          for (const c of params.crop as CropSpec[]) {
            const target = mediaList.filter((m) => m.kind === "image")[c.image_index];
            if (target && target.source === "path") {
              try {
                const cropped = await cropImageFile(target.path!, c, config.ffmpegPath, ctx.signal);
                target.path = cropped;
              } catch (err: any) {
                ctx.ui.notify(`[multi-content-proxy] crop failed: ${err?.message ?? err}`, "warning");
              }
            }
          }
        } else {
          ctx.ui.notify("[multi-content-proxy] ffmpeg not found — ignoring crop", "warning");
        }
      }

      // ffmpeg for video frames.
      const needsFfmpeg = mediaList.some((m) => m.kind === "video" && config.videoStrategy === "frames");
      let effective = config;
      if (needsFfmpeg && !(await ffmpegAvailable(config.ffmpegPath))) {
        ctx.ui.notify("[multi-content-proxy] ffmpeg missing — video uses native strategy", "warning");
        effective = { ...config, videoStrategy: "native" };
      }

      try {
        const answers: string[] = [];
        for (const media of mediaList) {
          const text = await callForMedia(media, params.question as string, effective, ctx, {
            frames: params.frames as number | undefined,
          });
          answers.push(`### ${media.kind}: ${media.label}\n${text}`);
        }
        const out = answers.join("\n\n");
        state.toolCache.set(cacheKey, out);
        return { content: [{ type: "text", text: out }], details: {} };
      } catch (err: any) {
        return {
          content: [{ type: "text", text: `analyze_media failed: ${err?.message ?? err}` }],
          details: { error: String(err?.message ?? err) },
        };
      }
    },
  });

  // ── Configuration command ─────────────────────────────────────────────────
  pi.registerCommand("multi-content-proxy", {
    description: "Configure the multi-content-proxy (mode, model, endpoints, strategies).",
    handler: async (args: string, ctx: ExtensionContext) => {
      const trimmed = args.trim();
      if (!trimmed) {
        await showStatus(ctx);
        return;
      }
      const [cmd, ...rest] = trimmed.split(/\s+/);
      const arg = rest.join(" ");
      try {
        // --- mode ---
        if (cmd === "off" || cmd === "fallback" || cmd === "always") {
          await savePersisted({ mode: cmd as ProxyMode });
          ctx.ui.notify(`multi-content-proxy mode → ${cmd}`, "info");
          return;
        }

        // --- per-modality provider setters ---
        // aliases: model/base-url/api-key  ->  image-*
        const alias: Record<string, string> = {
          model: "image-model",
          "base-url": "image-base-url",
          "api-key": "image-api-key",
        };
        const realCmd = alias[cmd] ?? cmd;
        const provMatch = /^(image|audio|video)-(model|base-url|api-key)$/.exec(realCmd);
        if (provMatch) {
          const kind = provMatch[1] as MediaKind;
          const field = provMatch[2] as "model" | "base-url" | "api-key";
          const cfgField = field === "base-url" ? "baseUrl" : (field === "api-key" ? "apiKey" : "model");
          if (!arg) {
            ctx.ui.notify(`usage: /multi-content-proxy ${realCmd} <value>  (or 'clear' for audio/video to fall back to image)`, "warning");
            return;
          }
          if (kind !== "image" && arg.toLowerCase() === "clear") {
            await clearProvider(kind, ctx);
            return;
          }
          await setProviderField(kind, cfgField, arg, ctx);
          return;
        }

        // --- strategy / limits ---
        if (cmd === "video-strategy") {
          if (arg !== "native" && arg !== "frames") return ctx.ui.notify("video-strategy must be native|frames", "warning");
          await savePersisted({ videoStrategy: arg as VideoStrategy });
          ctx.ui.notify(`video-strategy → ${arg}`, "info");
          return;
        }
        if (cmd === "audio-strategy") {
          if (arg !== "describe" && arg !== "transcribe") return ctx.ui.notify("audio-strategy must be describe|transcribe", "warning");
          await savePersisted({ audioStrategy: arg as AudioStrategy });
          ctx.ui.notify(`audio-strategy → ${arg}`, "info");
          return;
        }
        if (cmd === "max-frames") {
          await savePersisted({ maxFrames: Math.max(1, parseInt(arg, 10) || 4) });
          ctx.ui.notify(`max-frames → ${arg}`, "info");
          return;
        }
        if (cmd === "max-bytes") {
          const bytes = parseMaxBytes(arg);
          if (bytes === null) return ctx.ui.notify("max-bytes: use <n> (bytes) or <n>mb", "warning");
          await savePersisted({ maxBytes: bytes });
          ctx.ui.notify(`max-bytes → ${(bytes / 1024 / 1024).toFixed(1)}MB`, "info");
          return;
        }
        if (cmd === "ffmpeg") {
          if (!arg) return ctx.ui.notify("usage: /multi-content-proxy ffmpeg <path>", "warning");
          await savePersisted({ ffmpegPath: arg });
          ctx.ui.notify(`ffmpeg → ${arg}`, "info");
          return;
        }
        if (cmd === "status") {
          if (arg !== "on" && arg !== "off") return ctx.ui.notify("status must be on|off", "warning");
          await savePersisted({ statusLine: arg as "on" | "off" });
          ctx.ui.notify(`status line → ${arg}`, "info");
          return;
        }
        if (cmd === "consent") {
          if (arg !== "yes" && arg !== "no" && arg !== "ask") return ctx.ui.notify("consent must be yes|no|ask", "warning");
          await savePersisted({ consent: arg as "yes" | "no" | "ask" });
          ctx.ui.notify(`consent → ${arg}`, "info");
          return;
        }
        if (cmd === "folders") {
          await handleFolders(arg, ctx);
          return;
        }
        if (cmd === "test") {
          await runTest(arg, ctx);
          return;
        }
        if (cmd === "reset-consent") {
          await savePersisted({ consent: "ask" });
          ctx.ui.notify("consent reset to ask", "info");
          return;
        }
        ctx.ui.notify(`unknown subcommand: ${cmd}`, "warning");
        await showStatus(ctx);
      } catch (err: any) {
        ctx.ui.notify(`multi-content-proxy error: ${err?.message ?? err}`, "error");
      }
    },
  });
}

  // ── Natural-language configuration tool ─────────────────────────────────────
  // Lets the user configure the extension just by telling the assistant, e.g.
  // "帮我配置 multi-content-proxy 的 baseurl 为 …/v1 模型名为 dots3-note-prev
  //  APIKEY为 ak_xxx". The model parses that into this tool call.
  function normalizeBaseUrl(u: string): string {
    let s = (u || "").trim();
    if (s.endsWith("/chat/completions")) s = s.slice(0, -"/chat/completions".length);
    else if (s.endsWith("/audio/transcriptions")) s = s.slice(0, -"/audio/transcriptions".length);
    if (s.endsWith("/")) s = s.slice(0, -1);
    return s;
  }

  // Only register the configuration tool when the extension is NOT yet configured,
  // so it never appears in the model's context once setup is complete.
  const isConfigured = (c: ReturnType<typeof resolveConfig>): boolean => !!(c.image.baseUrl && c.image.apiKey);
  const configuredAtLoad = isConfigured(resolveConfig());

  if (!configuredAtLoad) {
  pi.tool({
    name: "configure_multi_content_proxy",
    description: [
      "Configure the multi-content-proxy pi extension by setting its API endpoint, model, and key.",
      "Call this whenever the user asks to set up / configure multi-content-proxy, or supplies its base URL, model name, or API key — including in natural language such as '帮我配置拓展 multi-content-proxy 的 baseurl 为 https://note3-prev-api.askdiandian.com/v1 模型名为 dots3-note-prev APIKEY为 ak_xxx'.",
      "baseUrl is the OpenAI-compatible …/v1 root (a trailing /chat/completions is stripped automatically). model is the multimodal model id (applied to image + video). apiKey is applied globally and to all providers. audio/video inherit from image unless they were already configured separately.",
    ].join(" "),
    input: {
      baseUrl: Type.Optional(Type.String({ description: "OpenAI-compatible base URL ending in /v1, e.g. https://note3-prev-api.askdiandian.com/v1" })),
      model: Type.Optional(Type.String({ description: "Multimodal model id, e.g. dots3-note-prev. Applied to image + video providers." })),
      apiKey: Type.Optional(Type.String({ description: "API key for the endpoint, e.g. ak_xxx." })),
      consent: Type.Optional(Type.Union([Type.Literal("yes"), Type.Literal("no"), Type.Literal("ask")], { description: "Media egress consent: yes | no | ask." })),
      mode: Type.Optional(Type.Union([Type.Literal("always"), Type.Literal("fallback"), Type.Literal("off")], { description: "Proxy mode: always | fallback | off." })),
    },
    execute: async (params, ctx) => {
      const cur = resolveConfig();
      const changed: string[] = [];

      if (params.baseUrl) {
        const u = normalizeBaseUrl(params.baseUrl);
        await setProviderField("image", "baseUrl", u, ctx);
        if (cur.audio) await setProviderField("audio", "baseUrl", u, ctx);
        if (cur.video) await setProviderField("video", "baseUrl", u, ctx);
        changed.push(`baseUrl → ${u} (image${cur.audio ? "+audio" : ""}${cur.video ? "+video" : ""})`);
      }
      if (params.model) {
        await setProviderField("image", "model", params.model, ctx);
        if (cur.video) await setProviderField("video", "model", params.model, ctx);
        changed.push(`model → ${params.model} (image+video)`);
      }
      if (params.apiKey) {
        await savePersisted({ apiKey: params.apiKey });
        await setProviderField("image", "apiKey", params.apiKey, ctx);
        if (cur.audio) await setProviderField("audio", "apiKey", params.apiKey, ctx);
        if (cur.video) await setProviderField("video", "apiKey", params.apiKey, ctx);
        changed.push("apiKey → set (hidden)");
      }
      if (params.consent) { await savePersisted({ consent: params.consent }); changed.push(`consent → ${params.consent}`); }
      if (params.mode) { await savePersisted({ mode: params.mode }); changed.push(`mode → ${params.mode}`); }

      const c = resolveConfig();
      await showStatus(ctx);
      return {
        content: [{
          type: "text",
          text:
            `✅ multi-content-proxy 已配置完成：\n` +
            changed.map((x) => `- ${x}`).join("\n") +
            `\n\n当前 image 配置：\n  baseUrl: ${c.image.baseUrl}\n  model:   ${c.image.model}\n  apiKey:  ${maskKey(c.apiKey || c.image.apiKey || "")}`,
        }],
      };
    },
  });
  }

  // Once the extension is configured, drop the configuration tool from the active
  // tool set so it doesn't clutter the model's context. Re-show it if the config
  // is later cleared (e.g. apiKey removed).
  pi.on("before_agent_start", () => {
    if (configuredAtLoad) return; // tool was never registered; nothing to toggle
    const configured = isConfigured(resolveConfig());
    const active = pi.getActiveTools();
    const has = active.includes("configure_multi_content_proxy");
    if (configured && has) {
      pi.setActiveTools(active.filter((t) => t !== "configure_multi_content_proxy"));
    } else if (!configured && !has) {
      pi.setActiveTools([...active, "configure_multi_content_proxy"]);
    }
  });

// ── Command helpers ─────────────────────────────────────────────────────────

function classifyFromExt(ext: string): MediaKind | undefined {
  return classifyExt(ext);
}

function resolveLocalMedia(ref: string, cwd: string): MediaFile | undefined {
  let p = ref;
  if (ref.startsWith("~")) p = resolve(homedir(), ref.slice(1));
  else if (!isAbsolute(ref)) p = resolve(cwd, ref);
  try {
    if (!statSync(p).isFile()) return undefined;
  } catch {
    return undefined;
  }
  const m = /\.([a-z0-9]+)$/i.exec(p);
  const kind = m ? classifyExt("." + m[1]) : undefined;
  if (!kind) return undefined;
  return { kind, path: p, label: basename(p), source: "path" };
}

function parseMaxBytes(arg: string): number | null {
  const t = arg.trim().toLowerCase();
  const mb = /^(\d+(?:\.\d+)?)\s*mb$/.exec(t);
  if (mb) return Math.round(parseFloat(mb[1]) * 1024 * 1024);
  if (/^\d+$/.test(t)) return parseInt(t, 10);
  return null;
}

/** Set one field of a modality provider, inheriting unset fields from the image (base) provider. */
async function setProviderField(
  kind: MediaKind,
  field: "baseUrl" | "apiKey" | "model",
  value: string,
  ctx: ExtensionContext,
): Promise<void> {
  const c = resolveConfig();
  const current = kind === "image" ? c.image : kind === "audio" ? c.audio ?? c.image : c.video ?? c.image;
  const merged = { ...c.image, ...current, [field]: value };
  if (kind === "image") await savePersisted({ image: merged });
  else if (kind === "audio") await savePersisted({ audio: merged });
  else await savePersisted({ video: merged });
  ctx.ui.notify(`${kind} ${field} saved`, "info");
}

/** Remove the dedicated audio/video provider so it falls back to the image provider. */
async function clearProvider(kind: MediaKind, ctx: ExtensionContext): Promise<void> {
  if (kind === "image") {
    ctx.ui.notify("image is the base provider and cannot be cleared", "warning");
    return;
  }
  if (kind === "audio") await savePersisted({ audio: undefined });
  else await savePersisted({ video: undefined });
  ctx.ui.notify(`${kind} provider cleared — now falls back to image`, "info");
}

async function showStatus(ctx: ExtensionContext): Promise<void> {
  const c = resolveConfig();
  const line = (label: string, p: { baseUrl: string; apiKey: string; model: string }, fallback?: boolean) =>
    `  ${label.padEnd(7)} ${fallback ? "(fallback→image) " : ""}model=${p.model}  url=${p.baseUrl || "(not set)"}  key=${maskKey(p.apiKey)}`;
  const lines = [
    "multi-content-proxy",
    `  mode:              ${c.mode}`,
    line("image", c.image),
    line("audio", c.audio ?? c.image, !c.audio),
    line("video", c.video ?? c.image, !c.video),
    `  image/audio/video: ${c.enableImage}/${c.enableAudio}/${c.enableVideo}`,
    `  video-strategy:    ${c.videoStrategy}`,
    `  audio-strategy:    ${c.audioStrategy}`,
    `  max-bytes:         ${(c.maxBytes / 1024 / 1024).toFixed(1)}MB`,
    `  max-frames:        ${c.maxFrames}`,
    `  ffmpeg:            ${c.ffmpegPath}`,
    `  consent:           ${c.consent}`,
    `  status-line:       ${c.statusLine}`,
  ];
  ctx.ui.notify(lines.join("\n"), "info");
}

async function handleFolders(arg: string, ctx: ExtensionContext): Promise<void> {
  const c = resolveConfig();
  const [sub, ...rest] = arg.split(/\s+/);
  const path = rest.join(" ");
  if (sub === "clear") {
    await savePersisted({ allowedFolders: [] });
    ctx.ui.notify("allowed folders cleared (any readable file allowed)", "info");
    return;
  }
  if (sub === "add" && path) {
    const abs = resolve(path);
    await savePersisted({ allowedFolders: [...new Set([...c.allowedFolders, abs])] });
    ctx.ui.notify(`added allowed folder: ${abs}`, "info");
    return;
  }
  if (sub === "remove" && path) {
    const abs = resolve(path);
    await savePersisted({ allowedFolders: c.allowedFolders.filter((f) => resolve(f) !== abs) });
    ctx.ui.notify(`removed allowed folder: ${abs}`, "info");
    return;
  }
  ctx.ui.notify(`allowed folders: ${c.allowedFolders.length ? c.allowedFolders.join(", ") : "(any)"}`, "info");
}

async function runTest(arg: string, ctx: ExtensionContext): Promise<void> {
  if (!arg) return ctx.ui.notify("usage: /multi-content-proxy test <path|url>", "warning");
  const c = resolveConfig();
  const resolved = resolveLocalMedia(arg, ctx.cwd);
  const media: MediaFile | undefined =
    resolved ??
    (/^https?:\/\//i.test(arg)
      ? {
          kind: (classifyExt("." + (/\.([a-z0-9]+)$/i.exec(arg)?.[1] ?? "")) || "image") as MediaKind,
          url: arg,
          label: arg,
          source: "url",
        }
      : undefined);
  if (!media) return ctx.ui.notify(`cannot resolve test target: ${arg}`, "warning");
  if (!providerFor(media.kind, c).baseUrl) return ctx.ui.notify("base-url not configured", "warning");
  ctx.ui.notify(`[multi-content-proxy] testing ${media.label} …`, "info");
  try {
    const text = await callForMedia(media, defaultMediaPrompt(media.kind), c, ctx);
    ctx.ui.notify(`${media.label}:\n${text.slice(0, 4000)}`, "info");
  } catch (err: any) {
    ctx.ui.notify(`test failed: ${err?.message ?? err}`, "error");
  }
}
