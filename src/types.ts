/**
 * Shared types for the multi-content-proxy pi extension.
 *
 * The proxy forwards parsed media to an OpenAI-compatible multimodal endpoint
 * (dots.ai / OpenRouter / any chat-completions server that accepts rich
 * content parts). The content-part shape follows the de-facto standard:
 *
 *   - image : { type: "image_url",      image_url: { url } }
 *   - video : { type: "video_url",      video_url: { url } }   (native)
 *             or several image_url parts + one audio_url         (frames)
 *   - audio : { type: "audio_url",      audio_url: { url } }   (data URI)
 */

export type MediaKind = "image" | "audio" | "video";

export type MediaSource = "path" | "attachment" | "url";

/** A media item discovered in the conversation (file, attachment, or URL). */
export interface MediaFile {
  kind: MediaKind;
  /** Filesystem path (source "path"). */
  path?: string;
  /** Raw base64 (no data: prefix) — used for attachments. */
  data?: string;
  /** MIME type — required for attachments / raw bytes. */
  mimeType?: string;
  /** Remote http(s) URL (source "url"). */
  url?: string;
  /** Human-friendly label (filename or "attached"). */
  label: string;
  source: MediaSource;
}

/** An OpenAI-compatible multimodal content part.
 *
 * dots.ai (and most OpenAI-compatible multimodal servers) accept image/audio/
 * video as data-URI content parts using the `image_url` / `audio_url` /
 * `video_url` shapes. (The older OpenAI `input_audio` shape is intentionally
 * not used here because dots.ai rejects it.) */
export type MediaPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string } }
  | { type: "video_url"; video_url: { url: string } }
  | { type: "audio_url"; audio_url: { url: string } };

/** Result returned by the remote multimodal proxy. */
export interface ProxyResult {
  text: string;
  model?: string;
  usage?: {
    promptTokens?: number;
    completionTokens?: number;
    totalTokens?: number;
  };
}
