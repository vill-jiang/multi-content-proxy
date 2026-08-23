/**
 * Remote multimodal proxy client.
 *
 * Talks to any OpenAI-compatible chat-completions endpoint that accepts rich
 * content parts (image_url / video_url / audio_url as data URIs) — e.g. dots.ai's platform
 * API, OpenRouter, or a local vLLM/Ollama. Optionally falls back to a separate
 * /audio/transcriptions endpoint for speech-only transcription.
 */

import { Buffer } from "node:buffer";
import type { MediaPart, ProxyResult } from "./types.js";
import type { ProviderConfig } from "./config.js";

const REQUEST_TIMEOUT_MS = 600_000; // dots.ai is a reasoning model; large media needs headroom

function withTimeout(signal: AbortSignal | undefined, ms: number): AbortSignal {
  const timeout = AbortSignal.timeout(ms);
  return signal ? AbortSignal.any([signal, timeout]) : timeout;
}

/** Pull text out of an OpenAI-style message (string or content-array). */
function extractText(message: any): string {
  if (!message || typeof message !== "object") return "";
  const content = message.content;
  if (typeof content === "string" && content.trim()) return content;
  if (Array.isArray(content)) {
    const joined = content
      .map((part: any) => (typeof part === "string" ? part : part?.text))
      .filter(Boolean)
      .join("");
    if (joined.trim()) return joined;
  }
  if (typeof message.reasoning_content === "string" && message.reasoning_content.trim()) {
    return message.reasoning_content;
  }
  return "";
}

/**
 * Call the multimodal chat-completions endpoint with the given media parts
 * and a text prompt. Returns the model's text plus light usage info.
 */
export async function callMultimodalProxy(opts: {
  provider: ProviderConfig;
  parts: MediaPart[];
  prompt: string;
  /** Generation budget. dots.ai is a reasoning model: audio/video need a large
   *  budget (e.g. 8192) or it spends it all thinking and returns empty content. */
  maxTokens?: number;
  signal?: AbortSignal;
}): Promise<ProxyResult> {
  const { provider, parts, prompt, signal, maxTokens = 4096 } = opts;

  const body = {
    model: provider.model,
    stream: false,
    max_tokens: maxTokens,
    messages: [
      {
        role: "user",
        content: [...parts, { type: "text", text: prompt }],
      },
    ],
  };

  let res: Response;
  try {
    res = await fetch(provider.baseUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(provider.apiKey ? { Authorization: `Bearer ${provider.apiKey}` } : {}),
      },
      body: JSON.stringify(body),
      signal: withTimeout(signal, REQUEST_TIMEOUT_MS),
    });
  } catch (err: any) {
    if (err?.name === "TimeoutError") throw new Error(`proxy request timed out (${REQUEST_TIMEOUT_MS / 1000}s)`);
    if (err?.name === "AbortError") throw new AbortError("cancelled");
    throw new Error(`proxy request failed: ${err?.message ?? err}`);
  }

  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(`proxy HTTP ${res.status}: ${t.slice(0, 300)}`);
  }

  const data = (await res.json().catch(() => null)) as any;
  const choice = data?.choices?.[0];
  const text = extractText(choice?.message);
  if (!text) {
    const hint = data?.error?.message ? `: ${String(data.error.message).slice(0, 200)}` : "";
    throw new Error(`proxy returned no content${hint}`);
  }

  const u = choice?.message?.usage ?? data?.usage;
  const usage = u
    ? {
        promptTokens: u.prompt_tokens,
        completionTokens: u.completion_tokens,
        totalTokens: u.total_tokens,
      }
    : undefined;

  return { text, model: provider.model, usage };
}

/**
 * Speech-to-text via a separate /audio/transcriptions endpoint (OpenAI-style
 * multipart form). Used when audioStrategy === "transcribe".
 */
export async function callStt(opts: {
  provider: ProviderConfig;
  data: string;
  format: string;
  signal?: AbortSignal;
}): Promise<string> {
  const { provider, data, format, signal } = opts;
  const url = provider.baseUrl.replace(/chat\/completions\/?$/i, "audio/transcriptions");

  const form = new FormData();
  form.append("model", provider.model);
  const mime = format === "wav" ? "audio/wav" : format === "ogg" ? "audio/ogg" : "audio/mpeg";
  form.append(
    "file",
    new Blob([Buffer.from(data, "base64")], { type: mime }),
    `audio.${format}`,
  );

  let res: Response;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: provider.apiKey ? { Authorization: `Bearer ${provider.apiKey}` } : {},
      body: form,
      signal: withTimeout(signal, REQUEST_TIMEOUT_MS),
    });
  } catch (err: any) {
    if (err?.name === "AbortError") throw new AbortError("cancelled");
    throw new Error(`STT request failed: ${err?.message ?? err}`);
  }

  if (!res.ok) {
    const t = await res.text().catch(() => "");
    throw new Error(`STT HTTP ${res.status}: ${t.slice(0, 300)}`);
  }
  const dataJson = (await res.json().catch(() => null)) as any;
  const text = typeof dataJson?.text === "string" ? dataJson.text : await res.text().catch(() => "");
  if (!text.trim()) throw new Error("STT returned no text");
  return text;
}

class AbortError extends Error {
  constructor(message = "cancelled") {
    super(message);
    this.name = "AbortError";
  }
}
