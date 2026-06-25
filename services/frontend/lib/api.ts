import type { ChatMessage, ChatResponse } from "./types";

const AGENT_URL = process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8000";

function getErrorMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail?: unknown }).detail;

    if (typeof detail === "string") {
      return detail;
    }

    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            const message = (item as { msg?: unknown }).msg;
            if (typeof message === "string") return message;
          }
          return String(item);
        })
        .join("\n");
    }
  }

  return fallback;
}

export async function sendMessage(messages: ChatMessage[]): Promise<string> {
  const res = await fetch(`${AGENT_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    let message = text || res.statusText;

    try {
      message = getErrorMessage(JSON.parse(text), message);
    } catch {
      // Non-JSON errors can still be shown as plain text.
    }

    if (res.status === 429 && (!message || message === res.statusText)) {
      const retryAfter = res.headers.get("Retry-After");
      message = retryAfter
        ? `Rate limit reached. Please try again in ${retryAfter} seconds.`
        : "Rate limit reached. Please try again soon.";
    }

    throw new Error(message);
  }
  const data = (await res.json()) as ChatResponse;
  return data.response;
}
