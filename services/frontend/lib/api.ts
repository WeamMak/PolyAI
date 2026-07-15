import type { ChatMessage, ChatResponse } from "./types";

const AGENT_URL = process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8000";

function getFallbackErrorMessage(status: number, statusText: string): string {
  if (status >= 500) {
    return "The agent service is temporarily unavailable. Please try again later.";
  }

  return statusText || "Something went wrong. Please try again.";
}

function getErrorMessage(body: unknown, fallback: string, status: number): string {
  if (status >= 500) {
    return fallback;
  }

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

export async function sendMessage(
  messages: ChatMessage[],
  chatId: string | null,
  activeImageS3Key: string | null
): Promise<ChatResponse> {
  const requestMessages = messages.map((message, index) => ({
    role: message.role,
    content: message.content,
    ...(index === messages.length - 1 && message.image_base64
      ? { image_base64: message.image_base64 }
      : {}),
  }));

  const res = await fetch(`${AGENT_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: requestMessages,
      ...(chatId ? { chat_id: chatId } : {}),
      ...(activeImageS3Key
        ? { active_image_s3_key: activeImageS3Key }
        : {}),
    }),
  });
  if (!res.ok) {
    const fallbackMessage = getFallbackErrorMessage(res.status, res.statusText);
    const text = await res.text().catch(() => "");
    let message = fallbackMessage;

    try {
      message = getErrorMessage(JSON.parse(text), message, res.status);
    } catch {
      // Keep the fallback for non-JSON errors.
    }

    if (
      res.status === 429 &&
      (!message || message === res.statusText || message === fallbackMessage)
    ) {
      const retryAfter = res.headers.get("Retry-After");
      message = retryAfter
        ? `Rate limit reached. Please try again in ${retryAfter} seconds.`
        : "Rate limit reached. Please try again soon.";
    }

    throw new Error(message);
  }
  return (await res.json()) as ChatResponse;
}
