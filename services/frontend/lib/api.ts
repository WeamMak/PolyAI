import type { ChatMessage, ChatResponse } from "./types";

const AGENT_URL = process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8000";

export async function sendMessage(messages: ChatMessage[]): Promise<ChatResponse> {
  const requestMessages = messages.map((message) => ({
    role: message.role,
    content: message.content,
    ...(message.image_base64 ? { image_base64: message.image_base64 } : {}),
  }));

  const res = await fetch(`${AGENT_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: requestMessages }),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(text || res.statusText);
  }
  return (await res.json()) as ChatResponse;
}
