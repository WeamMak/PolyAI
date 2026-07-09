export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  image_base64?: string;
  image_media_type?: string;
  prediction_id?: string | null;
  annotated_image?: string | null;
  annotated_image_media_type?: string | null;
}

export interface ChatResponse {
  response: string;
  chat_id: string;
  active_image_s3_key: string | null;
  prediction_id: string | null;
  annotated_image: string | null;
  annotated_image_media_type: string | null;
  tokens_used: {
    input: number;
    output: number;
    total: number;
  };
  agent_loop_time_s: number;
  iterations: number;
  tools_called: string[];
  context_limit_exceeded: boolean;
}
