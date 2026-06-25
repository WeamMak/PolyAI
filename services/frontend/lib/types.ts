export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  image_base64?: string;
}

export interface ChatResponse {
  response: string;
  prediction_id: string | null;
  annotated_image: string | null;
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
