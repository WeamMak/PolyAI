import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/lib/types";

export default function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={cn("flex", isUser ? "justify-end" : "justify-start")}>
      <div
        className={cn(
          "max-w-[80%] rounded-2xl px-4 py-3 text-sm shadow-sm",
          isUser
            ? "bg-primary text-primary-foreground rounded-br-sm"
            : "bg-muted text-foreground rounded-bl-sm border border-border/50"
        )}
      >
        {message.image_base64 && (
          <img
            src={`data:image/jpeg;base64,${message.image_base64}`}
            alt="uploaded"
            className="mb-2 max-h-48 rounded-lg object-contain"
          />
        )}
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="prose prose-sm max-w-none dark:prose-invert prose-p:my-1 prose-ul:my-1 prose-li:my-0">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                img: () => null,
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
        {!isUser && message.annotated_image && (
          <div className="mt-3 flex justify-center rounded-lg border border-border/60 bg-background p-2">
            <img
              src={`data:image/png;base64,${message.annotated_image}`}
              alt="image result"
              className="h-auto max-h-80 w-auto max-w-full rounded-md object-contain"
            />
          </div>
        )}
      </div>
    </div>
  );
}
