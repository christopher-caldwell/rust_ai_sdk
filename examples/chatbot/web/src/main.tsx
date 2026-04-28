import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { useState } from "react";
import { createRoot } from "react-dom/client";
import "./style.css";

type ToolLikePart = {
    type: string;
    state?: string;
    toolName?: string;
    toolCallId?: string;
    input?: unknown;
    output?: unknown;
    errorText?: string;
};

function App() {
    const [input, setInput] = useState("");
    const { messages, sendMessage, status, error, stop } = useChat({
        transport: new DefaultChatTransport({
            api: "/api/chat",
        }),
    });

    const isBusy = status === "submitted" || status === "streaming";

    return (
        <main className="shell">
            <aside className="sidebar">
                <div className="brand">
                    <span className="mark">AI</span>
                    <div>
                        <strong>Rust SDK Chat</strong>
                        <small>Axum + Vercel AI SDK</small>
                    </div>
                </div>
                <div className="hint">
                    <span>Try:</span>
                    <button
                        type="button"
                        onClick={() =>
                            setInput("What is the weather in Paris?")
                        }
                    >
                        Weather in Paris
                    </button>
                    <button
                        type="button"
                        onClick={() =>
                            setInput("What time is it in America/Chicago?")
                        }
                    >
                        Current time
                    </button>
                </div>
            </aside>

            <section className="chat-panel" aria-label="Chat">
                <header className="topbar">
                    <div>
                        <h1>Chatbot Example</h1>
                        <p>
                            Streaming through the local Axum server and this
                            Rust SDK.
                        </p>
                    </div>
                    <span className={`status status-${status}`}>{status}</span>
                </header>

                <div className="messages">
                    {messages.length === 0 ? (
                        <EmptyState />
                    ) : (
                        messages.map((message) => (
                            <MessageBubble key={message.id} message={message} />
                        ))
                    )}
                    {error ? (
                        <div className="error">
                            Something went wrong: {error.message}
                        </div>
                    ) : null}
                </div>

                <form
                    className="composer"
                    onSubmit={(event) => {
                        event.preventDefault();
                        const text = input.trim();
                        if (!text || isBusy) {
                            return;
                        }
                        sendMessage({ text });
                        setInput("");
                    }}
                >
                    <textarea
                        value={input}
                        onChange={(event) => setInput(event.target.value)}
                        onKeyDown={(event) => {
                            if (event.key === "Enter" && !event.shiftKey) {
                                event.preventDefault();
                                event.currentTarget.form?.requestSubmit();
                            }
                        }}
                        placeholder="Message the Rust SDK chatbot..."
                        rows={1}
                    />
                    {isBusy ? (
                        <button
                            type="button"
                            className="secondary"
                            onClick={() => stop()}
                        >
                            Stop
                        </button>
                    ) : (
                        <button type="submit" disabled={!input.trim()}>
                            Send
                        </button>
                    )}
                </form>
            </section>
        </main>
    );
}

function EmptyState() {
    return (
        <div className="empty">
            <div className="orb" />
            <h2>Local AI SDK bridge</h2>
            <p>
                Ask for weather or time to see server-side tool calls stream
                into the UI.
            </p>
        </div>
    );
}

function MessageBubble({ message }: { message: UIMessage }) {
    return (
        <article className={`message message-${message.role}`}>
            <div className="avatar">{message.role === "user" ? "U" : "AI"}</div>
            <div className="bubble">
                {message.parts.map((part, index) => (
                    <MessagePart key={`${message.id}-${index}`} part={part} />
                ))}
            </div>
        </article>
    );
}

function MessagePart({ part }: { part: UIMessage["parts"][number] }) {
    if (part.type === "step-start") {
        return null;
    }

    if (part.type === "text") {
        return <p className="text-part">{part.text}</p>;
    }

    if (part.type === "dynamic-tool" || part.type.startsWith("tool-")) {
        const tool = part as ToolLikePart;
        const name = tool.toolName ?? part.type.replace(/^tool-/, "");
        return (
            <details
                className="tool-part"
                open={tool.state !== "output-available"}
            >
                <summary>
                    <span>{name}</span>
                    <code>{tool.state ?? "tool"}</code>
                </summary>
                {tool.input !== undefined ? (
                    <pre>{JSON.stringify(tool.input, null, 2)}</pre>
                ) : null}
                {tool.output !== undefined ? (
                    <pre>{JSON.stringify(tool.output, null, 2)}</pre>
                ) : null}
                {tool.errorText ? (
                    <p className="tool-error">{tool.errorText}</p>
                ) : null}
            </details>
        );
    }

    return <pre className="raw-part">{JSON.stringify(part, null, 2)}</pre>;
}

createRoot(document.getElementById("app")!).render(<App />);
