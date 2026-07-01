# Chatbot Example

Full-stack example that connects a Vite React chat UI to an Axum server using
this crate. The browser uses `@ai-sdk/react`; the Rust server talks to the
model through provider-neutral SDK types and streams the AI SDK UI-message SSE
protocol back to the browser.

## Pieces

- [server](server/README.md): concise Axum adapter that uses
  `compose_text_request(...)` and `stream_text_messages(...)`.
- [server-explicit](server-explicit/README.md): same behavior with request
  conversion and `TextRequest` construction shown explicitly.
- [web](web/README.md): Vite React app using `useChat()` and a `/api/chat`
  proxy.

## Run It

Create a server environment file:

```sh
cd examples/chatbot/server
cp .env.example .env
```

Fill in:

```sh
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-5.4-nano
PORT=3001
```

Start the default server:

```sh
cd examples/chatbot
just server
```

Start the web app in another terminal:

```sh
cd examples/chatbot
just web
```

Vite usually serves the app at `http://127.0.0.1:5173` and proxies `/api` to
`http://127.0.0.1:3001`.

To run the explicit server instead:

```sh
cd examples/chatbot
just server-explicit
```

## What To Try

Ask for weather or time, for example:

```text
What is the weather in Paris?
What time is it in America/Chicago?
```

Both tools are deterministic demo tools implemented in the Rust server. The
browser never executes tools directly.

## Checks

```sh
cd examples/chatbot
just server-check
just server-explicit-check
just web-build
```
