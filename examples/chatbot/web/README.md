# Chatbot Web App

Vite React frontend for the chatbot example. It uses `@ai-sdk/react` with
`DefaultChatTransport` and sends chat requests to `/api/chat`.

## Run It

Start one of the Rust servers first:

```sh
cd examples/chatbot
just server
```

Then start the web app:

```sh
cd examples/chatbot
just web
```

Or from this directory:

```sh
npm install
npm run dev
```

The Vite dev server runs on `http://127.0.0.1:5173` by default. Its proxy sends
`/api` traffic to `http://127.0.0.1:3001`.

## How It Works

- `src/main.tsx` calls `useChat(...)` with `DefaultChatTransport`.
- User messages are posted to `/api/chat`.
- The Rust server streams AI SDK UI-message chunks back over SSE.
- Text parts render as chat bubbles.
- Tool parts render in expandable detail sections so you can inspect tool
  inputs and outputs.

The UI includes shortcuts for the demo server tools:

```text
Weather in Paris
Current time
```

## Build

```sh
npm run build
```
