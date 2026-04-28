# Detailed implementation prompt for evolving `rust_ai_sdk` into a tool-aware SDK suitable for a Rust chat server behind a Vite frontend

You are working inside the repository `christopher-caldwell/rust_ai_sdk`.

## Why this work exists

The end goal is not simply to add abstract “tool support” for its own sake. The real product goal is to enable a deployment shape where:

- the browser UI is a fully client-side Vite application served statically, for example from Nginx,
- a Rust server owns the AI provider credentials and all server-side orchestration,
- the Rust server streams chat output to the frontend in the same practical role that a Next.js route handler often serves in Vercel AI SDK examples,
- the Rust server can perform controlled tool execution when the model requests it,
- the Rust server can do end-of-stream actions, such as persisting the final assistant message and metadata to a database,
- the frontend can still have good ergonomics, and ideally keep compatibility with a client-side AI SDK style workflow rather than requiring a heavy custom browser protocol from scratch.

This means the repository must evolve from “provider calls that can generate text and stream text” into a provider-neutral SDK that can express richer chat semantics. Right now the SDK is already a good start, but it is still too text-centric for that architecture.

The purpose of this work is to give the outer Rust chat server a clean, durable SDK boundary so the server does not need to parse provider-specific tool call wire formats, and so the server does not need to fake tool support by scraping structured meaning back out of plain text.

The SDK should remain focused on provider-neutral modeling, provider adapters, runtime behavior, and examples. It should not become the application server itself. Database writes, HTTP route shapes, auth, persistence policy, and browser transport protocol details belong to the outer application that will consume this SDK.

## Current repository state you must respect

The current SDK shape is intentionally minimal.

The inspected core indicates the following:

- `TextRequest` currently contains `messages`, `max_output_tokens`, and `temperature`.
- `Message` currently uses a `Role` plus plain string content.
- `StreamEvent` currently supports only `TextDelta` and `Finished`.
- `LanguageModel` currently exposes `generate`, `stream`, `model_id`, and `provider_name`.
- `TextResult` currently returns only final text plus finish reason, usage, and response metadata.
- The OpenAI provider already has working generate and stream behavior, including accumulation of metadata and usage across chunked streaming.
- There are runnable examples for OpenAI generate and stream.
- Anthropic-related provider work exists or is beginning, but the immediate design should not depend on Anthropic being feature-complete first.

This is important because the next work should extend the existing architecture rather than replace it with a totally different design.

## The architectural problem to solve

A tool-capable chat runtime cannot be represented cleanly with the current primitives because:

- a message cannot represent a tool call or a tool result,
- a request cannot describe which tools are available to the model,
- a stream cannot emit structured tool call progress,
- a final result cannot return a structured assistant message if the assistant produced non-text content,
- the runtime does not yet provide a reusable model → tool request → tool result → model continuation loop.

These are not optional polish items. They are the core missing capabilities required for a Rust server to act as the chat orchestrator.

## Non-goals

Do not build the outer application server in this repository.

Do not implement database persistence in this repository.

Do not implement browser-specific stream protocol adapters in this repository.

Do not build resumable streams, chat history storage, auth, or deployment code here.

Do not overfit the core types to one provider’s naming or payload shape.

Do not force MCP into the first version of this design. The immediate need is controlled tool execution with a clean SDK boundary. MCP can be layered later by an application or by future SDK extension, but it should not distort the first-class core abstractions.

## High-level design target

The SDK should support a Rust server being able to do this cleanly:

1. Receive chat messages from a frontend.
2. Convert them into SDK request types.
3. Provide an explicit set of allowed tools to the model.
4. Start a streamed model turn.
5. Receive structured events from the SDK that distinguish normal text streaming from tool call streaming.
6. Detect when the model has completed one or more tool calls.
7. Execute those tools outside the SDK using the application’s own tool registry and policy.
8. Append tool results back into the message history.
9. Continue the model loop using the SDK again.
10. End with a final assistant message and associated metadata that can be persisted by the application.

This means the SDK must support both low-level provider access and one or more higher-level orchestration helpers.

## Design principles to preserve

Preserve provider-neutral abstractions.

Preserve the clean split between `core`, `providers`, and `runtime`.

Preserve the ability to use the SDK for simple text-only generation and text-only streaming.

Preserve a low-friction API for examples and ergonomics. The API should not become so abstract that basic usage becomes painful.

Keep the core types understandable. Avoid clever type systems that make extension difficult.

Prefer additive change where practical, but if a breaking change is clearly the right design choice for message modeling, make it deliberately and consistently.

## The concrete additions required

### 1. Replace plain string message content with structured message parts

The current `Message` model is too narrow because it assumes all content is plain text. That is not sufficient for a tool-aware assistant.

Introduce a structured content model. The exact naming can vary, but the shape should support first-class message parts such as:

- text,
- tool call,
- tool result.

A reasonable target shape would be conceptually similar to:

- `Message { role, parts: Vec<MessagePart> }`
- `MessagePart::Text(...)`
- `MessagePart::ToolCall(...)`
- `MessagePart::ToolResult(...)`

Keep the design extensible enough that future parts could exist later if needed.

The migration should also preserve convenience for simple text-only prompts, such as helper constructors that create a user text message without a lot of ceremony.

### 2. Add provider-neutral tool definitions to the request model

The model request needs a way to express the tool surface available for a turn.

Extend the request type so that it can carry:

- a list of tool definitions,
- optional tool choice policy.

Tool definitions should be provider-neutral and should describe:

- tool name,
- optional description,
- input schema.

Use a JSON-schema-compatible representation for the input schema boundary rather than an application-specific Rust callback type. The SDK should model the contract of the tool, not the application’s execution function.

This allows the provider adapters to translate that abstract tool definition into provider-specific request payloads.

### 3. Expand stream events to represent structured tool call progress

The current stream event model only supports text deltas and a final completion event. That is insufficient for a chat orchestrator.

Introduce structured stream events that let the caller distinguish between normal text generation and tool activity.

The design should support at least the equivalent of:

- text delta,
- tool call started,
- tool call input delta,
- tool call finished or ready,
- final finished event.

If one provider emits arguments incrementally and another emits them only at the end, the core API should still make sense for both.

It is acceptable for the stream event model to focus on the common useful contract rather than exposing every provider-specific streaming quirk.

### 4. Add a richer final result type

A text-only `TextResult` is no longer sufficient for tool-aware chat. The outer application needs a structured final assistant message or a structured intermediate outcome indicating that tools are required.

Introduce a richer final result, for example a `ChatResult` or similarly named type, that contains:

- the final assistant message in structured message form,
- finish reason,
- usage,
- response metadata.

If you keep `TextResult` for backwards compatibility or ergonomics, it should be a thin specialization or convenience path rather than the only final result type.

### 5. Add a reusable higher-level chat loop runtime

The SDK currently has the provider-level request/stream primitives. That is good and should remain. But the repository also needs a reusable runtime helper that makes tool-aware orchestration straightforward for the outer Rust server.

Add a runtime function or small runtime module that can run one model turn and produce an explicit structured outcome, such as:

- completed normally with a final assistant message,
- requires tool execution with one or more tool calls returned in structured form.

This helper should not execute tools itself unless there is a deliberately designed callback-based API that still keeps application ownership of policy. The application should remain in control of whether a tool is allowed, how it is executed, what timeouts apply, and how results are logged.

The runtime helper exists to avoid every consuming application having to rebuild the model-response parsing and tool-request detection loop from scratch.

### 6. Implement the provider translation for OpenAI first

OpenAI is already the most mature path in the repository and should be the first implementation of the richer abstractions.

Update the OpenAI request translation so it can include tools.

Update the OpenAI response translation so it can produce structured assistant message parts.

Update the OpenAI stream translation so it can emit tool-aware stream events.

Preserve the quality of the current stream implementation, especially around usage accumulation, finish handling, metadata carry-forward, and tests.

### 7. Only then extend Anthropic in the same model

After the core types and the OpenAI mapping feel correct, extend Anthropic support to the same abstractions.

Do not design the core around imagined future Anthropic behavior before the OpenAI path is working cleanly. But also do not hardcode OpenAI-shaped assumptions into the core if you can avoid it.

## Required examples

The repository must include corresponding runnable examples. This is not optional.

Create examples that prove both output behavior and API ergonomics.

At a minimum, add:

### Example A: text-only generate
A minimal generate example using the new message model with no tools, proving that the ergonomic path for simple usage is still good.

### Example B: text-only stream
A minimal stream example using the new message model with no tools, proving backwards-compatible streaming ergonomics remain acceptable.

### Example C: tool-aware non-streaming turn
An example where the model is given a simple tool definition, requests the tool, the example executes the tool in application code, feeds the result back into the model, and prints the final assistant answer.

The tool should be extremely simple and deterministic, such as weather lookup from a hardcoded map, arithmetic, or a small fake business lookup. The point is SDK ergonomics, not the external integration.

### Example D: tool-aware streaming turn
An example where the model streams output, emits a tool request, the example handles the tool outside the SDK, re-enters the model loop, and streams the final answer.

This is the most important example because it most closely mirrors the actual target architecture.

### Example E: event inspection example
An example dedicated to printing or logging the structured stream events so the ergonomics of the event API are visible.

These examples must be runnable and documented. If the examples require API keys for specific providers, make that explicit. If some examples can run purely against mocked or fake model outputs, that is acceptable too, especially for demonstrating runtime loop ergonomics.

## Required tests

Add tests that cover the new core behavior.

At a minimum:

- message modeling tests,
- request translation tests for tool definitions,
- stream event tests for tool-related streaming,
- final result structure tests,
- runtime helper tests for a normal completion path,
- runtime helper tests for a tool-request path,
- runtime helper tests for multi-step model → tool → model flow,
- provider-specific tests for OpenAI mapping.

Where provider streaming is hard to test against live endpoints, use mocked streaming payloads as is already done in the repository.

## Acceptance criteria

This work is complete only when all of the following are true:

1. The SDK can represent tool calls and tool results as first-class structured content.
2. The request model can describe available tools in a provider-neutral way.
3. The stream API can emit structured tool-related events.
4. The result model can return a structured final assistant message.
5. The runtime provides a reusable way to detect tool requests and continue the loop.
6. OpenAI supports the new abstractions.
7. The repository includes runnable examples that demonstrate simple text usage and tool-aware usage.
8. The examples make the API ergonomics clear enough for a consumer to understand how to build the outer Rust chat server.
9. Existing simple text-only usage remains possible without excessive ceremony.
10. The code remains organized cleanly by core abstractions, provider adapters, runtime helpers, and examples.

## Implementation guidance

Do this in deliberate stages rather than trying to land everything in one pass.

A good sequence is:

1. evolve the core message types,
2. evolve the request model,
3. evolve the stream event model,
4. add richer final result types,
5. build runtime helpers,
6. wire OpenAI translation,
7. add examples,
8. add tests,
9. only then extend the second provider path.

When in doubt, bias toward explicit and boring over magical and implicit. This SDK is intended to be a clean foundation for a Rust application server, not a clever macro-heavy framework.

## Important ergonomics requirement

Do not design the API only for theoretical cleanliness. The examples should read like something a real application author would want to use.

That means:

- helper constructors are welcome,
- common flows should not require deeply nested boilerplate,
- provider-neutral core types should still be straightforward to instantiate,
- tool execution should remain application-owned without making the runtime impossible to use.

A reader should come away from the examples able to imagine writing the outer Rust chat server immediately afterward.

## Final deliverables expected from this work

You should produce:

- updated core types,
- updated request types,
- updated stream event types,
- updated result types,
- runtime helpers for tool-aware chat loops,
- OpenAI provider implementation for the new abstractions,
- tests,
- runnable examples,
- any minimal supporting docs needed so the examples are understandable.

Do not stop at type design alone. The examples are part of the required deliverable because they validate both correctness and ergonomics.
