# pie-ai (Rust)

Rust port of [`@earendil-works/pie-ai`](https://github.com/earendil-works/pi) — a unified streaming
LLM client. Every supported model (Anthropic, OpenAI Responses / Completions / Codex, Google
Gemini, Vertex, Bedrock, Mistral, Azure, Cloudflare, OpenAI-compatible endpoints) is reached
through a single `stream(model, context, options)` call that yields a normalized
`AssistantMessageEventStream`.

## Status

A feature-gated source-level port mirroring the TypeScript layout. The table records implemented
behavior and the explicit remaining limitations.

| Layer | Status |
|-------|--------|
| Core types (`types.rs`) | implemented |
| Event stream (`utils/event_stream.rs`) | implemented + tested |
| Registry + entry points (`api_registry.rs` / `stream.rs`) | implemented |
| Anthropic | working — SSE, text/thinking/tool_use, cache_control, budget thinking, compat |
| OpenAI Responses | working — text/reasoning/function_call, cache key + 24h retention |
| OpenAI Completions | working — catch-all; parallel tool_calls, reasoning_content, `[DONE]` |
| OpenAI Codex | working (HTTP/SSE path) — instructions body, codex headers; WebSocket transport TODO |
| Azure OpenAI Responses | working — reuses Responses consumer; deployment-name + api-key header |
| Google Gemini | working — text/thought parts, functionCall, thinking budget |
| Google Vertex | working — access token or service-account ADC via `GOOGLE_APPLICATION_CREDENTIALS`; reuses Gemini consumer |
| Amazon Bedrock | working — Converse Stream with Bearer token or SigV4 environment credentials |
| Mistral | working — chat-completions-shaped; x-affinity, alphanumeric tool ids |
| Cloudflare | working through OpenAI/Anthropic-compatible providers; base-url placeholders resolve from `CLOUDFLARE_ACCOUNT_ID` / `CLOUDFLARE_GATEWAY_ID` |
| Faux | scriptable — queue `AssistantMessage`s, replayed as event sequences |
| Models registry | generated catalog loaded from `models.generated.json` |
| Cost accounting | working for streamed usage across built-in providers; `Usage.cost` is populated from `Model.cost` |
| Cross-provider `transform_messages` | implemented helper (image downgrade, thinking, id normalization, synthetic results); provider-local conversion remains the default and global handoff is caller-owned |
| Context-overflow detection (`overflow.rs`) | implemented — all provider error patterns |
| Anthropic OAuth (PKCE) | implemented — authorize URL, local listener, exchange, refresh |
| OpenAI Codex / Copilot OAuth | not implemented |
| Images | unsupported — `images(...)` returns an explicit error |

Mock-SSE end-to-end tests (`tests/anthropic_sse_e2e.rs`) prove the shared HTTP→SSE→event pipeline
against a local server (no API key). `scripts/regen_models.sh` regenerates the model catalog from
the TS source of truth.

### Tests

```
cargo test -p ai --features all-providers
```

## Layout

```
src/
  lib.rs                         barrel
  types.rs                       core types
  stream.rs                      stream() / streamSimple() / complete()
  api_registry.rs                registerApiProvider / getApiProvider
  models.rs / models_generated.rs
  images.rs / image_models.rs      image generation model metadata; generation unsupported
  env_api_keys.rs
  oauth.rs
  session_resources.rs
  providers/
    anthropic.rs
    openai_responses.rs
    openai_completions.rs
    ...                          one file per wire protocol or vendor
  utils/
    event_stream.rs
    sanitize_unicode.rs
    json_parse.rs
    ...
    oauth/
      anthropic.rs
      github_copilot.rs
      ...
```

Each Rust file has the same name as its TS counterpart (snake_case instead of kebab-case).
`scripts/regen_models.sh` reads the TypeScript model catalog from the `TS_PATH` environment
variable.

## Build

```bash
cargo check
cargo build --features all-providers
cargo run --example anthropic_hello --features anthropic   # needs ANTHROPIC_API_KEY
```
