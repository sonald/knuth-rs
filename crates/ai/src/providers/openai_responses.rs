//! OpenAI Responses provider. Partial 1:1 port of
//! `packages/ai/src/providers/openai-responses.ts` (~312 lines) plus the shared SSE→event
//! pipeline that lives in `openai-responses-shared.ts` on the TS side.
//!
//! Implemented:
//! - Provider trait + registration scaffold
//! - `OpenAIResponsesOptions` typed knobs
//! - HTTP request shape (POST /v1/responses, streaming JSON SSE)
//! - SSE → AssistantMessageEvent mapping for the happy path
//! - `prompt_cache_key` + `prompt_cache_retention` ("24h" when retention is long)
//! - `reasoning.effort` + `reasoning.summary` + `include: ["reasoning.encrypted_content"]`
//! - service_tier knob (cost multiplier TODO)
//!
//! TODO:
//! - Cross-provider transform_messages
//! - GitHub Copilot dynamic headers
//! - Tool-call id `call|item` normalization across provider handoffs
//! - service_tier pricing multiplier
//! - `output_text.done`/`function_call_arguments.done` final-state reconciliation

use async_trait::async_trait;
use serde::Serialize;
use serde_json::{Map, Value, json};
use std::collections::HashMap;

use crate::api_registry::ApiProvider;
use crate::types::*;
use crate::utils::abort::{self as abort_utils, AbortErrorOrReqwest, AbortableNext};
use crate::utils::event_stream::{AssistantMessageEventSender, AssistantMessageEventStream};
use crate::utils::sse::SseStream;

const OPENAI_BASE_URL: &str = "https://api.openai.com";

#[derive(Clone, Debug)]
pub(crate) struct Compat {
    pub send_session_id_header: bool,
    pub supports_long_cache_retention: bool,
    /// Replay assistant thinking content as `{"type":"reasoning"}` input items.
    /// Needed by servers that do byte-exact KV prefix matching on the rendered
    /// history (e.g. ds4 / DeepSeek V4 local): omitting the reasoning changes
    /// the rendered prefix and invalidates their cache checkpoints.
    pub replay_reasoning_content: bool,
}

pub(crate) fn resolve_compat(model: &Model) -> Compat {
    let read_bool = |key: &str, default: bool| -> bool {
        model
            .compat
            .as_ref()
            .and_then(|c| c.get(key))
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    };
    Compat {
        send_session_id_header: read_bool("sendSessionIdHeader", true),
        supports_long_cache_retention: read_bool("supportsLongCacheRetention", true),
        replay_reasoning_content: read_bool("requiresReasoningContentOnAssistantMessages", false),
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct OpenAIResponsesOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_summary: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

#[derive(Default)]
pub struct OpenAIResponsesProvider {}

#[async_trait]
impl ApiProvider for OpenAIResponsesProvider {
    fn api(&self) -> &str {
        KnownApi::OpenAIResponses.as_str()
    }

    fn stream(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&StreamOptions>,
    ) -> AssistantMessageEventStream {
        let (stream, sender) = AssistantMessageEventStream::new();
        let model = model.clone();
        let context = context.clone();
        let options = options.cloned().unwrap_or_default();
        tokio::spawn(async move {
            run(model, context, options, sender).await;
        });
        stream
    }

    fn stream_simple(
        &self,
        model: &Model,
        context: &Context,
        options: Option<&SimpleStreamOptions>,
    ) -> AssistantMessageEventStream {
        let translated = options
            .map(|o| {
                let mut base = o.base.clone();
                if let Some(level) = o.reasoning {
                    if let Some(mapped) = map_reasoning_effort(level) {
                        base.provider_extras
                            .insert("reasoning_effort".to_string(), json!(mapped));
                    }
                }
                base
            })
            .unwrap_or_default();
        self.stream(model, context, Some(&translated))
    }
}

fn map_reasoning_effort(level: ThinkingLevel) -> Option<&'static str> {
    Some(match level {
        ThinkingLevel::Minimal => "minimal",
        ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High => "high",
        // OpenAI Responses API does not natively accept "xhigh" — providers map via
        // `thinkingLevelMap` to whatever the concrete model accepts.
        ThinkingLevel::Xhigh => "xhigh",
    })
}

// ────────────────────────────────────────────────────────────────────────────────────────────
// HTTP + SSE pipeline
// ────────────────────────────────────────────────────────────────────────────────────────────

async fn run(
    model: Model,
    context: Context,
    options: StreamOptions,
    mut sender: AssistantMessageEventSender,
) {
    let api_key = match resolve_openai_compatible_api_key(&model, &options) {
        Some(k) => k,
        None => {
            push_error(
                &mut sender,
                &model,
                missing_openai_compatible_api_key_message(&model),
            );
            return;
        }
    };

    let compat = resolve_compat(&model);
    let body = match build_request_body(&model, &context, &options, &compat) {
        Ok(b) => b,
        Err(e) => {
            push_error(&mut sender, &model, format!("build request body: {e}"));
            return;
        }
    };

    let client = match crate::utils::node_http_proxy::build_client(options.timeout_ms) {
        Ok(c) => c,
        Err(e) => {
            push_error(&mut sender, &model, format!("http client: {e}"));
            return;
        }
    };

    let base = if model.base_url.is_empty() {
        Ok(OPENAI_BASE_URL.to_string())
    } else {
        #[cfg(feature = "cloudflare")]
        {
            if crate::providers::cloudflare::is_cloudflare_provider(&model.provider.0) {
                crate::providers::cloudflare::resolve_cloudflare_base_url(&model)
            } else {
                Ok(model.base_url.clone())
            }
        }
        #[cfg(not(feature = "cloudflare"))]
        {
            Ok::<_, String>(model.base_url.clone())
        }
    };
    let base = match base {
        Ok(base) => base,
        Err(error) => {
            push_error(&mut sender, &model, error);
            return;
        }
    };
    let url = crate::utils::openai_compat_url::build_responses_url(&base);
    let mut req = client
        .post(&url)
        .bearer_auth(api_key)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream");

    if let Some(sid) = &options.session_id {
        if compat.send_session_id_header {
            req = req.header("session_id", sid.as_str());
        }
        req = req.header("x-client-request-id", sid.as_str());
    }

    let custom_headers = match crate::utils::headers::merged_model_and_option_headers(
        model.headers.as_ref(),
        options.headers.as_ref(),
    ) {
        Ok(headers) => headers,
        Err(error) => {
            push_error(
                &mut sender,
                &model,
                format!("custom request headers: {error}"),
            );
            return;
        }
    };
    req = req.headers(custom_headers);

    let req = req.json(&body);
    let resp = match crate::utils::retry::send_with_retry(&options, req).await {
        Ok(r) => r,
        Err(e) => {
            if e.is_aborted() {
                abort_utils::push_aborted(&mut sender, &model);
            } else {
                push_error(&mut sender, &model, format!("http error: {e}"));
            }
            return;
        }
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let txt = match abort_utils::response_text_or_abort(resp, options.abort.as_ref()).await {
            Ok(txt) => txt,
            Err(AbortErrorOrReqwest::Aborted) => {
                abort_utils::push_aborted(&mut sender, &model);
                return;
            }
            Err(AbortErrorOrReqwest::Reqwest(_)) => String::new(),
        };
        push_error(&mut sender, &model, format!("HTTP {status}: {txt}"));
        return;
    }

    consume_responses_sse(resp, &model, &mut sender, options.abort.as_ref()).await;
}

/// Shared Responses-API SSE consumer. Reused by the Azure provider, which differs only in URL
/// shape and auth header. Pushes `Start`, drains the SSE stream into events, and emits the
/// terminal `Done`/`Error`.
pub(crate) async fn consume_responses_sse(
    resp: reqwest::Response,
    model: &Model,
    sender: &mut AssistantMessageEventSender,
    abort_token: Option<&tokio_util::sync::CancellationToken>,
) {
    let mut partial = empty_partial(model);
    let mut state = StreamState::default();
    sender.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });

    let mut sse = SseStream::new(resp.bytes_stream());
    loop {
        if sender.is_closed() {
            return;
        }
        let item = match abort_utils::next_or_abort(&mut sse, abort_token).await {
            AbortableNext::Item(item) => item,
            AbortableNext::Eof => break,
            AbortableNext::Aborted => {
                abort_utils::push_aborted(sender, model);
                return;
            }
        };
        match item {
            Err(e) => {
                push_error(sender, model, format!("sse: {e}"));
                return;
            }
            Ok(ev) => {
                if !handle_event(&ev, &mut partial, &mut state, sender) {
                    return;
                }
            }
        }
    }

    partial.stop_reason = StopReason::Error;
    partial.error_message = Some("openai-responses stream ended before terminal event".into());
    sender.push(AssistantMessageEvent::Error {
        reason: ErrorReason::Error,
        error: partial,
    });
}

// ────────────────────────────────────────────────────────────────────────────────────────────
// SSE → event translation
// ────────────────────────────────────────────────────────────────────────────────────────────

#[derive(Default)]
struct StreamState {
    item_to_content: HashMap<String, usize>,
    output_to_content: HashMap<usize, usize>,
    tool_arg_buffers: HashMap<usize, String>,
}

fn handle_event(
    ev: &crate::utils::sse::SseEvent,
    partial: &mut AssistantMessage,
    state: &mut StreamState,
    sender: &mut AssistantMessageEventSender,
) -> bool {
    let Ok(payload): Result<Value, _> = serde_json::from_str(&ev.data) else {
        return true;
    };
    let kind = ev
        .event
        .as_deref()
        .or_else(|| payload.get("type").and_then(|v| v.as_str()))
        .unwrap_or("");
    match kind {
        "response.created" | "response.in_progress" => {
            if let Some(id) = payload.pointer("/response/id").and_then(|v| v.as_str()) {
                partial.response_id = Some(id.to_string());
            }
        }
        "response.output_item.added" => on_output_item_added(&payload, partial, state, sender),
        "response.output_item.done" => on_output_item_done(&payload, partial, state, sender),
        "response.output_text.delta" => on_text_delta(&payload, partial, sender),
        "response.output_text.done" => on_text_done(&payload, partial, sender),
        "response.reasoning_summary_text.delta" => on_thinking_delta(&payload, partial, sender),
        "response.reasoning_summary_text.done" => on_thinking_done(&payload, partial),
        "response.function_call_arguments.delta" => {
            on_tool_args_delta(&payload, partial, state, sender)
        }
        "response.function_call_arguments.done" => on_tool_args_done(&payload, partial, state),
        "response.completed" => {
            if let Some(u) = payload.pointer("/response/usage") {
                update_usage(&mut partial.usage, u);
            }
            let stop = openai_stop_reason(&payload);
            partial.stop_reason = stop;
            let reason = match stop {
                StopReason::ToolUse => DoneReason::ToolUse,
                StopReason::Length => DoneReason::Length,
                _ => DoneReason::Stop,
            };
            sender.push(AssistantMessageEvent::Done {
                reason,
                message: partial.clone(),
            });
            return false;
        }
        "response.incomplete" => {
            if let Some(u) = payload.pointer("/response/usage") {
                update_usage(&mut partial.usage, u);
            }
            partial.stop_reason = StopReason::Length;
            sender.push(AssistantMessageEvent::Done {
                reason: DoneReason::Length,
                message: partial.clone(),
            });
            return false;
        }
        "response.failed" | "response.error" | "error" => {
            let msg = payload
                .pointer("/error/message")
                .or_else(|| payload.pointer("/response/error/message"))
                .and_then(|v| v.as_str())
                .unwrap_or("openai-responses error")
                .to_string();
            partial.stop_reason = StopReason::Error;
            partial.error_message = Some(msg);
            sender.push(AssistantMessageEvent::Error {
                reason: ErrorReason::Error,
                error: partial.clone(),
            });
            return false;
        }
        _ => {}
    }
    true
}

fn on_output_item_added(
    payload: &Value,
    partial: &mut AssistantMessage,
    state: &mut StreamState,
    sender: &mut AssistantMessageEventSender,
) {
    let item = &payload["item"];
    let output_index = payload["output_index"].as_u64().map(|n| n as usize);
    match item["type"].as_str().unwrap_or("") {
        "reasoning" => {
            let idx = partial.content.len();
            partial
                .content
                .push(ContentBlock::Thinking(ThinkingContent::default()));
            remember_item_index(item, output_index, idx, state);
            sender.push(AssistantMessageEvent::ThinkingStart {
                content_index: idx,
                partial: partial.clone(),
            });
        }
        "function_call" => {
            let call_id = item["call_id"].as_str().unwrap_or("").to_string();
            let id = responses_tool_call_id(&call_id, item["id"].as_str());
            let name = item["name"].as_str().unwrap_or("").to_string();
            let idx = partial.content.len();
            partial.content.push(ContentBlock::ToolCall(ToolCall {
                id,
                name,
                arguments: Map::new(),
                thought_signature: None,
            }));
            remember_item_index(item, output_index, idx, state);
            if let Some(args) = item["arguments"].as_str().filter(|s| !s.is_empty()) {
                state.tool_arg_buffers.insert(idx, args.to_string());
            }
            sender.push(AssistantMessageEvent::ToolCallStart {
                content_index: idx,
                partial: partial.clone(),
            });
        }
        _ => {}
    }
}

fn on_output_item_done(
    payload: &Value,
    partial: &mut AssistantMessage,
    state: &mut StreamState,
    sender: &mut AssistantMessageEventSender,
) {
    let item = &payload["item"];
    match item["type"].as_str().unwrap_or("") {
        "reasoning" => {
            let Some(idx) = content_index_for_event(payload, state)
                .or_else(|| find_last_block(partial, |b| matches!(b, ContentBlock::Thinking(_))))
            else {
                return;
            };
            let summary = item["summary"]
                .as_array()
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n")
                })
                .unwrap_or_default();
            let content = item["content"]
                .as_array()
                .map(|parts| {
                    parts
                        .iter()
                        .filter_map(|p| p["text"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n\n")
                })
                .unwrap_or_default();
            let mut final_text = None;
            if let Some(ContentBlock::Thinking(tc)) = partial.content.get_mut(idx) {
                if !summary.is_empty() || !content.is_empty() {
                    tc.thinking = if summary.is_empty() { content } else { summary };
                }
                tc.thinking_signature = Some(item.to_string());
                final_text = Some(tc.thinking.clone());
            }
            if let Some(content) = final_text {
                sender.push(AssistantMessageEvent::ThinkingEnd {
                    content_index: idx,
                    content,
                    partial: partial.clone(),
                });
            }
        }
        "function_call" => {
            let Some(idx) = content_index_for_event(payload, state)
                .or_else(|| find_last_block(partial, |b| matches!(b, ContentBlock::ToolCall(_))))
            else {
                return;
            };
            let raw = item["arguments"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .or_else(|| state.tool_arg_buffers.remove(&idx))
                .unwrap_or_default();
            if let Ok(Value::Object(map)) = crate::utils::json_parse::parse_partial_json(&raw) {
                if let Some(ContentBlock::ToolCall(tc)) = partial.content.get_mut(idx) {
                    tc.arguments = map;
                }
            }
            if let Some(ContentBlock::ToolCall(tc)) = partial.content.get(idx).cloned() {
                sender.push(AssistantMessageEvent::ToolCallEnd {
                    content_index: idx,
                    tool_call: tc,
                    partial: partial.clone(),
                });
            }
        }
        _ => {}
    }
}

fn remember_item_index(
    item: &Value,
    output_index: Option<usize>,
    content_index: usize,
    state: &mut StreamState,
) {
    if let Some(id) = item["id"].as_str().filter(|s| !s.is_empty()) {
        state.item_to_content.insert(id.to_string(), content_index);
    }
    if let Some(index) = output_index {
        state.output_to_content.insert(index, content_index);
    }
}

fn content_index_for_event(payload: &Value, state: &StreamState) -> Option<usize> {
    payload["item_id"]
        .as_str()
        .and_then(|id| state.item_to_content.get(id).copied())
        .or_else(|| {
            payload["output_index"]
                .as_u64()
                .and_then(|n| state.output_to_content.get(&(n as usize)).copied())
        })
        .or_else(|| {
            payload["item"]["id"]
                .as_str()
                .and_then(|id| state.item_to_content.get(id).copied())
        })
}

fn find_last_block(
    partial: &AssistantMessage,
    pred: impl Fn(&ContentBlock) -> bool,
) -> Option<usize> {
    partial.content.iter().rposition(pred)
}

fn responses_tool_call_id(call_id: &str, item_id: Option<&str>) -> String {
    match item_id.filter(|id| !id.is_empty()) {
        Some(item_id) => format!("{call_id}|{item_id}"),
        None => call_id.to_string(),
    }
}

fn on_text_delta(
    payload: &Value,
    partial: &mut AssistantMessage,
    sender: &mut AssistantMessageEventSender,
) {
    let delta = payload["delta"].as_str().unwrap_or("").to_string();
    let idx = match partial.content.last() {
        Some(ContentBlock::Text(_)) => partial.content.len() - 1,
        _ => {
            let i = partial.content.len();
            partial.content.push(ContentBlock::text(""));
            sender.push(AssistantMessageEvent::TextStart {
                content_index: i,
                partial: partial.clone(),
            });
            i
        }
    };
    if let Some(ContentBlock::Text(tc)) = partial.content.get_mut(idx) {
        tc.text.push_str(&delta);
    }
    sender.push(AssistantMessageEvent::TextDelta {
        content_index: idx,
        delta,
        partial: partial.clone(),
    });
}

fn on_text_done(
    payload: &Value,
    partial: &mut AssistantMessage,
    sender: &mut AssistantMessageEventSender,
) {
    if let Some(ContentBlock::Text(tc)) = partial.content.last().cloned() {
        let idx = partial.content.len() - 1;
        let text = payload["text"]
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or(tc.text);
        sender.push(AssistantMessageEvent::TextEnd {
            content_index: idx,
            content: text,
            partial: partial.clone(),
        });
    }
}

fn on_thinking_delta(
    payload: &Value,
    partial: &mut AssistantMessage,
    sender: &mut AssistantMessageEventSender,
) {
    let delta = payload["delta"].as_str().unwrap_or("").to_string();
    let idx = match partial
        .content
        .iter()
        .rposition(|b| matches!(b, ContentBlock::Thinking(_)))
    {
        Some(i) => i,
        None => {
            let i = partial.content.len();
            partial
                .content
                .push(ContentBlock::Thinking(ThinkingContent::default()));
            sender.push(AssistantMessageEvent::ThinkingStart {
                content_index: i,
                partial: partial.clone(),
            });
            i
        }
    };
    if let Some(ContentBlock::Thinking(tc)) = partial.content.get_mut(idx) {
        tc.thinking.push_str(&delta);
    }
    sender.push(AssistantMessageEvent::ThinkingDelta {
        content_index: idx,
        delta,
        partial: partial.clone(),
    });
}

fn on_thinking_done(payload: &Value, partial: &mut AssistantMessage) {
    if let Some(idx) = partial
        .content
        .iter()
        .rposition(|b| matches!(b, ContentBlock::Thinking(_)))
    {
        let content = payload["text"]
            .as_str()
            .map(|s| s.to_string())
            .unwrap_or_default();
        if let Some(ContentBlock::Thinking(tc)) = partial.content.get_mut(idx) {
            tc.thinking = content;
        }
    }
}

fn on_tool_args_delta(
    payload: &Value,
    partial: &mut AssistantMessage,
    state: &mut StreamState,
    sender: &mut AssistantMessageEventSender,
) {
    let delta = payload["delta"].as_str().unwrap_or("").to_string();
    if let Some(idx) = content_index_for_event(payload, state)
        .or_else(|| find_last_block(partial, |b| matches!(b, ContentBlock::ToolCall(_))))
    {
        state
            .tool_arg_buffers
            .entry(idx)
            .or_default()
            .push_str(&delta);
        sender.push(AssistantMessageEvent::ToolCallDelta {
            content_index: idx,
            delta,
            partial: partial.clone(),
        });
    }
}

fn on_tool_args_done(payload: &Value, partial: &mut AssistantMessage, state: &mut StreamState) {
    let Some(idx) = content_index_for_event(payload, state)
        .or_else(|| find_last_block(partial, |b| matches!(b, ContentBlock::ToolCall(_))))
    else {
        return;
    };
    let raw = payload["arguments"].as_str().unwrap_or("");
    state.tool_arg_buffers.insert(idx, raw.to_string());
    if let Ok(Value::Object(map)) = crate::utils::json_parse::parse_partial_json(raw) {
        if let Some(ContentBlock::ToolCall(tc)) = partial.content.get_mut(idx) {
            tc.arguments = map;
        }
    }
}

fn openai_stop_reason(payload: &Value) -> StopReason {
    if let Some(items) = payload
        .pointer("/response/output")
        .and_then(|v| v.as_array())
    {
        if items.iter().any(|i| i["type"] == "function_call") {
            return StopReason::ToolUse;
        }
    }
    match payload
        .pointer("/response/status")
        .and_then(|v| v.as_str())
        .unwrap_or("")
    {
        "incomplete" => StopReason::Length,
        _ => StopReason::Stop,
    }
}

fn update_usage(usage: &mut Usage, val: &Value) {
    if let Some(n) = val.get("input_tokens").and_then(|v| v.as_u64()) {
        usage.input += n;
    }
    if let Some(n) = val.get("output_tokens").and_then(|v| v.as_u64()) {
        usage.output += n;
    }
    if let Some(n) = val
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(|v| v.as_u64())
    {
        usage.cache_read += n;
    }
    // Non-standard but reported by local inference servers (ds4): tokens newly
    // written into the prompt cache this request.
    if let Some(n) = val
        .pointer("/input_tokens_details/cache_write_tokens")
        .and_then(|v| v.as_u64())
    {
        usage.cache_write += n;
    }
    usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
}

// ────────────────────────────────────────────────────────────────────────────────────────────
// Request body construction
// ────────────────────────────────────────────────────────────────────────────────────────────

pub(crate) fn build_request_body(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    compat: &Compat,
) -> Result<Value, String> {
    let messages = convert_messages(
        &context.messages,
        context.system_prompt.as_deref(),
        compat.replay_reasoning_content,
    );
    let mut body = json!({
        "model": model.id,
        "input": messages,
        "stream": true,
        "store": false,
    });

    let retention = options.cache_retention.unwrap_or(CacheRetention::Short);
    if !matches!(retention, CacheRetention::None) {
        if let Some(sid) = &options.session_id {
            body["prompt_cache_key"] = json!(sid);
        }
        if matches!(retention, CacheRetention::Long) && compat.supports_long_cache_retention {
            body["prompt_cache_retention"] = json!("24h");
        }
    }

    if let Some(max) = options.max_tokens {
        body["max_output_tokens"] = json!(max);
    }
    if let Some(t) = options.temperature {
        body["temperature"] = json!(t);
    }
    if let Some(tier) = options.provider_extras.get("service_tier") {
        body["service_tier"] = tier.clone();
    }

    if let Some(tools) = &context.tools {
        body["tools"] = json!(serialize_tools(tools));
    }

    if model.reasoning {
        if let Some(effort) = options
            .provider_extras
            .get("reasoning_effort")
            .and_then(|v| v.as_str())
        {
            let mapped = model
                .thinking_level_map
                .as_ref()
                .and_then(|m| {
                    let lvl = match effort {
                        "minimal" => ModelThinkingLevel::Minimal,
                        "low" => ModelThinkingLevel::Low,
                        "medium" => ModelThinkingLevel::Medium,
                        "high" => ModelThinkingLevel::High,
                        "xhigh" => ModelThinkingLevel::Xhigh,
                        _ => ModelThinkingLevel::Medium,
                    };
                    m.get(&lvl).cloned().flatten()
                })
                .unwrap_or_else(|| effort.to_string());
            body["reasoning"] = json!({
                "effort": mapped,
                "summary": options
                    .provider_extras
                    .get("reasoning_summary")
                    .and_then(|v| v.as_str())
                    .unwrap_or("auto"),
            });
            body["include"] = json!(["reasoning.encrypted_content"]);
        }
    }

    Ok(body)
}

pub(crate) fn serialize_tools(tools: &[Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "type": "function",
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })
        })
        .collect()
}

pub(crate) fn convert_messages(
    msgs: &[Message],
    system_prompt: Option<&str>,
    replay_reasoning: bool,
) -> Vec<Value> {
    let mut out = Vec::with_capacity(msgs.len() + 1);
    if let Some(sys) = system_prompt {
        out.push(json!({
            "role": "system",
            "content": [{ "type": "input_text", "text": sys }],
        }));
    }
    for m in msgs {
        match m {
            Message::User(u) => {
                let content = user_content_to_value(&u.content);
                out.push(json!({ "role": "user", "content": content }));
            }
            Message::Assistant(a) => {
                let mut content = Vec::new();
                let mut function_calls = Vec::new();
                for b in &a.content {
                    match b {
                        ContentBlock::Text(t) => content.push(json!({
                            "type": "output_text",
                            "text": t.text,
                        })),
                        // Servers that consume reasoning items merge them into
                        // the *following* assistant message, so this must be
                        // emitted before the message / function_call items.
                        ContentBlock::Thinking(th) if replay_reasoning => {
                            if let Some(raw) = th
                                .thinking_signature
                                .as_ref()
                                .and_then(|sig| serde_json::from_str::<Value>(sig).ok())
                                .filter(|v| v["type"] == "reasoning")
                            {
                                out.push(raw);
                            } else if !th.thinking.is_empty() {
                                out.push(json!({
                                    "type": "reasoning",
                                    "summary": [{ "type": "summary_text", "text": th.thinking }],
                                }));
                            }
                        }
                        ContentBlock::Thinking(_) => {}
                        ContentBlock::ToolCall(tc) => {
                            let (call_id, item_id) = split_responses_tool_call_id(&tc.id);
                            let mut call = json!({
                                "type": "function_call",
                                "call_id": call_id,
                                "name": tc.name,
                                "arguments": serde_json::to_string(&tc.arguments).unwrap_or_default(),
                            });
                            if let Some(item_id) = item_id {
                                call["id"] = json!(item_id);
                            }
                            function_calls.push(call);
                        }
                        ContentBlock::Image(_) => {}
                    }
                }
                if !content.is_empty() {
                    out.push(json!({ "role": "assistant", "content": content }));
                }
                out.extend(function_calls);
            }
            Message::ToolResult(tr) => {
                let text_parts: Vec<String> = tr
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        UserContentBlock::Text(t) => Some(t.text.clone()),
                        _ => None,
                    })
                    .collect();
                out.push(json!({
                    "type": "function_call_output",
                    "call_id": split_responses_tool_call_id(&tr.tool_call_id).0,
                    "output": text_parts.join("\n"),
                }));
            }
        }
    }
    out
}

fn split_responses_tool_call_id(id: &str) -> (&str, Option<&str>) {
    match id.split_once('|') {
        Some((call_id, item_id)) => (call_id, Some(item_id)),
        None => (id, None),
    }
}

fn user_content_to_value(content: &UserContent) -> Value {
    match content {
        UserContent::Text(s) => json!([{ "type": "input_text", "text": s }]),
        UserContent::Blocks(blocks) => {
            let arr: Vec<Value> = blocks
                .iter()
                .map(|b| match b {
                    UserContentBlock::Text(t) => json!({ "type": "input_text", "text": t.text }),
                    UserContentBlock::Image(i) => json!({
                        "type": "input_image",
                        "image_url": format!("data:{};base64,{}", i.mime_type, i.data),
                    }),
                })
                .collect();
            Value::Array(arr)
        }
    }
}

fn resolve_openai_compatible_api_key(model: &Model, options: &StreamOptions) -> Option<String> {
    options
        .api_key
        .clone()
        .or_else(|| crate::env_api_keys::get_env_api_key(&model.provider.0))
        .or_else(|| {
            if model.provider.0 == "openai" {
                crate::env_api_keys::get_env_api_key("openai")
            } else {
                None
            }
        })
}

fn missing_openai_compatible_api_key_message(model: &Model) -> String {
    let vars = crate::env_api_keys::env_var_names(&model.provider.0);
    if vars.is_empty() {
        format!(
            "no API key for provider: {}; pass options.api_key or configure a provider-specific credential",
            model.provider.0
        )
    } else {
        format!(
            "no API key for provider: {}; set {} or pass options.api_key",
            model.provider.0,
            vars.join(" or ")
        )
    }
}

pub(crate) fn empty_partial(model: &Model) -> AssistantMessage {
    AssistantMessage {
        role: AssistantRole::Assistant,
        content: vec![],
        api: model.api.clone(),
        provider: model.provider.clone(),
        model: model.id.clone(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: chrono::Utc::now().timestamp_millis(),
    }
}

pub(crate) fn push_error(sender: &mut AssistantMessageEventSender, model: &Model, msg: String) {
    let mut p = empty_partial(model);
    p.stop_reason = StopReason::Error;
    p.error_message = Some(msg);
    sender.push(AssistantMessageEvent::Error {
        reason: ErrorReason::Error,
        error: p,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn mk_model() -> Model {
        Model {
            id: "gpt-5".into(),
            name: "GPT-5".into(),
            api: Api::known(KnownApi::OpenAIResponses),
            provider: Provider::from("openai"),
            base_url: "https://api.openai.com".into(),
            reasoning: true,
            thinking_level_map: None,
            input: vec![],
            cost: ModelCost::default(),
            context_window: 200_000,
            max_tokens: 16_384,
            headers: None,
            compat: None,
        }
    }

    #[test]
    fn body_includes_system_prompt() {
        let m = mk_model();
        let ctx = Context {
            system_prompt: Some("be helpful".into()),
            messages: vec![Message::User(UserMessage {
                role: UserRole::User,
                content: UserContent::Text("hi".into()),
                timestamp: 0,
            })],
            tools: None,
        };
        let body = build_request_body(&m, &ctx, &Default::default(), &resolve_compat(&m)).unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(input[0]["role"], "system");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
    }

    #[test]
    fn long_retention_sets_24h_and_cache_key() {
        let m = mk_model();
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                role: UserRole::User,
                content: UserContent::Text("hi".into()),
                timestamp: 0,
            })],
            tools: None,
        };
        let opts = StreamOptions {
            cache_retention: Some(CacheRetention::Long),
            session_id: Some("sess-1".into()),
            ..Default::default()
        };
        let body = build_request_body(&m, &ctx, &opts, &resolve_compat(&m)).unwrap();
        assert_eq!(body["prompt_cache_key"], "sess-1");
        assert_eq!(body["prompt_cache_retention"], "24h");
    }

    #[test]
    fn reasoning_block_emitted_when_effort_set() {
        let m = mk_model();
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message::User(UserMessage {
                role: UserRole::User,
                content: UserContent::Text("hi".into()),
                timestamp: 0,
            })],
            tools: None,
        };
        let mut opts = StreamOptions::default();
        opts.provider_extras
            .insert("reasoning_effort".into(), json!("high"));
        let body = build_request_body(&m, &ctx, &opts, &resolve_compat(&m)).unwrap();
        assert_eq!(body["reasoning"]["effort"], "high");
        assert_eq!(body["reasoning"]["summary"], "auto");
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
    }

    fn assistant_msg_with_thinking() -> Message {
        Message::Assistant(AssistantMessage {
            role: AssistantRole::Assistant,
            content: vec![
                ContentBlock::Thinking(ThinkingContent {
                    thinking: "let me check".into(),
                    thinking_signature: None,
                    redacted: false,
                }),
                ContentBlock::Text(TextContent {
                    text: "done".into(),
                    text_signature: None,
                }),
            ],
            api: Api::known(KnownApi::OpenAIResponses),
            provider: Provider::from("openai"),
            model: "gpt-5".into(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        })
    }

    fn sse_event(kind: &str, payload: Value) -> crate::utils::sse::SseEvent {
        crate::utils::sse::SseEvent {
            event: Some(kind.into()),
            data: payload.to_string(),
        }
    }

    fn assistant_message(content: Vec<ContentBlock>) -> Message {
        Message::Assistant(AssistantMessage {
            role: AssistantRole::Assistant,
            content,
            api: Api::known(KnownApi::OpenAIResponses),
            provider: Provider::from("openai"),
            model: "gpt-5".into(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason: StopReason::Stop,
            error_message: None,
            timestamp: 0,
        })
    }

    #[test]
    fn thinking_replayed_as_reasoning_item_when_compat_requires() {
        // ds4 (DeepSeek V4 local server) does byte-exact KV prefix matching on
        // the rendered history; it accepts `{"type":"reasoning"}` input items
        // and merges them into the following assistant message. Replaying the
        // thinking text keeps the rendered prefix identical to what the server
        // sampled, so disk KV checkpoints stay valid after eviction/restart.
        let mut m = mk_model();
        m.compat = Some(json!({ "requiresReasoningContentOnAssistantMessages": true }));
        let ctx = Context {
            system_prompt: None,
            messages: vec![assistant_msg_with_thinking()],
            tools: None,
        };
        let body = build_request_body(&m, &ctx, &Default::default(), &resolve_compat(&m)).unwrap();
        let input = body["input"].as_array().unwrap();
        let reasoning_idx = input
            .iter()
            .position(|v| v["type"] == "reasoning")
            .expect("reasoning input item");
        assert_eq!(
            input[reasoning_idx]["summary"],
            json!([{ "type": "summary_text", "text": "let me check" }])
        );
        let assistant_idx = input
            .iter()
            .position(|v| v["role"] == "assistant")
            .expect("assistant message item");
        assert!(
            reasoning_idx < assistant_idx,
            "reasoning item must precede the assistant message it belongs to"
        );
    }

    #[test]
    fn reasoning_item_done_is_captured_and_replayed() {
        let m = mk_model();
        let (_stream, mut sender) = AssistantMessageEventStream::new();
        let mut partial = empty_partial(&m);
        let mut state = StreamState::default();

        handle_event(
            &sse_event(
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": { "type": "reasoning", "id": "rs_1", "summary": [] },
                }),
            ),
            &mut partial,
            &mut state,
            &mut sender,
        );
        handle_event(
            &sse_event(
                "response.output_item.done",
                json!({
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [{ "type": "summary_text", "text": "checked" }],
                        "encrypted_content": "enc",
                    },
                }),
            ),
            &mut partial,
            &mut state,
            &mut sender,
        );

        let Some(ContentBlock::Thinking(thinking)) = partial.content.first() else {
            panic!("expected thinking block");
        };
        assert_eq!(thinking.thinking, "checked");
        assert_eq!(
            serde_json::from_str::<Value>(thinking.thinking_signature.as_deref().unwrap()).unwrap()
                ["encrypted_content"],
            "enc"
        );

        let input = convert_messages(
            &[assistant_message(vec![ContentBlock::Thinking(
                thinking.clone(),
            )])],
            None,
            true,
        );
        assert_eq!(input[0]["type"], "reasoning");
        assert_eq!(input[0]["id"], "rs_1");
        assert_eq!(input[0]["encrypted_content"], "enc");
    }

    #[test]
    fn empty_summary_reasoning_signature_is_replayed() {
        let raw = json!({
            "type": "reasoning",
            "id": "rs_empty",
            "summary": [],
            "encrypted_content": "enc",
        });
        let input = convert_messages(
            &[assistant_message(vec![ContentBlock::Thinking(
                ThinkingContent {
                    thinking: String::new(),
                    thinking_signature: Some(raw.to_string()),
                    redacted: false,
                },
            )])],
            None,
            true,
        );

        assert_eq!(input, vec![raw]);
    }

    #[test]
    fn thinking_dropped_without_compat_flag() {
        let m = mk_model();
        let ctx = Context {
            system_prompt: None,
            messages: vec![assistant_msg_with_thinking()],
            tools: None,
        };
        let body = build_request_body(&m, &ctx, &Default::default(), &resolve_compat(&m)).unwrap();
        let input = body["input"].as_array().unwrap();
        assert!(input.iter().all(|v| v["type"] != "reasoning"));
    }

    #[test]
    fn usage_reads_cached_and_cache_write_tokens() {
        // ds4 reports both cached_tokens (KV prefix hits) and cache_write_tokens
        // (new suffix written into the live KV) under input_tokens_details.
        let mut usage = Usage::default();
        update_usage(
            &mut usage,
            &json!({
                "input_tokens": 100,
                "output_tokens": 10,
                "input_tokens_details": {
                    "cached_tokens": 80,
                    "cache_write_tokens": 20,
                },
            }),
        );
        assert_eq!(usage.cache_read, 80);
        assert_eq!(usage.cache_write, 20);
    }

    #[test]
    fn tool_call_serializes_as_function_call() {
        let m = mk_model();
        let mut args = Map::new();
        args.insert("x".into(), json!(1));
        let ctx = Context {
            system_prompt: None,
            messages: vec![Message::Assistant(AssistantMessage {
                role: AssistantRole::Assistant,
                content: vec![ContentBlock::ToolCall(ToolCall {
                    id: "call_123".into(),
                    name: "calc".into(),
                    arguments: args,
                    thought_signature: None,
                })],
                api: Api::known(KnownApi::OpenAIResponses),
                provider: Provider::from("openai"),
                model: "gpt-5".into(),
                response_model: None,
                response_id: None,
                diagnostics: None,
                usage: Usage::default(),
                stop_reason: StopReason::ToolUse,
                error_message: None,
                timestamp: 0,
            })],
            tools: None,
        };
        let body = build_request_body(&m, &ctx, &Default::default(), &resolve_compat(&m)).unwrap();
        let input = body["input"].as_array().unwrap();
        let fc = input
            .iter()
            .find(|v| v["type"] == "function_call")
            .expect("function_call output item");
        assert_eq!(fc["call_id"], "call_123");
        assert_eq!(fc["name"], "calc");
        assert!(fc["arguments"].as_str().unwrap().contains("\"x\":1"));
    }

    #[test]
    fn tool_call_id_preserves_response_item_id_for_replay() {
        let m = mk_model();
        let (_stream, mut sender) = AssistantMessageEventStream::new();
        let mut partial = empty_partial(&m);
        let mut state = StreamState::default();

        handle_event(
            &sse_event(
                "response.output_item.added",
                json!({
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "calc",
                        "arguments": "",
                    },
                }),
            ),
            &mut partial,
            &mut state,
            &mut sender,
        );

        let Some(ContentBlock::ToolCall(tool_call)) = partial.content.first() else {
            panic!("expected tool call block");
        };
        assert_eq!(tool_call.id, "call_1|fc_1");

        let input = convert_messages(
            &[assistant_message(vec![ContentBlock::ToolCall(
                tool_call.clone(),
            )])],
            None,
            false,
        );
        assert_eq!(input[0]["type"], "function_call");
        assert_eq!(input[0]["call_id"], "call_1");
        assert_eq!(input[0]["id"], "fc_1");
    }

    #[test]
    fn tool_argument_deltas_route_by_item_id() {
        let m = mk_model();
        let (_stream, mut sender) = AssistantMessageEventStream::new();
        let mut partial = empty_partial(&m);
        let mut state = StreamState::default();

        for (output_index, item_id, call_id, name) in [
            (0, "fc_1", "call_1", "first"),
            (1, "fc_2", "call_2", "second"),
        ] {
            handle_event(
                &sse_event(
                    "response.output_item.added",
                    json!({
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "type": "function_call",
                            "id": item_id,
                            "call_id": call_id,
                            "name": name,
                            "arguments": "",
                        },
                    }),
                ),
                &mut partial,
                &mut state,
                &mut sender,
            );
        }

        handle_event(
            &sse_event(
                "response.function_call_arguments.delta",
                json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "delta": "{\"x\":",
                }),
            ),
            &mut partial,
            &mut state,
            &mut sender,
        );
        handle_event(
            &sse_event(
                "response.function_call_arguments.delta",
                json!({
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_2",
                    "delta": "{\"y\":",
                }),
            ),
            &mut partial,
            &mut state,
            &mut sender,
        );
        handle_event(
            &sse_event(
                "response.function_call_arguments.done",
                json!({
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_1",
                    "arguments": "{\"x\":1}",
                }),
            ),
            &mut partial,
            &mut state,
            &mut sender,
        );
        handle_event(
            &sse_event(
                "response.function_call_arguments.done",
                json!({
                    "type": "response.function_call_arguments.done",
                    "item_id": "fc_2",
                    "arguments": "{\"y\":2}",
                }),
            ),
            &mut partial,
            &mut state,
            &mut sender,
        );

        let args: Vec<_> = partial
            .content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::ToolCall(tc) => Some(tc.arguments.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(args[0]["x"], 1);
        assert_eq!(args[1]["y"], 2);
    }

    async fn serve_once(body: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = socket.read(&mut buf).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.flush().await.unwrap();
        });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn eof_before_terminal_event_emits_error() {
        let body = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n";
        let url = serve_once(body).await;
        let resp = reqwest::Client::new().get(url).send().await.unwrap();
        let m = mk_model();
        let (mut stream, mut sender) = AssistantMessageEventStream::new();

        consume_responses_sse(resp, &m, &mut sender, None).await;
        drop(sender);

        let mut saw_error = false;
        while let Some(event) = stream.next().await {
            if let AssistantMessageEvent::Error { error, .. } = event {
                saw_error = true;
                assert_eq!(error.stop_reason, StopReason::Error);
                assert!(error.error_message.unwrap().contains("terminal event"));
            }
        }
        assert!(saw_error);
    }
}
