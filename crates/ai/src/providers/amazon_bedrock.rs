//! Amazon Bedrock provider (`bedrock-converse-stream`). Partial 1:1 port of
//! `packages/ai/src/providers/amazon-bedrock.ts` (~950 LOC).
//!
//! Implemented:
//! - Bearer-token auth (AWS Bedrock API keys — `AWS_BEARER_TOKEN_BEDROCK` or `options.api_key`)
//!   with AWS environment-credential SigV4 fallback
//! - Converse Stream request body (messages / system / inferenceConfig / toolConfig)
//! - AWS binary eventstream frame decoding → AssistantMessageEvent
//!   (messageStart / contentBlockStart / contentBlockDelta / contentBlockStop / messageStop /
//!   metadata for usage)
//! - text / reasoningContent / toolUse content blocks
//!
//! TODO:
//! - prompt caching (`cachePoint` blocks)
//! - thinking display modes
//! - image content blocks in tool results

use async_trait::async_trait;
use chrono::Utc;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HOST, HeaderName, HeaderValue};
use serde_json::{Map, Value, json};

use crate::api_registry::ApiProvider;
use crate::models::calculate_usage_cost;
use crate::types::*;
use crate::utils::abort::{self as abort_utils, AbortErrorOrReqwest, AbortableNext};
use crate::utils::aws_eventstream::AwsEventStream;
use crate::utils::event_stream::{AssistantMessageEventSender, AssistantMessageEventStream};

#[derive(Default)]
pub struct AmazonBedrockProvider {}

#[async_trait]
impl ApiProvider for AmazonBedrockProvider {
    fn api(&self) -> &str {
        KnownApi::BedrockConverseStream.as_str()
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
        match translate_simple_options(model, options) {
            Ok(options) => self.stream(model, context, Some(&options)),
            Err(error) => {
                let (stream, mut sender) = AssistantMessageEventStream::new();
                push_error(&mut sender, model, error);
                stream
            }
        }
    }
}

fn translate_simple_options(
    model: &Model,
    options: Option<&SimpleStreamOptions>,
) -> Result<StreamOptions, String> {
    let Some(options) = options else {
        return Ok(StreamOptions::default());
    };
    let mut translated = options.base.clone();
    let Some(level) = options.reasoning else {
        return Ok(translated);
    };

    let additional_fields = if supports_adaptive_thinking(model) {
        json!({
            "thinking": { "type": "adaptive" },
            "output_config": { "effort": reasoning_effort(model, level) },
        })
    } else if supports_fixed_budget_thinking(model) {
        const MIN_THINKING_BUDGET: u32 = 1024;
        let configured_budget = options
            .thinking_budgets
            .as_ref()
            .and_then(|budgets| match level {
                ThinkingLevel::Minimal => budgets.minimal,
                ThinkingLevel::Low => budgets.low,
                ThinkingLevel::Medium => budgets.medium,
                ThinkingLevel::High | ThinkingLevel::Xhigh => budgets.high,
            });
        let default_budget = match level {
            ThinkingLevel::Minimal => 1024,
            ThinkingLevel::Low => 2048,
            ThinkingLevel::Medium => 8192,
            ThinkingLevel::High | ThinkingLevel::Xhigh => 16384,
        };
        let max_tokens = translated.max_tokens.unwrap_or(model.max_tokens);
        if max_tokens <= MIN_THINKING_BUDGET {
            return Err(format!(
                "Bedrock model {} requires maxTokens greater than the minimum Claude thinking budget",
                model.id
            ));
        }
        let budget = configured_budget
            .unwrap_or(default_budget)
            .max(MIN_THINKING_BUDGET)
            .min(max_tokens - 1);
        translated.max_tokens = Some(max_tokens);
        json!({
            "thinking": {
                "type": "enabled",
                "budget_tokens": budget,
            }
        })
    } else if is_nova_2_lite(model) {
        let effort = match level {
            ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
            ThinkingLevel::Medium => "medium",
            ThinkingLevel::High | ThinkingLevel::Xhigh => "high",
        };
        if effort == "high" {
            translated.max_tokens = None;
            translated.temperature = None;
        }
        json!({
            "reasoningConfig": {
                "type": "enabled",
                "maxReasoningEffort": effort,
            }
        })
    } else {
        return Err(format!(
            "Bedrock simple reasoning has no documented configurable protocol for model {}",
            model.id
        ));
    };
    translated.provider_extras.insert(
        "bedrock_additional_model_request_fields".into(),
        additional_fields,
    );
    Ok(translated)
}

fn model_id_matches(id: &str, canonical_id: &str) -> bool {
    id == canonical_id
        || ["au.", "eu.", "global.", "jp.", "us."]
            .iter()
            .any(|prefix| id.strip_prefix(prefix) == Some(canonical_id))
}

fn supports_adaptive_thinking(model: &Model) -> bool {
    [
        "anthropic.claude-opus-4-6-v1",
        "anthropic.claude-opus-4-7",
        "anthropic.claude-sonnet-4-6",
    ]
    .iter()
    .any(|id| model_id_matches(&model.id, id))
}

fn supports_fixed_budget_thinking(model: &Model) -> bool {
    [
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-opus-4-1-20250805-v1:0",
        "anthropic.claude-opus-4-5-20251101-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
    ]
    .iter()
    .any(|id| model_id_matches(&model.id, id))
}

fn reasoning_effort(model: &Model, level: ThinkingLevel) -> String {
    let mapped_level = match level {
        ThinkingLevel::Minimal => ModelThinkingLevel::Minimal,
        ThinkingLevel::Low => ModelThinkingLevel::Low,
        ThinkingLevel::Medium => ModelThinkingLevel::Medium,
        ThinkingLevel::High => ModelThinkingLevel::High,
        ThinkingLevel::Xhigh => ModelThinkingLevel::Xhigh,
    };
    if let Some(mapped) = model
        .thinking_level_map
        .as_ref()
        .and_then(|levels| levels.get(&mapped_level))
        .and_then(|level| level.as_ref())
    {
        return mapped.clone();
    }
    match level {
        ThinkingLevel::Minimal | ThinkingLevel::Low => "low",
        ThinkingLevel::Medium => "medium",
        ThinkingLevel::High | ThinkingLevel::Xhigh => "high",
    }
    .into()
}

fn is_nova_2_lite(model: &Model) -> bool {
    model_id_matches(&model.id, "amazon.nova-2-lite-v1:0")
}

async fn run(
    model: Model,
    context: Context,
    options: StreamOptions,
    mut sender: AssistantMessageEventSender,
) {
    let bearer_token = options
        .api_key
        .clone()
        .filter(|token| !token.is_empty())
        .or_else(|| {
            std::env::var("AWS_BEARER_TOKEN_BEDROCK")
                .ok()
                .filter(|token| !token.is_empty())
        });
    let sigv4_creds = if bearer_token.is_none() {
        match crate::bedrock_provider::BedrockCreds::from_env() {
            Some(creds) => Some(creds),
            None => {
                push_error(
                    &mut sender,
                    &model,
                    "Bedrock auth missing: set AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY".into(),
                );
                return;
            }
        }
    } else {
        None
    };

    let body = build_request_body(&context, &options);
    let payload = match serde_json::to_vec(&body) {
        Ok(payload) => payload,
        Err(error) => {
            push_error(
                &mut sender,
                &model,
                format!("Bedrock request body serialization failed: {error}"),
            );
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
    let base = model.base_url.trim_end_matches('/');
    let url = format!("{base}/model/{}/converse-stream", model.id);
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
    let mut request_headers = custom_headers;
    if !request_headers.contains_key(CONTENT_TYPE) {
        request_headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    }
    if !request_headers.contains_key(ACCEPT) {
        request_headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/vnd.amazon.eventstream"),
        );
    }
    if let Some(token) = bearer_token {
        let authorization = match HeaderValue::from_str(&format!("Bearer {token}")) {
            Ok(authorization) => authorization,
            Err(error) => {
                push_error(
                    &mut sender,
                    &model,
                    format!("Bedrock bearer token is not a valid header value: {error}"),
                );
                return;
            }
        };
        request_headers.insert(AUTHORIZATION, authorization);
    } else if let Some(creds) = sigv4_creds {
        let parsed_url = match url::Url::parse(&url) {
            Ok(parsed_url) => parsed_url,
            Err(error) => {
                push_error(
                    &mut sender,
                    &model,
                    format!("Bedrock request URL is invalid: {error}"),
                );
                return;
            }
        };
        let signing_region = match resolve_signing_region(&model.id, &parsed_url, &creds.region) {
            Ok(region) => region,
            Err(error) => {
                push_error(&mut sender, &model, format!("Bedrock SigV4: {error}"));
                return;
            }
        };
        request_headers.remove(HOST);
        request_headers.remove(AUTHORIZATION);
        let signing_headers = match request_headers
            .iter()
            .filter(|(name, _)| sigv4_signs_header(name))
            .map(|(name, value)| {
                value
                    .to_str()
                    .map(|value| (name.as_str(), value))
                    .map_err(|error| format!("header {name:?} cannot be signed: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(headers) => headers,
            Err(error) => {
                push_error(&mut sender, &model, format!("Bedrock SigV4: {error}"));
                return;
            }
        };
        let amz_date = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        let signed = crate::sigv4::sign(&crate::sigv4::SigningRequest {
            method: "POST",
            url: &parsed_url,
            headers: &signing_headers,
            payload: &payload,
            region: &signing_region,
            service: "bedrock",
            access_key: &creds.access_key,
            secret_key: &creds.secret_key,
            session_token: creds.session_token.as_deref(),
            amz_date: &amz_date,
        });
        drop(signing_headers);
        let authorization = match HeaderValue::from_str(&signed.authorization) {
            Ok(authorization) => authorization,
            Err(error) => {
                push_error(
                    &mut sender,
                    &model,
                    format!("Bedrock SigV4 authorization header is invalid: {error}"),
                );
                return;
            }
        };
        request_headers.insert(AUTHORIZATION, authorization);
        for (name, value) in signed.headers {
            let name = match HeaderName::from_bytes(name.as_bytes()) {
                Ok(name) => name,
                Err(error) => {
                    push_error(
                        &mut sender,
                        &model,
                        format!("Bedrock SigV4 generated an invalid header name: {error}"),
                    );
                    return;
                }
            };
            let value = match HeaderValue::from_str(&value) {
                Ok(value) => value,
                Err(error) => {
                    push_error(
                        &mut sender,
                        &model,
                        format!("Bedrock SigV4 generated an invalid header value: {error}"),
                    );
                    return;
                }
            };
            request_headers.insert(name, value);
        }
    }

    let req = client.post(&url).headers(request_headers).body(payload);
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
        push_error(
            &mut sender,
            &model,
            format!("Bedrock API error ({status}): {txt}"),
        );
        return;
    }

    consume(resp, &model, sender, options.abort.as_ref()).await;
}

fn sigv4_signs_header(name: &HeaderName) -> bool {
    !matches!(
        name.as_str(),
        "accept"
            | "authorization"
            | "connection"
            | "content-length"
            | "host"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
            | "user-agent"
            | "x-amzn-trace-id"
    )
}

fn resolve_signing_region(
    model_id: &str,
    url: &url::Url,
    fallback: &str,
) -> Result<String, String> {
    let arn_region = bedrock_arn_region(model_id);
    let endpoint_region = standard_bedrock_endpoint_region(url);
    match (arn_region, endpoint_region) {
        (Some(arn), Some(endpoint)) if arn != endpoint => Err(format!(
            "Bedrock model ARN region {arn} conflicts with endpoint region {endpoint}"
        )),
        (Some(region), _) | (None, Some(region)) => Ok(region.to_string()),
        (None, None) => Ok(fallback.to_string()),
    }
}

fn bedrock_arn_region(model_id: &str) -> Option<&str> {
    let mut parts = model_id.split(':');
    if parts.next()? != "arn" {
        return None;
    }
    let partition = parts.next()?;
    if !partition.starts_with("aws") || parts.next()? != "bedrock" {
        return None;
    }
    let region = parts.next()?;
    (!region.is_empty()).then_some(region)
}

fn standard_bedrock_endpoint_region(url: &url::Url) -> Option<&str> {
    let host = url.host_str()?;
    let regional_host = host
        .strip_suffix(".amazonaws.com.cn")
        .or_else(|| host.strip_suffix(".amazonaws.com"))?;
    let region = regional_host
        .strip_prefix("bedrock-runtime.")
        .or_else(|| regional_host.strip_prefix("bedrock-runtime-fips."))?;
    (!region.is_empty()
        && region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'))
    .then_some(region)
}

async fn consume(
    resp: reqwest::Response,
    model: &Model,
    mut sender: AssistantMessageEventSender,
    abort_token: Option<&tokio_util::sync::CancellationToken>,
) {
    let mut partial = empty_partial(model);
    sender.push(AssistantMessageEvent::Start {
        partial: partial.clone(),
    });

    // Bedrock indexes content blocks; map contentBlockIndex → our content vec position.
    let mut tool_args: std::collections::HashMap<u64, String> = std::collections::HashMap::new();
    let mut index_map: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut saw_terminal = false;

    let mut es = AwsEventStream::new(resp.bytes_stream());
    loop {
        if sender.is_closed() {
            return;
        }
        let item = match abort_utils::next_or_abort(&mut es, abort_token).await {
            AbortableNext::Item(item) => item,
            AbortableNext::Eof => break,
            AbortableNext::Aborted => {
                abort_utils::push_aborted(&mut sender, model);
                return;
            }
        };
        let frame = match item {
            Ok(f) => f,
            Err(e) => {
                push_error(&mut sender, model, format!("eventstream: {e}"));
                return;
            }
        };
        if let Some(exc) = &frame.exception_type {
            let body = String::from_utf8_lossy(&frame.payload);
            partial.stop_reason = StopReason::Error;
            partial.error_message = Some(format!("{exc}: {body}"));
            calculate_usage_cost(model, &mut partial.usage);
            sender.push(AssistantMessageEvent::Error {
                reason: ErrorReason::Error,
                error: partial,
            });
            return;
        }
        let Ok(payload): Result<Value, _> = serde_json::from_slice(&frame.payload) else {
            continue;
        };
        match frame.event_type.as_deref() {
            Some("contentBlockStart") => {
                let bidx = payload["contentBlockIndex"].as_u64().unwrap_or(0);
                if let Some(tu) = payload.pointer("/start/toolUse") {
                    let id = tu["toolUseId"].as_str().unwrap_or("").to_string();
                    let name = tu["name"].as_str().unwrap_or("").to_string();
                    let pos = partial.content.len();
                    partial.content.push(ContentBlock::ToolCall(ToolCall {
                        id,
                        name,
                        arguments: Map::new(),
                        thought_signature: None,
                    }));
                    index_map.insert(bidx, pos);
                    tool_args.insert(bidx, String::new());
                    sender.push(AssistantMessageEvent::ToolCallStart {
                        content_index: pos,
                        partial: partial.clone(),
                    });
                }
            }
            Some("contentBlockDelta") => {
                let bidx = payload["contentBlockIndex"].as_u64().unwrap_or(0);
                let delta = &payload["delta"];
                if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                    let pos = *index_map.entry(bidx).or_insert_with(|| {
                        let p = partial.content.len();
                        partial.content.push(ContentBlock::text(""));
                        p
                    });
                    let is_first = matches!(partial.content.get(pos), Some(ContentBlock::Text(t)) if t.text.is_empty());
                    if is_first {
                        sender.push(AssistantMessageEvent::TextStart {
                            content_index: pos,
                            partial: partial.clone(),
                        });
                    }
                    if let Some(ContentBlock::Text(t)) = partial.content.get_mut(pos) {
                        t.text.push_str(text);
                    }
                    sender.push(AssistantMessageEvent::TextDelta {
                        content_index: pos,
                        delta: text.to_string(),
                        partial: partial.clone(),
                    });
                } else if let Some(rc) = delta.get("reasoningContent") {
                    if let Some(text) = rc.get("text").and_then(|v| v.as_str()) {
                        let pos = *index_map.entry(bidx).or_insert_with(|| {
                            let p = partial.content.len();
                            partial
                                .content
                                .push(ContentBlock::Thinking(ThinkingContent::default()));
                            p
                        });
                        let is_first = matches!(
                            partial.content.get(pos),
                            Some(ContentBlock::Thinking(t)) if t.thinking.is_empty()
                        );
                        if is_first {
                            sender.push(AssistantMessageEvent::ThinkingStart {
                                content_index: pos,
                                partial: partial.clone(),
                            });
                        }
                        if let Some(ContentBlock::Thinking(t)) = partial.content.get_mut(pos) {
                            t.thinking.push_str(text);
                        }
                        sender.push(AssistantMessageEvent::ThinkingDelta {
                            content_index: pos,
                            delta: text.to_string(),
                            partial: partial.clone(),
                        });
                    }
                } else if let Some(input) = delta.pointer("/toolUse/input").and_then(|v| v.as_str())
                {
                    if let Some(pos) = index_map.get(&bidx).copied() {
                        tool_args.entry(bidx).or_default().push_str(input);
                        sender.push(AssistantMessageEvent::ToolCallDelta {
                            content_index: pos,
                            delta: input.to_string(),
                            partial: partial.clone(),
                        });
                    }
                }
            }
            Some("contentBlockStop") => {
                let bidx = payload["contentBlockIndex"].as_u64().unwrap_or(0);
                if let Some(pos) = index_map.get(&bidx).copied() {
                    let snapshot = partial.content.get(pos).cloned();
                    match snapshot {
                        Some(ContentBlock::Text(t)) => {
                            sender.push(AssistantMessageEvent::TextEnd {
                                content_index: pos,
                                content: t.text,
                                partial: partial.clone(),
                            });
                        }
                        Some(ContentBlock::Thinking(t)) => {
                            sender.push(AssistantMessageEvent::ThinkingEnd {
                                content_index: pos,
                                content: t.thinking,
                                partial: partial.clone(),
                            });
                        }
                        Some(ContentBlock::ToolCall(_)) => {
                            let raw = tool_args.get(&bidx).map(|s| s.as_str()).unwrap_or("");
                            if let Ok(Value::Object(m)) =
                                crate::utils::json_parse::parse_partial_json(raw)
                            {
                                if let Some(ContentBlock::ToolCall(tc)) =
                                    partial.content.get_mut(pos)
                                {
                                    tc.arguments = m;
                                }
                            }
                            if let Some(ContentBlock::ToolCall(tc)) =
                                partial.content.get(pos).cloned()
                            {
                                sender.push(AssistantMessageEvent::ToolCallEnd {
                                    content_index: pos,
                                    tool_call: tc,
                                    partial: partial.clone(),
                                });
                            }
                        }
                        _ => {}
                    }
                }
            }
            Some("messageStop") => {
                saw_terminal = true;
                partial.stop_reason = map_stop_reason(payload["stopReason"].as_str().unwrap_or(""));
            }
            Some("metadata") => {
                if let Some(u) = payload.get("usage") {
                    update_usage(&mut partial.usage, u);
                }
            }
            _ => {}
        }
    }

    calculate_usage_cost(model, &mut partial.usage);

    if !saw_terminal {
        partial.stop_reason = StopReason::Error;
        partial.error_message = Some("bedrock stream ended before terminal event".into());
        sender.push(AssistantMessageEvent::Error {
            reason: ErrorReason::Error,
            error: partial,
        });
        return;
    }

    let reason = match partial.stop_reason {
        StopReason::ToolUse => DoneReason::ToolUse,
        StopReason::Length => DoneReason::Length,
        _ => DoneReason::Stop,
    };
    sender.push(AssistantMessageEvent::Done {
        reason,
        message: partial,
    });
}

fn map_stop_reason(s: &str) -> StopReason {
    match s {
        "end_turn" | "stop_sequence" => StopReason::Stop,
        "max_tokens" => StopReason::Length,
        "tool_use" => StopReason::ToolUse,
        _ => StopReason::Stop,
    }
}

fn update_usage(usage: &mut Usage, u: &Value) {
    usage.input = u.get("inputTokens").and_then(|v| v.as_u64()).unwrap_or(0);
    usage.output = u.get("outputTokens").and_then(|v| v.as_u64()).unwrap_or(0);
    usage.cache_read = u
        .get("cacheReadInputTokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    usage.cache_write = u
        .get("cacheWriteInputTokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    usage.total_tokens = usage.input + usage.output + usage.cache_read + usage.cache_write;
}

fn build_request_body(context: &Context, options: &StreamOptions) -> Value {
    let mut body = json!({ "messages": convert_messages(&context.messages) });
    let nova_high = options
        .provider_extras
        .get("bedrock_additional_model_request_fields")
        .and_then(|fields| fields.pointer("/reasoningConfig/maxReasoningEffort"))
        .and_then(Value::as_str)
        == Some("high");
    if !nova_high {
        body["inferenceConfig"] = inference_config(options);
    }
    if let Some(sys) = &context.system_prompt {
        body["system"] = json!([{ "text": sys }]);
    }
    if let Some(tools) = &context.tools {
        if !tools.is_empty() {
            body["toolConfig"] = json!({ "tools": serialize_tools(tools) });
        }
    }
    if let Some(fields) = options
        .provider_extras
        .get("bedrock_additional_model_request_fields")
    {
        body["additionalModelRequestFields"] = fields.clone();
    }
    body
}

fn inference_config(options: &StreamOptions) -> Value {
    let mut cfg = serde_json::Map::new();
    cfg.insert(
        "maxTokens".into(),
        json!(options.max_tokens.unwrap_or(4096)),
    );
    if let Some(t) = options.temperature {
        cfg.insert("temperature".into(), json!(t));
    }
    Value::Object(cfg)
}

fn serialize_tools(tools: &[Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|t| {
            json!({
                "toolSpec": {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": { "json": t.parameters },
                }
            })
        })
        .collect()
}

fn convert_messages(msgs: &[Message]) -> Vec<Value> {
    let mut out = Vec::with_capacity(msgs.len());
    for m in msgs {
        match m {
            Message::User(u) => {
                let content = match &u.content {
                    UserContent::Text(s) => vec![json!({ "text": s })],
                    UserContent::Blocks(blocks) => blocks.iter().map(user_block).collect(),
                };
                out.push(json!({ "role": "user", "content": content }));
            }
            Message::Assistant(a) => {
                let mut content = Vec::new();
                for b in &a.content {
                    match b {
                        ContentBlock::Text(t) => content.push(json!({ "text": t.text })),
                        ContentBlock::ToolCall(tc) => content.push(json!({
                            "toolUse": { "toolUseId": tc.id, "name": tc.name, "input": tc.arguments }
                        })),
                        _ => {}
                    }
                }
                if !content.is_empty() {
                    out.push(json!({ "role": "assistant", "content": content }));
                }
            }
            Message::ToolResult(tr) => {
                let inner: Vec<Value> = tr
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        UserContentBlock::Text(t) => Some(json!({ "text": t.text })),
                        _ => None,
                    })
                    .collect();
                out.push(json!({
                    "role": "user",
                    "content": [{
                        "toolResult": {
                            "toolUseId": tr.tool_call_id,
                            "content": inner,
                            "status": if tr.is_error { "error" } else { "success" },
                        }
                    }],
                }));
            }
        }
    }
    out
}

fn user_block(b: &UserContentBlock) -> Value {
    match b {
        UserContentBlock::Text(t) => json!({ "text": t.text }),
        UserContentBlock::Image(i) => {
            let format = i.mime_type.strip_prefix("image/").unwrap_or("png");
            json!({ "image": { "format": format, "source": { "bytes": i.data } } })
        }
    }
}

fn empty_partial(model: &Model) -> AssistantMessage {
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

fn push_error(sender: &mut AssistantMessageEventSender, model: &Model, msg: String) {
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

    #[test]
    fn body_has_converse_shape() {
        let ctx = Context {
            system_prompt: Some("sys".into()),
            messages: vec![Message::User(UserMessage {
                role: UserRole::User,
                content: UserContent::Text("hi".into()),
                timestamp: 0,
            })],
            tools: Some(vec![Tool {
                name: "t".into(),
                description: "d".into(),
                parameters: json!({ "type": "object" }),
            }]),
        };
        let opts = StreamOptions {
            max_tokens: Some(512),
            ..Default::default()
        };
        let body = build_request_body(&ctx, &opts);
        assert_eq!(body["system"][0]["text"], "sys");
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"][0]["text"], "hi");
        assert_eq!(body["inferenceConfig"]["maxTokens"], 512);
        assert_eq!(body["toolConfig"]["tools"][0]["toolSpec"]["name"], "t");
    }

    #[test]
    fn tool_result_converts() {
        let msgs = vec![Message::ToolResult(ToolResultMessage {
            role: ToolResultRole::ToolResult,
            tool_call_id: "tu_1".into(),
            tool_name: "t".into(),
            content: vec![UserContentBlock::text("ok")],
            details: None,
            is_error: false,
            timestamp: 0,
        })];
        let out = convert_messages(&msgs);
        assert_eq!(out[0]["content"][0]["toolResult"]["toolUseId"], "tu_1");
        assert_eq!(out[0]["content"][0]["toolResult"]["status"], "success");
    }

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(map_stop_reason("end_turn"), StopReason::Stop);
        assert_eq!(map_stop_reason("max_tokens"), StopReason::Length);
        assert_eq!(map_stop_reason("tool_use"), StopReason::ToolUse);
    }

    #[test]
    fn bedrock_stream_simple_rejects_unknown_claude_protocol() {
        let mut model = crate::get_model(
            &Provider::from("amazon-bedrock"),
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
        )
        .expect("built-in Claude model");
        model.id = "anthropic.claude-future-v1:0".into();
        let options = SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        };

        assert_eq!(
            translate_simple_options(&model, Some(&options)).unwrap_err(),
            "Bedrock simple reasoning has no documented configurable protocol for model anthropic.claude-future-v1:0"
        );
    }

    #[test]
    fn bedrock_signing_region_prefers_standard_endpoint_over_env_fallback() {
        let url = url::Url::parse(
            "https://bedrock-runtime.eu-central-1.amazonaws.com/model/test/converse-stream",
        )
        .unwrap();

        assert_eq!(
            resolve_signing_region("test-model", &url, "us-east-1").unwrap(),
            "eu-central-1"
        );
    }

    #[test]
    fn bedrock_signing_region_rejects_arn_endpoint_conflict() {
        let url = url::Url::parse(
            "https://bedrock-runtime.eu-central-1.amazonaws.com/model/test/converse-stream",
        )
        .unwrap();

        assert_eq!(
            resolve_signing_region(
                "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/test",
                &url,
                "us-east-1",
            )
            .unwrap_err(),
            "Bedrock model ARN region us-west-2 conflicts with endpoint region eu-central-1"
        );
    }

    #[test]
    fn bedrock_stream_simple_preserves_base_options_and_omits_absent_reasoning() {
        let model = crate::get_model(&Provider::from("amazon-bedrock"), "amazon.nova-lite-v1:0")
            .expect("built-in Nova model");
        let abort = tokio_util::sync::CancellationToken::new();
        let options = SimpleStreamOptions {
            base: StreamOptions {
                max_tokens: Some(321),
                headers: Some(std::collections::HashMap::from([(
                    "x-test-header".into(),
                    "kept".into(),
                )])),
                abort: Some(abort.clone()),
                ..Default::default()
            },
            reasoning: None,
            ..Default::default()
        };
        let translated = translate_simple_options(&model, Some(&options)).unwrap();
        let context = Context {
            tools: Some(vec![Tool {
                name: "lookup".into(),
                description: "look up a value".into(),
                parameters: json!({ "type": "object" }),
            }]),
            ..Default::default()
        };
        let body = build_request_body(&context, &translated);

        assert_eq!(body["inferenceConfig"]["maxTokens"], 321);
        assert_eq!(body["toolConfig"]["tools"][0]["toolSpec"]["name"], "lookup");
        assert!(body.get("additionalModelRequestFields").is_none());
        assert_eq!(
            translated
                .headers
                .as_ref()
                .and_then(|headers| headers.get("x-test-header"))
                .map(String::as_str),
            Some("kept")
        );
        abort.cancel();
        assert!(translated.abort.as_ref().unwrap().is_cancelled());
    }
}
