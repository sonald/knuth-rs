mod support;

#[cfg(any(
    feature = "openai-completions",
    feature = "amazon-bedrock",
    feature = "google-vertex"
))]
use std::collections::HashMap;

#[cfg(any(feature = "openai-responses", feature = "openai-codex-responses"))]
use ai::Message;
#[cfg(any(feature = "openai-codex-responses", feature = "amazon-bedrock"))]
use ai::Provider;
#[cfg(feature = "openai-codex-responses")]
use ai::{Api, AssistantMessage, ContentBlock, StopReason, ThinkingContent, ToolCall, Usage};
#[cfg(any(
    feature = "openai-completions",
    feature = "openai-responses",
    feature = "openai-codex-responses",
    feature = "amazon-bedrock",
    feature = "mistral",
    feature = "google",
    feature = "google-vertex",
    all(
        feature = "cloudflare",
        any(feature = "anthropic", feature = "openai-responses")
    )
))]
use ai::{AssistantMessageEvent, Context, KnownApi, StreamOptions, stream};
#[cfg(feature = "amazon-bedrock")]
use ai::{SimpleStreamOptions, ThinkingLevel, stream_simple};
#[cfg(any(feature = "openai-responses", feature = "openai-codex-responses"))]
use ai::{UserContent, UserMessage, UserRole};
#[cfg(any(
    feature = "openai-completions",
    feature = "openai-responses",
    feature = "openai-codex-responses",
    feature = "amazon-bedrock",
    feature = "mistral",
    feature = "google",
    feature = "google-vertex",
    all(
        feature = "cloudflare",
        any(feature = "anthropic", feature = "openai-responses")
    )
))]
use futures::StreamExt;
#[cfg(feature = "amazon-bedrock")]
use sha2::{Digest, Sha256};

#[cfg(any(
    feature = "amazon-bedrock",
    feature = "google-vertex",
    all(
        feature = "cloudflare",
        any(
            feature = "openai-completions",
            feature = "anthropic",
            feature = "openai-responses"
        )
    )
))]
use support::EnvVarGuard;

#[cfg(feature = "openai-completions")]
const COMPLETIONS_SSE: &[u8] =
    br#"data: {"id":"response_1","choices":[{"finish_reason":"stop","delta":{}}]}

data: [DONE]

"#;

#[cfg(all(feature = "anthropic", feature = "cloudflare"))]
const ANTHROPIC_SSE: &[u8] = br#"event: message_start
data: {"type":"message_start","message":{"id":"msg_1","usage":{}}}

event: message_stop
data: {"type":"message_stop"}

"#;

#[cfg(all(feature = "openai-responses", feature = "cloudflare"))]
const RESPONSES_SSE: &[u8] = br#"event: response.created
data: {"type":"response.created","response":{"id":"resp_1"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{}}}

"#;

#[cfg(feature = "openai-codex-responses")]
const CODEX_RESPONSES_SSE: &[u8] = br#"event: response.created
data: {"type":"response.created","response":{"id":"resp_codex"}}

event: response.completed
data: {"type":"response.completed","response":{"id":"resp_codex","status":"completed","output":[],"usage":{}}}

"#;

#[cfg(feature = "openai-responses")]
const RESPONSES_ROUND_TRIP_SSE: &[u8] = br#"data: {"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","id":"rs_1","summary":[]}}

data: {"type":"response.output_item.done","output_index":0,"item":{"type":"reasoning","id":"rs_1","encrypted_content":"encrypted","summary":[{"type":"summary_text","text":"plan"}]}}

data: {"type":"response.output_item.added","output_index":1,"item":{"type":"message","id":"msg_commentary","phase":"commentary","content":[]}}

data: {"type":"response.content_part.added","item_id":"msg_commentary","output_index":1,"content_index":0,"part":{"type":"output_text","text":""}}

data: {"type":"response.output_text.delta","item_id":"msg_commentary","output_index":1,"content_index":0,"delta":"checking"}

data: {"type":"response.output_text.done","item_id":"msg_commentary","output_index":1,"content_index":0,"text":"checking"}

data: {"type":"response.output_item.done","output_index":1,"item":{"type":"message","id":"msg_commentary","phase":"commentary","content":[{"type":"output_text","text":"checking"}]}}

data: {"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":""}}

data: {"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":2,"arguments":"{\"path\":\"README.md\"}"}

data: {"type":"response.output_item.done","output_index":2,"item":{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read","arguments":"{\"path\":\"README.md\"}"}}

data: {"type":"response.output_item.added","output_index":3,"item":{"type":"message","id":"msg_final","phase":"final_answer","content":[]}}

data: {"type":"response.content_part.added","item_id":"msg_final","output_index":3,"content_index":0,"part":{"type":"output_text","text":""}}

data: {"type":"response.output_text.done","item_id":"msg_final","output_index":3,"content_index":0,"text":"finished"}

data: {"type":"response.output_item.done","output_index":3,"item":{"type":"message","id":"msg_final","phase":"final_answer","content":[{"type":"output_text","text":"finished"}]}}

data: {"type":"response.completed","response":{"id":"resp_round_trip","status":"completed","output":[],"usage":{}}}

"#;

#[cfg(feature = "openai-completions")]
async fn capture_completions_request(
    model_headers: Option<HashMap<String, String>>,
    option_headers: Option<HashMap<String, String>>,
) -> String {
    let (base_url, captured) =
        support::serve_capture_once(COMPLETIONS_SSE, "text/event-stream").await;
    let mut model = support::model(
        KnownApi::OpenAICompletions,
        "openai",
        "test-model",
        base_url,
    );
    model.headers = model_headers;
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        headers: option_headers,
        ..Default::default()
    };

    let mut events = stream(&model, &Context::default(), Some(&options));
    let mut saw_done = false;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Done { .. } => saw_done = true,
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected stream error: {:?}", error.error_message)
            }
            _ => {}
        }
    }
    assert!(saw_done, "expected normally terminated stream");
    captured.await.unwrap().request
}

#[cfg(feature = "openai-codex-responses")]
async fn capture_codex_request(context: Context, options: StreamOptions) -> serde_json::Value {
    let (base_url, captured) =
        support::serve_capture_once(CODEX_RESPONSES_SSE, "text/event-stream").await;
    let model = support::model(
        KnownApi::OpenAICodexResponses,
        "openai-codex",
        "gpt-5-codex",
        base_url,
    );

    let mut events = stream(&model, &context, Some(&options));
    let mut saw_done = false;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Done { .. } => saw_done = true,
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected stream error: {:?}", error.error_message)
            }
            _ => {}
        }
    }
    assert!(saw_done, "expected normally terminated stream");

    let request = captured.await.unwrap().request;
    let (_, body) = request
        .split_once("\r\n\r\n")
        .expect("captured HTTP request body");
    serde_json::from_str(body).expect("valid request JSON")
}

#[cfg(feature = "openai-responses")]
#[tokio::test]
async fn responses_store_false_round_trip_preserves_interleaved_output_items() {
    let first_url = support::serve_once(RESPONSES_ROUND_TRIP_SSE, "text/event-stream").await;
    let mut model = support::model(KnownApi::OpenAIResponses, "openai", "gpt-test", first_url);
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut first_stream = stream(&model, &Context::default(), Some(&options));
    let mut first_message = None;
    while let Some(event) = first_stream.next().await {
        match event {
            AssistantMessageEvent::Done { message, .. } => first_message = Some(message),
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected first-turn error: {:?}", error.error_message)
            }
            _ => {}
        }
    }

    let second_response = br#"data: {"type":"response.completed","response":{"id":"resp_2","status":"completed","output":[],"usage":{}}}

"#;
    let (second_url, captured) =
        support::serve_capture_once(second_response, "text/event-stream").await;
    model.base_url = second_url;
    let context = Context {
        system_prompt: None,
        messages: vec![
            Message::Assistant(first_message.expect("first-turn Done message")),
            Message::User(UserMessage {
                role: UserRole::User,
                content: UserContent::Text("continue".into()),
                timestamp: 0,
            }),
        ],
        tools: None,
    };
    let mut second_stream = stream(&model, &context, Some(&options));
    let mut second_done = false;
    while let Some(event) = second_stream.next().await {
        match event {
            AssistantMessageEvent::Done { .. } => second_done = true,
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected second-turn error: {:?}", error.error_message)
            }
            _ => {}
        }
    }
    assert!(second_done, "second turn must terminate with Done");

    let request = captured.await.unwrap().request;
    let (_, body) = request.split_once("\r\n\r\n").unwrap();
    let body: serde_json::Value = serde_json::from_str(body).unwrap();
    let input = body["input"].as_array().unwrap();
    assert_eq!(
        input[..4]
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["reasoning", "message", "function_call", "message"]
    );
    assert_eq!(input[1]["id"], "msg_commentary");
    assert_eq!(input[1]["phase"], "commentary");
    assert_eq!(input[1]["content"][0]["text"], "checking");
    assert_eq!(input[2]["id"], "fc_1");
    assert_eq!(input[2]["call_id"], "call_1");
    assert_eq!(input[3]["id"], "msg_final");
    assert_eq!(input[3]["phase"], "final_answer");
    assert_eq!(input[3]["content"][0]["text"], "finished");
    assert_eq!(input[4]["role"], "user");
}

#[cfg(feature = "openai-codex-responses")]
fn codex_options(max_tokens: Option<u32>) -> StreamOptions {
    StreamOptions {
        api_key: Some("test-key".into()),
        max_tokens,
        ..Default::default()
    }
}

#[cfg(feature = "openai-codex-responses")]
#[tokio::test]
async fn codex_request_includes_max_output_tokens() {
    let body = capture_codex_request(Context::default(), codex_options(Some(1234))).await;

    assert_eq!(body["max_output_tokens"], 1234);
    assert!(body.get("max_tokens").is_none());
}

#[cfg(feature = "openai-codex-responses")]
#[tokio::test]
async fn codex_request_omits_max_output_tokens_by_default() {
    let body = capture_codex_request(Context::default(), codex_options(None)).await;

    assert!(body.get("max_output_tokens").is_none());
    assert!(body.get("max_tokens").is_none());
}

#[cfg(feature = "openai-codex-responses")]
#[tokio::test]
async fn codex_replays_encrypted_reasoning_items() {
    let encrypted_item = serde_json::json!({
        "type": "reasoning",
        "id": "rs_123",
        "encrypted_content": "encrypted-payload",
        "summary": [{ "type": "summary_text", "text": "brief summary" }]
    });
    let assistant = Message::Assistant(AssistantMessage {
        role: Default::default(),
        content: vec![
            ContentBlock::Thinking(ThinkingContent {
                thinking: "brief summary".into(),
                thinking_signature: Some(encrypted_item.to_string()),
                redacted: false,
            }),
            ContentBlock::Thinking(ThinkingContent {
                thinking: "ordinary local thinking".into(),
                thinking_signature: None,
                redacted: false,
            }),
            ContentBlock::text("answer"),
        ],
        api: Api::known(KnownApi::OpenAICodexResponses),
        provider: Provider::from("openai-codex"),
        model: "gpt-5-codex".into(),
        response_model: None,
        response_id: Some("resp_previous".into()),
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::Stop,
        error_message: None,
        timestamp: 0,
    });
    let context = Context {
        system_prompt: None,
        messages: vec![assistant],
        tools: None,
    };

    let body = capture_codex_request(context, codex_options(None)).await;
    let input = body["input"].as_array().expect("input array");
    let reasoning_items: Vec<_> = input
        .iter()
        .filter(|item| item["type"] == "reasoning")
        .collect();

    assert_eq!(reasoning_items, vec![&encrypted_item]);
}

#[cfg(feature = "openai-codex-responses")]
#[tokio::test]
async fn codex_replays_only_well_formed_encrypted_reasoning_in_mixed_history() {
    let encrypted_item = serde_json::json!({
        "type": "reasoning",
        "id": "rs_valid",
        "encrypted_content": "encrypted-payload",
        "summary": [],
    });
    let thinking = |signature: Option<String>| {
        ContentBlock::Thinking(ThinkingContent {
            thinking: "not replayable".into(),
            thinking_signature: signature,
            redacted: false,
        })
    };
    let assistant = Message::Assistant(AssistantMessage {
        role: Default::default(),
        content: vec![
            ContentBlock::text("before"),
            thinking(Some("{".into())),
            ContentBlock::ToolCall(ToolCall {
                id: "call_1|fc_1".into(),
                name: "read".into(),
                arguments: Default::default(),
                thought_signature: None,
            }),
            thinking(Some(
                serde_json::json!({
                    "type": "message",
                    "encrypted_content": "wrong-type",
                })
                .to_string(),
            )),
            ContentBlock::text("middle"),
            thinking(Some(
                serde_json::json!({ "type": "reasoning", "id": "rs_missing" }).to_string(),
            )),
            thinking(Some(
                serde_json::json!({
                    "type": "reasoning",
                    "encrypted_content": "missing-id",
                })
                .to_string(),
            )),
            thinking(Some(
                serde_json::json!({
                    "type": "reasoning",
                    "id": "",
                    "encrypted_content": "empty-id",
                })
                .to_string(),
            )),
            thinking(Some(
                serde_json::json!({
                    "type": "reasoning",
                    "id": "rs_non_string",
                    "encrypted_content": 7,
                })
                .to_string(),
            )),
            thinking(None),
            thinking(Some(encrypted_item.to_string())),
            ContentBlock::text("after"),
        ],
        api: Api::known(KnownApi::OpenAICodexResponses),
        provider: Provider::from("openai-codex"),
        model: "gpt-5-codex".into(),
        response_model: None,
        response_id: Some("resp_previous".into()),
        diagnostics: None,
        usage: Usage::default(),
        stop_reason: StopReason::ToolUse,
        error_message: None,
        timestamp: 0,
    });
    let context = Context {
        system_prompt: None,
        messages: vec![
            assistant,
            Message::User(UserMessage {
                role: UserRole::User,
                content: UserContent::Text("continue".into()),
                timestamp: 0,
            }),
        ],
        tools: None,
    };

    let body = capture_codex_request(context, codex_options(None)).await;
    let input = body["input"].as_array().unwrap();
    assert_eq!(
        input[..5]
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "message",
            "function_call",
            "message",
            "reasoning",
            "message"
        ]
    );
    assert_eq!(input[0]["content"][0]["text"], "before");
    assert_eq!(input[1]["call_id"], "call_1");
    assert_eq!(input[2]["content"][0]["text"], "middle");
    assert_eq!(input[3], encrypted_item);
    assert_eq!(input[4]["content"][0]["text"], "after");
    assert_eq!(input[5]["role"], "user");
    assert_eq!(
        input
            .iter()
            .filter(|item| item["type"] == "reasoning")
            .count(),
        1
    );
}

#[cfg(any(
    feature = "openai-completions",
    feature = "amazon-bedrock",
    feature = "google-vertex"
))]
fn header_values<'a>(request: &'a str, target: &str) -> Vec<&'a str> {
    request
        .lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case(target))
        .map(|(_, value)| value.trim())
        .collect()
}

#[cfg(any(
    feature = "amazon-bedrock",
    feature = "mistral",
    feature = "google",
    feature = "google-vertex"
))]
async fn stream_to_done(
    model: &ai::Model,
    context: &Context,
    options: &StreamOptions,
) -> ai::AssistantMessage {
    let mut events = stream(model, context, Some(options));
    let mut done = None;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Done { message, .. } => done = Some(message),
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected stream error: {:?}", error.error_message)
            }
            _ => {}
        }
    }
    done.expect("expected normally terminated stream")
}

#[cfg(feature = "mistral")]
const MISTRAL_REASONING_SSE: &[u8] = br#"data: {"id":"mistral-reasoning","choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"plan"}]},{"type":"text","text":"answer"},{"type":"thinking","thinking":[{"type":"text","text":"review"}]},{"type":"text","text":"final"}]},"finish_reason":"stop"}]}

data: [DONE]

"#;

#[cfg(feature = "mistral")]
const MISTRAL_DONE_SSE: &[u8] =
    br#"data: {"id":"mistral-done","choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]

"#;

#[cfg(feature = "mistral")]
#[tokio::test]
async fn mistral_public_stream_replays_thinking_and_text_chunks_in_block_order() {
    let first_user = ai::Message::User(ai::UserMessage {
        role: ai::UserRole::User,
        content: ai::UserContent::Text("first turn".into()),
        timestamp: 0,
    });
    let options = StreamOptions {
        api_key: Some("mistral-test-key".into()),
        ..Default::default()
    };
    let first_context = Context {
        messages: vec![first_user.clone()],
        ..Default::default()
    };
    let (base_url, first_request) =
        support::serve_capture_once(MISTRAL_REASONING_SSE, "text/event-stream").await;
    let model = support::model(
        KnownApi::MistralConversations,
        "mistral",
        "mistral-small-latest",
        base_url,
    );

    let assistant = stream_to_done(&model, &first_context, &options).await;
    let _ = first_request.await.expect("first captured request");
    assert!(matches!(
        assistant.content.as_slice(),
        [
            ai::ContentBlock::Thinking(first_thinking),
            ai::ContentBlock::Text(first_text),
            ai::ContentBlock::Thinking(second_thinking),
            ai::ContentBlock::Text(second_text),
        ] if first_thinking.thinking == "plan"
            && first_text.text == "answer"
            && second_thinking.thinking == "review"
            && second_text.text == "final"
    ));

    let next_context = Context {
        messages: vec![
            first_user,
            ai::Message::Assistant(assistant),
            ai::Message::User(ai::UserMessage {
                role: ai::UserRole::User,
                content: ai::UserContent::Text("second turn".into()),
                timestamp: 1,
            }),
        ],
        ..Default::default()
    };
    let (base_url, next_request) =
        support::serve_capture_once(MISTRAL_DONE_SSE, "text/event-stream").await;
    let model = support::model(
        KnownApi::MistralConversations,
        "mistral",
        "mistral-small-latest",
        base_url,
    );
    let _ = stream_to_done(&model, &next_context, &options).await;

    let request = next_request.await.expect("next captured request").request;
    let body = captured_request_json(&request);
    assert_eq!(
        body["messages"][1]["content"],
        serde_json::json!([
            {
                "type": "thinking",
                "thinking": [{ "type": "text", "text": "plan" }]
            },
            { "type": "text", "text": "answer" },
            {
                "type": "thinking",
                "thinking": [{ "type": "text", "text": "review" }]
            },
            { "type": "text", "text": "final" }
        ])
    );
}

#[cfg(any(
    feature = "amazon-bedrock",
    feature = "mistral",
    feature = "google",
    feature = "google-vertex"
))]
fn captured_request_json(request: &str) -> serde_json::Value {
    serde_json::from_str(
        request
            .split_once("\r\n\r\n")
            .expect("captured request body")
            .1,
    )
    .expect("valid captured request JSON")
}

#[cfg(any(feature = "google", feature = "google-vertex"))]
const GOOGLE_SIGNATURE_SSE: &[u8] = br#"data: {"responseId":"signature-response","candidates":[{"content":{"parts":[{"thought":true,"text":"plan","thoughtSignature":"dGhpbmtpbmc="},{"text":"answer","thoughtSignature":"dGV4dA=="},{"text":"plain"},{"text":"","thoughtSignature":"ZW1wdHk="}]},"finishReason":"STOP"}]}

"#;

#[cfg(any(feature = "google", feature = "google-vertex"))]
const GOOGLE_DONE_SSE: &[u8] = br#"data: {"responseId":"vertex-response","candidates":[{"content":{"parts":[{"text":"vertex ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":20,"toolUsePromptTokenCount":5,"candidatesTokenCount":10,"thoughtsTokenCount":5}}

"#;

#[cfg(any(feature = "google", feature = "google-vertex"))]
async fn assert_google_signature_round_trip(
    api: KnownApi,
    provider: &str,
    options: &StreamOptions,
) {
    let first_user = ai::Message::User(ai::UserMessage {
        role: ai::UserRole::User,
        content: ai::UserContent::Text("first turn".into()),
        timestamp: 0,
    });
    let first_context = Context {
        messages: vec![first_user.clone()],
        ..Default::default()
    };
    let (base_url, first_request) =
        support::serve_capture_once(GOOGLE_SIGNATURE_SSE, "text/event-stream").await;
    let model = support::model(api, provider, "gemini-test", base_url);

    let assistant = stream_to_done(&model, &first_context, options).await;
    let _ = first_request.await.expect("first captured request");
    match assistant.content.as_slice() {
        [
            ai::ContentBlock::Thinking(thinking),
            ai::ContentBlock::Text(text),
            ai::ContentBlock::Text(plain),
            ai::ContentBlock::Text(empty),
        ] => {
            assert_eq!(thinking.thinking, "plan");
            assert_eq!(thinking.thinking_signature.as_deref(), Some("dGhpbmtpbmc="));
            assert_eq!(text.text, "answer");
            assert_eq!(text.text_signature.as_deref(), Some("dGV4dA=="));
            assert_eq!(plain.text, "plain");
            assert_eq!(plain.text_signature, None);
            assert!(empty.text.is_empty());
            assert_eq!(empty.text_signature.as_deref(), Some("ZW1wdHk="));
        }
        content => panic!("unexpected signed Google content blocks: {content:?}"),
    }

    let next_user = ai::Message::User(ai::UserMessage {
        role: ai::UserRole::User,
        content: ai::UserContent::Text("second turn".into()),
        timestamp: 1,
    });
    let next_context = Context {
        messages: vec![first_user, ai::Message::Assistant(assistant), next_user],
        ..Default::default()
    };
    let (base_url, next_request) =
        support::serve_capture_once(GOOGLE_DONE_SSE, "text/event-stream").await;
    let model = support::model(api, provider, "gemini-test", base_url);
    let _ = stream_to_done(&model, &next_context, options).await;
    let request = next_request.await.expect("next captured request").request;
    let body = captured_request_json(&request);

    assert_eq!(
        body["contents"][1]["parts"],
        serde_json::json!([
            {
                "thought": true,
                "text": "plan",
                "thoughtSignature": "dGhpbmtpbmc="
            },
            { "text": "answer", "thoughtSignature": "dGV4dA==" },
            { "text": "plain" },
            { "text": "", "thoughtSignature": "ZW1wdHk=" }
        ])
    );
}

#[cfg(feature = "google")]
#[tokio::test]
async fn google_public_stream_replays_text_thinking_and_empty_part_signatures() {
    assert_google_signature_round_trip(
        KnownApi::GoogleGenerativeAi,
        "google",
        &StreamOptions {
            api_key: Some("google-test-key".into()),
            ..Default::default()
        },
    )
    .await;
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_public_stream_replays_text_thinking_and_empty_part_signatures() {
    let _lock = support::env_lock().lock().await;
    let _project = EnvVarGuard::set("GOOGLE_VERTEX_PROJECT", "signature-project");
    let _location = EnvVarGuard::set("GOOGLE_VERTEX_LOCATION", "us-central1");

    assert_google_signature_round_trip(
        KnownApi::GoogleVertex,
        "google-vertex",
        &StreamOptions {
            api_key: Some("vertex-test-token".into()),
            ..Default::default()
        },
    )
    .await;
}

#[cfg(feature = "amazon-bedrock")]
fn bedrock_reasoning_body(signature_chunks: &[&str]) -> &'static [u8] {
    let mut body = Vec::new();
    for text in ["plan ", "carefully"] {
        let payload = serde_json::to_vec(&serde_json::json!({
            "contentBlockIndex": 0,
            "delta": { "reasoningContent": { "text": text } }
        }))
        .unwrap();
        body.extend(support::aws_eventstream_frame(
            "contentBlockDelta",
            &payload,
        ));
    }
    for signature in signature_chunks {
        let payload = serde_json::to_vec(&serde_json::json!({
            "contentBlockIndex": 0,
            "delta": { "reasoningContent": { "signature": signature } }
        }))
        .unwrap();
        body.extend(support::aws_eventstream_frame(
            "contentBlockDelta",
            &payload,
        ));
    }
    body.extend(support::aws_eventstream_frame(
        "contentBlockStop",
        br#"{"contentBlockIndex":0}"#,
    ));
    body.extend(support::aws_eventstream_frame(
        "contentBlockDelta",
        br#"{"contentBlockIndex":1,"delta":{"text":"answer"}}"#,
    ));
    body.extend(support::aws_eventstream_frame(
        "contentBlockStop",
        br#"{"contentBlockIndex":1}"#,
    ));
    body.extend(support::aws_eventstream_frame(
        "messageStop",
        br#"{"stopReason":"end_turn"}"#,
    ));
    Box::leak(body.into_boxed_slice())
}

#[cfg(feature = "amazon-bedrock")]
async fn assert_bedrock_reasoning_round_trip(
    model_id: &str,
    signature_chunks: &[&str],
    expected_signature: Option<&str>,
) {
    let first_user = ai::Message::User(ai::UserMessage {
        role: ai::UserRole::User,
        content: ai::UserContent::Text("first turn".into()),
        timestamp: 0,
    });
    let first_context = Context {
        messages: vec![first_user.clone()],
        ..Default::default()
    };
    let options = StreamOptions {
        api_key: Some("bedrock-bearer".into()),
        ..Default::default()
    };
    let (base_url, first_request) = support::serve_capture_once(
        bedrock_reasoning_body(signature_chunks),
        "application/vnd.amazon.eventstream",
    )
    .await;
    let model = support::model(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        model_id,
        base_url,
    );

    let assistant = stream_to_done(&model, &first_context, &options).await;
    let _ = first_request.await.expect("first captured request");
    match assistant.content.as_slice() {
        [
            ai::ContentBlock::Thinking(thinking),
            ai::ContentBlock::Text(text),
        ] => {
            assert_eq!(thinking.thinking, "plan carefully");
            assert_eq!(thinking.thinking_signature.as_deref(), expected_signature);
            assert_eq!(text.text, "answer");
        }
        content => panic!("unexpected Bedrock reasoning blocks: {content:?}"),
    }

    let next_context = Context {
        messages: vec![
            first_user,
            ai::Message::Assistant(assistant),
            ai::Message::User(ai::UserMessage {
                role: ai::UserRole::User,
                content: ai::UserContent::Text("second turn".into()),
                timestamp: 1,
            }),
        ],
        ..Default::default()
    };
    let (base_url, next_request) =
        support::serve_capture_once(bedrock_done_body(), "application/vnd.amazon.eventstream")
            .await;
    let model = support::model(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        model_id,
        base_url,
    );
    let _ = stream_to_done(&model, &next_context, &options).await;
    let request = next_request.await.expect("next captured request").request;
    let body = captured_request_json(&request);
    let reasoning_text = match expected_signature {
        Some(signature) => serde_json::json!({
            "reasoningContent": {
                "reasoningText": {
                    "text": "plan carefully",
                    "signature": signature
                }
            }
        }),
        None => serde_json::json!({
            "reasoningContent": {
                "reasoningText": { "text": "plan carefully" }
            }
        }),
    };

    assert_eq!(
        body["messages"][1]["content"],
        serde_json::json!([reasoning_text, { "text": "answer" }])
    );
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_public_stream_replays_claude_and_nova_reasoning_in_block_order() {
    assert_bedrock_reasoning_round_trip(
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        &["claude-", "signature"],
        Some("claude-signature"),
    )
    .await;
    assert_bedrock_reasoning_round_trip("amazon.nova-2-lite-v1:0", &[], None).await;
}

#[cfg(any(feature = "amazon-bedrock", feature = "google-vertex"))]
async fn stream_to_error(
    model: &ai::Model,
    context: &Context,
    options: &StreamOptions,
) -> ai::AssistantMessage {
    let mut events = stream(model, context, Some(options));
    let mut error = None;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Error {
                error: error_message,
                ..
            } => error = Some(error_message),
            AssistantMessageEvent::Done { message, .. } => {
                panic!("unexpected Done event: {message:?}")
            }
            _ => {}
        }
    }
    error.expect("expected Error event")
}

#[cfg(feature = "amazon-bedrock")]
fn bedrock_done_body() -> &'static [u8] {
    let mut body = support::aws_eventstream_frame("messageStop", br#"{"stopReason":"end_turn"}"#);
    body.extend(support::aws_eventstream_frame(
        "metadata",
        br#"{"usage":{"inputTokens":5,"outputTokens":2}}"#,
    ));
    Box::leak(body.into_boxed_slice())
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_prefers_bearer_token_but_accepts_sigv4_creds() {
    let _lock = support::env_lock().lock().await;
    let _bearer_absent = EnvVarGuard::remove("AWS_BEARER_TOKEN_BEDROCK");
    let _access_key = EnvVarGuard::set("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE");
    let _secret_key = EnvVarGuard::set("AWS_SECRET_ACCESS_KEY", "secret");
    let _session_token = EnvVarGuard::set("AWS_SESSION_TOKEN", "session-token");
    let _region = EnvVarGuard::set("AWS_REGION", "us-west-2");

    let (base_url, captured) =
        support::serve_capture_once(bedrock_done_body(), "application/vnd.amazon.eventstream")
            .await;
    let mut model = support::model(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        "test-model",
        base_url.clone(),
    );
    model.headers = Some(HashMap::from([
        (
            "content-type".into(),
            "application/json; charset=utf-8".into(),
        ),
        ("x-amz-custom-test".into(), "aws-value".into()),
        ("x-bedrock-custom".into(), "model-value".into()),
    ]));
    let context = Context {
        system_prompt: None,
        messages: vec![ai::Message::User(ai::UserMessage {
            role: ai::UserRole::User,
            content: ai::UserContent::Text("signed body".into()),
            timestamp: 0,
        })],
        tools: None,
    };
    let options = StreamOptions {
        headers: Some(HashMap::from([(
            "x-bedrock-custom".into(),
            "option-value".into(),
        )])),
        ..Default::default()
    };

    stream_to_done(&model, &context, &options).await;
    let request = captured.await.unwrap().request;
    assert!(request.starts_with("POST /model/test-model/converse-stream HTTP/1.1\r\n"));
    let content_type = header_values(&request, "content-type");
    assert_eq!(content_type, ["application/json; charset=utf-8"]);
    assert_eq!(
        header_values(&request, "accept"),
        ["application/vnd.amazon.eventstream"]
    );
    let session_token = header_values(&request, "x-amz-security-token");
    assert_eq!(session_token, ["session-token"]);
    let custom_amz_header = header_values(&request, "x-amz-custom-test");
    assert_eq!(custom_amz_header, ["aws-value"]);
    let custom_bedrock_header = header_values(&request, "x-bedrock-custom");
    assert_eq!(custom_bedrock_header, ["option-value"]);
    let amz_date = header_values(&request, "x-amz-date");
    assert_eq!(amz_date.len(), 1);
    let body_text = request
        .split_once("\r\n\r\n")
        .expect("captured request body")
        .1;
    assert_eq!(
        header_values(&request, "x-amz-content-sha256"),
        [hex::encode(Sha256::digest(body_text.as_bytes()))]
    );
    let body: serde_json::Value = serde_json::from_str(body_text).unwrap();
    assert_eq!(body["messages"][0]["content"][0]["text"], "signed body");
    let authorization = header_values(&request, "authorization");
    assert_eq!(authorization.len(), 1);
    assert!(authorization[0].contains(
        "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-custom-test;x-amz-date;x-amz-security-token;x-bedrock-custom"
    ));
    assert!(!authorization[0].contains("SignedHeaders=accept;"));

    // The fixed botocore vector in sigv4.rs proves canonicalization. This integration check
    // intentionally reuses that signer only to prove the provider gave it the final wire URL,
    // credentials, date, signable headers, and payload captured by the local server.
    let request_target = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .expect("captured request target");
    let signed_url = url::Url::parse(&format!("{base_url}{request_target}"))
        .expect("wire URL for signature comparison");
    assert_eq!(signed_url.path(), request_target);
    let expected_signature = ai::sigv4::sign(&ai::sigv4::SigningRequest {
        method: "POST",
        url: &signed_url,
        headers: &[
            ("content-type", content_type[0]),
            ("x-amz-custom-test", custom_amz_header[0]),
            ("x-bedrock-custom", custom_bedrock_header[0]),
        ],
        payload: body_text.as_bytes(),
        region: "us-west-2",
        service: "bedrock",
        access_key: "AKIDEXAMPLE",
        secret_key: "secret",
        session_token: Some(session_token[0]),
        amz_date: amz_date[0],
    });
    assert_eq!(authorization, [expected_signature.authorization]);

    let _bearer = EnvVarGuard::set("AWS_BEARER_TOKEN_BEDROCK", "bearer-wins");
    let (base_url, captured) =
        support::serve_capture_once(bedrock_done_body(), "application/vnd.amazon.eventstream")
            .await;
    let model = support::model(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        "test-model",
        base_url,
    );
    stream_to_done(&model, &context, &options).await;
    let request = captured.await.unwrap().request;
    assert_eq!(
        header_values(&request, "authorization"),
        ["Bearer bearer-wins"]
    );
    assert!(header_values(&request, "x-amz-date").is_empty());

    let (base_url, captured) =
        support::serve_capture_once(bedrock_done_body(), "application/vnd.amazon.eventstream")
            .await;
    let model = support::model(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        "test-model",
        base_url,
    );
    let options = StreamOptions {
        api_key: Some("options-wins".into()),
        ..Default::default()
    };
    stream_to_done(&model, &context, &options).await;
    assert_eq!(
        header_values(&captured.await.unwrap().request, "authorization"),
        ["Bearer options-wins"]
    );
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_sigv4_uses_inference_profile_arn_region() {
    let _lock = support::env_lock().lock().await;
    let _bearer_absent = EnvVarGuard::remove("AWS_BEARER_TOKEN_BEDROCK");
    let _access_key = EnvVarGuard::set("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE");
    let _secret_key = EnvVarGuard::set("AWS_SECRET_ACCESS_KEY", "secret");
    let _session_token = EnvVarGuard::remove("AWS_SESSION_TOKEN");
    let _region = EnvVarGuard::set("AWS_REGION", "eu-central-1");

    let (base_url, captured) =
        support::serve_capture_once(bedrock_done_body(), "application/vnd.amazon.eventstream")
            .await;
    let model_id =
        "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/test-profile";
    let request_target = "/model/arn%3Aaws%3Abedrock%3Aus-west-2%3A123456789012%3Aapplication-inference-profile%2Ftest-profile/converse-stream";
    let model = support::model(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        model_id,
        base_url.clone(),
    );

    stream_to_done(&model, &Context::default(), &StreamOptions::default()).await;
    let request = captured.await.unwrap().request;
    assert!(request.starts_with(&format!("POST {request_target} HTTP/1.1\r\n")));
    let authorization = header_values(&request, "authorization");
    assert_eq!(authorization.len(), 1);
    assert!(
        authorization[0].contains("/us-west-2/bedrock/aws4_request"),
        "ARN region must override the eu-central-1 environment fallback: {}",
        authorization[0]
    );

    let captured_target = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .expect("captured request target");
    assert_eq!(captured_target, request_target);
    let signed_url = url::Url::parse(&format!("{base_url}{captured_target}"))
        .expect("wire URL for signature comparison");
    assert_eq!(signed_url.path(), captured_target);
    let content_type = header_values(&request, "content-type");
    assert_eq!(content_type, ["application/json"]);
    let amz_date = header_values(&request, "x-amz-date");
    assert_eq!(amz_date.len(), 1);
    let body = request.split_once("\r\n\r\n").expect("request body").1;
    let expected_signature = ai::sigv4::sign(&ai::sigv4::SigningRequest {
        method: "POST",
        url: &signed_url,
        headers: &[("content-type", content_type[0])],
        payload: body.as_bytes(),
        region: "us-west-2",
        service: "bedrock",
        access_key: "AKIDEXAMPLE",
        secret_key: "secret",
        session_token: None,
        amz_date: amz_date[0],
    });
    assert_eq!(
        authorization,
        [expected_signature.authorization],
        "provider-to-signer wiring must sign the encoded ARN wire path with the current canonical double-encode semantics"
    );
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_sigv4_uses_sagemaker_endpoint_arn_region() {
    let _lock = support::env_lock().lock().await;
    let _bearer_absent = EnvVarGuard::remove("AWS_BEARER_TOKEN_BEDROCK");
    let _access_key = EnvVarGuard::set("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE");
    let _secret_key = EnvVarGuard::set("AWS_SECRET_ACCESS_KEY", "secret");
    let _session_token = EnvVarGuard::remove("AWS_SESSION_TOKEN");
    let _region = EnvVarGuard::set("AWS_REGION", "eu-central-1");
    let (base_url, captured) =
        support::serve_capture_once(bedrock_done_body(), "application/vnd.amazon.eventstream")
            .await;
    let model = support::model(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        "arn:aws:sagemaker:us-west-2:123456789012:endpoint/test-endpoint",
        base_url,
    );

    stream_to_done(&model, &Context::default(), &StreamOptions::default()).await;
    let request = captured.await.unwrap().request;
    let authorization = header_values(&request, "authorization");
    assert_eq!(authorization.len(), 1);
    assert!(authorization[0].contains("/us-west-2/bedrock/aws4_request"));
    assert!(request.starts_with(
        "POST /model/arn%3Aaws%3Asagemaker%3Aus-west-2%3A123456789012%3Aendpoint%2Ftest-endpoint/converse-stream HTTP/1.1\r\n"
    ));
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_sagemaker_arn_rejects_endpoint_region_conflict_from_public_stream() {
    let _lock = support::env_lock().lock().await;
    let _bearer_absent = EnvVarGuard::remove("AWS_BEARER_TOKEN_BEDROCK");
    let _access_key = EnvVarGuard::set("AWS_ACCESS_KEY_ID", "AKIDEXAMPLE");
    let _secret_key = EnvVarGuard::set("AWS_SECRET_ACCESS_KEY", "secret");
    let _session_token = EnvVarGuard::remove("AWS_SESSION_TOKEN");
    let _region = EnvVarGuard::set("AWS_REGION", "us-east-1");
    let model = support::model(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        "arn:aws:sagemaker:us-west-2:123456789012:endpoint/test-endpoint",
        "http://bedrock-runtime.eu-central-1.amazonaws.com:9".into(),
    );
    let abort = tokio_util::sync::CancellationToken::new();
    abort.cancel();
    let options = StreamOptions {
        abort: Some(abort),
        timeout_ms: Some(50),
        ..Default::default()
    };

    let error = stream_to_error(&model, &Context::default(), &options).await;
    assert_eq!(
        error.error_message.as_deref(),
        Some(
            "Bedrock SigV4: Bedrock model ARN region us-west-2 conflicts with endpoint region eu-central-1"
        )
    );
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_missing_auth_emits_named_error_without_network_request() {
    let _lock = support::env_lock().lock().await;
    let _bearer = EnvVarGuard::remove("AWS_BEARER_TOKEN_BEDROCK");
    let _access_key = EnvVarGuard::remove("AWS_ACCESS_KEY_ID");
    let _secret_key = EnvVarGuard::remove("AWS_SECRET_ACCESS_KEY");
    let _session_token = EnvVarGuard::remove("AWS_SESSION_TOKEN");
    let (base_url, captured) =
        support::serve_capture_once(bedrock_done_body(), "application/vnd.amazon.eventstream")
            .await;
    let model = support::model(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        "test-model",
        base_url,
    );

    let error = stream_to_error(&model, &Context::default(), &StreamOptions::default()).await;
    assert_eq!(
        error.error_message.as_deref(),
        Some(
            "Bedrock auth missing: set AWS_BEARER_TOKEN_BEDROCK or AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY"
        )
    );
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(50), captured)
            .await
            .is_err(),
        "missing credentials must stop before opening the HTTP connection"
    );
}

#[cfg(feature = "amazon-bedrock")]
async fn capture_bedrock_simple_body(
    model_id: &str,
    mut options: SimpleStreamOptions,
) -> serde_json::Value {
    let (base_url, captured) =
        support::serve_capture_once(bedrock_done_body(), "application/vnd.amazon.eventstream")
            .await;
    let mut model = ai::get_model(&Provider::from("amazon-bedrock"), model_id)
        .unwrap_or_else(|| panic!("missing built-in Bedrock model {model_id}"));
    model.base_url = base_url;
    options.base.api_key = Some("simple-bearer".into());

    let mut events = stream_simple(&model, &Context::default(), Some(&options));
    let mut saw_done = false;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Done { .. } => saw_done = true,
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected stream error: {:?}", error.error_message)
            }
            _ => {}
        }
    }
    assert!(saw_done, "expected normally terminated stream");
    let request = captured.await.unwrap().request;
    serde_json::from_str(
        request
            .split_once("\r\n\r\n")
            .expect("Bedrock request body")
            .1,
    )
    .unwrap()
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_simple_reasoning_uses_claude_adaptive_fields() {
    let body = capture_bedrock_simple_body(
        "global.anthropic.claude-opus-4-6-v1",
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(
        body["additionalModelRequestFields"]["thinking"],
        serde_json::json!({ "type": "adaptive" })
    );
    assert_eq!(
        body["additionalModelRequestFields"]["output_config"],
        serde_json::json!({ "effort": "medium" })
    );
    assert!(
        body["additionalModelRequestFields"]
            .get("reasoning")
            .is_none()
    );
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_simple_reasoning_keeps_claude_budget_below_max_tokens() {
    let model = ai::get_model(
        &Provider::from("amazon-bedrock"),
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
    )
    .expect("built-in Claude 4.5 model");
    let body = capture_bedrock_simple_body(
        &model.id,
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(
        body["additionalModelRequestFields"]["thinking"]["type"],
        "enabled"
    );
    assert_eq!(
        body["additionalModelRequestFields"]["thinking"]["budget_tokens"],
        8192
    );
    let max_tokens = body["inferenceConfig"]["maxTokens"]
        .as_u64()
        .expect("Claude maxTokens");
    assert_eq!(max_tokens, u64::from(model.max_tokens));
    assert!(8192 < max_tokens);
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_simple_reasoning_respects_explicit_max_tokens() {
    let body = capture_bedrock_simple_body(
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        SimpleStreamOptions {
            base: StreamOptions {
                max_tokens: Some(4096),
                ..Default::default()
            },
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    )
    .await;

    let max_tokens = body["inferenceConfig"]["maxTokens"]
        .as_u64()
        .expect("Claude maxTokens");
    let budget = body["additionalModelRequestFields"]["thinking"]["budget_tokens"]
        .as_u64()
        .expect("Claude thinking budget");
    assert_eq!(max_tokens, 4096);
    assert!(budget < max_tokens);
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_simple_reasoning_uses_nova_reasoning_config() {
    let body = capture_bedrock_simple_body(
        "amazon.nova-2-lite-v1:0",
        SimpleStreamOptions {
            reasoning: Some(ThinkingLevel::Medium),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(
        body["additionalModelRequestFields"]["reasoningConfig"],
        serde_json::json!({
            "type": "enabled",
            "maxReasoningEffort": "medium"
        })
    );
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_simple_reasoning_nova_high_omits_inference_limits() {
    let body = capture_bedrock_simple_body(
        "amazon.nova-2-lite-v1:0",
        SimpleStreamOptions {
            base: StreamOptions {
                max_tokens: Some(2048),
                temperature: Some(0.5),
                ..Default::default()
            },
            reasoning: Some(ThinkingLevel::High),
            ..Default::default()
        },
    )
    .await;

    assert_eq!(
        body["additionalModelRequestFields"]["reasoningConfig"]["maxReasoningEffort"],
        "high"
    );
    assert!(body.get("inferenceConfig").is_none());
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_simple_reasoning_fails_closed_for_unmapped_builtin_model() {
    let (base_url, captured) =
        support::serve_capture_once(bedrock_done_body(), "application/vnd.amazon.eventstream")
            .await;
    let mut model = ai::get_model(&Provider::from("amazon-bedrock"), "deepseek.r1-v1:0")
        .expect("built-in DeepSeek model");
    model.base_url = base_url;
    let options = SimpleStreamOptions {
        base: StreamOptions {
            api_key: Some("simple-bearer".into()),
            ..Default::default()
        },
        reasoning: Some(ThinkingLevel::Medium),
        ..Default::default()
    };

    let mut events = stream_simple(&model, &Context::default(), Some(&options));
    let mut terminal = None;
    while let Some(event) = events.next().await {
        if event.is_terminal() {
            terminal = Some(event);
        }
    }
    let AssistantMessageEvent::Error { reason, error } = terminal.expect("terminal event") else {
        panic!("unmapped reasoning model must fail closed");
    };
    assert_eq!(reason, ai::ErrorReason::Error);
    assert_eq!(
        error.error_message.as_deref(),
        Some(
            "Bedrock simple reasoning has no documented configurable protocol for model deepseek.r1-v1:0"
        )
    );
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(50), captured)
            .await
            .is_err(),
        "unsupported reasoning must fail before opening the HTTP connection"
    );
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_can_select_adc_when_access_token_absent() {
    let _lock = support::env_lock().lock().await;
    let (token_uri, token_request) = support::serve_capture_once(
        br#"{"access_token":"adc-local-token","expires_in":3600,"scope":"scope"}"#,
        "application/json",
    )
    .await;
    let credentials = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        credentials.path(),
        serde_json::to_vec(&serde_json::json!({
            "client_email": "svc@test-project.iam.gserviceaccount.com",
            "private_key": support::TEST_RSA_PRIVATE_KEY,
            "token_uri": token_uri,
            "project_id": "service-account-project"
        }))
        .unwrap(),
    )
    .unwrap();
    let _access_token = EnvVarGuard::remove("GOOGLE_VERTEX_ACCESS_TOKEN");
    let _project = EnvVarGuard::remove("GOOGLE_VERTEX_PROJECT");
    let _credentials = EnvVarGuard::set(
        "GOOGLE_APPLICATION_CREDENTIALS",
        credentials.path().to_str().unwrap(),
    );
    let _location = EnvVarGuard::set("GOOGLE_VERTEX_LOCATION", "europe-west4");

    let (base_url, model_request) =
        support::serve_capture_once(GOOGLE_DONE_SSE, "text/event-stream").await;
    let mut model = support::model(
        KnownApi::GoogleVertex,
        "google-vertex",
        "gemini-test",
        base_url,
    );
    model.headers = Some(HashMap::from([
        ("x-model-header".into(), "model-value".into()),
        ("x-shared-header".into(), "model-value".into()),
    ]));
    let options = StreamOptions {
        max_tokens: Some(321),
        headers: Some(HashMap::from([
            ("x-option-header".into(), "option-value".into()),
            ("x-shared-header".into(), "option-value".into()),
        ])),
        ..Default::default()
    };
    let context = Context {
        messages: vec![ai::Message::User(ai::UserMessage {
            role: ai::UserRole::User,
            content: ai::UserContent::Text("vertex body".into()),
            timestamp: 0,
        })],
        ..Default::default()
    };

    let message = stream_to_done(&model, &context, &options).await;
    assert!(matches!(
        message.content.as_slice(),
        [ai::ContentBlock::Text(text)] if text.text == "vertex ok"
    ));

    let token_request = token_request.await.unwrap().request;
    let form = token_request
        .split_once("\r\n\r\n")
        .expect("token request body")
        .1;
    let fields: HashMap<_, _> = url::form_urlencoded::parse(form.as_bytes())
        .into_owned()
        .collect();
    assert_eq!(
        fields.get("grant_type").map(String::as_str),
        Some("urn:ietf:params:oauth:grant-type:jwt-bearer")
    );
    assert_eq!(
        fields
            .get("assertion")
            .map(|assertion| assertion.split('.').count()),
        Some(3)
    );

    let model_request = model_request.await.unwrap().request;
    assert!(model_request.starts_with(
        "POST /v1/projects/service-account-project/locations/europe-west4/publishers/google/models/gemini-test:streamGenerateContent?alt=sse HTTP/1.1\r\n"
    ));
    assert_eq!(
        header_values(&model_request, "authorization"),
        ["Bearer adc-local-token"]
    );
    assert_eq!(
        header_values(&model_request, "x-model-header"),
        ["model-value"]
    );
    assert_eq!(
        header_values(&model_request, "x-option-header"),
        ["option-value"]
    );
    assert_eq!(
        header_values(&model_request, "x-shared-header"),
        ["option-value"]
    );
    let body: serde_json::Value = serde_json::from_str(
        model_request
            .split_once("\r\n\r\n")
            .expect("Vertex request body")
            .1,
    )
    .unwrap();
    assert_eq!(body["contents"][0]["parts"][0]["text"], "vertex body");
    assert_eq!(body["generationConfig"]["maxOutputTokens"], 321);
}

#[cfg(feature = "google-vertex")]
fn write_service_account(
    token_uri: &str,
    project_id: &str,
    private_key: &str,
) -> tempfile::NamedTempFile {
    let credentials = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(
        credentials.path(),
        serde_json::to_vec(&serde_json::json!({
            "client_email": "svc@test-project.iam.gserviceaccount.com",
            "private_key": private_key,
            "token_uri": token_uri,
            "project_id": project_id
        }))
        .unwrap(),
    )
    .unwrap();
    credentials
}

#[cfg(feature = "google-vertex")]
async fn vertex_error_for_credentials(path: &std::path::Path) -> String {
    let _access_token = EnvVarGuard::remove("GOOGLE_VERTEX_ACCESS_TOKEN");
    let _project = EnvVarGuard::set("GOOGLE_VERTEX_PROJECT", "error-project");
    let _credentials = EnvVarGuard::set(
        "GOOGLE_APPLICATION_CREDENTIALS",
        path.to_str().expect("UTF-8 test path"),
    );
    let model = support::model(
        KnownApi::GoogleVertex,
        "google-vertex",
        "gemini-test",
        "http://127.0.0.1:9".into(),
    );
    stream_to_error(&model, &Context::default(), &StreamOptions::default())
        .await
        .error_message
        .expect("named Vertex error")
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_auth_priority_is_options_then_env_then_adc() {
    let _lock = support::env_lock().lock().await;
    let _env_token = EnvVarGuard::set("GOOGLE_VERTEX_ACCESS_TOKEN", "env-token");
    let _project = EnvVarGuard::set("GOOGLE_VERTEX_PROJECT", "priority-project");
    let _credentials = EnvVarGuard::set(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "/path/that/must/not/be/read.json",
    );

    let (base_url, captured) =
        support::serve_capture_once(GOOGLE_DONE_SSE, "text/event-stream").await;
    let model = support::model(
        KnownApi::GoogleVertex,
        "google-vertex",
        "gemini-test",
        base_url,
    );
    let options = StreamOptions {
        api_key: Some("options-token".into()),
        ..Default::default()
    };
    stream_to_done(&model, &Context::default(), &options).await;
    assert_eq!(
        header_values(&captured.await.unwrap().request, "authorization"),
        ["Bearer options-token"]
    );

    let (base_url, captured) =
        support::serve_capture_once(GOOGLE_DONE_SSE, "text/event-stream").await;
    let model = support::model(
        KnownApi::GoogleVertex,
        "google-vertex",
        "gemini-test",
        base_url,
    );
    stream_to_done(&model, &Context::default(), &StreamOptions::default()).await;
    assert_eq!(
        header_values(&captured.await.unwrap().request, "authorization"),
        ["Bearer env-token"]
    );
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_explicit_project_overrides_service_account_project() {
    let _lock = support::env_lock().lock().await;
    let (token_uri, token_request) = support::serve_capture_once(
        br#"{"access_token":"adc-project-token","expires_in":3600}"#,
        "application/json",
    )
    .await;
    let credentials = write_service_account(
        &token_uri,
        "service-account-project",
        support::TEST_RSA_PRIVATE_KEY,
    );
    let _access_token = EnvVarGuard::remove("GOOGLE_VERTEX_ACCESS_TOKEN");
    let _project = EnvVarGuard::set("GOOGLE_VERTEX_PROJECT", "explicit-project");
    let _credentials = EnvVarGuard::set(
        "GOOGLE_APPLICATION_CREDENTIALS",
        credentials.path().to_str().unwrap(),
    );
    let _location = EnvVarGuard::set("GOOGLE_VERTEX_LOCATION", "us-central1");
    let (base_url, model_request) =
        support::serve_capture_once(GOOGLE_DONE_SSE, "text/event-stream").await;
    let model = support::model(
        KnownApi::GoogleVertex,
        "google-vertex",
        "gemini-test",
        base_url,
    );

    stream_to_done(&model, &Context::default(), &StreamOptions::default()).await;
    token_request.await.unwrap();
    assert!(model_request.await.unwrap().request.starts_with(
        "POST /v1/projects/explicit-project/locations/us-central1/publishers/google/models/gemini-test:streamGenerateContent?alt=sse HTTP/1.1\r\n"
    ));
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_adc_file_error_is_public_and_redacted() {
    let _lock = support::env_lock().lock().await;
    let tempdir = tempfile::tempdir().unwrap();
    let path = tempdir.path().join("missing-private-key-sentinel.json");
    let error = vertex_error_for_credentials(&path).await;
    assert!(error.contains("Vertex ADC auth failed while loading credentials: io:"));
    assert!(!error.contains("PRIVATE KEY"));
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_adc_json_error_is_public_and_redacted() {
    let _lock = support::env_lock().lock().await;
    let credentials = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(credentials.path(), "not-json private-key-json-sentinel").unwrap();
    let error = vertex_error_for_credentials(credentials.path()).await;
    assert!(error.contains("parse credentials:"));
    assert!(!error.contains("private-key-json-sentinel"));
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_adc_jwt_error_is_public_and_redacted() {
    let _lock = support::env_lock().lock().await;
    let private_key = "private-key-jwt-sentinel";
    let credentials =
        write_service_account("http://127.0.0.1:9/token", "error-project", private_key);
    let error = vertex_error_for_credentials(credentials.path()).await;
    assert!(error.contains("sign jwt:"));
    assert!(!error.contains(private_key));
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_adc_token_http_error_is_public_and_redacted() {
    let _lock = support::env_lock().lock().await;
    let (token_uri, captured) = support::serve_capture_once_with_status(
        "500 Internal Server Error",
        br#"{"error":"token-http-response-sentinel"}"#,
        "application/json",
    )
    .await;
    let credentials =
        write_service_account(&token_uri, "error-project", support::TEST_RSA_PRIVATE_KEY);
    let error = vertex_error_for_credentials(credentials.path()).await;
    let request = captured.await.unwrap().request;
    let assertion = url::form_urlencoded::parse(
        request
            .split_once("\r\n\r\n")
            .expect("token request body")
            .1
            .as_bytes(),
    )
    .find_map(|(name, value)| (name == "assertion").then(|| value.into_owned()))
    .expect("JWT assertion");
    assert!(error.contains("token exchange: HTTP 500 Internal Server Error"));
    assert!(!error.contains("token-http-response-sentinel"));
    assert!(!error.contains(&assertion));
    assert!(!error.contains("BEGIN PRIVATE KEY"));
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_adc_token_response_error_is_public_and_redacted() {
    let _lock = support::env_lock().lock().await;
    let (token_uri, captured) = support::serve_capture_once(
        br#"{"secret":"token-response-sentinel"}"#,
        "application/json",
    )
    .await;
    let credentials =
        write_service_account(&token_uri, "error-project", support::TEST_RSA_PRIVATE_KEY);
    let error = vertex_error_for_credentials(credentials.path()).await;
    let request = captured.await.unwrap().request;
    let assertion = url::form_urlencoded::parse(
        request
            .split_once("\r\n\r\n")
            .expect("token request body")
            .1
            .as_bytes(),
    )
    .find_map(|(name, value)| (name == "assertion").then(|| value.into_owned()))
    .expect("JWT assertion");
    assert!(error.contains("parse token response:"));
    assert!(!error.contains("token-response-sentinel"));
    assert!(!error.contains(&assertion));
    assert!(!error.contains("BEGIN PRIVATE KEY"));
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_adc_token_exchange_honors_abort() {
    let _lock = support::env_lock().lock().await;
    let (token_uri, token_request) = support::serve_capture_hanging_once().await;
    let credentials =
        write_service_account(&token_uri, "abort-project", support::TEST_RSA_PRIVATE_KEY);
    let _access_token = EnvVarGuard::remove("GOOGLE_VERTEX_ACCESS_TOKEN");
    let _project = EnvVarGuard::remove("GOOGLE_VERTEX_PROJECT");
    let _credentials = EnvVarGuard::set(
        "GOOGLE_APPLICATION_CREDENTIALS",
        credentials.path().to_str().unwrap(),
    );
    let model = support::model(
        KnownApi::GoogleVertex,
        "google-vertex",
        "gemini-test",
        "http://127.0.0.1:9".into(),
    );
    let abort = tokio_util::sync::CancellationToken::new();
    let options = StreamOptions {
        abort: Some(abort.clone()),
        ..Default::default()
    };
    let mut events = stream(&model, &Context::default(), Some(&options));

    tokio::time::timeout(std::time::Duration::from_secs(1), token_request)
        .await
        .expect("ADC token request must start")
        .expect("captured ADC token request");
    abort.cancel();
    let terminal = tokio::time::timeout(std::time::Duration::from_millis(500), async {
        while let Some(event) = events.next().await {
            if event.is_terminal() {
                return event;
            }
        }
        panic!("Vertex stream ended without a terminal event");
    })
    .await
    .expect("abort must stop a hanging ADC exchange");

    let AssistantMessageEvent::Error { reason, error } = terminal else {
        panic!("abort must emit Error");
    };
    assert_eq!(reason, ai::ErrorReason::Aborted);
    assert_eq!(error.stop_reason, ai::StopReason::Aborted);
    assert_eq!(error.error_message.as_deref(), Some("aborted"));
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_adc_token_exchange_honors_timeout_ms() {
    let _lock = support::env_lock().lock().await;
    let (token_uri, token_request) = support::serve_capture_hanging_once().await;
    let credentials =
        write_service_account(&token_uri, "timeout-project", support::TEST_RSA_PRIVATE_KEY);
    let _access_token = EnvVarGuard::remove("GOOGLE_VERTEX_ACCESS_TOKEN");
    let _project = EnvVarGuard::remove("GOOGLE_VERTEX_PROJECT");
    let _credentials = EnvVarGuard::set(
        "GOOGLE_APPLICATION_CREDENTIALS",
        credentials.path().to_str().unwrap(),
    );
    let model = support::model(
        KnownApi::GoogleVertex,
        "google-vertex",
        "gemini-test",
        "http://127.0.0.1:9".into(),
    );
    let options = StreamOptions {
        timeout_ms: Some(50),
        ..Default::default()
    };
    let mut events = stream(&model, &Context::default(), Some(&options));

    tokio::time::timeout(std::time::Duration::from_secs(1), token_request)
        .await
        .expect("ADC token request must start")
        .expect("captured ADC token request");
    let terminal = tokio::time::timeout(std::time::Duration::from_millis(500), async {
        while let Some(event) = events.next().await {
            if event.is_terminal() {
                return event;
            }
        }
        panic!("Vertex stream ended without a terminal event");
    })
    .await
    .expect("timeout_ms must stop a hanging ADC exchange");

    let AssistantMessageEvent::Error { reason, error } = terminal else {
        panic!("timeout must emit Error");
    };
    assert_eq!(reason, ai::ErrorReason::Error);
    assert_eq!(error.stop_reason, ai::StopReason::Error);
    let message = error.error_message.expect("named ADC timeout error");
    assert!(message.starts_with("Vertex ADC auth failed during token exchange: token exchange:"));
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_adc_token_response_body_honors_abort() {
    let _lock = support::env_lock().lock().await;
    let (token_uri, token_request) = support::serve_hanging_response_body_once().await;
    let credentials = write_service_account(
        &token_uri,
        "body-abort-project",
        support::TEST_RSA_PRIVATE_KEY,
    );
    let _access_token = EnvVarGuard::remove("GOOGLE_VERTEX_ACCESS_TOKEN");
    let _project = EnvVarGuard::remove("GOOGLE_VERTEX_PROJECT");
    let _credentials = EnvVarGuard::set(
        "GOOGLE_APPLICATION_CREDENTIALS",
        credentials.path().to_str().unwrap(),
    );
    let model = support::model(
        KnownApi::GoogleVertex,
        "google-vertex",
        "gemini-test",
        "http://127.0.0.1:9".into(),
    );
    let abort = tokio_util::sync::CancellationToken::new();
    let options = StreamOptions {
        abort: Some(abort.clone()),
        ..Default::default()
    };
    let mut events = stream(&model, &Context::default(), Some(&options));

    tokio::time::timeout(std::time::Duration::from_secs(1), token_request)
        .await
        .expect("ADC token response headers must arrive")
        .expect("captured ADC token request");
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    abort.cancel();
    let terminal = tokio::time::timeout(std::time::Duration::from_millis(500), async {
        while let Some(event) = events.next().await {
            if event.is_terminal() {
                return event;
            }
        }
        panic!("Vertex stream ended without a terminal event");
    })
    .await
    .expect("abort must stop a hanging ADC token response body");

    let AssistantMessageEvent::Error { reason, error } = terminal else {
        panic!("abort must emit Error");
    };
    assert_eq!(reason, ai::ErrorReason::Aborted);
    assert_eq!(error.stop_reason, ai::StopReason::Aborted);
    assert_eq!(error.error_message.as_deref(), Some("aborted"));
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_adc_token_response_body_honors_timeout_ms() {
    let _lock = support::env_lock().lock().await;
    let (token_uri, token_request) = support::serve_hanging_response_body_once().await;
    let credentials = write_service_account(
        &token_uri,
        "body-timeout-project",
        support::TEST_RSA_PRIVATE_KEY,
    );
    let _access_token = EnvVarGuard::remove("GOOGLE_VERTEX_ACCESS_TOKEN");
    let _project = EnvVarGuard::remove("GOOGLE_VERTEX_PROJECT");
    let _credentials = EnvVarGuard::set(
        "GOOGLE_APPLICATION_CREDENTIALS",
        credentials.path().to_str().unwrap(),
    );
    let model = support::model(
        KnownApi::GoogleVertex,
        "google-vertex",
        "gemini-test",
        "http://127.0.0.1:9".into(),
    );
    let options = StreamOptions {
        timeout_ms: Some(50),
        ..Default::default()
    };
    let mut events = stream(&model, &Context::default(), Some(&options));

    tokio::time::timeout(std::time::Duration::from_secs(1), token_request)
        .await
        .expect("ADC token response headers must arrive")
        .expect("captured ADC token request");
    let terminal = tokio::time::timeout(std::time::Duration::from_millis(500), async {
        while let Some(event) = events.next().await {
            if event.is_terminal() {
                return event;
            }
        }
        panic!("Vertex stream ended without a terminal event");
    })
    .await
    .expect("timeout_ms must stop a hanging ADC token response body");

    let AssistantMessageEvent::Error { reason, error } = terminal else {
        panic!("timeout must emit Error");
    };
    assert_eq!(reason, ai::ErrorReason::Error);
    assert_eq!(error.stop_reason, ai::StopReason::Error);
    let message = error.error_message.expect("named ADC timeout error");
    assert!(message.starts_with("Vertex ADC auth failed during token exchange: token exchange:"));
}

#[cfg(feature = "openai-completions")]
#[tokio::test]
async fn openai_completions_merges_model_headers_then_options_headers() {
    let request = capture_completions_request(
        Some(HashMap::from([
            ("x-model-header".into(), "model".into()),
            ("x-shared-header".into(), "model".into()),
        ])),
        Some(HashMap::from([
            ("x-options-header".into(), "options".into()),
            ("x-shared-header".into(), "options".into()),
        ])),
    )
    .await;

    assert_eq!(header_values(&request, "x-model-header"), ["model"]);
    assert_eq!(header_values(&request, "x-options-header"), ["options"]);
    assert_eq!(header_values(&request, "x-shared-header"), ["options"]);
}

#[cfg(feature = "openai-completions")]
#[tokio::test]
async fn model_headers_override_provider_defaults() {
    let request = capture_completions_request(
        Some(HashMap::from([(
            "Content-Type".into(),
            "application/model+json".into(),
        )])),
        None,
    )
    .await;

    assert_eq!(
        header_values(&request, "content-type"),
        ["application/model+json"]
    );
}

#[cfg(feature = "openai-completions")]
#[tokio::test]
async fn options_headers_override_provider_defaults() {
    let request = capture_completions_request(
        None,
        Some(HashMap::from([(
            "Accept".into(),
            "application/options".into(),
        )])),
    )
    .await;

    assert_eq!(header_values(&request, "accept"), ["application/options"]);
}

#[cfg(feature = "openai-completions")]
#[tokio::test]
async fn options_headers_override_model_headers_case_insensitively() {
    let request = capture_completions_request(
        Some(HashMap::from([("X-Shared-Header".into(), "model".into())])),
        Some(HashMap::from([(
            "x-shared-header".into(),
            "options".into(),
        )])),
    )
    .await;

    assert_eq!(header_values(&request, "x-shared-header"), ["options"]);
}

#[cfg(feature = "openai-completions")]
#[tokio::test]
async fn invalid_custom_header_emits_error() {
    let mut model = support::model(
        KnownApi::OpenAICompletions,
        "openai",
        "test-model",
        "http://127.0.0.1:1".into(),
    );
    model.headers = Some(HashMap::from([(
        "invalid header name".into(),
        "value".into(),
    )]));
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };

    let mut events = stream(&model, &Context::default(), Some(&options));
    let mut error_message = None;
    let mut saw_done = false;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Error { error, .. } => error_message = error.error_message,
            AssistantMessageEvent::Done { .. } => saw_done = true,
            _ => {}
        }
    }

    assert!(!saw_done, "invalid custom header must not produce Done");
    assert!(
        error_message
            .as_deref()
            .is_some_and(|message| message.contains("custom request headers")),
        "error message: {error_message:?}"
    );
}

#[cfg(all(feature = "openai-completions", feature = "cloudflare"))]
#[tokio::test]
async fn cloudflare_placeholders_are_resolved_before_request() {
    const ENV_NAME: &str = "CLOUDFLARE_ACCOUNT_ID";

    let _lock = support::env_lock().lock().await;
    let _env = EnvVarGuard::set(ENV_NAME, "acct123");
    let (base, captured) = support::serve_capture_once(COMPLETIONS_SSE, "text/event-stream").await;
    let model = support::model(
        KnownApi::OpenAICompletions,
        "cloudflare-workers-ai",
        "@cf/meta/llama",
        format!("{base}/accounts/{{CLOUDFLARE_ACCOUNT_ID}}/ai/v1"),
    );
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &Context::default(), Some(&options));
    let mut saw_done = false;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Done { .. } => saw_done = true,
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected stream error: {:?}", error.error_message)
            }
            _ => {}
        }
    }

    let request = captured.await.unwrap().request;
    assert!(saw_done, "expected normally terminated stream");
    assert!(request.starts_with("POST /accounts/acct123/ai/v1/chat/completions "));
}

#[cfg(all(feature = "anthropic", feature = "cloudflare"))]
#[tokio::test]
async fn cloudflare_anthropic_placeholders_are_resolved_before_request() {
    let _lock = support::env_lock().lock().await;
    let _account = EnvVarGuard::set("CLOUDFLARE_ACCOUNT_ID", "acct123");
    let _gateway = EnvVarGuard::set("CLOUDFLARE_GATEWAY_ID", "gateway456");
    let (base, captured) = support::serve_capture_once(ANTHROPIC_SSE, "text/event-stream").await;
    let model = support::model(
        KnownApi::AnthropicMessages,
        "cloudflare-ai-gateway",
        "claude-test",
        format!("{base}/accounts/{{CLOUDFLARE_ACCOUNT_ID}}/{{CLOUDFLARE_GATEWAY_ID}}/anthropic"),
    );
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &Context::default(), Some(&options));
    let mut saw_done = false;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Done { .. } => saw_done = true,
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected stream error: {:?}", error.error_message)
            }
            _ => {}
        }
    }

    let request = captured.await.unwrap().request;
    assert!(saw_done, "expected normally terminated stream");
    assert!(request.starts_with("POST /accounts/acct123/gateway456/anthropic/v1/messages "));
}

#[cfg(all(feature = "openai-responses", feature = "cloudflare"))]
#[tokio::test]
async fn cloudflare_responses_placeholders_are_resolved_before_request() {
    let _lock = support::env_lock().lock().await;
    let _account = EnvVarGuard::set("CLOUDFLARE_ACCOUNT_ID", "acct123");
    let _gateway = EnvVarGuard::set("CLOUDFLARE_GATEWAY_ID", "gateway456");
    let (base, captured) = support::serve_capture_once(RESPONSES_SSE, "text/event-stream").await;
    let model = support::model(
        KnownApi::OpenAIResponses,
        "cloudflare-ai-gateway",
        "response-test",
        format!("{base}/accounts/{{CLOUDFLARE_ACCOUNT_ID}}/{{CLOUDFLARE_GATEWAY_ID}}/openai"),
    );
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &Context::default(), Some(&options));
    let mut saw_done = false;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Done { .. } => saw_done = true,
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected stream error: {:?}", error.error_message)
            }
            _ => {}
        }
    }

    let request = captured.await.unwrap().request;
    assert!(saw_done, "expected normally terminated stream");
    assert!(request.starts_with("POST /accounts/acct123/gateway456/openai/v1/responses "));
}

#[cfg(all(feature = "openai-completions", feature = "cloudflare"))]
#[tokio::test]
async fn cloudflare_missing_placeholder_emits_named_error() {
    const ENV_NAME: &str = "CLOUDFLARE_GATEWAY_ID";

    let _lock = support::env_lock().lock().await;
    let _env = EnvVarGuard::remove(ENV_NAME);
    let model = support::model(
        KnownApi::OpenAICompletions,
        "cloudflare-ai-gateway",
        "test-model",
        "http://127.0.0.1:1/{CLOUDFLARE_GATEWAY_ID}/v1".into(),
    );
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &Context::default(), Some(&options));
    let mut error_message = None;
    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::Error { error, .. } = event {
            error_message = error.error_message;
        }
    }

    assert!(
        error_message
            .as_deref()
            .is_some_and(|message| message.contains(ENV_NAME)),
        "error message: {error_message:?}"
    );
}
