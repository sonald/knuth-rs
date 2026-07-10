mod support;

use ai::{
    AssistantMessageEvent, Context, KnownApi, Message, StopReason, StreamOptions, UserContent,
    UserMessage, UserRole, stream,
};
#[cfg(feature = "mistral")]
use ai::{ContentBlock, SimpleStreamOptions, ThinkingLevel, stream_simple};
use futures::StreamExt;
#[cfg(feature = "google-vertex")]
use support::EnvVarGuard;

fn context() -> Context {
    Context {
        system_prompt: None,
        messages: vec![Message::User(UserMessage {
            role: UserRole::User,
            content: UserContent::Text("hello".into()),
            timestamp: 0,
        })],
        tools: None,
    }
}

#[cfg(any(feature = "google", feature = "amazon-bedrock", feature = "mistral"))]
async fn terminal_event(
    api: KnownApi,
    provider: &str,
    model_id: &str,
    base_url: String,
) -> AssistantMessageEvent {
    let model = support::model(api, provider, model_id, base_url);
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &context(), Some(&options));
    let mut terminal = None;
    while let Some(event) = events.next().await {
        if event.is_terminal() {
            assert!(
                terminal.is_none(),
                "provider emitted multiple terminal events"
            );
            terminal = Some(event);
        }
    }
    terminal.expect("provider terminal event")
}

async fn assert_eof_is_error(
    api: KnownApi,
    provider: &str,
    base_url: String,
    expect_usage_cost: bool,
) {
    let model = support::model(api, provider, "test-model", base_url);
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &context(), Some(&options));
    let mut saw_error = false;
    let mut saw_done = false;

    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Error { error, .. } => {
                saw_error = true;
                assert_eq!(error.stop_reason, StopReason::Error);
                assert!(
                    error
                        .error_message
                        .as_deref()
                        .unwrap_or_default()
                        .contains("stream ended before terminal event"),
                    "error message: {:?}",
                    error.error_message
                );
                if expect_usage_cost {
                    assert!(error.usage.cost.total > 0.0);
                }
            }
            AssistantMessageEvent::Done { .. } => saw_done = true,
            _ => {}
        }
    }

    assert!(saw_error, "expected an error event");
    assert!(!saw_done, "unexpected Done event");
}

#[cfg(feature = "anthropic")]
#[tokio::test]
async fn anthropic_eof_before_message_stop_is_error() {
    let base_url = support::serve_once(
        b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{\"input_tokens\":10}}}\n\n",
        "text/event-stream",
    )
    .await;

    assert_eof_is_error(KnownApi::AnthropicMessages, "anthropic", base_url, true).await;
}

#[cfg(feature = "openai-completions")]
#[tokio::test]
async fn openai_completions_eof_before_done_or_finish_reason_is_error() {
    let base_url = support::serve_once(
        b"data: {\"id\":\"chatcmpl_1\",\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1}}\n\n",
        "text/event-stream",
    )
    .await;

    assert_eof_is_error(KnownApi::OpenAICompletions, "openai", base_url, true).await;
}

#[cfg(feature = "openai-completions")]
#[tokio::test]
async fn openai_completions_done_usage_has_nonzero_cost() {
    let base_url = support::serve_once(
        br#"data: {"id":"chatcmpl_1","choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":100,"completion_tokens":10,"prompt_tokens_details":{"cached_tokens":80}}}

data: [DONE]

"#,
        "text/event-stream",
    )
    .await;
    let model = support::model(
        KnownApi::OpenAICompletions,
        "openai",
        "test-model",
        base_url,
    );
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &context(), Some(&options));
    let mut message = None;

    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::Done { message: done, .. } = event {
            message = Some(done);
        }
    }

    let message = message.expect("expected Done event");
    assert!(message.usage.cost.total > 0.0);
    assert_eq!(message.usage.input, 20);
    assert_eq!(message.usage.cache_read, 80);
}

#[cfg(feature = "mistral")]
#[tokio::test]
async fn mistral_eof_before_done_or_finish_reason_is_error() {
    let base_url = support::serve_once(
        b"data: {\"id\":\"mistral_1\",\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":1}}\n\n",
        "text/event-stream",
    )
    .await;

    assert_eof_is_error(KnownApi::MistralConversations, "mistral", base_url, true).await;
}

#[cfg(feature = "mistral")]
fn mistral_terminal_sse(reason: &str) -> &'static [u8] {
    let payload = serde_json::json!({
        "id": "mistral-terminal",
        "choices": [{ "delta": {}, "finish_reason": reason }]
    });
    Box::leak(
        format!("data: {payload}\n\ndata: [DONE]\n\n")
            .into_bytes()
            .into_boxed_slice(),
    )
}

#[cfg(feature = "mistral")]
#[tokio::test]
async fn mistral_finish_reasons_keep_success_mappings_and_fail_closed() {
    for (provider_reason, expected_reason, expected_stop) in [
        ("stop", ai::DoneReason::Stop, StopReason::Stop),
        ("length", ai::DoneReason::Length, StopReason::Length),
        ("model_length", ai::DoneReason::Length, StopReason::Length),
        ("tool_calls", ai::DoneReason::ToolUse, StopReason::ToolUse),
    ] {
        let base_url =
            support::serve_once(mistral_terminal_sse(provider_reason), "text/event-stream").await;
        match terminal_event(
            KnownApi::MistralConversations,
            "mistral",
            "mistral-test",
            base_url,
        )
        .await
        {
            AssistantMessageEvent::Done { reason, message } => {
                assert_eq!(reason, expected_reason, "provider reason {provider_reason}");
                assert_eq!(
                    message.stop_reason, expected_stop,
                    "provider reason {provider_reason}"
                );
            }
            event => panic!("expected Done for {provider_reason}, got {event:?}"),
        }
    }

    for provider_reason in ["error", "future_mistral_failure"] {
        let base_url =
            support::serve_once(mistral_terminal_sse(provider_reason), "text/event-stream").await;
        match terminal_event(
            KnownApi::MistralConversations,
            "mistral",
            "mistral-test",
            base_url,
        )
        .await
        {
            AssistantMessageEvent::Error { error, .. } => {
                assert_eq!(error.stop_reason, StopReason::Error);
                assert!(
                    error
                        .error_message
                        .as_deref()
                        .is_some_and(|message| message.contains(provider_reason)),
                    "raw finish reason was not retained: {:?}",
                    error.error_message
                );
            }
            event => panic!("expected Error for {provider_reason}, got {event:?}"),
        }
    }
}

#[cfg(feature = "mistral")]
#[tokio::test]
async fn mistral_stream_simple_preserves_interleaved_reasoning_and_text_chunks() {
    const BODY: &[u8] = br#"data: {"id":"mistral-reasoning","choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"plan "}]},{"type":"text","text":"answer "}]},"finish_reason":null}]}

data: {"id":"mistral-reasoning","choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"reconsider"}]},{"type":"text","text":"done"}]},"finish_reason":null}]}

data: {"id":"mistral-reasoning","choices":[{"delta":{"content":"tail"},"finish_reason":"stop"}]}

data: [DONE]

"#;
    let (base_url, captured) = support::serve_capture_once(BODY, "text/event-stream").await;
    let mut model = support::model(
        KnownApi::MistralConversations,
        "mistral",
        "mistral-small-latest",
        base_url,
    );
    model.reasoning = true;
    let options = SimpleStreamOptions {
        base: StreamOptions {
            api_key: Some("test-key".into()),
            ..Default::default()
        },
        reasoning: Some(ThinkingLevel::High),
        ..Default::default()
    };
    let mut events = stream_simple(&model, &context(), Some(&options));
    let mut starts = Vec::new();
    let mut deltas = Vec::new();
    let mut ends = Vec::new();
    let mut message = None;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::ThinkingStart { content_index, .. } => {
                starts.push(("thinking", content_index));
            }
            AssistantMessageEvent::TextStart { content_index, .. } => {
                starts.push(("text", content_index));
            }
            AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                ..
            } => deltas.push(("thinking", content_index, delta)),
            AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                ..
            } => deltas.push(("text", content_index, delta)),
            AssistantMessageEvent::ThinkingEnd { content_index, .. } => {
                ends.push(("thinking", content_index));
            }
            AssistantMessageEvent::TextEnd { content_index, .. } => {
                ends.push(("text", content_index));
            }
            AssistantMessageEvent::Done { message: done, .. } => message = Some(done),
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected Mistral error: {:?}", error.error_message)
            }
            _ => {}
        }
    }

    assert_eq!(
        starts,
        vec![("thinking", 0), ("text", 1), ("thinking", 2), ("text", 3)]
    );
    assert_eq!(
        deltas,
        vec![
            ("thinking", 0, "plan ".into()),
            ("text", 1, "answer ".into()),
            ("thinking", 2, "reconsider".into()),
            ("text", 3, "done".into()),
            ("text", 3, "tail".into()),
        ]
    );
    assert_eq!(
        ends,
        vec![("thinking", 0), ("text", 1), ("thinking", 2), ("text", 3)]
    );
    let message = message.expect("Mistral Done event");
    assert!(matches!(
        message.content.as_slice(),
        [
            ContentBlock::Thinking(first),
            ContentBlock::Text(first_text),
            ContentBlock::Thinking(second),
            ContentBlock::Text(second_text),
        ] if first.thinking == "plan "
            && first_text.text == "answer "
            && second.thinking == "reconsider"
            && second_text.text == "donetail"
    ));

    let request = captured.await.expect("captured Mistral request").request;
    let body: serde_json::Value = serde_json::from_str(
        request
            .split_once("\r\n\r\n")
            .expect("Mistral request body")
            .1,
    )
    .unwrap();
    assert_eq!(body["reasoning_effort"], "high");
}

#[cfg(feature = "mistral")]
#[tokio::test]
async fn mistral_parallel_tool_calls_do_not_merge_arguments() {
    let base_url = support::serve_once(
        br#"data: {"id":"mistral_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"alpha1234","function":{"name":"a","arguments":"{\"x\":1}"}},{"index":1,"id":"bravo5678","function":{"name":"b","arguments":"{\"y\":2}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]

"#,
        "text/event-stream",
    )
    .await;
    let model = support::model(
        KnownApi::MistralConversations,
        "mistral",
        "test-model",
        base_url,
    );
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &context(), Some(&options));
    let mut message = None;

    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::Done { message: done, .. } = event {
            message = Some(done);
        }
    }

    let message = message.expect("expected Done event");
    assert_eq!(message.content.len(), 2);
    assert!(matches!(
        &message.content[0],
        ContentBlock::ToolCall(call)
            if call.id == "alpha1234" && call.name == "a"
                && call.arguments.get("x") == Some(&serde_json::json!(1))
    ));
    assert!(matches!(
        &message.content[1],
        ContentBlock::ToolCall(call)
            if call.id == "bravo5678" && call.name == "b"
                && call.arguments.get("y") == Some(&serde_json::json!(2))
    ));
}

#[cfg(feature = "mistral")]
#[tokio::test]
async fn mistral_parallel_tool_call_events_preserve_content_indices() {
    let base_url = support::serve_once(
        br#"data: {"id":"mistral_1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call-a-too-long","function":{"name":"a"}},{"index":1,"id":"bravo5678","function":{"name":"b"}}]},"finish_reason":null}]}

data: {"id":"mistral_1","choices":[{"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"y\":"}},{"index":0,"function":{"arguments":"{\"x\":"}}]},"finish_reason":null}]}

data: {"id":"mistral_1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"1}"}},{"index":1,"function":{"arguments":"2}"}}]},"finish_reason":"tool_calls"}]}

data: [DONE]

"#,
        "text/event-stream",
    )
    .await;
    let model = support::model(
        KnownApi::MistralConversations,
        "mistral",
        "test-model",
        base_url,
    );
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &context(), Some(&options));
    let mut starts = Vec::new();
    let mut deltas = Vec::new();
    let mut ends = Vec::new();
    let mut message = None;

    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::ToolCallStart { content_index, .. } => {
                starts.push(content_index)
            }
            AssistantMessageEvent::ToolCallDelta {
                content_index,
                delta,
                ..
            } => deltas.push((content_index, delta)),
            AssistantMessageEvent::ToolCallEnd {
                content_index,
                tool_call,
                ..
            } => ends.push((content_index, tool_call)),
            AssistantMessageEvent::Done { message: done, .. } => message = Some(done),
            _ => {}
        }
    }

    assert_eq!(starts, vec![0, 1]);
    assert_eq!(
        deltas,
        vec![
            (1, "{\"y\":".to_string()),
            (0, "{\"x\":".to_string()),
            (0, "1}".to_string()),
            (1, "2}".to_string()),
        ]
    );
    assert_eq!(
        ends.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert_eq!(ends[1].1.id, "bravo5678");
    assert_eq!(ends[0].1.name, "a");
    assert_eq!(ends[0].1.arguments.get("x"), Some(&serde_json::json!(1)));
    assert_eq!(ends[1].1.arguments.get("y"), Some(&serde_json::json!(2)));
    assert_eq!(ends[0].1.id.len(), 9);
    assert!(ends[0].1.id.chars().all(|ch| ch.is_ascii_alphanumeric()));

    let message = message.expect("expected Done event");
    assert!(matches!(
        &message.content[0],
        ContentBlock::ToolCall(call)
            if call.name == "a"
                && call.arguments.get("x") == Some(&serde_json::json!(1))
                && call.id.len() == 9
                && call.id.chars().all(|ch| ch.is_ascii_alphanumeric())
    ));
    assert!(matches!(
        &message.content[1],
        ContentBlock::ToolCall(call)
            if call.id == "bravo5678" && call.name == "b"
                && call.arguments.get("y") == Some(&serde_json::json!(2))
    ));
}

#[cfg(feature = "google")]
#[tokio::test]
async fn google_eof_before_finish_reason_is_error() {
    let base_url = support::serve_once(
        b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":1}}\n\n",
        "text/event-stream",
    )
    .await;

    assert_eof_is_error(KnownApi::GoogleGenerativeAi, "google", base_url, true).await;
}

#[cfg(feature = "google")]
fn google_terminal_sse(reason: &str, with_tool_call: bool) -> &'static [u8] {
    let parts = if with_tool_call {
        serde_json::json!([{
            "functionCall": { "name": "lookup", "args": {} },
            "thoughtSignature": "dG9vbA=="
        }])
    } else {
        serde_json::json!([{ "text": "answer" }])
    };
    let payload = serde_json::json!({
        "candidates": [{
            "content": { "parts": parts },
            "finishReason": reason
        }]
    });
    Box::leak(
        format!("data: {payload}\n\n")
            .into_bytes()
            .into_boxed_slice(),
    )
}

#[cfg(feature = "google")]
#[tokio::test]
async fn google_finish_reasons_keep_success_mappings_and_fail_closed() {
    for (provider_reason, with_tool_call, expected_reason, expected_stop) in [
        ("STOP", false, ai::DoneReason::Stop, StopReason::Stop),
        (
            "MAX_TOKENS",
            false,
            ai::DoneReason::Length,
            StopReason::Length,
        ),
        (
            "MAX_TOKENS",
            true,
            ai::DoneReason::Length,
            StopReason::Length,
        ),
        ("STOP", true, ai::DoneReason::ToolUse, StopReason::ToolUse),
    ] {
        let base_url = support::serve_once(
            google_terminal_sse(provider_reason, with_tool_call),
            "text/event-stream",
        )
        .await;
        match terminal_event(
            KnownApi::GoogleGenerativeAi,
            "google",
            "gemini-test",
            base_url,
        )
        .await
        {
            AssistantMessageEvent::Done { reason, message } => {
                assert_eq!(reason, expected_reason, "provider reason {provider_reason}");
                assert_eq!(
                    message.stop_reason, expected_stop,
                    "provider reason {provider_reason}"
                );
            }
            event => panic!("expected Done for {provider_reason}, got {event:?}"),
        }
    }

    for provider_reason in [
        "MALFORMED_FUNCTION_CALL",
        "MISSING_THOUGHT_SIGNATURE",
        "FUTURE_GEMINI_FAILURE",
    ] {
        let base_url = support::serve_once(
            google_terminal_sse(
                provider_reason,
                provider_reason == "MALFORMED_FUNCTION_CALL",
            ),
            "text/event-stream",
        )
        .await;
        match terminal_event(
            KnownApi::GoogleGenerativeAi,
            "google",
            "gemini-test",
            base_url,
        )
        .await
        {
            AssistantMessageEvent::Error { error, .. } => {
                assert_eq!(error.stop_reason, StopReason::Error);
                assert!(
                    error
                        .error_message
                        .as_deref()
                        .is_some_and(|message| message.contains(provider_reason)),
                    "raw finish reason was not retained: {:?}",
                    error.error_message
                );
            }
            event => panic!("expected Error for {provider_reason}, got {event:?}"),
        }
    }
}

#[cfg(feature = "google-vertex")]
#[tokio::test]
async fn vertex_wrapper_done_usage_has_nonzero_cost() {
    let _lock = support::env_lock().lock().await;
    let _project = EnvVarGuard::set("GOOGLE_VERTEX_PROJECT", "usage-project");
    let _location = EnvVarGuard::set("GOOGLE_VERTEX_LOCATION", "us-central1");
    let base_url = support::serve_once(
        br#"data: {"responseId":"vertex-usage","candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":20,"toolUsePromptTokenCount":5,"candidatesTokenCount":10,"thoughtsTokenCount":5}}

"#,
        "text/event-stream",
    )
    .await;
    let model = support::model(
        KnownApi::GoogleVertex,
        "google-vertex",
        "gemini-test",
        base_url,
    );
    let options = StreamOptions {
        api_key: Some("vertex-test-token".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &context(), Some(&options));
    let mut message = None;
    while let Some(event) = events.next().await {
        match event {
            AssistantMessageEvent::Done { message: done, .. } => message = Some(done),
            AssistantMessageEvent::Error { error, .. } => {
                panic!("unexpected Vertex error: {:?}", error.error_message)
            }
            _ => {}
        }
    }

    let message = message.expect("expected Done event from Vertex wrapper");
    assert_eq!(message.usage.input, 85);
    assert_eq!(message.usage.output, 15);
    assert_eq!(message.usage.cache_read, 20);
    assert_eq!(message.usage.cache_write, 0);
    assert_eq!(message.usage.total_tokens, 120);
    assert!((message.usage.cost.input - 0.000_085).abs() < 1e-12);
    assert!((message.usage.cost.output - 0.000_030).abs() < 1e-12);
    assert!((message.usage.cost.cache_read - 0.000_005).abs() < 1e-12);
    assert!((message.usage.cost.total - 0.000_120).abs() < 1e-12);
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_eof_before_message_stop_is_error() {
    let body = support::aws_eventstream_frame(
        "metadata",
        br#"{"usage":{"inputTokens":10,"outputTokens":1}}"#,
    );
    let base_url = support::serve_once(
        Box::leak(body.into_boxed_slice()),
        "application/vnd.amazon.eventstream",
    )
    .await;

    assert_eof_is_error(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        base_url,
        true,
    )
    .await;
}

#[cfg(feature = "amazon-bedrock")]
fn bedrock_terminal_body(reason: &str) -> &'static [u8] {
    let payload = serde_json::to_vec(&serde_json::json!({ "stopReason": reason })).unwrap();
    Box::leak(support::aws_eventstream_frame("messageStop", &payload).into_boxed_slice())
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_stop_reasons_keep_success_mappings_and_fail_closed() {
    for (provider_reason, expected_reason, expected_stop) in [
        ("end_turn", ai::DoneReason::Stop, StopReason::Stop),
        ("stop_sequence", ai::DoneReason::Stop, StopReason::Stop),
        ("max_tokens", ai::DoneReason::Length, StopReason::Length),
        (
            "model_context_window_exceeded",
            ai::DoneReason::Length,
            StopReason::Length,
        ),
        ("tool_use", ai::DoneReason::ToolUse, StopReason::ToolUse),
    ] {
        let base_url = support::serve_once(
            bedrock_terminal_body(provider_reason),
            "application/vnd.amazon.eventstream",
        )
        .await;
        match terminal_event(
            KnownApi::BedrockConverseStream,
            "amazon-bedrock",
            "bedrock-test",
            base_url,
        )
        .await
        {
            AssistantMessageEvent::Done { reason, message } => {
                assert_eq!(reason, expected_reason, "provider reason {provider_reason}");
                assert_eq!(
                    message.stop_reason, expected_stop,
                    "provider reason {provider_reason}"
                );
            }
            event => panic!("expected Done for {provider_reason}, got {event:?}"),
        }
    }

    for provider_reason in [
        "content_filtered",
        "guardrail_intervened",
        "malformed_tool_use",
        "future_bedrock_failure",
    ] {
        let base_url = support::serve_once(
            bedrock_terminal_body(provider_reason),
            "application/vnd.amazon.eventstream",
        )
        .await;
        match terminal_event(
            KnownApi::BedrockConverseStream,
            "amazon-bedrock",
            "bedrock-test",
            base_url,
        )
        .await
        {
            AssistantMessageEvent::Error { error, .. } => {
                assert_eq!(error.stop_reason, StopReason::Error);
                assert!(
                    error
                        .error_message
                        .as_deref()
                        .is_some_and(|message| message.contains(provider_reason)),
                    "raw stop reason was not retained: {:?}",
                    error.error_message
                );
            }
            event => panic!("expected Error for {provider_reason}, got {event:?}"),
        }
    }
}

#[cfg(feature = "openai-responses")]
#[tokio::test]
async fn openai_responses_eof_before_terminal_event_is_error() {
    let base_url = support::serve_once(
        b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
        "text/event-stream",
    )
    .await;

    assert_eof_is_error(KnownApi::OpenAIResponses, "openai", base_url, false).await;
}

#[cfg(feature = "openai-responses")]
#[tokio::test]
async fn openai_responses_done_usage_has_nonzero_cost() {
    let base_url = support::serve_once(
        br#"data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":100,"output_tokens":10,"input_tokens_details":{"cached_tokens":80}}}}

"#,
        "text/event-stream",
    )
    .await;
    let model = support::model(KnownApi::OpenAIResponses, "openai", "test-model", base_url);
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &context(), Some(&options));
    let mut message = None;

    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::Done { message: done, .. } = event {
            message = Some(done);
        }
    }

    let message = message.expect("expected Done event");
    assert!(message.usage.cost.total > 0.0);
    assert_eq!(message.usage.input, 20);
    assert_eq!(message.usage.cache_read, 80);
}

#[cfg(feature = "openai-codex-responses")]
#[tokio::test]
async fn codex_responses_wrapper_done_usage_has_nonzero_cost() {
    let base_url = support::serve_once(
        br#"data: {"type":"response.completed","response":{"id":"resp_codex","status":"completed","output":[],"usage":{"input_tokens":100,"output_tokens":10,"input_tokens_details":{"cached_tokens":80,"cache_write_tokens":20}}}}

"#,
        "text/event-stream",
    )
    .await;
    let model = support::model(
        KnownApi::OpenAICodexResponses,
        "openai-codex",
        "gpt-5-codex",
        base_url,
    );
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &context(), Some(&options));
    let mut message = None;

    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::Done { message: done, .. } = event {
            message = Some(done);
        }
    }

    let message = message.expect("expected Done event from Codex wrapper");
    assert_eq!(message.usage.input, 0);
    assert_eq!(message.usage.output, 10);
    assert_eq!(message.usage.cache_read, 80);
    assert_eq!(message.usage.cache_write, 20);
    assert_eq!(message.usage.total_tokens, 110);
    assert!(message.usage.cost.total > 0.0);
}

#[cfg(feature = "azure-openai-responses")]
#[tokio::test]
async fn azure_openai_responses_wrapper_done_usage_has_nonzero_cost() {
    let base_url = support::serve_once(
        br#"data: {"type":"response.completed","response":{"id":"resp_azure","status":"completed","output":[],"usage":{"input_tokens":100,"output_tokens":10,"input_tokens_details":{"cached_tokens":80,"cache_write_tokens":20}}}}

"#,
        "text/event-stream",
    )
    .await;
    let model = support::model(
        KnownApi::AzureOpenAIResponses,
        "azure-openai-responses",
        "test-deployment",
        base_url,
    );
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        ..Default::default()
    };
    let mut events = stream(&model, &context(), Some(&options));
    let mut message = None;

    while let Some(event) = events.next().await {
        if let AssistantMessageEvent::Done { message: done, .. } = event {
            message = Some(done);
        }
    }

    let message = message.expect("expected Done event from Azure wrapper");
    assert_eq!(message.usage.input, 0);
    assert_eq!(message.usage.output, 10);
    assert_eq!(message.usage.cache_read, 80);
    assert_eq!(message.usage.cache_write, 20);
    assert_eq!(message.usage.total_tokens, 110);
    assert!(message.usage.cost.total > 0.0);
}
