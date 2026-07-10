mod support;

use ai::{
    AssistantMessageEvent, ContentBlock, Context, KnownApi, Message, StopReason, StreamOptions,
    UserContent, UserMessage, UserRole, stream,
};
use futures::StreamExt;

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

async fn assert_eof_is_error(api: KnownApi, provider: &str, base_url: String) {
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
        b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"usage\":{}}}\n\n",
        "text/event-stream",
    )
    .await;

    assert_eof_is_error(KnownApi::AnthropicMessages, "anthropic", base_url).await;
}

#[cfg(feature = "openai-completions")]
#[tokio::test]
async fn openai_completions_eof_before_done_or_finish_reason_is_error() {
    let base_url = support::serve_once(
        b"data: {\"id\":\"chatcmpl_1\",\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
        "text/event-stream",
    )
    .await;

    assert_eof_is_error(KnownApi::OpenAICompletions, "openai", base_url).await;
}

#[cfg(feature = "mistral")]
#[tokio::test]
async fn mistral_eof_before_done_or_finish_reason_is_error() {
    let base_url = support::serve_once(
        b"data: {\"id\":\"mistral_1\",\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
        "text/event-stream",
    )
    .await;

    assert_eof_is_error(KnownApi::MistralConversations, "mistral", base_url).await;
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
        b"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n",
        "text/event-stream",
    )
    .await;

    assert_eof_is_error(KnownApi::GoogleGenerativeAi, "google", base_url).await;
}

#[cfg(feature = "amazon-bedrock")]
#[tokio::test]
async fn bedrock_eof_before_message_stop_is_error() {
    let base_url = support::serve_once(b"", "application/vnd.amazon.eventstream").await;

    assert_eof_is_error(KnownApi::BedrockConverseStream, "amazon-bedrock", base_url).await;
}

#[cfg(feature = "openai-responses")]
#[tokio::test]
async fn openai_responses_eof_before_terminal_event_is_error() {
    let base_url = support::serve_once(
        b"data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}\n\n",
        "text/event-stream",
    )
    .await;

    assert_eof_is_error(KnownApi::OpenAIResponses, "openai", base_url).await;
}
