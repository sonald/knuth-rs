mod support;

#[cfg(feature = "openai-completions")]
use std::collections::HashMap;
#[cfg(all(
    feature = "cloudflare",
    any(
        feature = "openai-completions",
        feature = "anthropic",
        feature = "openai-responses"
    )
))]
use std::ffi::OsString;

#[cfg(any(
    feature = "openai-completions",
    all(
        feature = "cloudflare",
        any(feature = "anthropic", feature = "openai-responses")
    )
))]
use ai::{AssistantMessageEvent, Context, KnownApi, StreamOptions, stream};
#[cfg(any(
    feature = "openai-completions",
    all(
        feature = "cloudflare",
        any(feature = "anthropic", feature = "openai-responses")
    )
))]
use futures::StreamExt;

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

#[cfg(all(
    feature = "cloudflare",
    any(
        feature = "openai-completions",
        feature = "anthropic",
        feature = "openai-responses"
    )
))]
struct EnvVarGuard {
    name: &'static str,
    previous: Option<OsString>,
}

#[cfg(all(
    feature = "cloudflare",
    any(
        feature = "openai-completions",
        feature = "anthropic",
        feature = "openai-responses"
    )
))]
impl EnvVarGuard {
    fn set(name: &'static str, value: &str) -> Self {
        let previous = std::env::var_os(name);
        unsafe {
            std::env::set_var(name, value);
        }
        Self { name, previous }
    }

    fn remove(name: &'static str) -> Self {
        let previous = std::env::var_os(name);
        unsafe {
            std::env::remove_var(name);
        }
        Self { name, previous }
    }
}

#[cfg(all(
    feature = "cloudflare",
    any(
        feature = "openai-completions",
        feature = "anthropic",
        feature = "openai-responses"
    )
))]
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            match self.previous.take() {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }
}

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

#[cfg(feature = "openai-completions")]
fn header_values<'a>(request: &'a str, target: &str) -> Vec<&'a str> {
    request
        .lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case(target))
        .map(|(_, value)| value.trim())
        .collect()
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
