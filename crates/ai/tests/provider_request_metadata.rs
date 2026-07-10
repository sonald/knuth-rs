#![cfg(all(feature = "openai-completions", feature = "cloudflare"))]

mod support;

use std::collections::HashMap;

use ai::{AssistantMessageEvent, Context, KnownApi, StreamOptions, stream};
use futures::StreamExt;

const COMPLETIONS_SSE: &[u8] =
    br#"data: {"id":"response_1","choices":[{"finish_reason":"stop","delta":{}}]}

data: [DONE]

"#;

fn restore_env(name: &str, previous: Option<std::ffi::OsString>) {
    unsafe {
        match previous {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }
}

#[tokio::test]
async fn openai_completions_merges_model_headers_then_options_headers() {
    let (base_url, captured) =
        support::serve_capture_once(COMPLETIONS_SSE, "text/event-stream").await;
    let mut model = support::model(
        KnownApi::OpenAICompletions,
        "openai",
        "test-model",
        base_url,
    );
    model.headers = Some(HashMap::from([
        ("x-model-header".into(), "model".into()),
        ("x-shared-header".into(), "model".into()),
    ]));
    let options = StreamOptions {
        api_key: Some("test-key".into()),
        headers: Some(HashMap::from([
            ("x-options-header".into(), "options".into()),
            ("x-shared-header".into(), "options".into()),
        ])),
        ..Default::default()
    };

    let mut events = stream(&model, &Context::default(), Some(&options));
    while events.next().await.is_some() {}

    let request = captured.await.unwrap().request.to_ascii_lowercase();
    assert!(request.contains("\r\nx-model-header: model\r\n"));
    assert!(request.contains("\r\nx-options-header: options\r\n"));
    assert!(request.contains("\r\nx-shared-header: options\r\n"));
    assert!(!request.contains("\r\nx-shared-header: model\r\n"));
}

#[tokio::test]
async fn cloudflare_placeholders_are_resolved_before_request() {
    const ENV_NAME: &str = "CLOUDFLARE_ACCOUNT_ID";

    let _guard = support::env_lock().lock().await;
    let previous = std::env::var_os(ENV_NAME);
    unsafe {
        std::env::set_var(ENV_NAME, "acct123");
    }

    let result = async {
        let (base, captured) =
            support::serve_capture_once(COMPLETIONS_SSE, "text/event-stream").await;
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
        while events.next().await.is_some() {}
        captured.await
    }
    .await;

    restore_env(ENV_NAME, previous);
    let request = result.unwrap().request;
    assert!(request.starts_with("POST /accounts/acct123/ai/v1/chat/completions "));
}

#[tokio::test]
async fn cloudflare_missing_placeholder_emits_named_error() {
    const ENV_NAME: &str = "CLOUDFLARE_GATEWAY_ID";

    let _guard = support::env_lock().lock().await;
    let previous = std::env::var_os(ENV_NAME);
    unsafe {
        std::env::remove_var(ENV_NAME);
    }

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

    restore_env(ENV_NAME, previous);
    assert!(
        error_message
            .as_deref()
            .is_some_and(|message| message.contains(ENV_NAME)),
        "error message: {error_message:?}"
    );
}
