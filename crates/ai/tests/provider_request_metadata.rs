mod support;

#[cfg(any(feature = "openai-completions", feature = "google-vertex"))]
use std::collections::HashMap;

#[cfg(feature = "openai-codex-responses")]
use ai::{
    Api, AssistantMessage, ContentBlock, Message, Provider, StopReason, ThinkingContent, Usage,
};
#[cfg(any(
    feature = "openai-completions",
    feature = "openai-codex-responses",
    feature = "amazon-bedrock",
    feature = "google-vertex",
    all(
        feature = "cloudflare",
        any(feature = "anthropic", feature = "openai-responses")
    )
))]
use ai::{AssistantMessageEvent, Context, KnownApi, StreamOptions, stream};
#[cfg(any(
    feature = "openai-completions",
    feature = "openai-codex-responses",
    feature = "amazon-bedrock",
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

#[cfg(any(feature = "amazon-bedrock", feature = "google-vertex"))]
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
    let signed_url =
        url::Url::parse(&format!("{base_url}/model/test-model/converse-stream")).unwrap();
    let model = support::model(
        KnownApi::BedrockConverseStream,
        "amazon-bedrock",
        "test-model",
        base_url,
    );
    let context = Context {
        system_prompt: None,
        messages: vec![ai::Message::User(ai::UserMessage {
            role: ai::UserRole::User,
            content: ai::UserContent::Text("signed body".into()),
            timestamp: 0,
        })],
        tools: None,
    };
    let options = StreamOptions::default();

    stream_to_done(&model, &context, &options).await;
    let request = captured.await.unwrap().request;
    assert!(request.starts_with("POST /model/test-model/converse-stream HTTP/1.1\r\n"));
    assert_eq!(
        header_values(&request, "content-type"),
        ["application/json"]
    );
    assert_eq!(
        header_values(&request, "accept"),
        ["application/vnd.amazon.eventstream"]
    );
    assert_eq!(
        header_values(&request, "x-amz-security-token"),
        ["session-token"]
    );
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
    let expected_signature = ai::sigv4::sign(&ai::sigv4::SigningRequest {
        method: "POST",
        url: &signed_url,
        headers: &[],
        payload: body_text.as_bytes(),
        region: "us-west-2",
        service: "bedrock",
        access_key: "AKIDEXAMPLE",
        secret_key: "secret",
        session_token: Some("session-token"),
        amz_date: amz_date[0],
    });
    assert_eq!(
        header_values(&request, "authorization"),
        [expected_signature.authorization]
    );

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

#[cfg(feature = "google-vertex")]
const VERTEX_DONE_SSE: &[u8] = br#"data: {"responseId":"vertex-response","candidates":[{"content":{"parts":[{"text":"vertex ok"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":20,"toolUsePromptTokenCount":5,"candidatesTokenCount":10,"thoughtsTokenCount":5}}

"#;

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
        support::serve_capture_once(VERTEX_DONE_SSE, "text/event-stream").await;
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
        support::serve_capture_once(VERTEX_DONE_SSE, "text/event-stream").await;
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
        support::serve_capture_once(VERTEX_DONE_SSE, "text/event-stream").await;
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
        support::serve_capture_once(VERTEX_DONE_SSE, "text/event-stream").await;
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
