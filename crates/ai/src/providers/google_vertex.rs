//! Google Vertex AI provider (`google-vertex`). Partial 1:1 port of
//! `packages/ai/src/providers/google-vertex.ts` (~570 LOC).
//!
//! Reuses the Gemini message/tool conversion and SSE consumer from `google`/`google_shared`.
//! Vertex differs from the Gemini API in two ways:
//! - endpoint: `{base}/v1/projects/{project}/locations/{loc}/publishers/google/models/{id}:streamGenerateContent`
//! - auth: OAuth Bearer access token (service account / ADC) instead of `x-goog-api-key`
//!
//! Authentication prefers `options.api_key`, then `GOOGLE_VERTEX_ACCESS_TOKEN`, then the
//! service-account ADC JWT exchange in `vertex_adc`. Project selection prefers
//! `GOOGLE_VERTEX_PROJECT`, then the service account's `project_id`.

use async_trait::async_trait;

use crate::api_registry::ApiProvider;
use crate::providers::google::{
    build_request_body, consume_gemini_sse, push_error, translate_simple,
};
use crate::types::*;
use crate::utils::abort::{self as abort_utils, AbortErrorOrReqwest};
use crate::utils::event_stream::{AssistantMessageEventSender, AssistantMessageEventStream};

#[derive(Default)]
pub struct GoogleVertexProvider {}

#[async_trait]
impl ApiProvider for GoogleVertexProvider {
    fn api(&self) -> &str {
        KnownApi::GoogleVertex.as_str()
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
        self.stream(model, context, Some(&translate_simple(options)))
    }
}

fn vertex_location() -> String {
    std::env::var("GOOGLE_VERTEX_LOCATION").unwrap_or_else(|_| "us-central1".to_string())
}

fn vertex_host(location: &str) -> String {
    if location == "global" {
        "https://aiplatform.googleapis.com".to_string()
    } else {
        format!("https://{location}-aiplatform.googleapis.com")
    }
}

fn vertex_request_url(model: &Model, project: &str, location: &str) -> String {
    let base = model.base_url.trim_end_matches('/');
    let host = if base.is_empty() || base == "https://{location}-aiplatform.googleapis.com" {
        vertex_host(location)
    } else {
        base.replace("{location}", location)
    };
    format!(
        "{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{}:streamGenerateContent?alt=sse",
        model.id
    )
}

async fn run(
    model: Model,
    context: Context,
    options: StreamOptions,
    mut sender: AssistantMessageEventSender,
) {
    let explicit_project = std::env::var("GOOGLE_VERTEX_PROJECT")
        .ok()
        .filter(|project| !project.is_empty());
    let option_token = options.api_key.clone().filter(|token| !token.is_empty());
    let env_token = std::env::var("GOOGLE_VERTEX_ACCESS_TOKEN")
        .ok()
        .filter(|token| !token.is_empty());

    let (token, adc_project) = if let Some(token) = option_token {
        (token, None)
    } else if let Some(token) = env_token {
        (token, None)
    } else {
        let service_account = match crate::vertex_adc::load_service_account(None) {
            Ok(service_account) => service_account,
            Err(error) => {
                push_error(
                    &mut sender,
                    &model,
                    format!("Vertex ADC auth failed while loading credentials: {error}"),
                );
                return;
            }
        };
        let project = service_account
            .project_id
            .clone()
            .filter(|project| !project.is_empty());
        let access_token = match crate::vertex_adc::fetch_access_token_for_service_account(
            &service_account,
            None,
            &options,
        )
        .await
        {
            Ok(access_token) => access_token.token,
            Err(crate::vertex_adc::AdcExchangeError::Aborted) => {
                abort_utils::push_aborted(&mut sender, &model);
                return;
            }
            Err(crate::vertex_adc::AdcExchangeError::Adc(error)) => {
                push_error(
                    &mut sender,
                    &model,
                    format!("Vertex ADC auth failed during token exchange: {error}"),
                );
                return;
            }
        };
        (access_token, project)
    };

    let project = if let Some(project) = explicit_project {
        project
    } else if let Some(project) = adc_project {
        project
    } else {
        match crate::vertex_adc::load_service_account(None) {
            Ok(service_account) => match service_account
                .project_id
                .filter(|project| !project.is_empty())
            {
                Some(project) => project,
                None => {
                    push_error(
                        &mut sender,
                        &model,
                        "Vertex project missing: set GOOGLE_VERTEX_PROJECT or include project_id in the service-account credentials".into(),
                    );
                    return;
                }
            },
            Err(error) => {
                push_error(
                    &mut sender,
                    &model,
                    format!(
                        "Vertex project resolution failed while loading service-account credentials: {error}"
                    ),
                );
                return;
            }
        }
    };
    let location = vertex_location();

    let body = build_request_body(&context, &options);
    let client = match crate::utils::node_http_proxy::build_client(options.timeout_ms) {
        Ok(c) => c,
        Err(e) => {
            push_error(&mut sender, &model, format!("http client: {e}"));
            return;
        }
    };

    let url = vertex_request_url(&model, &project, &location);
    let mut req = client
        .post(&url)
        .bearer_auth(token)
        .header("content-type", "application/json")
        .header("accept", "text/event-stream");
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
        push_error(
            &mut sender,
            &model,
            format!("Vertex API error ({status}): {txt}"),
        );
        return;
    }

    consume_gemini_sse(resp, &model, sender, options.abort.as_ref()).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_model_url_resolves_regional_and_global_location() {
        let model = crate::get_model(&Provider::from("google-vertex"), "gemini-2.5-pro")
            .expect("built-in Vertex model");
        assert_eq!(
            model.base_url,
            "https://{location}-aiplatform.googleapis.com"
        );
        assert_eq!(
            vertex_request_url(&model, "test-project", "europe-west4"),
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/test-project/locations/europe-west4/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
        assert_eq!(
            vertex_request_url(&model, "test-project", "global"),
            "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global/publishers/google/models/gemini-2.5-pro:streamGenerateContent?alt=sse"
        );
    }
}
