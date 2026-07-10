//! Lazy provider registration. 1:1 port of `packages/ai/src/providers/register-builtins.ts`.
//!
//! The TS file uses side-effect imports + lazy `async () => import("./anthropic.js")` wrappers
//! so importing `pie-ai` doesn't drag every provider's SDK into cold start. Rust uses Cargo
//! features and registry insertion-if-absent so every ensure call restores missing built-ins
//! without replacing a custom provider for the same API.

use std::sync::{Mutex, OnceLock};

fn provider_lifecycle_lock() -> &'static Mutex<()> {
    static CELL: OnceLock<Mutex<()>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(()))
}

pub(crate) fn with_provider_lifecycle<T>(f: impl FnOnce() -> T) -> T {
    let _guard = provider_lifecycle_lock()
        .lock()
        .expect("provider lifecycle lock poisoned");
    f()
}

/// Register all enabled built-in providers. Idempotent.
pub fn ensure() {
    with_provider_lifecycle(register_enabled);
}

pub(crate) fn ensure_and_get(
    api: &crate::types::Api,
) -> Option<crate::api_registry::RegisteredHandle> {
    with_provider_lifecycle(|| {
        register_enabled();
        crate::api_registry::get_api_provider(api)
    })
}

fn register_enabled() {
    #[cfg(feature = "anthropic")]
    crate::api_registry::register_api_provider_if_absent(
        Box::new(crate::providers::anthropic::AnthropicProvider::default()),
        Some("builtin".into()),
    );

    #[cfg(feature = "faux")]
    crate::api_registry::register_api_provider_if_absent(
        Box::new(crate::providers::faux::FauxProvider::default()),
        Some("builtin".into()),
    );

    #[cfg(feature = "openai-responses")]
    crate::api_registry::register_api_provider_if_absent(
        Box::new(crate::providers::openai_responses::OpenAIResponsesProvider::default()),
        Some("builtin".into()),
    );

    #[cfg(feature = "openai-completions")]
    crate::api_registry::register_api_provider_if_absent(
        Box::new(crate::providers::openai_completions::OpenAICompletionsProvider::default()),
        Some("builtin".into()),
    );

    #[cfg(feature = "google")]
    crate::api_registry::register_api_provider_if_absent(
        Box::new(crate::providers::google::GoogleProvider::default()),
        Some("builtin".into()),
    );

    #[cfg(feature = "mistral")]
    crate::api_registry::register_api_provider_if_absent(
        Box::new(crate::providers::mistral::MistralProvider::default()),
        Some("builtin".into()),
    );

    #[cfg(feature = "azure-openai-responses")]
    crate::api_registry::register_api_provider_if_absent(
        Box::new(crate::providers::azure_openai_responses::AzureOpenAIResponsesProvider::default()),
        Some("builtin".into()),
    );

    #[cfg(feature = "google-vertex")]
    crate::api_registry::register_api_provider_if_absent(
        Box::new(crate::providers::google_vertex::GoogleVertexProvider::default()),
        Some("builtin".into()),
    );

    #[cfg(feature = "amazon-bedrock")]
    crate::api_registry::register_api_provider_if_absent(
        Box::new(crate::providers::amazon_bedrock::AmazonBedrockProvider::default()),
        Some("builtin".into()),
    );

    #[cfg(feature = "openai-codex-responses")]
    crate::api_registry::register_api_provider_if_absent(
        Box::new(crate::providers::openai_codex_responses::OpenAICodexResponsesProvider::default()),
        Some("builtin".into()),
    );

    // TODO: register remaining providers as their implementations land.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_registry::{
        ApiProvider, clear_api_providers, get_api_provider, list_api_ids, register_api_provider,
        registry_test_lock,
    };
    use crate::types::{
        Api, AssistantMessageEvent, Context, Model, ModelCost, Provider, SimpleStreamOptions,
        StreamOptions,
    };
    use crate::utils::event_stream::AssistantMessageEventStream;
    use futures::StreamExt;

    struct RestoreBuiltins;

    impl Drop for RestoreBuiltins {
        fn drop(&mut self) {
            clear_api_providers();
            ensure();
        }
    }

    fn faux_model() -> Model {
        Model {
            id: "faux-model".into(),
            name: "Faux Model".into(),
            api: Api::from("faux"),
            provider: Provider::from("faux"),
            base_url: String::new(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![],
            cost: ModelCost::default(),
            context_window: 1,
            max_tokens: 1,
            headers: None,
            compat: None,
        }
    }

    struct CustomProvider {
        api: &'static str,
    }

    #[async_trait::async_trait]
    impl ApiProvider for CustomProvider {
        fn api(&self) -> &str {
            self.api
        }

        fn stream(
            &self,
            _model: &Model,
            _context: &Context,
            _options: Option<&StreamOptions>,
        ) -> AssistantMessageEventStream {
            crate::api_registry::error_stream("custom provider".into())
        }

        fn stream_simple(
            &self,
            _model: &Model,
            _context: &Context,
            _options: Option<&SimpleStreamOptions>,
        ) -> AssistantMessageEventStream {
            crate::api_registry::error_stream("custom provider".into())
        }
    }

    #[cfg(feature = "faux")]
    #[tokio::test]
    async fn stream_re_registers_builtins_after_clear() {
        let _guard = registry_test_lock().lock().await;
        let _restore = RestoreBuiltins;
        ensure();
        clear_api_providers();

        let mut stream = crate::stream::stream(&faux_model(), &Context::default(), None);
        assert!(matches!(
            stream.next().await,
            Some(AssistantMessageEvent::Start { .. })
        ));
    }

    #[tokio::test]
    async fn ensure_restores_missing_builtins_when_registry_contains_custom_provider() {
        let _guard = registry_test_lock().lock().await;
        let _restore = RestoreBuiltins;
        clear_api_providers();
        ensure();
        let built_in_apis = list_api_ids();

        clear_api_providers();
        register_api_provider(
            Box::new(CustomProvider { api: "custom-api" }),
            Some("custom".into()),
        );
        ensure();

        assert!(get_api_provider(&Api::from("custom-api")).is_some());
        let restored_apis = list_api_ids();
        assert!(built_in_apis.iter().all(|api| restored_apis.contains(api)));
    }

    #[cfg(feature = "faux")]
    #[tokio::test]
    async fn ensure_preserves_custom_override_for_builtin_api() {
        let _guard = registry_test_lock().lock().await;
        let _restore = RestoreBuiltins;
        clear_api_providers();
        register_api_provider(
            Box::new(CustomProvider { api: "faux" }),
            Some("custom".into()),
        );
        ensure();

        let mut stream = crate::stream::stream(&faux_model(), &Context::default(), None);
        assert!(matches!(
            stream.next().await,
            Some(AssistantMessageEvent::Error { error, .. })
                if error.error_message.as_deref() == Some("custom provider")
        ));
    }
}
