//! Top-level streaming entry points. 1:1 port of `packages/ai/src/stream.ts`.
//!
//! The TS file does a side-effect import of `providers/register-builtins.js` to ensure
//! providers are registered before the first call. In Rust, feature-gated providers register
//! themselves on first use via [`crate::providers::register_builtins::ensure`].

use crate::api_registry::error_stream;
use crate::types::{AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions};
use crate::utils::event_stream::AssistantMessageEventStream;

pub use crate::env_api_keys::get_env_api_key;

#[cfg(test)]
fn resolve_test_barrier() -> &'static std::sync::Mutex<Option<std::sync::Arc<std::sync::Barrier>>> {
    use std::sync::{Arc, Barrier, Mutex, OnceLock};

    static CELL: OnceLock<Mutex<Option<Arc<Barrier>>>> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(None))
}

#[cfg(test)]
fn pause_resolve_for_test() {
    let barrier = resolve_test_barrier()
        .lock()
        .expect("resolve test barrier poisoned")
        .clone();
    if let Some(barrier) = barrier {
        barrier.wait();
        barrier.wait();
    }
}

fn resolve(model: &Model) -> Result<crate::api_registry::RegisteredHandle, String> {
    let handle = crate::providers::register_builtins::ensure_and_get(&model.api);
    #[cfg(test)]
    pause_resolve_for_test();
    handle.ok_or_else(|| format!("No API provider registered for api: {}", model.api.0))
}

pub fn stream(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> AssistantMessageEventStream {
    match resolve(model) {
        Ok(handle) => handle.stream(model, context, options),
        Err(msg) => error_stream(msg),
    }
}

pub async fn complete(
    model: &Model,
    context: &Context,
    options: Option<&StreamOptions>,
) -> Option<AssistantMessage> {
    stream(model, context, options).result().await
}

pub fn stream_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> AssistantMessageEventStream {
    match resolve(model) {
        Ok(handle) => handle.stream_simple(model, context, options),
        Err(msg) => error_stream(msg),
    }
}

pub async fn complete_simple(
    model: &Model,
    context: &Context,
    options: Option<&SimpleStreamOptions>,
) -> Option<AssistantMessage> {
    stream_simple(model, context, options).result().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api_registry::{clear_api_providers, registry_test_lock};
    use crate::types::{Api, Model, ModelCost, Provider};
    use std::sync::{Arc, Barrier};

    struct RestoreBuiltins;

    impl Drop for RestoreBuiltins {
        fn drop(&mut self) {
            clear_api_providers();
            crate::providers::register_builtins::ensure();
        }
    }

    struct ResolvePause;

    impl ResolvePause {
        fn install() -> Arc<Barrier> {
            let barrier = Arc::new(Barrier::new(2));
            *resolve_test_barrier()
                .lock()
                .expect("resolve test barrier poisoned") = Some(barrier.clone());
            barrier
        }
    }

    impl Drop for ResolvePause {
        fn drop(&mut self) {
            *resolve_test_barrier()
                .lock()
                .expect("resolve test barrier poisoned") = None;
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

    #[cfg(feature = "faux")]
    #[tokio::test]
    async fn clear_racing_with_stream_lookup_does_not_return_missing_provider_error() {
        let _guard = registry_test_lock().lock().await;
        let _restore = RestoreBuiltins;
        clear_api_providers();
        let _pause = ResolvePause;
        let barrier = ResolvePause::install();
        let model = faux_model();
        let resolve_task = tokio::task::spawn_blocking(move || resolve(&model));

        barrier.wait();
        clear_api_providers();
        barrier.wait();

        assert!(resolve_task.await.expect("resolve task panicked").is_ok());
    }
}
