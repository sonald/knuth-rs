//! Top-level streaming entry points. 1:1 port of `packages/ai/src/stream.ts`.
//!
//! The TS file does a side-effect import of `providers/register-builtins.js` to ensure
//! providers are registered before the first call. In Rust, feature-gated providers register
//! themselves on first use via [`crate::providers::register_builtins::ensure`].

use crate::api_registry::error_stream;
use crate::types::{AssistantMessage, Context, Model, SimpleStreamOptions, StreamOptions};
use crate::utils::event_stream::AssistantMessageEventStream;

pub use crate::env_api_keys::get_env_api_key;

fn resolve(model: &Model) -> Result<crate::api_registry::RegisteredHandle, String> {
    let handle = crate::providers::register_builtins::ensure_and_get(&model.api);
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
    use crate::providers::register_builtins::{
        clear_ensure_and_get_test_hook, install_ensure_and_get_test_hook,
        provider_lifecycle_is_locked_for_test,
    };
    use crate::types::{Api, Model, ModelCost, Provider};
    use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
    use std::thread::JoinHandle;

    struct RestoreBuiltins;

    impl Drop for RestoreBuiltins {
        fn drop(&mut self) {
            clear_api_providers();
            crate::providers::register_builtins::ensure();
        }
    }

    struct ClearRace {
        resume: Sender<()>,
        clear_thread: Option<JoinHandle<()>>,
    }

    impl ClearRace {
        fn install() -> (Self, Receiver<()>) {
            let (entered_tx, entered_rx) = mpsc::channel();
            let (resume_tx, resume_rx) = mpsc::channel();
            install_ensure_and_get_test_hook(entered_tx, resume_rx);
            (
                Self {
                    resume: resume_tx,
                    clear_thread: None,
                },
                entered_rx,
            )
        }

        fn release(&self) {
            let _ = self.resume.send(());
        }

        fn start_clear(&mut self) -> (Receiver<()>, Receiver<()>) {
            let (clear_ready_tx, clear_ready_rx) = mpsc::channel();
            let (clear_done_tx, clear_done_rx) = mpsc::channel();
            self.clear_thread = Some(std::thread::spawn(move || {
                let _ = clear_ready_tx.send(());
                clear_api_providers();
                let _ = clear_done_tx.send(());
            }));
            (clear_ready_rx, clear_done_rx)
        }

        fn join_clear(&mut self) -> std::thread::Result<()> {
            self.clear_thread
                .take()
                .expect("clear thread was not started")
                .join()
        }
    }

    impl Drop for ClearRace {
        fn drop(&mut self) {
            self.release();
            if let Some(clear_thread) = self.clear_thread.take() {
                let _ = clear_thread.join();
            }
            clear_ensure_and_get_test_hook();
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
        let (mut race, entered_rx) = ClearRace::install();
        let model = faux_model();
        let resolve_task = tokio::task::spawn_blocking(move || resolve(&model));

        entered_rx
            .recv()
            .expect("resolve did not reach internal hook");
        let lifecycle_was_held = provider_lifecycle_is_locked_for_test();
        let (clear_ready_rx, clear_done_rx) = race.start_clear();

        clear_ready_rx
            .recv()
            .expect("clear thread did not begin clear attempt");
        let done_before_release = clear_done_rx.try_recv();
        race.release();
        let resolve_result = resolve_task.await;
        let clear_join_result = race.join_clear();
        let done_after_release = if matches!(done_before_release, Err(TryRecvError::Empty)) {
            clear_done_rx.recv()
        } else {
            Ok(())
        };

        assert!(lifecycle_was_held);
        assert!(matches!(done_before_release, Err(TryRecvError::Empty)));
        assert!(resolve_result.expect("resolve task panicked").is_ok());
        assert!(clear_join_result.is_ok());
        assert!(done_after_release.is_ok());
    }
}
