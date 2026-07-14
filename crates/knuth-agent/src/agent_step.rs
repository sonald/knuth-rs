use std::ops::ControlFlow;

use ai::{AssistantMessage, DoneReason, Model, StreamOptions, stream};
use async_trait::async_trait;
use futures::StreamExt;
use knuth_core::{AgentEvent, ModelStepEndReason};
use tokio::sync::mpsc;
use tracing::debug;
use uuid::Uuid;

use crate::*;

#[derive(Debug, thiserror::Error)]
pub enum AgentStepError {
    #[error("Failed to send event to store: {0}")]
    EventSendError(String),

    #[error("LLM error: {0}")]
    LLMError(String),
}

#[derive(Debug)]
pub enum AgentStepActorMessage {
    StreamEvent(ai::AssistantMessageEvent),
    StreamEnded,
    Cancel,
}

struct AgentStepActor {
    model: Model,
    options: StreamOptions,
    store_tx: mpsc::Sender<AgentActorMessage>,
    context: ai::Context,
    step_id: Uuid,

    step_reason: ModelStepEndReason,
    assistant_message: Option<AssistantMessage>,
}

#[async_trait]
impl Actor for AgentStepActor {
    type Message = AgentStepActorMessage;

    async fn on_start(&mut self, ctx: &mut ActorContext<Self::Message>) {
        let _ = self
            .emit(AgentEvent::ModelStepStarted {
                step_id: self.step_id,
            })
            .await;

        let Some(tx) = ctx.upgrade() else { return };

        let shutdown = ctx.shutdown.clone();
        let context = self.context.clone();
        let model = self.model.clone();
        let options = self.options.clone();

        tokio::spawn(async move {
            let mut stream = stream(&model, &context, Some(&options));
            loop {
                tokio::select! {
                    _ = shutdown.cancelled() => break,
                    event = stream.next() => {
                        match event {
                            Some(event) => {
                                if tx.send(AgentStepActorMessage::StreamEvent(event)).await.is_err() {
                                    break;
                                }
                            }
                            None => {
                                let _ = tx.send(AgentStepActorMessage::StreamEnded).await;
                                break;
                            }
                        }
                    }
                }
            }
        });
    }

    async fn handle(
        &mut self,
        message: Self::Message,
        _: &mut ActorContext<Self::Message>,
    ) -> ControlFlow<()> {
        match message {
            AgentStepActorMessage::StreamEvent(event) => match self.handle_event(event).await {
                Ok(()) => ControlFlow::Continue(()),
                Err(e) => {
                    self.step_reason = ModelStepEndReason::Error(e.to_string());
                    ControlFlow::Break(())
                }
            },
            AgentStepActorMessage::StreamEnded => {
                // A stream that closes without a Done (or Error) event is a broken
                // stream, not a successful step.
                if self.assistant_message.is_none() {
                    self.step_reason = ModelStepEndReason::Error(
                        "stream ended without a Done event".to_string(),
                    );
                }
                ControlFlow::Break(())
            }
            AgentStepActorMessage::Cancel => {
                self.step_reason = ModelStepEndReason::Cancelled;
                ControlFlow::Break(())
            }
        }
    }

    async fn on_stop(&mut self, ctx: &mut ActorContext<Self::Message>) {
        if ctx.shutdown.is_cancelled() && self.assistant_message.is_none() {
            self.step_reason = ModelStepEndReason::Cancelled;
        }

        // Completion is a control signal, not a domain event: the harness
        // derives and commits `ModelStepEnded` itself.
        let _ = self
            .store_tx
            .send(AgentActorMessage::StepFinished {
                step_id: self.step_id,
                reason: self.step_reason.clone(),
                assistant_message: self.assistant_message.take(),
            })
            .await;
    }
}

impl AgentStepActor {
    pub fn new(
        model: Model,
        options: StreamOptions,
        store_tx: mpsc::Sender<AgentActorMessage>,
        context: ai::Context,
        step_id: Uuid,
    ) -> Self {
        Self {
            model,
            options,
            store_tx,
            context,
            step_id,
            step_reason: ModelStepEndReason::Success,
            assistant_message: None,
        }
    }

    pub async fn emit(&mut self, event: AgentEvent) -> Result<(), AgentStepError> {
        self.store_tx
            .send(AgentActorMessage::Turn(self.step_id, event))
            .await
            .map_err(|e| AgentStepError::EventSendError(e.to_string()))
    }

    async fn handle_event(
        &mut self,
        event: ai::AssistantMessageEvent,
    ) -> Result<(), AgentStepError> {
        match event {
            ai::AssistantMessageEvent::Start { .. } => {}
            ai::AssistantMessageEvent::Done { message, reason } => {
                self.step_reason = match reason {
                    DoneReason::Stop => ModelStepEndReason::Success,
                    DoneReason::Length => ModelStepEndReason::Length,
                    DoneReason::ToolUse => ModelStepEndReason::ToolUse,
                };
                self.assistant_message = Some(message);
                debug!("Done reason: {:?}", reason);
            }
            ai::AssistantMessageEvent::Error { error, .. } => {
                return Err(AgentStepError::LLMError(
                    error.error_message.unwrap_or("Unknown error".to_string()),
                ));
            }

            ai::AssistantMessageEvent::TextStart { content_index, .. } => {
                self.emit(AgentEvent::AssistantMessageTextStarted { content_index })
                    .await?;
            }
            ai::AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                ..
            } => {
                self.emit(AgentEvent::AssistantMessageTextDelta {
                    content_index,
                    delta,
                })
                .await?;
            }
            ai::AssistantMessageEvent::TextEnd {
                content_index,
                content,
                partial,
                ..
            } => {
                self.emit(AgentEvent::AssistantMessageTextCompleted {
                    content_index,
                    text_content: content,
                    assistant_message: partial,
                })
                .await?;
            }

            ai::AssistantMessageEvent::ThinkingStart { content_index, .. } => {
                self.emit(AgentEvent::AssistantMessageThinkingStarted { content_index })
                    .await?;
            }
            ai::AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                ..
            } => {
                self.emit(AgentEvent::AssistantMessageThinkingDelta {
                    content_index,
                    delta,
                })
                .await?;
            }
            ai::AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
                ..
            } => {
                self.emit(AgentEvent::AssistantMessageThinkingCompleted {
                    content_index,
                    content,
                })
                .await?;
            }

            ai::AssistantMessageEvent::ToolCallEnd { .. } => {}

            ai::AssistantMessageEvent::ToolCallStart { .. } => {}
            ai::AssistantMessageEvent::ToolCallDelta { .. } => {}
        }

        Ok(())
    }
}

#[derive(Debug)]
pub struct AgentStepRunner {
    step_id: Uuid,
    step_runtime: ActorRuntime<AgentStepActorMessage>,
}

impl AgentStepRunner {
    pub async fn new(
        model: Model,
        options: StreamOptions,
        store_tx: mpsc::Sender<AgentActorMessage>,
        context: ai::Context,
    ) -> Self {
        let step_id = Uuid::now_v7();
        let step_runtime = spawn_actor(
            AgentStepActor::new(model, options, store_tx, context, step_id),
            100,
        )
        .await;

        Self {
            step_id,
            step_runtime,
        }
    }

    pub fn step_id(&self) -> Uuid {
        self.step_id
    }

    pub async fn cancel(&self) {
        let _ = self
            .step_runtime
            .handle()
            .send(AgentStepActorMessage::Cancel)
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ai::providers::faux::{clear_faux_responses, set_faux_responses};
    use ai::{
        Api, AssistantMessage, AssistantMessageEvent, AssistantRole, ContentBlock, KnownApi,
        ModelCost, Provider, StopReason, ToolCall, Usage,
    };
    use serde_json::Map;
    use std::time::Duration;
    use tokio::time::timeout;

    fn mk_model() -> Model {
        Model {
            id: "faux".into(),
            name: "Faux".into(),
            api: Api::from("faux"),
            provider: Provider::from("faux"),
            base_url: String::new(),
            reasoning: false,
            thinking_level_map: None,
            input: vec![],
            cost: ModelCost::default(),
            context_window: 128_000,
            max_tokens: 4096,
            headers: None,
            compat: None,
        }
    }

    fn mk_message(stop_reason: StopReason, content: Vec<ContentBlock>) -> AssistantMessage {
        AssistantMessage {
            role: AssistantRole::Assistant,
            content,
            api: Api::known(KnownApi::OpenAIResponses),
            provider: Provider::from("openai"),
            model: "test-model".into(),
            response_model: None,
            response_id: None,
            diagnostics: None,
            usage: Usage::default(),
            stop_reason,
            error_message: None,
            timestamp: 0,
        }
    }

    fn mk_actor(store_tx: mpsc::Sender<AgentActorMessage>) -> AgentStepActor {
        AgentStepActor::new(
            mk_model(),
            StreamOptions::default(),
            store_tx,
            ai::Context {
                system_prompt: None,
                messages: vec![],
                tools: None,
            },
            Uuid::now_v7(),
        )
    }

    #[tokio::test]
    async fn length_done_reason_ends_turn_without_panic() {
        let (tx, _rx) = mpsc::channel(4);
        let mut step = mk_actor(tx);
        let message = mk_message(StopReason::Length, vec![ContentBlock::text("partial")]);

        step.handle_event(AssistantMessageEvent::Done {
            reason: DoneReason::Length,
            message,
        })
        .await
        .unwrap();

        assert_eq!(step.step_reason, ModelStepEndReason::Length);
        assert_eq!(
            step.assistant_message.unwrap().stop_reason,
            StopReason::Length
        );
    }

    #[tokio::test]
    async fn tooluse_done_reason_ends_turn_without_panic() {
        let (tx, _rx) = mpsc::channel(4);
        let mut step = mk_actor(tx);
        let message = mk_message(
            StopReason::ToolUse,
            vec![ContentBlock::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: Map::new(),
                thought_signature: None,
            })],
        );

        step.handle_event(AssistantMessageEvent::Done {
            reason: DoneReason::ToolUse,
            message,
        })
        .await
        .unwrap();

        assert_eq!(step.step_reason, ModelStepEndReason::ToolUse);
        assert_eq!(
            step.assistant_message.unwrap().stop_reason,
            StopReason::ToolUse
        );
    }

    #[tokio::test]
    async fn start_step_emits_started_and_ended_with_length_reason() {
        let _guard = crate::test_support::faux_lock();
        clear_faux_responses();
        let (tx, mut rx) = mpsc::channel(16);
        let message = mk_message(StopReason::Length, vec![ContentBlock::text("partial")]);
        set_faux_responses(vec![message]);

        let runner = AgentStepRunner::new(
            mk_model(),
            StreamOptions::default(),
            tx,
            ai::Context {
                system_prompt: None,
                messages: vec![],
                tools: None,
            },
        )
        .await;

        let mut messages = Vec::new();
        loop {
            let message = timeout(Duration::from_secs(1), rx.recv())
                .await
                .expect("step should emit events")
                .expect("step channel should stay open until ModelStepEnded");
            let ended = matches!(message, AgentActorMessage::StepFinished { .. });
            messages.push(message);
            if ended {
                break;
            }
        }
        clear_faux_responses();

        let expected_step_id = match messages.first().expect("expected actor messages") {
            AgentActorMessage::Turn(
                step_id,
                AgentEvent::ModelStepStarted {
                    step_id: event_step_id,
                },
            ) => {
                assert_eq!(step_id, event_step_id);
                *step_id
            }
            _ => panic!("first message should be ModelStepStarted"),
        };
        // step_id stays readable after the actor has exited.
        assert_eq!(runner.step_id(), expected_step_id);

        for message in &messages {
            match message {
                AgentActorMessage::Turn(step_id, event) => {
                    assert_eq!(*step_id, expected_step_id);
                    if let AgentEvent::ModelStepStarted { step_id } = event {
                        assert_eq!(*step_id, expected_step_id);
                    }
                }
                AgentActorMessage::StepFinished { step_id, .. } => {
                    assert_eq!(*step_id, expected_step_id);
                }
                AgentActorMessage::Command(_) => panic!("unexpected actor command"),
                AgentActorMessage::ToolFinished { .. } => panic!("unexpected ToolFinished"),
            }
        }

        let lifecycle = messages
            .iter()
            .filter_map(|message| match message {
                AgentActorMessage::Turn(_, AgentEvent::ModelStepStarted { .. }) => Some("started"),
                AgentActorMessage::StepFinished { .. } => Some("finished"),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(lifecycle, ["started", "finished"]);

        match messages.last().expect("expected actor messages") {
            AgentActorMessage::StepFinished {
                step_id,
                reason,
                assistant_message,
            } => {
                assert_eq!(*step_id, expected_step_id);
                assert_eq!(*reason, ModelStepEndReason::Length);
                assert_eq!(
                    assistant_message
                        .as_ref()
                        .expect("StepFinished should include assistant_message")
                        .stop_reason,
                    StopReason::Length
                );
            }
            _ => unreachable!(),
        }
    }

    fn recv_step_finished(
        rx: &mut mpsc::Receiver<AgentActorMessage>,
    ) -> (ModelStepEndReason, Option<AssistantMessage>) {
        loop {
            match rx.try_recv().expect("expected a StepFinished message") {
                AgentActorMessage::StepFinished {
                    reason,
                    assistant_message,
                    ..
                } => return (reason, assistant_message),
                _ => continue,
            }
        }
    }

    #[tokio::test]
    async fn stream_ended_without_done_reports_error() {
        let (tx, _rx) = mpsc::channel(4);
        let mut step = mk_actor(tx);
        let mut ctx = test_actor_context();

        let flow = step.handle(AgentStepActorMessage::StreamEnded, &mut ctx).await;

        assert!(flow.is_break());
        assert!(matches!(step.step_reason, ModelStepEndReason::Error(_)));
    }

    #[tokio::test]
    async fn stream_ended_after_done_keeps_done_reason() {
        let (tx, _rx) = mpsc::channel(4);
        let mut step = mk_actor(tx);
        let mut ctx = test_actor_context();

        step.handle_event(AssistantMessageEvent::Done {
            reason: DoneReason::Stop,
            message: mk_message(StopReason::Stop, vec![ContentBlock::text("done")]),
        })
        .await
        .unwrap();
        let flow = step.handle(AgentStepActorMessage::StreamEnded, &mut ctx).await;

        assert!(flow.is_break());
        assert_eq!(step.step_reason, ModelStepEndReason::Success);
    }

    #[tokio::test]
    async fn cancel_message_breaks_with_cancelled_reason() {
        let (tx, _rx) = mpsc::channel(4);
        let mut step = mk_actor(tx);
        let mut ctx = test_actor_context();

        let flow = step.handle(AgentStepActorMessage::Cancel, &mut ctx).await;

        assert!(flow.is_break());
        assert_eq!(step.step_reason, ModelStepEndReason::Cancelled);
    }

    #[tokio::test]
    async fn shutdown_without_done_reports_cancelled() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut step = mk_actor(tx);
        let mut ctx = test_actor_context();
        ctx.shutdown.cancel();

        step.on_stop(&mut ctx).await;

        let (reason, assistant_message) = recv_step_finished(&mut rx);
        assert_eq!(reason, ModelStepEndReason::Cancelled);
        assert!(assistant_message.is_none());
    }

    #[tokio::test]
    async fn shutdown_after_done_keeps_completed_reason() {
        let (tx, mut rx) = mpsc::channel(4);
        let mut step = mk_actor(tx);
        step.handle_event(AssistantMessageEvent::Done {
            reason: DoneReason::Stop,
            message: mk_message(StopReason::Stop, vec![ContentBlock::text("done")]),
        })
        .await
        .unwrap();
        let mut ctx = test_actor_context();
        ctx.shutdown.cancel();

        step.on_stop(&mut ctx).await;

        let (reason, assistant_message) = recv_step_finished(&mut rx);
        assert_eq!(reason, ModelStepEndReason::Success);
        assert_eq!(
            assistant_message.expect("completed step keeps its message").stop_reason,
            StopReason::Stop
        );
    }
}
