use std::ops::ControlFlow;

use ai::{AssistantMessage, DoneReason, Model, StreamOptions, stream};
use async_trait::async_trait;
use futures::StreamExt;
use knuth_core::{AgentEvent, ModelStepEndReason};
use tokio::sync::{mpsc, oneshot};
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
    GetStepId(oneshot::Sender<Uuid>),
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
            .emit(TurnMessage::Event(AgentEvent::ModelStepStarted {
                step_id: self.step_id,
            }))
            .await;

        let Some(tx) = ctx.ugprade() else { return };

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
                                let _ = tx.send(AgentStepActorMessage::StreamEvent(event)).await;
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
            AgentStepActorMessage::StreamEnded => ControlFlow::Break(()),
            AgentStepActorMessage::Cancel => {
                self.step_reason = ModelStepEndReason::Cancelled;
                ControlFlow::Break(())
            }
            AgentStepActorMessage::GetStepId(tx) => {
                let _ = tx.send(self.step_id.clone());
                ControlFlow::Continue(())
            }
        }
    }

    async fn on_stop(&mut self, ctx: &mut ActorContext<Self::Message>) {
        if ctx.shutdown.is_cancelled() {
            self.step_reason = ModelStepEndReason::Cancelled;
        }

        let _ = self
            .emit(TurnMessage::Event(AgentEvent::ModelStepEnded {
                step_id: self.step_id,
                reason: self.step_reason.clone(),
                assistant_message: self.assistant_message.clone(),
            }))
            .await;
    }
}

impl AgentStepActor {
    pub fn new(
        model: Model,
        options: StreamOptions,
        store_tx: mpsc::Sender<AgentActorMessage>,
        context: ai::Context,
    ) -> Self {
        Self {
            model,
            options,
            store_tx,
            context,
            step_id: Uuid::now_v7(),
            step_reason: ModelStepEndReason::Success,
            assistant_message: None,
        }
    }

    pub async fn emit(&mut self, event: TurnMessage) -> Result<(), AgentStepError> {
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
                self.emit(TurnMessage::Event(
                    AgentEvent::AssistantMessageTextStarted { content_index },
                ))
                .await?;
            }
            ai::AssistantMessageEvent::TextDelta {
                content_index,
                delta,
                ..
            } => {
                self.emit(TurnMessage::Event(AgentEvent::AssistantMessageTextDelta {
                    content_index,
                    delta,
                }))
                .await?;
            }
            ai::AssistantMessageEvent::TextEnd {
                content_index,
                content,
                partial,
                ..
            } => {
                self.emit(TurnMessage::Event(
                    AgentEvent::AssistantMessageTextCompleted {
                        content_index,
                        text_content: content,
                        assistant_message: partial,
                    },
                ))
                .await?;
            }

            ai::AssistantMessageEvent::ThinkingStart { content_index, .. } => {
                self.emit(TurnMessage::Event(
                    AgentEvent::AssistantMessageThinkingStarted { content_index },
                ))
                .await?;
            }
            ai::AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta,
                ..
            } => {
                self.emit(TurnMessage::Event(
                    AgentEvent::AssistantMessageThinkingDelta {
                        content_index,
                        delta,
                    },
                ))
                .await?;
            }
            ai::AssistantMessageEvent::ThinkingEnd {
                content_index,
                content,
                ..
            } => {
                self.emit(TurnMessage::Event(
                    AgentEvent::AssistantMessageThinkingCompleted {
                        content_index,
                        content,
                    },
                ))
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
    step_runtime: ActorRuntime<AgentStepActorMessage>,
}

impl AgentStepRunner {
    pub async fn new(
        model: Model,
        options: StreamOptions,
        store_tx: mpsc::Sender<AgentActorMessage>,
        context: ai::Context,
    ) -> Self {
        let step_runtime =
            spawn_actor(AgentStepActor::new(model, options, store_tx, context), 100).await;

        Self { step_runtime }
    }

    pub async fn step_id(&self) -> Uuid {
        let (tx, rx) = oneshot::channel();
        self.step_runtime
            .handle()
            .send(AgentStepActorMessage::GetStepId(tx))
            .await
            .unwrap();
        rx.await.expect("GetStepId should not block")
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
            let ended = matches!(
                message,
                AgentActorMessage::Turn(_, TurnMessage::Event(AgentEvent::ModelStepEnded { .. }))
            );
            messages.push(message);
            if ended {
                break;
            }
        }
        runner.step_runtime.shutdown().await;
        clear_faux_responses();

        let expected_step_id = match messages.first().expect("expected actor messages") {
            AgentActorMessage::Turn(
                step_id,
                TurnMessage::Event(AgentEvent::ModelStepStarted {
                    step_id: event_step_id,
                }),
            ) => {
                assert_eq!(step_id, event_step_id);
                *step_id
            }
            _ => panic!("first message should be ModelStepStarted"),
        };

        for message in &messages {
            match message {
                AgentActorMessage::Turn(step_id, TurnMessage::Event(event)) => {
                    assert_eq!(*step_id, expected_step_id);
                    match event {
                        AgentEvent::ModelStepStarted { step_id }
                        | AgentEvent::ModelStepEnded { step_id, .. } => {
                            assert_eq!(*step_id, expected_step_id);
                        }
                        _ => {}
                    }
                }
                AgentActorMessage::Turn(_, TurnMessage::Finished { .. }) => {
                    panic!("unexpected TurnMessage::Finished");
                }
                AgentActorMessage::Command(_) => panic!("unexpected actor command"),
            }
        }

        let lifecycle = messages
            .iter()
            .filter_map(|message| match message {
                AgentActorMessage::Turn(
                    _,
                    TurnMessage::Event(AgentEvent::ModelStepStarted { .. }),
                ) => Some("started"),
                AgentActorMessage::Turn(
                    _,
                    TurnMessage::Event(AgentEvent::ModelStepEnded { .. }),
                ) => Some("ended"),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(lifecycle, ["started", "ended"]);

        match messages.last().expect("expected actor messages") {
            AgentActorMessage::Turn(
                step_id,
                TurnMessage::Event(AgentEvent::ModelStepEnded {
                    step_id: event_step_id,
                    reason,
                    assistant_message,
                }),
            ) => {
                assert_eq!(*step_id, expected_step_id);
                assert_eq!(*event_step_id, expected_step_id);
                assert_eq!(*reason, ModelStepEndReason::Length);
                assert_eq!(
                    assistant_message
                        .as_ref()
                        .expect("ModelStepEnded should include assistant_message")
                        .stop_reason,
                    StopReason::Length
                );
            }
            _ => unreachable!(),
        }
    }
}
