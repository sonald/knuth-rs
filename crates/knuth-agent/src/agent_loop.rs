use ai::{AssistantMessage, DoneReason, Model, StreamOptions, stream};
use futures::StreamExt;
use knuth_core::{AgentEvent, ModelStepEndReason};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use uuid::Uuid;

use crate::{AgentActorMessage, TurnMessage};

#[derive(Debug, thiserror::Error)]
pub enum AgentStepLoopError {
    #[error("Failed to send event to store: {0}")]
    EventSendError(String),

    #[error("LLM error: {0}")]
    LLMError(String),
}

#[derive(Debug)]
pub struct AgentStepRunner {
    model: Model,
    options: StreamOptions,
    store_tx: mpsc::Sender<AgentActorMessage>,
    context: ai::Context,
    step_id: Uuid,
    last_message_id: Option<Uuid>,
    cancel_token: CancellationToken,

    step_reason: ModelStepEndReason,
    assistant_message: Option<AssistantMessage>,
}

impl AgentStepRunner {
    pub fn new(
        model: Model,
        options: StreamOptions,
        store_tx: mpsc::Sender<AgentActorMessage>,
        context: ai::Context,
        cancel_token: CancellationToken,
    ) -> Self {
        Self {
            model,
            options,
            store_tx,
            context,
            step_id: Uuid::now_v7(),
            last_message_id: None,
            cancel_token,
            step_reason: ModelStepEndReason::Success,
            assistant_message: None,
        }
    }

    pub fn step_id(&self) -> Uuid {
        self.step_id
    }

    pub async fn emit(&mut self, event: TurnMessage) -> Result<(), AgentStepLoopError> {
        self.store_tx
            .send(AgentActorMessage::Turn(self.step_id, event))
            .await
            .map_err(|e| AgentStepLoopError::EventSendError(e.to_string()))
    }

    pub(crate) async fn start_step(&mut self) -> Result<(), AgentStepLoopError> {
        let mut stream = stream(&self.model, &self.context, Some(&self.options));

        self.emit(TurnMessage::Event(AgentEvent::ModelStepStarted {
            step_id: self.step_id,
        }))
        .await?;


        loop {
            tokio::select! {
                biased;

                _ = self.cancel_token.cancelled() => {
                    self.step_reason = ModelStepEndReason::Cancelled;
                    break;
                }

                event = stream.next() => {
                    let Some(event) = event else {
                        debug!("stream ended, breaking");
                        break;
                    };

                    if let Err(e) = self.handle_event(event).await {
                        self.step_reason = ModelStepEndReason::Error(e.to_string());
                        break;
                    }
                }
            }
        }

        self.emit(TurnMessage::Event(AgentEvent::ModelStepEnded { 
            step_id: self.step_id, reason: self.step_reason.clone(), assistant_message: self.assistant_message.clone() 
        })).await?;

        Ok(())
    }

    async fn handle_event(
        &mut self,
        event: ai::AssistantMessageEvent,
    ) -> Result<(), AgentStepLoopError> {
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
                return Err(AgentStepLoopError::LLMError(
                    error.error_message.unwrap_or("Unknown error".to_string()),
                ));
            }

            ai::AssistantMessageEvent::TextStart { content_index, .. } => {
                self.emit(TurnMessage::Event(
                    AgentEvent::AssistantMessageTextStarted {
                        content_index,
                    },
                ))
                .await?;
            }
            ai::AssistantMessageEvent::TextDelta { content_index, delta, .. } => {
                self.emit(TurnMessage::Event(AgentEvent::AssistantMessageTextDelta {
                    content_index,
                    delta,
                }))
                .await?;
            }
            ai::AssistantMessageEvent::TextEnd {
                content_index, content, partial, ..
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
                self.last_message_id = Some(Uuid::now_v7());
                self.emit(TurnMessage::Event(
                    AgentEvent::AssistantMessageThinkingStarted {
                        content_index
                    },
                ))
                .await?;
            }
            ai::AssistantMessageEvent::ThinkingDelta { content_index, delta, .. } => {
                self.emit(TurnMessage::Event(
                    AgentEvent::AssistantMessageThinkingDelta {
                        content_index,
                        delta,
                    },
                ))
                .await?;
            }
            ai::AssistantMessageEvent::ThinkingEnd { content_index, content, .. } => {
                self.emit(TurnMessage::Event(
                    AgentEvent::AssistantMessageThinkingCompleted {
                        content_index,
                        content,
                    },
                ))
                .await?;
            }

            ai::AssistantMessageEvent::ToolCallEnd { .. } => {
            }

            ai::AssistantMessageEvent::ToolCallStart { .. } => {}
            ai::AssistantMessageEvent::ToolCallDelta { .. } => {}
        }

        Ok(())
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

    fn mk_loop(store_tx: mpsc::Sender<AgentActorMessage>) -> AgentStepRunner {
        AgentStepRunner::new(
            mk_model(),
            StreamOptions::default(),
            store_tx,
            ai::Context {
                system_prompt: None,
                messages: vec![],
                tools: None,
            },
            CancellationToken::new(),
        )
    }

    #[tokio::test]
    async fn length_done_reason_ends_turn_without_panic() {
        let (tx, _rx) = mpsc::channel(4);
        let mut agent_loop = mk_loop(tx);
        let message = mk_message(StopReason::Length, vec![ContentBlock::text("partial")]);

        agent_loop
            .handle_event(AssistantMessageEvent::Done {
                reason: DoneReason::Length,
                message,
            })
            .await
            .unwrap();

        assert_eq!(agent_loop.step_reason, ModelStepEndReason::Length);
        assert_eq!(
            agent_loop.assistant_message.unwrap().stop_reason,
            StopReason::Length
        );
    }

    #[tokio::test]
    async fn tooluse_done_reason_ends_turn_without_panic() {
        let (tx, _rx) = mpsc::channel(4);
        let mut agent_loop = mk_loop(tx);
        let message = mk_message(
            StopReason::ToolUse,
            vec![ContentBlock::ToolCall(ToolCall {
                id: "call_1".into(),
                name: "search".into(),
                arguments: Map::new(),
                thought_signature: None,
            })],
        );

        agent_loop
            .handle_event(AssistantMessageEvent::Done {
                reason: DoneReason::ToolUse,
                message,
            })
            .await
            .unwrap();

        assert_eq!(agent_loop.step_reason, ModelStepEndReason::ToolUse);
        assert_eq!(
            agent_loop.assistant_message.unwrap().stop_reason,
            StopReason::ToolUse
        );
    }

    #[tokio::test]
    async fn start_step_emits_started_and_ended_with_length_reason() {
        clear_faux_responses();
        let (tx, mut rx) = mpsc::channel(16);
        let mut agent_loop = mk_loop(tx);
        let expected_step_id = agent_loop.step_id();
        let message = mk_message(StopReason::Length, vec![ContentBlock::text("partial")]);
        set_faux_responses(vec![message]);

        let result = timeout(Duration::from_secs(1), agent_loop.start_step()).await;
        clear_faux_responses();
        result.expect("start_step should not block").unwrap();

        let messages = std::iter::from_fn(|| rx.try_recv().ok()).collect::<Vec<_>>();
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
