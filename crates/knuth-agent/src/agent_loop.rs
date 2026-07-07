use ai::{DoneReason, Model, StreamOptions, stream};
use futures::StreamExt;
use knuth_core::{AgentEvent, TurnEndReason, TurnOutcome};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use uuid::Uuid;

use crate::{AgentActorMessage, TurnMessage};

#[derive(Debug, thiserror::Error)]
pub enum AgentTurnLoopError {
    #[error("Failed to send event to store: {0}")]
    EventSendError(String),
}

#[derive(Debug)]
pub struct AgentTurnRunner {
    model: Model,
    options: StreamOptions,
    store_tx: mpsc::Sender<AgentActorMessage>,
    context: ai::Context,
    turn_id: Uuid,
    last_message_id: Option<Uuid>,
    cancel_token: CancellationToken,

    has_tool_calls: bool
}

impl AgentTurnRunner {
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
            turn_id: Uuid::now_v7(),
            last_message_id: None,
            has_tool_calls: false,
            cancel_token,
        }
    }

    pub async fn emit(&mut self, event: TurnMessage) -> Result<(), AgentTurnLoopError> {
        self.store_tx
            .send(AgentActorMessage::Turn(event))
            .await
            .map_err(|e| AgentTurnLoopError::EventSendError(e.to_string()))
    }

    pub(crate) async fn start_turn_loop(
        &mut self,
        max_turns: Option<usize>,
    ) -> Result<TurnOutcome, AgentTurnLoopError> {
        let mut stream = stream(&self.model, &self.context, Some(&self.options));

        let mut turn = 0;
        let max_turns = max_turns.unwrap_or(1);

        while turn < max_turns {
            loop {
                tokio::select! {
                    biased;

                    _ = self.cancel_token.cancelled() => {
                        return Ok(TurnOutcome {
                            turn_id: self.turn_id,
                            reason: TurnEndReason::Cancelled,
                        });
                    }

                    event = stream.next() => {
                        let Some(event) = event else { debug!("stream ended, breaking"); break; };
                        self.handle_event(event).await?;
                    }

                }
            }

            if !self.has_tool_calls {
                break;
            }

            turn += 1;
        }

        if self.has_tool_calls {
            //TODO: 
        }

        Ok(TurnOutcome {
            turn_id: self.turn_id,
            reason: TurnEndReason::Success,
        })
    }

    async fn handle_event(
        &mut self,
        event: ai::AssistantMessageEvent,
    ) -> Result<(), AgentTurnLoopError> {
        match event {
            ai::AssistantMessageEvent::Start { .. } => {
                self.emit(TurnMessage::Event(AgentEvent::AgentTurnStarted {
                    turn_id: self.turn_id,
                }))
                .await?;
            }
            ai::AssistantMessageEvent::Done { message, reason } => {
                let turn_reason = match reason {
                    DoneReason::Stop => TurnEndReason::Success,
                    DoneReason::Length => TurnEndReason::Length,
                    DoneReason::ToolUse => TurnEndReason::ToolUse,
                };
                self.emit(TurnMessage::Event(AgentEvent::AgentTurnEnded {
                    turn_id: self.turn_id,
                    reason: turn_reason,
                    assistant_message: Some(message),
                }))
                .await?;
            }
            ai::AssistantMessageEvent::Error { error, .. } => {
                self.emit(TurnMessage::Event(AgentEvent::ErrorOccurred {
                    message: error.error_message.unwrap_or("Unknown error".to_string()),
                    details: None,
                }))
                .await?;

                self.emit(TurnMessage::Event(AgentEvent::AgentTurnEnded {
                    turn_id: self.turn_id,
                    reason: TurnEndReason::Error,
                    assistant_message: None,
                }))
                .await?;
            }

            ai::AssistantMessageEvent::TextStart { .. } => {
                self.last_message_id = Some(Uuid::now_v7());
                self.emit(TurnMessage::Event(AgentEvent::AssistantMessageTextStarted {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                }))
                .await?;
            }
            ai::AssistantMessageEvent::TextDelta { delta, .. } => {
                self.emit(TurnMessage::Event(AgentEvent::AssistantMessageTextDelta {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                    delta,
                }))
                .await?;
            }
            ai::AssistantMessageEvent::TextEnd {
                content, partial, ..
            } => {
                self.emit(TurnMessage::Event(AgentEvent::AssistantMessageTextCompleted {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                    text_content: content,
                    assistant_message: partial,
                }))
                .await?;
            }

            ai::AssistantMessageEvent::ThinkingStart { .. } => {
                self.last_message_id = Some(Uuid::now_v7());
                self.emit(TurnMessage::Event(AgentEvent::AssistantMessageThinkingStarted {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                }))
                .await?;
            }
            ai::AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                self.emit(TurnMessage::Event(AgentEvent::AssistantMessageThinkingDelta {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                    delta,
                }))
                .await?;
            }
            ai::AssistantMessageEvent::ThinkingEnd { content, .. } => {
                self.emit(TurnMessage::Event(AgentEvent::AssistantMessageThinkingCompleted {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                    content,
                }))
                .await?;
            }

            ai::AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                self.emit(TurnMessage::Event(AgentEvent::ToolCallRequested {
                    tool_call_id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                }))
                .await?;
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
    use ai::{
        Api, AssistantMessage, AssistantMessageEvent, AssistantRole, ContentBlock, KnownApi,
        ModelCost, Provider, StopReason, ToolCall, Usage,
    };
    use serde_json::Map;

    fn mk_model() -> Model {
        Model {
            id: "test-model".into(),
            name: "Test Model".into(),
            api: Api::known(KnownApi::OpenAIResponses),
            provider: Provider::from("openai"),
            base_url: "https://api.openai.com".into(),
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

    fn mk_loop(store_tx: mpsc::Sender<AgentActorMessage>) -> AgentTurnRunner {
        AgentTurnRunner::new(
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
        let (tx, mut rx) = mpsc::channel(4);
        let mut agent_loop = mk_loop(tx);
        let message = mk_message(StopReason::Length, vec![ContentBlock::text("partial")]);

        agent_loop
            .handle_event(AssistantMessageEvent::Done {
                reason: DoneReason::Length,
                message,
            })
            .await
            .unwrap();

        let Some(AgentActorMessage::Turn(TurnMessage::Event(AgentEvent::AgentTurnEnded {
            reason,
            assistant_message,
            ..
        }))) = rx.recv().await
        else {
            panic!("expected AgentTurnEnded");
        };
        assert_eq!(reason, TurnEndReason::Length);
        assert_eq!(assistant_message.unwrap().stop_reason, StopReason::Length);
    }

    #[tokio::test]
    async fn tooluse_done_reason_ends_turn_without_panic() {
        let (tx, mut rx) = mpsc::channel(4);
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

        let Some(AgentActorMessage::Turn(TurnMessage::Event(AgentEvent::AgentTurnEnded {
            reason,
            assistant_message,
            ..
        }))) = rx.recv().await
        else {
            panic!("expected AgentTurnEnded");
        };
        assert_eq!(reason, TurnEndReason::ToolUse);
        assert_eq!(assistant_message.unwrap().stop_reason, StopReason::ToolUse);
    }
}
