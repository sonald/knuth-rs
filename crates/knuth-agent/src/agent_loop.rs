use ai::{DoneReason, Model, StreamOptions, stream};
use futures::StreamExt;
use knuth_core::{AgentEvent, TurnEndReason, TurnOutcome};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use uuid::Uuid;

#[derive(Debug, thiserror::Error)]
pub enum AgentLoopError {
    #[error("Failed to send event to store: {0}")]
    EventSendError(String),
}

#[derive(Debug)]
pub struct AgentLoop {
    model: Model,
    options: StreamOptions,
    store_tx: mpsc::Sender<AgentEvent>,
    context: ai::Context,
    turn_cancel: CancellationToken,
    turn_id: Uuid,
    last_message_id: Option<Uuid>,
}

impl AgentLoop {
    pub fn new(
        model: Model,
        options: StreamOptions,
        store_tx: mpsc::Sender<AgentEvent>,
        context: ai::Context,
        turn_cancel: CancellationToken,
    ) -> Self {
        Self {
            model,
            options,
            store_tx,
            context,
            turn_cancel,
            turn_id: Uuid::now_v7(),
            last_message_id: None,
        }
    }

    async fn emit(&mut self, event: AgentEvent) -> Result<(), AgentLoopError> {
        self.store_tx
            .send(event)
            .await
            .map_err(|e| AgentLoopError::EventSendError(e.to_string()))
    }

    pub(crate) async fn start_agent_loop(
        &mut self,
        _max_turns: Option<usize>,
    ) -> Result<TurnOutcome, AgentLoopError> {
        let mut stream = stream(&self.model, &self.context, Some(&self.options));

        loop {
            tokio::select! {
                biased;

                _ = self.turn_cancel.cancelled() => {
                    debug!("Agent loop cancelled");
                    self.emit(AgentEvent::AgentTurnEnded {
                        turn_id: self.turn_id,
                        reason: TurnEndReason::Cancelled,
                        assistant_message: None,
                    })
                    .await?;

                    return Ok(TurnOutcome {
                        turn_id: self.turn_id,
                        reason: TurnEndReason::Cancelled,
                    });
                }

                event = stream.next() => {
                    let Some(event) = event else {
                        debug!("Agent loop stream ended");
                        //TODO: check if the last message is a text message only
                        return Ok(TurnOutcome {
                            turn_id: self.turn_id,
                            reason: TurnEndReason::Success,
                        });
                    };
                    self.handle_event(event).await?;
                }

            }
        }

        // Ok(TurnOutcome {
        //     turn_id: self.turn_id,
        //     reason: TurnEndReason::Success,
        // })
    }

    async fn handle_event(
        &mut self,
        event: ai::AssistantMessageEvent,
    ) -> Result<(), AgentLoopError> {
        match event {
            ai::AssistantMessageEvent::Start { .. } => {
                self.emit(AgentEvent::AgentTurnStarted {
                    turn_id: self.turn_id,
                })
                .await?;
            }
            ai::AssistantMessageEvent::Done { message, reason } => {
                let turn_reason = match reason {
                    DoneReason::Stop => TurnEndReason::Success,
                    DoneReason::Length => TurnEndReason::Length,
                    DoneReason::ToolUse => TurnEndReason::ToolUse,
                };
                self.emit(AgentEvent::AgentTurnEnded {
                    turn_id: self.turn_id,
                    reason: turn_reason,
                    assistant_message: Some(message),
                })
                .await?;
            }
            ai::AssistantMessageEvent::Error { error, .. } => {
                self.emit(AgentEvent::ErrorOccurred {
                    message: error.error_message.unwrap_or("Unknown error".to_string()),
                    details: None,
                })
                .await?;

                self.emit(AgentEvent::AgentTurnEnded {
                    turn_id: self.turn_id,
                    reason: TurnEndReason::Error,
                    assistant_message: None,
                })
                .await?;
            }

            ai::AssistantMessageEvent::TextStart { .. } => {
                self.last_message_id = Some(Uuid::now_v7());
                self.emit(AgentEvent::AssistantMessageTextStarted {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                })
                .await?;
            }
            ai::AssistantMessageEvent::TextDelta { delta, .. } => {
                self.emit(AgentEvent::AssistantMessageTextDelta {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                    delta,
                })
                .await?;
            }
            ai::AssistantMessageEvent::TextEnd {
                content, partial, ..
            } => {
                self.emit(AgentEvent::AssistantMessageTextCompleted {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                    text_content: content,
                    assistant_message: partial,
                })
                .await?;
            }

            ai::AssistantMessageEvent::ThinkingStart { .. } => {
                self.last_message_id = Some(Uuid::now_v7());
                self.emit(AgentEvent::AssistantMessageThinkingStarted {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                })
                .await?;
            }
            ai::AssistantMessageEvent::ThinkingDelta { delta, .. } => {
                self.emit(AgentEvent::AssistantMessageThinkingDelta {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                    delta,
                })
                .await?;
            }
            ai::AssistantMessageEvent::ThinkingEnd { content, .. } => {
                self.emit(AgentEvent::AssistantMessageThinkingCompleted {
                    message_id: self.last_message_id.expect("last_message_id not set"),
                    content,
                })
                .await?;
            }

            ai::AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                self.emit(AgentEvent::ToolCallRequested {
                    tool_call_id: tool_call.id.clone(),
                    name: tool_call.name.clone(),
                    arguments: tool_call.arguments.clone(),
                })
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

    fn mk_loop(store_tx: mpsc::Sender<AgentEvent>) -> AgentLoop {
        AgentLoop::new(
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

        let Some(AgentEvent::AgentTurnEnded {
            reason,
            assistant_message,
            ..
        }) = rx.recv().await
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

        let Some(AgentEvent::AgentTurnEnded {
            reason,
            assistant_message,
            ..
        }) = rx.recv().await
        else {
            panic!("expected AgentTurnEnded");
        };
        assert_eq!(reason, TurnEndReason::ToolUse);
        assert_eq!(assistant_message.unwrap().stop_reason, StopReason::ToolUse);
    }
}
