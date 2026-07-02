use ai::{Model, StreamOptions, stream};
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

    pub(crate) async fn start_agent_loop(&mut self) -> Result<TurnOutcome, AgentLoopError> {
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
                        return Ok(TurnOutcome {
                            turn_id: self.turn_id,
                            reason: TurnEndReason::Success,
                        });
                     };
                    self.handle_event(event).await?;
                }

            }
        }
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
            ai::AssistantMessageEvent::Done { message, .. } => {
                self.emit(AgentEvent::AgentTurnEnded {
                    turn_id: self.turn_id,
                    reason: TurnEndReason::Success,
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
                self.emit(AgentEvent::AssistantMessageStarted {
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
                self.emit(AgentEvent::AssistantMessageCompleted {
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
