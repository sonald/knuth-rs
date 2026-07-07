use ai::{AssistantMessage, DoneReason, Model, StreamOptions, stream};
use futures::StreamExt;
use knuth_core::{AgentEvent, TurnEndReason};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use uuid::Uuid;

use crate::{AgentActorMessage, TurnMessage};

#[derive(Debug, thiserror::Error)]
pub enum AgentTurnLoopError {
    #[error("Failed to send event to store: {0}")]
    EventSendError(String),

    #[error("LLM error: {0}")]
    LLMError(String),
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

    has_tool_calls: bool,
    turn_reason: TurnEndReason,
    assistant_message: Option<AssistantMessage>,
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
            turn_reason: TurnEndReason::Success,
            assistant_message: None,
        }
    }

    pub fn turn_id(&self) -> Uuid {
        self.turn_id
    }

    pub async fn emit(&mut self, event: TurnMessage) -> Result<(), AgentTurnLoopError> {
        self.store_tx
            .send(AgentActorMessage::Turn(self.turn_id, event))
            .await
            .map_err(|e| AgentTurnLoopError::EventSendError(e.to_string()))
    }

    pub(crate) async fn start_turn_loop(
        &mut self,
        max_turns: Option<usize>,
    ) -> Result<(), AgentTurnLoopError> {
        let mut stream = stream(&self.model, &self.context, Some(&self.options));

        let mut turn = 0;
        let max_turns = max_turns.unwrap_or(1);

        let mut error = None;

        'outer: while turn < max_turns {
            loop {
                tokio::select! {
                    biased;

                    _ = self.cancel_token.cancelled() => {
                        self.turn_reason = TurnEndReason::Cancelled;
                        break 'outer;
                    }

                    event = stream.next() => {
                        let Some(event) = event else {
                            debug!("stream ended, breaking");
                            break;
                        };

                        if let Err(e) = self.handle_event(event).await {
                            self.turn_reason = TurnEndReason::Error;
                            error = Some(e);
                            break 'outer;
                        }
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

        self.emit(TurnMessage::Finished {
            reason: self.turn_reason.clone(),
            error,
            assistant_message: self.assistant_message.clone(),
        })
        .await?;

        Ok(())
    }

    async fn handle_event(
        &mut self,
        event: ai::AssistantMessageEvent,
    ) -> Result<(), AgentTurnLoopError> {
        match event {
            ai::AssistantMessageEvent::Start { .. } => {}
            ai::AssistantMessageEvent::Done { message, reason } => {
                self.turn_reason = match reason {
                    DoneReason::Stop => TurnEndReason::Success,
                    DoneReason::Length => TurnEndReason::Length,
                    DoneReason::ToolUse => TurnEndReason::ToolUse,
                };
                self.assistant_message = Some(message);
            }
            ai::AssistantMessageEvent::Error { error, .. } => {
                return Err(AgentTurnLoopError::LLMError(
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

            ai::AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                self.has_tool_calls = true;
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
    use ai::providers::faux::{clear_faux_responses, set_faux_responses};
    use ai::{
        Api, AssistantMessage, AssistantMessageEvent, AssistantRole, ContentBlock, KnownApi,
        ModelCost, Provider, StopReason, ToolCall, Usage,
    };
    use serde_json::Map;

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

        assert_eq!(agent_loop.turn_reason, TurnEndReason::Length);
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

        assert_eq!(agent_loop.turn_reason, TurnEndReason::ToolUse);
        assert_eq!(
            agent_loop.assistant_message.unwrap().stop_reason,
            StopReason::ToolUse
        );
    }

    #[tokio::test]
    async fn start_turn_loop_sends_finished_with_done_reason() {
        clear_faux_responses();
        let (tx, mut rx) = mpsc::channel(8);
        let mut agent_loop = mk_loop(tx);
        let message = mk_message(StopReason::Length, vec![ContentBlock::text("partial")]);
        set_faux_responses(vec![message]);

        agent_loop.start_turn_loop(Some(1)).await.unwrap();
        clear_faux_responses();

        while let Some(message) = rx.recv().await {
            if let AgentActorMessage::Turn(
                turn_id,
                TurnMessage::Finished {
                    reason,
                    error,
                    assistant_message,
                },
            ) = message
            {
                assert_eq!(turn_id, agent_loop.turn_id());
                assert_eq!(reason, TurnEndReason::Length);
                assert!(error.is_none());
                assert_eq!(assistant_message.unwrap().stop_reason, StopReason::Length);
                return;
            }
        }

        panic!("expected Finished");
    }
}
