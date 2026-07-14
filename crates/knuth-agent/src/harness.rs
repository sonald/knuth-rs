use async_trait::async_trait;
use std::collections::{HashSet, VecDeque};
use std::ops::ControlFlow;
use std::sync::Arc;

use crate::{
    Actor, ActorContext, ActorRuntime, AgentStepRunner, AgentToolRegistry, AskError, BashTool,
    EditFileTool, EventLog, PythonTool, ReadFileTool, ToolOutcome, WriteFileTool, spawn_actor,
};
use ai::{
    AssistantMessage, ContentBlock, ImageContent, Model, StreamOptions, ToolCall, UserContent,
    UserContentBlock,
};
use knuth_core::{
    AgentEvent, AgentSubscription, EventStoreError, InMemoryEventStore, ModelStepEndReason,
    SessionEndReason, UserMessageIntent,
};
use tokio::sync::mpsc::error::SendError;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;
use tracing::debug;
use uuid::Uuid;

const SUBSCRIPTION_BUFFER: usize = 100;

#[derive(Debug)]
pub struct AgentConfig {
    pub model: Model,
    pub options: StreamOptions,
}

#[derive(Debug, thiserror::Error)]
pub enum AgentSessionError {
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    #[error("Agent is already running")]
    AgentIsRunning,

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Store error: {0}")]
    StoreError(#[from] EventStoreError),

    #[error("Failed to send event to store: {0}")]
    ChannelSendError(#[from] SendError<AgentEvent>),

    #[error("Actor task failed: {0}")]
    ActorTaskFailed(#[from] tokio::task::JoinError),

    #[error("Failed to load image '{path}': {source}")]
    ImageLoadFailed {
        path: String,
        source: std::io::Error,
    },

    #[error("command failed: {0}")]
    AskError(#[from] AskError),
}

#[derive(Debug)]
struct PendingInput {
    content: UserContent,
    intent: UserMessageIntent,
}

pub enum AgentActorMessage {
    Command(AgentCommand),
    /// A domain event reported by the model step identified by the `Uuid`.
    /// Committed to the log verbatim; never drives the state machine.
    Turn(Uuid, AgentEvent),
    /// Control signal from a step runner: the step is over. The actor derives
    /// and commits the `ModelStepEnded` domain event itself, then advances the
    /// turn state machine.
    StepFinished {
        step_id: Uuid,
        reason: ModelStepEndReason,
        assistant_message: Option<AssistantMessage>,
    },
    /// Control signal from a spawned tool task: one tool call completed.
    ToolFinished {
        turn_id: Uuid,
        tool_call_id: String,
        tool_name: String,
        result: String,
    },
}

pub enum AgentCommand {
    SetSystemPrompt(String),
    SubmitInput {
        intent: UserMessageIntent,
        input: String,
        images: Vec<ImageContent>, // save for later use
        reply: oneshot::Sender<Result<(), AgentSessionError>>,
    },
    Subscribe {
        from_seq: Option<u64>,
        reply: oneshot::Sender<AgentSubscription>,
    },
    Cancel {},
}

impl std::fmt::Display for AgentCommand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentCommand::SetSystemPrompt(prompt) => write!(f, "SetSystemPrompt({})", prompt),
            AgentCommand::SubmitInput { .. } => write!(f, "SubmitInput"),
            AgentCommand::Subscribe { .. } => write!(f, "Subscribe"),
            AgentCommand::Cancel {} => write!(f, "Cancel"),
        }
    }
}

/// Lifecycle of the turn currently being processed. `turn_id` is minted when
/// an input is dispatched and carried through every phase, so
/// `AgentTurnStarted`/`AgentTurnEnded` always refer to the same turn.
#[derive(Debug)]
enum TurnState {
    Idle,
    /// A model step is streaming.
    Streaming { turn_id: Uuid, step: AgentStepRunner },
    /// Tool calls from the last step are executing in background tasks;
    /// `pending` holds the tool_call_ids we are still waiting on.
    RunningTools {
        turn_id: Uuid,
        pending: HashSet<String>,
        cancel: CancellationToken,
    },
}

/// Internal actor for the agent session.
///
/// Owns turn orchestration only: queueing inputs, driving the
/// idle → streaming → running-tools state machine, and reacting to step/tool
/// completion. Persistence + projection + fan-out live in [`EventLog`]; tool
/// lookup lives in [`AgentToolRegistry`]; model streaming lives in
/// [`AgentStepRunner`].
#[derive(Debug)]
pub(crate) struct AgentActor {
    id: Uuid,
    config: AgentConfig,

    log: EventLog,
    tools: AgentToolRegistry,

    turn: TurnState,
    pending_input_queue: VecDeque<PendingInput>,
}

#[async_trait]
impl Actor for AgentActor {
    type Message = AgentActorMessage;

    async fn on_start(&mut self, ctx: &mut ActorContext<Self::Message>) {
        match self
            .log
            .commit(AgentEvent::SessionStarted {
                session_id: self.id,
            })
            .await
        {
            Ok(_) => {
                debug!("Session started");
            }
            Err(e) => {
                debug!("Failed to start session: {:?}", e);
                ctx.shutdown.cancel();
            }
        }
    }

    async fn handle(
        &mut self,
        message: Self::Message,
        ctx: &mut ActorContext<Self::Message>,
    ) -> ControlFlow<()> {
        match message {
            AgentActorMessage::Command(command) => {
                debug!("Handling command: {}", command);
                if let Err(e) = self.handle_command(command, ctx).await {
                    return self.handle_session_error(e).await;
                }
            }

            AgentActorMessage::Turn(_, event) => {
                if let Err(e) = self.log.commit(event).await {
                    return self.handle_session_error(e.into()).await;
                }
            }

            AgentActorMessage::StepFinished {
                step_id,
                reason,
                assistant_message,
            } => {
                if let Err(e) = self
                    .handle_step_ended(step_id, reason, assistant_message, ctx)
                    .await
                {
                    return self.handle_session_error(e).await;
                }
            }

            AgentActorMessage::ToolFinished {
                turn_id,
                tool_call_id,
                tool_name,
                result,
            } => {
                if let Err(e) = self
                    .handle_tool_finished(turn_id, tool_call_id, tool_name, result, ctx)
                    .await
                {
                    return self.handle_session_error(e).await;
                }
            }
        }
        ControlFlow::Continue(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext<Self::Message>) {
        //TODO: reason comes from handle() break result
        match self
            .log
            .commit(AgentEvent::SessionEnded {
                reason: SessionEndReason::Success,
            })
            .await
        {
            Ok(_) => {
                debug!("Session ended");
            }
            Err(e) => {
                debug!("Failed to end session: {:?}", e);
            }
        }
    }
}

impl AgentActor {
    pub fn new(session_id: Uuid, config: AgentConfig) -> Self {
        let mut tools = AgentToolRegistry::new();
        tools.register(Arc::new(BashTool {}));
        tools.register(Arc::new(ReadFileTool {}));
        tools.register(Arc::new(WriteFileTool {}));
        tools.register(Arc::new(EditFileTool {}));
        tools.register(Arc::new(PythonTool {}));

        Self {
            id: session_id,
            config,
            log: EventLog::new(Box::new(InMemoryEventStore::new())),
            tools,
            turn: TurnState::Idle,
            pending_input_queue: VecDeque::new(),
        }
    }

    async fn handle_session_error(&mut self, error: AgentSessionError) -> ControlFlow<()> {
        match error {
            AgentSessionError::StoreError(_) => return ControlFlow::Break(()),
            e => {
                let _ = self
                    .log
                    .commit(AgentEvent::ErrorOccurred {
                        message: e.to_string(),
                        details: None,
                    })
                    .await;
                ControlFlow::Continue(())
            }
        }
    }

    async fn handle_command(
        &mut self,
        command: AgentCommand,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        match command {
            AgentCommand::SubmitInput {
                intent,
                input,
                images,
                reply,
            } => {
                let result = self.submit_input(input, images, intent, ctx).await;
                let _ = reply.send(result);
            }
            AgentCommand::Subscribe { reply, from_seq } => self.subscribe(reply, from_seq).await?,
            AgentCommand::SetSystemPrompt(prompt) => {
                self.log
                    .commit(AgentEvent::SystemPromptSet { prompt })
                    .await?;
            }
            AgentCommand::Cancel {} => match &self.turn {
                TurnState::Idle => {}
                TurnState::Streaming { step, .. } => step.cancel().await,
                TurnState::RunningTools { cancel, .. } => cancel.cancel(),
            },
        }

        Ok(())
    }

    async fn submit_input(
        &mut self,
        input: String,
        images: Vec<ImageContent>,
        intent: UserMessageIntent,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        let content = {
            let mut v = vec![];
            v.push(UserContentBlock::text(input));
            for image in images {
                v.push(UserContentBlock::Image(image));
            }
            UserContent::Blocks(v)
        };
        self.pending_input_queue.push_back(PendingInput { content, intent });
        self.try_dispatch_next_input(ctx).await?;
        Ok(())
    }

    async fn try_dispatch_next_input(
        &mut self,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        if !matches!(self.turn, TurnState::Idle) {
            return Ok(());
        }

        let Some(input) = self.pending_input_queue.pop_front() else {
            return Ok(());
        };

        let turn_id = Uuid::now_v7();
        self.log
            .commit(AgentEvent::AgentTurnStarted { turn_id })
            .await?;
        self.log
            .commit(AgentEvent::UserMessageCommitted {
                content: input.content,
                intent: input.intent,
            })
            .await?;

        self.continue_step(turn_id, ctx).await
    }

    /// Runs the next model step of `turn_id` against the current conversation.
    async fn continue_step(
        &mut self,
        turn_id: Uuid,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        let step = self.spawn_model_step(ctx).await?;
        self.turn = TurnState::Streaming { turn_id, step };
        Ok(())
    }

    async fn handle_step_ended(
        &mut self,
        step_id: Uuid,
        reason: ModelStepEndReason,
        assistant_message: Option<AssistantMessage>,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        debug!("Model step ended, reason: {:?}", reason);

        let turn_id = match std::mem::replace(&mut self.turn, TurnState::Idle) {
            TurnState::Streaming { turn_id, .. } => turn_id,
            other => {
                // A step we no longer track (e.g. raced with shutdown): record
                // the event but leave the state machine alone.
                self.turn = other;
                debug!("step {step_id} ended outside Streaming state");
                self.log
                    .commit(AgentEvent::ModelStepEnded {
                        step_id,
                        reason,
                        assistant_message,
                    })
                    .await?;
                return Ok(());
            }
        };

        self.log
            .commit(AgentEvent::ModelStepEnded {
                step_id,
                reason: reason.clone(),
                assistant_message: assistant_message.clone(),
            })
            .await?;

        let tool_calls: Vec<ToolCall> = assistant_message
            .as_ref()
            .map(|msg| {
                msg.content
                    .iter()
                    .filter_map(|content| match content {
                        ContentBlock::ToolCall(tool_call) => Some(tool_call.clone()),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();

        match reason {
            ModelStepEndReason::Error(error) => {
                self.log
                    .commit(AgentEvent::ErrorOccurred {
                        message: error,
                        details: None,
                    })
                    .await?;
                self.end_turn(turn_id, ctx).await
            }
            ModelStepEndReason::Cancelled => self.end_turn(turn_id, ctx).await,
            _ if !tool_calls.is_empty() => self.spawn_tools(turn_id, tool_calls, ctx).await,
            _ => self.end_turn(turn_id, ctx).await,
        }
    }

    /// Spawns one background task per tool call; each reports back with a
    /// `ToolFinished` message so the actor stays responsive (e.g. to `Cancel`)
    /// while tools run.
    async fn spawn_tools(
        &mut self,
        turn_id: Uuid,
        tool_calls: Vec<ToolCall>,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        let Some(self_tx) = ctx.upgrade() else {
            return Err(AgentSessionError::InvalidState(
                "actor mailbox is gone".to_string(),
            ));
        };
        let cancel = ctx.shutdown.child_token();
        let mut pending = HashSet::new();

        for call in tool_calls {
            self.log
                .commit(AgentEvent::ToolExecutionStarted {
                    tool_call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .await?;

            let Some(tool) = self.tools.get(&call.name) else {
                self.log
                    .commit(AgentEvent::ToolExecutionEnded {
                        tool_call_id: call.id.clone(),
                        tool_name: call.name.clone(),
                        result: format!("Invalid tool name: {}", call.name),
                    })
                    .await?;
                continue;
            };

            pending.insert(call.id.clone());
            let tx = self_tx.clone();
            let cancel = cancel.clone();
            tokio::spawn(async move {
                let result = match tool.invoke(call.arguments, cancel).await {
                    Ok(ToolOutcome::Success(value)) => value
                        .get("output")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .unwrap_or_else(|| value.to_string()),
                    Err(e) => e,
                };
                let _ = tx
                    .send(AgentActorMessage::ToolFinished {
                        turn_id,
                        tool_call_id: call.id,
                        tool_name: call.name,
                        result,
                    })
                    .await;
            });
        }

        if pending.is_empty() {
            // Every call had an unknown tool name; their error results are
            // already committed, so go straight back to the model.
            self.continue_step(turn_id, ctx).await
        } else {
            self.turn = TurnState::RunningTools {
                turn_id,
                pending,
                cancel,
            };
            Ok(())
        }
    }

    async fn handle_tool_finished(
        &mut self,
        turn_id: Uuid,
        tool_call_id: String,
        tool_name: String,
        result: String,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        self.log
            .commit(AgentEvent::ToolExecutionEnded {
                tool_call_id: tool_call_id.clone(),
                tool_name,
                result,
            })
            .await?;

        let TurnState::RunningTools {
            turn_id: current_turn,
            pending,
            cancel,
        } = &mut self.turn
        else {
            debug!("ToolFinished for turn {turn_id} arrived outside RunningTools state");
            return Ok(());
        };
        if *current_turn != turn_id {
            debug!("ToolFinished for stale turn {turn_id}");
            return Ok(());
        }

        pending.remove(&tool_call_id);
        if !pending.is_empty() {
            return Ok(());
        }

        let cancelled = cancel.is_cancelled();
        if cancelled {
            self.end_turn(turn_id, ctx).await
        } else {
            self.continue_step(turn_id, ctx).await
        }
    }

    async fn end_turn(
        &mut self,
        turn_id: Uuid,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        self.log
            .commit(AgentEvent::AgentTurnEnded { turn_id })
            .await?;
        self.turn = TurnState::Idle;
        self.try_dispatch_next_input(ctx).await
    }

    fn assemble_context(&self) -> ai::Context {
        ai::Context {
            system_prompt: Some(self.log.system_prompt().to_string()),
            messages: self.log.messages().to_vec(),
            tools: Some(self.tools.schemas()),
        }
    }

    async fn spawn_model_step(
        &mut self,
        actor_ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<AgentStepRunner, AgentSessionError> {
        let Some(store_tx) = actor_ctx.upgrade() else {
            return Err(AgentSessionError::InvalidState(
                "actor mailbox is gone".to_string(),
            ));
        };

        debug!("current conversation state: \n{}", self.log.conversation());

        Ok(AgentStepRunner::new(
            self.config.model.clone(),
            self.config.options.clone(),
            store_tx,
            self.assemble_context(),
        )
        .await)
    }

    //TODO: impl from_seq
    async fn subscribe(
        &mut self,
        reply: oneshot::Sender<AgentSubscription>,
        _from_seq: Option<u64>,
    ) -> Result<(), AgentSessionError> {
        let _ = reply.send(self.log.subscribe(SUBSCRIPTION_BUFFER));
        Ok(())
    }
}

pub struct AgentSession {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    runtime: ActorRuntime<AgentActorMessage>,
}

impl AgentSession {
    pub async fn build(name: String, description: String, config: AgentConfig) -> Self {
        let id = Uuid::now_v7();

        let actor = AgentActor::new(id, config);
        let runtime = spawn_actor(actor, 100).await;

        Self {
            id,
            name,
            description,
            runtime,
        }
    }

    async fn load_images(
        &self,
        images: Vec<String>,
    ) -> Result<Vec<ImageContent>, AgentSessionError> {
        use base64::Engine;

        let mut image_contents = vec![];
        for path in images {
            let bytes = tokio::fs::read(&path).await.map_err(|source| {
                AgentSessionError::ImageLoadFailed {
                    path: path.clone(),
                    source,
                }
            })?;
            let data = base64::engine::general_purpose::STANDARD.encode(&bytes);
            let mime_type = mime_type_from_path(&path);
            image_contents.push(ImageContent { data, mime_type });
        }
        Ok(image_contents)
    }

    pub async fn submit_input(
        &mut self,
        input: String,
        images: Vec<String>,
    ) -> Result<(), AgentSessionError> {
        let images = self.load_images(images).await?;
        self.runtime.handle()
            .ask(|reply| AgentActorMessage::Command(AgentCommand::SubmitInput {
                intent: UserMessageIntent::Normal,
                reply: reply,
                input: input,
                images: images,
            })).await?
    }

    pub async fn subscribe(
        &mut self,
        from_seq: Option<u64>,
    ) -> Result<AgentSubscription, AgentSessionError> {
        Ok(self.runtime.handle()
            .ask(move |reply| AgentActorMessage::Command(AgentCommand::Subscribe {
                reply,
                from_seq,
            })).await?)
    }

    pub async fn set_system_prompt(&mut self, prompt: String) -> Result<(), AgentSessionError> {
        Ok(self.runtime
            .handle()
            .send(AgentActorMessage::Command(AgentCommand::SetSystemPrompt(
                prompt,
            )))
            .await?)
    }

    pub async fn close(self) -> Result<(), AgentSessionError> {
        self.runtime.shutdown().await;
        Ok(())
    }

    pub async fn cancel_current_turn(&mut self) -> Result<(), AgentSessionError> {
        self.runtime
            .handle()
            .send(AgentActorMessage::Command(AgentCommand::Cancel {}))
            .await?;
        Ok(())
    }
}

fn mime_type_from_path(path: &str) -> String {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        _ => "application/octet-stream",
    }
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::faux_lock;
    use ai::providers::faux::{
        clear_faux_responses, faux_assistant_message, faux_text, faux_tool_call,
        set_faux_responses,
    };
    use ai::{Api, ModelCost, Provider};
    use futures::StreamExt;
    use serde_json::Map;
    use std::time::Duration;
    use tokio::time::timeout;

    fn faux_model() -> Model {
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

    async fn mk_session() -> AgentSession {
        AgentSession::build(
            "test".to_string(),
            "".to_string(),
            AgentConfig {
                model: faux_model(),
                options: StreamOptions::default(),
            },
        )
        .await
    }

    async fn next_event(sub: &mut AgentSubscription) -> AgentEvent {
        timeout(Duration::from_secs(5), sub.next())
            .await
            .expect("timed out waiting for event")
            .expect("subscription closed unexpectedly")
            .event
    }

    #[tokio::test]
    async fn turn_with_tool_call_keeps_turn_id_and_resumes_after_tools() {
        let _guard = faux_lock();
        clear_faux_responses();
        let mut args = Map::new();
        args.insert("command".into(), serde_json::json!("printf tool-ran"));
        set_faux_responses(vec![
            faux_assistant_message(vec![faux_tool_call("bash", args)]),
            faux_assistant_message(vec![faux_text("done")]),
        ]);

        let mut session = mk_session().await;
        let mut sub = session.subscribe(None).await.unwrap();
        session.submit_input("run it".to_string(), vec![]).await.unwrap();

        let mut turn_started = None;
        let mut tool_result = None;
        let mut step_reasons = vec![];
        let turn_ended = loop {
            match next_event(&mut sub).await {
                AgentEvent::AgentTurnStarted { turn_id } => turn_started = Some(turn_id),
                AgentEvent::ToolExecutionEnded { result, .. } => tool_result = Some(result),
                AgentEvent::ModelStepEnded { reason, .. } => step_reasons.push(reason),
                AgentEvent::AgentTurnEnded { turn_id } => break turn_id,
                _ => {}
            }
        };
        clear_faux_responses();

        assert_eq!(
            turn_started.expect("turn should start"),
            turn_ended,
            "AgentTurnStarted and AgentTurnEnded must carry the same turn_id"
        );
        assert_eq!(tool_result.as_deref(), Some("tool-ran"));
        assert_eq!(
            step_reasons,
            vec![ModelStepEndReason::ToolUse, ModelStepEndReason::Success]
        );

        session.close().await.unwrap();
    }

    #[tokio::test]
    async fn queued_inputs_run_as_separate_turns() {
        let _guard = faux_lock();
        clear_faux_responses();
        set_faux_responses(vec![
            faux_assistant_message(vec![faux_text("first")]),
            faux_assistant_message(vec![faux_text("second")]),
        ]);

        let mut session = mk_session().await;
        let mut sub = session.subscribe(None).await.unwrap();
        session.submit_input("one".to_string(), vec![]).await.unwrap();
        session.submit_input("two".to_string(), vec![]).await.unwrap();

        let mut turns = vec![];
        while turns.len() < 2 {
            if let AgentEvent::AgentTurnEnded { turn_id } = next_event(&mut sub).await {
                turns.push(turn_id);
            }
        }
        clear_faux_responses();

        assert_ne!(turns[0], turns[1], "each input should get its own turn");
        session.close().await.unwrap();
    }
}
