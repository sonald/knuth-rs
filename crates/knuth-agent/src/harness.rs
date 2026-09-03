use async_trait::async_trait;
use knuth_core::ids::*;
use std::collections::{HashSet, VecDeque};
use std::ops::ControlFlow;
use std::sync::Arc;

use crate::{
    Actor, ActorContext, ActorRuntime, AgentStepRunner, AskError, EventLog, spawn_actor,
    tools::policy::{PolicyContext, PolicyEngineTrait},
};
use ai::{
    AssistantMessage, ContentBlock, ImageContent, Model, StreamOptions, ToolCall, UserContent,
    UserContentBlock,
};
use knuth_core::{
    AgentEvent, AgentSubscription, EventStoreError, InMemoryEventStore, LiveEvent,
    ModelStepEndReason, SessionEndReason, ToolOutcome, UserMessageIntent,
};
use tokio::sync::mpsc::error::SendError;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;
use tracing::debug;

const SUBSCRIPTION_BUFFER: usize = 100;

pub struct AgentConfig {
    pub model: Model,
    pub options: StreamOptions,
    pub policy_engine: Arc<dyn PolicyEngineTrait>,
    pub policy_context: PolicyContext,
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
    Step(StepId, AgentEvent),
    /// Streaming progress from a model step. Fanned out to subscribers and
    /// then dropped; never persisted and never drives the state machine.
    Live(LiveEvent),
    /// Control signal from a step runner: the step is over. The actor derives
    /// and commits the `ModelStepEnded` domain event itself, then advances the
    /// turn state machine.
    StepFinished {
        step_id: StepId,
        generation: Generation,
        reason: ModelStepEndReason,
        assistant_message: Option<AssistantMessage>,
    },
    /// Control signal from a spawned tool task: one tool call completed.
    ToolFinished {
        turn_id: TurnId,
        step_id: StepId,
        invocation_id: ToolInvocationId,
        tool_call_id: String,
        tool_name: String,
        outcome: ToolOutcome,
        content: Vec<u8>,
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
    Streaming {
        turn_id: TurnId,
        step_id: StepId,
        generation: Generation,
        step: AgentStepRunner,
    },
    /// Tool calls from the last step are executing in background tasks;
    /// `pending` holds the tool_call_ids we are still waiting on.
    RunningTools {
        turn_id: TurnId,
        pending: HashSet<ToolInvocationId>,
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
pub(crate) struct AgentActor {
    id: SessionId,
    config: AgentConfig,

    log: EventLog,

    generation: Generation,
    next_step_id: Option<StepId>,

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

            AgentActorMessage::Step(_, event) => {
                if let Err(e) = self.log.commit(event).await {
                    return self.handle_session_error(e.into()).await;
                }
            }

            AgentActorMessage::Live(event) => self.log.publish_live(event),

            AgentActorMessage::StepFinished {
                step_id,
                generation,
                reason,
                assistant_message,
            } => {
                if let Err(e) = self
                    .handle_step_ended(step_id, generation, reason, assistant_message, ctx)
                    .await
                {
                    return self.handle_session_error(e).await;
                }
            }

            AgentActorMessage::ToolFinished { .. } => {
                if let Err(e) = self.handle_tool_finished(&message, ctx).await {
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
    pub fn new(session_id: SessionId, config: AgentConfig) -> Self {
        Self {
            id: session_id,
            config,
            log: EventLog::new(Box::new(InMemoryEventStore::new())),
            generation: Generation::new(),
            next_step_id: None,
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
        self.pending_input_queue
            .push_back(PendingInput { content, intent });
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

        let turn_id = TurnId::new();

        self.log
            .commit(AgentEvent::AgentTurnStarted { turn_id })
            .await?;
        self.log
            .commit(AgentEvent::UserMessageCommitted {
                message_id: MessageId::new(),
                content: input.content,
                intent: input.intent,
            })
            .await?;

        self.continue_step(turn_id, ctx).await
    }

    /// Runs the next model step of `turn_id` against the current conversation.
    async fn continue_step(
        &mut self,
        turn_id: TurnId,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        let step = self.spawn_model_step(ctx).await?;
        self.turn = TurnState::Streaming {
            turn_id: turn_id.clone(),
            step_id: step.step_id(),
            generation: step.generation(),
            step,
        };
        Ok(())
    }

    async fn handle_step_ended(
        &mut self,
        step_id: StepId,
        generation: Generation,
        reason: ModelStepEndReason,
        assistant_message: Option<AssistantMessage>,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        debug!("Model step ended, reason: {:?}", reason);
        let is_current = matches! {
            self.turn,
           TurnState::Streaming { step_id: current_step_id, generation: current_generation, ..}
                if current_step_id == step_id && current_generation == generation
        };

        if !is_current {
            debug!("StepFinished for stale step {step_id}");
            return Ok(());
        }

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
            _ if !tool_calls.is_empty() => {
                self.dispatch_tool_calls(turn_id, step_id, tool_calls, ctx)
                    .await
            }
            _ => self.end_turn(turn_id, ctx).await,
        }
    }

    /// Spawns one background task per tool call; each reports back with a
    /// `ToolFinished` message so the actor stays responsive (e.g. to `Cancel`)
    /// while tools run.
    async fn dispatch_tool_calls(
        &mut self,
        turn_id: TurnId,
        step_id: StepId,
        tool_calls: Vec<ToolCall>,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        let Some(mailbox) = ctx.upgrade() else {
            return Err(AgentSessionError::InvalidState(
                "actor mailbox is gone".to_string(),
            ));
        };
        let batch_cancel = ctx.shutdown.child_token();
        let policy_engine = Arc::clone(&self.config.policy_engine);
        let mut pending = HashSet::new();

        for call in tool_calls {
            let invocation_id = ToolInvocationId::new();

            self.log
                .commit(AgentEvent::ToolExecutionRequested {
                    invocation_id,
                    step_id,
                    tool_call_id: call.id.clone(),
                    tool_name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .await?;
            self.log.publish_live(LiveEvent::ToolExecutionStarted {
                step_id,
                tool_call_id: call.id.clone(),
                tool_name: call.name.clone(),
                arguments: call.arguments.clone(),
            });

            pending.insert(invocation_id);

            let mailbox = mailbox.clone();
            let cancel = batch_cancel.clone();
            let policy_engine = Arc::clone(&policy_engine);
            let policy_context = self.config.policy_context.clone();
            let turn_id = turn_id.clone();

            tokio::spawn(async move {
                let result = policy_engine.execute(&call, cancel, &policy_context).await;
                let _ = mailbox
                    .send(AgentActorMessage::ToolFinished {
                        turn_id: turn_id,
                        step_id,
                        invocation_id,
                        tool_call_id: call.id,
                        tool_name: call.name,
                        outcome: result.outcome,
                        content: result.content,
                    })
                    .await;
            });
        }

        self.turn = TurnState::RunningTools {
            turn_id: turn_id,
            pending,
            cancel: batch_cancel,
        };
        Ok(())
    }

    /// Records the outcome of one invocation: the durable `ToolResultReceived`
    /// that replay depends on, plus the live counterpart for observers.
    async fn record_tool_result(
        &mut self,
        message: &AgentActorMessage,
    ) -> Result<(), AgentSessionError> {
        let (step_id, invocation_id, tool_call_id, tool_name, outcome, content) = match message {
            AgentActorMessage::ToolFinished {
                step_id,
                invocation_id,
                tool_call_id,
                tool_name,
                outcome,
                content,
                ..
            } => (
                step_id.clone(),
                invocation_id.clone(),
                tool_call_id.clone(),
                tool_name.clone(),
                outcome.clone(),
                content.clone(),
            ),
            _ => unreachable!(),
        };

        self.log
            .commit(AgentEvent::ToolResultReceived {
                invocation_id,
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                outcome,
                content: content.clone(),
            })
            .await?;

        self.log.publish_live(LiveEvent::ToolExecutionEnded {
            step_id,
            tool_call_id,
            tool_name,
            result: String::from_utf8_lossy(&content).into_owned(),
        });

        Ok(())
    }

    async fn handle_tool_finished(
        &mut self,
        message: &AgentActorMessage,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        let (turn_id, invocation_id) = match message {
            AgentActorMessage::ToolFinished {
                turn_id,
                invocation_id,
                ..
            } => (turn_id, invocation_id),
            _ => return Ok(()),
        };

        let (is_last, cancelled) = match &mut self.turn {
            TurnState::RunningTools {
                turn_id: current_turn,
                pending,
                cancel,
            } => {
                if current_turn == turn_id && pending.remove(&invocation_id) {
                    (pending.is_empty(), cancel.is_cancelled())
                } else {
                    return Ok(());
                }
            }
            _ => return Ok(()),
        };

        self.record_tool_result(message).await?;
        if !is_last {
            return Ok(());
        }

        if cancelled {
            self.end_turn(*turn_id, ctx).await
        } else {
            self.continue_step(*turn_id, ctx).await
        }
    }

    async fn end_turn(
        &mut self,
        turn_id: TurnId,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        self.log
            .commit(AgentEvent::AgentTurnEnded {
                turn_id: turn_id.clone(),
            })
            .await?;
        self.turn = TurnState::Idle;
        self.try_dispatch_next_input(ctx).await
    }

    fn assemble_context(&self) -> ai::Context {
        ai::Context {
            system_prompt: Some(self.log.system_prompt().to_string()),
            messages: self.log.messages().to_vec(),
            tools: Some(self.config.policy_engine.schemas()),
        }
    }

    fn next_step_id(&mut self) -> (StepId, Generation) {
        self.generation = self.generation.next();
        self.next_step_id = Some(StepId::new());
        (self.next_step_id.unwrap(), self.generation)
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

        let (step_id, generation) = self.next_step_id();

        Ok(AgentStepRunner::new(
            step_id,
            generation,
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
    pub id: SessionId,
    pub name: String,
    pub description: String,
    runtime: ActorRuntime<AgentActorMessage>,
}

impl AgentSession {
    pub async fn build(name: String, description: String, config: AgentConfig) -> Self {
        let id = SessionId::new();

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
        self.runtime
            .handle()
            .ask(|reply| {
                AgentActorMessage::Command(AgentCommand::SubmitInput {
                    intent: UserMessageIntent::Normal,
                    reply: reply,
                    input: input,
                    images: images,
                })
            })
            .await?
    }

    pub async fn subscribe(
        &mut self,
        from_seq: Option<u64>,
    ) -> Result<AgentSubscription, AgentSessionError> {
        Ok(self
            .runtime
            .handle()
            .ask(move |reply| {
                AgentActorMessage::Command(AgentCommand::Subscribe { reply, from_seq })
            })
            .await?)
    }

    pub async fn set_system_prompt(&mut self, prompt: String) -> Result<(), AgentSessionError> {
        Ok(self
            .runtime
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
    use crate::tools::policy::{PolicyContext, PolicyEngineTrait, PolicyMode};
    use crate::{AgentToolRegistry, ToolError, ToolResult};
    use ai::providers::faux::{
        clear_faux_responses, faux_assistant_message, faux_text, faux_tool_call, set_faux_responses,
    };
    use ai::{Api, ModelCost, Provider, ToolCall};
    use futures::StreamExt;
    use knuth_core::SessionEvent;
    use serde_json::Map;
    use std::time::Duration;
    use tokio::time::timeout;
    use tokio_util::sync::CancellationToken;

    struct RegistryPolicyEngine {
        registry: AgentToolRegistry,
    }

    #[async_trait]
    impl PolicyEngineTrait for RegistryPolicyEngine {
        fn schemas(&self) -> Vec<ai::Tool> {
            self.registry.schemas()
        }

        async fn execute(
            &self,
            tool_call: &ToolCall,
            cancel: CancellationToken,
            _ctx: &PolicyContext,
        ) -> ToolResult {
            let Some(tool) = self.registry.get(tool_call.name.as_str()) else {
                return ToolResult {
                    outcome: ToolOutcome::Error,
                    content: ToolError::InvalidTool(tool_call.name.clone())
                        .to_string()
                        .into_bytes(),
                };
            };
            match tool.execute(tool_call.arguments.clone(), cancel).await {
                Ok(result) => result,
                Err(error) => ToolResult {
                    outcome: ToolOutcome::Error,
                    content: error.to_string().into_bytes(),
                },
            }
        }
    }

    struct DenyAllEngine;

    #[async_trait]
    impl PolicyEngineTrait for DenyAllEngine {
        fn schemas(&self) -> Vec<ai::Tool> {
            Vec::new()
        }

        async fn execute(
            &self,
            _tool_call: &ToolCall,
            _cancel: CancellationToken,
            _ctx: &PolicyContext,
        ) -> ToolResult {
            ToolResult {
                outcome: ToolOutcome::PolicyDenied,
                content: b"denied".to_vec(),
            }
        }
    }

    fn default_tool_registry() -> AgentToolRegistry {
        let mut registry = AgentToolRegistry::new();
        registry.load_default();
        registry
    }

    fn passthrough_engine() -> Arc<dyn PolicyEngineTrait> {
        Arc::new(RegistryPolicyEngine {
            registry: default_tool_registry(),
        })
    }

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

    fn test_config(policy_engine: Arc<dyn PolicyEngineTrait>) -> AgentConfig {
        AgentConfig {
            model: faux_model(),
            options: StreamOptions::default(),
            policy_engine,
            policy_context: PolicyContext {
                mode: PolicyMode::Auto,
            },
        }
    }

    async fn mk_session() -> AgentSession {
        AgentSession::build(
            "test".to_string(),
            "".to_string(),
            test_config(passthrough_engine()),
        )
        .await
    }

    async fn next_session_event(sub: &mut AgentSubscription) -> SessionEvent {
        timeout(Duration::from_secs(5), sub.next())
            .await
            .expect("timed out waiting for event")
            .expect("subscription closed unexpectedly")
    }

    /// Skips live progress so assertions see only the replayable history.
    async fn next_durable_event(sub: &mut AgentSubscription) -> AgentEvent {
        loop {
            if let SessionEvent::Durable(stored) = next_session_event(sub).await {
                return stored.event;
            }
        }
    }

    #[tokio::test]
    async fn stale_step_completion_is_not_published() {
        let actor = AgentActor::new(SessionId::new(), test_config(passthrough_engine()));
        let runtime = spawn_actor(actor, 8).await;
        let mut sub = runtime
            .handle()
            .ask(|reply| {
                AgentActorMessage::Command(AgentCommand::Subscribe {
                    from_seq: None,
                    reply,
                })
            })
            .await
            .unwrap();

        runtime
            .handle()
            .send(AgentActorMessage::StepFinished {
                step_id: StepId::new(),
                generation: Generation::new(),
                reason: ModelStepEndReason::Success,
                assistant_message: None,
            })
            .await
            .unwrap();
        runtime
            .handle()
            .send(AgentActorMessage::Command(AgentCommand::SetSystemPrompt(
                "after stale completion".to_string(),
            )))
            .await
            .unwrap();

        assert!(matches!(
            next_durable_event(&mut sub).await,
            AgentEvent::SystemPromptSet { prompt } if prompt == "after stale completion"
        ));
        runtime.shutdown().await;
    }

    #[tokio::test]
    async fn stale_tool_completion_is_not_persisted() {
        let actor = AgentActor::new(SessionId::new(), test_config(passthrough_engine()));
        let runtime = spawn_actor(actor, 8).await;
        let mut sub = runtime
            .handle()
            .ask(|reply| {
                AgentActorMessage::Command(AgentCommand::Subscribe {
                    from_seq: None,
                    reply,
                })
            })
            .await
            .unwrap();

        runtime
            .handle()
            .send(AgentActorMessage::ToolFinished {
                turn_id: TurnId::new(),
                step_id: StepId::new(),
                invocation_id: ToolInvocationId::new(),
                tool_call_id: "stale-call".into(),
                tool_name: "bash".into(),
                outcome: ToolOutcome::Error,
                content: b"stale result".to_vec(),
            })
            .await
            .unwrap();
        runtime
            .handle()
            .send(AgentActorMessage::Command(AgentCommand::SetSystemPrompt(
                "after stale tool completion".to_string(),
            )))
            .await
            .unwrap();

        assert!(matches!(
            next_durable_event(&mut sub).await,
            AgentEvent::SystemPromptSet { prompt } if prompt == "after stale tool completion"
        ));
        runtime.shutdown().await;
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
        session
            .submit_input("run it".to_string(), vec![])
            .await
            .unwrap();

        let mut turn_started = None;
        let mut requested = None;
        let mut received = None;
        let mut live_started_step = None;
        let mut live_ended = None;
        let mut step_ends = vec![];
        let turn_ended = loop {
            match next_session_event(&mut sub).await {
                SessionEvent::Durable(stored) => match stored.event {
                    AgentEvent::AgentTurnStarted { turn_id } => turn_started = Some(turn_id),
                    AgentEvent::ToolExecutionRequested {
                        invocation_id,
                        step_id,
                        ..
                    } => requested = Some((invocation_id, step_id)),
                    AgentEvent::ToolResultReceived {
                        invocation_id,
                        outcome,
                        content,
                        ..
                    } => received = Some((invocation_id, outcome, content)),
                    AgentEvent::ModelStepEnded {
                        step_id, reason, ..
                    } => step_ends.push((step_id, reason)),
                    AgentEvent::AgentTurnEnded { turn_id } => break turn_id,
                    _ => {}
                },
                SessionEvent::Live(event) => match event {
                    LiveEvent::ToolExecutionStarted { step_id, .. } => {
                        live_started_step = Some(step_id)
                    }
                    LiveEvent::ToolExecutionEnded {
                        step_id, result, ..
                    } => live_ended = Some((step_id, result)),
                    _ => {}
                },
            }
        };
        clear_faux_responses();

        assert_eq!(
            turn_started.expect("turn should start"),
            turn_ended,
            "AgentTurnStarted and AgentTurnEnded must carry the same turn_id"
        );

        let (requested_invocation, requested_step) = requested.expect("tool should be requested");
        let (received_invocation, outcome, content) =
            received.expect("tool should report a result");
        assert_eq!(
            requested_invocation, received_invocation,
            "the durable pair must share one invocation_id"
        );
        assert_eq!(outcome, ToolOutcome::ExecSuccess);
        assert!(
            String::from_utf8_lossy(&content).contains("tool-ran"),
            "tool result should include command stdout, got {content:?}"
        );

        let (live_ended_step, live_result) =
            live_ended.expect("tool should report live completion");
        assert!(
            live_result.contains("tool-ran"),
            "live result should mirror the durable content, got {live_result:?}"
        );
        assert_eq!(requested_step, step_ends[0].0);
        assert_eq!(live_started_step, Some(step_ends[0].0));
        assert_eq!(live_ended_step, step_ends[0].0);
        assert_eq!(
            step_ends
                .into_iter()
                .map(|(_, reason)| reason)
                .collect::<Vec<_>>(),
            vec![ModelStepEndReason::ToolUse, ModelStepEndReason::Success]
        );

        session.close().await.unwrap();
    }

    #[tokio::test]
    async fn policy_denial_is_recorded_and_turn_resumes() {
        let _guard = faux_lock();
        clear_faux_responses();

        let mut args = Map::new();
        args.insert(
            "path".into(),
            serde_json::Value::String("/tmp/unused.txt".into()),
        );
        args.insert(
            "content".into(),
            serde_json::Value::String("must not be written".into()),
        );
        set_faux_responses(vec![
            faux_assistant_message(vec![faux_tool_call("write_file", args)]),
            faux_assistant_message(vec![faux_text("done")]),
        ]);

        let mut session = AgentSession::build(
            "test".to_string(),
            "".to_string(),
            test_config(Arc::new(DenyAllEngine)),
        )
        .await;
        let mut sub = session.subscribe(None).await.unwrap();
        session
            .submit_input("try writing".to_string(), vec![])
            .await
            .unwrap();

        let mut outcome = None;
        loop {
            match next_durable_event(&mut sub).await {
                AgentEvent::ToolResultReceived {
                    outcome: tool_outcome,
                    ..
                } => outcome = Some(tool_outcome),
                AgentEvent::AgentTurnEnded { .. } => break,
                _ => {}
            }
        }
        clear_faux_responses();

        assert_eq!(outcome, Some(ToolOutcome::PolicyDenied));
        session.close().await.unwrap();
    }

    #[tokio::test]
    async fn duplicate_tool_call_ids_wait_for_both_invocations() {
        let _guard = faux_lock();
        clear_faux_responses();

        let mut slow_args = Map::new();
        slow_args.insert(
            "command".into(),
            serde_json::Value::String("sleep 0.25; printf slow".into()),
        );
        let mut fast_args = Map::new();
        fast_args.insert(
            "command".into(),
            serde_json::Value::String("printf fast".into()),
        );
        let duplicate_id = "duplicate-call-id".to_string();
        set_faux_responses(vec![
            faux_assistant_message(vec![
                ContentBlock::ToolCall(ToolCall {
                    id: duplicate_id.clone(),
                    name: "bash".into(),
                    arguments: slow_args,
                    thought_signature: None,
                }),
                ContentBlock::ToolCall(ToolCall {
                    id: duplicate_id,
                    name: "bash".into(),
                    arguments: fast_args,
                    thought_signature: None,
                }),
            ]),
            faux_assistant_message(vec![faux_text("done")]),
        ]);

        let mut session = mk_session().await;
        let mut sub = session.subscribe(None).await.unwrap();
        session
            .submit_input("run both".to_string(), vec![])
            .await
            .unwrap();

        let mut requested = HashSet::new();
        let mut received = HashSet::new();
        loop {
            match next_durable_event(&mut sub).await {
                AgentEvent::ToolExecutionRequested { invocation_id, .. } => {
                    requested.insert(invocation_id);
                }
                AgentEvent::ToolResultReceived { invocation_id, .. } => {
                    received.insert(invocation_id);
                }
                AgentEvent::AgentTurnEnded { .. } => break,
                _ => {}
            }
        }
        clear_faux_responses();

        assert_eq!(requested.len(), 2, "each call needs its own invocation");
        assert_eq!(
            received, requested,
            "turn ended before every invocation finished"
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
        session
            .submit_input("one".to_string(), vec![])
            .await
            .unwrap();
        session
            .submit_input("two".to_string(), vec![])
            .await
            .unwrap();

        let mut turns = vec![];
        while turns.len() < 2 {
            if let AgentEvent::AgentTurnEnded { turn_id } = next_durable_event(&mut sub).await {
                turns.push(turn_id);
            }
        }
        clear_faux_responses();

        assert_ne!(turns[0], turns[1], "each input should get its own turn");
        session.close().await.unwrap();
    }
}
