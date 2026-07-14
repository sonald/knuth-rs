use async_trait::async_trait;
use std::collections::VecDeque;
use std::ops::ControlFlow;

use crate::{
    Actor, ActorContext, ActorRuntime, AgentStepRunner, AgentToolRegistry, BashTool, ToolInput,
    ToolOutcome, spawn_actor,
};
use ai::{
    AssistantMessage, ContentBlock, ImageContent, Message, Model, StreamOptions, ToolResultMessage,
    ToolResultRole, UserContent, UserContentBlock, UserMessage, UserRole,
};
use knuth_core::{
    AgentEvent, AgentSubscription, EventStore, EventStoreError, InMemoryEventStore,
    ModelStepEndReason, SessionEndReason, StoredEvent, UserMessageIntent,
};
use tokio::sync::mpsc::error::{SendError, TrySendError};
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;
use tracing::debug;
use uuid::Uuid;

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

    #[error("Failed to send command to actor: {0}")]
    ActorSendError(#[from] mpsc::error::SendError<AgentActorMessage>),

    #[error("Failed to receive reply: {0}")]
    OneshotRecvError(#[from] oneshot::error::RecvError),

    #[error("Actor task failed: {0}")]
    ActorTaskFailed(#[from] tokio::task::JoinError),

    #[error("Failed to load image '{path}': {source}")]
    ImageLoadFailed {
        path: String,
        source: std::io::Error,
    },
}

#[derive(Debug)]
pub enum AgentActorState {
    Idle,
    Running(AgentStepRunner),
}

#[derive(Debug, Default)]
struct ConversationState {
    messages: Vec<Message>,
    system_prompt: String,
}

impl ConversationState {
    fn new(system_prompt: String) -> Self {
        Self {
            messages: vec![],
            system_prompt: system_prompt,
        }
    }

    fn add_message(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn apply_event(&mut self, event: StoredEvent) {
        match event.event {
            AgentEvent::ModelStepEnded {
                reason,
                assistant_message: Some(assistant_message),
                ..
            } if matches!(
                reason,
                ModelStepEndReason::Success
                    | ModelStepEndReason::Length
                    | ModelStepEndReason::ToolUse
            ) =>
            {
                self.add_message(Message::Assistant(assistant_message));
            }
            AgentEvent::UserMessageCommitted { content, .. } => {
                self.add_message(Message::User(UserMessage {
                    role: UserRole::User,
                    content: content,
                    timestamp: event.timestamp.timestamp(),
                }));
            }
            AgentEvent::ToolExecutionEnded {
                tool_call_id,
                tool_name,
                result,
            } => {
                debug!("append ToolExecutionEnded event to conversation state");

                let content = vec![UserContentBlock::text(result)];
                self.add_message(Message::ToolResult(ToolResultMessage {
                    role: ToolResultRole::ToolResult,
                    tool_call_id,
                    tool_name,
                    content,
                    details: None,
                    is_error: false,
                    timestamp: event.timestamp.timestamp(),
                }));
            }
            AgentEvent::SystemPromptSet { prompt } => {
                self.system_prompt = prompt;
            }
            _ => {}
        }
    }
}

impl std::fmt::Display for ConversationState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConversationState (system_prompt: {} chars, {} messages)",
            self.system_prompt.len(),
            self.messages.len()
        )?;
        for (i, message) in self.messages.iter().enumerate() {
            write!(f, "\n  [{i}] {message}")?;
        }
        Ok(())
    }
}

#[derive(Debug)]
struct PendingInput {
    content: UserContent,
    intent: UserMessageIntent,
}

pub enum AgentActorMessage {
    Command(AgentCommand),
    Turn(Uuid, TurnMessage),
}

#[derive(Debug)]
pub enum TurnMessage {
    Event(AgentEvent),
    Finished {
        reason: ModelStepEndReason,
        assistant_message: Option<AssistantMessage>,
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

/// internal actor for the agent session
#[derive(Debug)]
pub(crate) struct AgentActor {
    id: Uuid,
    conversation_state: ConversationState,
    config: AgentConfig,

    subscriptions: Vec<mpsc::Sender<StoredEvent>>,
    store: Box<dyn EventStore>,

    state: AgentActorState,
    tool_registry: AgentToolRegistry,

    pending_input_queue: VecDeque<PendingInput>,
}

#[async_trait]
impl Actor for AgentActor {
    type Message = AgentActorMessage;

    async fn on_start(&mut self, ctx: &mut ActorContext<Self::Message>) {
        match self
            .persist_and_publish(AgentEvent::SessionStarted {
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

            AgentActorMessage::Turn(
                _,
                TurnMessage::Event(AgentEvent::ModelStepEnded {
                    step_id,
                    reason,
                    assistant_message,
                }),
            ) => {
                if let Err(e) = self
                    .handle_step_ended(step_id, reason, assistant_message, ctx)
                    .await
                {
                    return self.handle_session_error(e).await;
                }
            }

            AgentActorMessage::Turn(_, TurnMessage::Event(event)) => {
                if let Err(e) = self.persist_and_publish(event.clone()).await {
                    return self.handle_session_error(e).await;
                }
            }

            AgentActorMessage::Turn(_, TurnMessage::Finished { .. }) => {}
        }
        ControlFlow::Continue(())
    }

    async fn on_stop(&mut self, _ctx: &mut ActorContext<Self::Message>) {
        //TODO: reason comes from handle() break result
        match self
            .persist_and_publish(AgentEvent::SessionEnded {
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
        let store = Box::new(InMemoryEventStore::new());

        let mut tool_registry = AgentToolRegistry::new();
        tool_registry.register(Box::new(BashTool {}));

        Self {
            id: session_id,
            conversation_state: ConversationState::new("".to_string()),
            config: config,
            subscriptions: Vec::new(),
            state: AgentActorState::Idle,
            store: store,
            pending_input_queue: VecDeque::new(),
            tool_registry: tool_registry,
        }
    }

    fn notify_subscriptions(&mut self, event: StoredEvent) {
        self.subscriptions
            .retain_mut(|s| match s.try_send(event.clone()) {
                Ok(_) => true,
                Err(TrySendError::Full(e)) => {
                    debug!(
                        "AgentActor: Failed to send event to subscription: {}, remove subscription",
                        e
                    );
                    false
                }
                Err(TrySendError::Closed(e)) => {
                    debug!(
                        "AgentActor: Subscription closed: {}, remove subscription",
                        e
                    );
                    false
                }
            });
    }

    async fn execute_tool(
        &mut self,
        tool_call_id: String,
        name: String,
        arguments: ToolInput,
        cancel_token: CancellationToken,
    ) -> Result<(), AgentSessionError> {
        self.persist_and_publish(AgentEvent::ToolExecutionStarted {
            tool_call_id: tool_call_id.clone(),
            tool_name: name.clone(),
            arguments: arguments.clone(),
        })
        .await?;

        if let Some(tool) = self.tool_registry.get(&name) {
            let result = tool.invoke(arguments, cancel_token).await;
            debug!("tool result: {:?}", result);

            let tool_name = tool.schema().name.clone();
            match result {
                Ok(ToolOutcome::Success(result)) => {
                    let result = result.get("output").unwrap().as_str().unwrap().to_string();
                    self.persist_and_publish(AgentEvent::ToolExecutionEnded {
                        tool_call_id,
                        tool_name,
                        result,
                    })
                    .await?;
                }
                Err(e) => {
                    self.persist_and_publish(AgentEvent::ToolExecutionEnded {
                        tool_call_id,
                        tool_name,
                        result: e.to_string(),
                    })
                    .await?;
                }
            }
        } else {
            self.persist_and_publish(AgentEvent::ToolExecutionEnded {
                tool_call_id,
                tool_name: name.clone(),
                result: format!("Invalid tool name: {}", name),
            })
            .await?;
        }

        Ok(())
    }

    async fn handle_session_error(&mut self, error: AgentSessionError) -> ControlFlow<()> {
        match error {
            AgentSessionError::StoreError(_) => return ControlFlow::Break(()),
            e => {
                let _ = self
                    .persist_and_publish(AgentEvent::ErrorOccurred {
                        message: e.to_string(),
                        details: None,
                    })
                    .await;
                ControlFlow::Continue(())
            }
        }
    }

    async fn persist_and_publish(&mut self, event: AgentEvent) -> Result<(), AgentSessionError> {
        let stored = self.store.append(event.clone()).await?;
        self.conversation_state.apply_event(stored.clone());
        self.notify_subscriptions(stored);

        Ok(())
    }

    async fn process_pending_tool_calls(
        &mut self,
        assistant_message: &Option<AssistantMessage>,
        cancel_token: CancellationToken,
    ) -> Result<(), AgentSessionError> {
        let tool_call_events: Vec<_> = assistant_message
            .as_ref()
            .map(|msg| {
                msg.content
                    .iter()
                    .filter_map(|content| match content {
                        ContentBlock::ToolCall(tool_call) => Some(tool_call),
                        _ => None,
                    })
                    .collect()
            })
            .unwrap_or_default();

        for e in tool_call_events {
            self.execute_tool(
                e.id.clone(),
                e.name.clone(),
                e.arguments.clone().into(),
                cancel_token.clone(),
            )
            .await?;
        }

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
        self.state = AgentActorState::Idle;

        if let ModelStepEndReason::Error(error) = &reason {
            self.persist_and_publish(AgentEvent::ErrorOccurred {
                message: error.to_string(),
                details: None,
            })
            .await?;
            debug!("Error occurred in model step: {:?}", error);
            return Ok(());
        }

        let has_tool_call = if let Some(msg) = &assistant_message {
            msg.content
                .iter()
                .any(|content| matches!(content, ContentBlock::ToolCall(_)))
        } else {
            false
        };

        self.persist_and_publish(AgentEvent::ModelStepEnded {
            step_id,
            reason: reason.clone(),
            assistant_message: assistant_message.clone(),
        })
        .await?;

        if has_tool_call {
            self.process_pending_tool_calls(&assistant_message, ctx.shutdown.child_token())
                .await?;
            self.continue_step(ctx).await
        } else {
            self.persist_and_publish(AgentEvent::AgentTurnEnded {
                turn_id: Uuid::now_v7(), //TODO: need to get the turn id from turn start
            })
            .await?;

            self.try_dispatch_next_input(ctx).await
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
                self.persist_and_publish(AgentEvent::SystemPromptSet { prompt: prompt })
                    .await?;
            }
            AgentCommand::Cancel {} => {
                if let AgentActorState::Running(step) = &self.state {
                    step.cancel().await;
                }
            }
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
        self.pending_input_queue.push_back(PendingInput {
            content,
            intent: intent,
        });
        self.try_dispatch_next_input(ctx).await?;
        Ok(())
    }

    async fn assemble_context(&self) -> Result<ai::Context, AgentSessionError> {
        let context = ai::Context {
            system_prompt: Some(self.conversation_state.system_prompt.clone()),
            messages: self.conversation_state.messages.clone(),
            tools: Some(self.tool_registry.schemas()),
        };
        Ok(context)
    }

    async fn continue_step(
        &mut self,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        let context = self.assemble_context().await?;
        debug!("current conversation state: \n{}", self.conversation_state);
        self.spawn_model_step(context, ctx).await?;
        Ok(())
    }

    async fn try_dispatch_next_input(
        &mut self,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        if matches!(self.state, AgentActorState::Running(_)) {
            return Ok(());
        }

        let Some(input) = self.pending_input_queue.pop_front() else {
            return Ok(());
        };

        self.persist_and_publish(AgentEvent::AgentTurnStarted {
            turn_id: Uuid::now_v7(),
        })
        .await?;

        self.persist_and_publish(AgentEvent::UserMessageCommitted {
            content: input.content,
            intent: input.intent,
        })
        .await?;

        let context = self.assemble_context().await?;
        debug!("current conversation state: \n{}", self.conversation_state);
        self.spawn_model_step(context, ctx).await?;
        Ok(())
    }

    async fn spawn_model_step(
        &mut self,
        context: ai::Context,
        actor_ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        let Some(store_tx) = actor_ctx.ugprade() else {
            return Err(AgentSessionError::InvalidState(
                "Actor context is not upgraded".to_string(),
            ));
        };

        let current_step = AgentStepRunner::new(
            self.config.model.clone(),
            self.config.options.clone(),
            store_tx,
            context,
        )
        .await;

        self.state = AgentActorState::Running(current_step);
        Ok(())
    }

    //TODO: impl from_seq
    async fn subscribe(
        &mut self,
        reply: oneshot::Sender<AgentSubscription>,
        _from_seq: Option<u64>,
    ) -> Result<(), AgentSessionError> {
        let (tx, rx) = mpsc::channel(100);
        self.subscriptions.push(tx);
        let _ = reply.send(AgentSubscription::new(rx));
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
        let (reply, receiver) = oneshot::channel();
        let images = self.load_images(images).await?;
        self.runtime
            .handle()
            .send(AgentActorMessage::Command(AgentCommand::SubmitInput {
                intent: UserMessageIntent::Normal,
                reply: reply,
                input: input,
                images: images,
            }))
            .await?;

        receiver.await?
    }

    pub async fn subscribe(
        &mut self,
        from_seq: Option<u64>,
    ) -> Result<AgentSubscription, AgentSessionError> {
        let (reply, receiver) = oneshot::channel();
        self.runtime
            .handle()
            .send(AgentActorMessage::Command(AgentCommand::Subscribe {
                reply,
                from_seq,
            }))
            .await?;

        Ok(receiver.await?)
    }

    pub async fn set_system_prompt(&mut self, prompt: String) -> Result<(), AgentSessionError> {
        self.runtime
            .handle()
            .send(AgentActorMessage::Command(AgentCommand::SetSystemPrompt(
                prompt,
            )))
            .await?;
        Ok(())
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
