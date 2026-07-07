use async_trait::async_trait;
use std::collections::VecDeque;
use std::ops::ControlFlow;

use crate::agent_loop::AgentTurnLoopError;
use crate::{Actor, ActorContext, ActorRuntime, AgentTurnRunner, spawn_actor};
use ai::{
    AssistantMessage, ImageContent, Message, Model, StreamOptions, UserContent, UserContentBlock,
    UserMessage, UserRole,
};
use knuth_core::{
    AgentEvent, AgentSubscription, EventStore, EventStoreError, InMemoryEventStore,
    SessionEndReason, StoredEvent, TurnEndReason, UserMessageIntent,
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

#[derive(Debug, Clone)]
pub enum AgentActorState {
    Idle,
    Running(CancellationToken),
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
            AgentEvent::AgentTurnEnded {
                assistant_message: Some(assistant_message),
                ..
            } => {
                self.add_message(Message::Assistant(assistant_message));
            }
            AgentEvent::UserMessageCommitted { content, .. } => {
                self.add_message(Message::User(UserMessage {
                    role: UserRole::User,
                    content: content,
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
        reason: TurnEndReason,
        error: Option<AgentTurnLoopError>,
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
                    match e {
                        AgentSessionError::StoreError(_) => return ControlFlow::Break(()),
                        e => {
                            let _ = self
                                .persist_and_publish(AgentEvent::ErrorOccurred {
                                    message: e.to_string(),
                                    details: None,
                                })
                                .await;
                        }
                    }
                }
            }

            AgentActorMessage::Turn(_, TurnMessage::Event(event)) => {
                if let Err(e) = self.persist_and_publish(event).await {
                    match e {
                        AgentSessionError::StoreError(_) => return ControlFlow::Break(()),
                        e => {
                            let _ = self
                                .persist_and_publish(AgentEvent::ErrorOccurred {
                                    message: e.to_string(),
                                    details: None,
                                })
                                .await;
                        }
                    }

                }
            }

            AgentActorMessage::Turn(
                turn_id,
                TurnMessage::Finished {
                    reason,
                    error,
                    assistant_message,
                },
            ) => {
                let _ = self
                    .handle_turn_ended(turn_id, reason, error, assistant_message, ctx)
                    .await;
            }
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

        Self {
            id: session_id,
            conversation_state: ConversationState::new("".to_string()),
            config: config,
            subscriptions: Vec::new(),
            state: AgentActorState::Idle,
            store: store,
            pending_input_queue: VecDeque::new(),
        }
    }

    fn notify_subscriptions(&mut self, event: StoredEvent) {
        self.subscriptions
            .retain_mut(|s| match s.try_send(event.clone()) {
                Ok(_) => true,
                Err(TrySendError::Full(e)) => {
                    debug!("AgentActor: Failed to send event to subscription: {}, remove subscription", e);
                    false
                }
                Err(TrySendError::Closed(e)) => {
                    debug!("AgentActor: Subscription closed: {}, remove subscription", e);
                    false
                }
            });
    }

    async fn persist_and_publish(&mut self, event: AgentEvent) -> Result<(), AgentSessionError> {
        let stored = self.store.append(event.clone()).await?;
        self.conversation_state.apply_event(stored.clone());
        self.notify_subscriptions(stored);

        Ok(())
    }

    async fn handle_turn_ended(
        &mut self,
        turn_id: Uuid,
        reason: TurnEndReason,
        error: Option<AgentTurnLoopError>,
        assistant_message: Option<AssistantMessage>,
        ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        debug!("AgentActor: Turn ended, reason: {:?}", reason);

        if let Some(error) = error {
            self.persist_and_publish(AgentEvent::ErrorOccurred {
                message: error.to_string(),
                details: None,
            })
            .await?;
            debug!("AgentActor: Error occurred in agent loop: {:?}", error);
        }

        self.persist_and_publish(AgentEvent::AgentTurnEnded {
            turn_id,
            reason,
            assistant_message,
        })
        .await?;

        self.state = AgentActorState::Idle;
        self.try_dispatch_next_input(ctx).await
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
            AgentCommand::Cancel {} => {}
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
            tools: None,
        };
        Ok(context)
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

        self.persist_and_publish(AgentEvent::UserMessageCommitted {
            content: input.content,
            intent: input.intent,
        })
        .await?;

        let context = self.assemble_context().await?;
        debug!("current conversation state: \n{}", self.conversation_state);
        self.spawn_turn(context, ctx).await?;
        Ok(())
    }

    async fn spawn_turn(
        &mut self,
        context: ai::Context,
        actor_ctx: &mut ActorContext<AgentActorMessage>,
    ) -> Result<(), AgentSessionError> {
        let Some(store_tx) = actor_ctx.ugprade() else {
            return Err(AgentSessionError::InvalidState(
                "Actor context is not upgraded".to_string(),
            ));
        };
        let turn_cancel = actor_ctx.shutdown.child_token();

        let mut current_turn = AgentTurnRunner::new(
            self.config.model.clone(),
            self.config.options.clone(),
            store_tx,
            context,
            turn_cancel.clone(),
        );

        self.persist_and_publish(AgentEvent::AgentTurnStarted {
            turn_id: current_turn.turn_id(),
        })
        .await?;

        tokio::task::spawn(async move {
            let _ = current_turn.start_turn_loop(Some(10)).await;
        });

        self.state = AgentActorState::Running(turn_cancel);

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
