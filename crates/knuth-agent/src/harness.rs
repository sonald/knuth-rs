use std::collections::VecDeque;

use crate::AgentLoop;
use crate::agent_loop::AgentLoopError;
use ai::{
    ImageContent, Message, Model, StreamOptions, UserContent, UserContentBlock, UserMessage,
    UserRole,
};
use chrono::Utc;
use knuth_core::{
    AgentEvent, AgentSubscription, EventStore, EventStoreError, InMemoryEventStore,
    SessionEndReason, TurnOutcome, UserMessageIntent,
};
use tokio::sync::mpsc;
use tokio::sync::mpsc::error::{SendError, TrySendError};
use tokio::task::JoinHandle;
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
    ActorSendError(#[from] SendError<AgentCommand>),

    #[error("Actor task failed: {0}")]
    ActorTaskFailed(#[from] tokio::task::JoinError),

    #[error("Failed to load image '{path}': {source}")]
    ImageLoadFailed {
        path: String,
        source: std::io::Error,
    },
}

#[derive(Debug)]
pub enum AgentCommand {
    SetSystemPrompt(String),
    SubmitInput {
        input: String,
        images: Vec<ImageContent>, // save for later use
    },
    Followup {
        input: String,
        images: Vec<ImageContent>, // save for later use
    },
    Steer {
        input: String,
        images: Vec<ImageContent>, // save for later use
    },
    Subscribe {
        tx: mpsc::Sender<AgentEvent>,
    },
    Cancel {},
    Stop {},
}

#[derive(Debug, Clone, PartialEq)]
pub enum AgentActorState {
    Idle,
    Running,
    Cancelled,
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

    pub async fn apply_event(&mut self, event: AgentEvent) -> Result<(), AgentSessionError> {
        match event {
            AgentEvent::AgentTurnEnded {
                assistant_message, ..
            } => {
                if let Some(message) = assistant_message {
                    self.add_message(Message::Assistant(message));
                }
            }
            AgentEvent::UserMessageCommitted { content, .. } => {
                self.add_message(Message::User(UserMessage {
                    role: UserRole::User,
                    content: content,
                    timestamp: Utc::now().timestamp(),
                }));
            }
            _ => {}
        }
        Ok(())
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

/// internal actor for the agent session
#[derive(Debug)]
pub(crate) struct AgentActor {
    id: Uuid,
    conversation_state: ConversationState,
    config: AgentConfig,
    shutdown_token: CancellationToken,
    current_turn_cancel: Option<CancellationToken>,

    subscriptions: Vec<mpsc::Sender<AgentEvent>>,

    event_tx: mpsc::Sender<AgentEvent>,
    event_rx: mpsc::Receiver<AgentEvent>,
    store: Box<dyn EventStore>,

    state: AgentActorState,
    agent_loop_task: Option<JoinHandle<Result<TurnOutcome, AgentLoopError>>>,

    pending_input_queue: VecDeque<PendingInput>,
}

impl Drop for AgentActor {
    fn drop(&mut self) {
        debug!("AgentActor: Dropping actor");
        self.shutdown_token.cancel();
    }
}

async fn poll_agent_loop(
    task: &mut Option<JoinHandle<Result<TurnOutcome, AgentLoopError>>>,
) -> Result<TurnOutcome, AgentLoopError> {
    match task {
        Some(task) => match task.await {
            Ok(inner) => {
                return inner;
            }
            Err(e) => {
                return Err(AgentLoopError::EventSendError(e.to_string()));
            }
        },
        None => std::future::pending().await,
    }
}

impl AgentActor {
    pub fn new(session_id: Uuid, config: AgentConfig) -> Self {
        let store = Box::new(InMemoryEventStore::new());
        let (event_tx, event_rx) = mpsc::channel(100);

        Self {
            id: session_id,
            conversation_state: ConversationState::new("".to_string()),
            config: config,
            shutdown_token: CancellationToken::new(),
            current_turn_cancel: None,
            subscriptions: Vec::new(),
            state: AgentActorState::Idle,
            event_tx,
            event_rx,
            store: store,
            agent_loop_task: None,
            pending_input_queue: VecDeque::new(),
        }
    }

    async fn commit_event(&mut self, event: AgentEvent) -> Result<(), AgentSessionError> {
        self.persist_and_publish(event).await?;
        return Ok(());
    }

    async fn notify_subscriptions(&mut self, event: AgentEvent) -> Result<(), AgentSessionError> {
        self.subscriptions
            .retain_mut(|s| match s.try_send(event.clone()) {
                Ok(_) => true,
                Err(TrySendError::Full(e)) => {
                    debug!("AgentActor: Failed to send event to subscription: {:?}", e);
                    true
                }
                Err(TrySendError::Closed(e)) => {
                    debug!("AgentActor: Subscription closed: {:?}", e);
                    false
                }
            });

        Ok(())
    }

    pub async fn start(
        &mut self,
        mut cmd_rx: mpsc::Receiver<AgentCommand>,
    ) -> Result<(), AgentSessionError> {
        self.commit_event(AgentEvent::SessionStarted {
            session_id: self.id,
        })
        .await?;

        let reason = loop {
            tokio::select! {
                maybe_command = cmd_rx.recv() => {
                    let Some(command) = maybe_command else { break SessionEndReason::Success };
                    debug!("AgentActor: Executing command: {:?}", command);
                    if let Err(e) = self.handle_command(command).await {
                        match e {
                            AgentSessionError::StoreError(_) => break SessionEndReason::Error,
                            e => {
                                let _ = self.persist_and_publish(AgentEvent::ErrorOccurred {
                                        message: e.to_string(), details: None
                                }).await;
                            }
                        }
                    }
                }

                _ = self.shutdown_token.cancelled() => {
                    break SessionEndReason::Cancelled;
                }

                maybe_event = self.event_rx.recv() => {
                    let Some(event) = maybe_event else { break SessionEndReason::Success };
                    self.persist_and_publish(event).await?;
                }

                result = poll_agent_loop(&mut self.agent_loop_task) => {
                    while let Ok(event) = self.event_rx.try_recv() {
                        self.persist_and_publish(event).await?;
                    }
                    self.handle_turn_ended(result).await?;
                }
            }
        };

        // drain the remaining events from the store
        while let Ok(event) = self.event_rx.try_recv() {
            debug!("Draining event: {:?}", event.name());
            self.persist_and_publish(event).await?;
        }

        // write the end event to the store manually
        let end_event = AgentEvent::SessionEnded { reason };
        self.persist_and_publish(end_event).await?;

        Ok(())
    }

    async fn persist_and_publish(&mut self, event: AgentEvent) -> Result<(), AgentSessionError> {
        self.store.append(event.clone()).await?;
        self.conversation_state.apply_event(event.clone()).await?;
        self.notify_subscriptions(event).await?;
        Ok(())
    }

    async fn handle_turn_ended(
        &mut self,
        result: Result<TurnOutcome, AgentLoopError>,
    ) -> Result<(), AgentSessionError> {
        debug!("AgentActor: Turn ended, result: {:?}", result);
        match result {
            Ok(_) => {}
            Err(e) => {
                self.commit_event(AgentEvent::ErrorOccurred {
                    message: e.to_string(),
                    details: None,
                })
                .await?;
                debug!("AgentActor: Error occurred in agent loop: {:?}", e);
            }
        }

        self.agent_loop_task = None;
        self.current_turn_cancel = None;
        self.state = AgentActorState::Idle;

        self.try_dispatch_next_input().await
    }

    async fn wait_current_turn_to_finish(&mut self) -> Result<(), AgentSessionError> {
        if self.agent_loop_task.is_none() {
            return Ok(());
        }

        loop {
            tokio::select! {
                maybe_event = self.event_rx.recv() => {
                    if let Some(event) = maybe_event {
                        self.persist_and_publish(event).await?;
                    }
                }

                result = poll_agent_loop(&mut self.agent_loop_task) => {
                    while let Ok(event) = self.event_rx.try_recv() {
                        self.persist_and_publish(event).await?;
                    }

                    self.handle_turn_ended(result).await?;
                    return Ok(());
                }
            }
        }
    }

    async fn handle_command(&mut self, command: AgentCommand) -> Result<(), AgentSessionError> {
        match command {
            AgentCommand::SubmitInput { input, images } => {
                self.submit_input(input, images, UserMessageIntent::Normal)
                    .await?;
            }
            AgentCommand::Followup { input, images } => {
                self.submit_input(input, images, UserMessageIntent::Followup)
                    .await?;
            }
            AgentCommand::Steer { input, images } => {
                self.submit_input(input, images, UserMessageIntent::Normal)
                    .await?;
            }
            AgentCommand::Subscribe { tx } => self.subscribe(tx).await?,
            AgentCommand::Cancel {} => {
                if let Some(turn_cancel) = &self.current_turn_cancel {
                    turn_cancel.cancel();
                }
            }
            AgentCommand::SetSystemPrompt(prompt) => {
                self.conversation_state.system_prompt = prompt;
            }
            AgentCommand::Stop {} => {
                self.shutdown_token.cancel();

                if let Some(turn_cancel) = &self.current_turn_cancel {
                    turn_cancel.cancel();
                }

                self.wait_current_turn_to_finish().await?
            }
        }

        Ok(())
    }

    async fn submit_input(
        &mut self,
        input: String,
        images: Vec<ImageContent>,
        intent: UserMessageIntent,
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
        self.try_dispatch_next_input().await?;
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

    async fn try_dispatch_next_input(&mut self) -> Result<(), AgentSessionError> {
        if self.state != AgentActorState::Idle {
            return Ok(());
        }

        let Some(input) = self.pending_input_queue.pop_front() else {
            return Ok(());
        };

        let message_id = Uuid::now_v7();
        self.commit_event(AgentEvent::UserMessageCommitted {
            message_id: message_id,
            content: input.content,
            intent: input.intent,
        })
        .await?;

        let context = self.assemble_context().await?;
        debug!("current conversation state: \n{}", self.conversation_state);
        self.spawn_agent_loop(context).await?;
        self.state = AgentActorState::Running;
        Ok(())
    }

    async fn spawn_agent_loop(&mut self, context: ai::Context) -> Result<(), AgentSessionError> {
        let store_tx = self.event_tx.clone();
        let turn_cancel = self.shutdown_token.child_token();
        self.current_turn_cancel = Some(turn_cancel.clone());

        let mut agent_loop = AgentLoop::new(
            self.config.model.clone(),
            self.config.options.clone(),
            store_tx,
            context,
            turn_cancel,
        );

        self.agent_loop_task = Some(tokio::task::spawn(async move {
            agent_loop.start_agent_loop(Some(10)).await
        }));

        Ok(())
    }

    async fn subscribe(&mut self, tx: mpsc::Sender<AgentEvent>) -> Result<(), AgentSessionError> {
        self.subscriptions.push(tx);
        Ok(())
    }
}

pub struct AgentSession {
    pub id: Uuid,
    pub name: String,
    pub description: String,

    actor_task: JoinHandle<Result<(), AgentSessionError>>,
    cmd_tx: mpsc::Sender<AgentCommand>,
}

impl AgentSession {
    pub fn new(name: String, description: String, config: AgentConfig) -> Self {
        let session_id = Uuid::now_v7();
        let (cmd_tx, cmd_rx) = mpsc::channel(100);

        let mut actor = AgentActor::new(session_id, config);
        let actor_task = tokio::task::spawn(async move {
            actor.start(cmd_rx).await?;

            Ok(())
        });

        Self {
            id: session_id,
            name: name,
            description: description,
            actor_task: actor_task,
            cmd_tx: cmd_tx,
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
        self.cmd_tx
            .send(AgentCommand::SubmitInput {
                input: input,
                images: images,
            })
            .await?;
        Ok(())
    }

    pub async fn subscribe(&mut self) -> Result<AgentSubscription, AgentSessionError> {
        let (tx, rx) = mpsc::channel(100);
        self.cmd_tx.send(AgentCommand::Subscribe { tx: tx }).await?;

        let subscription = AgentSubscription::new(rx);
        Ok(subscription)
    }

    pub async fn set_system_prompt(&mut self, prompt: String) -> Result<(), AgentSessionError> {
        self.cmd_tx
            .send(AgentCommand::SetSystemPrompt(prompt))
            .await?;
        Ok(())
    }

    pub async fn close(self) -> Result<(), AgentSessionError> {
        self.cmd_tx.send(AgentCommand::Stop {}).await?;

        let result = self.actor_task.await?;
        return result;
    }

    pub async fn cancel_current_turn(&mut self) -> Result<(), AgentSessionError> {
        self.cmd_tx.send(AgentCommand::Cancel {}).await?;
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
