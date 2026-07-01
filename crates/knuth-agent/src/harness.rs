use std::collections::VecDeque;
use std::sync::Arc;

use ai::{
    ImageContent, Message, Model, StreamOptions, UserContent, UserMessage, UserRole,
};
use knuth_core::{
    AgentEvent, AgentSubscription, EventStore, EventStoreError, InMemoryEventStore, SessionEndReason, UserMessageIntent,
};
use tokio::sync::mpsc::error::{SendError, TrySendError};
use tokio::sync::{Mutex, mpsc};
use tokio::task::JoinHandle;
use tracing::{debug, info};
use uuid::Uuid;
use tokio_util::sync::CancellationToken;
use crate::AgentLoop;
use crate::agent_loop::AgentLoopError;

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

    #[error("Store error: {0}")]
    StoreError(#[from] EventStoreError),

    #[error("Failed to send event to store: {0}")]
    ChannelSendError(#[from] SendError<AgentEvent>),
}

#[derive(Debug)]
pub enum AgentCommand {
    SubmitInput {
        input: String,
        images: Vec<ImageContent>, // save for later use
    },
    Subscribe {
        tx: mpsc::Sender<AgentEvent>,
    },
    Cancel {},
}

#[derive(Debug, Clone)]
pub enum AgentActorState {
    Idle,
    Running,
    Paused,
    WaitApproval,
    Cancelled,
}

/// internal actor for the agent session
#[derive(Debug)]
pub(crate) struct AgentActor {
    pub id: Uuid,
    pub messages: Arc<Mutex<Vec<Message>>>, // cached messages for the agent
    pub config: AgentConfig,
    pub shutdown_token: CancellationToken,
    pub current_turn_cancel: Option<CancellationToken>,

    pub subscriptions: Vec<mpsc::Sender<AgentEvent>>,

    store_tx: mpsc::Sender<AgentEvent>,
    store_rx: mpsc::Receiver<AgentEvent>,
    store: Box<dyn EventStore>,

    state: AgentActorState,
    agent_loop_task: Option<JoinHandle<Result<(), AgentLoopError>>>,

    pending_input_queue: VecDeque<Message>,
}

impl Drop for AgentActor {
    fn drop(&mut self) {
        debug!("AgentActor: Dropping actor");
        self.shutdown_token.cancel();
    }
}

async fn poll_agent_loop(task: &mut Option<JoinHandle<Result<(), AgentLoopError>>>) -> Result<(), AgentLoopError> {
    match task {
        Some(task) => {
            match task.await {
                Ok(inner) => { return inner; }
                Err(e) => { return Err(AgentLoopError::StoreError(e.to_string())); }
            }
        }
        None => std::future::pending().await,
    }
}

impl AgentActor {
    pub fn new(
        session_id: Uuid,
        config: AgentConfig,
    ) -> Self {

        let store = Box::new(InMemoryEventStore::new());
        let (store_tx, store_rx) = mpsc::channel(100);

        Self {
            id: session_id,
            messages: Arc::new(Mutex::new(vec![])),
            config: config,
            shutdown_token: CancellationToken::new(),
            current_turn_cancel: None,
            subscriptions: Vec::new(),
            state: AgentActorState::Idle,
            store_tx: store_tx,
            store_rx: store_rx,
            store: store,
            agent_loop_task: None,
            pending_input_queue: VecDeque::new(),
        }
    }

    async fn store_event(&mut self, event: AgentEvent) -> Result<(), AgentSessionError> {
        self.store_tx.send(event).await?;
        return Ok(());
    }

    async fn notify_subscriptions(&mut self, event: AgentEvent) -> Result<(), AgentSessionError> {
        self.subscriptions.retain_mut(|s| match s.try_send(event.clone()) {
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

    pub async fn start(&mut self, mut cmd_rx: mpsc::Receiver<AgentCommand>) -> Result<(), AgentSessionError> {
        self.store_event(AgentEvent::SessionStarted {
            session_id: self.id,
        })
        .await?;

        let reason = loop {
            tokio::select! {
                maybe_command = cmd_rx.recv() => {
                    let Some(command) = maybe_command else { break SessionEndReason::Success };
                        debug!("AgentActor: Executing command: {:?}", command);
                        self.handle_command(command).await?;
                }

                _ = self.shutdown_token.cancelled() => {
                    break SessionEndReason::Cancelled;
                }

                maybe_event = self.store_rx.recv() => {
                    let Some(event) = maybe_event else { break SessionEndReason::Success };
                    self.store.append(event.clone()).await?;
                    self.notify_subscriptions(event).await?;
                }

                result = poll_agent_loop(&mut self.agent_loop_task) => {
                    self.handle_turn_ended(result).await?;
                }

            }
        };

        // drain the remaining events from the store
        while let Ok(event) = self.store_rx.try_recv() {
            self.store.append(event.clone()).await?;
            self.notify_subscriptions(event).await?;
        }

        // write the end event to the store manually
        let end_event = AgentEvent::SessionEnded { reason };
        self.store.append(end_event.clone()).await?;
        self.notify_subscriptions(end_event).await?;

        Ok(())
    }

    async fn handle_turn_ended(&mut self, result: Result<(), AgentLoopError>) -> Result<(), AgentSessionError> {
        if let Err(e) = result {
            match e {
                AgentLoopError::LoopCancelled => {
                }
                _ => {
                    self.store_event(AgentEvent::ErrorOccurred { message: e.to_string(), details: None }).await?;
                    debug!("AgentActor: Error occurred in agent loop: {:?}", e);
                }
            }
        }


        self.agent_loop_task = None;
        self.current_turn_cancel = None;

        if self.pending_input_queue.is_empty() {
            self.state = AgentActorState::Idle;
            return Ok(());
        }

        //TODO: handle followup input
        return Ok(());
    }

    async fn handle_command(&mut self, command: AgentCommand) -> Result<(), AgentSessionError> {
        match command {
            AgentCommand::SubmitInput { input, images } => {
                self.submit_input(input, images).await?;
            }
            AgentCommand::Subscribe { tx } => self.subscribe(tx).await?,
            AgentCommand::Cancel {} => {
                if let Some(turn_cancel) = &self.current_turn_cancel {
                    turn_cancel.cancel();
                }
            }
        }

        Ok(())
    }


    //TODO: make this a pipeline 
    async fn assemble_system_prompt(&self) -> Result<String, AgentSessionError> {
        let system_prompt = "You are a terse assistant. ".into();
        Ok(system_prompt)
    }

    async fn assemble_context(&self, input: Message) -> Result<ai::Context, AgentSessionError> {
        let mut messages = self.messages.lock().await;
        messages.push(input);

        let context = ai::Context {
            system_prompt: self.assemble_system_prompt().await.ok(),
            messages: messages.clone(),
            tools: None,
        };
        Ok(context)
    }

    //TODO: take care of steer/followup/etc.
    async fn submit_input(
        &mut self,
        input: String,
        _images: Vec<ImageContent>,
    ) -> Result<(), AgentSessionError> {
        let intent = match self.state {
            AgentActorState::Idle =>  UserMessageIntent::Normal ,
            AgentActorState::Running =>  UserMessageIntent::Followup,
            _ => unimplemented!()
        };

        self.store_event(AgentEvent::UserMessageReceived {
            message_id: Uuid::now_v7(),
            content: UserContent::Text(input.clone()),
            intent: intent,
        })
        .await?;

        //TODO: support images
        let input = Message::User(UserMessage {
            role: UserRole::User,
            content: UserContent::Text(input.clone()),
            timestamp: 0,
        });

        match self.state {
            AgentActorState::Idle => {
                let context = self.assemble_context(input).await?;
                self.spawn_agent_loop(context).await?;
                self.state = AgentActorState::Running;
            }

            AgentActorState::Running => {
                self.pending_input_queue.push_back(input);
            }
            _ => unimplemented!()
        }
        Ok(())
    }

    async fn spawn_agent_loop(&mut self, context: ai::Context) -> Result<(), AgentSessionError> {
        let store_tx = self.store_tx.clone();
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
            agent_loop.start_agent_loop().await?;
            Ok(())
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

    system_prompt: String,
    actor_task: JoinHandle<Result<(), AgentSessionError>>,
    cmd_tx: mpsc::Sender<AgentCommand>,
}

impl AgentSession {
    pub fn new(
        name: String,
        description: String,
        system_prompt: String,
        config: AgentConfig,
    ) -> Self {
        let session_id = Uuid::now_v7();
        let (cmd_tx, cmd_rx) = mpsc::channel(100);

        let mut actor = AgentActor::new(session_id, config);
        let actor_task = tokio::task::spawn(async move {
            actor.start(cmd_rx).await?;

            Ok(())
        });

        Self {
            id: session_id,
            system_prompt: system_prompt,
            name: name,
            description: description,
            actor_task: actor_task,
            cmd_tx: cmd_tx,
        }
    }

    pub async fn submit_input(&mut self, input: String) -> Result<(), AgentSessionError> {
        self.cmd_tx
            .send(AgentCommand::SubmitInput {
                input: input,
                images: vec![],
            })
            .await
            .unwrap();
        Ok(())
    }

    pub async fn subscribe(&mut self) -> Result<AgentSubscription, AgentSessionError> {
        let (tx, rx) = mpsc::channel(100);
        self.cmd_tx
            .send(AgentCommand::Subscribe { tx: tx })
            .await
            .unwrap();

        let subscription = AgentSubscription::new(rx);
        Ok(subscription)
    }
}
