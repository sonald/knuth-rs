use async_trait::async_trait;
use std::fmt::Debug;
use std::ops::ControlFlow;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub struct ActorContext<M> {
    self_tx: mpsc::WeakSender<M>,
    pub shutdown: CancellationToken,
}

impl<M> ActorContext<M> {
    pub fn ugprade(&self) -> Option<mpsc::Sender<M>> {
        self.self_tx.upgrade()
    }
}

#[async_trait]
pub trait Actor: Send + 'static {
    type Message: Send + 'static;

    async fn on_start(&mut self, _ctx: &mut ActorContext<Self::Message>) {}
    async fn handle(
        &mut self,
        message: Self::Message,
        ctx: &mut ActorContext<Self::Message>,
    ) -> ControlFlow<()>;
    async fn on_stop(&mut self, _ctx: &mut ActorContext<Self::Message>) {}
}

#[derive(Debug, Clone)]
pub struct ActorHandle<M> {
    tx: mpsc::Sender<M>,
}

impl<M> ActorHandle<M> {
    pub async fn send(&self, message: M) -> Result<(), mpsc::error::SendError<M>> {
        self.tx.send(message).await
    }

    pub fn addr(&self) -> ActorHandle<M> {
        ActorHandle {
            tx: self.tx.clone(),
        }
    }
}

impl<A: Actor> ActorHandle<A> {}

#[derive(Debug)]
pub struct ActorRuntime<M> {
    handle: ActorHandle<M>,
    task: JoinHandle<()>,
    shutdown: CancellationToken,
}

impl<M> ActorRuntime<M> {
    pub async fn shutdown(self) {
        self.shutdown.cancel();
        let _ = self.task.await;
    }

    pub fn handle(&self) -> &ActorHandle<M> {
        &self.handle
    }
}

pub async fn spawn_actor<A: Actor>(mut actor: A, mailbox_size: usize) -> ActorRuntime<A::Message> {
    let (tx, mut rx) = mpsc::channel(mailbox_size);
    let shutdown = CancellationToken::new();

    let ctrl = shutdown.clone();

    let mut ctx = ActorContext {
        self_tx: tx.downgrade(),
        shutdown: shutdown.clone(),
    };

    let task = tokio::spawn(async move {
        actor.on_start(&mut ctx).await;
        loop {
            tokio::select! {
                biased;

                _ = ctrl.cancelled() => break,
                maybe_message = rx.recv() => match maybe_message {
                    Some(message) => {
                        match actor.handle(message, &mut ctx).await {
                            ControlFlow::Continue(_) => {}
                            ControlFlow::Break(_) => {
                                break
                            }
                        }
                    }
                    None => break,
                }
            }
        }
        actor.on_stop(&mut ctx).await;
    });

    ActorRuntime {
        handle: ActorHandle { tx },
        task,
        shutdown,
    }
}
