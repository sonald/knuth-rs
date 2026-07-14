use async_trait::async_trait;
use std::fmt::Debug;
use std::ops::ControlFlow;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

pub struct ActorContext<M> {
    self_tx: mpsc::WeakSender<M>,
    pub shutdown: CancellationToken,
}

impl<M> ActorContext<M> {
    pub fn upgrade(&self) -> Option<mpsc::Sender<M>> {
        self.self_tx.upgrade()
    }
}

/// Build a detached `ActorContext` for unit-testing `handle`/`on_stop` without a runtime.
#[cfg(test)]
pub(crate) fn test_actor_context<M>() -> ActorContext<M> {
    let (tx, _rx) = mpsc::channel(1);
    ActorContext {
        self_tx: tx.downgrade(),
        shutdown: CancellationToken::new(),
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

#[derive(Debug, thiserror::Error)]
pub enum AskError {
    #[error("Mailbox send error")]
    MailboxSendError,

    #[error("actor dropped the reply channel")]
    ReplyDropped,
}

impl<M> ActorHandle<M> {
    pub async fn send(&self, message: M) -> Result<(), AskError> {
        self.tx.send(message).await.map_err(|_| AskError::MailboxSendError)
    }

    pub fn addr(&self) -> ActorHandle<M> {
        ActorHandle {
            tx: self.tx.clone(),
        }
    }

    pub async fn ask<R>(&self, build_msg: impl FnOnce(oneshot::Sender<R>) -> M) -> Result<R, AskError> {
        let (reply, receiver) = oneshot::channel();
        self.send(build_msg(reply))
            .await?;

        receiver.await.map_err(|_| AskError::ReplyDropped)
    }

}

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
