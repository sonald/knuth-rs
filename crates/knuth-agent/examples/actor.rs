use async_trait::async_trait;
use knuth_agent::actor::*;
use std::ops::ControlFlow;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinSet;

#[derive(Error, Debug)]
enum MyError {
    #[error("Failed to receive reply")]
    FailedToReceiveReply(#[from] oneshot::error::RecvError),
    #[error("Failed to send message")]
    FailedToSendMessage(#[from] mpsc::error::SendError<ActorMessage>),
}

struct MyActor {
    counter: u32,
}

enum ActorMessage {
    GetUniqueId { reply: oneshot::Sender<u32> },
}

impl MyActor {
    fn new() -> Self {
        Self { counter: 0 }
    }
}

#[async_trait]
impl Actor for MyActor {
    type Message = ActorMessage;

    async fn handle(
        &mut self,
        message: Self::Message,
        _ctx: &mut ActorContext<Self::Message>,
    ) -> ControlFlow<()> {
        match message {
            ActorMessage::GetUniqueId { reply } => {
                let id = self.counter;
                self.counter += 1;
                if let Err(_) = reply.send(id) {
                    return ControlFlow::Break(());
                }
            }
        }
        ControlFlow::Continue(())
    }
}

struct MyActorHandle {
    handle: ActorHandle<ActorMessage>,
}

impl Clone for MyActorHandle {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle.addr(),
        }
    }
}

impl MyActorHandle {
    fn new(handle: ActorHandle<ActorMessage>) -> Self {
        Self { handle }
    }

    async fn get_unique_id(&self) -> Result<u32, MyError> {
        let (tx, rx) = oneshot::channel();
        self.handle
            .send(ActorMessage::GetUniqueId { reply: tx })
            .await?;
        let id = rx.await?;
        Ok(id)
    }
}

#[tokio::main]
async fn main() -> Result<(), MyError> {
    let actor = MyActor::new();
    let runtime = spawn_actor(actor, 100).await;
    let h = MyActorHandle::new(runtime.handle().addr());

    let id = h.get_unique_id().await?;
    println!("Unique ID: {}", id);

    let mut join_set = JoinSet::new();
    for _ in 0..10 {
        let h2 = h.clone();
        join_set.spawn(async move {
            tokio::time::sleep(Duration::from_millis(1)).await;
            let id2 = h2.get_unique_id().await.unwrap();
            println!("Unique ID(inner): {}", id2);
        });
    }
    while let Some(result) = join_set.join_next().await {
        result.unwrap();
    }
    Ok(())
}
